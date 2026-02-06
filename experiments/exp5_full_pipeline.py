#!/usr/bin/env python3
"""Experiment 5: Full Pipeline Integration with relayout.

Monkey-patches _extract_features_uint8 to use relayout_output(), then tests
the full compute() pipeline for 7 displacements. Key test: vertical shifts
must produce correct vy (currently returns 0).

Tests both standard and pooled modes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from libredgetpu.optical_flow_module import OpticalFlow
from libredgetpu.delegate import relayout_output
from libredgetpu._quantize import dequantize


def monkey_patch_relayout(flow):
    """Monkey-patch _extract_features_uint8 to use relayout_output."""
    if flow._cached_mode:
        output_layer = flow._eo_exe.output_layers[0]
    elif flow._sa_exe is not None:
        output_layer = flow._sa_exe.output_layers[0]
    else:
        raise RuntimeError("No executable found")

    original_method = flow._extract_features_uint8

    def patched_extract(image):
        """Extract features with relayout applied."""
        image = np.asarray(image)
        if image.ndim == 4:
            image = image.squeeze(axis=0)
        if image.ndim == 3:
            image = image.squeeze(axis=-1) if image.shape[-1] == 1 else image
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape != (flow._height, flow._width):
            raise ValueError(f"Image shape {image.shape} != ({flow._height}, {flow._width})")

        image_normalized = image.astype(np.float32) / 255.0
        quantized = flow._quantize_input(image_normalized)
        raw_output = flow._execute_raw(quantized.tobytes())

        if output_layer.tile_layout is not None:
            return relayout_output(raw_output, output_layer)
        else:
            n = flow._out_h * flow._out_w * flow._num_filters
            return np.frombuffer(raw_output, dtype=np.uint8)[:n].reshape(
                flow._out_h, flow._out_w, flow._num_filters)

    flow._extract_features_uint8 = patched_extract
    return flow


def test_pipeline(flow, label, pool_factor=4):
    """Test the full compute pipeline with known displacements."""
    h, w = flow._height, flow._width
    np.random.seed(42)
    texture = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    # Test displacements (in INPUT pixel space)
    tests = [
        ("no shift", 0, 0),
        ("right 4px", 4, 0),
        ("right 8px", 8, 0),
        ("left 8px", -8, 0),
        ("down 4px", 0, 4),
        ("down 8px", 0, 8),
        ("up 8px", 0, -8),
        ("diag +4,+4", 4, 4),
    ]

    # Expected flow in pooled pixels
    if flow._fused_pool:
        pf = flow._fused_pool
    else:
        pf = flow._pool_factor

    print(f"\n{label}")
    print(f"{'Test':>15} | {'Shift(dx,dy)':>14} | {'Expected(vx,vy)':>18} | {'Measured(vx,vy)':>18} | {'Status':>8}")
    print("-" * 85)

    n_pass = 0
    n_total = len(tests)

    for name, dx, dy in tests:
        shifted = np.roll(texture, shift=(dy, dx), axis=(0, 1))
        vx, vy = flow.compute(texture, shifted)

        # Expected in pooled pixels
        exp_vx = dx / pf
        exp_vy = dy / pf

        err_vx = abs(vx - exp_vx)
        err_vy = abs(vy - exp_vy)
        tolerance = 0.7  # Half a pooled pixel

        passed = err_vx < tolerance and err_vy < tolerance
        if passed:
            n_pass += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"{name:>15} | ({dx:+d},{dy:+d}){'':>6} | ({exp_vx:+5.1f},{exp_vy:+5.1f}){'':>5} | ({vx:+5.2f},{vy:+5.2f}){'':>5} | {status:>8}")

    print("-" * 85)
    print(f"Result: {n_pass}/{n_total} passed")
    return n_pass, n_total


def main():
    print("Experiment 5: Full Pipeline Integration with Relayout")
    print("=" * 85)

    # Test standard mode (64x64, pooled=False)
    print("\n--- Standard Mode (pooled=False) ---")
    flow_std = OpticalFlow.from_template(64, pooled=False)
    flow_std.open()
    try:
        # First test WITHOUT relayout (baseline)
        print("\nBaseline (no relayout):")
        test_pipeline(flow_std, "Standard mode — NO relayout")

        # Now with relayout
        monkey_patch_relayout(flow_std)
        print("\nWith relayout:")
        n_pass_std, n_total_std = test_pipeline(flow_std, "Standard mode — WITH relayout")
    finally:
        flow_std.close()

    # Test pooled mode (64x64, pooled=True)
    print("\n\n--- Pooled Mode (pooled=True) ---")
    flow_pool = OpticalFlow.from_template(64, pooled=True)
    flow_pool.open()
    try:
        # First test WITHOUT relayout (baseline)
        print("\nBaseline (no relayout):")
        test_pipeline(flow_pool, "Pooled mode — NO relayout")

        # Now with relayout
        monkey_patch_relayout(flow_pool)
        print("\nWith relayout:")
        n_pass_pool, n_total_pool = test_pipeline(flow_pool, "Pooled mode — WITH relayout")
    finally:
        flow_pool.close()

    # Summary
    print("\n" + "=" * 85)
    print("SUMMARY")
    print(f"  Standard mode: {n_pass_std}/{n_total_std} passed")
    print(f"  Pooled mode:   {n_pass_pool}/{n_total_pool} passed")

    if n_pass_std >= 7 and n_pass_pool >= 7:
        print("\n✓ PASS: Full pipeline works with relayout!")
    elif n_pass_std >= 7:
        print("\n~ PARTIAL: Standard mode works, pooled mode has issues")
    else:
        print("\n✗ FAIL: Pipeline still broken with relayout")


if __name__ == "__main__":
    main()
