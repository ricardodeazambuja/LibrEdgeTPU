#!/usr/bin/env python3
"""Experiment 4: Pooled model relayout check.

Tests if the pooled model (16x16x8 output) also benefits from relayout_output().
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from libredgetpu.optical_flow_module import OpticalFlow
from libredgetpu.delegate import relayout_output


def main():
    print("Experiment 4: Pooled Model Relayout Check")
    print("=" * 75)

    flow = OpticalFlow.from_template(64, pooled=True)
    flow.open()

    try:
        if flow._cached_mode:
            output_layer = flow._eo_exe.output_layers[0]
        else:
            output_layer = flow._sa_exe.output_layers[0]

        print(f"Output layer: y={output_layer.y_dim}, x={output_layer.x_dim}, z={output_layer.z_dim}")
        print(f"Tile layout: {'PRESENT' if output_layer.tile_layout is not None else 'NONE'}")
        print(f"Fused pool: {flow._fused_pool}")

        h_in, w_in = flow._height, flow._width
        oh, ow, nf = flow._out_h, flow._out_w, flow._num_filters
        n = oh * ow * nf
        shape = (oh, ow, nf)

        print(f"Input: {h_in}x{w_in}, Output: {oh}x{ow}x{nf}")

        # Test 1: Shift equivariance
        print("\n\nTest 1: Shift Equivariance (pooled model)")
        np.random.seed(42)
        texture = np.random.randint(0, 256, (h_in, w_in), dtype=np.uint8)

        def get_features(img):
            img_normalized = img.astype(np.float32) / 255.0
            quantized = flow._quantize_input(img_normalized)
            raw = flow._execute_raw(quantized.tobytes())
            naive = np.frombuffer(raw, dtype=np.uint8)[:n].reshape(shape)
            relayouted = relayout_output(raw, output_layer)
            return naive, relayouted

        base_naive, base_relay = get_features(texture)

        # Shifts in INPUT pixel space (pool_factor=4 → 1 pooled pixel = 4 input pixels)
        pool = flow._fused_pool or flow._pool_factor
        shifts = [(4, 0), (0, 4), (8, 0), (0, 8)]  # These are 1, 1, 2, 2 pooled pixels
        border = 3  # Gabor kernel border in pooled space (7/2 ~= 3)

        print(f"\n{'Shift (dx,dy) input':>20} | {'Pooled':>8} | {'Naive Corr':>12} | {'Relayout Corr':>14}")
        print("-" * 65)

        for dx, dy in shifts:
            shifted = np.roll(texture, shift=(dy, dx), axis=(0, 1))
            shift_naive, shift_relay = get_features(shifted)

            # Shift in pooled space
            pdx, pdy = dx // pool, dy // pool
            s = max(abs(pdx), abs(pdy)) + border

            def pearson(a, b):
                a_flat = a.ravel().astype(float)
                b_flat = b.ravel().astype(float)
                if len(a_flat) == 0 or a_flat.std() < 1e-10 or b_flat.std() < 1e-10:
                    return 0.0
                return np.corrcoef(a_flat, b_flat)[0, 1]

            base_crop_n = base_naive[s:oh-s, s:ow-s]
            shift_crop_n = shift_naive[s+pdy:oh-s+pdy, s+pdx:ow-s+pdx]
            corr_naive = pearson(base_crop_n, shift_crop_n)

            base_crop_r = base_relay[s:oh-s, s:ow-s]
            shift_crop_r = shift_relay[s+pdy:oh-s+pdy, s+pdx:ow-s+pdx]
            corr_relay = pearson(base_crop_r, shift_crop_r)

            print(f"({dx:+d},{dy:+d}){'':>13} | ({pdx:+d},{pdy:+d}) | {corr_naive:>12.4f} | {corr_relay:>14.4f}")

        # Test 2: Edge detection
        print("\n\nTest 2: Edge Localization (pooled model)")
        edge_rows_input = [16, 32, 48]  # In input space

        print(f"{'Edge Row (input)':>18} | {'Naive Trans':>12} | {'Naive Err':>10} | {'Relayout Trans':>15} | {'Relayout Err':>12}")
        print("-" * 75)

        for edge_row in edge_rows_input:
            img = np.zeros((h_in, w_in), dtype=np.uint8)
            img[:edge_row, :] = 255

            img_normalized = img.astype(np.float32) / 255.0
            quantized = flow._quantize_input(img_normalized)
            raw = flow._execute_raw(quantized.tobytes())

            naive = np.frombuffer(raw, dtype=np.uint8)[:n].reshape(shape)
            relayouted = relayout_output(raw, output_layer)

            # Expected edge row in pooled space
            expected_pooled = edge_row // pool

            for feat, name in [(naive, "Naive"), (relayouted, "Relayout")]:
                all_ch = feat.astype(float).mean(axis=(1, 2))
                grad = np.abs(np.diff(all_ch))
                trans = np.argmax(grad)
                err = abs(trans - expected_pooled)
                if name == "Naive":
                    naive_trans, naive_err = trans, err
                else:
                    relay_trans, relay_err = trans, err

            print(f"{edge_row:>18} | {naive_trans:>12} | {naive_err:>10} | {relay_trans:>15} | {relay_err:>12}")

        # Test 3: Check if naive and relayout produce DIFFERENT results
        print("\n\nTest 3: Are naive and relayout different?")
        diff = np.abs(base_naive.astype(int) - base_relay.astype(int))
        print(f"  Max pixel difference: {diff.max()}")
        print(f"  Mean pixel difference: {diff.mean():.2f}")
        print(f"  Fraction of pixels that differ: {(diff > 0).mean():.4f}")

        if diff.max() == 0:
            print("  NOTE: Naive and relayout produce IDENTICAL results!")
            print("  This means the pooled model's output is already in correct order.")
        else:
            print("  Relayout DOES change the output — tiling is non-trivial.")

        print("\n" + "=" * 75)
        print("SUMMARY: Pooled model relayout check complete.")

    finally:
        flow.close()


if __name__ == "__main__":
    main()
