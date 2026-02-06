"""Dump Gabor weights from TFLite model to inspect layout."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.tflite_builder import _generate_gabor_kernels, build_optical_flow
from libredgetpu.tflite_parser import parse_full


def main():
    print("="*60)
    print("Gabor Weight Layout Inspection")
    print("="*60)

    # Generate model
    print("\nGenerating model...")
    tflite_bytes, metadata = build_optical_flow(64, 64)

    # Parse model
    print("Parsing model...")
    model = parse_full(tflite_bytes)

    # Find weight tensor
    print(f"\nModel has {len(model.tensors)} tensors:")
    for i, tensor in enumerate(model.tensors):
        print(f"  Tensor {i}: name='{tensor.name}', shape={tensor.shape}, type={tensor.dtype}")

    # Find Gabor weights (should be named "gabor_weights")
    weight_tensor = None
    weight_buffer_data = None
    for i, tensor in enumerate(model.tensors):
        if "gabor" in tensor.name.lower() or "weight" in tensor.name.lower():
            weight_tensor = tensor
            if tensor.buffer_index < len(model.buffers) and model.buffers[tensor.buffer_index] is not None:
                weight_buffer_data = model.buffers[tensor.buffer_index]
            print(f"\n✓ Found weight tensor {i}: '{tensor.name}'")
            print(f"  Shape: {tensor.shape}")
            print(f"  Buffer index: {tensor.buffer_index}")
            print(f"  Buffer size: {len(weight_buffer_data) if weight_buffer_data else 0} bytes")
            break

    if weight_tensor is None or weight_buffer_data is None:
        print("\n❌ Could not find Gabor weight tensor or buffer!")
        return

    # Decode weights
    weights_stored = np.frombuffer(weight_buffer_data, dtype=np.int8)
    print(f"\n  Weights as stored in buffer: {weights_stored.shape} int8 values")
    print(f"  First 64 values: {weights_stored[:64]}")

    # Reshape according to declared tensor shape
    try:
        weights_reshaped = weights_stored.reshape(weight_tensor.shape)
        print(f"\n  Weights reshaped to {weight_tensor.shape}:")
        print(f"    Shape interpretation: {weight_tensor.shape}")
        if len(weight_tensor.shape) == 4:
            print(f"    [dim0={weight_tensor.shape[0]}, dim1={weight_tensor.shape[1]}, "
                  f"dim2={weight_tensor.shape[2]}, dim3={weight_tensor.shape[3]}]")
    except Exception as e:
        print(f"\n❌ Could not reshape: {e}")
        return

    # Generate reference Gabor kernels
    print(f"\n{'='*60}")
    print("Reference Gabor Kernels (from _generate_gabor_kernels)")
    print(f"{'='*60}")
    gabor_float = _generate_gabor_kernels(ksize=7, orientations=4, sigmas=(1.5, 3.0))
    print(f"Generated shape: {gabor_float.shape} (HWIO format)")

    # Quantize reference
    gabor_abs_max = float(np.max(np.abs(gabor_float)))
    gabor_weight_scale = max(gabor_abs_max, 1e-6) / 127.0
    gabor_int8_hwio = np.clip(
        np.round(gabor_float / gabor_weight_scale), -127, 127
    ).astype(np.int8)
    print(f"Quantized shape: {gabor_int8_hwio.shape}")
    print(f"Weight scale: {gabor_weight_scale:.6f}")

    # Try to match stored weights with reference
    print(f"\n{'='*60}")
    print("Attempting to Match Layouts")
    print(f"{'='*60}")

    # The stored weights are in some layout matching weight_tensor.shape
    # Reference is in HWIO: [7, 7, 1, 8]
    # Tensor declares: [8, 7, 7, 1] (OHWI)

    # Test 1: Are stored weights actually HWIO (no transpose)?
    if weights_reshaped.shape == (8, 7, 7, 1):
        # Try to interpret as HWIO by flattening and comparing
        stored_flat = weights_reshaped.ravel()
        ref_flat = gabor_int8_hwio.ravel()
        if np.array_equal(stored_flat, ref_flat):
            print("\n✓ Stored weights match reference in HWIO order (no transpose)")
        else:
            diff = np.abs(stored_flat.astype(np.int16) - ref_flat.astype(np.int16))
            print(f"\n✗ Stored weights differ from HWIO reference:")
            print(f"  Mean abs diff: {diff.mean():.1f}")
            print(f"  Max abs diff: {diff.max()}")
            print(f"  Fraction equal: {np.mean(stored_flat == ref_flat):.4f}")

    # Test 2: Are stored weights actually OHWI (with transpose)?
    gabor_int8_ohwi = np.transpose(gabor_int8_hwio, (3, 0, 1, 2))  # HWIO -> OHWI
    if weights_reshaped.shape == gabor_int8_ohwi.shape:
        stored_flat = weights_reshaped.ravel()
        ref_flat = gabor_int8_ohwi.ravel()
        if np.array_equal(stored_flat, ref_flat):
            print("\n✓ Stored weights match reference in OHWI order (transposed)")
        else:
            diff = np.abs(stored_flat.astype(np.int16) - ref_flat.astype(np.int16))
            print(f"\n✗ Stored weights differ from OHWI reference:")
            print(f"  Mean abs diff: {diff.mean():.1f}")
            print(f"  Max abs diff: {diff.max()}")
            print(f"  Fraction equal: {np.mean(stored_flat == ref_flat):.4f}")

    # Per-channel comparison
    print(f"\n{'='*60}")
    print("Per-Channel Weight Statistics")
    print(f"{'='*60}")
    print(f"{'Chan':<6} {'Stored mean':<15} {'Ref HWIO mean':<18} {'Ref OHWI mean':<18}")
    print("-" * 60)
    for c in range(8):
        if weights_reshaped.shape == (8, 7, 7, 1):
            stored_c = weights_reshaped[c, :, :, :].ravel()
        else:
            stored_c = weights_reshaped[:, :, :, c].ravel() if len(weights_reshaped.shape) == 4 else None

        ref_hwio_c = gabor_int8_hwio[:, :, :, c].ravel()
        ref_ohwi_c = gabor_int8_ohwi[c, :, :, :].ravel()

        stored_mean = stored_c.mean() if stored_c is not None else float('nan')
        ref_hwio_mean = ref_hwio_c.mean()
        ref_ohwi_mean = ref_ohwi_c.mean()

        print(f"{c:<6} {stored_mean:<15.1f} {ref_hwio_mean:<18.1f} {ref_ohwi_mean:<18.1f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
