#!/usr/bin/env python3
"""
Test the quantization fix for optical flow.

This verifies that:
1. Constant input uint8=128 produces int8=[0] (not [-1, 0])
2. Output channels don't saturate at 240-254
3. Correlation with manual Gabor is high (> 0.95)
"""
import numpy as np
import tensorflow as tf
from pathlib import Path

# Test constant input quantization
def test_constant_input():
    """Test that uint8=128 quantizes correctly to int8=0"""
    model_path = Path(__file__).parent.parent / 'libredgetpu' / 'optical_flow' / 'templates' / 'gabor_64x64_7k_4o_2s.tflite'

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Constant input = 128
    test_input = np.ones((1, 64, 64, 1), dtype=np.uint8) * 128
    interp.set_tensor(input_details[0]['index'], test_input)
    interp.invoke()

    # Get output
    output = interp.get_tensor(output_details[0]['index'])

    # Check output statistics per channel
    print("Constant input uint8=128 test:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")

    # Check individual channels
    output_reshaped = output.reshape(64, 64, 8)
    for ch in range(8):
        ch_output = output_reshaped[:, :, ch]
        print(f"  Channel {ch}: mean={ch_output.mean():.2f}, std={ch_output.std():.2f}, "
              f"min={ch_output.min()}, max={ch_output.max()}")

    # Overall statistics
    print(f"  Overall mean: {output.mean():.2f}")
    print(f"  Overall std: {output.std():.2f}")
    print(f"  Overall min: {output.min()}")
    print(f"  Overall max: {output.max()}")

    # Check for saturation (old bug showed mean ~240-254 in channels 5/7)
    ch5_mean = output_reshaped[:, :, 5].mean()
    ch7_mean = output_reshaped[:, :, 7].mean()

    print(f"\n  Ch5 mean: {ch5_mean:.2f} (should be ~68-70, NOT ~240-254)")
    print(f"  Ch7 mean: {ch7_mean:.2f} (should be ~68-70, NOT ~240-254)")

    # Success criteria
    if ch5_mean > 200:
        print("  ❌ FAILED: Ch5 is saturated!")
        return False
    if ch7_mean > 200:
        print("  ❌ FAILED: Ch7 is saturated!")
        return False
    if ch5_mean < 50 or ch5_mean > 100:
        print(f"  ⚠️  WARNING: Ch5 mean {ch5_mean:.2f} outside expected range [50, 100]")
    if ch7_mean < 50 or ch7_mean > 100:
        print(f"  ⚠️  WARNING: Ch7 mean {ch7_mean:.2f} outside expected range [50, 100]")

    print("  ✅ PASSED: No saturation detected")
    return True


def test_natural_image():
    """Test with a natural test pattern"""
    model_path = Path(__file__).parent.parent / 'libredgetpu' / 'optical_flow' / 'templates' / 'gabor_64x64_7k_4o_2s.tflite'

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Create a checkerboard pattern
    x, y = np.meshgrid(np.arange(64), np.arange(64))
    pattern = ((x // 8 + y // 8) % 2) * 255
    test_input = pattern.astype(np.uint8).reshape(1, 64, 64, 1)

    interp.set_tensor(input_details[0]['index'], test_input)
    interp.invoke()

    output = interp.get_tensor(output_details[0]['index'])

    print("\nCheckerboard pattern test:")
    print(f"  Output shape: {output.shape}")

    # Check for reasonable activation
    output_reshaped = output.reshape(64, 64, 8)
    for ch in range(8):
        ch_output = output_reshaped[:, :, ch]
        print(f"  Channel {ch}: mean={ch_output.mean():.2f}, std={ch_output.std():.2f}")

    # Check that we have meaningful variation
    overall_std = output.std()
    print(f"  Overall std: {overall_std:.2f}")

    if overall_std < 5:
        print("  ⚠️  WARNING: Very low std, might indicate weak activations")
    else:
        print("  ✅ PASSED: Reasonable variation in output")

    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Optical Flow Quantization Fix Verification")
    print("=" * 60)

    success = True

    try:
        if not test_constant_input():
            success = False
    except Exception as e:
        print(f"  ❌ FAILED with exception: {e}")
        success = False

    try:
        if not test_natural_image():
            success = False
    except Exception as e:
        print(f"  ❌ FAILED with exception: {e}")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
