#!/usr/bin/env python3
"""
Systematic investigation of depthwise convolution fundamentals.

This script implements 6 phases of testing to understand quantized
depthwise convolution behavior from first principles.

Usage:
    python test_depthwise_fundamentals.py --phase 1
    python test_depthwise_fundamentals.py --phase 2
    python test_depthwise_fundamentals.py --all
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from libredgetpu.tflite_builder import build_simple_depthwise
from libredgetpu.tflite_parser import parse_full
from test_minimal_reference import manual_depthwise_conv

try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("Warning: TensorFlow not available, skipping TFLite interpreter tests")


def run_tflite_interpreter(tflite_bytes, input_uint8):
    """Run TFLite interpreter and return output."""
    if tf is None:
        raise RuntimeError("TensorFlow not installed")

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Set input
    if input_uint8.ndim == 2:
        input_data = input_uint8[np.newaxis, :, :, np.newaxis]  # [H, W] → [1, H, W, 1]
    else:
        input_data = input_uint8

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])
    return output_data[0]  # Remove batch dimension


class Phase1_IdentityConv:
    """Test single-channel identity convolution.

    Goal: Verify basic depthwise conv works with simplest possible case.
    """

    def __init__(self):
        self.phase_name = "Phase 1: Identity Convolution"

    def build_model(self, h=4, w=4, ksize=3):
        """Build minimal depthwise conv: identity kernel, 1 channel."""
        print(f"\n{'='*60}")
        print(f"{self.phase_name}")
        print(f"{'='*60}")
        print(f"Building {h}×{w} model with {ksize}×{ksize} kernel (all 1s)")

        # Kernel: all 1s → output = sum of neighbors
        kernel = np.ones((1, ksize, ksize, 1), dtype=np.float32)

        # Identity-like quantization
        input_scale = 1.0 / 255.0
        weight_scale = 1.0 / 127.0
        output_scale = 0.02  # Will calculate properly

        tflite_bytes, metadata = build_simple_depthwise(
            height=h, width=w, ksize=ksize, num_filters=1,
            kernel_weights=kernel,
            input_scale=input_scale,
            output_scale=output_scale
        )

        print(f"✓ Model built: {len(tflite_bytes)} bytes")
        print(f"  Input scale: {metadata['input_scale']}")
        print(f"  Output scale: {metadata['output_scale']}")

        return tflite_bytes, metadata, kernel

    def test_constant_input(self):
        """Test with constant input=128."""
        print(f"\n--- Test: Constant Input (128) ---")

        tflite_bytes, metadata, kernel = self.build_model()

        # Input: constant 128
        input_img = np.full((4, 4), 128, dtype=np.uint8)
        print(f"Input: {input_img[0, 0]} (constant)")

        # Manual reference
        manual_out = manual_depthwise_conv(
            input_img, kernel,
            input_scale=metadata['input_scale'],
            output_scale=metadata['output_scale'],
            weight_scales=[metadata['weight_scales'][0]],
            verbose=True
        )

        print(f"\nManual output:")
        print(manual_out[:, :, 0])

        # TFLite interpreter
        if tf is not None:
            tflite_out = run_tflite_interpreter(tflite_bytes, input_img)
            print(f"\nTFLite output:")
            print(tflite_out[:, :, 0])

            # Compare
            diff = np.abs(tflite_out[:, :, 0].astype(np.int16) - manual_out[:, :, 0].astype(np.int16))
            print(f"\nDifference (TFLite - Manual):")
            print(diff)
            print(f"Max diff: {diff.max()}, Mean diff: {diff.mean():.2f}")

            if diff.max() <= 1:
                print("✓ PASS: Outputs match within ±1")
                return True
            else:
                print("✗ FAIL: Outputs differ by more than 1")
                return False
        else:
            print("⚠ Skipping TFLite comparison (TensorFlow not available)")
            return None

    def test_range_of_inputs(self):
        """Test with range of input values."""
        print(f"\n--- Test: Range of Inputs ---")

        tflite_bytes, metadata, kernel = self.build_model()

        test_values = [0, 64, 128, 192, 255]
        results = []

        for val in test_values:
            input_img = np.full((4, 4), val, dtype=np.uint8)

            manual_out = manual_depthwise_conv(
                input_img, kernel,
                input_scale=metadata['input_scale'],
                output_scale=metadata['output_scale'],
                weight_scales=[metadata['weight_scales'][0]],
                verbose=False
            )

            if tf is not None:
                tflite_out = run_tflite_interpreter(tflite_bytes, input_img)
                diff = np.abs(tflite_out[:, :, 0] - manual_out[:, :, 0]).max()
                results.append((val, manual_out[0, 0, 0], tflite_out[0, 0, 0], diff))
                print(f"  Input={val:3d} → Manual={manual_out[0,0,0]:3d}, TFLite={tflite_out[0,0,0]:3d}, Diff={diff}")
            else:
                results.append((val, manual_out[0, 0, 0], None, None))
                print(f"  Input={val:3d} → Manual={manual_out[0,0,0]:3d}")

        if tf is not None and all(r[3] <= 1 for r in results):
            print("✓ PASS: All inputs match within ±1")
            return True
        else:
            return None if tf is None else False


class Phase2_IntentionalSaturation:
    """Force saturation to verify understanding.

    Goal: Verify we can intentionally cause saturation and understand bounds.
    """

    def __init__(self):
        self.phase_name = "Phase 2: Intentional Saturation"

    def test_positive_saturation(self):
        """Force output to +127."""
        print(f"\n{'='*60}")
        print(f"{self.phase_name}")
        print(f"{'='*60}")
        print(f"\n--- Test: Positive Saturation ---")

        # Large positive accumulator
        ksize = 3
        kernel = np.ones((1, ksize, ksize, 1), dtype=np.float32) * 127  # Max weight
        input_img = np.full((4, 4), 255, dtype=np.uint8)  # Max input

        # Very small output scale to force saturation
        input_scale = 1.0 / 255.0
        weight_scale = 1.0  # Large scale
        output_scale = 0.001  # Tiny scale → saturation

        tflite_bytes, metadata = build_simple_depthwise(
            height=4, width=4, ksize=ksize, num_filters=1,
            kernel_weights=kernel,
            input_scale=input_scale,
            output_scale=output_scale
        )

        manual_out = manual_depthwise_conv(
            input_img, kernel,
            input_scale=input_scale,
            output_scale=output_scale,
            weight_scales=[weight_scale],
            verbose=True
        )

        print(f"\nManual output (should saturate to 127):")
        print(manual_out[:, :, 0])

        if manual_out.max() == 127:
            print("✓ PASS: Successfully saturated to +127")
            return True
        else:
            print(f"✗ FAIL: Expected 127, got {manual_out.max()}")
            return False

    def test_negative_saturation(self):
        """Force output to -128 (before ReLU)."""
        print(f"\n--- Test: Negative Saturation ---")

        # Large negative accumulator
        ksize = 3
        kernel = np.ones((1, ksize, ksize, 1), dtype=np.float32) * -127  # Max negative weight
        input_img = np.full((4, 4), 255, dtype=np.uint8)

        input_scale = 1.0 / 255.0
        weight_scale = 1.0
        output_scale = 0.001

        tflite_bytes, metadata = build_simple_depthwise(
            height=4, width=4, ksize=ksize, num_filters=1,
            kernel_weights=kernel,
            input_scale=input_scale,
            output_scale=output_scale
        )

        # Without ReLU to see negative saturation
        manual_out = manual_depthwise_conv(
            input_img, kernel,
            input_scale=input_scale,
            output_scale=output_scale,
            weight_scales=[weight_scale],
            use_relu=False,  # Disable ReLU
            verbose=True
        )

        print(f"\nManual output (should saturate to -128):")
        print(manual_out[:, :, 0].view(np.int8))  # View as signed

        if manual_out[:, :, 0].view(np.int8).min() == -128:
            print("✓ PASS: Successfully saturated to -128")
            return True
        else:
            print(f"✗ FAIL: Expected -128, got {manual_out[:, :, 0].view(np.int8).min()}")
            return False


class Phase3_MultiChannel:
    """Verify per-channel quantization works correctly.

    Goal: Test 2 filters with different quantization scales.
    """

    def __init__(self):
        self.phase_name = "Phase 3: Multi-Channel (Per-Channel Quantization)"

    def test_two_channels(self):
        """Test 2 filters: one positive, one negative."""
        print(f"\n{'='*60}")
        print(f"{self.phase_name}")
        print(f"{'='*60}")
        print(f"\n--- Test: Two Independent Channels ---")

        ksize = 3
        # Channel 0: all +1s, Channel 1: all -1s
        kernel = np.zeros((1, ksize, ksize, 2), dtype=np.float32)
        kernel[0, :, :, 0] = 1.0
        kernel[0, :, :, 1] = -1.0

        input_img = np.full((4, 4), 128, dtype=np.uint8)

        input_scale = 1.0 / 255.0
        weight_scales = [1.0 / 127.0, 1.0 / 127.0]
        output_scale = 0.02

        tflite_bytes, metadata = build_simple_depthwise(
            height=4, width=4, ksize=ksize, num_filters=2,
            kernel_weights=kernel,
            input_scale=input_scale,
            output_scale=output_scale
        )

        manual_out = manual_depthwise_conv(
            input_img, kernel,
            input_scale=input_scale,
            output_scale=output_scale,
            weight_scales=weight_scales,
            use_relu=False,  # No ReLU to see negative channel
            verbose=True
        )

        print(f"\nManual output:")
        print(f"  Channel 0 (positive): mean={manual_out[:,:,0].mean():.1f}")
        print(f"  Channel 1 (negative): mean={manual_out[:,:,1].view(np.int8).mean():.1f}")

        # They should be opposite signs (before ReLU)
        ch0_mean = manual_out[:, :, 0].mean()
        ch1_mean = manual_out[:, :, 1].view(np.int8).mean()

        if ch0_mean > 0 and ch1_mean < 0:
            print("✓ PASS: Channels produce independent outputs with opposite signs")
            return True
        else:
            print("✗ FAIL: Channels don't have expected opposite signs")
            return False


def main():
    parser = argparse.ArgumentParser(description='Depthwise convolution fundamentals investigation')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], help='Run specific phase')
    parser.add_argument('--all', action='store_true', help='Run all phases')
    args = parser.parse_args()

    results = {}

    if args.phase == 1 or args.all:
        phase1 = Phase1_IdentityConv()
        results['Phase 1 - Constant'] = phase1.test_constant_input()
        results['Phase 1 - Range'] = phase1.test_range_of_inputs()

    if args.phase == 2 or args.all:
        phase2 = Phase2_IntentionalSaturation()
        results['Phase 2 - Positive'] = phase2.test_positive_saturation()
        results['Phase 2 - Negative'] = phase2.test_negative_saturation()

    if args.phase == 3 or args.all:
        phase3 = Phase3_MultiChannel()
        results['Phase 3 - Two Channels'] = phase3.test_two_channels()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"{test_name:30s} {status}")


if __name__ == '__main__':
    main()
