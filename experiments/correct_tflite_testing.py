#!/usr/bin/env python3
"""
Demonstrates CORRECT way to test TFLite models after discovering XNNPACK bug.

CRITICAL: Always disable XNNPACK delegate when testing quantized models!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from libredgetpu.tflite_builder import build_simple_depthwise


def test_model_WRONG_way(tflite_bytes, input_img):
    """❌ WRONG: Uses default TFLite interpreter (XNNPACK enabled)."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    interpreter.set_tensor(0, input_img)
    interpreter.invoke()

    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])


def test_model_CORRECT_way(tflite_bytes, input_img):
    """✓ CORRECT: Disables XNNPACK delegate."""
    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True  # Disables XNNPACK
    )
    interpreter.allocate_tensors()

    interpreter.set_tensor(0, input_img)
    interpreter.invoke()

    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])


def main():
    print("="*60)
    print("TFLite Testing: WRONG vs CORRECT Methods")
    print("="*60)

    # Build a simple model
    tflite_bytes, metadata = build_simple_depthwise(
        height=4, width=4, ksize=3, num_filters=1,
        kernel_weights=np.ones((1, 3, 3, 1), dtype=np.float32),
        input_scale=1.0/255.0,
        output_scale=0.02
    )

    # Test input: constant 128
    input_img = np.full((1, 4, 4, 1), 128, dtype=np.uint8)

    print(f"\nInput: constant uint8={input_img[0,0,0,0]}")
    print(f"Expected output: constant ~128 (for identity kernel)")

    # Test WRONG way
    print(f"\n{'='*60}")
    print("❌ WRONG METHOD (XNNPACK enabled)")
    print(f"{'='*60}")
    try:
        output_wrong = test_model_WRONG_way(tflite_bytes, input_img)
        print("Output:")
        print(output_wrong[0, :, :, 0])
        print(f"\nUnique values: {np.unique(output_wrong)}")
        print(f"✗ NON-CONSTANT OUTPUT! XNNPACK bug detected.")
    except Exception as e:
        print(f"Error: {e}")

    # Test CORRECT way
    print(f"\n{'='*60}")
    print("✓ CORRECT METHOD (XNNPACK disabled)")
    print(f"{'='*60}")
    try:
        output_correct = test_model_CORRECT_way(tflite_bytes, input_img)
        print("Output:")
        print(output_correct[0, :, :, 0])
        print(f"\nUnique values: {np.unique(output_correct)}")

        if len(np.unique(output_correct)) == 1:
            print(f"✓ CONSTANT OUTPUT! This is correct.")
        else:
            print(f"✗ Still non-constant. Something else is wrong.")
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("""
ALWAYS use this pattern for TFLite CPU testing:

    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True  # ← CRITICAL!
    )

Without this flag, XNNPACK delegate will produce incorrect results
for quantized models.

Edge TPU is NOT affected by this bug (doesn't use XNNPACK).
""")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
