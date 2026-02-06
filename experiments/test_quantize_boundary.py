#!/usr/bin/env python3
"""Phase 1.5: Characterize QUANTIZE operator boundary instability.

Test QUANTIZE operator with range of input values to identify where
instability occurs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from libredgetpu.tflite_builder import build_simple_depthwise

def test_quantize_stability(input_value):
    """Test QUANTIZE operator with specific input value."""
    # Build model
    tflite_bytes, metadata = build_simple_depthwise(
        height=4, width=4, ksize=3, num_filters=1,
        kernel_weights=np.ones((1, 3, 3, 1), dtype=np.float32),
        input_scale=1.0/255.0,
        output_scale=0.02
    )

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    # Prepare input: constant value
    input_img = np.full((1, 4, 4, 1), input_value, dtype=np.uint8)

    # Set input and run
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details['index'], input_img)
    interpreter.invoke()

    # Get quantize_out tensor
    quantize_out = interpreter.get_tensor(1)  # Tensor 1 is quantize_out

    # Analyze
    unique_vals = np.unique(quantize_out)
    is_constant = len(unique_vals) == 1
    value_range = quantize_out.max() - quantize_out.min()

    # Theoretical expected value
    # uint8 → int8: round((uint8 - 0) * (1/255) / (1/255) + (-128))
    #             = round(uint8 - 128)
    expected = int(input_value) - 128

    return {
        'input': input_value,
        'expected': expected,
        'is_constant': is_constant,
        'unique_vals': unique_vals.tolist(),
        'value_range': int(value_range),
        'min': int(quantize_out.min()),
        'max': int(quantize_out.max()),
        'mean': float(quantize_out.mean()),
        'std': float(quantize_out.std()),
    }


def main():
    print("="*60)
    print("Phase 1.5: QUANTIZE Operator Boundary Instability")
    print("="*60)

    # Test range of input values
    test_values = [0, 32, 64, 96, 127, 128, 129, 160, 192, 224, 255]

    print(f"\n{'Input':>5} | {'Expected':>8} | {'Constant':>8} | {'Range':>5} | {'Min':>4} | {'Max':>4} | {'Mean':>7} | {'Std':>7}")
    print("-" * 80)

    results = []
    for val in test_values:
        result = test_quantize_stability(val)
        results.append(result)

        const_str = "✓ Yes" if result['is_constant'] else "✗ No"
        print(f"{result['input']:>5} | {result['expected']:>8} | {const_str:>8} | "
              f"{result['value_range']:>5} | {result['min']:>4} | {result['max']:>4} | "
              f"{result['mean']:>7.1f} | {result['std']:>7.2f}")

    # Detailed analysis of problem cases
    print(f"\n{'='*60}")
    print("Detailed Analysis of Non-Constant Cases")
    print(f"{'='*60}")

    for result in results:
        if not result['is_constant']:
            print(f"\n### Input uint8={result['input']} (expected int8={result['expected']}) ###")
            print(f"  Unique values: {result['unique_vals']}")
            print(f"  Range: {result['value_range']} (span of {result['max'] - result['min']})")
            print(f"  Mean: {result['mean']:.2f} (vs expected {result['expected']})")
            print(f"  Std: {result['std']:.2f}")

            # Get full tensor pattern
            tflite_bytes, _ = build_simple_depthwise(
                height=4, width=4, ksize=3, num_filters=1,
                kernel_weights=np.ones((1, 3, 3, 1), dtype=np.float32),
                input_scale=1.0/255.0, output_scale=0.02
            )
            interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
            interpreter.allocate_tensors()
            input_img = np.full((1, 4, 4, 1), result['input'], dtype=np.uint8)
            interpreter.set_tensor(0, input_img)
            interpreter.invoke()
            quantize_out = interpreter.get_tensor(1)

            print(f"  Spatial pattern:")
            print("  " + str(quantize_out[0, :, :, 0]).replace('\n', '\n  '))

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    unstable = [r for r in results if not r['is_constant']]
    stable = [r for r in results if r['is_constant']]

    print(f"\nStable inputs ({len(stable)}): {[r['input'] for r in stable]}")
    print(f"Unstable inputs ({len(unstable)}): {[r['input'] for r in unstable]}")

    if unstable:
        print(f"\nWorst instability:")
        worst = max(unstable, key=lambda r: r['value_range'])
        print(f"  Input: {worst['input']}")
        print(f"  Expected: {worst['expected']}")
        print(f"  Actual range: [{worst['min']}, {worst['max']}] (span {worst['value_range']})")
        print(f"  Error: {abs(worst['mean'] - worst['expected']):.2f}")

    # Check if instability is symmetric around 128
    around_128 = [r for r in results if 127 <= r['input'] <= 129]
    print(f"\nInstability around uint8=128:")
    for r in around_128:
        status = "UNSTABLE" if not r['is_constant'] else "stable"
        print(f"  {r['input']:>3}: {status:>10} (range={r['value_range']}, mean={r['mean']:.1f})")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
