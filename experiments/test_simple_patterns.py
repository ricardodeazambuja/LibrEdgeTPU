"""Test Edge TPU vs TFLite CPU with simple pattern inputs.

Uses constant images (all zeros, all 255, checkerboard) to isolate
whether the issue is in conv, pooling, or quantization.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.tflite_builder import build_optical_flow_pooled
from libredgetpu import OpticalFlow

try:
    import tensorflow as tf
except ImportError:
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)


def run_tflite_cpu(tflite_bytes, image):
    """Run uncompiled TFLite model on CPU interpreter, return uint8."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Prepare input
    if image.ndim == 2:
        image = image.reshape(1, image.shape[0], image.shape[1], 1)

    # Set input and run
    interpreter.set_tensor(input_details['index'], image)
    interpreter.invoke()

    # Get raw uint8 output
    output = interpreter.get_tensor(output_details['index'])
    return output.squeeze()


def test_pattern(name, image, tflite_bytes, flow):
    """Test a single pattern."""
    print(f"\n{'='*60}")
    print(f"Pattern: {name}")
    print(f"{'='*60}")
    print(f"Input: shape={image.shape}, min={image.min()}, max={image.max()}, mean={image.mean():.1f}")

    # Run CPU
    feat_cpu = run_tflite_cpu(tflite_bytes, image)

    # Run TPU
    feat_tpu = flow._extract_features_uint8(image)

    print(f"\nCPU output: min={feat_cpu.min()}, max={feat_cpu.max()}, mean={feat_cpu.mean():.1f}")
    print(f"TPU output: min={feat_tpu.min()}, max={feat_tpu.max()}, mean={feat_tpu.mean():.1f}")

    # Per-channel comparison
    print(f"\nPer-channel uint8 values:")
    print(f"{'Channel':<10} {'CPU mean':<15} {'TPU mean':<15} {'MAE':<10} {'Max diff':<10}")
    print("-" * 60)
    for c in range(8):
        cpu_c = feat_cpu[:, :, c]
        tpu_c = feat_tpu[:, :, c]
        cpu_mean = cpu_c.mean()
        tpu_mean = tpu_c.mean()
        mae = np.abs(cpu_c.astype(np.int16) - tpu_c.astype(np.int16)).mean()
        max_diff = np.abs(cpu_c.astype(np.int16) - tpu_c.astype(np.int16)).max()
        print(f"{c:<10} {cpu_mean:<15.1f} {tpu_mean:<15.1f} {mae:<10.1f} {max_diff:<10}")

    # Overall comparison
    diff = feat_cpu.astype(np.int16) - feat_tpu.astype(np.int16)
    mae = np.abs(diff).mean()
    max_diff = np.abs(diff).max()
    frac_equal = np.mean(diff == 0)

    print(f"\nOverall:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Max diff: {max_diff}")
    print(f"  Fraction equal: {frac_equal:.4f}")

    if mae < 1.0:
        print(f"  ✅ Excellent match (MAE < 1.0)")
    elif mae < 5.0:
        print(f"  ✅ Good match (MAE < 5.0)")
    elif mae < 20.0:
        print(f"  ⚠️  Moderate mismatch (MAE < 20.0)")
    else:
        print(f"  ❌ Poor match (MAE >= 20.0)")


def main():
    print("="*60)
    print("Simple Pattern Test")
    print("="*60)

    size = 64
    pool_factor = 4

    # Generate models
    print(f"\nGenerating models...")
    tflite_bytes, metadata = build_optical_flow_pooled(size, size, pool_factor=pool_factor)

    # Initialize Edge TPU
    flow = OpticalFlow.from_template(size, pooled=True)
    flow.open()

    # Test patterns
    patterns = [
        ("All zeros", np.zeros((size, size), dtype=np.uint8)),
        ("All 128 (mid-gray)", np.full((size, size), 128, dtype=np.uint8)),
        ("All 255", np.full((size, size), 255, dtype=np.uint8)),
        ("Horizontal gradient", np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))),
        ("Vertical gradient", np.tile(np.linspace(0, 255, size, dtype=np.uint8).reshape(-1, 1), (1, size))),
        ("Checkerboard 8x8", np.kron([[128*((i+j)%2) for i in range(8)] for j in range(8)], np.ones((8, 8), dtype=np.uint8))),
        ("Random (seed=42)", (np.random.RandomState(42).rand(size, size) * 255).astype(np.uint8)),
    ]

    for name, image in patterns:
        test_pattern(name, image, tflite_bytes, flow)

    flow.close()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print("If MAE is low for constant inputs but high for random:")
    print("  → Issue is in Conv2D Gabor feature extraction")
    print("If MAE is high even for constant inputs:")
    print("  → Issue is in quantization or pooling")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
