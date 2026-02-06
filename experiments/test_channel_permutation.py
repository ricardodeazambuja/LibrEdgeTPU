"""Test if Edge TPU channels are permuted relative to TFLite CPU.

Checks all possible channel permutations to see if the Edge TPU output
is just a reordering of the correct channels.
"""

import numpy as np
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.tflite_builder import build_optical_flow_pooled
from libredgetpu import OpticalFlow

try:
    import tensorflow as tf
except ImportError:
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)


def create_test_image(h, w, seed=42):
    """Create a random test image."""
    np.random.seed(seed)
    return (np.random.rand(h, w) * 255).astype(np.uint8)


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


def main():
    print("="*60)
    print("Channel Permutation Test")
    print("="*60)

    size = 64
    pool_factor = 4

    # Generate uncompiled TFLite model
    print(f"\nGenerating models...")
    tflite_bytes, metadata = build_optical_flow_pooled(size, size, pool_factor=pool_factor)

    # Create test image
    image = create_test_image(size, size)

    # Run on TFLite CPU (ground truth)
    print("Running TFLite CPU...")
    feat_cpu = run_tflite_cpu(tflite_bytes, image)
    print(f"CPU output shape: {feat_cpu.shape}")

    # Run on Edge TPU
    print("Running Edge TPU...")
    flow = OpticalFlow.from_template(size, pooled=True)
    flow.open()
    feat_tpu = flow._extract_features_uint8(image)
    flow.close()
    print(f"TPU output shape: {feat_tpu.shape}")

    # Check all possible 8-channel permutations
    n_channels = feat_cpu.shape[2]
    print(f"\nTesting {n_channels}! = {np.math.factorial(n_channels)} permutations...")
    print("(This may take a while for 8 channels...)")

    best_perm = None
    best_corr = -1.0
    best_mae = float('inf')

    # For 8 channels, 8! = 40320 is tractable
    for perm in itertools.permutations(range(n_channels)):
        # Permute TPU channels
        feat_tpu_perm = feat_tpu[:, :, list(perm)]

        # Compute correlation
        cpu_flat = feat_cpu.ravel().astype(np.float32)
        tpu_flat = feat_tpu_perm.ravel().astype(np.float32)
        corr = np.corrcoef(cpu_flat, tpu_flat)[0, 1]

        # Compute MAE
        mae = np.abs(feat_cpu.astype(np.int16) - feat_tpu_perm.astype(np.int16)).mean()

        if corr > best_corr:
            best_corr = corr
            best_perm = perm
            best_mae = mae

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    if best_perm == tuple(range(n_channels)):
        print(f"\n✅ No permutation needed (identity permutation is best)")
        print(f"   Correlation: {best_corr:.6f}")
        print(f"   MAE: {best_mae:.3f}")
    else:
        print(f"\n⚠️  PERMUTATION DETECTED!")
        print(f"   Best permutation: {best_perm}")
        print(f"   (TPU channel i contains CPU channel {best_perm[i]})")
        print(f"   Correlation: {best_corr:.6f}")
        print(f"   MAE: {best_mae:.3f}")

        # Show mapping
        print(f"\n   Channel mapping (TPU <- CPU):")
        for tpu_idx, cpu_idx in enumerate(best_perm):
            print(f"     TPU channel {tpu_idx} <- CPU channel {cpu_idx}")

    # Compare with no permutation
    feat_tpu_noperm = feat_tpu
    cpu_flat = feat_cpu.ravel().astype(np.float32)
    tpu_flat = feat_tpu_noperm.ravel().astype(np.float32)
    corr_noperm = np.corrcoef(cpu_flat, tpu_flat)[0, 1]
    mae_noperm = np.abs(feat_cpu.astype(np.int16) - feat_tpu_noperm.astype(np.int16)).mean()

    print(f"\n   No permutation (current):")
    print(f"     Correlation: {corr_noperm:.6f}")
    print(f"     MAE: {mae_noperm:.3f}")

    print(f"\n   Improvement with best permutation:")
    print(f"     Δ Correlation: +{best_corr - corr_noperm:.6f}")
    print(f"     Δ MAE: {mae_noperm - best_mae:.3f}")

    if best_corr > 0.99:
        print(f"\n✅ Perfect match found (correlation > 0.99)!")
        print(f"   The Edge TPU is computing correctly, just with permuted channels.")
    elif best_corr > 0.95:
        print(f"\n⚠️  Good match (correlation > 0.95), likely a permutation issue.")
    else:
        print(f"\n❌ No good permutation found (best correlation = {best_corr:.6f})")
        print(f"   The issue is more than just channel reordering.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
