"""Test if Edge TPU output is spatially flipped/transposed vs TFLite CPU.

Checks various spatial transformations (flip H, flip V, transpose, rotate).
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


def main():
    print("="*60)
    print("Spatial Transform Test")
    print("="*60)

    size = 64
    pool_factor = 4

    # Generate models
    print(f"\nGenerating models...")
    tflite_bytes, metadata = build_optical_flow_pooled(size, size, pool_factor=pool_factor)

    # Initialize Edge TPU
    flow = OpticalFlow.from_template(size, pooled=True)
    flow.open()

    # Create a spatially-varying test pattern (horizontal gradient)
    np.random.seed(42)
    image = (np.random.rand(size, size) * 255).astype(np.uint8)
    print(f"Test image: shape={image.shape}, dtype={image.dtype}")

    # Run CPU and TPU
    print("\nRunning CPU...")
    feat_cpu = run_tflite_cpu(tflite_bytes, image)

    print("Running TPU...")
    feat_tpu = flow._extract_features_uint8(image)

    flow.close()

    print(f"\nCPU shape: {feat_cpu.shape}")
    print(f"TPU shape: {feat_tpu.shape}")

    # Baseline (no transform)
    diff = feat_cpu.astype(np.int16) - feat_tpu.astype(np.int16)
    mae_baseline = np.abs(diff).mean()
    print(f"\nBaseline (no transform): MAE = {mae_baseline:.3f}")

    # Test various spatial transforms
    transforms = []

    # Flips
    for axis in [(0,), (1,), (0, 1)]:
        name = f"flip axis {axis}"
        feat_tpu_t = np.flip(feat_tpu, axis=axis)
        diff = feat_cpu.astype(np.int16) - feat_tpu_t.astype(np.int16)
        mae = np.abs(diff).mean()
        transforms.append((name, mae))

    # Transpose (swap H and W)
    feat_tpu_t = np.transpose(feat_tpu, (1, 0, 2))
    diff = feat_cpu.astype(np.int16) - feat_tpu_t.astype(np.int16)
    mae = np.abs(diff).mean()
    transforms.append(("transpose (H<->W)", mae))

    # Rotations (90, 180, 270 degrees)
    for k in [1, 2, 3]:
        name = f"rotate {k*90}°"
        feat_tpu_t = np.rot90(feat_tpu, k=k, axes=(0, 1))
        diff = feat_cpu.astype(np.int16) - feat_tpu_t.astype(np.int16)
        mae = np.abs(diff).mean()
        transforms.append((name, mae))

    # Combined transforms
    # Transpose + flip
    for axis in [(0,), (1,)]:
        name = f"transpose + flip axis {axis}"
        feat_tpu_t = np.flip(np.transpose(feat_tpu, (1, 0, 2)), axis=axis)
        diff = feat_cpu.astype(np.int16) - feat_tpu_t.astype(np.int16)
        mae = np.abs(diff).mean()
        transforms.append((name, mae))

    # Sort by MAE
    transforms.sort(key=lambda x: x[1])

    print(f"\n{'='*60}")
    print("Results (sorted by MAE)")
    print(f"{'='*60}")
    print(f"{'Transform':<30} {'MAE':<10}")
    print("-" * 60)
    for name, mae in transforms:
        marker = "✅" if mae < 1.0 else ("⚠️" if mae < mae_baseline * 0.9 else "")
        print(f"{name:<30} {mae:<10.3f} {marker}")

    best_name, best_mae = transforms[0]
    if best_mae < 1.0:
        print(f"\n✅ Found perfect match: {best_name} (MAE = {best_mae:.3f})")
    elif best_mae < mae_baseline * 0.5:
        print(f"\n⚠️  Found significant improvement: {best_name}")
        print(f"   MAE: {best_mae:.3f} (baseline: {mae_baseline:.3f})")
        print(f"   Improvement: {(1 - best_mae/mae_baseline)*100:.1f}%")
    else:
        print(f"\n❌ No good spatial transform found")
        print(f"   Best: {best_name} with MAE = {best_mae:.3f}")
        print(f"   Baseline: {mae_baseline:.3f}")
        print(f"   The issue is not a simple spatial transformation.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
