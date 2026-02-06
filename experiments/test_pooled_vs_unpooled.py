"""Compare pooled vs non-pooled optical flow models.

Tests if the bug is specific to the fused Conv+Pool model or also
present in the standard Conv-only model.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.tflite_builder import build_optical_flow, build_optical_flow_pooled
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


def test_model(name, tflite_bytes, flow, image):
    """Test a single model variant."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Run CPU
    print("Running TFLite CPU...")
    feat_cpu = run_tflite_cpu(tflite_bytes, image)

    # Run TPU
    print("Running Edge TPU...")
    feat_tpu = flow._extract_features_uint8(image)

    print(f"\nCPU output: shape={feat_cpu.shape}, range=[{feat_cpu.min()}, {feat_cpu.max()}], mean={feat_cpu.mean():.1f}")
    print(f"TPU output: shape={feat_tpu.shape}, range=[{feat_tpu.min()}, {feat_tpu.max()}], mean={feat_tpu.mean():.1f}")

    # Overall comparison
    if feat_cpu.shape != feat_tpu.shape:
        print(f"❌ Shape mismatch!")
        return

    diff = feat_cpu.astype(np.int16) - feat_tpu.astype(np.int16)
    mae = np.abs(diff).mean()
    max_diff = np.abs(diff).max()
    frac_equal = np.mean(diff == 0)

    corr_flat = np.corrcoef(feat_cpu.ravel().astype(np.float32),
                            feat_tpu.ravel().astype(np.float32))[0, 1]

    print(f"\nOverall comparison:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Max diff: {max_diff}")
    print(f"  Fraction equal: {frac_equal:.4f}")
    print(f"  Correlation: {corr_flat:.6f}")

    # Per-channel comparison
    print(f"\nPer-channel MAE:")
    for c in range(feat_cpu.shape[2]):
        cpu_c = feat_cpu[:, :, c]
        tpu_c = feat_tpu[:, :, c]
        mae_c = np.abs(cpu_c.astype(np.int16) - tpu_c.astype(np.int16)).mean()
        corr_c = np.corrcoef(cpu_c.ravel().astype(np.float32),
                             tpu_c.ravel().astype(np.float32))[0, 1]
        marker = "✅" if mae_c < 1.0 else ("⚠️" if mae_c < 10.0 else "❌")
        print(f"  Channel {c}: MAE={mae_c:6.1f}, corr={corr_c:.3f} {marker}")

    if mae < 1.0:
        print(f"\n✅ Excellent match (MAE < 1.0)")
        return "good"
    elif mae < 5.0:
        print(f"\n✅ Good match (MAE < 5.0)")
        return "good"
    elif mae < 20.0:
        print(f"\n⚠️  Moderate mismatch (MAE < 20.0)")
        return "moderate"
    else:
        print(f"\n❌ Poor match (MAE >= 20.0)")
        return "bad"


def main():
    print("="*60)
    print("Pooled vs Non-Pooled Model Comparison")
    print("="*60)

    size = 64
    pool_factor = 4

    # Create test image
    np.random.seed(42)
    image = (np.random.rand(size, size) * 255).astype(np.uint8)
    print(f"\nTest image: shape={image.shape}, dtype={image.dtype}")

    # Test 1: Non-pooled model
    print(f"\n\n{'#'*60}")
    print("# TEST 1: NON-POOLED MODEL (Conv only)")
    print(f"{'#'*60}")

    tflite_unpooled, meta_unpooled = build_optical_flow(size, size)
    print(f"Model size: {len(tflite_unpooled)} bytes")

    try:
        flow_unpooled = OpticalFlow.from_template(size, pooled=False)
        flow_unpooled.open()
        result_unpooled = test_model("Non-pooled (Conv only)", tflite_unpooled, flow_unpooled, image)
        flow_unpooled.close()
    except Exception as e:
        print(f"❌ Non-pooled test failed: {e}")
        import traceback
        traceback.print_exc()
        result_unpooled = "error"

    # Test 2: Pooled model
    print(f"\n\n{'#'*60}")
    print("# TEST 2: POOLED MODEL (Conv + AVG_POOL)")
    print(f"{'#'*60}")

    tflite_pooled, meta_pooled = build_optical_flow_pooled(size, size, pool_factor=pool_factor)
    print(f"Model size: {len(tflite_pooled)} bytes")

    try:
        flow_pooled = OpticalFlow.from_template(size, pooled=True)
        flow_pooled.open()
        result_pooled = test_model("Pooled (Conv + AVG_POOL)", tflite_pooled, flow_pooled, image)
        flow_pooled.close()
    except Exception as e:
        print(f"❌ Pooled test failed: {e}")
        import traceback
        traceback.print_exc()
        result_pooled = "error"

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Non-pooled model: {result_unpooled}")
    print(f"Pooled model: {result_pooled}")

    if result_unpooled == "good" and result_pooled != "good":
        print(f"\n🔍 BUG ISOLATED: The issue is specific to the POOLED model!")
        print(f"   → The edgetpu_compiler or Edge TPU has a bug in Conv+AVG_POOL fusion")
    elif result_unpooled != "good" and result_pooled != "good":
        print(f"\n🔍 BUG IS GENERAL: Both models have issues")
        print(f"   → The issue is in the Conv2D (Gabor) or basic quantization")
    elif result_unpooled == "good" and result_pooled == "good":
        print(f"\n✅ Both models work correctly!")
    else:
        print(f"\n🤷 Mixed results - needs more investigation")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
