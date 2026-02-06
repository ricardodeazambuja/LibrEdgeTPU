"""Compare Edge TPU vs TFLite CPU interpreter for optical flow.

This uses the TFLite CPU interpreter as ground truth to verify:
1. The model file is correct
2. The Edge TPU is computing the same thing
3. Any discrepancies are in our driver, not the model
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu import OpticalFlow
from libredgetpu.optical_flow.templates import get_pooled_template

try:
    import tensorflow as tf
except ImportError:
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)


def create_test_image(h, w, seed=42):
    """Create a random test image."""
    np.random.seed(seed)
    return (np.random.rand(h, w) * 255).astype(np.uint8)


def run_tflite_cpu(tflite_path, image):
    """Run TFLite model on CPU interpreter."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Prepare input
    if image.ndim == 2:
        image = image.reshape(1, image.shape[0], image.shape[1], 1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = image.reshape(1, image.shape[0], image.shape[1], 1)

    # Set input
    interpreter.set_tensor(input_details['index'], image)

    # Run
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details['index'])

    # Dequantize (CPU interpreter returns quantized uint8)
    scale = output_details['quantization'][0]
    zero_point = output_details['quantization'][1]

    output_float = (output.astype(np.float32) - zero_point) * scale

    return output_float.squeeze()


def main():
    print("="*60)
    print("TFLite CPU vs Edge TPU Comparison")
    print("="*60)

    size = 64
    tflite_path, json_path = get_pooled_template(size, pool_factor=4)

    print(f"\nModel: {tflite_path}")

    # Create test image
    image = create_test_image(size, size)
    print(f"Test image: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")

    # Run on TFLite CPU
    print("\n--- TFLite CPU Interpreter ---")
    try:
        feat_cpu = run_tflite_cpu(tflite_path, image)
        print(f"Output shape: {feat_cpu.shape}")
        print(f"Output range: [{feat_cpu.min():.3f}, {feat_cpu.max():.3f}]")
        print(f"Output mean/std: {feat_cpu.mean():.3f} / {feat_cpu.std():.3f}")
    except Exception as e:
        print(f"❌ TFLite CPU failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run on Edge TPU
    print("\n--- Edge TPU ---")
    try:
        flow = OpticalFlow.from_template(size, pooled=True)
        flow.open()

        feat_tpu = flow.extract_features(image)
        print(f"Output shape: {feat_tpu.shape}")
        print(f"Output range: [{feat_tpu.min():.3f}, {feat_tpu.max():.3f}]")
        print(f"Output mean/std: {feat_tpu.mean():.3f} / {feat_tpu.std():.3f}")

        flow.close()
    except Exception as e:
        print(f"❌ Edge TPU failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare
    print("\n--- Comparison ---")
    print(f"Shape match: {feat_cpu.shape == feat_tpu.shape}")

    if feat_cpu.shape == feat_tpu.shape:
        # Compute correlation
        corr = np.corrcoef(feat_cpu.ravel(), feat_tpu.ravel())[0, 1]
        print(f"Pearson correlation: {corr:.6f}")

        # MSE
        mse = np.mean((feat_cpu - feat_tpu) ** 2)
        rmse = np.sqrt(mse)
        print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")

        # Max absolute error
        max_err = np.max(np.abs(feat_cpu - feat_tpu))
        print(f"Max absolute error: {max_err:.6f}")

        if corr > 0.99:
            print("\n✅ Edge TPU matches TFLite CPU reference (correlation > 0.99)")
        elif corr > 0.95:
            print("\n⚠️  Edge TPU mostly matches TFLite CPU (correlation > 0.95)")
        else:
            print(f"\n❌ Edge TPU DOES NOT match TFLite CPU (correlation = {corr:.6f})")

            # Per-channel comparison
            print("\nPer-channel correlation:")
            for c in range(feat_cpu.shape[2]):
                cpu_c = feat_cpu[:, :, c].ravel()
                tpu_c = feat_tpu[:, :, c].ravel()
                corr_c = np.corrcoef(cpu_c, tpu_c)[0, 1]
                print(f"  Channel {c}: {corr_c:.6f}")
    else:
        print(f"❌ Shape mismatch: CPU={feat_cpu.shape}, TPU={feat_tpu.shape}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
