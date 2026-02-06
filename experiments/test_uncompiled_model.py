"""Compare uncompiled TFLite model (CPU) vs compiled Edge TPU model.

This generates a fresh uncompiled model, runs it on TFLite CPU interpreter,
then compares with the compiled Edge TPU model to identify discrepancies.
"""

import numpy as np
import sys
import os
import tempfile

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
    """Run uncompiled TFLite model on CPU interpreter."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"\nTFLite CPU Interpreter:")
    print(f"  Input: {input_details['shape']}, dtype={input_details['dtype']}")
    print(f"    Quantization: scale={input_details['quantization'][0]}, zp={input_details['quantization'][1]}")
    print(f"  Output: {output_details['shape']}, dtype={output_details['dtype']}")
    print(f"    Quantization: scale={output_details['quantization'][0]}, zp={output_details['quantization'][1]}")

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

    # Return raw uint8 and dequantized float32
    scale = output_details['quantization'][0]
    zero_point = output_details['quantization'][1]

    output_float = (output.astype(np.float32) - zero_point) * scale

    return output.squeeze(), output_float.squeeze()


def main():
    print("="*60)
    print("Uncompiled TFLite CPU vs Compiled Edge TPU")
    print("="*60)

    size = 64
    pool_factor = 4

    # Generate uncompiled TFLite model
    print(f"\nGenerating uncompiled TFLite model ({size}x{size}, pool={pool_factor})...")
    tflite_bytes, metadata = build_optical_flow_pooled(size, size, pool_factor=pool_factor)
    print(f"Model size: {len(tflite_bytes)} bytes")
    print(f"Metadata: {metadata}")

    # Create test image
    image = create_test_image(size, size)
    print(f"\nTest image: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")

    # Run on uncompiled TFLite CPU
    print("\n" + "="*60)
    print("Running on UNCOMPILED TFLite CPU")
    print("="*60)
    try:
        feat_cpu_uint8, feat_cpu_float = run_tflite_cpu(tflite_bytes, image)
        print(f"\nRaw uint8 output:")
        print(f"  Shape: {feat_cpu_uint8.shape}")
        print(f"  Range: [{feat_cpu_uint8.min()}, {feat_cpu_uint8.max()}]")
        print(f"  Mean/std: {feat_cpu_uint8.mean():.3f} / {feat_cpu_uint8.std():.3f}")
        print(f"\nDequantized float32 output:")
        print(f"  Shape: {feat_cpu_float.shape}")
        print(f"  Range: [{feat_cpu_float.min():.3f}, {feat_cpu_float.max():.3f}]")
        print(f"  Mean/std: {feat_cpu_float.mean():.3f} / {feat_cpu_float.std():.3f}")
    except Exception as e:
        print(f"❌ TFLite CPU failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run on compiled Edge TPU
    print("\n" + "="*60)
    print("Running on COMPILED Edge TPU")
    print("="*60)
    try:
        flow = OpticalFlow.from_template(size, pooled=True)
        flow.open()

        print(f"\nEdge TPU model:")
        print(f"  Input: shape={flow._input_info.shape}, scale={flow._input_info.scale}, zp={flow._input_info.zero_point}")
        print(f"  Output: shape={flow._output_info.shape}, scale={flow._output_info.scale}, zp={flow._output_info.zero_point}")

        # Get raw uint8 output
        feat_tpu_uint8 = flow._extract_features_uint8(image)
        # Get dequantized float32 output
        feat_tpu_float = flow.extract_features(image)

        print(f"\nRaw uint8 output:")
        print(f"  Shape: {feat_tpu_uint8.shape}")
        print(f"  Range: [{feat_tpu_uint8.min()}, {feat_tpu_uint8.max()}]")
        print(f"  Mean/std: {feat_tpu_uint8.mean():.3f} / {feat_tpu_uint8.std():.3f}")
        print(f"\nDequantized float32 output:")
        print(f"  Shape: {feat_tpu_float.shape}")
        print(f"  Range: [{feat_tpu_float.min():.3f}, {feat_tpu_float.max():.3f}]")
        print(f"  Mean/std: {feat_tpu_float.mean():.3f} / {feat_tpu_float.std():.3f}")

        flow.close()
    except Exception as e:
        print(f"❌ Edge TPU failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    print(f"\nShape match (uint8): {feat_cpu_uint8.shape == feat_tpu_uint8.shape}")
    print(f"Shape match (float32): {feat_cpu_float.shape == feat_tpu_float.shape}")

    if feat_cpu_uint8.shape == feat_tpu_uint8.shape:
        # Compare uint8 (raw quantized values)
        print("\n--- Raw uint8 comparison ---")
        diff = feat_cpu_uint8.astype(np.int16) - feat_tpu_uint8.astype(np.int16)
        print(f"Difference range: [{diff.min()}, {diff.max()}]")
        print(f"Mean absolute difference: {np.abs(diff).mean():.3f}")
        print(f"Max absolute difference: {np.abs(diff).max()}")
        print(f"Fraction exactly equal: {np.mean(diff == 0):.4f}")

        # Per-channel stats
        print("\nPer-channel uint8 MAE:")
        for c in range(feat_cpu_uint8.shape[2]):
            mae = np.abs(feat_cpu_uint8[:, :, c] - feat_tpu_uint8[:, :, c]).mean()
            print(f"  Channel {c}: {mae:.3f}")

    if feat_cpu_float.shape == feat_tpu_float.shape:
        # Compare float32 (dequantized)
        print("\n--- Dequantized float32 comparison ---")
        corr = np.corrcoef(feat_cpu_float.ravel(), feat_tpu_float.ravel())[0, 1]
        print(f"Pearson correlation: {corr:.6f}")

        mse = np.mean((feat_cpu_float - feat_tpu_float) ** 2)
        rmse = np.sqrt(mse)
        print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")

        max_err = np.max(np.abs(feat_cpu_float - feat_tpu_float))
        print(f"Max absolute error: {max_err:.6f}")

        # Per-channel correlation
        print("\nPer-channel float32 correlation:")
        for c in range(feat_cpu_float.shape[2]):
            cpu_c = feat_cpu_float[:, :, c].ravel()
            tpu_c = feat_tpu_float[:, :, c].ravel()
            corr_c = np.corrcoef(cpu_c, tpu_c)[0, 1]
            print(f"  Channel {c}: {corr_c:.6f}")

        if corr > 0.99:
            print("\n✅ Edge TPU matches TFLite CPU reference (correlation > 0.99)")
        elif corr > 0.95:
            print("\n⚠️  Edge TPU mostly matches TFLite CPU (correlation > 0.95)")
        else:
            print(f"\n❌ Edge TPU DOES NOT match TFLite CPU (correlation = {corr:.6f})")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
