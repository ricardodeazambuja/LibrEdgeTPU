"""Test if TFLite CPU interpreter works with the corrected OHWI model."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.tflite_builder import build_optical_flow
from libredgetpu.optical_flow.templates import get_template

try:
    import tensorflow as tf
except ImportError:
    print("⚠️  TensorFlow not installed")
    sys.exit(1)

def manual_gabor_conv_cpu(image):
    """Manual Gabor (ground truth)."""
    from libredgetpu.tflite_builder import _generate_gabor_kernels
    import scipy.ndimage

    kernels = _generate_gabor_kernels(7, 4, (1.5, 3.0))
    n_filters = 8
    h, w = image.shape

    input_int8 = image.astype(np.int16) - 128

    gabor_abs_max = float(np.max(np.abs(kernels)))
    gabor_weight_scale = max(gabor_abs_max, 1e-6) / 127.0
    kernels_int8 = np.clip(np.round(kernels / gabor_weight_scale), -127, 127).astype(np.int8)

    conv_output_max = 1.0 * gabor_weight_scale * 7 * 7 * 127
    conv_output_scale = conv_output_max / 127.0

    features = np.zeros((h, w, n_filters), dtype=np.float32)
    for i in range(n_filters):
        kernel = kernels_int8[:, :, 0, i].astype(np.float32)
        conv_int = scipy.ndimage.convolve(input_int8.astype(np.float32), kernel, mode='constant', cval=0.0)
        conv_float = conv_int * gabor_weight_scale
        conv_float = np.maximum(conv_float, 0.0)
        features[:, :, i] = conv_float

    features_uint8 = np.clip(np.round(features / conv_output_scale), 0, 255).astype(np.uint8)
    return features_uint8


def run_tflite_cpu(tflite_bytes, image):
    """Run TFLite CPU interpreter."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if image.ndim == 2:
        image = image.reshape(1, image.shape[0], image.shape[1], 1)

    interpreter.set_tensor(input_details['index'], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details['index'])
    return output.squeeze()


def main():
    print("="*60)
    print("TFLite CPU vs Manual Gabor (NEW CORRECTED MODEL)")
    print("="*60)

    size = 64
    np.random.seed(42)
    image = (np.random.rand(size, size) * 255).astype(np.uint8)

    # Manual (ground truth)
    print("\n--- Manual Gabor ---")
    feat_manual = manual_gabor_conv_cpu(image)
    print(f"Output: shape={feat_manual.shape}, mean={feat_manual.mean():.1f}")

    # TFLite CPU (corrected model)
    print("\n--- TFLite CPU (CORRECTED with OHWI weights) ---")
    tflite_bytes, _ = build_optical_flow(size, size)
    feat_tflite = run_tflite_cpu(tflite_bytes, image)
    print(f"Output: shape={feat_tflite.shape}, mean={feat_tflite.mean():.1f}")

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    diff = feat_manual.astype(np.int16) - feat_tflite.astype(np.int16)
    mae = np.abs(diff).mean()
    corr = np.corrcoef(feat_manual.ravel(), feat_tflite.ravel())[0, 1]

    print(f"\nOverall:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Correlation: {corr:.6f}")

    # Per-channel
    print(f"\nPer-channel:")
    for c in range(8):
        manual_c = feat_manual[:, :, c]
        tflite_c = feat_tflite[:, :, c]
        mae_c = np.abs(manual_c.astype(np.int16) - tflite_c.astype(np.int16)).mean()
        corr_c = np.corrcoef(manual_c.ravel(), tflite_c.ravel())[0, 1]
        marker = "✅" if mae_c < 1.0 else "❌"
        print(f"  Channel {c}: MAE={mae_c:.1f}, corr={corr_c:.3f} {marker}")

    if corr > 0.99:
        print(f"\n✅ TFLite CPU WORKS CORRECTLY with corrected OHWI model!")
        print(f"   → The bug is in edgetpu_compiler or Edge TPU hardware, NOT our model!")
    else:
        print(f"\n❌ TFLite CPU still broken (corr={corr:.6f})")
        print(f"   → The model itself might still be wrong")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
