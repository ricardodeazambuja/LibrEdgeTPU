"""Test optical flow against manual CPU Gabor convolution (ground truth).

This bypasses TFLite entirely and computes Gabor features using numpy/scipy
as the absolute ground truth.
"""

import numpy as np
import sys
import os
import scipy.ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.tflite_builder import _generate_gabor_kernels, build_optical_flow
from libredgetpu import OpticalFlow


def manual_gabor_conv_cpu(image, ksize=7, orientations=4, sigmas=(1.5, 3.0)):
    """Manually compute Gabor convolution on CPU using scipy (ground truth).

    Returns:
        Features [H, W, n_filters] as uint8 (quantized, matching TFLite output)
    """
    # Generate Gabor kernels
    kernels = _generate_gabor_kernels(ksize, orientations, sigmas)  # [ksize, ksize, 1, n_filters]
    n_filters = kernels.shape[3]
    h, w = image.shape

    # Simulate TFLite quantization
    input_int8 = image.astype(np.int16) - 128  # uint8 [0,255] -> int8 [-128,127]

    # Quantize Gabor kernels with PER-CHANNEL quantization (matching TFLite model)
    per_ch_scales = []
    kernels_int8 = np.zeros_like(kernels, dtype=np.int8)
    for i in range(n_filters):
        kernel_ch = kernels[:, :, :, i]  # [ksize, ksize, 1]
        ch_abs_max = float(np.max(np.abs(kernel_ch)))
        ch_scale = max(ch_abs_max, 1e-6) / 127.0
        per_ch_scales.append(ch_scale)
        kernels_int8[:, :, :, i] = np.clip(
            np.round(kernel_ch / ch_scale), -127, 127
        ).astype(np.int8)

    # Average scale for output quantization (matches TFLite)
    avg_weight_scale = np.mean(per_ch_scales)
    conv_output_max = 1.0 * avg_weight_scale * ksize * ksize * 127
    conv_output_scale = conv_output_max / 127.0

    # Convolve each channel
    features = np.zeros((h, w, n_filters), dtype=np.float32)
    for i in range(n_filters):
        kernel = kernels_int8[:, :, 0, i].astype(np.float32)  # [ksize, ksize]
        # Conv with int16 accumulation
        conv_int = scipy.ndimage.convolve(
            input_int8.astype(np.float32), kernel, mode='constant', cval=0.0
        )
        # Dequantize with THIS channel's scale
        conv_float = conv_int * per_ch_scales[i]
        # ReLU
        conv_float = np.maximum(conv_float, 0.0)
        features[:, :, i] = conv_float

    # Quantize to uint8 output
    features_uint8 = np.clip(
        np.round(features / conv_output_scale), 0, 255
    ).astype(np.uint8)

    return features_uint8


def main():
    print("="*60)
    print("Manual Gabor Convolution vs Edge TPU")
    print("="*60)

    size = 64

    # Create test image
    np.random.seed(42)
    image = (np.random.rand(size, size) * 255).astype(np.uint8)
    print(f"\nTest image: shape={image.shape}, dtype={image.dtype}")

    # Ground truth: Manual CPU Gabor
    print("\n--- Manual CPU Gabor (Ground Truth) ---")
    feat_manual = manual_gabor_conv_cpu(image)
    print(f"Output: shape={feat_manual.shape}, range=[{feat_manual.min()}, {feat_manual.max()}], mean={feat_manual.mean():.1f}")

    # Edge TPU
    print("\n--- Edge TPU ---")
    try:
        flow = OpticalFlow.from_template(size, pooled=False)
        flow.open()
        feat_tpu = flow._extract_features_uint8(image)
        flow.close()
        print(f"Output: shape={feat_tpu.shape}, range=[{feat_tpu.min()}, {feat_tpu.max()}], mean={feat_tpu.mean():.1f}")
    except Exception as e:
        print(f"❌ Edge TPU failed: {e}")
        return

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    if feat_manual.shape != feat_tpu.shape:
        print(f"❌ Shape mismatch: manual={feat_manual.shape}, TPU={feat_tpu.shape}")
        return

    # Overall
    diff = feat_manual.astype(np.int16) - feat_tpu.astype(np.int16)
    mae = np.abs(diff).mean()
    max_diff = np.abs(diff).max()
    frac_equal = np.mean(diff == 0)
    corr = np.corrcoef(feat_manual.ravel().astype(np.float32),
                       feat_tpu.ravel().astype(np.float32))[0, 1]

    print(f"\nOverall:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Max diff: {max_diff}")
    print(f"  Fraction equal: {frac_equal:.4f}")
    print(f"  Correlation: {corr:.6f}")

    # Per-channel
    print(f"\nPer-channel:")
    print(f"{'Chan':<6} {'Manual mean':<15} {'TPU mean':<15} {'MAE':<10} {'Corr':<10}")
    print("-" * 60)
    for c in range(8):
        manual_c = feat_manual[:, :, c]
        tpu_c = feat_tpu[:, :, c]
        manual_mean = manual_c.mean()
        tpu_mean = tpu_c.mean()
        mae_c = np.abs(manual_c.astype(np.int16) - tpu_c.astype(np.int16)).mean()
        corr_c = np.corrcoef(manual_c.ravel().astype(np.float32),
                             tpu_c.ravel().astype(np.float32))[0, 1]
        marker = "✅" if mae_c < 1.0 else ("⚠️" if mae_c < 10.0 else "❌")
        print(f"{c:<6} {manual_mean:<15.1f} {tpu_mean:<15.1f} {mae_c:<10.1f} {corr_c:<10.3f} {marker}")

    if corr > 0.99:
        print(f"\n✅ Edge TPU matches manual CPU Gabor (correlation > 0.99)")
    elif corr > 0.95:
        print(f"\n⚠️  Edge TPU mostly matches (correlation > 0.95)")
    else:
        print(f"\n❌ Edge TPU does NOT match manual CPU Gabor (correlation = {corr:.6f})")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
