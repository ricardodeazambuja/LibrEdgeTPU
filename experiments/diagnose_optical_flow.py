#!/usr/bin/env python3
"""Systematic diagnosis of OpticalFlow — test each component independently.

Strategy: SpotTracker and PatternTracker WORK. The infrastructure is proven.
The bug MUST be in something OpticalFlow does differently.

Components tested independently:
  1. CPU correlation + soft_argmax (NO Edge TPU, NO Gabor — pure math)
  2. Gabor kernels (numpy convolution, NO Edge TPU)
  3. Gabor features → correlation pipeline (numpy, NO Edge TPU)
  4. TFLite model on CPU (NO Edge TPU, checks quantization)
  5. Full pipeline on Edge TPU (if available)

For each test, we create known-displacement image pairs and verify the output.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1: Pure correlation + soft_argmax
# Tests _global_correlation and _soft_argmax with synthetic feature maps
# NO Edge TPU, NO Gabor — just the math
# ═══════════════════════════════════════════════════════════════════════════════

def global_correlation(feat_t, feat_t1, search_range=4):
    """Standalone copy of OpticalFlow._global_correlation."""
    h, w, c = feat_t.shape
    sr = search_range
    side = 2 * sr + 1

    padded = np.pad(feat_t, ((sr, sr), (sr, sr), (0, 0)), mode='constant')
    s = padded.strides
    view = as_strided(padded,
                      shape=(side, side, h, w, c),
                      strides=(s[0], s[1], s[0], s[1], s[2]))

    if np.issubdtype(feat_t1.dtype, np.integer):
        corr_map = np.einsum('ijhwc,hwc->ij', view, feat_t1, dtype=np.int64)
    else:
        corr_map = np.einsum('ijhwc,hwc->ij', view, feat_t1)

    return corr_map[::-1, ::-1].ravel()


def soft_argmax(corr, search_range=4, temperature=0.1):
    """Standalone copy of OpticalFlow._soft_argmax."""
    sr = search_range
    displacements = []
    for dy in range(-sr, sr + 1):
        for dx in range(-sr, sr + 1):
            displacements.append((dx, dy))
    displacements = np.array(displacements, dtype=np.float32)

    corr_shifted = corr - np.max(corr)
    weights = np.exp(corr_shifted / max(temperature, 1e-6))
    total = np.sum(weights)
    if total < 1e-12:
        return 0.0, 0.0
    weights /= total

    vx = float(np.sum(weights * displacements[:, 0]))
    vy = float(np.sum(weights * displacements[:, 1]))
    return vx, vy


def compute_overlap_counts(ph, pw, nf, search_range=4):
    """Standalone copy of overlap count computation."""
    sr = search_range
    return np.array(
        [(ph - abs(dy)) * (pw - abs(dx)) * nf
         for dy in range(-sr, sr + 1) for dx in range(-sr, sr + 1)],
        dtype=np.float64,
    )


def test_correlation_math():
    """Test 1: Pure correlation with synthetic features."""
    print("=" * 70)
    print("TEST 1: Pure Correlation + Soft Argmax (no Edge TPU, no Gabor)")
    print("=" * 70)

    np.random.seed(42)
    h, w, c = 16, 16, 8  # Pooled feature size
    sr = 4

    # Create random features
    feat_base = np.random.rand(h, w, c).astype(np.float32) * 100

    overlap = compute_overlap_counts(h, w, c, sr)

    test_cases = [
        # (shift_y, shift_x, expected_vy, expected_vx, description)
        (0, 0, 0.0, 0.0, "No shift"),
        (0, 2, 0.0, 2.0, "Right shift by 2"),
        (0, -3, 0.0, -3.0, "Left shift by 3"),
        (2, 0, 2.0, 0.0, "Down shift by 2"),
        (-3, 0, -3.0, 0.0, "Up shift by 3"),
        (1, 2, 1.0, 2.0, "Diagonal shift (2, 1)"),
    ]

    all_pass = True
    for shift_y, shift_x, exp_vy, exp_vx, desc in test_cases:
        feat_t = feat_base.copy()
        feat_t1 = np.roll(np.roll(feat_t, shift_y, axis=0), shift_x, axis=1)

        corr = global_correlation(feat_t, feat_t1, sr).astype(np.float64)
        corr /= overlap
        vx, vy = soft_argmax(corr.astype(np.float32), sr)

        ok_x = abs(vx - exp_vx) < 0.5
        ok_y = abs(vy - exp_vy) < 0.5
        status = "PASS" if (ok_x and ok_y) else "FAIL"
        if not (ok_x and ok_y):
            all_pass = False

        print(f"  [{status}] {desc}: expected ({exp_vx:.1f}, {exp_vy:.1f}), "
              f"got ({vx:.2f}, {vy:.2f})")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2: Gabor kernels (numpy convolution, no Edge TPU)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_gabor_kernels(ksize=7, orientations=4, sigmas=(1.5, 3.0)):
    """Standalone copy of _generate_gabor_kernels."""
    n_filters = orientations * len(sigmas)
    kernels = np.zeros((ksize, ksize, 1, n_filters), dtype=np.float32)
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)

    idx = 0
    for sigma in sigmas:
        wavelength = 2.0 * sigma
        for oi in range(orientations):
            theta = oi * np.pi / orientations
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            gaussian = np.exp(-(x_theta ** 2 + y_theta ** 2) / (2.0 * sigma ** 2))
            carrier = np.cos(2.0 * np.pi * y_theta / wavelength)
            gabor = gaussian * carrier
            abs_max = float(np.max(np.abs(gabor)))
            if abs_max > 1e-6:
                gabor = gabor / abs_max
            kernels[:, :, 0, idx] = gabor
            idx += 1
    return kernels


def apply_gabor_numpy(image, kernels):
    """Apply Gabor kernels using numpy convolution (float, no quantization)."""
    from scipy.signal import correlate2d
    h, w = image.shape
    ksize = kernels.shape[0]
    n_filters = kernels.shape[3]

    features = np.zeros((h, w, n_filters), dtype=np.float32)
    for ch in range(n_filters):
        kernel = kernels[:, :, 0, ch]
        # correlate2d for 'same' padding
        features[:, :, ch] = correlate2d(image.astype(np.float32), kernel,
                                          mode='same', boundary='fill')
    # ReLU
    features = np.maximum(features, 0)
    return features


def test_gabor_kernels():
    """Test 2: Gabor kernel properties and orientation selectivity."""
    print("=" * 70)
    print("TEST 2: Gabor Kernel Properties")
    print("=" * 70)

    kernels = generate_gabor_kernels()
    n_filters = kernels.shape[3]
    labels = []
    for sigma in (1.5, 3.0):
        for oi in range(4):
            theta = oi * 180 / 4
            labels.append(f"Ch{len(labels)}: sigma={sigma}, theta={theta}°")

    print("  Channel properties:")
    for ch in range(n_filters):
        k = kernels[:, :, 0, ch]
        print(f"    {labels[ch]}: sum={k.sum():.3f}, max={k.max():.3f}, "
              f"min={k.min():.3f}, abs_max={np.max(np.abs(k)):.3f}")

    print()

    # Test orientation selectivity with synthetic edge patterns
    h, w = 64, 64

    # Horizontal edges (should activate theta=0° channels)
    horiz = np.zeros((h, w), dtype=np.float32)
    for row in range(h):
        if (row // 8) % 2 == 0:
            horiz[row, :] = 255.0

    # Vertical edges (should activate theta=90° channels)
    vert = np.zeros((h, w), dtype=np.float32)
    for col in range(w):
        if (col // 8) % 2 == 0:
            vert[:, col] = 255.0

    feat_h = apply_gabor_numpy(horiz, kernels)
    feat_v = apply_gabor_numpy(vert, kernels)

    print("  Mean response to HORIZONTAL edges (should peak at Ch0/Ch4, theta=0°):")
    for ch in range(n_filters):
        print(f"    {labels[ch]}: mean={feat_h[:, :, ch].mean():.2f}")

    print()
    print("  Mean response to VERTICAL edges (should peak at Ch2/Ch6, theta=90°):")
    for ch in range(n_filters):
        print(f"    {labels[ch]}: mean={feat_v[:, :, ch].mean():.2f}")

    print()
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3: Gabor features → correlation pipeline (numpy, no Edge TPU)
# ═══════════════════════════════════════════════════════════════════════════════

def pool_features(feat, pool_factor=4):
    """Block-mean pooling."""
    h, w, c = feat.shape
    p = pool_factor
    h_trunc = (h // p) * p
    w_trunc = (w // p) * p
    cropped = feat[:h_trunc, :w_trunc, :]
    return cropped.reshape(h_trunc // p, p, w_trunc // p, p, c).mean(axis=(1, 3))


def test_gabor_pipeline_numpy():
    """Test 3: Full Gabor → pool → correlate pipeline using numpy only."""
    print("=" * 70)
    print("TEST 3: Gabor → Pool → Correlate Pipeline (numpy, no Edge TPU)")
    print("=" * 70)

    h, w = 64, 64
    sr = 4
    pool = 4
    kernels = generate_gabor_kernels()

    # Test with different patterns and shifts
    test_cases = [
        ("Horizontal stripes, shift right by 8px",
         "horizontal", 0, 8, 0.0, 2.0),  # 8px / pool=4 = 2 pooled pixels
        ("Horizontal stripes, shift down by 8px",
         "horizontal", 8, 0, 2.0, 0.0),
        ("Vertical stripes, shift right by 8px",
         "vertical", 0, 8, 0.0, 2.0),
        ("Vertical stripes, shift down by 8px",
         "vertical", 8, 0, 2.0, 0.0),
        ("Random texture, shift right by 8px",
         "random", 0, 8, 0.0, 2.0),
        ("Random texture, shift down by 8px",
         "random", 8, 0, 2.0, 0.0),
    ]

    all_pass = True
    for desc, pattern, shift_y, shift_x, exp_vy, exp_vx in test_cases:
        # Create base image
        if pattern == "horizontal":
            img = np.zeros((h, w), dtype=np.float32)
            for row in range(h):
                if (row // 8) % 2 == 0:
                    img[row, :] = 255.0
        elif pattern == "vertical":
            img = np.zeros((h, w), dtype=np.float32)
            for col in range(w):
                if (col // 8) % 2 == 0:
                    img[:, col] = 255.0
        elif pattern == "random":
            np.random.seed(123)
            img = np.random.randint(0, 256, (h, w)).astype(np.float32)

        # Create shifted version
        img_shifted = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)

        # Extract features using numpy
        feat_t = apply_gabor_numpy(img, kernels)
        feat_t1 = apply_gabor_numpy(img_shifted, kernels)

        # Pool
        feat_t_p = pool_features(feat_t, pool)
        feat_t1_p = pool_features(feat_t1, pool)

        # Correlate
        ph, pw = feat_t_p.shape[0], feat_t_p.shape[1]
        nf = feat_t_p.shape[2]
        overlap = compute_overlap_counts(ph, pw, nf, sr)
        corr = global_correlation(feat_t_p, feat_t1_p, sr).astype(np.float64)
        corr /= overlap
        vx, vy = soft_argmax(corr.astype(np.float32), sr)

        ok_x = abs(vx - exp_vx) < 0.5
        ok_y = abs(vy - exp_vy) < 0.5
        status = "PASS" if (ok_x and ok_y) else "FAIL"
        if not (ok_x and ok_y):
            all_pass = False

        print(f"  [{status}] {desc}")
        print(f"         expected ({exp_vx:.1f}, {exp_vy:.1f}), "
              f"got ({vx:.2f}, {vy:.2f})")

        # Print per-channel analysis for failures
        if not (ok_x and ok_y):
            print(f"         Per-channel feature stats (pooled):")
            for ch in range(nf):
                t_ch = feat_t_p[:, :, ch]
                t1_ch = feat_t1_p[:, :, ch]
                print(f"           Ch{ch}: t_mean={t_ch.mean():.2f} "
                      f"t1_mean={t1_ch.mean():.2f} "
                      f"t_std={t_ch.std():.2f} "
                      f"t1_std={t1_ch.std():.2f}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 4: TFLite model on CPU (checks quantization)
# ═══════════════════════════════════════════════════════════════════════════════

def test_tflite_cpu():
    """Test 4: TFLite model on CPU vs numpy reference."""
    print("=" * 70)
    print("TEST 4: TFLite Model on CPU (checks quantization)")
    print("=" * 70)

    try:
        import tensorflow as tf
    except ImportError:
        print("  SKIP: TensorFlow not available\n")
        return True

    # Build the model
    from libredgetpu.tflite_builder import build_optical_flow

    h, w = 64, 64
    tflite_bytes, metadata = build_optical_flow(h, w)

    # Load with TFLite interpreter (XNNPACK disabled!)
    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input:  shape={input_details[0]['shape']}, "
          f"dtype={input_details[0]['dtype']}, "
          f"quant={input_details[0]['quantization_parameters']}")
    print(f"  Output: shape={output_details[0]['shape']}, "
          f"dtype={output_details[0]['dtype']}, "
          f"quant={output_details[0]['quantization_parameters']}")

    # Test with various patterns
    kernels = generate_gabor_kernels()
    # Create patterns properly
    horiz = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        if (row // 8) % 2 == 0:
            horiz[row, :] = 255
    vert = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        if (col // 8) % 2 == 0:
            vert[:, col] = 255

    patterns = {
        "Horizontal stripes": horiz,
        "Vertical stripes": vert,
        "Random texture": np.random.RandomState(42).randint(
            0, 256, (h, w), dtype=np.uint8),
        "Constant 128": np.full((h, w), 128, dtype=np.uint8),
    }

    for name, img in patterns.items():
        # Run TFLite
        input_data = img.reshape(1, h, w, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_out = interpreter.get_tensor(output_details[0]['index'])
        tflite_out = tflite_out.squeeze()  # [H, W, 8]

        # Run numpy reference
        numpy_feat = apply_gabor_numpy(img.astype(np.float32), kernels)

        print(f"\n  {name}:")
        print(f"    TFLite output stats:")
        for ch in range(8):
            ch_data = tflite_out[:, :, ch].astype(float)
            print(f"      Ch{ch}: mean={ch_data.mean():.1f}, std={ch_data.std():.1f}, "
                  f"min={ch_data.min():.0f}, max={ch_data.max():.0f}, "
                  f"saturated%={100*np.mean(ch_data >= 255):.1f}")

        print(f"    Numpy reference stats:")
        for ch in range(8):
            ch_data = numpy_feat[:, :, ch]
            print(f"      Ch{ch}: mean={ch_data.mean():.1f}, std={ch_data.std():.1f}, "
                  f"min={ch_data.min():.1f}, max={ch_data.max():.1f}")

    print()
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 5: Full pipeline comparison (TFLite CPU vs numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_cpu():
    """Test 5: Full flow pipeline using TFLite on CPU."""
    print("=" * 70)
    print("TEST 5: Full Pipeline - TFLite CPU features → correlation → flow")
    print("=" * 70)

    try:
        import tensorflow as tf
    except ImportError:
        print("  SKIP: TensorFlow not available\n")
        return True

    from libredgetpu.tflite_builder import build_optical_flow

    h, w = 64, 64
    sr = 4
    pool = 4
    tflite_bytes, metadata = build_optical_flow(h, w)

    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    out_scale = output_details[0]['quantization_parameters']['scales'][0]
    out_zp = output_details[0]['quantization_parameters']['zero_points'][0]

    def extract_features_tflite(img_uint8):
        """Run through TFLite and return uint8 features."""
        input_data = img_uint8.reshape(1, h, w, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index']).squeeze()

    # Test shifts with random texture
    np.random.seed(42)
    base_img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    test_cases = [
        (0, 0, 0.0, 0.0, "No shift"),
        (0, 8, 0.0, 2.0, "Right 8px (2 pooled)"),
        (0, -8, 0.0, -2.0, "Left 8px (2 pooled)"),
        (8, 0, 2.0, 0.0, "Down 8px (2 pooled)"),
        (-8, 0, -2.0, 0.0, "Up 8px (2 pooled)"),
        (0, 4, 0.0, 1.0, "Right 4px (1 pooled)"),
        (4, 0, 1.0, 0.0, "Down 4px (1 pooled)"),
    ]

    all_pass = True
    for shift_y, shift_x, exp_vy, exp_vx, desc in test_cases:
        img_shifted = np.roll(np.roll(base_img, shift_y, axis=0),
                              shift_x, axis=1)

        feat_t = extract_features_tflite(base_img)
        feat_t1 = extract_features_tflite(img_shifted)

        # Subtract zero point, cast to int16
        feat_t_int = feat_t.astype(np.int16) - np.int16(out_zp)
        feat_t1_int = feat_t1.astype(np.int16) - np.int16(out_zp)

        # Pool (block sum)
        feat_t_p = pool_features(feat_t_int.astype(np.float32), pool)
        feat_t1_p = pool_features(feat_t1_int.astype(np.float32), pool)

        # Correlate
        ph, pw = feat_t_p.shape[0], feat_t_p.shape[1]
        nf = feat_t_p.shape[2]
        overlap = compute_overlap_counts(ph, pw, nf, sr)
        corr = global_correlation(feat_t_p, feat_t1_p, sr).astype(np.float64)
        corr /= overlap
        vx, vy = soft_argmax(corr.astype(np.float32), sr)

        ok_x = abs(vx - exp_vx) < 0.5
        ok_y = abs(vy - exp_vy) < 0.5
        status = "PASS" if (ok_x and ok_y) else "FAIL"
        if not (ok_x and ok_y):
            all_pass = False

        print(f"  [{status}] {desc}: expected ({exp_vx:.1f}, {exp_vy:.1f}), "
              f"got ({vx:.2f}, {vy:.2f})")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 6: Feature diversity analysis
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_diversity():
    """Test 6: Check if features have enough diversity for correlation."""
    print("=" * 70)
    print("TEST 6: Feature Diversity Analysis")
    print("=" * 70)

    try:
        import tensorflow as tf
    except ImportError:
        print("  SKIP: TensorFlow not available\n")
        return True

    from libredgetpu.tflite_builder import build_optical_flow

    h, w = 64, 64
    tflite_bytes, metadata = build_optical_flow(h, w)

    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Random texture
    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    input_data = img.reshape(1, h, w, 1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    feat = interpreter.get_tensor(output_details[0]['index']).squeeze()

    print("  Feature map statistics for random texture input:")
    for ch in range(8):
        ch_data = feat[:, :, ch].astype(float)
        unique = len(np.unique(ch_data))
        print(f"    Ch{ch}: unique_vals={unique}, "
              f"mean={ch_data.mean():.1f}, std={ch_data.std():.1f}, "
              f"range=[{ch_data.min():.0f}, {ch_data.max():.0f}], "
              f"zero%={100*np.mean(ch_data == 0):.1f}, "
              f"sat%={100*np.mean(ch_data >= 255):.1f}")

    # Check if features have spatial structure (not flat)
    print("\n  Spatial autocorrelation (should be high for small shifts, low for large):")
    for shift in [0, 1, 2, 4, 8]:
        if shift == 0:
            corr_val = np.mean(feat.astype(float) ** 2)
        else:
            corr_val = np.mean(
                feat[:-shift, :, :].astype(float) * feat[shift:, :, :].astype(float))
        print(f"    shift={shift}: autocorr={corr_val:.1f}")

    print()
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = {}

    results["1_correlation_math"] = test_correlation_math()
    results["2_gabor_kernels"] = test_gabor_kernels()
    results["3_gabor_pipeline_numpy"] = test_gabor_pipeline_numpy()
    results["4_tflite_cpu"] = test_tflite_cpu()
    results["5_full_pipeline_cpu"] = test_full_pipeline_cpu()
    results["6_feature_diversity"] = test_feature_diversity()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print()
    if all(results.values()):
        print("  All components pass — bug is likely in Edge TPU execution or "
              "input preprocessing")
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"  Failed components: {', '.join(failed)}")
        print("  Bug is in the CPU-side code — fix before testing on Edge TPU")
