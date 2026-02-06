#!/usr/bin/env python3
"""Deep investigation of Edge TPU feature maps.

Key finding: Edge TPU features have near-zero correlation with CPU features,
AND vertical shifts fail while horizontal shifts work.

Tests:
1. Edge TPU feature reproducibility (same input → same output?)
2. Edge TPU feature spatial structure (constant along Y? constant along X?)
3. Edge TPU self-correlation (shifted inputs → shifted features?)
4. CPU without XNNPACK as ground truth
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.optical_flow_module import OpticalFlow


def main():
    h, w = 64, 64

    print("Loading Edge TPU model (standard mode)...")
    flow = OpticalFlow.from_template(64, pooled=False)
    flow.open()

    # ── Test 1: Reproducibility ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Edge TPU Feature Reproducibility")
    print("=" * 70)

    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    feat1 = flow._extract_features_uint8(img)
    feat2 = flow._extract_features_uint8(img)

    diff = feat1.astype(np.int16) - feat2.astype(np.int16)
    print(f"  Same input, two invocations:")
    print(f"    max_abs_diff = {np.max(np.abs(diff))}")
    print(f"    n_different = {np.sum(diff != 0)} / {diff.size}")
    if np.max(np.abs(diff)) == 0:
        print("    RESULT: Perfectly reproducible ✓")
    else:
        print("    RESULT: NOT reproducible ✗")

    # ── Test 2: Spatial Structure ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Edge TPU Feature Spatial Structure")
    print("=" * 70)

    feat = flow._extract_features_uint8(img)

    print("  Per-channel statistics:")
    for ch in range(8):
        ch_data = feat[:, :, ch].astype(float)
        # Check row-constancy (same value across columns)
        row_std = np.std(ch_data, axis=1).mean()  # avg std across columns per row
        # Check column-constancy (same value across rows)
        col_std = np.std(ch_data, axis=0).mean()  # avg std across rows per column
        print(f"    Ch{ch}: mean={ch_data.mean():.1f}, "
              f"row_variation={row_std:.1f}, col_variation={col_std:.1f}, "
              f"zero%={100 * np.mean(ch_data == 0):.1f}, "
              f"sat%={100 * np.mean(ch_data == 255):.1f}")

    print("\n  Vertical vs Horizontal autocorrelation of features:")
    for ch in range(8):
        ch_data = feat[:, :, ch].astype(float)
        # Horizontal autocorrelation (shift along X)
        h_corr_1 = np.corrcoef(ch_data[:, :-1].ravel(), ch_data[:, 1:].ravel())[0, 1]
        h_corr_4 = np.corrcoef(ch_data[:, :-4].ravel(), ch_data[:, 4:].ravel())[0, 1]
        # Vertical autocorrelation (shift along Y)
        v_corr_1 = np.corrcoef(ch_data[:-1, :].ravel(), ch_data[1:, :].ravel())[0, 1]
        v_corr_4 = np.corrcoef(ch_data[:-4, :].ravel(), ch_data[4:, :].ravel())[0, 1]
        print(f"    Ch{ch}: H_acorr(1)={h_corr_1:.3f} H_acorr(4)={h_corr_4:.3f} "
              f"| V_acorr(1)={v_corr_1:.3f} V_acorr(4)={v_corr_4:.3f}")

    # ── Test 3: Edge TPU Self-Correlation ─────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: Edge TPU Self-Correlation (shifted input → shifted features?)")
    print("=" * 70)

    feat_base = flow._extract_features_uint8(img)

    for shift_y, shift_x, desc in [(0, 4, "Right 4px"), (4, 0, "Down 4px"),
                                    (0, 8, "Right 8px"), (8, 0, "Down 8px")]:
        img_shifted = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
        feat_shifted = flow._extract_features_uint8(img_shifted)

        # Compare: if features correctly track shifts, then
        # feat_shifted should look like feat_base shifted by the same amount
        if shift_x > 0:
            # Compare overlapping region (columns shift_x: onwards)
            overlap_base = feat_base[:, shift_x:, :]
            overlap_shifted = feat_shifted[:, :-shift_x, :]
        elif shift_y > 0:
            overlap_base = feat_base[shift_y:, :, :]
            overlap_shifted = feat_shifted[:-shift_y, :, :]
        else:
            overlap_base = feat_base
            overlap_shifted = feat_shifted

        per_ch_corr = []
        for ch in range(8):
            base_ch = overlap_base[:, :, ch].astype(float).ravel()
            shift_ch = overlap_shifted[:, :, ch].astype(float).ravel()
            if base_ch.std() < 0.01 or shift_ch.std() < 0.01:
                per_ch_corr.append(float('nan'))
            else:
                per_ch_corr.append(np.corrcoef(base_ch, shift_ch)[0, 1])

        mean_corr = np.nanmean(per_ch_corr)
        corr_str = " ".join([f"Ch{i}={c:.3f}" for i, c in enumerate(per_ch_corr)])
        print(f"  {desc}: mean_corr={mean_corr:.3f}")
        print(f"    {corr_str}")

    # ── Test 4: CPU without XNNPACK as ground truth ───────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: CPU without XNNPACK vs Edge TPU")
    print("=" * 70)

    try:
        import tensorflow as tf
        from libredgetpu.tflite_builder import build_optical_flow

        tflite_bytes, metadata = build_optical_flow(h, w)

        # Try to disable XNNPACK
        try:
            interpreter = tf.lite.Interpreter(
                model_content=tflite_bytes,
                experimental_preserve_all_tensors=True,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
        except Exception:
            # Fallback if the experimental flag isn't available
            interpreter = tf.lite.Interpreter(
                model_content=tflite_bytes,
                experimental_preserve_all_tensors=True
            )
            print("  WARNING: Could not disable XNNPACK delegate")

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Run CPU
        input_data = img.reshape(1, h, w, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        cpu_feat = interpreter.get_tensor(output_details[0]['index']).squeeze()

        # Run Edge TPU
        tpu_feat = flow._extract_features_uint8(img)

        print(f"\n  Per-channel CPU vs TPU correlation:")
        for ch in range(8):
            cpu_ch = cpu_feat[:, :, ch].astype(float)
            tpu_ch = tpu_feat[:, :, ch].astype(float)
            if cpu_ch.std() < 0.01 or tpu_ch.std() < 0.01:
                corr = float('nan')
            else:
                corr = np.corrcoef(cpu_ch.ravel(), tpu_ch.ravel())[0, 1]
            diff = cpu_ch - tpu_ch
            print(f"    Ch{ch}: corr={corr:.4f} "
                  f"cpu_mean={cpu_ch.mean():.1f} tpu_mean={tpu_ch.mean():.1f} "
                  f"diff_abs_mean={np.abs(diff).mean():.1f}")

        # Test full pipeline with CPU (no XNNPACK) features
        print(f"\n  Full pipeline with CPU (no XNNPACK) features:")
        from experiments.diagnose_optical_flow import (
            global_correlation, soft_argmax, compute_overlap_counts
        )

        def cpu_pipeline(img1, img2):
            """Run OpticalFlow pipeline using CPU features."""
            inp1 = img1.reshape(1, h, w, 1)
            interpreter.set_tensor(input_details[0]['index'], inp1)
            interpreter.invoke()
            f1 = interpreter.get_tensor(output_details[0]['index']).squeeze()

            inp2 = img2.reshape(1, h, w, 1)
            interpreter.set_tensor(input_details[0]['index'], inp2)
            interpreter.invoke()
            f2 = interpreter.get_tensor(output_details[0]['index']).squeeze()

            out_zp = output_details[0]['quantization_parameters']['zero_points'][0]
            f1_int = f1.astype(np.int16) - np.int16(out_zp)
            f2_int = f2.astype(np.int16) - np.int16(out_zp)

            # Pool
            pool = 4
            f1_p = f1_int.reshape(h // pool, pool, w // pool, pool, 8).sum(axis=(1, 3), dtype=np.int32).astype(np.float32)
            f2_p = f2_int.reshape(h // pool, pool, w // pool, pool, 8).sum(axis=(1, 3), dtype=np.int32).astype(np.float32)

            sr = 4
            overlap = compute_overlap_counts(h // pool, w // pool, 8, sr)
            corr = global_correlation(f1_p, f2_p, sr).astype(np.float64)
            corr /= overlap
            return soft_argmax(corr.astype(np.float32), sr)

        for shift_y, shift_x, exp_vy, exp_vx, desc in [
            (0, 8, 0.0, 2.0, "Right 8px"),
            (8, 0, 2.0, 0.0, "Down 8px"),
        ]:
            img_shifted = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
            vx_cpu, vy_cpu = cpu_pipeline(img, img_shifted)
            vx_tpu, vy_tpu = flow.compute(img, img_shifted)
            print(f"    {desc}: expected=({exp_vx:.1f}, {exp_vy:.1f}), "
                  f"CPU=({vx_cpu:.2f}, {vy_cpu:.2f}), "
                  f"TPU=({vx_tpu:.2f}, {vy_tpu:.2f})")

    except ImportError:
        print("  SKIP: TensorFlow not available")

    # ── Test 5: Print actual feature map values for visual inspection ─────
    print("\n" + "=" * 70)
    print("TEST 5: Feature Map Values (top-left 4x4 block of each channel)")
    print("=" * 70)

    feat = flow._extract_features_uint8(img)
    for ch in range(8):
        print(f"\n  Ch{ch}:")
        block = feat[:4, :4, ch]
        for row in range(4):
            vals = " ".join(f"{v:3d}" for v in block[row])
            print(f"    [{vals}]")

    flow.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
