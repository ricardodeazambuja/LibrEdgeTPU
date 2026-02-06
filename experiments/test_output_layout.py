#!/usr/bin/env python3
"""Test hypothesis: Edge TPU outputs in NCHW format, but we interpret as NHWC.

If true, reinterpreting the raw bytes as (C, H, W) and transposing to (H, W, C)
should produce features that match the CPU reference.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.optical_flow_module import OpticalFlow


def main():
    h, w, c = 64, 64, 8

    print("Loading Edge TPU model (standard mode)...")
    flow = OpticalFlow.from_template(64, pooled=False)
    flow.open()

    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    # Get raw Edge TPU output
    image_normalized = img.astype(np.float32) / 255.0
    from libredgetpu._quantize import quantize_uint8
    quantized = quantize_uint8(image_normalized,
                                flow._input_info.scale,
                                flow._input_info.zero_point)
    raw_output = flow._execute_raw(quantized.tobytes())
    n = h * w * c

    raw_bytes = np.frombuffer(raw_output, dtype=np.uint8)[:n]

    # Interpretation A: NHWC (current)
    feat_nhwc = raw_bytes.reshape(h, w, c)

    # Interpretation B: NCHW → transpose to HWC
    feat_nchw = raw_bytes.reshape(c, h, w)
    feat_nchw_to_hwc = np.transpose(feat_nchw, (1, 2, 0))

    # Interpretation C: Maybe (W, H, C)?
    feat_whc = raw_bytes.reshape(w, h, c)
    feat_whc_to_hwc = np.transpose(feat_whc, (1, 0, 2))

    # Get CPU reference
    try:
        import tensorflow as tf
        from libredgetpu.tflite_builder import build_optical_flow

        tflite_bytes, metadata = build_optical_flow(h, w)
        interpreter = tf.lite.Interpreter(
            model_content=tflite_bytes,
            experimental_preserve_all_tensors=True,
        )
        interpreter.allocate_tensors()
        inp_det = interpreter.get_input_details()
        out_det = interpreter.get_output_details()

        interpreter.set_tensor(inp_det[0]['index'], img.reshape(1, h, w, 1))
        interpreter.invoke()
        cpu_feat = interpreter.get_tensor(out_det[0]['index']).squeeze()

        print("\n" + "=" * 70)
        print("Layout Hypothesis Testing")
        print("=" * 70)

        for name, feat in [
            ("A: NHWC (current)", feat_nhwc),
            ("B: NCHW → HWC", feat_nchw_to_hwc),
            ("C: WHC → HWC", feat_whc_to_hwc),
        ]:
            per_ch_corr = []
            for ch in range(c):
                cpu_ch = cpu_feat[:, :, ch].astype(float).ravel()
                tpu_ch = feat[:, :, ch].astype(float).ravel()
                if cpu_ch.std() < 0.01 or tpu_ch.std() < 0.01:
                    per_ch_corr.append(float('nan'))
                else:
                    per_ch_corr.append(np.corrcoef(cpu_ch, tpu_ch)[0, 1])

            mean_corr = np.nanmean(per_ch_corr)
            corr_str = " ".join(f"{c:.3f}" for c in per_ch_corr)
            print(f"\n  {name}: mean_corr={mean_corr:.4f}")
            print(f"    Per-channel: {corr_str}")

    except ImportError:
        print("  SKIP: TensorFlow not available")

    # ── Test each layout with flow pipeline ───────────────────────────────
    print("\n" + "=" * 70)
    print("Flow Pipeline with Different Layouts")
    print("=" * 70)

    from experiments.diagnose_optical_flow import (
        global_correlation, soft_argmax, compute_overlap_counts
    )

    img_shifted_right = np.roll(img, 8, axis=1)
    img_shifted_down = np.roll(img, 8, axis=0)

    for layout_name, layout_fn in [
        ("NHWC (current)", lambda raw: raw.reshape(h, w, c)),
        ("NCHW → HWC", lambda raw: np.transpose(raw.reshape(c, h, w), (1, 2, 0))),
        ("WHC → HWC", lambda raw: np.transpose(raw.reshape(w, h, c), (1, 0, 2))),
    ]:
        def extract_with_layout(image):
            image_norm = image.astype(np.float32) / 255.0
            q = quantize_uint8(image_norm, flow._input_info.scale,
                               flow._input_info.zero_point)
            raw = flow._execute_raw(q.tobytes())
            return layout_fn(np.frombuffer(raw, dtype=np.uint8)[:n])

        # Get features
        f_base = extract_with_layout(img)
        f_right = extract_with_layout(img_shifted_right)
        f_down = extract_with_layout(img_shifted_down)

        # Run correlation
        zp = np.int16(flow._output_info.zero_point)
        pool = 4
        sr = 4

        def compute_flow(feat_t, feat_t1):
            ft = (feat_t.astype(np.int16) - zp).reshape(
                h // pool, pool, w // pool, pool, c).sum(axis=(1, 3), dtype=np.int32).astype(np.float32)
            ft1 = (feat_t1.astype(np.int16) - zp).reshape(
                h // pool, pool, w // pool, pool, c).sum(axis=(1, 3), dtype=np.int32).astype(np.float32)
            overlap = compute_overlap_counts(h // pool, w // pool, c, sr)
            corr = global_correlation(ft, ft1, sr).astype(np.float64)
            corr /= overlap
            return soft_argmax(corr.astype(np.float32), sr)

        vx_r, vy_r = compute_flow(f_base, f_right)
        vx_d, vy_d = compute_flow(f_base, f_down)

        print(f"\n  {layout_name}:")
        ok_r = abs(vx_r - 2.0) < 0.5 and abs(vy_r) < 0.5
        ok_d = abs(vy_d - 2.0) < 0.5 and abs(vx_d) < 0.5
        print(f"    Right 8px: expected (2.0, 0.0), got ({vx_r:.2f}, {vy_r:.2f}) "
              f"{'✓' if ok_r else '✗'}")
        print(f"    Down 8px:  expected (0.0, 2.0), got ({vx_d:.2f}, {vy_d:.2f}) "
              f"{'✓' if ok_d else '✗'}")

        # Self-correlation check: do shifted features track the input shift?
        for shift_name, f_shifted, shift_y, shift_x in [
            ("Right 4px", extract_with_layout(np.roll(img, 4, axis=1)), 0, 4),
            ("Down 4px", extract_with_layout(np.roll(img, 4, axis=0)), 4, 0),
        ]:
            if shift_x > 0:
                overlap_base = f_base[:, shift_x:, :]
                overlap_shifted = f_shifted[:, :-shift_x, :]
            else:
                overlap_base = f_base[shift_y:, :, :]
                overlap_shifted = f_shifted[:-shift_y, :, :]

            corrs = []
            for ch in range(c):
                b = overlap_base[:, :, ch].astype(float).ravel()
                s = overlap_shifted[:, :, ch].astype(float).ravel()
                if b.std() < 0.01 or s.std() < 0.01:
                    corrs.append(float('nan'))
                else:
                    corrs.append(np.corrcoef(b, s)[0, 1])
            mean_c = np.nanmean(corrs)
            print(f"    Self-corr {shift_name}: mean={mean_c:.3f} "
                  f"({' '.join(f'{c:.2f}' for c in corrs)})")

    flow.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
