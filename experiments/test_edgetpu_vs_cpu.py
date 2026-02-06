#!/usr/bin/env python3
"""Phase 2b: Compare Edge TPU features vs TFLite CPU features.

This is the ONLY untested component. If features match, the pipeline is correct
and the issue is in the GUI integration. If they differ, the bug is in
the Edge TPU execution path (quantization, model, firmware, etc).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_edgetpu_vs_cpu():
    """Compare Edge TPU features vs TFLite CPU features."""
    import tensorflow as tf
    from libredgetpu.tflite_builder import build_optical_flow
    from libredgetpu.optical_flow_module import OpticalFlow

    h, w = 64, 64

    # ── Build TFLite model and get CPU reference ──────────────────────────
    tflite_bytes, metadata = build_optical_flow(h, w)

    interpreter = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def extract_cpu(img_uint8):
        input_data = img_uint8.reshape(1, h, w, 1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index']).squeeze()

    # ── Load Edge TPU model ───────────────────────────────────────────────
    print("Loading Edge TPU model...")
    flow = OpticalFlow.from_template(64, pooled=False)
    flow.open()

    # ── Test images ───────────────────────────────────────────────────────
    np.random.seed(42)
    images = {
        "Random texture": np.random.randint(0, 256, (h, w), dtype=np.uint8),
        "Constant 128": np.full((h, w), 128, dtype=np.uint8),
        "Gradient H": np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1)),
        "Gradient V": np.tile(np.linspace(0, 255, h, dtype=np.uint8), (w, 1)).T.astype(np.uint8),
    }

    print("\n" + "=" * 70)
    print("Edge TPU vs TFLite CPU Feature Comparison")
    print("=" * 70)

    for name, img in images.items():
        print(f"\n  {name}:")

        # CPU features
        cpu_feat = extract_cpu(img)

        # Edge TPU features
        tpu_feat = flow._extract_features_uint8(img)

        # Compare
        diff = cpu_feat.astype(np.int16) - tpu_feat.astype(np.int16)

        print(f"    CPU  shape={cpu_feat.shape}, dtype={cpu_feat.dtype}")
        print(f"    TPU  shape={tpu_feat.shape}, dtype={tpu_feat.dtype}")
        print(f"    Diff: mean={diff.mean():.2f}, std={diff.std():.2f}, "
              f"max_abs={np.max(np.abs(diff))}")

        # Per-channel comparison
        for ch in range(8):
            cpu_ch = cpu_feat[:, :, ch].astype(float)
            tpu_ch = tpu_feat[:, :, ch].astype(float)
            ch_diff = cpu_ch - tpu_ch
            corr = np.corrcoef(cpu_ch.ravel(), tpu_ch.ravel())[0, 1]
            print(f"      Ch{ch}: cpu_mean={cpu_ch.mean():.1f} tpu_mean={tpu_ch.mean():.1f} "
                  f"diff_mean={ch_diff.mean():.2f} diff_std={ch_diff.std():.2f} "
                  f"corr={corr:.4f}")

    # ── Full flow pipeline comparison ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Full Pipeline: Edge TPU Flow vs Expected")
    print("=" * 70)

    np.random.seed(42)
    base_img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    test_cases = [
        (0, 0, 0.0, 0.0, "No shift"),
        (0, 8, 0.0, 2.0, "Right 8px (2 pooled)"),
        (0, -8, 0.0, -2.0, "Left 8px"),
        (8, 0, 2.0, 0.0, "Down 8px (2 pooled)"),
        (-8, 0, -2.0, 0.0, "Up 8px"),
        (0, 4, 0.0, 1.0, "Right 4px (1 pooled)"),
        (4, 0, 1.0, 0.0, "Down 4px (1 pooled)"),
    ]

    all_pass = True
    for shift_y, shift_x, exp_vy, exp_vx, desc in test_cases:
        img_shifted = np.roll(np.roll(base_img, shift_y, axis=0),
                              shift_x, axis=1)

        vx, vy = flow.compute(base_img, img_shifted)

        ok_x = abs(vx - exp_vx) < 0.5
        ok_y = abs(vy - exp_vy) < 0.5
        status = "PASS" if (ok_x and ok_y) else "FAIL"
        if not (ok_x and ok_y):
            all_pass = False

        print(f"  [{status}] {desc}: expected ({exp_vx:.1f}, {exp_vy:.1f}), "
              f"got ({vx:.2f}, {vy:.2f})")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # ── Test with simulated panning pattern ───────────────────────────────
    print("\n" + "=" * 70)
    print("Panning Pattern Simulation (varied vertical stripes)")
    print("=" * 70)

    # Create a wide canvas with varied vertical stripes (like the GUI does)
    canvas_width = w * 3
    canvas = np.zeros((h, canvas_width), dtype=np.uint8)
    np.random.seed(42)
    x_pos = 0
    while x_pos < canvas_width:
        stripe_width = np.random.randint(8, 32)
        intensity = 255 if (x_pos // 20) % 2 == 0 else 0
        canvas[:, x_pos:x_pos + stripe_width] = intensity
        x_pos += stripe_width

    # Add horizontal dashes for 2D texture
    for x in range(0, canvas_width, 60):
        for y_offset in [10, 25, 42, 55]:
            y = (y_offset + (x // 120) * 3) % h
            dash_end = min(x + 20 + (x % 30), canvas_width)
            canvas[y, x:dash_end] = 180

    for shift in [1, 2, 4, 8]:
        frame_t = canvas[:, 0:w]
        frame_t1 = canvas[:, shift:shift + w]

        vx, vy = flow.compute(frame_t, frame_t1)
        exp_vx = shift / 4.0  # pool_factor=4
        ok_x = abs(vx - exp_vx) < 0.5
        ok_y = abs(vy) < 0.5
        status = "PASS" if (ok_x and ok_y) else "FAIL"
        print(f"  [{status}] Pan right {shift}px: expected ({exp_vx:.2f}, 0.0), "
              f"got ({vx:.2f}, {vy:.2f})")

    flow.close()
    print("\nDone.")


if __name__ == "__main__":
    test_edgetpu_vs_cpu()
