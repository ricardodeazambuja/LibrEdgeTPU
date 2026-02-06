#!/usr/bin/env python3
"""Test VALID padding as a workaround for the per-tile SAME padding bug.

If the Edge TPU applies SAME padding per-tile instead of globally,
we can work around it by:
1. Pre-padding the input ourselves (add ksize//2 on each side)
2. Using VALID padding in the model
3. The output will be exactly the original size

Test plan:
A. Build 7x7 Conv2D with VALID padding (input 70x70 → output 64x64x8)
B. Build 7x7 Conv2D with SAME padding (input 64x64 → output 64x64x8) [current, broken]
C. Compare edge localization and self-correlation
"""

import numpy as np
import sys
import os
import json
import subprocess
import tempfile
import flatbuffers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.tflite_builder import (
    _generate_gabor_kernels,
    _build_buffer, _build_tensor, _build_quantization,
    _build_conv2d_options, _build_operator, _build_operator_code,
    _build_subgraph, _build_model,
    TensorType, Padding, Activation, BuiltinOp, BuiltinOptions,
)


def build_gabor_valid(height, width, ksize=7):
    """Build Gabor model with VALID padding.

    Input: [1, height+ksize-1, width+ksize-1, 1] (pre-padded)
    Output: [1, height, width, 8]
    """
    orientations = 4
    sigmas = (1.5, 3.0)
    n_filters = orientations * len(sigmas)
    pad = ksize // 2
    in_h = height + 2 * pad  # Pre-padded input height
    in_w = width + 2 * pad   # Pre-padded input width

    # Generate kernels in OHWI format for CONV_2D
    gabor_hwio = _generate_gabor_kernels(ksize, orientations, sigmas)
    gabor_float = np.transpose(gabor_hwio, (3, 0, 1, 2))  # [n_filters, ksize, ksize, 1]

    input_scale = 1.0 / 255.0
    q_int8_scale = 1.0 / 255.0
    q_int8_zp = -128

    per_ch_weight_scales = []
    gabor_int8 = np.zeros_like(gabor_float, dtype=np.int8)
    for ch in range(n_filters):
        kernel_ch = gabor_float[ch, :, :, 0]
        ch_abs_max = float(np.max(np.abs(kernel_ch)))
        ch_scale = max(ch_abs_max, 1e-6) / 127.0
        per_ch_weight_scales.append(ch_scale)
        gabor_int8[ch, :, :, 0] = np.clip(
            np.round(kernel_ch / ch_scale), -127, 127
        ).astype(np.int8)

    per_ch_bias_scales = [q_int8_scale * ws for ws in per_ch_weight_scales]
    conv_bias = np.zeros(n_filters, dtype=np.int32)
    gabor_weight_scale = np.mean(per_ch_weight_scales)

    import math
    worst_case_acc_per_output = ksize * ksize * 127.0
    conservative_acc = worst_case_acc_per_output * 1.5
    conv_output_max = conservative_acc * q_int8_scale * gabor_weight_scale
    conv_output_scale = conv_output_max / 127.0
    conv_output_zp = -128
    final_output_scale = conv_output_scale
    final_output_zp = 0

    builder = flatbuffers.Builder(4096 + len(gabor_int8.tobytes()))

    buf_list = [_build_buffer(builder, None)]
    buf_list.append(_build_buffer(builder, None))    # input
    buf_list.append(_build_buffer(builder, None))    # quantize out
    buf_list.append(_build_buffer(builder, gabor_int8.tobytes()))  # weights
    buf_list.append(_build_buffer(builder, conv_bias.tobytes()))   # bias
    buf_list.append(_build_buffer(builder, None))    # conv out
    buf_list.append(_build_buffer(builder, None))    # output
    buf_list.append(_build_buffer(builder, None))
    buf_list.append(_build_buffer(builder, None))

    per_ch_weight_zps = [0] * n_filters
    per_ch_bias_zps = [0] * n_filters

    q_input = _build_quantization(builder, [input_scale], [0])
    q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
    q_weights = _build_quantization(builder, per_ch_weight_scales, per_ch_weight_zps,
                                     quantized_dimension=0)
    q_bias = _build_quantization(builder, per_ch_bias_scales, per_ch_bias_zps,
                                  quantized_dimension=0)
    q_conv_out = _build_quantization(builder, [conv_output_scale], [conv_output_zp])
    q_final = _build_quantization(builder, [final_output_scale], [final_output_zp])

    # NOTE: Input is pre-padded: [1, in_h, in_w, 1]
    t0 = _build_tensor(builder, "input", [1, in_h, in_w, 1],
                        TensorType.UINT8, 1, q_input)
    t1 = _build_tensor(builder, "quantize_out", [1, in_h, in_w, 1],
                        TensorType.INT8, 2, q_int8)
    t2 = _build_tensor(builder, "gabor_weights", [n_filters, ksize, ksize, 1],
                        TensorType.INT8, 3, q_weights)
    t3 = _build_tensor(builder, "conv_bias", [n_filters],
                        TensorType.INT32, 4, q_bias)
    # Output is the original size (VALID padding crops)
    t4 = _build_tensor(builder, "conv_out", [1, height, width, n_filters],
                        TensorType.INT8, 5, q_conv_out)
    t5 = _build_tensor(builder, "output", [1, height, width, n_filters],
                        TensorType.UINT8, 6, q_final)

    tensors = [t0, t1, t2, t3, t4, t5]

    # VALID padding — no automatic padding by the operator
    conv_opts = _build_conv2d_options(builder, padding=Padding.VALID,
                                      activation=Activation.RELU)

    op0 = _build_operator(builder, 0, [0], [1])
    op1 = _build_operator(builder, 1, [1, 2, 3], [4],
                          int(BuiltinOptions.Conv2DOptions), conv_opts)
    op2 = _build_operator(builder, 0, [4], [5])

    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.CONV_2D)

    sg = _build_subgraph(builder, tensors=tensors, inputs=[0], outputs=[5],
                         operators=[op0, op1, op2], name="main")

    tflite_bytes = _build_model(builder, operator_codes=[oc0, oc1],
                                 subgraphs=[sg], buffers=buf_list,
                                 description=f"Gabor VALID {height}x{width}",
                                 version=3)

    metadata = {
        "height": height,
        "width": width,
        "ksize": ksize,
        "num_filters": n_filters,
        "input_scale": float(input_scale),
        "input_zero_point": 0,
        "output_scale": float(final_output_scale),
        "output_zero_point": int(final_output_zp),
        "output_count": height * width * n_filters,
        "gabor_weight_scale": float(gabor_weight_scale),
        # Pre-padded input dimensions
        "padded_height": in_h,
        "padded_width": in_w,
        "pad": pad,
    }

    return tflite_bytes, metadata


def main():
    h, w = 64, 64
    ksize = 7
    pad = ksize // 2
    tmpdir = tempfile.mkdtemp()
    print(f"Working dir: {tmpdir}")

    # Build VALID padding model
    print("Building Gabor model with VALID padding...")
    tflite_bytes, metadata = build_gabor_valid(h, w, ksize)
    tflite_path = os.path.join(tmpdir, "gabor_valid.tflite")
    json_path = os.path.join(tmpdir, "gabor_valid_edgetpu.tflite.json")

    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Compile
    print("Compiling...")
    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", tmpdir, tflite_path],
        capture_output=True, text=True
    )
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr}")
        return

    edgetpu_path = os.path.join(tmpdir, "gabor_valid_edgetpu.tflite")
    if not os.path.exists(edgetpu_path):
        print("  Compiled model not found!")
        for f in os.listdir(tmpdir):
            print(f"    {f}")
        return

    # Copy metadata for the compiled model
    with open(edgetpu_path + ".json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Load on Edge TPU
    print("\nLoading on Edge TPU...")
    from libredgetpu._base import EdgeTPUModelBase
    from libredgetpu._quantize import quantize_uint8

    class ValidModel(EdgeTPUModelBase):
        def _default_output_size(self):
            return 64 * 64 * 8

    model = ValidModel(edgetpu_path, metadata_path=edgetpu_path + ".json")
    model.open()

    def extract_features_valid(img_64x64):
        """Pad input manually, send to Edge TPU, get 64x64x8 output."""
        # Pre-pad with zeros (black)
        padded = np.pad(img_64x64, ((pad, pad), (pad, pad)), mode='constant',
                        constant_values=0)
        # Normalize and quantize (same as OpticalFlow)
        img_norm = padded.astype(np.float32) / 255.0
        quantized = quantize_uint8(img_norm, model._input_info.scale,
                                    model._input_info.zero_point)
        raw = model._execute_raw(quantized.tobytes())
        n = h * w * 8
        return np.frombuffer(raw, dtype=np.uint8)[:n].reshape(h, w, 8)

    # ── Test 1: Edge Localization ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALID PADDING: Edge Localization Test")
    print("=" * 70)

    print("\n  Moving horizontal edge (Ch0 peak should track edge position):")
    for edge_row in [16, 24, 32, 40, 48]:
        img = np.zeros((h, w), dtype=np.uint8)
        img[:edge_row, :] = 255
        feat = extract_features_valid(img)
        row_means = feat[:, :, 0].mean(axis=1)
        peak = np.argmax(row_means)
        err = abs(peak - edge_row)
        status = "✓" if err <= 4 else "✗"
        print(f"    Edge at row {edge_row}: Ch0 peak at row {peak} (error={err}) {status}")

    print()
    print("  Moving vertical edge (Ch2 peak should track edge position):")
    for edge_col in [16, 24, 32, 40, 48]:
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, :edge_col] = 255
        feat = extract_features_valid(img)
        col_means = feat[:, :, 2].mean(axis=0)
        peak = np.argmax(col_means)
        err = abs(peak - edge_col)
        status = "✓" if err <= 4 else "✗"
        print(f"    Edge at col {edge_col}: Ch2 peak at col {peak} (error={err}) {status}")

    # ── Test 2: Self-Correlation ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALID PADDING: Self-Correlation Test")
    print("=" * 70)

    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    feat_base = extract_features_valid(img)

    for shift_y, shift_x, desc in [(0, 4, "Right 4px"), (4, 0, "Down 4px"),
                                    (0, 8, "Right 8px"), (8, 0, "Down 8px")]:
        img_shifted = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
        feat_shifted = extract_features_valid(img_shifted)

        m = 7
        if shift_x > 0:
            base_int = feat_base[m:-m, m+shift_x:-m, :]
            shift_int = feat_shifted[m:-m, m:-m-shift_x, :]
        else:
            base_int = feat_base[m+shift_y:-m, m:-m, :]
            shift_int = feat_shifted[m:-m-shift_y, m:-m, :]

        per_ch = []
        for ch in range(8):
            b = base_int[:, :, ch].astype(float).ravel()
            s = shift_int[:, :, ch].astype(float).ravel()
            if b.std() < 0.01 or s.std() < 0.01:
                per_ch.append(float('nan'))
            else:
                per_ch.append(np.corrcoef(b, s)[0, 1])

        mean_c = np.nanmean(per_ch)
        print(f"  {desc}: mean_corr={mean_c:.3f} "
              f"({' '.join(f'{c:.3f}' for c in per_ch)})")

    # ── Test 3: Full Flow Pipeline ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALID PADDING: Full Flow Pipeline Test")
    print("=" * 70)

    from experiments.diagnose_optical_flow import (
        global_correlation, soft_argmax, compute_overlap_counts
    )

    np.random.seed(42)
    base_img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    out_zp = 0  # from metadata

    pool = 4
    sr = 4

    for shift_y, shift_x, exp_vy, exp_vx, desc in [
        (0, 0, 0.0, 0.0, "No shift"),
        (0, 8, 0.0, 2.0, "Right 8px"),
        (8, 0, 2.0, 0.0, "Down 8px"),
        (0, -8, 0.0, -2.0, "Left 8px"),
        (-8, 0, -2.0, 0.0, "Up 8px"),
        (0, 4, 0.0, 1.0, "Right 4px"),
        (4, 0, 1.0, 0.0, "Down 4px"),
    ]:
        img_shifted = np.roll(np.roll(base_img, shift_y, axis=0),
                              shift_x, axis=1)

        f_base = extract_features_valid(base_img)
        f_shift = extract_features_valid(img_shifted)

        f_base_int = f_base.astype(np.int16) - np.int16(out_zp)
        f_shift_int = f_shift.astype(np.int16) - np.int16(out_zp)

        f_base_p = f_base_int.reshape(h // pool, pool, w // pool, pool, 8).sum(
            axis=(1, 3), dtype=np.int32).astype(np.float32)
        f_shift_p = f_shift_int.reshape(h // pool, pool, w // pool, pool, 8).sum(
            axis=(1, 3), dtype=np.int32).astype(np.float32)

        overlap = compute_overlap_counts(h // pool, w // pool, 8, sr)
        corr = global_correlation(f_base_p, f_shift_p, sr).astype(np.float64)
        corr /= overlap
        vx, vy = soft_argmax(corr.astype(np.float32), sr)

        ok_x = abs(vx - exp_vx) < 0.5
        ok_y = abs(vy - exp_vy) < 0.5
        status = "PASS" if (ok_x and ok_y) else "FAIL"
        print(f"  [{status}] {desc}: expected ({exp_vx:.1f}, {exp_vy:.1f}), "
              f"got ({vx:.2f}, {vy:.2f})")

    model.close()
    print(f"\nTemp dir: {tmpdir}")
    print("Done.")


if __name__ == "__main__":
    main()
