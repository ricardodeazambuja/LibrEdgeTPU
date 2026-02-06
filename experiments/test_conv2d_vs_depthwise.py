#!/usr/bin/env python3
"""Test Conv2D vs DepthwiseConv2D on Edge TPU.

ROOT CAUSE HYPOTHESIS: edgetpu_compiler handles DEPTHWISE_CONV_2D with
depth_multiplier=8 incorrectly, producing tiled features that don't
preserve spatial position.

FIX HYPOTHESIS: Using regular CONV_2D (which is mathematically identical
for 1 input channel) should work correctly because the compiler handles
CONV_2D with different codegen.

This test:
1. Builds the model with CONV_2D [8, 7, 7, 1] weights
2. Compiles it with edgetpu_compiler
3. Tests edge localization on the Edge TPU
4. If it works, we found the fix!
"""

import numpy as np
import sys
import os
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_optical_flow_conv2d(height, width, ksize=7, orientations=4,
                               sigmas=(1.5, 3.0)):
    """Build optical flow model using standard CONV_2D instead of DEPTHWISE_CONV_2D."""
    import flatbuffers
    from libredgetpu.tflite_builder import (
        _generate_gabor_kernels,
        _build_buffer, _build_tensor, _build_quantization,
        _build_conv2d_options, _build_operator, _build_operator_code,
        _build_subgraph, _build_model,
        TensorType, Padding, Activation, BuiltinOp, BuiltinOptions,
    )

    n_filters = orientations * len(sigmas)

    # Generate Gabor kernels in HWIO format [ksize, ksize, 1, n_filters]
    gabor_hwio = _generate_gabor_kernels(ksize, orientations, sigmas)

    # For CONV_2D: weight format is OHWI = [out_ch, kH, kW, in_ch]
    # = [n_filters, ksize, ksize, 1]
    gabor_float = np.transpose(gabor_hwio, (3, 0, 1, 2))  # [n_filters, ksize, ksize, 1]

    # Quantization parameters (same as current depthwise version)
    input_scale = 1.0 / 255.0
    input_zp = 0
    q_int8_scale = 1.0 / 255.0
    q_int8_zp = -128

    # Per-channel weight quantization
    per_ch_weight_scales = []
    gabor_int8 = np.zeros_like(gabor_float, dtype=np.int8)

    for ch in range(n_filters):
        kernel_ch = gabor_float[ch, :, :, 0]  # [ksize, ksize] - CONV_2D OHWI indexing
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

    # Build FlatBuffer
    weight_bytes = gabor_int8.tobytes()
    builder = flatbuffers.Builder(4096 + len(weight_bytes))

    buf_list = [_build_buffer(builder, None)]       # 0: sentinel
    buf_list.append(_build_buffer(builder, None))    # 1: input
    buf_list.append(_build_buffer(builder, None))    # 2: quantize out
    buf_list.append(_build_buffer(builder, gabor_int8.tobytes()))  # 3: weights
    buf_list.append(_build_buffer(builder, conv_bias.tobytes()))   # 4: bias
    buf_list.append(_build_buffer(builder, None))    # 5: conv out
    buf_list.append(_build_buffer(builder, None))    # 6: final output
    buf_list.append(_build_buffer(builder, None))    # 7
    buf_list.append(_build_buffer(builder, None))    # 8

    per_ch_weight_zps = [0] * n_filters
    per_ch_bias_zps = [0] * n_filters

    q_input = _build_quantization(builder, [input_scale], [input_zp])
    q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
    # CONV_2D: quantized_dimension=0 (output channel axis in OHWI)
    q_weights = _build_quantization(builder, per_ch_weight_scales, per_ch_weight_zps,
                                     quantized_dimension=0)
    q_bias = _build_quantization(builder, per_ch_bias_scales, per_ch_bias_zps,
                                  quantized_dimension=0)
    q_conv_out = _build_quantization(builder, [conv_output_scale], [conv_output_zp])
    q_final = _build_quantization(builder, [final_output_scale], [final_output_zp])

    # Tensors
    t0 = _build_tensor(builder, "input", [1, height, width, 1],
                        TensorType.UINT8, 1, q_input)
    t1 = _build_tensor(builder, "quantize_out", [1, height, width, 1],
                        TensorType.INT8, 2, q_int8)
    # CONV_2D weights: [n_filters, ksize, ksize, 1] = OHWI format
    t2 = _build_tensor(builder, "gabor_weights",
                        [n_filters, ksize, ksize, 1],
                        TensorType.INT8, 3, q_weights)
    t3 = _build_tensor(builder, "conv_bias", [n_filters],
                        TensorType.INT32, 4, q_bias)
    t4 = _build_tensor(builder, "conv_out", [1, height, width, n_filters],
                        TensorType.INT8, 5, q_conv_out)
    t5 = _build_tensor(builder, "output", [1, height, width, n_filters],
                        TensorType.UINT8, 6, q_final)

    tensors = [t0, t1, t2, t3, t4, t5]

    # CONV_2D options (not DEPTHWISE_CONV_2D!)
    conv_opts = _build_conv2d_options(builder, padding=Padding.SAME,
                                      activation=Activation.RELU)

    # Operators
    op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE uint8→int8
    op1 = _build_operator(builder, 1, [1, 2, 3], [4],  # CONV_2D
                          int(BuiltinOptions.Conv2DOptions), conv_opts)
    op2 = _build_operator(builder, 0, [4], [5])  # QUANTIZE int8→uint8

    operators = [op0, op1, op2]

    # Operator codes - CONV_2D, not DEPTHWISE_CONV_2D!
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.CONV_2D)
    opcodes = [oc0, oc1]

    sg = _build_subgraph(builder,
                         tensors=tensors,
                         inputs=[0],
                         outputs=[5],
                         operators=operators,
                         name="main")

    desc = f"OpticalFlow Conv2D Gabor {height}x{width} (libredgetpu)"
    tflite_bytes = _build_model(
        builder,
        operator_codes=opcodes,
        subgraphs=[sg],
        buffers=buf_list,
        description=desc,
        version=3,
    )

    metadata = {
        "height": height,
        "width": width,
        "ksize": ksize,
        "orientations": orientations,
        "sigmas": list(sigmas),
        "num_filters": n_filters,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zp),
        "output_scale": float(final_output_scale),
        "output_zero_point": int(final_output_zp),
        "output_count": height * width * n_filters,
        "gabor_weight_scale": float(gabor_weight_scale),
    }

    return tflite_bytes, metadata


def main():
    h, w = 64, 64

    # Build CONV_2D model
    print("Building Conv2D model...")
    tflite_bytes, metadata = build_optical_flow_conv2d(h, w)

    # Save to temp file
    tmpdir = tempfile.mkdtemp()
    tflite_path = os.path.join(tmpdir, "gabor_conv2d.tflite")
    json_path = os.path.join(tmpdir, "gabor_conv2d.tflite.json")

    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved to {tflite_path} ({len(tflite_bytes)} bytes)")

    # Compile with edgetpu_compiler
    print("\nCompiling with edgetpu_compiler...")
    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", tmpdir, tflite_path],
        capture_output=True, text=True
    )
    print(f"  stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr}")
        print("  FAILED to compile!")
        return

    edgetpu_path = os.path.join(tmpdir, "gabor_conv2d_edgetpu.tflite")
    edgetpu_json = os.path.join(tmpdir, "gabor_conv2d_edgetpu.tflite.json")

    if not os.path.exists(edgetpu_path):
        print(f"  Expected output not found at {edgetpu_path}")
        # List files in tmpdir
        for f in os.listdir(tmpdir):
            print(f"    {f}")
        return

    # Copy metadata
    with open(edgetpu_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Compiled: {edgetpu_path}")

    # Load on Edge TPU
    print("\nLoading on Edge TPU...")
    from libredgetpu.optical_flow_module import OpticalFlow

    flow = OpticalFlow(edgetpu_path, metadata_path=edgetpu_json)
    flow.open()

    # Test edge localization
    print("\n" + "=" * 70)
    print("CONV_2D Edge Localization Test")
    print("=" * 70)

    print("\n  Moving horizontal edge (Ch0 peak should track edge position):")
    for edge_row in [16, 24, 32, 40, 48]:
        img = np.zeros((h, w), dtype=np.uint8)
        img[:edge_row, :] = 255
        feat = flow._extract_features_uint8(img)
        row_means = feat[:, :, 0].mean(axis=1)
        peak = np.argmax(row_means)
        err = abs(peak - edge_row)
        status = "✓" if err <= 4 else "✗"
        print(f"    Edge at row {edge_row}: Ch0 peak at row {peak} "
              f"(error={err}) {status}")

    print()
    print("  Moving vertical edge (Ch2 peak should track edge position):")
    for edge_col in [16, 24, 32, 40, 48]:
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, :edge_col] = 255
        feat = flow._extract_features_uint8(img)
        col_means = feat[:, :, 2].mean(axis=0)
        peak = np.argmax(col_means)
        err = abs(peak - edge_col)
        status = "✓" if err <= 4 else "✗"
        print(f"    Edge at col {edge_col}: Ch2 peak at col {peak} "
              f"(error={err}) {status}")

    # Test self-correlation
    print("\n" + "=" * 70)
    print("CONV_2D Self-Correlation Test")
    print("=" * 70)

    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    feat_base = flow._extract_features_uint8(img)

    for shift_y, shift_x, desc in [(0, 4, "Right 4px"), (4, 0, "Down 4px"),
                                    (0, 8, "Right 8px"), (8, 0, "Down 8px")]:
        img_shifted = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
        feat_shifted = flow._extract_features_uint8(img_shifted)

        m = 7  # Exclude boundary
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

    # Test full flow pipeline
    print("\n" + "=" * 70)
    print("CONV_2D Full Flow Pipeline Test")
    print("=" * 70)

    np.random.seed(42)
    base_img = np.random.randint(0, 256, (h, w), dtype=np.uint8)

    for shift_y, shift_x, exp_vy, exp_vx, desc in [
        (0, 0, 0.0, 0.0, "No shift"),
        (0, 8, 0.0, 2.0, "Right 8px"),
        (0, -8, 0.0, -2.0, "Left 8px"),
        (8, 0, 2.0, 0.0, "Down 8px"),
        (-8, 0, -2.0, 0.0, "Up 8px"),
        (0, 4, 0.0, 1.0, "Right 4px"),
        (4, 0, 1.0, 0.0, "Down 4px"),
    ]:
        img_shifted = np.roll(np.roll(base_img, shift_y, axis=0),
                              shift_x, axis=1)
        vx, vy = flow.compute(base_img, img_shifted)

        ok_x = abs(vx - exp_vx) < 0.5
        ok_y = abs(vy - exp_vy) < 0.5
        status = "PASS" if (ok_x and ok_y) else "FAIL"
        print(f"  [{status}] {desc}: expected ({exp_vx:.1f}, {exp_vy:.1f}), "
              f"got ({vx:.2f}, {vy:.2f})")

    flow.close()
    print(f"\nTemp dir: {tmpdir}")
    print("Done.")


if __name__ == "__main__":
    main()
