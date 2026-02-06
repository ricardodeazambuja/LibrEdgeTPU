#!/usr/bin/env python3
"""Test if the Edge TPU preserves spatial structure with minimal models.

Tests progressively more complex models to find where tiling starts:
1. QUANTIZE only (identity passthrough)
2. 1x1 Conv2D (channel expansion, no spatial mixing)
3. 3x3 Conv2D with SAME padding
4. 7x7 Conv2D with SAME padding (our actual model)

If the DMA output is tiled differently than NHWC, even the identity
passthrough will show the wrong spatial structure.
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
    _build_buffer, _build_tensor, _build_quantization,
    _build_conv2d_options, _build_operator, _build_operator_code,
    _build_subgraph, _build_model,
    TensorType, Padding, Activation, BuiltinOp, BuiltinOptions,
)
from libredgetpu._base import EdgeTPUModelBase


def build_quantize_only(height, width):
    """Minimal model: just QUANTIZE uint8→uint8 (identity)."""
    builder = flatbuffers.Builder(1024)

    buf_list = [_build_buffer(builder, None)]  # sentinel
    buf_list.append(_build_buffer(builder, None))  # input
    buf_list.append(_build_buffer(builder, None))  # output

    q_in = _build_quantization(builder, [1.0], [0])
    q_out = _build_quantization(builder, [1.0], [0])

    t0 = _build_tensor(builder, "input", [1, height, width, 1],
                        TensorType.UINT8, 1, q_in)
    t1 = _build_tensor(builder, "output", [1, height, width, 1],
                        TensorType.UINT8, 2, q_out)

    op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE

    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)

    sg = _build_subgraph(builder, tensors=[t0, t1], inputs=[0], outputs=[1],
                         operators=[op0], name="main")

    tflite_bytes = _build_model(builder, operator_codes=[oc0],
                                 subgraphs=[sg], buffers=buf_list,
                                 description="Identity", version=3)
    return tflite_bytes


def build_conv1x1(height, width, n_out=8):
    """Model: QUANTIZE → Conv2D(1x1) → QUANTIZE. No spatial mixing."""
    builder = flatbuffers.Builder(2048)

    # 1x1 weights: each output channel copies the input (identity)
    weights = np.ones((n_out, 1, 1, 1), dtype=np.int8) * 127
    bias = np.zeros(n_out, dtype=np.int32)

    input_scale = 1.0 / 255.0
    q_int8_scale = 1.0 / 255.0
    weight_scale = 1.0 / 127.0
    output_scale = 1.0 / 255.0  # Should reproduce input

    buf_list = [_build_buffer(builder, None)]
    buf_list.append(_build_buffer(builder, None))  # input
    buf_list.append(_build_buffer(builder, None))  # quantize out
    buf_list.append(_build_buffer(builder, weights.tobytes()))  # weights
    buf_list.append(_build_buffer(builder, bias.tobytes()))  # bias
    buf_list.append(_build_buffer(builder, None))  # conv out
    buf_list.append(_build_buffer(builder, None))  # output

    q_input = _build_quantization(builder, [input_scale], [0])
    q_int8 = _build_quantization(builder, [q_int8_scale], [-128])
    q_weights = _build_quantization(builder, [weight_scale] * n_out, [0] * n_out,
                                     quantized_dimension=0)
    bias_scales = [q_int8_scale * weight_scale] * n_out
    q_bias = _build_quantization(builder, bias_scales, [0] * n_out,
                                  quantized_dimension=0)
    q_conv_out = _build_quantization(builder, [output_scale], [-128])
    q_final = _build_quantization(builder, [output_scale], [0])

    t0 = _build_tensor(builder, "input", [1, height, width, 1],
                        TensorType.UINT8, 1, q_input)
    t1 = _build_tensor(builder, "quant_out", [1, height, width, 1],
                        TensorType.INT8, 2, q_int8)
    t2 = _build_tensor(builder, "weights", [n_out, 1, 1, 1],
                        TensorType.INT8, 3, q_weights)
    t3 = _build_tensor(builder, "bias", [n_out],
                        TensorType.INT32, 4, q_bias)
    t4 = _build_tensor(builder, "conv_out", [1, height, width, n_out],
                        TensorType.INT8, 5, q_conv_out)
    t5 = _build_tensor(builder, "output", [1, height, width, n_out],
                        TensorType.UINT8, 6, q_final)

    conv_opts = _build_conv2d_options(builder, padding=Padding.VALID,
                                      activation=Activation.NONE)
    op0 = _build_operator(builder, 0, [0], [1])
    op1 = _build_operator(builder, 1, [1, 2, 3], [4],
                          int(BuiltinOptions.Conv2DOptions), conv_opts)
    op2 = _build_operator(builder, 0, [4], [5])

    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.CONV_2D)

    sg = _build_subgraph(builder, tensors=[t0, t1, t2, t3, t4, t5],
                         inputs=[0], outputs=[5],
                         operators=[op0, op1, op2], name="main")

    tflite_bytes = _build_model(builder, operator_codes=[oc0, oc1],
                                 subgraphs=[sg], buffers=buf_list,
                                 description="Conv1x1", version=3)

    metadata = {
        "height": height, "width": width, "num_filters": n_out,
        "input_scale": input_scale, "input_zero_point": 0,
        "output_scale": output_scale, "output_zero_point": 0,
    }
    return tflite_bytes, metadata


def compile_and_load(tflite_bytes, name, tmpdir, metadata=None):
    """Compile with edgetpu_compiler and load."""
    tflite_path = os.path.join(tmpdir, f"{name}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", tmpdir, tflite_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Compilation failed for {name}: {result.stderr}")
        return None

    edgetpu_path = os.path.join(tmpdir, f"{name}_edgetpu.tflite")
    if not os.path.exists(edgetpu_path):
        print(f"  Output not found: {edgetpu_path}")
        return None

    # Save metadata
    json_path = edgetpu_path + ".json"
    if metadata:
        with open(json_path, "w") as f:
            json.dump(metadata, f)
    else:
        with open(json_path, "w") as f:
            json.dump({"height": 64, "width": 64}, f)

    return edgetpu_path, json_path


def test_spatial_structure(name, raw_output, h, w, n_ch, input_img):
    """Test if output preserves spatial structure."""
    feat = np.frombuffer(raw_output, dtype=np.uint8)[:h * w * n_ch].reshape(h, w, n_ch)

    print(f"\n  {name}:")

    if n_ch == 1:
        # Compare with input directly
        diff = feat[:, :, 0].astype(int) - input_img.astype(int)
        print(f"    Input → Output diff: mean={diff.mean():.2f}, "
              f"max_abs={np.max(np.abs(diff))}")
        # Check row structure
        for r in [0, 4, 8, 16, 32, 48, 60]:
            if r < h:
                vals = feat[r, :4, 0]
                input_vals = input_img[r, :4]
                print(f"    Row {r:2d}: out=[{', '.join(f'{v:3d}' for v in vals)}] "
                      f"in=[{', '.join(f'{v:3d}' for v in input_vals)}]")
    else:
        # Check spatial coherence for multi-channel output
        for ch in [0, 2, 4]:
            ch_data = feat[:, :, ch].astype(float)
            # Vertical autocorrelation
            v_corr = np.corrcoef(ch_data[:-1, :].ravel(),
                                  ch_data[1:, :].ravel())[0, 1]
            # Horizontal autocorrelation
            h_corr = np.corrcoef(ch_data[:, :-1].ravel(),
                                  ch_data[:, 1:].ravel())[0, 1]
            print(f"    Ch{ch}: H_autocorr={h_corr:.3f}, V_autocorr={v_corr:.3f}")

        # Check if gradient input produces gradient output
        for ch in [0]:
            ch_data = feat[:, :, ch].astype(float)
            row_means = ch_data.mean(axis=1)
            col_means = ch_data.mean(axis=0)
            print(f"    Ch0 row means (first 8): {[f'{v:.1f}' for v in row_means[:8]]}")
            print(f"    Ch0 col means (first 8): {[f'{v:.1f}' for v in col_means[:8]]}")


def main():
    h, w = 64, 64
    tmpdir = tempfile.mkdtemp()
    print(f"Working dir: {tmpdir}")

    # Create gradient images
    grad_h = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))  # varies horizontally
    grad_v = np.tile(np.linspace(0, 255, h, dtype=np.uint8), (w, 1)).T.astype(np.uint8)  # varies vertically

    # Test 1: Conv1x1 (channel expansion, no spatial mixing)
    print("\n" + "=" * 70)
    print("TEST: Conv2D 1×1 (no spatial mixing, just channel expansion)")
    print("=" * 70)

    tflite_bytes, metadata = build_conv1x1(h, w, n_out=8)
    result = compile_and_load(tflite_bytes, "conv1x1", tmpdir, metadata)

    if result:
        edgetpu_path, json_path = result
        from libredgetpu._base import EdgeTPUModelBase

        class SimpleModel(EdgeTPUModelBase):
            def _default_output_size(self):
                return 64 * 64 * 8

        model = SimpleModel(edgetpu_path, metadata_path=json_path)
        model.open()

        for name, img in [("Gradient H", grad_h), ("Gradient V", grad_v)]:
            # Normalize like OpticalFlow does
            img_norm = img.astype(np.float32) / 255.0
            from libredgetpu._quantize import quantize_uint8
            quantized = quantize_uint8(img_norm,
                                        model._input_info.scale,
                                        model._input_info.zero_point)
            raw = model._execute_raw(quantized.tobytes())
            test_spatial_structure(name, raw, h, w, 8, img)

        # Test edge localization with 1x1 conv
        print("\n  Edge localization with 1x1 conv:")
        for edge_row in [16, 32, 48]:
            img = np.zeros((h, w), dtype=np.uint8)
            img[:edge_row, :] = 255
            img_norm = img.astype(np.float32) / 255.0
            quantized = quantize_uint8(img_norm, model._input_info.scale,
                                        model._input_info.zero_point)
            raw = model._execute_raw(quantized.tobytes())
            feat = np.frombuffer(raw, dtype=np.uint8)[:h * w * 8].reshape(h, w, 8)
            row_means = feat[:, :, 0].mean(axis=1)
            peak = np.argmax(row_means)
            # Print transition rows
            print(f"    Edge at row {edge_row}: peak at {peak}")
            for r in range(max(0, edge_row - 4), min(h, edge_row + 4)):
                vals = [f"{feat[r, 0, ch]:3d}" for ch in range(8)]
                print(f"      row {r}: [{' '.join(vals)}]")

        model.close()

    print(f"\nAll temp files in: {tmpdir}")
    print("Done.")


if __name__ == "__main__":
    main()
