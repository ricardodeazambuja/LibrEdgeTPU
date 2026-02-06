"""Test all possible transpose permutations to find the correct one."""

import numpy as np
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tensorflow as tf
except ImportError:
    print("⚠️  TensorFlow not installed")
    sys.exit(1)

import scipy.ndimage
from libredgetpu.tflite_builder import _generate_gabor_kernels, _build_buffer, _build_quantization, _build_tensor, _build_conv2d_options, _build_operator, _build_operator_code, _build_subgraph, _build_model
from flatbuffers import Builder
from libredgetpu.tflite_parser import TensorType, Padding, Activation, BuiltinOp, BuiltinOptions


def manual_gabor_cpu(image):
    """Manual Gabor (ground truth)."""
    kernels = _generate_gabor_kernels(7, 4, (1.5, 3.0))  # [7, 7, 1, 8]
    n_filters = 8
    h, w = image.shape
    input_int8 = image.astype(np.int16) - 128

    gabor_abs_max = float(np.max(np.abs(kernels)))
    gabor_weight_scale = max(gabor_abs_max, 1e-6) / 127.0
    kernels_int8 = np.clip(np.round(kernels / gabor_weight_scale), -127, 127).astype(np.int8)

    conv_output_max = 1.0 * gabor_weight_scale * 7 * 7 * 127
    conv_output_scale = conv_output_max / 127.0

    features = np.zeros((h, w, n_filters), dtype=np.float32)
    for i in range(8):
        kernel = kernels_int8[:, :, 0, i].astype(np.float32)
        conv_int = scipy.ndimage.convolve(input_int8.astype(np.float32), kernel, mode='constant', cval=0.0)
        conv_float = conv_int * gabor_weight_scale
        conv_float = np.maximum(conv_float, 0.0)
        features[:, :, i] = conv_float

    return np.clip(np.round(features / conv_output_scale), 0, 255).astype(np.uint8)


def build_model_with_transpose(transpose_axes):
    """Build optical flow model with specific transpose."""
    from libredgetpu.tflite_builder import build_optical_flow

    # Generate kernels
    gabor_float = _generate_gabor_kernels(7, 4, (1.5, 3.0))  # [7, 7, 1, 8]

    # Quantize
    gabor_abs_max = float(np.max(np.abs(gabor_float)))
    gabor_weight_scale = max(gabor_abs_max, 1e-6) / 127.0
    gabor_int8 = np.clip(np.round(gabor_float / gabor_weight_scale), -127, 127).astype(np.int8)

    # Apply transpose
    if transpose_axes is not None:
        gabor_transposed = np.transpose(gabor_int8, transpose_axes)
    else:
        gabor_transposed = gabor_int8

    # Build minimal model (QUANTIZE → CONV → QUANTIZE)
    size = 64
    builder = Builder(8192)

    # Buffers
    bufs = [_build_buffer(builder, None)]
    bufs.append(_build_buffer(builder, None))  # input
    bufs.append(_build_buffer(builder, None))  # quantize out
    bufs.append(_build_buffer(builder, gabor_transposed.tobytes()))  # weights
    bufs.append(_build_buffer(builder, np.zeros(8, dtype=np.int32).tobytes()))  # bias
    bufs.append(_build_buffer(builder, None))  # conv out
    bufs.append(_build_buffer(builder, None))  # final out

    # Quant params
    q_in = _build_quantization(builder, [1.0], [0])
    q_int8 = _build_quantization(builder, [1.0], [-128])
    conv_output_max = 1.0 * gabor_weight_scale * 7 * 7 * 127
    conv_output_scale = conv_output_max / 127.0
    q_conv = _build_quantization(builder, [conv_output_scale], [-128])
    q_out = _build_quantization(builder, [conv_output_scale], [0])
    q_w = _build_quantization(builder, [gabor_weight_scale]*8, [0]*8)
    q_b = _build_quantization(builder, [1.0 * gabor_weight_scale]*8, [0]*8)

    # Tensors
    t0 = _build_tensor(builder, "in", [1, size, size, 1], TensorType.UINT8, 1, q_in)
    t1 = _build_tensor(builder, "qi", [1, size, size, 1], TensorType.INT8, 2, q_int8)
    t2 = _build_tensor(builder, "w", [8, 7, 7, 1], TensorType.INT8, 3, q_w)
    t3 = _build_tensor(builder, "b", [8], TensorType.INT32, 4, q_b)
    t4 = _build_tensor(builder, "cv", [1, size, size, 8], TensorType.INT8, 5, q_conv)
    t5 = _build_tensor(builder, "out", [1, size, size, 8], TensorType.UINT8, 6, q_out)

    tensors = [t0, t1, t2, t3, t4, t5]

    # Ops
    conv_opts = _build_conv2d_options(builder, Padding.SAME, Activation.RELU)
    op0 = _build_operator(builder, 0, [0], [1])
    op1 = _build_operator(builder, 1, [1, 2, 3], [4], int(BuiltinOptions.Conv2DOptions), conv_opts)
    op2 = _build_operator(builder, 0, [4], [5])

    ops = [op0, op1, op2]

    # Opcodes
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.CONV_2D)
    opcodes = [oc0, oc1]

    # Subgraph
    sg = _build_subgraph(builder, tensors, [0], [5], ops, "main")

    # Model
    return _build_model(builder, opcodes, [sg], bufs, "test", 3)


def test_transpose(transpose_axes, feat_manual, image):
    """Test a specific transpose."""
    try:
        tflite_bytes = build_model_with_transpose(transpose_axes)

        # Run TFLite CPU
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        interp.set_tensor(inp['index'], image.reshape(1, 64, 64, 1))
        interp.invoke()
        feat_tflite = interp.get_tensor(out['index']).squeeze()

        # Compare
        mae = np.abs(feat_manual.astype(np.int16) - feat_tflite.astype(np.int16)).mean()
        corr = np.corrcoef(feat_manual.ravel(), feat_tflite.ravel())[0, 1]

        return mae, corr
    except Exception as e:
        return None, None


def main():
    print("="*60)
    print("Testing All Transpose Permutations")
    print("="*60)

    size = 64
    np.random.seed(42)
    image = (np.random.rand(size, size) * 255).astype(np.uint8)

    # Ground truth
    print("\nGenerating ground truth...")
    feat_manual = manual_gabor_cpu(image)

    # Test all permutations of (0, 1, 2, 3)
    print("\nTesting all 24 permutations + no transpose...")
    results = []

    # No transpose
    mae, corr = test_transpose(None, feat_manual, image)
    if mae is not None:
        results.append(("None (HWIO as-is)", mae, corr))
        print(f"  None: MAE={mae:.1f}, corr={corr:.3f}")

    # All permutations
    for perm in itertools.permutations([0, 1, 2, 3]):
        mae, corr = test_transpose(perm, feat_manual, image)
        if mae is not None:
            results.append((str(perm), mae, corr))
            print(f"  {perm}: MAE={mae:.1f}, corr={corr:.3f}")

    # Sort by correlation (best first)
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'='*60}")
    print("Top 5 Results (by correlation)")
    print(f"{'='*60}")
    for i, (name, mae, corr) in enumerate(results[:5]):
        marker = "✅" if corr > 0.99 else ("⚠️" if corr > 0.95 else "")
        print(f"{i+1}. {name:<25} MAE={mae:<6.1f} corr={corr:.6f} {marker}")

    best_name, best_mae, best_corr = results[0]
    if best_corr > 0.99:
        print(f"\n✅ FOUND IT! Transpose {best_name} gives perfect match!")
    else:
        print(f"\n❌ No perfect transpose found. Best: {best_name} with corr={best_corr:.6f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
