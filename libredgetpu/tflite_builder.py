"""TFLite FlatBuffer builder — constructs valid .tflite files without TensorFlow.

Uses the `flatbuffers` library (already a runtime dependency) to build TFLite
models from scratch.  Supports:

- ``build_dense(n)`` — Dense(N,N) for MatMulEngine template generation
- ``build_spot_tracker(h, w, ...)`` — soft argmax tracker for SpotTracker
- ``build_looming(h, w, ...)`` — Sobel edge density for LoomingDetector
- ``build_pattern_tracker(sh, sw, kh, kw, ...)`` — Conv2D correlation for PatternTracker
- ``build_optical_flow(h, w, ...)`` — Gabor feature extraction for OpticalFlow
- ``build_optical_flow_pooled(h, w, ...)`` — Gabor + AVG_POOL for OpticalFlow (reduced USB)

The FlatBuffer schema field indices and enum values are taken directly from the
TensorFlow Lite schema (schema.fbs v3).
"""

import math
from enum import IntEnum

import flatbuffers
import numpy as np

__all__ = ["build_dense", "build_spot_tracker", "build_looming", "build_pattern_tracker",
           "build_optical_flow", "build_optical_flow_pooled", "build_simple_depthwise"]


# ── TFLite schema enums ─────────────────────────────────────────────────────
# Values from tensorflow/lite/schema/schema.fbs

class TensorType(IntEnum):
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    COMPLEX64 = 8
    INT8 = 9


class BuiltinOp(IntEnum):
    ADD = 0
    AVERAGE_POOL_2D = 1
    CONCATENATION = 2
    CONV_2D = 3
    DEPTHWISE_CONV_2D = 4
    DEQUANTIZE = 6
    FULLY_CONNECTED = 9
    L2_NORMALIZATION = 17
    MUL = 18
    RESHAPE = 22
    SOFTMAX = 25
    RESIZE_BILINEAR = 23
    PAD = 27
    CUSTOM = 32
    SQUEEZE = 34
    ARG_MAX = 56
    RELU = 19
    RELU6 = 21
    FAKE_QUANT = 80
    PACK = 83
    QUANTIZE = 114
    HARD_SWISH = 117


class BuiltinOptions(IntEnum):
    """Union type discriminator for Operator.builtin_options.

    In FlatBuffers, a union ``foo:FooUnion`` generates two vtable slots:
    ``foo_type`` (uint8 discriminator) followed by ``foo`` (uoffset to table).
    The discriminator values match the 1-based position in the union declaration
    (0 = NONE).
    """
    NONE = 0
    Conv2DOptions = 1
    DepthwiseConv2DOptions = 2
    ConcatEmbeddingsOptions = 3
    LSHProjectionOptions = 4
    Pool2DOptions = 5
    SVDFOptions = 6
    RNNOptions = 7
    FullyConnectedOptions = 8
    SoftmaxOptions = 9
    ConcatenationOptions = 10
    AddOptions = 11
    ReshapeOptions = 17
    MulOptions = 21


class Padding(IntEnum):
    SAME = 0
    VALID = 1


class Activation(IntEnum):
    NONE = 0
    RELU = 1
    RELU_N1_TO_1 = 2
    RELU6 = 3
    TANH = 4
    SIGN_BIT = 5


# ── Low-level FlatBuffer table builders ─────────────────────────────────────
#
# FlatBuffer construction is bottom-up: leaves first, root last.
# Each helper returns an offset that callers embed into parent tables.

def _build_buffer(builder, data=None):
    """Build a TFLite Buffer table.  *data* is bytes or None for empty."""
    if data is not None:
        data_vec = builder.CreateByteVector(data)
    # Buffer table: field 0 = data (vector of uint8)
    builder.StartObject(1)
    if data is not None:
        builder.PrependUOffsetTRelativeSlot(0, data_vec, 0)
    return builder.EndObject()


def _build_quantization(builder, scales, zero_points, quantized_dimension=None):
    """Build a QuantizationParameters table.

    Args:
        scales: list of float
        zero_points: list of int (stored as int64 in TFLite)
        quantized_dimension: For per-channel quantization, the axis index along
            which the scales/zero_points apply (e.g., 3 for output channels in HWIO).
            If None or len(scales)==1, per-layer quantization is used.
    """
    # Pre-create vectors (must be done before StartObject)
    # Scales vector (float32)
    builder.StartVector(4, len(scales), 4)
    for s in reversed(scales):
        builder.PrependFloat32(s)
    scales_vec = builder.EndVector()

    # Zero-points vector (int64)
    builder.StartVector(8, len(zero_points), 8)
    for zp in reversed(zero_points):
        builder.PrependInt64(zp)
    zp_vec = builder.EndVector()

    # QuantizationParameters table:
    #   field 0: min (vector of float) - unused
    #   field 1: max (vector of float) - unused
    #   field 2: scale (vector of float)
    #   field 3: zero_point (vector of int64)
    #   field 4: details_type (QuantizationDetails union) - unused
    #   field 5: details (union table) - unused
    #   field 6: quantized_dimension (int32) - for per-channel quantization
    builder.StartObject(7)
    builder.PrependUOffsetTRelativeSlot(2, scales_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, zp_vec, 0)
    if quantized_dimension is not None and len(scales) > 1:
        builder.PrependInt32Slot(6, quantized_dimension, 0)
    return builder.EndObject()


def _build_tensor(builder, name, shape, dtype, buffer_idx, quant_offset=None):
    """Build a TFLite Tensor table.

    Args:
        name: tensor name string
        shape: list of int
        dtype: TensorType enum value
        buffer_idx: index into Model.buffers
        quant_offset: offset from _build_quantization, or None
    """
    # Pre-create sub-objects
    name_off = builder.CreateString(name)

    builder.StartVector(4, len(shape), 4)
    for dim in reversed(shape):
        builder.PrependInt32(dim)
    shape_vec = builder.EndVector()

    # Tensor table:
    #   field 0: shape (vector of int32)
    #   field 1: type (TensorType, uint8)
    #   field 2: buffer (uint32 — index into Model.buffers)
    #   field 3: name (string)
    #   field 4: quantization (QuantizationParameters table)
    builder.StartObject(5)
    builder.PrependUOffsetTRelativeSlot(0, shape_vec, 0)
    builder.PrependUint8Slot(1, int(dtype), 0)
    builder.PrependUint32Slot(2, buffer_idx, 0)
    builder.PrependUOffsetTRelativeSlot(3, name_off, 0)
    if quant_offset is not None:
        builder.PrependUOffsetTRelativeSlot(4, quant_offset, 0)
    return builder.EndObject()


def _build_operator_code(builder, builtin_code):
    """Build an OperatorCode table.

    Uses both deprecated_builtin_code (int8, field 0) and
    builtin_code (int32, field 3) for compatibility with all TFLite runtimes.
    """
    # OperatorCode table:
    #   field 0: deprecated_builtin_code (int8) — old runtimes
    #   field 1: custom_code (string) — old schema
    #   field 2: version (int32, default 1)
    #   field 3: builtin_code (int32) — new schema v3a+
    #   field 4: custom_code (string) — new schema
    code = int(builtin_code)
    # Deprecated field is int8, so clamp to 127 for codes > 127
    deprecated_code = min(code, 127)
    builder.StartObject(5)
    builder.PrependInt8Slot(0, deprecated_code, 0)
    builder.PrependInt32Slot(3, code, 0)
    return builder.EndObject()


def _build_fc_options(builder):
    """Build a FullyConnectedOptions table with all defaults.

    FullyConnectedOptions:
        field 0: fused_activation_function (ActivationFunctionType, default NONE=0)
        field 1: weights_format (FullyConnectedOptionsWeightsFormat, default DEFAULT=0)
        field 2: keep_num_dims (bool, default false)
        field 3: asymmetric_quantize_inputs (bool, default false)
        field 4: quantized_bias_type (TensorType, default FLOAT32=0)
    """
    builder.StartObject(5)
    return builder.EndObject()


# ── Option builders for additional ops ────────────────────────────────────────

def _build_conv2d_options(builder, padding=Padding.VALID, stride_w=1, stride_h=1,
                          activation=Activation.NONE, dilation_w=1, dilation_h=1):
    """Build a Conv2DOptions table."""
    # Conv2DOptions:
    #   field 0: padding
    #   field 1: stride_w          (schema default = 0)
    #   field 2: stride_h          (schema default = 0)
    #   field 3: fused_activation_function
    #   field 4: dilation_w_factor (schema default = 1)
    #   field 5: dilation_h_factor (schema default = 1)
    #   field 6: quantized_bias_type
    # NOTE: FlatBuffers skips fields matching their default, so the default
    # arg to PrependInt32Slot MUST match the schema default, not the Python
    # parameter default.  stride defaults to 0 in schema; dilation to 1.
    builder.StartObject(7)
    builder.PrependInt8Slot(0, int(padding), 0)
    builder.PrependInt32Slot(1, stride_w, 0)
    builder.PrependInt32Slot(2, stride_h, 0)
    builder.PrependInt8Slot(3, int(activation), 0)
    builder.PrependInt32Slot(4, dilation_w, 1)
    builder.PrependInt32Slot(5, dilation_h, 1)
    return builder.EndObject()


def _build_depthwise_conv2d_options(builder, padding=Padding.VALID, stride_w=1, stride_h=1,
                                     depth_multiplier=1, activation=Activation.NONE,
                                     dilation_w=1, dilation_h=1):
    """Build a DepthwiseConv2DOptions table.

    DepthwiseConv2DOptions schema fields:
      field 0: padding
      field 1: stride_w          (schema default = 0)
      field 2: stride_h          (schema default = 0)
      field 3: depth_multiplier  (schema default = 0)
      field 4: fused_activation_function
      field 5: dilation_w_factor (schema default = 1)
      field 6: dilation_h_factor (schema default = 1)
    """
    builder.StartObject(7)
    builder.PrependInt8Slot(0, int(padding), 0)
    builder.PrependInt32Slot(1, stride_w, 0)
    builder.PrependInt32Slot(2, stride_h, 0)
    builder.PrependInt32Slot(3, depth_multiplier, 0)
    builder.PrependInt8Slot(4, int(activation), 0)
    builder.PrependInt32Slot(5, dilation_w, 1)
    builder.PrependInt32Slot(6, dilation_h, 1)
    return builder.EndObject()


def _build_add_options(builder, activation=Activation.NONE, pot_scale_int16=True):
    """Build an AddOptions table.

    AddOptions:
        field 0: fused_activation_function
        field 1: pot_scale_int16 (bool, default true)
    """
    builder.StartObject(2)
    builder.PrependInt8Slot(0, int(activation), 0)
    builder.PrependBoolSlot(1, pot_scale_int16, True)
    return builder.EndObject()


def _build_mul_options(builder, activation=Activation.NONE):
    """Build a MulOptions table.

    MulOptions:
        field 0: fused_activation_function
    """
    builder.StartObject(1)
    builder.PrependInt8Slot(0, int(activation), 0)
    return builder.EndObject()


def _build_softmax_options(builder, beta=1.0):
    """Build a SoftmaxOptions table."""
    # SoftmaxOptions: field 0: beta (float32)
    builder.StartObject(1)
    builder.PrependFloat32Slot(0, beta, 0.0)
    return builder.EndObject()


def _build_concatenation_options(builder, axis=0, activation=Activation.NONE):
    """Build a ConcatenationOptions table."""
    # ConcatenationOptions: field 0: axis, field 1: fused_activation_function
    builder.StartObject(2)
    builder.PrependInt32Slot(0, axis, 0)
    builder.PrependInt8Slot(1, int(activation), 0)
    return builder.EndObject()


def _build_pool2d_options(builder, padding=Padding.VALID, stride_w=1, stride_h=1,
                          filter_w=1, filter_h=1, activation=Activation.NONE):
    """Build a Pool2DOptions table."""
    # Pool2DOptions:
    #   field 0: padding       (schema default = 0)
    #   field 1: stride_w      (schema default = 0)
    #   field 2: stride_h      (schema default = 0)
    #   field 3: filter_width  (schema default = 0)
    #   field 4: filter_height (schema default = 0)
    #   field 5: fused_activation_function
    builder.StartObject(6)
    builder.PrependInt8Slot(0, int(padding), 0)
    builder.PrependInt32Slot(1, stride_w, 0)
    builder.PrependInt32Slot(2, stride_h, 0)
    builder.PrependInt32Slot(3, filter_w, 0)
    builder.PrependInt32Slot(4, filter_h, 0)
    builder.PrependInt8Slot(5, int(activation), 0)
    return builder.EndObject()


def _build_reshape_options(builder, new_shape):
    """Build a ReshapeOptions table."""
    # ReshapeOptions: field 0: new_shape (vector of int32)
    builder.StartVector(4, len(new_shape), 4)
    for dim in reversed(new_shape):
        builder.PrependInt32(dim)
    shape_vec = builder.EndVector()
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, shape_vec, 0)
    return builder.EndObject()


def _build_operator(builder, opcode_idx, inputs, outputs,
                    builtin_options_type=0, builtin_options=None):
    """Build an Operator table.

    Args:
        opcode_idx: index into Model.operator_codes
        inputs: list of tensor indices
        outputs: list of tensor indices
        builtin_options_type: BuiltinOptions union discriminator (uint8)
        builtin_options: offset from an options builder, or None
    """
    # Pre-create vectors
    builder.StartVector(4, len(inputs), 4)
    for idx in reversed(inputs):
        builder.PrependInt32(idx)
    inputs_vec = builder.EndVector()

    builder.StartVector(4, len(outputs), 4)
    for idx in reversed(outputs):
        builder.PrependInt32(idx)
    outputs_vec = builder.EndVector()

    # Operator table:
    #   field 0: opcode_index (uint32)
    #   field 1: inputs (vector of int32)
    #   field 2: outputs (vector of int32)
    #   field 3: builtin_options_type (uint8, union discriminator)
    #   field 4: builtin_options (union value, uoffset)
    #   field 5: custom_options (vector of uint8)
    #   field 6: custom_options_format
    #   field 7: mutating_variable_inputs
    #   field 8: intermediates
    num_fields = 9
    builder.StartObject(num_fields)
    builder.PrependUint32Slot(0, opcode_idx, 0)
    builder.PrependUOffsetTRelativeSlot(1, inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, outputs_vec, 0)
    if builtin_options is not None:
        builder.PrependUint8Slot(3, builtin_options_type, 0)
        builder.PrependUOffsetTRelativeSlot(4, builtin_options, 0)
    return builder.EndObject()


def _build_subgraph(builder, tensors, inputs, outputs, operators, name="main"):
    """Build a SubGraph table.

    Args:
        tensors: list of offsets from _build_tensor
        inputs: list of tensor indices (graph inputs)
        outputs: list of tensor indices (graph outputs)
        operators: list of offsets from _build_operator
        name: subgraph name string
    """
    name_off = builder.CreateString(name)

    # Tensors vector (vector of tables = vector of offsets)
    builder.StartVector(4, len(tensors), 4)
    for t in reversed(tensors):
        builder.PrependUOffsetTRelative(t)
    tensors_vec = builder.EndVector()

    # Inputs vector
    builder.StartVector(4, len(inputs), 4)
    for i in reversed(inputs):
        builder.PrependInt32(i)
    inputs_vec = builder.EndVector()

    # Outputs vector
    builder.StartVector(4, len(outputs), 4)
    for o in reversed(outputs):
        builder.PrependInt32(o)
    outputs_vec = builder.EndVector()

    # Operators vector
    builder.StartVector(4, len(operators), 4)
    for op in reversed(operators):
        builder.PrependUOffsetTRelative(op)
    operators_vec = builder.EndVector()

    # SubGraph table:
    #   field 0: tensors
    #   field 1: inputs
    #   field 2: outputs
    #   field 3: operators
    #   field 4: name
    builder.StartObject(5)
    builder.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
    builder.PrependUOffsetTRelativeSlot(1, inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, outputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, operators_vec, 0)
    builder.PrependUOffsetTRelativeSlot(4, name_off, 0)
    return builder.EndObject()


def _build_model(builder, operator_codes, subgraphs, buffers,
                 description="", version=3):
    """Build the root Model table and finalize the builder.

    Returns the raw bytes of the completed FlatBuffer with ``TFL3`` identifier.
    """
    desc_off = builder.CreateString(description)

    # operator_codes vector
    builder.StartVector(4, len(operator_codes), 4)
    for oc in reversed(operator_codes):
        builder.PrependUOffsetTRelative(oc)
    opcodes_vec = builder.EndVector()

    # subgraphs vector
    builder.StartVector(4, len(subgraphs), 4)
    for sg in reversed(subgraphs):
        builder.PrependUOffsetTRelative(sg)
    subgraphs_vec = builder.EndVector()

    # buffers vector
    builder.StartVector(4, len(buffers), 4)
    for b in reversed(buffers):
        builder.PrependUOffsetTRelative(b)
    buffers_vec = builder.EndVector()

    # Model table:
    #   field 0: version (int32)
    #   field 1: operator_codes
    #   field 2: subgraphs
    #   field 3: description (string)
    #   field 4: buffers
    builder.StartObject(5)
    builder.PrependInt32Slot(0, version, 0)
    builder.PrependUOffsetTRelativeSlot(1, opcodes_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, desc_off, 0)
    builder.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)
    model_off = builder.EndObject()

    builder.Finish(model_off, b"TFL3")
    return bytes(builder.Output())


# ── Dense model builder ─────────────────────────────────────────────────────

def build_dense(n, weights_int8=None):
    """Build a quantized Dense(N, N) TFLite model for MatMulEngine.

    Produces the same operator chain as TF's quantization pipeline:
    ``QUANTIZE(uint8→int8) → FULLY_CONNECTED → QUANTIZE(int8→uint8)``

    Args:
        n: Matrix dimension (square NxN).
        weights_int8: Optional [N, N] int8 numpy array.  If None, zero weights
            are used (the weights will be swapped at runtime anyway).

    Returns:
        (tflite_bytes, metadata) where metadata is a dict matching the sidecar
        JSON schema used by existing templates.
    """
    # ── Quantization parameters ──────────────────────────────────────────
    # Input: uint8 with scale = 2/255, zp = 127
    # This maps float [-1.0, +1.0] to uint8 [0, 254].
    input_scale = 2.0 / 255.0  # ≈ 0.007843
    input_zp = 127

    # Internal int8 after QUANTIZE: same scale, zp = input_zp - 128 = -1
    internal_scale = input_scale
    internal_zp = input_zp - 128  # -1

    # Weights: int8, symmetric (zp=0)
    if weights_int8 is not None:
        weights_int8 = np.asarray(weights_int8, dtype=np.int8)
        if weights_int8.shape != (n, n):
            raise ValueError(f"weights_int8 must be ({n}, {n}), got {weights_int8.shape}")
        max_abs = max(float(np.max(np.abs(weights_int8))), 1)
        weight_scale = max_abs / 127.0
    else:
        weights_int8 = np.zeros((n, n), dtype=np.int8)
        weight_scale = 1.0 / 127.0

    # FC output: int8
    # Worst-case accumulation: N elements each contributing input_scale * weight_scale
    # The output scale must be large enough to represent the worst-case sum.
    fc_output_scale = input_scale * weight_scale * n
    # Zero point for int8 output, centered: we use 1 to match TF's convention
    fc_output_zp = 1

    # Final output: uint8, same scale as FC output, zp = fc_output_zp + 128
    output_scale = fc_output_scale
    output_zp = fc_output_zp + 128

    # ── Weight buffer ────────────────────────────────────────────────────
    weight_bytes = weights_int8.tobytes()

    # ── Build FlatBuffer (bottom-up) ─────────────────────────────────────
    builder = flatbuffers.Builder(1024 + len(weight_bytes))

    # Buffers: TFLite models always start with an empty buffer at index 0
    # Buf 0: empty (sentinel)
    # Buf 1: empty (input tensor placeholder)
    # Buf 2: weight data
    # Buf 3: empty (quantize output placeholder)
    # Buf 4: empty (FC output placeholder)
    # Buf 5: empty (output tensor placeholder)
    buf0 = _build_buffer(builder, None)
    buf1 = _build_buffer(builder, None)
    buf2 = _build_buffer(builder, weight_bytes)
    buf3 = _build_buffer(builder, None)
    buf4 = _build_buffer(builder, None)
    buf5 = _build_buffer(builder, None)

    # Quantization parameter tables
    q_input = _build_quantization(builder, [input_scale], [input_zp])
    q_weights = _build_quantization(builder, [weight_scale], [0])
    q_internal = _build_quantization(builder, [internal_scale], [internal_zp])
    q_fc_output = _build_quantization(builder, [fc_output_scale], [fc_output_zp])
    q_output = _build_quantization(builder, [output_scale], [output_zp])

    # Tensors
    # T0: input uint8 [1, N]
    t0 = _build_tensor(builder, "input", [1, n], TensorType.UINT8, 1, q_input)
    # T1: weights int8 [N, N]
    t1 = _build_tensor(builder, "weights", [n, n], TensorType.INT8, 2, q_weights)
    # T2: quantize output int8 [1, N]
    t2 = _build_tensor(builder, "quantize_out", [1, n], TensorType.INT8, 3, q_internal)
    # T3: FC output int8 [1, N]
    t3 = _build_tensor(builder, "fc_out", [1, n], TensorType.INT8, 4, q_fc_output)
    # T4: output uint8 [1, N]
    t4 = _build_tensor(builder, "output", [1, n], TensorType.UINT8, 5, q_output)

    # FullyConnectedOptions (all defaults)
    fc_opts = _build_fc_options(builder)

    # Operators
    # Op0: QUANTIZE  inputs=[0] outputs=[2]  (uint8 → int8)
    op0 = _build_operator(builder, opcode_idx=0, inputs=[0], outputs=[2])
    # Op1: FC        inputs=[2, 1, -1] outputs=[3]  (int8 matmul, -1 = no bias)
    op1 = _build_operator(builder, opcode_idx=1, inputs=[2, 1, -1], outputs=[3],
                          builtin_options_type=int(BuiltinOptions.FullyConnectedOptions),
                          builtin_options=fc_opts)
    # Op2: QUANTIZE  inputs=[3] outputs=[4]  (int8 → uint8)
    op2 = _build_operator(builder, opcode_idx=0, inputs=[3], outputs=[4])

    # Operator codes
    # Index 0: QUANTIZE (114)
    # Index 1: FULLY_CONNECTED (9)
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.FULLY_CONNECTED)

    # Subgraph
    sg = _build_subgraph(builder,
                         tensors=[t0, t1, t2, t3, t4],
                         inputs=[0],
                         outputs=[4],
                         operators=[op0, op1, op2],
                         name="main")

    # Finalize
    tflite_bytes = _build_model(
        builder,
        operator_codes=[oc0, oc1],
        subgraphs=[sg],
        buffers=[buf0, buf1, buf2, buf3, buf4, buf5],
        description=f"Dense({n},{n}) for MatMulEngine (libredgetpu)",
        version=3,
    )

    # ── Metadata dict (matches sidecar JSON schema) ──────────────────────
    metadata = {
        "matrix_size": n,
        "weight_scale": float(weight_scale),
        "weight_zero_point": 0,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zp),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zp),
        "use_bias": False,
        "param_size": n * n,  # uncompiled weight size; updated after compilation
    }

    return tflite_bytes, metadata


# ── Coordinate grid helper ───────────────────────────────────────────────────

def _create_coordinate_grids(height, width, temperature=0.1):
    """Create normalized coordinate grids with temperature scaling baked in.

    Args:
        height: Grid height.
        width: Grid width.
        temperature: Softmax temperature (lower = sharper peak).

    Returns:
        (x_coords, y_coords) as numpy arrays of shape (height, width).
        Coordinates are normalized to [-1/temperature, +1/temperature] range.
    """
    x_coords = np.zeros((height, width), dtype=np.float32)
    for j in range(width):
        x_coords[:, j] = (j - (width - 1) / 2) / max((width - 1) / 2, 1) / temperature

    y_coords = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        y_coords[i, :] = (i - (height - 1) / 2) / max((height - 1) / 2, 1) / temperature

    return x_coords, y_coords


# ── Spot tracker model builder ───────────────────────────────────────────────

def build_spot_tracker(height, width, variant="bright", temperature=0.1,
                       color_weights=None):
    """Build a quantized spot tracker TFLite model for SpotTracker.

    Produces the same operator chain as TF's quantization pipeline:

    - **bright**: ``QUANTIZE → RESHAPE → SOFTMAX → FC(x) → FC(y) → CONCAT``
    - **color_***: ``QUANTIZE → CONV_2D(1x1, fused ReLU) → RESHAPE → SOFTMAX → FC(x) → FC(y) → CONCAT``

    Args:
        height: Input image height.
        width: Input image width.
        variant: "bright" for grayscale, or "color_{name}" for color tracking.
        temperature: Softmax temperature (default 0.1 for sharp peaks).
        color_weights: [R, G, B] float coefficients for color variant.
            Required when variant starts with "color_".

    Returns:
        (tflite_bytes, metadata) where metadata is a dict matching the
        sidecar JSON schema used by existing SpotTracker templates.
    """
    is_color = variant.startswith("color_")
    channels = 3 if is_color else 1

    if is_color and color_weights is None:
        raise ValueError("color_weights required for color variant")

    n_pixels = height * width

    # ── Coordinate grids ────────────────────────────────────────────────
    x_coords, y_coords = _create_coordinate_grids(height, width, temperature)
    y_coords_offset = y_coords + (1.0 / temperature)

    # FC weight shape: [output_dim, input_dim] = [1, N] in TFLite convention
    x_weights = x_coords.reshape(1, n_pixels).astype(np.float32)
    y_weights = y_coords_offset.reshape(1, n_pixels).astype(np.float32)

    # ── Quantization parameters (analytical, matching TF output) ────────
    # Input: uint8 identity mapping (scale=1.0, zp=0)
    input_scale = 1.0
    input_zp = 0

    # After QUANTIZE: int8 (scale=1.0, zp=-128)
    q_int8_scale = 1.0
    q_int8_zp = -128

    # Softmax output: always scale=1/256, zp=-128 (TFLite convention)
    softmax_scale = 1.0 / 256.0
    softmax_zp = -128

    # X weights: range [-1/T, +1/T], symmetric int8
    x_abs_max = float(np.max(np.abs(x_weights)))
    x_weight_scale = max(x_abs_max, 1e-6) / 127.0
    x_weights_int8 = np.clip(np.round(x_weights / x_weight_scale), -127, 127).astype(np.int8)

    # Y weights: range [0, +2/T], symmetric int8
    y_abs_max = float(np.max(np.abs(y_weights)))
    y_weight_scale = max(y_abs_max, 1e-6) / 127.0
    y_weights_int8 = np.clip(np.round(y_weights / y_weight_scale), -127, 127).astype(np.int8)

    # FC output scale: softmax_scale * weight_scale * n_pixels
    # But for softmax outputs, max accumulation is bounded by sum(|w|) since
    # softmax sums to 1. The actual max output is max(|w|) (weighted average).
    # We use a conservative bound: scale = (x_range + y_range) / 255
    # to cover the concat output range.
    x_fc_scale = softmax_scale * x_weight_scale * n_pixels
    y_fc_scale = softmax_scale * y_weight_scale * n_pixels

    # Concat output: must cover both x and y ranges in one scale/zp
    # x range: [-1/T, +1/T], y range: [0, +2/T]
    # Combined range: [-1/T, +2/T], span = 3/T
    concat_range_min = -1.0 / temperature
    concat_range_max = 2.0 / temperature
    concat_span = concat_range_max - concat_range_min
    output_scale = concat_span / 255.0
    # zp = -128 - round(min / scale)
    output_zp = int(round(-128 - concat_range_min / output_scale))
    # Clamp to int8 range
    output_zp = max(-128, min(127, output_zp))

    # FC x output: same scale as concat, different zp for x's symmetric range
    # We use the concat scale for both FCs so concat doesn't need requantization
    x_fc_output_scale = output_scale
    y_fc_output_scale = output_scale
    # x: range [-1/T, +1/T] mapped to int8
    x_fc_zp = int(round(-concat_range_min / output_scale - 128))
    x_fc_zp = max(-128, min(127, x_fc_zp))
    # y: range [0, +2/T] mapped to int8
    y_fc_zp = output_zp  # same as output since y has same min as concat
    # Actually both FCs should use the output quant so concat is a no-op
    x_fc_zp = output_zp
    y_fc_zp = output_zp

    # Color filter: Conv2D 1x1 weights
    if is_color:
        color_arr = np.array(color_weights, dtype=np.float32).reshape(1, 1, 3, 1)
        color_abs_max = max(float(np.max(np.abs(color_arr))), 1e-6)
        color_weight_scale = color_abs_max / 127.0
        color_weights_int8 = np.clip(
            np.round(color_arr / color_weight_scale), -127, 127
        ).astype(np.int8)
        # Conv2D output with ReLU: TFLite conv computes sum((input - input_zp) * weight).
        # With input_zp=-128, effective input = [0, 255] for uint8 data.
        # Use actual weight sum for a tighter bound.
        max_effective_input = (127 - q_int8_zp) * q_int8_scale  # 255 for uint8
        color_abs_sum = float(np.sum(np.abs(color_weights_int8))) * color_weight_scale
        conv_output_scale = max_effective_input * color_abs_sum / 127.0
        conv_output_zp = -128  # ReLU clamps to [0, +inf], so zp=-128

    # ── Build FlatBuffer ────────────────────────────────────────────────
    data_size = n_pixels * 2 + 64  # x + y weights + overhead
    if is_color:
        data_size += 16
    builder = flatbuffers.Builder(4096 + data_size)

    # Buffers: index 0 is always empty sentinel
    buf_list = [_build_buffer(builder, None)]  # buf 0: sentinel

    # Tensor/buffer layout depends on variant
    if is_color:
        # Color variant: 13 tensors, 16 buffers (matching TF output)
        # T0: input [1,H,W,3] uint8 -> buf 1
        buf_list.append(_build_buffer(builder, None))  # buf 1: input
        # T1: quantize out [1,H,W,3] int8 -> buf 2
        buf_list.append(_build_buffer(builder, None))  # buf 2: q_out
        # T2: x_weights [1, n_pixels] int8 -> buf 3
        buf_list.append(_build_buffer(builder, x_weights_int8.tobytes()))  # buf 3
        # T3: y_weights [1, n_pixels] int8 -> buf 4
        buf_list.append(_build_buffer(builder, y_weights_int8.tobytes()))  # buf 4
        # T4: conv_bias [1] int32 -> buf 5 (zero)
        conv_bias = np.zeros(1, dtype=np.int32)
        buf_list.append(_build_buffer(builder, conv_bias.tobytes()))  # buf 5
        # T5: conv_weights [1,1,3,1] int8 -> buf 6
        buf_list.append(_build_buffer(builder, color_weights_int8.tobytes()))  # buf 6
        # T6: conv_out [1,H,W,1] int8 -> buf 7
        buf_list.append(_build_buffer(builder, None))  # buf 7
        # T7: shape_const [2] int32 -> buf 8
        shape_data = np.array([1, n_pixels], dtype=np.int32)
        buf_list.append(_build_buffer(builder, shape_data.tobytes()))  # buf 8
        # T8: reshape_out [1, n_pixels] int8 -> buf 9
        buf_list.append(_build_buffer(builder, None))  # buf 9
        # T9: softmax_out [1, n_pixels] int8 -> buf 10
        buf_list.append(_build_buffer(builder, None))  # buf 10
        # T10: x_fc_out [1, 1] int8 -> buf 11
        buf_list.append(_build_buffer(builder, None))  # buf 11
        # T11: y_fc_out [1, 1] int8 -> buf 12
        buf_list.append(_build_buffer(builder, None))  # buf 12
        # T12: concat_out [1, 2] int8 -> buf 13 (output)
        buf_list.append(_build_buffer(builder, None))  # buf 13
        # Extra empty buffers to match TF's 16 buffers
        buf_list.append(_build_buffer(builder, None))  # buf 14
        buf_list.append(_build_buffer(builder, None))  # buf 15

        # Quantization params
        q_input = _build_quantization(builder, [input_scale], [input_zp])
        q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
        q_x_w = _build_quantization(builder, [x_weight_scale], [0])
        q_y_w = _build_quantization(builder, [y_weight_scale], [0])
        conv_bias_scale = q_int8_scale * color_weight_scale
        q_conv_bias = _build_quantization(builder, [conv_bias_scale], [0])
        q_conv_w = _build_quantization(builder, [color_weight_scale], [0])
        q_conv_out = _build_quantization(builder, [conv_output_scale], [conv_output_zp])
        q_softmax = _build_quantization(builder, [softmax_scale], [softmax_zp])
        q_fc_out = _build_quantization(builder, [output_scale], [output_zp])
        q_output = _build_quantization(builder, [output_scale], [output_zp])

        # Tensors
        t0 = _build_tensor(builder, "input", [1, height, width, channels], TensorType.UINT8, 1, q_input)
        t1 = _build_tensor(builder, "quantize_out", [1, height, width, channels], TensorType.INT8, 2, q_int8)
        t2 = _build_tensor(builder, "x_weights", [1, n_pixels], TensorType.INT8, 3, q_x_w)
        t3 = _build_tensor(builder, "y_weights", [1, n_pixels], TensorType.INT8, 4, q_y_w)
        t4 = _build_tensor(builder, "conv_bias", [1], TensorType.INT32, 5, q_conv_bias)
        t5 = _build_tensor(builder, "conv_weights", [1, 1, 3, 1], TensorType.INT8, 6, q_conv_w)
        t6 = _build_tensor(builder, "conv_out", [1, height, width, 1], TensorType.INT8, 7, q_conv_out)
        t7 = _build_tensor(builder, "shape_const", [2], TensorType.INT32, 8)
        t8 = _build_tensor(builder, "reshape_out", [1, n_pixels], TensorType.INT8, 9, q_conv_out)
        t9 = _build_tensor(builder, "softmax_out", [1, n_pixels], TensorType.INT8, 10, q_softmax)
        t10 = _build_tensor(builder, "x_fc_out", [1, 1], TensorType.INT8, 11, q_fc_out)
        t11 = _build_tensor(builder, "y_fc_out", [1, 1], TensorType.INT8, 12, q_fc_out)
        t12 = _build_tensor(builder, "output", [1, 2], TensorType.INT8, 13, q_output)
        tensors = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]

        # Options
        conv_opts = _build_conv2d_options(builder, padding=Padding.SAME,
                                          activation=Activation.RELU)
        reshape_opts = _build_reshape_options(builder, [1, n_pixels])
        softmax_opts = _build_softmax_options(builder, beta=1.0)
        fc_opts = _build_fc_options(builder)
        fc_opts2 = _build_fc_options(builder)
        concat_opts = _build_concatenation_options(builder, axis=1)

        # Operators: QUANTIZE -> CONV_2D -> RESHAPE -> SOFTMAX -> FC(x) -> FC(y) -> CONCAT
        # opcode indices: 0=QUANTIZE, 1=CONV_2D, 2=RESHAPE, 3=SOFTMAX, 4=FC, 5=CONCATENATION
        op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE
        op1 = _build_operator(builder, 1, [1, 5, 4], [6],  # CONV_2D
                              int(BuiltinOptions.Conv2DOptions), conv_opts)
        op2 = _build_operator(builder, 2, [6, 7], [8],  # RESHAPE
                              int(BuiltinOptions.ReshapeOptions), reshape_opts)
        op3 = _build_operator(builder, 3, [8], [9],  # SOFTMAX
                              int(BuiltinOptions.SoftmaxOptions), softmax_opts)
        op4 = _build_operator(builder, 4, [9, 2, -1], [10],  # FC(x)
                              int(BuiltinOptions.FullyConnectedOptions), fc_opts)
        op5 = _build_operator(builder, 4, [9, 3, -1], [11],  # FC(y)
                              int(BuiltinOptions.FullyConnectedOptions), fc_opts2)
        op6 = _build_operator(builder, 5, [10, 11], [12],  # CONCAT
                              int(BuiltinOptions.ConcatenationOptions), concat_opts)
        operators = [op0, op1, op2, op3, op4, op5, op6]

        # Operator codes
        oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
        oc1 = _build_operator_code(builder, BuiltinOp.CONV_2D)
        oc2 = _build_operator_code(builder, BuiltinOp.RESHAPE)
        oc3 = _build_operator_code(builder, BuiltinOp.SOFTMAX)
        oc4 = _build_operator_code(builder, BuiltinOp.FULLY_CONNECTED)
        oc5 = _build_operator_code(builder, BuiltinOp.CONCATENATION)
        opcodes = [oc0, oc1, oc2, oc3, oc4, oc5]

        output_tensor_idx = 12
    else:
        # Bright variant: 10 tensors, 13 buffers (matching TF output)
        # T0: input [1,H,W,1] uint8 -> buf 1
        buf_list.append(_build_buffer(builder, None))  # buf 1: input
        # T1: quantize out [1,H,W,1] int8 -> buf 2
        buf_list.append(_build_buffer(builder, None))  # buf 2: q_out
        # T2: x_weights [1, n_pixels] int8 -> buf 3
        buf_list.append(_build_buffer(builder, x_weights_int8.tobytes()))  # buf 3
        # T3: y_weights [1, n_pixels] int8 -> buf 4
        buf_list.append(_build_buffer(builder, y_weights_int8.tobytes()))  # buf 4
        # T4: shape_const [2] int32 -> buf 5
        shape_data = np.array([1, n_pixels], dtype=np.int32)
        buf_list.append(_build_buffer(builder, shape_data.tobytes()))  # buf 5
        # T5: reshape_out [1, n_pixels] int8 -> buf 6
        buf_list.append(_build_buffer(builder, None))  # buf 6
        # T6: softmax_out [1, n_pixels] int8 -> buf 7
        buf_list.append(_build_buffer(builder, None))  # buf 7
        # T7: x_fc_out [1, 1] int8 -> buf 8
        buf_list.append(_build_buffer(builder, None))  # buf 8
        # T8: y_fc_out [1, 1] int8 -> buf 9
        buf_list.append(_build_buffer(builder, None))  # buf 9
        # T9: concat_out [1, 2] int8 -> buf 10 (output)
        buf_list.append(_build_buffer(builder, None))  # buf 10
        # Extra empty buffers to match TF's 13 buffers
        buf_list.append(_build_buffer(builder, None))  # buf 11
        buf_list.append(_build_buffer(builder, None))  # buf 12

        # Quantization params
        q_input = _build_quantization(builder, [input_scale], [input_zp])
        q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
        q_x_w = _build_quantization(builder, [x_weight_scale], [0])
        q_y_w = _build_quantization(builder, [y_weight_scale], [0])
        q_softmax = _build_quantization(builder, [softmax_scale], [softmax_zp])
        q_fc_out = _build_quantization(builder, [output_scale], [output_zp])
        q_output = _build_quantization(builder, [output_scale], [output_zp])

        # Tensors
        t0 = _build_tensor(builder, "input", [1, height, width, 1], TensorType.UINT8, 1, q_input)
        t1 = _build_tensor(builder, "quantize_out", [1, height, width, 1], TensorType.INT8, 2, q_int8)
        t2 = _build_tensor(builder, "x_weights", [1, n_pixels], TensorType.INT8, 3, q_x_w)
        t3 = _build_tensor(builder, "y_weights", [1, n_pixels], TensorType.INT8, 4, q_y_w)
        t4 = _build_tensor(builder, "shape_const", [2], TensorType.INT32, 5)
        t5 = _build_tensor(builder, "reshape_out", [1, n_pixels], TensorType.INT8, 6, q_int8)
        t6 = _build_tensor(builder, "softmax_out", [1, n_pixels], TensorType.INT8, 7, q_softmax)
        t7 = _build_tensor(builder, "x_fc_out", [1, 1], TensorType.INT8, 8, q_fc_out)
        t8 = _build_tensor(builder, "y_fc_out", [1, 1], TensorType.INT8, 9, q_fc_out)
        t9 = _build_tensor(builder, "output", [1, 2], TensorType.INT8, 10, q_output)
        tensors = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]

        # Options
        reshape_opts = _build_reshape_options(builder, [1, n_pixels])
        softmax_opts = _build_softmax_options(builder, beta=1.0)
        fc_opts = _build_fc_options(builder)
        fc_opts2 = _build_fc_options(builder)
        concat_opts = _build_concatenation_options(builder, axis=1)

        # Operators: QUANTIZE -> RESHAPE -> SOFTMAX -> FC(x) -> FC(y) -> CONCAT
        # opcode indices: 0=QUANTIZE, 1=RESHAPE, 2=SOFTMAX, 3=FC, 4=CONCATENATION
        op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE
        op1 = _build_operator(builder, 1, [1, 4], [5],  # RESHAPE
                              int(BuiltinOptions.ReshapeOptions), reshape_opts)
        op2 = _build_operator(builder, 2, [5], [6],  # SOFTMAX
                              int(BuiltinOptions.SoftmaxOptions), softmax_opts)
        op3 = _build_operator(builder, 3, [6, 2, -1], [7],  # FC(x)
                              int(BuiltinOptions.FullyConnectedOptions), fc_opts)
        op4 = _build_operator(builder, 3, [6, 3, -1], [8],  # FC(y)
                              int(BuiltinOptions.FullyConnectedOptions), fc_opts2)
        op5 = _build_operator(builder, 4, [7, 8], [9],  # CONCAT
                              int(BuiltinOptions.ConcatenationOptions), concat_opts)
        operators = [op0, op1, op2, op3, op4, op5]

        # Operator codes
        oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
        oc1 = _build_operator_code(builder, BuiltinOp.RESHAPE)
        oc2 = _build_operator_code(builder, BuiltinOp.SOFTMAX)
        oc3 = _build_operator_code(builder, BuiltinOp.FULLY_CONNECTED)
        oc4 = _build_operator_code(builder, BuiltinOp.CONCATENATION)
        opcodes = [oc0, oc1, oc2, oc3, oc4]

        output_tensor_idx = 9

    # Subgraph
    sg = _build_subgraph(builder,
                         tensors=tensors,
                         inputs=[0],
                         outputs=[output_tensor_idx],
                         operators=operators,
                         name="main")

    # Finalize
    desc = f"SpotTracker {variant} {height}x{width} (libredgetpu)"
    tflite_bytes = _build_model(
        builder,
        operator_codes=opcodes,
        subgraphs=[sg],
        buffers=buf_list,
        description=desc,
        version=3,
    )

    # ── Metadata ────────────────────────────────────────────────────────
    metadata = {
        "height": height,
        "width": width,
        "channels": channels,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zp),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zp),
        "output_count": 2,
        "variant": variant,
        "temperature": float(temperature),
        "y_offset": 1.0 / temperature,
    }
    if is_color:
        metadata["color_weight_scale"] = float(color_weight_scale)

    return tflite_bytes, metadata


# ── Looming detector model builder ──────────────────────────────────────────

def build_looming(height, width, zones=3):
    """Build a quantized looming detection TFLite model for LoomingDetector.

    Produces the same operator chain as TF's quantization pipeline:
    ``QUANTIZE → CONV_2D(sobel_x) + CONV_2D(sobel_y) → MUL(gx²) + MUL(gy²)
    → ADD → AVG_POOL_2D → RESHAPE → QUANTIZE(int8→uint8)``

    Args:
        height: Input image height (must be >= zones).
        width: Input image width (must be >= zones).
        zones: Number of zones per dimension (default 3 for 3x3=9 zones).

    Returns:
        (tflite_bytes, metadata) where metadata is a dict matching the
        sidecar JSON schema used by existing LoomingDetector templates.
    """
    if zones != 3:
        raise ValueError(f"Only zones=3 is currently supported, got {zones}")
    if height < zones or width < zones:
        raise ValueError(
            f"Image dimensions ({height}x{width}) must be >= zones ({zones})"
        )

    pool_h = height // zones
    pool_w = width // zones

    # ── Sobel kernels (scaled by 1/8 to prevent overflow) ───────────────
    # These are the same kernels used by the TF generator
    sobel_x_float = np.array([
        [-1/8, 0, 1/8],
        [-1/4, 0, 1/4],
        [-1/8, 0, 1/8],
    ], dtype=np.float32)

    sobel_y_float = np.array([
        [-1/8, -1/4, -1/8],
        [0, 0, 0],
        [1/8, 1/4, 1/8],
    ], dtype=np.float32)

    # ── Quantization parameters (analytical) ────────────────────────────
    # Input: uint8 identity (scale=1.0, zp=0)
    input_scale = 1.0
    input_zp = 0

    # After QUANTIZE: int8 (scale=1.0, zp=-128)
    q_int8_scale = 1.0
    q_int8_zp = -128

    # Sobel weights: int8 symmetric, scale = max(|kernel|) / 127
    sobel_abs_max = float(np.max(np.abs(sobel_x_float)))  # 0.25
    sobel_weight_scale = sobel_abs_max / 127.0  # ≈ 0.001969
    sobel_x_int8 = np.clip(
        np.round(sobel_x_float / sobel_weight_scale), -127, 127
    ).astype(np.int8).reshape(3, 3, 1, 1)
    sobel_y_int8 = np.clip(
        np.round(sobel_y_float / sobel_weight_scale), -127, 127
    ).astype(np.int8).reshape(3, 3, 1, 1)

    # Sobel bias: zero (INT32)
    sobel_bias = np.zeros(1, dtype=np.int32)
    sobel_bias_scale = q_int8_scale * sobel_weight_scale

    # Sobel output: scale = input_scale * weight_scale * (sum of kernel)
    # Actual: each output pixel is dot product of 3x3 patch with kernel
    # Max absolute output: 255 * (1/8 + 1/4 + 1/8 + 0 + 0 + 0 + 1/8 + 1/4 + 1/8) = 255
    # With int8 input range [-128, 127]: max abs ≈ 127 * 1.0 = 127
    # Sobel output scale matches TF: ≈ q_int8_scale * sobel_weight_scale * 4 (sum of |kernel|)
    # From TF output: sobel_output scale ≈ 0.5, but we can compute it analytically
    sobel_output_scale = q_int8_scale * sobel_weight_scale * 9  # conservative
    # TF uses: scale ≈ 0.5 (half the input range), zp=0
    # We match TF's convention: symmetric output centered at 0
    sobel_output_scale = 0.5
    sobel_output_zp = 0

    # MUL: gx² or gy². Output is non-negative.
    # Max: (127 * 0.5)² = 4032.25
    # With int8 [−128, 127]: map [0, 4032] to [−128, 127]
    mul_output_max = (127 * sobel_output_scale) ** 2
    mul_output_scale = mul_output_max / 255.0  # ≈ 15.87
    mul_output_zp = -128

    # ADD: gx² + gy². Max: 2 * mul_output_max
    add_output_max = 2 * mul_output_max
    add_output_scale = add_output_max / 255.0  # ≈ 31.75
    add_output_zp = -128

    # Pool and reshape: same as ADD output
    pool_output_scale = add_output_scale
    pool_output_zp = add_output_zp

    # Final uint8 output: same range as pool but mapped to uint8
    final_output_scale = pool_output_scale
    final_output_zp = 0  # non-negative values, so zp=0 maps 0→0

    # ── Build FlatBuffer ────────────────────────────────────────────────
    builder = flatbuffers.Builder(4096)

    # Buffers: 18 buffers (matching TF output)
    buf_list = [_build_buffer(builder, None)]  # buf 0: sentinel

    # T0: input [1,H,W,1] uint8 -> buf 1
    buf_list.append(_build_buffer(builder, None))  # buf 1
    # T1: quantize out [1,H,W,1] int8 -> buf 2
    buf_list.append(_build_buffer(builder, None))  # buf 2
    # T2: sobel_y_weights [3,3,1,1] int8 -> buf 3
    buf_list.append(_build_buffer(builder, sobel_y_int8.tobytes()))  # buf 3
    # T3: sobel_y_bias / sobel_x_bias [1] int32 -> buf 4 (shared)
    buf_list.append(_build_buffer(builder, sobel_bias.tobytes()))  # buf 4
    # T4: sobel_y_out [1,H,W,1] int8 -> buf 5
    buf_list.append(_build_buffer(builder, None))  # buf 5
    # T5: sobel_x_weights [3,3,1,1] int8 -> buf 6
    buf_list.append(_build_buffer(builder, sobel_x_int8.tobytes()))  # buf 6
    # T6: sobel_x_bias [1] int32 -> buf 4 (shared — same zero bias)
    # Note: TF shares the bias buffer. We'll use the same buf index.
    # T7: sobel_x_out [1,H,W,1] int8 -> buf 7
    buf_list.append(_build_buffer(builder, None))  # buf 7
    # T8: gx_squared [1,H,W,1] int8 -> buf 8
    buf_list.append(_build_buffer(builder, None))  # buf 8
    # T9: gy_squared [1,H,W,1] int8 -> buf 9
    buf_list.append(_build_buffer(builder, None))  # buf 9
    # T10: edge_mag_sq [1,H,W,1] int8 -> buf 10
    buf_list.append(_build_buffer(builder, None))  # buf 10
    # T11: pooled [1,3,3,1] int8 -> buf 11
    buf_list.append(_build_buffer(builder, None))  # buf 11
    # T12: shape_const [2] int32 -> buf 12
    shape_data = np.array([1, zones * zones], dtype=np.int32)
    buf_list.append(_build_buffer(builder, shape_data.tobytes()))  # buf 12
    # T13: reshape_out [1, 9] int8 -> buf 13
    buf_list.append(_build_buffer(builder, None))  # buf 13
    # T14: output [1, 9] uint8 -> buf 14
    buf_list.append(_build_buffer(builder, None))  # buf 14
    # Extra buffers to reach 18
    buf_list.append(_build_buffer(builder, None))  # buf 15
    buf_list.append(_build_buffer(builder, None))  # buf 16
    buf_list.append(_build_buffer(builder, None))  # buf 17

    # Quantization parameters
    q_input = _build_quantization(builder, [input_scale], [input_zp])
    q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
    q_sobel_w = _build_quantization(builder, [sobel_weight_scale], [0])
    q_sobel_bias = _build_quantization(builder, [sobel_bias_scale], [0])
    q_sobel_out = _build_quantization(builder, [sobel_output_scale], [sobel_output_zp])
    q_mul_out = _build_quantization(builder, [mul_output_scale], [mul_output_zp])
    q_add_out = _build_quantization(builder, [add_output_scale], [add_output_zp])
    q_pool_out = _build_quantization(builder, [pool_output_scale], [pool_output_zp])
    q_final = _build_quantization(builder, [final_output_scale], [final_output_zp])

    # Tensors (15 total, matching TF)
    t0 = _build_tensor(builder, "input", [1, height, width, 1], TensorType.UINT8, 1, q_input)
    t1 = _build_tensor(builder, "quantize_out", [1, height, width, 1], TensorType.INT8, 2, q_int8)
    t2 = _build_tensor(builder, "sobel_y_weights", [1, 3, 3, 1], TensorType.INT8, 3, q_sobel_w)
    t3 = _build_tensor(builder, "sobel_y_bias", [1], TensorType.INT32, 4, q_sobel_bias)
    t4 = _build_tensor(builder, "sobel_y_out", [1, height, width, 1], TensorType.INT8, 5, q_sobel_out)
    t5 = _build_tensor(builder, "sobel_x_weights", [1, 3, 3, 1], TensorType.INT8, 6, q_sobel_w)
    t6 = _build_tensor(builder, "sobel_x_bias", [1], TensorType.INT32, 4, q_sobel_bias)  # shared buf 4
    t7 = _build_tensor(builder, "sobel_x_out", [1, height, width, 1], TensorType.INT8, 7, q_sobel_out)
    t8 = _build_tensor(builder, "gx_squared", [1, height, width, 1], TensorType.INT8, 8, q_mul_out)
    t9 = _build_tensor(builder, "gy_squared", [1, height, width, 1], TensorType.INT8, 9, q_mul_out)
    t10 = _build_tensor(builder, "edge_mag_sq", [1, height, width, 1], TensorType.INT8, 10, q_add_out)
    t11 = _build_tensor(builder, "pooled", [1, zones, zones, 1], TensorType.INT8, 11, q_pool_out)
    t12 = _build_tensor(builder, "shape_const", [2], TensorType.INT32, 12)
    t13 = _build_tensor(builder, "reshape_out", [1, zones * zones], TensorType.INT8, 13, q_pool_out)
    t14 = _build_tensor(builder, "output", [1, zones * zones], TensorType.UINT8, 14, q_final)

    tensors = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14]

    # Options
    conv_same_opts = _build_conv2d_options(builder, padding=Padding.SAME)
    conv_same_opts2 = _build_conv2d_options(builder, padding=Padding.SAME)
    mul_opts = _build_mul_options(builder)
    mul_opts2 = _build_mul_options(builder)
    add_opts = _build_add_options(builder)
    pool_opts = _build_pool2d_options(builder, padding=Padding.VALID,
                                      stride_w=pool_w, stride_h=pool_h,
                                      filter_w=pool_w, filter_h=pool_h)
    reshape_opts = _build_reshape_options(builder, [1, zones * zones])

    # Operators (9 total)
    # opcode indices: 0=QUANTIZE, 1=CONV_2D, 2=MUL, 3=ADD, 4=AVERAGE_POOL_2D, 5=RESHAPE
    op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE
    op1 = _build_operator(builder, 1, [1, 2, 3], [4],  # CONV_2D sobel_y
                          int(BuiltinOptions.Conv2DOptions), conv_same_opts)
    op2 = _build_operator(builder, 1, [1, 5, 6], [7],  # CONV_2D sobel_x
                          int(BuiltinOptions.Conv2DOptions), conv_same_opts2)
    op3 = _build_operator(builder, 2, [7, 7], [8],  # MUL gx²
                          int(BuiltinOptions.MulOptions), mul_opts)
    op4 = _build_operator(builder, 2, [4, 4], [9],  # MUL gy²
                          int(BuiltinOptions.MulOptions), mul_opts2)
    op5 = _build_operator(builder, 3, [8, 9], [10],  # ADD
                          int(BuiltinOptions.AddOptions), add_opts)
    op6 = _build_operator(builder, 4, [10], [11],  # AVG_POOL_2D
                          int(BuiltinOptions.Pool2DOptions), pool_opts)
    op7 = _build_operator(builder, 5, [11, 12], [13],  # RESHAPE
                          int(BuiltinOptions.ReshapeOptions), reshape_opts)
    op8 = _build_operator(builder, 0, [13], [14])  # QUANTIZE int8→uint8

    operators = [op0, op1, op2, op3, op4, op5, op6, op7, op8]

    # Operator codes
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.CONV_2D)
    oc2 = _build_operator_code(builder, BuiltinOp.MUL)
    oc3 = _build_operator_code(builder, BuiltinOp.ADD)
    oc4 = _build_operator_code(builder, BuiltinOp.AVERAGE_POOL_2D)
    oc5 = _build_operator_code(builder, BuiltinOp.RESHAPE)
    opcodes = [oc0, oc1, oc2, oc3, oc4, oc5]

    # Subgraph
    sg = _build_subgraph(builder,
                         tensors=tensors,
                         inputs=[0],
                         outputs=[14],
                         operators=operators,
                         name="main")

    # Finalize
    desc = f"LoomingDetector {height}x{width} {zones}x{zones} (libredgetpu)"
    tflite_bytes = _build_model(
        builder,
        operator_codes=opcodes,
        subgraphs=[sg],
        buffers=buf_list,
        description=desc,
        version=3,
    )

    # ── Metadata ────────────────────────────────────────────────────────
    metadata = {
        "height": height,
        "width": width,
        "zones": zones,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zp),
        "output_scale": float(final_output_scale),
        "output_zero_point": int(final_output_zp),
        "output_count": zones * zones,
    }

    return tflite_bytes, metadata


# ── Pattern tracker model builder ────────────────────────────────────────────

def build_pattern_tracker(search_h, search_w, kernel_h, kernel_w,
                          channels=1, temperature=0.1,
                          conv_weights_int8=None):
    """Build a quantized pattern tracker TFLite model for PatternTracker.

    Produces the same operator chain as TF's quantization pipeline:
    ``QUANTIZE → CONV_2D(kernel, VALID, fused ReLU) → RESHAPE → SOFTMAX
    → FC(x) → FC(y) → CONCAT``

    Args:
        search_h, search_w: Search image dimensions.
        kernel_h, kernel_w: Template/kernel dimensions.
        channels: Input channels (1=grayscale, 3=RGB).
        temperature: Softmax temperature (default 0.1).
        conv_weights_int8: Optional [1, kernel_h, kernel_w, channels] int8 array.
            If None, a Gaussian blob kernel is generated.

    Returns:
        (tflite_bytes, metadata) where metadata is a dict matching the
        sidecar JSON schema used by existing PatternTracker templates.
    """
    if kernel_h >= search_h or kernel_w >= search_w:
        raise ValueError(
            f"Kernel ({kernel_h}x{kernel_w}) must be smaller than "
            f"search image ({search_h}x{search_w})"
        )

    out_h = search_h - kernel_h + 1
    out_w = search_w - kernel_w + 1
    n_positions = out_h * out_w

    # ── Conv2D kernel ───────────────────────────────────────────────────
    n_weights = kernel_h * kernel_w * channels
    if conv_weights_int8 is not None:
        conv_weights_int8 = np.asarray(conv_weights_int8, dtype=np.int8)
        expected_shape = (1, kernel_h, kernel_w, channels)
        if conv_weights_int8.shape != expected_shape:
            raise ValueError(
                f"conv_weights_int8 must be {expected_shape}, "
                f"got {conv_weights_int8.shape}"
            )
        conv_abs_max = max(float(np.max(np.abs(conv_weights_int8))), 1)
        conv_weight_scale = conv_abs_max / 127.0
    else:
        # Default: Gaussian blob kernel (same as TF generator)
        cy, cx = kernel_h / 2, kernel_w / 2
        sigma = max(kernel_h, kernel_w) / 4
        yy, xx = np.mgrid[:kernel_h, :kernel_w]
        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.max()  # normalize to [0, 1]
        if channels == 1:
            kernel_float = gaussian.reshape(1, kernel_h, kernel_w, 1).astype(np.float32)
        else:
            kernel_float = np.stack([gaussian] * channels, axis=2).reshape(
                1, kernel_h, kernel_w, channels
            ).astype(np.float32) / channels
        conv_abs_max = max(float(np.max(np.abs(kernel_float))), 1e-6)
        conv_weight_scale = conv_abs_max / 127.0
        conv_weights_int8 = np.clip(
            np.round(kernel_float / conv_weight_scale), -127, 127
        ).astype(np.int8)

    # ── Coordinate grids ────────────────────────────────────────────────
    x_coords, y_coords = _create_coordinate_grids(out_h, out_w, temperature)
    y_coords_offset = y_coords + (1.0 / temperature)

    # FC weight shape: [output_dim, input_dim] = [1, N] in TFLite convention
    x_weights = x_coords.reshape(1, n_positions).astype(np.float32)
    y_weights = y_coords_offset.reshape(1, n_positions).astype(np.float32)

    # ── Quantization parameters ─────────────────────────────────────────
    input_scale = 1.0
    input_zp = 0

    q_int8_scale = 1.0
    q_int8_zp = -128

    # Conv2D output with fused ReLU
    # TFLite conv computes: acc = sum((input - input_zp) * weight)
    # With input_zp=-128, effective input = int8 + 128 = [0, 255] for uint8 data.
    # Use actual weight sum for a tighter bound (avoids wasting int8 precision).
    max_effective_input = (127 - q_int8_zp) * q_int8_scale  # 255 for uint8
    weight_abs_sum = float(np.sum(np.abs(conv_weights_int8))) * conv_weight_scale
    conv_output_max = max_effective_input * weight_abs_sum
    conv_output_scale = conv_output_max / 127.0  # map [0, max] to [-128, 127] with zp=-128
    conv_output_zp = -128  # ReLU → non-negative

    # Softmax: scale=1/256, zp=-128
    softmax_scale = 1.0 / 256.0
    softmax_zp = -128

    # X weights: range [-1/T, +1/T], symmetric int8
    x_abs_max = float(np.max(np.abs(x_weights)))
    x_weight_scale = max(x_abs_max, 1e-6) / 127.0
    x_weights_int8 = np.clip(
        np.round(x_weights / x_weight_scale), -127, 127
    ).astype(np.int8)

    # Y weights: range [0, +2/T], symmetric int8
    y_abs_max = float(np.max(np.abs(y_weights)))
    y_weight_scale = max(y_abs_max, 1e-6) / 127.0
    y_weights_int8 = np.clip(
        np.round(y_weights / y_weight_scale), -127, 127
    ).astype(np.int8)

    # Output: concat of x and y
    concat_range_min = -1.0 / temperature
    concat_range_max = 2.0 / temperature
    concat_span = concat_range_max - concat_range_min
    output_scale = concat_span / 255.0
    output_zp = int(round(-128 - concat_range_min / output_scale))
    output_zp = max(-128, min(127, output_zp))

    # ── Build FlatBuffer ────────────────────────────────────────────────
    data_size = n_weights + n_positions * 2 + 64
    builder = flatbuffers.Builder(4096 + data_size)

    # 13 tensors, 16 buffers (matching TF output for pattern tracker)
    buf_list = [_build_buffer(builder, None)]  # buf 0: sentinel

    # T0: input [1,sh,sw,ch] uint8 -> buf 1
    buf_list.append(_build_buffer(builder, None))  # buf 1
    # T1: quantize out [1,sh,sw,ch] int8 -> buf 2
    buf_list.append(_build_buffer(builder, None))  # buf 2
    # T2: x_weights [1, n_positions] int8 -> buf 3
    buf_list.append(_build_buffer(builder, x_weights_int8.tobytes()))  # buf 3
    # T3: y_weights [1, n_positions] int8 -> buf 4
    buf_list.append(_build_buffer(builder, y_weights_int8.tobytes()))  # buf 4
    # T4: conv_bias [1] int32 -> buf 5
    conv_bias = np.zeros(1, dtype=np.int32)
    buf_list.append(_build_buffer(builder, conv_bias.tobytes()))  # buf 5
    # T5: conv_weights [1, kernel_h, kernel_w, channels] int8 -> buf 6
    buf_list.append(_build_buffer(builder, conv_weights_int8.tobytes()))  # buf 6
    # T6: conv_out [1, out_h, out_w, 1] int8 -> buf 7
    buf_list.append(_build_buffer(builder, None))  # buf 7
    # T7: shape_const [2] int32 -> buf 8
    shape_data = np.array([1, n_positions], dtype=np.int32)
    buf_list.append(_build_buffer(builder, shape_data.tobytes()))  # buf 8
    # T8: reshape_out [1, n_positions] int8 -> buf 9
    buf_list.append(_build_buffer(builder, None))  # buf 9
    # T9: softmax_out [1, n_positions] int8 -> buf 10
    buf_list.append(_build_buffer(builder, None))  # buf 10
    # T10: x_fc_out [1, 1] int8 -> buf 11
    buf_list.append(_build_buffer(builder, None))  # buf 11
    # T11: y_fc_out [1, 1] int8 -> buf 12
    buf_list.append(_build_buffer(builder, None))  # buf 12
    # T12: concat_out [1, 2] int8 -> buf 13
    buf_list.append(_build_buffer(builder, None))  # buf 13
    # Extra buffers to reach 16
    buf_list.append(_build_buffer(builder, None))  # buf 14
    buf_list.append(_build_buffer(builder, None))  # buf 15

    # Quantization parameters
    q_input = _build_quantization(builder, [input_scale], [input_zp])
    q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
    q_x_w = _build_quantization(builder, [x_weight_scale], [0])
    q_y_w = _build_quantization(builder, [y_weight_scale], [0])
    conv_bias_scale = q_int8_scale * conv_weight_scale
    q_conv_bias = _build_quantization(builder, [conv_bias_scale], [0])
    q_conv_w = _build_quantization(builder, [conv_weight_scale], [0])
    q_conv_out = _build_quantization(builder, [conv_output_scale], [conv_output_zp])
    q_softmax = _build_quantization(builder, [softmax_scale], [softmax_zp])
    q_fc_out = _build_quantization(builder, [output_scale], [output_zp])
    q_output = _build_quantization(builder, [output_scale], [output_zp])

    # Tensors (13 total)
    t0 = _build_tensor(builder, "input", [1, search_h, search_w, channels], TensorType.UINT8, 1, q_input)
    t1 = _build_tensor(builder, "quantize_out", [1, search_h, search_w, channels], TensorType.INT8, 2, q_int8)
    t2 = _build_tensor(builder, "x_weights", [1, n_positions], TensorType.INT8, 3, q_x_w)
    t3 = _build_tensor(builder, "y_weights", [1, n_positions], TensorType.INT8, 4, q_y_w)
    t4 = _build_tensor(builder, "conv_bias", [1], TensorType.INT32, 5, q_conv_bias)
    t5 = _build_tensor(builder, "conv_weights", [1, kernel_h, kernel_w, channels], TensorType.INT8, 6, q_conv_w)
    t6 = _build_tensor(builder, "conv_out", [1, out_h, out_w, 1], TensorType.INT8, 7, q_conv_out)
    t7 = _build_tensor(builder, "shape_const", [2], TensorType.INT32, 8)
    t8 = _build_tensor(builder, "reshape_out", [1, n_positions], TensorType.INT8, 9, q_conv_out)
    t9 = _build_tensor(builder, "softmax_out", [1, n_positions], TensorType.INT8, 10, q_softmax)
    t10 = _build_tensor(builder, "x_fc_out", [1, 1], TensorType.INT8, 11, q_fc_out)
    t11 = _build_tensor(builder, "y_fc_out", [1, 1], TensorType.INT8, 12, q_fc_out)
    t12 = _build_tensor(builder, "output", [1, 2], TensorType.INT8, 13, q_output)

    tensors = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]

    # Options
    conv_opts = _build_conv2d_options(builder, padding=Padding.VALID,
                                      activation=Activation.RELU)
    reshape_opts = _build_reshape_options(builder, [1, n_positions])
    softmax_opts = _build_softmax_options(builder, beta=1.0)
    fc_opts = _build_fc_options(builder)
    fc_opts2 = _build_fc_options(builder)
    concat_opts = _build_concatenation_options(builder, axis=1)

    # Operators (7 total): QUANTIZE -> CONV_2D -> RESHAPE -> SOFTMAX -> FC(x) -> FC(y) -> CONCAT
    # opcode indices: 0=QUANTIZE, 1=CONV_2D, 2=RESHAPE, 3=SOFTMAX, 4=FC, 5=CONCATENATION
    op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE
    op1 = _build_operator(builder, 1, [1, 5, 4], [6],  # CONV_2D
                          int(BuiltinOptions.Conv2DOptions), conv_opts)
    op2 = _build_operator(builder, 2, [6, 7], [8],  # RESHAPE
                          int(BuiltinOptions.ReshapeOptions), reshape_opts)
    op3 = _build_operator(builder, 3, [8], [9],  # SOFTMAX
                          int(BuiltinOptions.SoftmaxOptions), softmax_opts)
    op4 = _build_operator(builder, 4, [9, 2, -1], [10],  # FC(x)
                          int(BuiltinOptions.FullyConnectedOptions), fc_opts)
    op5 = _build_operator(builder, 4, [9, 3, -1], [11],  # FC(y)
                          int(BuiltinOptions.FullyConnectedOptions), fc_opts2)
    op6 = _build_operator(builder, 5, [10, 11], [12],  # CONCAT
                          int(BuiltinOptions.ConcatenationOptions), concat_opts)

    operators = [op0, op1, op2, op3, op4, op5, op6]

    # Operator codes
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.CONV_2D)
    oc2 = _build_operator_code(builder, BuiltinOp.RESHAPE)
    oc3 = _build_operator_code(builder, BuiltinOp.SOFTMAX)
    oc4 = _build_operator_code(builder, BuiltinOp.FULLY_CONNECTED)
    oc5 = _build_operator_code(builder, BuiltinOp.CONCATENATION)
    opcodes = [oc0, oc1, oc2, oc3, oc4, oc5]

    # Subgraph
    sg = _build_subgraph(builder,
                         tensors=tensors,
                         inputs=[0],
                         outputs=[12],
                         operators=operators,
                         name="main")

    # Finalize
    desc = (f"PatternTracker {search_h}x{search_w} "
            f"kernel={kernel_h}x{kernel_w} ch={channels} (libredgetpu)")
    tflite_bytes = _build_model(
        builder,
        operator_codes=opcodes,
        subgraphs=[sg],
        buffers=buf_list,
        description=desc,
        version=3,
    )

    # ── Metadata ────────────────────────────────────────────────────────
    metadata = {
        "search_height": search_h,
        "search_width": search_w,
        "kernel_height": kernel_h,
        "kernel_width": kernel_w,
        "channels": channels,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zp),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zp),
        "output_count": 2,
        "temperature": float(temperature),
        "y_offset": 1.0 / temperature,
        "conv_weight_scale": float(conv_weight_scale),
        "conv_weight_zero_point": 0,
        "conv_weight_count": n_weights,
    }

    return tflite_bytes, metadata


# ── Gabor kernel helper ──────────────────────────────────────────────────────

def _generate_gabor_kernels(ksize=7, orientations=4, sigmas=(1.5, 3.0)):
    """Generate a bank of Gabor kernels for optical flow feature extraction.

    Orientation convention (standard Gabor): theta is the edge orientation angle.
    - theta=0 (0°): responds to horizontal edges (perpendicular to orientation 0°)
    - theta=π/4 (45°): responds to diagonal edges (NE-SW orientation)
    - theta=π/2 (90°): responds to vertical edges (perpendicular to orientation 90°)
    - theta=3π/4 (135°): responds to diagonal edges (NW-SE orientation)

    Args:
        ksize: Kernel size (must be odd).
        orientations: Number of evenly-spaced orientations (default 4: 0/45/90/135).
        sigmas: Tuple of Gaussian envelope scales (default (1.5, 3.0) for 2 scales).

    Returns:
        numpy float32 array of shape [ksize, ksize, 1, N] where N = orientations * len(sigmas).
        Kernels are normalized so max(|kernel|) ≈ 1.0 per filter.
    """
    n_filters = orientations * len(sigmas)
    kernels = np.zeros((ksize, ksize, 1, n_filters), dtype=np.float32)
    half = ksize // 2

    # Coordinate grid centered at kernel center
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)

    idx = 0
    for sigma in sigmas:
        # Wavelength ≈ 2 * sigma (one cycle per Gaussian envelope)
        wavelength = 2.0 * sigma
        for oi in range(orientations):
            theta = oi * np.pi / orientations
            # Rotated coordinates
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)
            # Gabor = Gaussian envelope × cosine carrier
            # Use y_theta for carrier to get horizontal stripes at theta=0
            gaussian = np.exp(-(x_theta ** 2 + y_theta ** 2) / (2.0 * sigma ** 2))
            carrier = np.cos(2.0 * np.pi * y_theta / wavelength)
            gabor = gaussian * carrier
            # Normalize to [-1, 1]
            abs_max = float(np.max(np.abs(gabor)))
            if abs_max > 1e-6:
                gabor = gabor / abs_max
            kernels[:, :, 0, idx] = gabor
            idx += 1

    return kernels


# ── Optical flow model builder ───────────────────────────────────────────────

def build_optical_flow(height, width, ksize=7, orientations=4,
                       sigmas=(1.5, 3.0)):
    """Build a quantized Gabor feature extraction TFLite model for OpticalFlow.

    Produces the operator chain:
    ``QUANTIZE(uint8→int8) → DEPTHWISE_CONV_2D(N Gabor filters, SAME, ReLU, depth_multiplier=N) → QUANTIZE(int8→uint8)``

    Uses depthwise convolution with depth_multiplier=N to apply N independent Gabor filters
    to a single input channel. This is semantically correct for Gabor filtering (no cross-channel
    mixing) and more efficient than standard Conv2D.

    The model extracts multi-scale, multi-orientation edge features using fixed
    Gabor filter kernels. The output features are used by the OpticalFlow class
    for CPU-side global correlation and soft argmax to compute (vx, vy).

    Args:
        height: Input image height.
        width: Input image width.
        ksize: Gabor kernel size (default 7).
        orientations: Number of orientations (default 4: 0/45/90/135).
        sigmas: Tuple of Gaussian envelope scales (default (1.5, 3.0)).

    Returns:
        (tflite_bytes, metadata) where metadata is a dict matching the
        sidecar JSON schema used by existing templates.
    """
    n_filters = orientations * len(sigmas)

    # ── Generate Gabor kernels ─────────────────────────────────────────
    gabor_hwio = _generate_gabor_kernels(ksize, orientations, sigmas)
    # Shape: [ksize, ksize, 1, n_filters] (HWIO format from generator)

    # Reshape for depthwise conv: [1, ksize, ksize, n_filters]
    # Depthwise filter format: [1, kernel_h, kernel_w, in_channels * depth_multiplier]
    gabor_float = np.transpose(gabor_hwio, (2, 0, 1, 3))  # [1, ksize, ksize, n_filters]

    # ── Quantization parameters ────────────────────────────────────────
    # Input: uint8 normalized to [0,1] range (TensorFlow-style)
    # This avoids quantization overflow at uint8=128 boundary
    input_scale = 1.0 / 255.0  # Normalize uint8 [0,255] to float [0,1]
    input_zp = 0

    # After QUANTIZE: int8 with same scale, zp=-128
    # Maps: uint8=0→int8=-128, uint8=128→int8≈0, uint8=255→int8=127
    q_int8_scale = 1.0 / 255.0
    q_int8_zp = -128

    # Gabor weights: int8 symmetric with PER-CHANNEL quantization
    # Each of the 8 Gabor kernels gets its own scale based on its max magnitude
    # gabor_float is [1, ksize, ksize, n_filters] for depthwise conv
    per_ch_weight_scales = []
    gabor_int8 = np.zeros_like(gabor_float, dtype=np.int8)

    for ch in range(n_filters):
        kernel_ch = gabor_float[0, :, :, ch]  # [ksize, ksize] - depthwise indexing
        ch_abs_max = float(np.max(np.abs(kernel_ch)))
        ch_scale = max(ch_abs_max, 1e-6) / 127.0
        per_ch_weight_scales.append(ch_scale)

        # Quantize this channel with its own scale
        gabor_int8[0, :, :, ch] = np.clip(
            np.round(kernel_ch / ch_scale), -127, 127
        ).astype(np.int8)

    # For bias scale computation, use PER-CHANNEL scales
    # bias_scale[i] = input_scale * weight_scale[i]
    per_ch_bias_scales = [q_int8_scale * ws for ws in per_ch_weight_scales]

    # Conv2D bias: zero (INT32), one per output channel
    conv_bias = np.zeros(n_filters, dtype=np.int32)

    # For average-based metrics, use mean of per-channel scales
    gabor_weight_scale = np.mean(per_ch_weight_scales)

    # Conv2D output with fused ReLU: non-negative
    # IMPORTANT: Edge TPU empirically produces much larger values than RMS estimate
    # Use full worst-case estimate to prevent saturation (empirically validated Feb 8, 2026)
    # Worst-case: ksize² * 127² / 127 = ksize² * 127
    # Testing showed 60% still saturated channels 5/7, so use 150% of worst-case
    import math
    worst_case_acc_per_output = ksize * ksize * 127.0
    conservative_acc = worst_case_acc_per_output * 1.5  # 150% of worst-case for safety margin
    conv_output_max = conservative_acc * q_int8_scale * gabor_weight_scale
    conv_output_scale = conv_output_max / 127.0
    conv_output_zp = -128  # ReLU → non-negative, so zp=-128

    # Old RMS-based estimate (too small, caused saturation):
    # rms_factor = 127.0 / math.sqrt(3.0)  # ≈ 73.3
    # typical_acc = math.sqrt(ksize²) * rms_factor² / 127.0
    # This gave scale ≈ 7.2e-05, but Edge TPU needs ≈ 0.002-0.003

    # Final uint8 output: same range as conv, mapped to uint8
    final_output_scale = conv_output_scale
    final_output_zp = 0  # non-negative, zp=0 maps 0→0

    # ── Build FlatBuffer ───────────────────────────────────────────────
    # Weights are in HWIO format (TFLite convention for Depthwise Conv2D)
    weight_bytes = gabor_int8.tobytes()
    builder = flatbuffers.Builder(4096 + len(weight_bytes))

    # Buffers
    buf_list = [_build_buffer(builder, None)]  # buf 0: sentinel
    buf_list.append(_build_buffer(builder, None))  # buf 1: input
    buf_list.append(_build_buffer(builder, None))  # buf 2: quantize out
    buf_list.append(_build_buffer(builder, gabor_int8.tobytes()))  # buf 3: weights
    buf_list.append(_build_buffer(builder, conv_bias.tobytes()))   # buf 4: bias
    buf_list.append(_build_buffer(builder, None))  # buf 5: conv out
    buf_list.append(_build_buffer(builder, None))  # buf 6: final output
    # Extra buffers for padding
    buf_list.append(_build_buffer(builder, None))  # buf 7
    buf_list.append(_build_buffer(builder, None))  # buf 8

    # Quantization parameters — per-channel for conv weights and bias
    # per_ch_weight_scales and per_ch_bias_scales already computed above
    per_ch_weight_zps = [0] * n_filters
    per_ch_bias_zps = [0] * n_filters

    q_input = _build_quantization(builder, [input_scale], [input_zp])
    q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
    # Per-channel quantization: axis 3 is the output channel dimension [1, H, W, C]
    q_weights = _build_quantization(builder, per_ch_weight_scales, per_ch_weight_zps,
                                     quantized_dimension=3)
    q_bias = _build_quantization(builder, per_ch_bias_scales, per_ch_bias_zps,
                                  quantized_dimension=0)
    q_conv_out = _build_quantization(builder, [conv_output_scale], [conv_output_zp])
    q_final = _build_quantization(builder, [final_output_scale], [final_output_zp])

    # Tensors
    t0 = _build_tensor(builder, "input", [1, height, width, 1],
                        TensorType.UINT8, 1, q_input)
    t1 = _build_tensor(builder, "quantize_out", [1, height, width, 1],
                        TensorType.INT8, 2, q_int8)
    t2 = _build_tensor(builder, "gabor_weights",
                        [1, ksize, ksize, n_filters],
                        TensorType.INT8, 3, q_weights)
    t3 = _build_tensor(builder, "conv_bias", [n_filters],
                        TensorType.INT32, 4, q_bias)
    t4 = _build_tensor(builder, "conv_out", [1, height, width, n_filters],
                        TensorType.INT8, 5, q_conv_out)
    t5 = _build_tensor(builder, "output", [1, height, width, n_filters],
                        TensorType.UINT8, 6, q_final)

    tensors = [t0, t1, t2, t3, t4, t5]

    # Options
    depthwise_opts = _build_depthwise_conv2d_options(
        builder, padding=Padding.SAME,
        depth_multiplier=n_filters,
        activation=Activation.RELU
    )

    # Operators
    # opcode 0: QUANTIZE, opcode 1: DEPTHWISE_CONV_2D
    op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE uint8→int8
    op1 = _build_operator(builder, 1, [1, 2, 3], [4],  # DEPTHWISE_CONV_2D
                          int(BuiltinOptions.DepthwiseConv2DOptions), depthwise_opts)
    op2 = _build_operator(builder, 0, [4], [5])  # QUANTIZE int8→uint8

    operators = [op0, op1, op2]

    # Operator codes
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.DEPTHWISE_CONV_2D)
    opcodes = [oc0, oc1]

    # Subgraph
    sg = _build_subgraph(builder,
                         tensors=tensors,
                         inputs=[0],
                         outputs=[5],
                         operators=operators,
                         name="main")

    # Finalize
    desc = (f"OpticalFlow Gabor {height}x{width} "
            f"k={ksize} o={orientations} s={len(sigmas)} (libredgetpu)")
    tflite_bytes = _build_model(
        builder,
        operator_codes=opcodes,
        subgraphs=[sg],
        buffers=buf_list,
        description=desc,
        version=3,
    )

    # ── Metadata ───────────────────────────────────────────────────────
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


# ── Optical flow pooled model builder ─────────────────────────────────────────

def build_optical_flow_pooled(height, width, ksize=7, orientations=4,
                              sigmas=(1.5, 3.0), pool_factor=4):
    """Build a quantized Gabor+Pool TFLite model for OpticalFlow.

    Produces the operator chain:
    ``QUANTIZE(uint8→int8) → DEPTHWISE_CONV_2D(N Gabor, SAME, ReLU, depth_multiplier=N) → AVG_POOL_2D
    → QUANTIZE(int8→uint8)``

    Uses depthwise convolution with depth_multiplier=N to apply N independent Gabor filters
    to a single input channel. This is semantically correct for Gabor filtering (no cross-channel
    mixing) and more efficient than standard Conv2D.

    By fusing AVG_POOL_2D into the Edge TPU model, the output shrinks from
    ``(H, W, N)`` to ``(H/P, W/P, N)`` — a ``P²×`` reduction in USB transfer
    bytes.  For the default 64×64 with pool_factor=4 this is 2048 vs 32768
    bytes per frame.

    Args:
        height: Input image height (must be divisible by pool_factor).
        width: Input image width (must be divisible by pool_factor).
        ksize: Gabor kernel size (default 7).
        orientations: Number of orientations (default 4: 0/45/90/135).
        sigmas: Tuple of Gaussian envelope scales (default (1.5, 3.0)).
        pool_factor: Spatial downsampling factor (default 4).

    Returns:
        (tflite_bytes, metadata) where metadata is a dict matching the
        sidecar JSON schema used by existing templates.  Includes
        ``"fused_pool": pool_factor`` to signal pooled mode to the runtime.

    Raises:
        ValueError: If height or width is not divisible by pool_factor.
    """
    if height % pool_factor != 0:
        raise ValueError(
            f"height ({height}) must be divisible by pool_factor ({pool_factor})"
        )
    if width % pool_factor != 0:
        raise ValueError(
            f"width ({width}) must be divisible by pool_factor ({pool_factor})"
        )

    n_filters = orientations * len(sigmas)
    out_h = height // pool_factor
    out_w = width // pool_factor

    # ── Generate Gabor kernels ─────────────────────────────────────────
    gabor_hwio = _generate_gabor_kernels(ksize, orientations, sigmas)
    # Shape: [ksize, ksize, 1, n_filters] (HWIO format from generator)

    # Reshape for depthwise conv: [1, ksize, ksize, n_filters]
    # Depthwise filter format: [1, kernel_h, kernel_w, in_channels * depth_multiplier]
    gabor_float = np.transpose(gabor_hwio, (2, 0, 1, 3))  # [1, ksize, ksize, n_filters]

    # ── Quantization parameters ────────────────────────────────────────
    # Input: uint8 normalized to [0,1] range (TensorFlow-style)
    # This avoids quantization overflow at uint8=128 boundary
    input_scale = 1.0 / 255.0  # Normalize uint8 [0,255] to float [0,1]
    input_zp = 0

    # After QUANTIZE: int8 with same scale, zp=-128
    # Maps: uint8=0→int8=-128, uint8=128→int8≈0, uint8=255→int8=127
    q_int8_scale = 1.0 / 255.0
    q_int8_zp = -128

    # Gabor weights: int8 symmetric with PER-CHANNEL quantization
    # Each of the 8 Gabor kernels gets its own scale based on its max magnitude
    # gabor_float is [1, ksize, ksize, n_filters] for depthwise conv
    per_ch_weight_scales = []
    gabor_int8 = np.zeros_like(gabor_float, dtype=np.int8)

    for ch in range(n_filters):
        kernel_ch = gabor_float[0, :, :, ch]  # [ksize, ksize] - depthwise indexing
        ch_abs_max = float(np.max(np.abs(kernel_ch)))
        ch_scale = max(ch_abs_max, 1e-6) / 127.0
        per_ch_weight_scales.append(ch_scale)

        # Quantize this channel with its own scale
        gabor_int8[0, :, :, ch] = np.clip(
            np.round(kernel_ch / ch_scale), -127, 127
        ).astype(np.int8)

    # For bias scale computation, use PER-CHANNEL scales
    # bias_scale[i] = input_scale * weight_scale[i]
    per_ch_bias_scales = [q_int8_scale * ws for ws in per_ch_weight_scales]

    # Conv2D bias: zero (INT32), one per output channel
    conv_bias = np.zeros(n_filters, dtype=np.int32)

    # For average-based metrics, use mean of per-channel scales
    gabor_weight_scale = np.mean(per_ch_weight_scales)

    # Conv2D output with fused ReLU: non-negative
    # IMPORTANT: Edge TPU empirically produces much larger values than RMS estimate
    # Use full worst-case estimate to prevent saturation (empirically validated Feb 8, 2026)
    # Worst-case: ksize² * 127² / 127 = ksize² * 127
    # Testing showed 60% still saturated channels 5/7, so use 150% of worst-case
    import math
    worst_case_acc_per_output = ksize * ksize * 127.0
    conservative_acc = worst_case_acc_per_output * 1.5  # 150% of worst-case for safety margin
    conv_output_max = conservative_acc * q_int8_scale * gabor_weight_scale
    conv_output_scale = conv_output_max / 127.0
    conv_output_zp = -128  # ReLU → non-negative, so zp=-128

    # Old RMS-based estimate (too small, caused saturation):
    # rms_factor = 127.0 / math.sqrt(3.0)  # ≈ 73.3
    # typical_acc = math.sqrt(ksize²) * rms_factor² / 127.0
    # This gave scale ≈ 7.2e-05, but Edge TPU needs ≈ 0.002-0.003

    # AVG_POOL preserves the quantization parameters of its input
    pool_output_scale = conv_output_scale
    pool_output_zp = conv_output_zp

    # Final uint8 output: same range as pool, mapped to uint8
    final_output_scale = pool_output_scale
    final_output_zp = 0  # non-negative, zp=0 maps 0→0

    # ── Build FlatBuffer ───────────────────────────────────────────────
    # Weights are in HWIO format (TFLite convention for Depthwise Conv2D)
    weight_bytes = gabor_int8.tobytes()
    builder = flatbuffers.Builder(4096 + len(weight_bytes))

    # Buffers: 10 total (sentinel + 7 tensors + 2 padding)
    buf_list = [_build_buffer(builder, None)]          # buf 0: sentinel
    buf_list.append(_build_buffer(builder, None))      # buf 1: input
    buf_list.append(_build_buffer(builder, None))      # buf 2: quantize out
    buf_list.append(_build_buffer(builder, gabor_int8.tobytes()))  # buf 3: weights
    buf_list.append(_build_buffer(builder, conv_bias.tobytes()))   # buf 4: bias
    buf_list.append(_build_buffer(builder, None))      # buf 5: conv out
    buf_list.append(_build_buffer(builder, None))      # buf 6: pool out
    buf_list.append(_build_buffer(builder, None))      # buf 7: final output
    buf_list.append(_build_buffer(builder, None))      # buf 8: padding
    buf_list.append(_build_buffer(builder, None))      # buf 9: padding

    # Quantization parameters — per-channel for conv weights and bias
    # per_ch_weight_scales and per_ch_bias_scales already computed above
    per_ch_weight_zps = [0] * n_filters
    per_ch_bias_zps = [0] * n_filters

    q_input = _build_quantization(builder, [input_scale], [input_zp])
    q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
    # Per-channel quantization: axis 3 is the output channel dimension [1, H, W, C]
    q_weights = _build_quantization(builder, per_ch_weight_scales, per_ch_weight_zps,
                                     quantized_dimension=3)
    q_bias = _build_quantization(builder, per_ch_bias_scales, per_ch_bias_zps,
                                  quantized_dimension=0)
    q_conv_out = _build_quantization(builder, [conv_output_scale], [conv_output_zp])
    q_pool_out = _build_quantization(builder, [pool_output_scale], [pool_output_zp])
    q_final = _build_quantization(builder, [final_output_scale], [final_output_zp])

    # Tensors (7 total)
    # T0: input [1, H, W, 1] uint8
    t0 = _build_tensor(builder, "input", [1, height, width, 1],
                        TensorType.UINT8, 1, q_input)
    # T1: quantize out [1, H, W, 1] int8
    t1 = _build_tensor(builder, "quantize_out", [1, height, width, 1],
                        TensorType.INT8, 2, q_int8)
    # T2: gabor weights [n_filters, ksize, ksize, 1] int8 OHWI format
    t2 = _build_tensor(builder, "gabor_weights",
                        [1, ksize, ksize, n_filters],
                        TensorType.INT8, 3, q_weights)
    # T3: conv bias [n_filters] int32
    t3 = _build_tensor(builder, "conv_bias", [n_filters],
                        TensorType.INT32, 4, q_bias)
    # T4: conv out [1, H, W, n_filters] int8
    t4 = _build_tensor(builder, "conv_out", [1, height, width, n_filters],
                        TensorType.INT8, 5, q_conv_out)
    # T5: pool out [1, H/P, W/P, n_filters] int8
    t5 = _build_tensor(builder, "pool_out", [1, out_h, out_w, n_filters],
                        TensorType.INT8, 6, q_pool_out)
    # T6: final output [1, H/P, W/P, n_filters] uint8
    t6 = _build_tensor(builder, "output", [1, out_h, out_w, n_filters],
                        TensorType.UINT8, 7, q_final)

    tensors = [t0, t1, t2, t3, t4, t5, t6]

    # Options
    depthwise_opts = _build_depthwise_conv2d_options(
        builder, padding=Padding.SAME,
        depth_multiplier=n_filters,
        activation=Activation.RELU
    )
    pool_opts = _build_pool2d_options(builder, padding=Padding.VALID,
                                      stride_w=pool_factor, stride_h=pool_factor,
                                      filter_w=pool_factor, filter_h=pool_factor)

    # Operators (4 total)
    # opcode 0: QUANTIZE, opcode 1: DEPTHWISE_CONV_2D, opcode 2: AVERAGE_POOL_2D
    op0 = _build_operator(builder, 0, [0], [1])       # QUANTIZE uint8→int8
    op1 = _build_operator(builder, 1, [1, 2, 3], [4],  # DEPTHWISE_CONV_2D Gabor
                          int(BuiltinOptions.DepthwiseConv2DOptions), depthwise_opts)
    op2 = _build_operator(builder, 2, [4], [5],        # AVG_POOL_2D
                          int(BuiltinOptions.Pool2DOptions), pool_opts)
    op3 = _build_operator(builder, 0, [5], [6])        # QUANTIZE int8→uint8

    operators = [op0, op1, op2, op3]

    # Operator codes
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.DEPTHWISE_CONV_2D)
    oc2 = _build_operator_code(builder, BuiltinOp.AVERAGE_POOL_2D)
    opcodes = [oc0, oc1, oc2]

    # Subgraph
    sg = _build_subgraph(builder,
                         tensors=tensors,
                         inputs=[0],
                         outputs=[6],
                         operators=operators,
                         name="main")

    # Finalize
    desc = (f"OpticalFlow Gabor+Pool {height}x{width} "
            f"k={ksize} o={orientations} s={len(sigmas)} "
            f"p={pool_factor} (libredgetpu)")
    tflite_bytes = _build_model(
        builder,
        operator_codes=opcodes,
        subgraphs=[sg],
        buffers=buf_list,
        description=desc,
        version=3,
    )

    # ── Metadata ───────────────────────────────────────────────────────
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
        "output_count": out_h * out_w * n_filters,
        "gabor_weight_scale": float(gabor_weight_scale),
        "fused_pool": pool_factor,
    }

    return tflite_bytes, metadata


# ── Simple depthwise conv for testing ─────────────────────────────────────────

def build_simple_depthwise(height, width, ksize, num_filters=1,
                          kernel_weights=None, input_scale=1.0/255.0,
                          output_scale=None):
    """Build minimal depthwise convolution model for testing.

    Produces the operator chain:
    ``QUANTIZE(uint8→int8) → DEPTHWISE_CONV_2D(SAME, ReLU) → QUANTIZE(int8→uint8)``

    This is a simplified version of build_optical_flow() for systematic testing
    of depthwise convolution fundamentals.

    Args:
        height: Input image height.
        width: Input image width.
        ksize: Kernel size.
        num_filters: Number of filters (depth_multiplier).
        kernel_weights: Custom kernel values [1, ksize, ksize, num_filters] (default: all 1s).
        input_scale: Input quantization scale (default: 1/255 for TF-style normalization).
        output_scale: Output quantization scale (default: auto-calculated from RMS estimate).

    Returns:
        (tflite_bytes, metadata) where metadata contains quantization parameters.
    """
    # ── Default kernel: all 1s ──────────────────────────────────────────
    if kernel_weights is None:
        kernel_weights = np.ones((1, ksize, ksize, num_filters), dtype=np.float32)

    # Validate shape
    expected_shape = (1, ksize, ksize, num_filters)
    if kernel_weights.shape != expected_shape:
        raise ValueError(f"kernel_weights shape {kernel_weights.shape} != expected {expected_shape}")

    # ── Quantization parameters ────────────────────────────────────────
    # Input: uint8 normalized to [0,1] range (TensorFlow-style)
    input_zp = 0

    # After QUANTIZE: int8 with same scale, zp=-128
    q_int8_scale = 1.0 / 255.0
    q_int8_zp = -128

    # Weights: int8 symmetric with PER-CHANNEL quantization
    per_ch_weight_scales = []
    kernel_int8 = np.zeros_like(kernel_weights, dtype=np.int8)

    for ch in range(num_filters):
        kernel_ch = kernel_weights[0, :, :, ch]  # [ksize, ksize]
        ch_abs_max = float(np.max(np.abs(kernel_ch)))
        ch_scale = max(ch_abs_max, 1e-6) / 127.0
        per_ch_weight_scales.append(ch_scale)

        # Quantize this channel with its own scale
        kernel_int8[0, :, :, ch] = np.clip(
            np.round(kernel_ch / ch_scale), -127, 127
        ).astype(np.int8)

    # Bias scale: per-channel
    per_ch_bias_scales = [q_int8_scale * ws for ws in per_ch_weight_scales]

    # Conv2D bias: zero (INT32), one per output channel
    conv_bias = np.zeros(num_filters, dtype=np.int32)

    # Average weight scale for metadata
    avg_weight_scale = np.mean(per_ch_weight_scales)

    # Output scale: auto-calculate using RMS estimate if not provided
    if output_scale is None:
        rms_factor = 127.0 / math.sqrt(3.0)  # ≈ 73.3
        typical_acc_per_output = math.sqrt(ksize * ksize) * rms_factor * rms_factor / 127.0
        conv_output_max = typical_acc_per_output * q_int8_scale * avg_weight_scale
        output_scale = conv_output_max / 127.0

    conv_output_scale = output_scale
    conv_output_zp = -128  # ReLU → non-negative

    # Final uint8 output: same range as conv
    final_output_scale = conv_output_scale
    final_output_zp = 0  # non-negative, zp=0 maps 0→0

    # ── Build FlatBuffer ───────────────────────────────────────────────
    weight_bytes = kernel_int8.tobytes()
    builder = flatbuffers.Builder(4096 + len(weight_bytes))

    # Buffers
    buf_list = [_build_buffer(builder, None)]  # buf 0: sentinel
    buf_list.append(_build_buffer(builder, None))  # buf 1: input
    buf_list.append(_build_buffer(builder, None))  # buf 2: quantize out
    buf_list.append(_build_buffer(builder, kernel_int8.tobytes()))  # buf 3: weights
    buf_list.append(_build_buffer(builder, conv_bias.tobytes()))   # buf 4: bias
    buf_list.append(_build_buffer(builder, None))  # buf 5: conv out
    buf_list.append(_build_buffer(builder, None))  # buf 6: final output
    buf_list.append(_build_buffer(builder, None))  # buf 7: padding
    buf_list.append(_build_buffer(builder, None))  # buf 8: padding

    # Quantization parameters — per-channel for conv weights and bias
    per_ch_weight_zps = [0] * num_filters
    per_ch_bias_zps = [0] * num_filters

    q_input = _build_quantization(builder, [input_scale], [input_zp])
    q_int8 = _build_quantization(builder, [q_int8_scale], [q_int8_zp])
    q_weights = _build_quantization(builder, per_ch_weight_scales, per_ch_weight_zps,
                                     quantized_dimension=3)
    q_bias = _build_quantization(builder, per_ch_bias_scales, per_ch_bias_zps,
                                  quantized_dimension=0)
    q_conv_out = _build_quantization(builder, [conv_output_scale], [conv_output_zp])
    q_final = _build_quantization(builder, [final_output_scale], [final_output_zp])

    # Tensors
    t0 = _build_tensor(builder, "input", [1, height, width, 1],
                        TensorType.UINT8, 1, q_input)
    t1 = _build_tensor(builder, "quantize_out", [1, height, width, 1],
                        TensorType.INT8, 2, q_int8)
    t2 = _build_tensor(builder, "weights",
                        [1, ksize, ksize, num_filters],
                        TensorType.INT8, 3, q_weights)
    t3 = _build_tensor(builder, "bias", [num_filters],
                        TensorType.INT32, 4, q_bias)
    t4 = _build_tensor(builder, "conv_out", [1, height, width, num_filters],
                        TensorType.INT8, 5, q_conv_out)
    t5 = _build_tensor(builder, "output", [1, height, width, num_filters],
                        TensorType.UINT8, 6, q_final)

    tensors = [t0, t1, t2, t3, t4, t5]

    # Options
    depthwise_opts = _build_depthwise_conv2d_options(
        builder, padding=Padding.SAME,
        depth_multiplier=num_filters,
        activation=Activation.RELU
    )

    # Operators
    op0 = _build_operator(builder, 0, [0], [1])  # QUANTIZE uint8→int8
    op1 = _build_operator(builder, 1, [1, 2, 3], [4],  # DEPTHWISE_CONV_2D
                          int(BuiltinOptions.DepthwiseConv2DOptions), depthwise_opts)
    op2 = _build_operator(builder, 0, [4], [5])  # QUANTIZE int8→uint8

    operators = [op0, op1, op2]

    # Operator codes
    oc0 = _build_operator_code(builder, BuiltinOp.QUANTIZE)
    oc1 = _build_operator_code(builder, BuiltinOp.DEPTHWISE_CONV_2D)
    opcodes = [oc0, oc1]

    # Subgraph
    sg = _build_subgraph(builder,
                         tensors=tensors,
                         inputs=[0],
                         outputs=[5],
                         operators=operators,
                         name="main")

    # Finalize
    desc = f"Simple Depthwise {height}x{width} k={ksize} n={num_filters} (libredgetpu test)"
    tflite_bytes = _build_model(
        builder,
        operator_codes=opcodes,
        subgraphs=[sg],
        buffers=buf_list,
        description=desc,
        version=3,
    )

    # ── Metadata ───────────────────────────────────────────────────────
    metadata = {
        "height": height,
        "width": width,
        "ksize": ksize,
        "num_filters": num_filters,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zp),
        "output_scale": float(final_output_scale),
        "output_zero_point": int(final_output_zp),
        "weight_scales": [float(s) for s in per_ch_weight_scales],
        "avg_weight_scale": float(avg_weight_scale),
    }

    return tflite_bytes, metadata
