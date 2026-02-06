"""Minimal TFLite flatbuffer parser using only struct. No tensorflow, no flatbuffers library.

Extracts the edgetpu-custom-op's customOptions bytes plus input/output tensor
quantization parameters from a compiled *_edgetpu.tflite model.

Also provides parse_full() for richer model introspection needed by
post-processing modules (DeepLabV3, PoseNet).
"""

import struct
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["parse", "parse_full", "TFLiteModel", "TFLiteModelFull", "TensorInfo"]


@dataclass
class TensorInfo:
    shape: List[int]
    dtype: int  # TFLite TensorType enum value
    scale: float = 0.0
    zero_point: int = 0
    name: str = ""
    buffer_index: int = 0  # index into Model.buffers


@dataclass
class OperatorInfo:
    opcode_index: int
    opcode_name: str  # builtin name or custom_code
    inputs: List[int]  # tensor indices
    outputs: List[int]  # tensor indices
    custom_options: Optional[bytes] = None
    builtin_options_type: int = 0


@dataclass
class TFLiteModelFull:
    """Rich parsed representation of an entire TFLite model."""
    tensors: List[TensorInfo]
    operators: List[OperatorInfo]
    buffers: List[Optional[bytes]]  # raw buffer data (None if empty)
    buffer_offsets: List[int]  # absolute file offset of each buffer's data (-1 if empty)
    graph_inputs: List[int]   # tensor indices
    graph_outputs: List[int]  # tensor indices


@dataclass
class TFLiteModel:
    custom_op_data: bytes
    input_tensor: TensorInfo
    output_tensor: TensorInfo


# ── FlatBuffer primitives ────────────────────────────────────────────────────

def _u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]

def _i32(buf, off):
    return struct.unpack_from("<i", buf, off)[0]

def _u16(buf, off):
    return struct.unpack_from("<H", buf, off)[0]

def _i8(buf, off):
    return struct.unpack_from("<b", buf, off)[0]

def _f32(buf, off):
    return struct.unpack_from("<f", buf, off)[0]

def _read_table(buf, table_pos):
    """Return (vtable, table_pos) for a flatbuffer table at *table_pos*."""
    vtable_soffset = _i32(buf, table_pos)
    vtable = table_pos - vtable_soffset
    return vtable, table_pos

def _field_offset(buf, vtable, table_pos, field_index):
    """Return absolute offset of field *field_index* (0-based) or None."""
    vtable_len = _u16(buf, vtable)
    voffset_pos = 4 + field_index * 2  # skip vtable_size(2) + table_size(2)
    if voffset_pos + 2 > vtable_len:
        return None
    off = _u16(buf, vtable + voffset_pos)
    if off == 0:
        return None
    return table_pos + off

def _read_offset(buf, pos):
    """Follow a uoffset_t at *pos* and return the target position."""
    return pos + _u32(buf, pos)

def _read_vector(buf, vec_field_pos):
    """Return (count, first_element_pos) for a flatbuffer vector field."""
    vec_pos = _read_offset(buf, vec_field_pos)
    count = _u32(buf, vec_pos)
    return count, vec_pos + 4

def _read_string(buf, str_field_pos):
    """Read a flatbuffer string at the offset stored at *str_field_pos*."""
    str_pos = _read_offset(buf, str_field_pos)
    length = _u32(buf, str_pos)
    return buf[str_pos + 4 : str_pos + 4 + length].decode("utf-8")


# ── TFLite schema navigation ────────────────────────────────────────────────

def _get_root_table(buf):
    root_offset = _u32(buf, 0)
    return _read_table(buf, root_offset)

def _get_field(buf, vtable, table_pos, idx):
    return _field_offset(buf, vtable, table_pos, idx)

def _get_vector_element_table(buf, vec_start, i):
    """Get table at index *i* of a vector-of-offsets."""
    elem_off_pos = vec_start + i * 4
    elem_pos = _read_offset(buf, elem_off_pos)
    return _read_table(buf, elem_pos)

def _parse_tensor(buf, vtable, table_pos):
    """Parse a TFLite Tensor table → TensorInfo."""
    # field 0: shape (vector of int32)
    shape = []
    f = _get_field(buf, vtable, table_pos, 0)
    if f is not None:
        count, start = _read_vector(buf, f)
        shape = [_i32(buf, start + i * 4) for i in range(count)]

    # field 1: type (uint8 — TensorType enum)
    dtype = 0
    f = _get_field(buf, vtable, table_pos, 1)
    if f is not None:
        dtype = buf[f]

    # field 3: name (string)
    name = ""
    f = _get_field(buf, vtable, table_pos, 3)
    if f is not None:
        try:
            name = _read_string(buf, f)
        except (UnicodeDecodeError, struct.error):
            pass

    # field 4: quantization (QuantizationParameters)
    scale = 0.0
    zero_point = 0
    f = _get_field(buf, vtable, table_pos, 4)
    if f is not None:
        qvt, qtp = _read_table(buf, _read_offset(buf, f))
        # QuantizationParameters field 2: scale (vector of float32)
        sf = _get_field(buf, qvt, qtp, 2)
        if sf is not None:
            cnt, s = _read_vector(buf, sf)
            if cnt > 0:
                scale = _f32(buf, s)
        # QuantizationParameters field 3: zero_point (vector of int64)
        zf = _get_field(buf, qvt, qtp, 3)
        if zf is not None:
            cnt, s = _read_vector(buf, zf)
            if cnt > 0:
                zero_point = struct.unpack_from("<q", buf, s)[0]

    if scale < 0:
        warnings.warn(
            f"Tensor {name!r} has negative quantization scale ({scale}); "
            f"using absolute value"
        )
        scale = abs(scale)

    return TensorInfo(shape=shape, dtype=dtype, scale=scale, zero_point=zero_point, name=name)


def parse(tflite_bytes: bytes) -> TFLiteModel:
    """Parse a compiled *_edgetpu.tflite model.

    Returns a TFLiteModel with the edgetpu-custom-op's customOptions and
    the input/output tensor quantization info.

    Raises ValueError if no edgetpu-custom-op is found.
    """
    buf = tflite_bytes if isinstance(tflite_bytes, (bytes, bytearray)) else bytes(tflite_bytes)

    # Model root table
    mvt, mtp = _get_root_table(buf)

    # Model field 1: operator_codes (vector of OperatorCode tables)
    opcodes_field = _get_field(buf, mvt, mtp, 1)
    opcodes_count, opcodes_start = _read_vector(buf, opcodes_field)

    # Build opcode index → custom_code mapping
    custom_code_map = {}
    for i in range(opcodes_count):
        ovt, otp = _get_vector_element_table(buf, opcodes_start, i)
        # OperatorCode field 4: custom_code (string) — in schema v3a+
        cc_field = _get_field(buf, ovt, otp, 4)
        if cc_field is not None:
            try:
                custom_code_map[i] = _read_string(buf, cc_field)
            except (UnicodeDecodeError, struct.error):
                pass
        # Also check deprecated field 1: custom_code (old schema)
        if i not in custom_code_map:
            cc_field_old = _get_field(buf, ovt, otp, 1)
            if cc_field_old is not None:
                try:
                    custom_code_map[i] = _read_string(buf, cc_field_old)
                except (UnicodeDecodeError, struct.error):
                    pass

    # Find the opcode index for edgetpu-custom-op
    edgetpu_opcode_idx = None
    for idx, name in custom_code_map.items():
        if name == "edgetpu-custom-op":
            edgetpu_opcode_idx = idx
            break

    if edgetpu_opcode_idx is None:
        raise ValueError("No edgetpu-custom-op found in operator_codes")

    # Model field 2: subgraphs (vector)
    sg_field = _get_field(buf, mvt, mtp, 2)
    sg_count, sg_start = _read_vector(buf, sg_field)
    sgvt, sgtp = _get_vector_element_table(buf, sg_start, 0)

    # SubGraph field 0: tensors (vector)
    tensors_field = _get_field(buf, sgvt, sgtp, 0)
    tensors_count, tensors_start = _read_vector(buf, tensors_field)

    # SubGraph field 1: inputs (vector of int32 — tensor indices)
    sg_inputs_field = _get_field(buf, sgvt, sgtp, 1)
    sg_inputs_count, sg_inputs_start = _read_vector(buf, sg_inputs_field)
    graph_input_indices = [_i32(buf, sg_inputs_start + j * 4) for j in range(sg_inputs_count)]

    # SubGraph field 2: outputs (vector of int32 — tensor indices)
    sg_outputs_field = _get_field(buf, sgvt, sgtp, 2)
    sg_outputs_count, sg_outputs_start = _read_vector(buf, sg_outputs_field)

    # SubGraph field 3: operators (vector)
    ops_field = _get_field(buf, sgvt, sgtp, 3)
    ops_count, ops_start = _read_vector(buf, ops_field)
    graph_output_indices = [_i32(buf, sg_outputs_start + j * 4) for j in range(sg_outputs_count)]

    # Find the edgetpu-custom-op operator and extract customOptions
    custom_op_data = None
    op_input_indices = None
    op_output_indices = None

    for i in range(ops_count):
        opvt, optp = _get_vector_element_table(buf, ops_start, i)
        # Operator field 0: opcode_index (uint32, default 0)
        oi_field = _get_field(buf, opvt, optp, 0)
        opcode_idx = _u32(buf, oi_field) if oi_field is not None else 0
        if opcode_idx != edgetpu_opcode_idx:
            continue

        # Operator field 5: custom_options (vector of uint8)
        co_field = _get_field(buf, opvt, optp, 5)
        if co_field is not None:
            count, start = _read_vector(buf, co_field)
            custom_op_data = buf[start : start + count]

        # Operator field 1: inputs (vector of int32)
        inp_field = _get_field(buf, opvt, optp, 1)
        if inp_field is not None:
            cnt, s = _read_vector(buf, inp_field)
            op_input_indices = [_i32(buf, s + j * 4) for j in range(cnt)]

        # Operator field 2: outputs (vector of int32)
        out_field = _get_field(buf, opvt, optp, 2)
        if out_field is not None:
            cnt, s = _read_vector(buf, out_field)
            op_output_indices = [_i32(buf, s + j * 4) for j in range(cnt)]
        break

    if custom_op_data is None:
        raise ValueError("edgetpu-custom-op found but has no customOptions")

    # Parse input tensor — use operator's first input, fall back to graph input
    input_indices = op_input_indices or graph_input_indices
    if not input_indices:
        raise ValueError("No input tensors found in model")
    input_idx = input_indices[0]
    if input_idx < 0 or input_idx >= tensors_count:
        raise ValueError(f"Input tensor index {input_idx} out of range [0, {tensors_count})")
    tvt, ttp = _get_vector_element_table(buf, tensors_start, input_idx)
    input_tensor = _parse_tensor(buf, tvt, ttp)

    # Parse output tensor — use operator's first output, fall back to graph output
    output_indices = op_output_indices or graph_output_indices
    if not output_indices:
        raise ValueError("No output tensors found in model")
    output_idx = output_indices[0]
    if output_idx < 0 or output_idx >= tensors_count:
        raise ValueError(f"Output tensor index {output_idx} out of range [0, {tensors_count})")
    tvt, ttp = _get_vector_element_table(buf, tensors_start, output_idx)
    output_tensor = _parse_tensor(buf, tvt, ttp)

    return TFLiteModel(
        custom_op_data=bytes(custom_op_data),
        input_tensor=input_tensor,
        output_tensor=output_tensor,
    )


# ── TFLite builtin opcode enum (subset used by compiled models) ──────────

_BUILTIN_OPCODE_NAMES = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D", 6: "DEQUANTIZE", 9: "FULLY_CONNECTED",
    17: "L2_NORMALIZATION", 18: "MUL", 22: "RESHAPE",
    23: "RESIZE_BILINEAR", 25: "SOFTMAX", 27: "PAD", 32: "CUSTOM",
    34: "SQUEEZE", 56: "ARG_MAX", 80: "FAKE_QUANT",
    87: "LOGICAL_NOT", 97: "RESIZE_NEAREST_NEIGHBOR",
    114: "QUANTIZE", 117: "HARD_SWISH",
}


def _parse_tensor_full(buf, vtable, table_pos):
    """Parse a TFLite Tensor table → TensorInfo with buffer_index."""
    info = _parse_tensor(buf, vtable, table_pos)
    # field 2: buffer (uint32 — index into Model.buffers)
    f = _get_field(buf, vtable, table_pos, 2)
    if f is not None:
        info.buffer_index = _u32(buf, f)
    return info


def parse_full(tflite_bytes: bytes) -> TFLiteModelFull:
    """Parse a TFLite model into a rich representation with all tensors,
    operators, and buffers.  No external dependencies required.

    This is needed by post-processing modules to extract conv weights,
    quant params, and operator graph structure.
    """
    buf = tflite_bytes if isinstance(tflite_bytes, (bytes, bytearray)) else bytes(tflite_bytes)

    mvt, mtp = _get_root_table(buf)

    # ── Buffers (Model field 4: buffers) ──
    # Model schema: 0=version, 1=operator_codes, 2=subgraphs, 3=description, 4=buffers
    buffers = []
    buffer_offsets = []
    buf_field = _get_field(buf, mvt, mtp, 4)
    if buf_field is not None:
        buf_count, buf_start = _read_vector(buf, buf_field)
        for i in range(buf_count):
            bvt, btp = _get_vector_element_table(buf, buf_start, i)
            # Buffer field 0: data (vector of uint8)
            df = _get_field(buf, bvt, btp, 0)
            if df is not None:
                cnt, s = _read_vector(buf, df)
                buffers.append(bytes(buf[s:s + cnt]) if cnt > 0 else None)
                buffer_offsets.append(s if cnt > 0 else -1)
            else:
                buffers.append(None)
                buffer_offsets.append(-1)

    # ── Operator codes (Model field 1) ──
    opcode_names = {}
    opcodes_field = _get_field(buf, mvt, mtp, 1)
    if opcodes_field is not None:
        opcodes_count, opcodes_start = _read_vector(buf, opcodes_field)
        for i in range(opcodes_count):
            ovt, otp = _get_vector_element_table(buf, opcodes_start, i)
            # Check custom_code first (field 4 new schema, field 1 old)
            custom_name = None
            for cc_idx in (4, 1):
                cc_field = _get_field(buf, ovt, otp, cc_idx)
                if cc_field is not None:
                    try:
                        custom_name = _read_string(buf, cc_field)
                        break
                    except (UnicodeDecodeError, struct.error):
                        pass
            if custom_name:
                opcode_names[i] = custom_name
            else:
                # OperatorCode field 0: deprecated_builtin_code (int8)
                code = 0
                f0 = _get_field(buf, ovt, otp, 0)
                if f0 is not None:
                    code = _i8(buf, f0)
                # field 3: builtin_code (int32, new schema v3a+)
                f3 = _get_field(buf, ovt, otp, 3)
                if f3 is not None:
                    new_code = _i32(buf, f3)
                    if new_code != 0:
                        code = new_code
                opcode_names[i] = _BUILTIN_OPCODE_NAMES.get(code, f"OP_{code}")

    # ── Subgraph 0 ──
    sg_field = _get_field(buf, mvt, mtp, 2)
    sg_count, sg_start = _read_vector(buf, sg_field)
    sgvt, sgtp = _get_vector_element_table(buf, sg_start, 0)

    # Tensors (SubGraph field 0)
    tensors = []
    tensors_field = _get_field(buf, sgvt, sgtp, 0)
    tensors_count, tensors_start = _read_vector(buf, tensors_field)
    for i in range(tensors_count):
        tvt, ttp = _get_vector_element_table(buf, tensors_start, i)
        tensors.append(_parse_tensor_full(buf, tvt, ttp))

    # Graph inputs (SubGraph field 1)
    sg_inputs_field = _get_field(buf, sgvt, sgtp, 1)
    sg_in_count, sg_in_start = _read_vector(buf, sg_inputs_field)
    graph_inputs = [_i32(buf, sg_in_start + j * 4) for j in range(sg_in_count)]

    # Graph outputs (SubGraph field 2)
    sg_outputs_field = _get_field(buf, sgvt, sgtp, 2)
    sg_out_count, sg_out_start = _read_vector(buf, sg_outputs_field)
    graph_outputs = [_i32(buf, sg_out_start + j * 4) for j in range(sg_out_count)]

    # Operators (SubGraph field 3)
    operators = []
    ops_field = _get_field(buf, sgvt, sgtp, 3)
    ops_count, ops_start = _read_vector(buf, ops_field)
    for i in range(ops_count):
        opvt, optp = _get_vector_element_table(buf, ops_start, i)

        # field 0: opcode_index
        oi_f = _get_field(buf, opvt, optp, 0)
        opcode_idx = _u32(buf, oi_f) if oi_f is not None else 0

        # field 1: inputs
        inp_f = _get_field(buf, opvt, optp, 1)
        inputs = []
        if inp_f is not None:
            cnt, s = _read_vector(buf, inp_f)
            inputs = [_i32(buf, s + j * 4) for j in range(cnt)]

        # field 2: outputs
        out_f = _get_field(buf, opvt, optp, 2)
        outputs = []
        if out_f is not None:
            cnt, s = _read_vector(buf, out_f)
            outputs = [_i32(buf, s + j * 4) for j in range(cnt)]

        # field 4: builtin_options_type (uint8)
        bot_f = _get_field(buf, opvt, optp, 4)
        builtin_options_type = buf[bot_f] if bot_f is not None else 0

        # field 5: custom_options (vector of uint8)
        co_f = _get_field(buf, opvt, optp, 5)
        custom_options = None
        if co_f is not None:
            cnt, s = _read_vector(buf, co_f)
            custom_options = bytes(buf[s:s + cnt]) if cnt > 0 else None

        operators.append(OperatorInfo(
            opcode_index=opcode_idx,
            opcode_name=opcode_names.get(opcode_idx, f"OP_{opcode_idx}"),
            inputs=inputs,
            outputs=outputs,
            custom_options=custom_options,
            builtin_options_type=builtin_options_type,
        ))

    return TFLiteModelFull(
        tensors=tensors,
        operators=operators,
        buffers=buffers,
        buffer_offsets=buffer_offsets,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
    )
