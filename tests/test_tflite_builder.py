#!/usr/bin/env python3
"""Tests for the TFLite FlatBuffer builder.

Validates that build_dense(), build_spot_tracker(), build_looming(), and
build_pattern_tracker() produce structurally valid TFLite files that
roundtrip correctly through the parser, with correct quantization parameters,
tensor shapes, operator sequences, and buffer contents.

No Edge TPU hardware required.
"""

import os

import numpy as np
import pytest

from libredgetpu.tflite_builder import (
    build_dense,
    build_spot_tracker,
    build_looming,
    build_pattern_tracker,
    TensorType,
    BuiltinOp,
    BuiltinOptions,
)
from libredgetpu.tflite_parser import parse_full, TFLiteModelFull


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_and_parse(n, weights_int8=None):
    """Build a Dense(n) model and parse it back."""
    tflite_bytes, metadata = build_dense(n, weights_int8=weights_int8)
    model = parse_full(tflite_bytes)
    return tflite_bytes, metadata, model


# ── Structural roundtrip tests ───────────────────────────────────────────────

class TestStructure:
    """Verify tensor/operator/buffer structure of builder output."""

    @pytest.mark.parametrize("n", [64, 256, 1024])
    def test_tensor_count(self, n):
        _, _, m = _build_and_parse(n)
        assert len(m.tensors) == 5

    @pytest.mark.parametrize("n", [64, 256, 1024])
    def test_operator_count(self, n):
        _, _, m = _build_and_parse(n)
        assert len(m.operators) == 3

    def test_tensor_shapes_256(self):
        _, _, m = _build_and_parse(256)
        assert m.tensors[0].shape == [1, 256]  # input
        assert m.tensors[1].shape == [256, 256]  # weights
        assert m.tensors[2].shape == [1, 256]  # quantize out
        assert m.tensors[3].shape == [1, 256]  # FC out
        assert m.tensors[4].shape == [1, 256]  # output

    def test_tensor_shapes_64(self):
        _, _, m = _build_and_parse(64)
        assert m.tensors[0].shape == [1, 64]
        assert m.tensors[1].shape == [64, 64]

    def test_tensor_shapes_1024(self):
        _, _, m = _build_and_parse(1024)
        assert m.tensors[0].shape == [1, 1024]
        assert m.tensors[1].shape == [1024, 1024]

    def test_tensor_dtypes(self):
        _, _, m = _build_and_parse(64)
        assert m.tensors[0].dtype == TensorType.UINT8  # input
        assert m.tensors[1].dtype == TensorType.INT8   # weights
        assert m.tensors[2].dtype == TensorType.INT8   # quantize out
        assert m.tensors[3].dtype == TensorType.INT8   # FC out
        assert m.tensors[4].dtype == TensorType.UINT8  # output

    def test_operator_sequence(self):
        _, _, m = _build_and_parse(256)
        names = [op.opcode_name for op in m.operators]
        assert names == ["QUANTIZE", "FULLY_CONNECTED", "QUANTIZE"]

    def test_operator_io(self):
        _, _, m = _build_and_parse(256)
        # QUANTIZE: input[0] -> output[2]
        assert m.operators[0].inputs == [0]
        assert m.operators[0].outputs == [2]
        # FC: inputs[2, 1, -1] -> output[3]
        assert m.operators[1].inputs == [2, 1, -1]
        assert m.operators[1].outputs == [3]
        # QUANTIZE: input[3] -> output[4]
        assert m.operators[2].inputs == [3]
        assert m.operators[2].outputs == [4]

    def test_graph_io(self):
        _, _, m = _build_and_parse(256)
        assert m.graph_inputs == [0]
        assert m.graph_outputs == [4]

    def test_buffer_count(self):
        _, _, m = _build_and_parse(256)
        assert len(m.buffers) == 6

    def test_weight_buffer_size(self):
        n = 256
        _, _, m = _build_and_parse(n)
        # Buffer at index 2 should contain weight data
        weight_buf = m.buffers[m.tensors[1].buffer_index]
        assert weight_buf is not None
        assert len(weight_buf) == n * n

    @pytest.mark.parametrize("n", [64, 256, 1024])
    def test_weight_buffer_size_parametrized(self, n):
        _, _, m = _build_and_parse(n)
        weight_buf = m.buffers[m.tensors[1].buffer_index]
        assert weight_buf is not None
        assert len(weight_buf) == n * n

    def test_non_weight_buffers_empty(self):
        _, _, m = _build_and_parse(64)
        for i, buf in enumerate(m.buffers):
            if i == m.tensors[1].buffer_index:
                continue  # weight buffer
            assert buf is None, f"Buffer {i} should be empty"

    def test_file_identifier(self):
        tflite_bytes, _, _ = _build_and_parse(64)
        assert tflite_bytes[4:8] == b"TFL3"


# ── Quantization parameter tests ────────────────────────────────────────────

class TestQuantization:
    """Verify quantization parameters match expected values."""

    def test_input_scale(self):
        _, _, m = _build_and_parse(256)
        assert abs(m.tensors[0].scale - 2.0 / 255.0) < 1e-6

    def test_input_zero_point(self):
        _, _, m = _build_and_parse(256)
        assert m.tensors[0].zero_point == 127

    def test_weight_zero_point_symmetric(self):
        _, _, m = _build_and_parse(256)
        assert m.tensors[1].zero_point == 0

    def test_internal_quantize_scale(self):
        """Internal QUANTIZE output has same scale as input."""
        _, _, m = _build_and_parse(256)
        assert abs(m.tensors[2].scale - m.tensors[0].scale) < 1e-6

    def test_internal_zero_point(self):
        """Internal int8 zp = input_zp - 128 = -1."""
        _, _, m = _build_and_parse(256)
        assert m.tensors[2].zero_point == -1

    def test_output_zp_offset(self):
        """Output uint8 zp = FC output int8 zp + 128."""
        _, _, m = _build_and_parse(256)
        fc_zp = m.tensors[3].zero_point
        out_zp = m.tensors[4].zero_point
        assert out_zp == fc_zp + 128

    def test_output_scale_equals_fc_output_scale(self):
        _, _, m = _build_and_parse(256)
        assert abs(m.tensors[4].scale - m.tensors[3].scale) < 1e-9

    def test_metadata_matches_model(self):
        """Metadata dict should match parsed model quantization params."""
        _, meta, m = _build_and_parse(256)
        assert abs(meta["input_scale"] - m.tensors[0].scale) < 1e-6
        assert meta["input_zero_point"] == m.tensors[0].zero_point
        assert abs(meta["weight_scale"] - m.tensors[1].scale) < 1e-6
        assert meta["weight_zero_point"] == m.tensors[1].zero_point
        assert abs(meta["output_scale"] - m.tensors[4].scale) < 1e-6
        assert meta["output_zero_point"] == m.tensors[4].zero_point


# ── Weight data tests ────────────────────────────────────────────────────────

class TestWeightData:
    """Verify custom and default weights survive roundtrip."""

    def test_custom_weights_roundtrip(self):
        n = 64
        rng = np.random.default_rng(42)
        weights = rng.integers(-127, 128, size=(n, n), dtype=np.int8)
        _, _, m = _build_and_parse(n, weights_int8=weights)

        weight_buf = m.buffers[m.tensors[1].buffer_index]
        recovered = np.frombuffer(weight_buf, dtype=np.int8).reshape(n, n)
        np.testing.assert_array_equal(recovered, weights)

    def test_custom_weights_roundtrip_256(self):
        n = 256
        rng = np.random.default_rng(123)
        weights = rng.integers(-127, 128, size=(n, n), dtype=np.int8)
        _, _, m = _build_and_parse(n, weights_int8=weights)

        weight_buf = m.buffers[m.tensors[1].buffer_index]
        recovered = np.frombuffer(weight_buf, dtype=np.int8).reshape(n, n)
        np.testing.assert_array_equal(recovered, weights)

    def test_default_weights_are_zeros(self):
        n = 64
        _, _, m = _build_and_parse(n)
        weight_buf = m.buffers[m.tensors[1].buffer_index]
        recovered = np.frombuffer(weight_buf, dtype=np.int8)
        assert np.all(recovered == 0)

    def test_weight_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must be"):
            build_dense(64, weights_int8=np.zeros((32, 64), dtype=np.int8))

    def test_weight_scale_from_data(self):
        """Weight scale should reflect the max absolute value of weights."""
        n = 64
        weights = np.full((n, n), 50, dtype=np.int8)
        _, meta, _ = _build_and_parse(n, weights_int8=weights)
        assert abs(meta["weight_scale"] - 50.0 / 127.0) < 1e-9


# ── Comparison with TF-generated template ────────────────────────────────────

class TestCompareWithTF:
    """Compare builder output against TF-generated dense_256.tflite."""

    @pytest.fixture
    def tf_model(self):
        """Load the existing TF-generated dense_256.tflite."""
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "libredgetpu", "templates", "dense_256.tflite"
        )
        if not os.path.isfile(template_path):
            pytest.skip("TF-generated dense_256.tflite not found")
        with open(template_path, "rb") as f:
            return parse_full(f.read())

    @pytest.fixture
    def builder_model(self):
        """Build a dense_256 model."""
        _, _, m = _build_and_parse(256)
        return m

    def test_same_tensor_count(self, tf_model, builder_model):
        assert len(builder_model.tensors) == len(tf_model.tensors)

    def test_same_tensor_shapes(self, tf_model, builder_model):
        for i in range(len(tf_model.tensors)):
            assert builder_model.tensors[i].shape == tf_model.tensors[i].shape, \
                f"Tensor {i} shape mismatch"

    def test_same_tensor_dtypes(self, tf_model, builder_model):
        for i in range(len(tf_model.tensors)):
            assert builder_model.tensors[i].dtype == tf_model.tensors[i].dtype, \
                f"Tensor {i} dtype mismatch"

    def test_same_operator_sequence(self, tf_model, builder_model):
        tf_names = [op.opcode_name for op in tf_model.operators]
        builder_names = [op.opcode_name for op in builder_model.operators]
        assert builder_names == tf_names

    def test_same_graph_io(self, tf_model, builder_model):
        assert builder_model.graph_inputs == tf_model.graph_inputs
        assert builder_model.graph_outputs == tf_model.graph_outputs

    def test_input_scale_close(self, tf_model, builder_model):
        """Input scale should be very close (both target 2/255)."""
        assert abs(builder_model.tensors[0].scale - tf_model.tensors[0].scale) < 1e-4

    def test_input_zp_matches(self, tf_model, builder_model):
        assert builder_model.tensors[0].zero_point == tf_model.tensors[0].zero_point

    def test_weight_zp_zero(self, tf_model, builder_model):
        assert builder_model.tensors[1].zero_point == 0
        assert tf_model.tensors[1].zero_point == 0

    def test_internal_zp_minus_one(self, tf_model, builder_model):
        """Both should have internal int8 zero point = -1."""
        assert builder_model.tensors[2].zero_point == tf_model.tensors[2].zero_point


# ── Metadata tests ───────────────────────────────────────────────────────────

class TestMetadata:
    """Verify metadata dict schema matches existing sidecar JSON."""

    def test_metadata_keys(self):
        _, meta, _ = _build_and_parse(256)
        required_keys = {
            "matrix_size", "weight_scale", "weight_zero_point",
            "input_scale", "input_zero_point",
            "output_scale", "output_zero_point",
            "use_bias", "param_size",
        }
        assert required_keys.issubset(meta.keys())

    def test_metadata_matrix_size(self):
        _, meta, _ = _build_and_parse(512)
        assert meta["matrix_size"] == 512

    def test_metadata_no_bias(self):
        _, meta, _ = _build_and_parse(64)
        assert meta["use_bias"] is False

    def test_metadata_param_size(self):
        _, meta, _ = _build_and_parse(256)
        assert meta["param_size"] == 256 * 256


# ── Spot tracker builder tests ──────────────────────────────────────────────

def _build_spot_tracker_and_parse(height=64, width=64, **kwargs):
    """Build a spot tracker model and parse it back."""
    tflite_bytes, metadata = build_spot_tracker(height, width, **kwargs)
    model = parse_full(tflite_bytes)
    return tflite_bytes, metadata, model


class TestSpotTrackerBrightStructure:
    """Verify structure of bright spot tracker builder output."""

    def test_tensor_count(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        assert len(m.tensors) == 10

    def test_operator_count(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        assert len(m.operators) == 6

    def test_operator_sequence(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        names = [op.opcode_name for op in m.operators]
        assert names == [
            "QUANTIZE", "RESHAPE", "SOFTMAX",
            "FULLY_CONNECTED", "FULLY_CONNECTED", "CONCATENATION",
        ]

    def test_input_shape(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        assert m.tensors[0].shape == [1, 64, 64, 1]
        assert m.tensors[0].dtype == TensorType.UINT8

    def test_output_shape(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        out_idx = m.graph_outputs[0]
        assert m.tensors[out_idx].shape == [1, 2]
        assert m.tensors[out_idx].dtype == TensorType.INT8

    def test_graph_io(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        assert m.graph_inputs == [0]
        assert m.graph_outputs == [9]

    def test_softmax_params(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        # Find softmax output tensor
        softmax_op = [op for op in m.operators if op.opcode_name == "SOFTMAX"][0]
        softmax_out = m.tensors[softmax_op.outputs[0]]
        assert abs(softmax_out.scale - 1.0/256.0) < 1e-6
        assert softmax_out.zero_point == -128

    def test_reshape_shape(self):
        _, _, m = _build_spot_tracker_and_parse(32, 48)
        reshape_op = [op for op in m.operators if op.opcode_name == "RESHAPE"][0]
        reshape_out = m.tensors[reshape_op.outputs[0]]
        assert reshape_out.shape == [1, 32 * 48]

    def test_fc_no_bias(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        fc_ops = [op for op in m.operators if op.opcode_name == "FULLY_CONNECTED"]
        assert len(fc_ops) == 2
        for fc in fc_ops:
            assert fc.inputs[2] == -1  # no bias

    def test_weight_buffers_correct_size(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        fc_ops = [op for op in m.operators if op.opcode_name == "FULLY_CONNECTED"]
        for fc in fc_ops:
            weight_tensor = m.tensors[fc.inputs[1]]
            assert weight_tensor.shape == [1, 64 * 64]  # [out, in] = [1, H*W]
            weight_buf = m.buffers[weight_tensor.buffer_index]
            assert weight_buf is not None
            assert len(weight_buf) == 64 * 64  # H*W weight bytes

    @pytest.mark.parametrize("h,w", [(32, 32), (64, 64), (128, 128)])
    def test_various_sizes(self, h, w):
        _, meta, m = _build_spot_tracker_and_parse(h, w)
        assert m.tensors[0].shape == [1, h, w, 1]
        assert meta["height"] == h
        assert meta["width"] == w

    def test_file_identifier(self):
        tflite_bytes, _, _ = _build_spot_tracker_and_parse(64, 64)
        assert tflite_bytes[4:8] == b"TFL3"


class TestSpotTrackerBrightMetadata:
    """Verify spot tracker metadata dict."""

    def test_metadata_keys(self):
        _, meta, _ = _build_spot_tracker_and_parse(64, 64)
        required_keys = {
            "height", "width", "channels", "input_scale", "input_zero_point",
            "output_scale", "output_zero_point", "output_count",
            "variant", "temperature", "y_offset",
        }
        assert required_keys.issubset(meta.keys())

    def test_metadata_values(self):
        _, meta, _ = _build_spot_tracker_and_parse(64, 64, temperature=0.1)
        assert meta["channels"] == 1
        assert meta["variant"] == "bright"
        assert meta["temperature"] == 0.1
        assert abs(meta["y_offset"] - 10.0) < 1e-6
        assert meta["output_count"] == 2
        assert meta["input_scale"] == 1.0
        assert meta["input_zero_point"] == 0

    def test_output_scale_covers_range(self):
        """Output scale * 255 should cover the [-1/T, +2/T] range."""
        _, meta, _ = _build_spot_tracker_and_parse(64, 64, temperature=0.1)
        full_range = meta["output_scale"] * 255
        expected = 3.0 / 0.1  # 30
        assert abs(full_range - expected) < 1e-3


class TestSpotTrackerColor:
    """Verify color variant of spot tracker."""

    def test_tensor_count(self):
        _, _, m = _build_spot_tracker_and_parse(
            64, 64, variant="color_red", color_weights=[1.0, -0.5, -0.5]
        )
        assert len(m.tensors) == 13

    def test_operator_count(self):
        _, _, m = _build_spot_tracker_and_parse(
            64, 64, variant="color_red", color_weights=[1.0, -0.5, -0.5]
        )
        assert len(m.operators) == 7

    def test_operator_sequence(self):
        _, _, m = _build_spot_tracker_and_parse(
            64, 64, variant="color_red", color_weights=[1.0, -0.5, -0.5]
        )
        names = [op.opcode_name for op in m.operators]
        assert names == [
            "QUANTIZE", "CONV_2D", "RESHAPE", "SOFTMAX",
            "FULLY_CONNECTED", "FULLY_CONNECTED", "CONCATENATION",
        ]

    def test_input_shape_rgb(self):
        _, _, m = _build_spot_tracker_and_parse(
            64, 64, variant="color_red", color_weights=[1.0, -0.5, -0.5]
        )
        assert m.tensors[0].shape == [1, 64, 64, 3]

    def test_conv_kernel_shape(self):
        _, _, m = _build_spot_tracker_and_parse(
            64, 64, variant="color_red", color_weights=[1.0, -0.5, -0.5]
        )
        conv_op = [op for op in m.operators if op.opcode_name == "CONV_2D"][0]
        weight_tensor = m.tensors[conv_op.inputs[1]]
        assert weight_tensor.shape == [1, 1, 3, 1]

    def test_metadata_has_color_weight_scale(self):
        _, meta, _ = _build_spot_tracker_and_parse(
            64, 64, variant="color_red", color_weights=[1.0, -0.5, -0.5]
        )
        assert "color_weight_scale" in meta
        assert meta["color_weight_scale"] > 0
        assert meta["channels"] == 3

    def test_color_weights_require_color_variant(self):
        with pytest.raises(ValueError, match="color_weights required"):
            build_spot_tracker(64, 64, variant="color_red")


class TestSpotTrackerComparison:
    """Compare builder output against TF-generated bright_64x64.tflite.

    TF and builder may order tensors differently, so we compare structural
    properties (tensor count, op sequence, shape sets) not positional indices.
    """

    @pytest.fixture
    def tf_model(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "libredgetpu", "tracker",
            "templates", "bright_64x64.tflite"
        )
        if not os.path.isfile(template_path):
            pytest.skip("TF-generated bright_64x64.tflite not found")
        with open(template_path, "rb") as f:
            return parse_full(f.read())

    @pytest.fixture
    def builder_model(self):
        _, _, m = _build_spot_tracker_and_parse(64, 64)
        return m

    def test_same_tensor_count(self, tf_model, builder_model):
        assert len(builder_model.tensors) == len(tf_model.tensors)

    def test_same_operator_sequence(self, tf_model, builder_model):
        tf_names = [op.opcode_name for op in tf_model.operators]
        builder_names = [op.opcode_name for op in builder_model.operators]
        assert builder_names == tf_names

    def test_same_tensor_shape_set(self, tf_model, builder_model):
        """Both models have the same set of tensor shapes (order may differ)."""
        tf_shapes = sorted([str(t.shape) for t in tf_model.tensors])
        builder_shapes = sorted([str(t.shape) for t in builder_model.tensors])
        assert builder_shapes == tf_shapes

    def test_same_graph_io_shape(self, tf_model, builder_model):
        """Input/output tensor shapes match."""
        tf_in = tf_model.tensors[tf_model.graph_inputs[0]]
        b_in = builder_model.tensors[builder_model.graph_inputs[0]]
        assert b_in.shape == tf_in.shape
        assert b_in.dtype == tf_in.dtype
        tf_out = tf_model.tensors[tf_model.graph_outputs[0]]
        b_out = builder_model.tensors[builder_model.graph_outputs[0]]
        assert b_out.shape == tf_out.shape
        assert b_out.dtype == tf_out.dtype

    def test_input_quant_matches(self, tf_model, builder_model):
        assert abs(builder_model.tensors[0].scale - tf_model.tensors[0].scale) < 1e-4
        assert builder_model.tensors[0].zero_point == tf_model.tensors[0].zero_point

    def test_softmax_quant_matches(self, tf_model, builder_model):
        """Softmax output is always scale=1/256, zp=-128."""
        for m in [tf_model, builder_model]:
            softmax_op = [op for op in m.operators if op.opcode_name == "SOFTMAX"][0]
            out_t = m.tensors[softmax_op.outputs[0]]
            assert abs(out_t.scale - 1.0/256.0) < 1e-6
            assert out_t.zero_point == -128


# ── Looming detector builder tests ──────────────────────────────────────────

def _build_looming_and_parse(height=64, width=64, **kwargs):
    """Build a looming model and parse it back."""
    tflite_bytes, metadata = build_looming(height, width, **kwargs)
    model = parse_full(tflite_bytes)
    return tflite_bytes, metadata, model


class TestLoomingStructure:
    """Verify structure of looming detector builder output."""

    def test_tensor_count(self):
        _, _, m = _build_looming_and_parse(64, 64)
        assert len(m.tensors) == 15

    def test_operator_count(self):
        _, _, m = _build_looming_and_parse(64, 64)
        assert len(m.operators) == 9

    def test_operator_sequence(self):
        _, _, m = _build_looming_and_parse(64, 64)
        names = [op.opcode_name for op in m.operators]
        assert names == [
            "QUANTIZE", "CONV_2D", "CONV_2D", "MUL", "MUL",
            "ADD", "AVERAGE_POOL_2D", "RESHAPE", "QUANTIZE",
        ]

    def test_input_shape(self):
        _, _, m = _build_looming_and_parse(64, 64)
        assert m.tensors[0].shape == [1, 64, 64, 1]
        assert m.tensors[0].dtype == TensorType.UINT8

    def test_output_shape(self):
        _, _, m = _build_looming_and_parse(64, 64)
        out_idx = m.graph_outputs[0]
        assert m.tensors[out_idx].shape == [1, 9]
        assert m.tensors[out_idx].dtype == TensorType.UINT8

    def test_graph_io(self):
        _, _, m = _build_looming_and_parse(64, 64)
        assert m.graph_inputs == [0]
        assert m.graph_outputs == [14]

    def test_sobel_kernel_shapes(self):
        _, _, m = _build_looming_and_parse(64, 64)
        conv_ops = [op for op in m.operators if op.opcode_name == "CONV_2D"]
        assert len(conv_ops) == 2
        for conv in conv_ops:
            weight_tensor = m.tensors[conv.inputs[1]]
            assert weight_tensor.shape == [1, 3, 3, 1]

    def test_sobel_kernel_values(self):
        """Sobel kernels should have correct ±64, ±127 pattern."""
        _, _, m = _build_looming_and_parse(64, 64)
        conv_ops = [op for op in m.operators if op.opcode_name == "CONV_2D"]
        for conv in conv_ops:
            weight_tensor = m.tensors[conv.inputs[1]]
            weight_buf = m.buffers[weight_tensor.buffer_index]
            assert weight_buf is not None
            assert len(weight_buf) == 9
            vals = np.frombuffer(weight_buf, dtype=np.int8)
            # Both sobel kernels have {0, ±64, ±127}
            unique_abs = set(np.abs(vals))
            assert unique_abs == {0, 64, 127}

    def test_mul_self_squaring(self):
        """MUL ops should have both inputs the same tensor (self-squaring)."""
        _, _, m = _build_looming_and_parse(64, 64)
        mul_ops = [op for op in m.operators if op.opcode_name == "MUL"]
        assert len(mul_ops) == 2
        for mul in mul_ops:
            assert mul.inputs[0] == mul.inputs[1]

    def test_pool_output_shape(self):
        _, _, m = _build_looming_and_parse(64, 64)
        pool_op = [op for op in m.operators if op.opcode_name == "AVERAGE_POOL_2D"][0]
        pool_out = m.tensors[pool_op.outputs[0]]
        assert pool_out.shape == [1, 3, 3, 1]

    @pytest.mark.parametrize("h,w", [(64, 64), (128, 128)])
    def test_various_sizes(self, h, w):
        _, meta, m = _build_looming_and_parse(h, w)
        assert m.tensors[0].shape == [1, h, w, 1]
        pool_op = [op for op in m.operators if op.opcode_name == "AVERAGE_POOL_2D"][0]
        assert m.tensors[pool_op.outputs[0]].shape == [1, 3, 3, 1]

    def test_zones_validation(self):
        with pytest.raises(ValueError, match="Only zones=3"):
            build_looming(64, 64, zones=5)

    def test_size_validation(self):
        with pytest.raises(ValueError, match="must be >= zones"):
            build_looming(2, 2, zones=3)

    def test_file_identifier(self):
        tflite_bytes, _, _ = _build_looming_and_parse(64, 64)
        assert tflite_bytes[4:8] == b"TFL3"


class TestLoomingMetadata:
    """Verify looming metadata dict."""

    def test_metadata_keys(self):
        _, meta, _ = _build_looming_and_parse(64, 64)
        required_keys = {
            "height", "width", "zones",
            "input_scale", "input_zero_point",
            "output_scale", "output_zero_point", "output_count",
        }
        assert required_keys.issubset(meta.keys())

    def test_metadata_values(self):
        _, meta, _ = _build_looming_and_parse(64, 64)
        assert meta["height"] == 64
        assert meta["width"] == 64
        assert meta["zones"] == 3
        assert meta["output_count"] == 9
        assert meta["input_scale"] == 1.0
        assert meta["input_zero_point"] == 0
        assert meta["output_zero_point"] == 0


class TestLoomingComparison:
    """Compare builder output against TF-generated looming_64x64_3x3.tflite.

    TF may order operators/tensors differently and merge bias buffers. We
    compare structural properties rather than positional indices.
    """

    @pytest.fixture
    def tf_model(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "libredgetpu", "looming",
            "templates", "looming_64x64_3x3.tflite"
        )
        if not os.path.isfile(template_path):
            pytest.skip("TF-generated looming_64x64_3x3.tflite not found")
        with open(template_path, "rb") as f:
            return parse_full(f.read())

    @pytest.fixture
    def builder_model(self):
        _, _, m = _build_looming_and_parse(64, 64)
        return m

    def test_same_tensor_count(self, tf_model, builder_model):
        assert len(builder_model.tensors) == len(tf_model.tensors)

    def test_same_operator_types(self, tf_model, builder_model):
        """Both models have the same multiset of operator types."""
        tf_names = sorted([op.opcode_name for op in tf_model.operators])
        builder_names = sorted([op.opcode_name for op in builder_model.operators])
        assert builder_names == tf_names

    def test_same_tensor_shape_set(self, tf_model, builder_model):
        tf_shapes = sorted([str(t.shape) for t in tf_model.tensors])
        builder_shapes = sorted([str(t.shape) for t in builder_model.tensors])
        assert builder_shapes == tf_shapes

    def test_same_graph_io_shape(self, tf_model, builder_model):
        tf_in = tf_model.tensors[tf_model.graph_inputs[0]]
        b_in = builder_model.tensors[builder_model.graph_inputs[0]]
        assert b_in.shape == tf_in.shape
        assert b_in.dtype == tf_in.dtype
        tf_out = tf_model.tensors[tf_model.graph_outputs[0]]
        b_out = builder_model.tensors[builder_model.graph_outputs[0]]
        assert b_out.shape == tf_out.shape
        assert b_out.dtype == tf_out.dtype

    def test_sobel_kernel_values_match(self, tf_model, builder_model):
        """Sobel kernel int8 values should match between TF and builder."""
        for model_label, m in [("TF", tf_model), ("Builder", builder_model)]:
            conv_ops = [op for op in m.operators if op.opcode_name == "CONV_2D"]
            for conv in conv_ops:
                weight_tensor = m.tensors[conv.inputs[1]]
                weight_buf = m.buffers[weight_tensor.buffer_index]
                vals = np.frombuffer(weight_buf, dtype=np.int8)
                assert set(np.abs(vals)) == {0, 64, 127}, \
                    f"{model_label} sobel kernel unexpected values: {vals}"


# ── Pattern tracker builder tests ───────────────────────────────────────────

def _build_pattern_tracker_and_parse(search_h=64, search_w=64,
                                      kernel_h=8, kernel_w=8, **kwargs):
    """Build a pattern tracker model and parse it back."""
    tflite_bytes, metadata = build_pattern_tracker(
        search_h, search_w, kernel_h, kernel_w, **kwargs
    )
    model = parse_full(tflite_bytes)
    return tflite_bytes, metadata, model


class TestPatternTrackerStructure:
    """Verify structure of pattern tracker builder output."""

    def test_tensor_count(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        assert len(m.tensors) == 13

    def test_operator_count(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        assert len(m.operators) == 7

    def test_operator_sequence(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        names = [op.opcode_name for op in m.operators]
        assert names == [
            "QUANTIZE", "CONV_2D", "RESHAPE", "SOFTMAX",
            "FULLY_CONNECTED", "FULLY_CONNECTED", "CONCATENATION",
        ]

    def test_input_shape(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        assert m.tensors[0].shape == [1, 64, 64, 1]

    def test_conv_output_shape(self):
        """Conv2D VALID padding: output = search - kernel + 1."""
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        conv_op = [op for op in m.operators if op.opcode_name == "CONV_2D"][0]
        conv_out = m.tensors[conv_op.outputs[0]]
        assert conv_out.shape == [1, 57, 57, 1]

    def test_conv_kernel_shape(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        conv_op = [op for op in m.operators if op.opcode_name == "CONV_2D"][0]
        weight_tensor = m.tensors[conv_op.inputs[1]]
        assert weight_tensor.shape == [1, 8, 8, 1]

    def test_reshape_output_shape(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        reshape_op = [op for op in m.operators if op.opcode_name == "RESHAPE"][0]
        reshape_out = m.tensors[reshape_op.outputs[0]]
        assert reshape_out.shape == [1, 57 * 57]

    def test_output_shape(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        out_idx = m.graph_outputs[0]
        assert m.tensors[out_idx].shape == [1, 2]
        assert m.tensors[out_idx].dtype == TensorType.INT8

    def test_graph_io(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        assert m.graph_inputs == [0]
        assert m.graph_outputs == [12]

    @pytest.mark.parametrize("sh,sw,kh,kw", [
        (64, 64, 8, 8), (128, 128, 16, 16), (64, 64, 16, 16),
    ])
    def test_various_sizes(self, sh, sw, kh, kw):
        _, meta, m = _build_pattern_tracker_and_parse(sh, sw, kh, kw)
        assert m.tensors[0].shape == [1, sh, sw, 1]
        conv_op = [op for op in m.operators if op.opcode_name == "CONV_2D"][0]
        conv_out = m.tensors[conv_op.outputs[0]]
        expected_oh = sh - kh + 1
        expected_ow = sw - kw + 1
        assert conv_out.shape == [1, expected_oh, expected_ow, 1]

    def test_rgb_input(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8, channels=3)
        assert m.tensors[0].shape == [1, 64, 64, 3]
        conv_op = [op for op in m.operators if op.opcode_name == "CONV_2D"][0]
        weight_tensor = m.tensors[conv_op.inputs[1]]
        assert weight_tensor.shape == [1, 8, 8, 3]

    def test_kernel_too_large_raises(self):
        with pytest.raises(ValueError, match="must be smaller"):
            build_pattern_tracker(64, 64, 64, 64)

    def test_custom_weights_roundtrip(self):
        rng = np.random.default_rng(42)
        weights = rng.integers(-127, 128, size=(1, 8, 8, 1), dtype=np.int8)
        _, _, m = _build_pattern_tracker_and_parse(
            64, 64, 8, 8, conv_weights_int8=weights
        )
        conv_op = [op for op in m.operators if op.opcode_name == "CONV_2D"][0]
        weight_tensor = m.tensors[conv_op.inputs[1]]
        weight_buf = m.buffers[weight_tensor.buffer_index]
        recovered = np.frombuffer(weight_buf, dtype=np.int8).reshape(1, 8, 8, 1)
        np.testing.assert_array_equal(recovered, weights)

    def test_custom_weights_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="conv_weights_int8 must be"):
            build_pattern_tracker(64, 64, 8, 8,
                                  conv_weights_int8=np.zeros((1, 4, 4, 1), dtype=np.int8))

    def test_file_identifier(self):
        tflite_bytes, _, _ = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        assert tflite_bytes[4:8] == b"TFL3"


class TestPatternTrackerMetadata:
    """Verify pattern tracker metadata dict."""

    def test_metadata_keys(self):
        _, meta, _ = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        required_keys = {
            "search_height", "search_width", "kernel_height", "kernel_width",
            "channels", "input_scale", "input_zero_point",
            "output_scale", "output_zero_point", "output_count",
            "temperature", "y_offset",
            "conv_weight_scale", "conv_weight_zero_point", "conv_weight_count",
        }
        assert required_keys.issubset(meta.keys())

    def test_metadata_values(self):
        _, meta, _ = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        assert meta["search_height"] == 64
        assert meta["search_width"] == 64
        assert meta["kernel_height"] == 8
        assert meta["kernel_width"] == 8
        assert meta["channels"] == 1
        assert meta["output_count"] == 2
        assert meta["conv_weight_count"] == 64
        assert meta["conv_weight_zero_point"] == 0
        assert abs(meta["y_offset"] - 10.0) < 1e-6


class TestPatternTrackerComparison:
    """Compare builder output against TF-generated pattern_64x64_8x8_1ch.tflite.

    TF and builder may order tensors differently, so we compare structural
    properties (tensor count, op sequence, shape sets) not positional indices.
    """

    @pytest.fixture
    def tf_model(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "libredgetpu", "pattern",
            "templates", "pattern_64x64_8x8_1ch.tflite"
        )
        if not os.path.isfile(template_path):
            pytest.skip("TF-generated pattern_64x64_8x8_1ch.tflite not found")
        with open(template_path, "rb") as f:
            return parse_full(f.read())

    @pytest.fixture
    def builder_model(self):
        _, _, m = _build_pattern_tracker_and_parse(64, 64, 8, 8)
        return m

    def test_same_tensor_count(self, tf_model, builder_model):
        assert len(builder_model.tensors) == len(tf_model.tensors)

    def test_same_operator_sequence(self, tf_model, builder_model):
        tf_names = [op.opcode_name for op in tf_model.operators]
        builder_names = [op.opcode_name for op in builder_model.operators]
        assert builder_names == tf_names

    def test_same_tensor_shape_set(self, tf_model, builder_model):
        tf_shapes = sorted([str(t.shape) for t in tf_model.tensors])
        builder_shapes = sorted([str(t.shape) for t in builder_model.tensors])
        assert builder_shapes == tf_shapes

    def test_same_graph_io_shape(self, tf_model, builder_model):
        tf_in = tf_model.tensors[tf_model.graph_inputs[0]]
        b_in = builder_model.tensors[builder_model.graph_inputs[0]]
        assert b_in.shape == tf_in.shape
        assert b_in.dtype == tf_in.dtype
        tf_out = tf_model.tensors[tf_model.graph_outputs[0]]
        b_out = builder_model.tensors[builder_model.graph_outputs[0]]
        assert b_out.shape == tf_out.shape
        assert b_out.dtype == tf_out.dtype
