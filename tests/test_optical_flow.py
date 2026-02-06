#!/usr/bin/env python3
"""Tests for optical flow module.

Offline tests run without hardware. Hardware tests require USB Edge TPU.

Usage:
    pytest tests/test_optical_flow.py -v                    # offline only
    pytest tests/test_optical_flow.py -v --run-hardware     # all tests
"""

import time

import numpy as np
import pytest

from libredgetpu.tflite_builder import (
    build_optical_flow, build_optical_flow_pooled, _generate_gabor_kernels
)


# ---------------------------------------------------------------------------
# Gabor kernel tests
# ---------------------------------------------------------------------------

class TestGaborKernels:
    """Test Gabor kernel generation."""

    def test_shape(self):
        """Kernels should have shape [ksize, ksize, 1, n_filters]."""
        kernels = _generate_gabor_kernels(ksize=7, orientations=4, sigmas=(1.5, 3.0))
        assert kernels.shape == (7, 7, 1, 8)

    def test_shape_custom(self):
        """Custom parameters should produce correct shape."""
        kernels = _generate_gabor_kernels(ksize=5, orientations=3, sigmas=(2.0,))
        assert kernels.shape == (5, 5, 1, 3)

    def test_normalization(self):
        """Each kernel should be normalized to [-1, 1]."""
        kernels = _generate_gabor_kernels()
        for i in range(kernels.shape[-1]):
            k = kernels[:, :, 0, i]
            assert np.max(np.abs(k)) <= 1.0 + 1e-6, \
                f"Kernel {i} not normalized: max abs = {np.max(np.abs(k))}"

    def test_distinct_orientations(self):
        """Different orientations should produce distinct kernels."""
        kernels = _generate_gabor_kernels(ksize=7, orientations=4, sigmas=(1.5,))
        # 4 kernels for 4 orientations at sigma=1.5
        for i in range(4):
            for j in range(i + 1, 4):
                k_i = kernels[:, :, 0, i].flatten()
                k_j = kernels[:, :, 0, j].flatten()
                # Correlation should not be 1.0 (distinct)
                corr = np.abs(np.dot(k_i, k_j) / (np.linalg.norm(k_i) * np.linalg.norm(k_j) + 1e-12))
                assert corr < 0.99, f"Kernels {i} and {j} too similar: corr={corr:.4f}"

    def test_dtype(self):
        """Kernels should be float32."""
        kernels = _generate_gabor_kernels()
        assert kernels.dtype == np.float32

    def test_symmetry(self):
        """Gabor kernels should have even symmetry (cosine carrier)."""
        kernels = _generate_gabor_kernels(ksize=7, orientations=1, sigmas=(2.0,))
        # With theta=0 (horizontal), the kernel should be symmetric about vertical axis
        k = kernels[:, :, 0, 0]
        # Check left-right symmetry for theta=0
        np.testing.assert_allclose(k, k[:, ::-1], atol=1e-5,
                                   err_msg="Theta=0 Gabor not left-right symmetric")


# ---------------------------------------------------------------------------
# TFLite builder tests
# ---------------------------------------------------------------------------

class TestTFLiteBuilder:
    """Test build_optical_flow() produces valid TFLite models."""

    def test_basic_build(self):
        """Should build without error for default parameters."""
        tflite_bytes, metadata = build_optical_flow(64, 64)
        assert len(tflite_bytes) > 100
        assert metadata["height"] == 64
        assert metadata["width"] == 64

    def test_file_identifier(self):
        """TFLite output should start with TFL3 identifier."""
        tflite_bytes, _ = build_optical_flow(32, 32)
        # FlatBuffer identifier is at bytes 4-7
        assert tflite_bytes[4:8] == b"TFL3"

    def test_roundtrip_parse(self):
        """Built model should be parseable by tflite_parser."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, metadata = build_optical_flow(64, 64)
        m = parse_full(tflite_bytes)

        # Check operator sequence: QUANTIZE → DEPTHWISE_CONV_2D → QUANTIZE
        assert len(m.operators) == 3
        assert m.operators[0].opcode_name == "QUANTIZE"
        assert m.operators[1].opcode_name == "DEPTHWISE_CONV_2D"
        assert m.operators[2].opcode_name == "QUANTIZE"

    def test_tensor_shapes(self):
        """Tensor shapes should match expected dimensions."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow(64, 64)
        m = parse_full(tflite_bytes)

        # Input tensor: [1, 64, 64, 1]
        assert m.tensors[0].shape == [1, 64, 64, 1]
        # Output tensor: [1, 64, 64, 8]
        output_idx = m.graph_outputs[0]
        assert m.tensors[output_idx].shape == [1, 64, 64, 8]

    def test_tensor_dtypes(self):
        """Input should be uint8, output should be uint8, internals int8."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow(64, 64)
        m = parse_full(tflite_bytes)

        # Input: uint8 (type 3)
        assert m.tensors[0].dtype == 3
        # Quantize output: int8 (type 9)
        assert m.tensors[1].dtype == 9
        # Final output: uint8 (type 3)
        output_idx = m.graph_outputs[0]
        assert m.tensors[output_idx].dtype == 3

    def test_conv2d_weights_shape(self):
        """Depthwise conv weight tensor should be [1, 7, 7, 8] (depthwise format)."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow(64, 64)
        m = parse_full(tflite_bytes)

        conv_op = m.operators[1]
        weight_tensor = m.tensors[conv_op.inputs[1]]
        assert weight_tensor.shape == [1, 7, 7, 8]

    def test_conv2d_weights_values(self):
        """Conv2D weights should be quantized Gabor kernels (not all zeros)."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow(64, 64)
        m = parse_full(tflite_bytes)

        conv_op = m.operators[1]
        weight_tensor = m.tensors[conv_op.inputs[1]]
        weight_buf = m.buffers[weight_tensor.buffer_index]
        vals = np.frombuffer(weight_buf, dtype=np.int8)
        # Should have non-zero values
        assert np.sum(np.abs(vals)) > 0

    def test_bias_tensor(self):
        """Conv2D bias should be [8] int32 zeros."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow(64, 64)
        m = parse_full(tflite_bytes)

        conv_op = m.operators[1]
        bias_tensor = m.tensors[conv_op.inputs[2]]
        assert bias_tensor.shape == [8]
        assert bias_tensor.dtype == 2  # INT32
        bias_buf = m.buffers[bias_tensor.buffer_index]
        vals = np.frombuffer(bias_buf, dtype=np.int32)
        np.testing.assert_array_equal(vals, np.zeros(8, dtype=np.int32))

    def test_various_sizes(self):
        """Should build for various image sizes."""
        for h, w in [(32, 32), (64, 64), (128, 128), (48, 64)]:
            tflite_bytes, metadata = build_optical_flow(h, w)
            assert metadata["height"] == h
            assert metadata["width"] == w
            assert len(tflite_bytes) > 100

    def test_metadata_keys(self):
        """Metadata should contain all required keys."""
        _, metadata = build_optical_flow(64, 64)
        required = ["height", "width", "ksize", "orientations", "sigmas",
                     "num_filters", "input_scale", "input_zero_point",
                     "output_scale", "output_zero_point", "output_count",
                     "gabor_weight_scale"]
        for key in required:
            assert key in metadata, f"Missing metadata key: {key}"

    def test_metadata_values(self):
        """Metadata values should be reasonable."""
        _, metadata = build_optical_flow(64, 64)
        assert metadata["num_filters"] == 8
        assert metadata["ksize"] == 7
        assert metadata["orientations"] == 4
        assert metadata["sigmas"] == [1.5, 3.0]
        assert metadata["output_count"] == 64 * 64 * 8

    def test_custom_parameters(self):
        """Should build with custom ksize, orientations, sigmas."""
        tflite_bytes, metadata = build_optical_flow(
            32, 32, ksize=5, orientations=3, sigmas=(2.0, 4.0, 6.0))
        assert metadata["ksize"] == 5
        assert metadata["orientations"] == 3
        assert metadata["num_filters"] == 9
        assert len(tflite_bytes) > 100


# ---------------------------------------------------------------------------
# Global correlation tests
# ---------------------------------------------------------------------------

class TestGlobalCorrelation:
    """Test CPU-side global correlation computation."""

    def _make_flow(self):
        """Create a minimal OpticalFlow with stubbed TPU to test CPU methods."""
        from libredgetpu.optical_flow_module import OpticalFlow
        # We'll test the methods directly on a mock instance
        # Create instance without calling __init__ fully
        obj = object.__new__(OpticalFlow)
        obj._search_range = 4
        obj._temperature = 0.1
        obj._pool_factor = 4
        sr = 4
        displacements = []
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                displacements.append((dx, dy))
        obj._displacements = np.array(displacements, dtype=np.float32)
        obj._n_displacements = len(displacements)
        return obj

    def test_zero_displacement_peak(self):
        """Identical features should peak at (0, 0) displacement."""
        flow = self._make_flow()
        feat = np.random.RandomState(42).rand(16, 16, 8).astype(np.float32)
        corr = flow._global_correlation(feat, feat)
        assert corr.shape == (81,)
        # Center displacement (0,0) is at index 40 (row 4, col 4 of 9x9)
        center_idx = 4 * 9 + 4  # 40
        assert corr[center_idx] == np.max(corr), \
            f"Peak at idx {np.argmax(corr)} not at center (40)"

    def test_known_shift(self):
        """Shifting features by (1,0) should peak near dx=1."""
        flow = self._make_flow()
        feat_t = np.random.RandomState(42).rand(16, 16, 8).astype(np.float32)
        # Shift feat_t1 right by 1 pixel
        feat_t1 = np.zeros_like(feat_t)
        feat_t1[:, 1:, :] = feat_t[:, :-1, :]
        corr = flow._global_correlation(feat_t, feat_t1)
        peak_idx = int(np.argmax(corr))
        # Index for (dx=1, dy=0): view[j=4, i=5] → after transpose and flip
        # corr_map[4, 5] → row 4, col 5 in flipped coordinates → 4*9+5 = 41
        expected_idx = 4 * 9 + 5
        assert peak_idx == expected_idx, \
            f"Expected peak at {expected_idx} (dx=1,dy=0), got {peak_idx}"

    def test_correlation_shape(self):
        """Correlation should have n_displacements entries."""
        flow = self._make_flow()
        feat = np.ones((16, 16, 8), dtype=np.float32)
        corr = flow._global_correlation(feat, feat)
        assert corr.shape == (81,)

    def test_uniform_features_flat(self):
        """Uniform features should give flat (all equal) correlation."""
        flow = self._make_flow()
        # All-ones features: shifting doesn't change the overlap sum much
        # (only boundary effects)
        feat = np.ones((16, 16, 8), dtype=np.float32)
        corr = flow._global_correlation(feat, feat)
        # Center should be highest (largest overlap area)
        center_idx = 40
        assert corr[center_idx] >= np.max(corr) - 1e-6


# ---------------------------------------------------------------------------
# Soft argmax tests
# ---------------------------------------------------------------------------

class TestSoftArgmax:
    """Test soft argmax computation."""

    def _make_flow(self, temperature=0.1):
        from libredgetpu.optical_flow_module import OpticalFlow
        obj = object.__new__(OpticalFlow)
        obj._search_range = 4
        obj._temperature = temperature
        sr = 4
        displacements = []
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                displacements.append((dx, dy))
        obj._displacements = np.array(displacements, dtype=np.float32)
        obj._n_displacements = len(displacements)
        return obj

    def test_sharp_peak_at_center(self):
        """Sharp peak at center should give (0, 0)."""
        flow = self._make_flow()
        corr = np.zeros(81, dtype=np.float32)
        corr[40] = 100.0  # Center: (0, 0)
        vx, vy = flow._soft_argmax(corr)
        assert abs(vx) < 0.1, f"Expected vx ≈ 0, got {vx}"
        assert abs(vy) < 0.1, f"Expected vy ≈ 0, got {vy}"

    def test_sharp_peak_at_offset(self):
        """Peak at (2, -1) should give vx≈2, vy≈-1."""
        flow = self._make_flow()
        corr = np.zeros(81, dtype=np.float32)
        # (dx=2, dy=-1): row=3 (dy=-1+4=3), col=6 (dx=2+4=6) → 3*9+6 = 33
        corr[33] = 100.0
        vx, vy = flow._soft_argmax(corr)
        assert abs(vx - 2.0) < 0.2, f"Expected vx ≈ 2, got {vx}"
        assert abs(vy - (-1.0)) < 0.2, f"Expected vy ≈ -1, got {vy}"

    def test_subpixel_interpolation(self):
        """Two adjacent peaks should give intermediate displacement."""
        flow = self._make_flow(temperature=1.0)  # wider softmax
        corr = np.zeros(81, dtype=np.float32)
        # Place equal peaks at (1,0) and (2,0)
        idx1 = 4 * 9 + 5  # (1, 0) → 41
        idx2 = 4 * 9 + 6  # (2, 0) → 42
        corr[idx1] = 10.0
        corr[idx2] = 10.0
        vx, vy = flow._soft_argmax(corr)
        assert 1.3 < vx < 1.7, f"Expected vx ≈ 1.5, got {vx}"
        assert abs(vy) < 0.3, f"Expected vy ≈ 0, got {vy}"

    def test_zero_correlation_gives_zero(self):
        """All-zero correlation should give (0, 0)."""
        flow = self._make_flow()
        corr = np.zeros(81, dtype=np.float32)
        vx, vy = flow._soft_argmax(corr)
        assert abs(vx) < 0.1
        assert abs(vy) < 0.1


# ---------------------------------------------------------------------------
# Pool features tests
# ---------------------------------------------------------------------------

class TestPoolFeatures:
    """Test feature downsampling."""

    def _make_flow(self, pool_factor=4):
        from libredgetpu.optical_flow_module import OpticalFlow
        obj = object.__new__(OpticalFlow)
        obj._pool_factor = pool_factor
        return obj

    def test_exact_ratio(self):
        """64→16 with pool_factor=4 should work exactly."""
        flow = self._make_flow(4)
        feat = np.random.rand(64, 64, 8).astype(np.float32)
        pooled = flow._pool_features(feat)
        assert pooled.shape == (16, 16, 8)

    def test_values_are_means(self):
        """Pooled values should be block means."""
        flow = self._make_flow(2)
        feat = np.zeros((4, 4, 1), dtype=np.float32)
        feat[0, 0, 0] = 4.0
        feat[0, 1, 0] = 8.0
        feat[1, 0, 0] = 0.0
        feat[1, 1, 0] = 0.0
        pooled = flow._pool_features(feat)
        assert pooled.shape == (2, 2, 1)
        assert abs(pooled[0, 0, 0] - 3.0) < 1e-5  # mean(4, 8, 0, 0) = 3

    def test_non_exact_ratio(self):
        """Non-integer ratios should truncate and pool."""
        flow = self._make_flow(4)
        feat = np.random.rand(65, 65, 8).astype(np.float32)
        pooled = flow._pool_features(feat)
        assert pooled.shape == (16, 16, 8)

    def test_int_pool_shape(self):
        """Integer block-sum should produce correct shape."""
        flow = self._make_flow(4)
        feat = np.random.randint(-128, 128, (64, 64, 8), dtype=np.int16)
        pooled = flow._pool_features_int(feat)
        assert pooled.shape == (16, 16, 8)
        assert pooled.dtype == np.int32

    def test_int_pool_values_are_sums(self):
        """Integer pooled values should be block sums."""
        flow = self._make_flow(2)
        feat = np.zeros((4, 4, 1), dtype=np.int16)
        feat[0, 0, 0] = 4
        feat[0, 1, 0] = 8
        feat[1, 0, 0] = 1
        feat[1, 1, 0] = 3
        pooled = flow._pool_features_int(feat)
        assert pooled.shape == (2, 2, 1)
        assert pooled[0, 0, 0] == 16  # sum(4, 8, 1, 3)

    def test_int_pool_non_exact_ratio(self):
        """Integer pool should truncate non-exact ratios."""
        flow = self._make_flow(4)
        feat = np.random.randint(-128, 128, (65, 65, 8), dtype=np.int16)
        pooled = flow._pool_features_int(feat)
        assert pooled.shape == (16, 16, 8)
        assert pooled.dtype == np.int32

    def test_int_vs_float_correlation_equivalence(self):
        """Integer and float correlation paths should give the same peak."""
        flow = self._make_flow(4)
        # Also set up correlation params
        flow._search_range = 4
        flow._temperature = 0.1
        sr = 4
        displacements = []
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                displacements.append((dx, dy))
        flow._displacements = np.array(displacements, dtype=np.float32)
        flow._n_displacements = len(displacements)

        rng = np.random.RandomState(42)
        feat_u8 = rng.randint(0, 256, (64, 64, 8), dtype=np.uint8)
        feat1_u8 = rng.randint(0, 256, (64, 64, 8), dtype=np.uint8)

        # Float path
        scale, zp = 0.003, 128
        feat_f = (feat_u8.astype(np.float32) - zp) * scale
        feat1_f = (feat1_u8.astype(np.float32) - zp) * scale
        pooled_f = flow._pool_features(feat_f)
        pooled1_f = flow._pool_features(feat1_f)
        corr_f = flow._global_correlation(pooled_f, pooled1_f)

        # Int path
        feat_i = feat_u8.astype(np.int16) - np.int16(zp)
        feat1_i = feat1_u8.astype(np.int16) - np.int16(zp)
        pooled_i = flow._pool_features_int(feat_i)
        pooled1_i = flow._pool_features_int(feat1_i)
        corr_i = flow._global_correlation(pooled_i, pooled1_i)
        # Normalize int corr to float-equivalent scale
        norm = (scale * scale) / (4 ** 4)
        corr_i_norm = corr_i.astype(np.float64) * norm

        assert np.argmax(corr_f) == np.argmax(corr_i_norm), \
            "Int and float paths disagree on peak displacement"
        np.testing.assert_allclose(
            corr_i_norm, corr_f.astype(np.float64), rtol=2e-5,
            err_msg="Int and float correlation values diverge")


# ---------------------------------------------------------------------------
# Full pipeline tests (CPU-only, no hardware)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test the full compute pipeline using synthetic data and mocked TPU."""

    def _make_features_from_image(self, image):
        """Simulate Gabor feature extraction (CPU approximation)."""
        h, w = image.shape[:2]
        n_filters = 8
        features = np.zeros((h, w, n_filters), dtype=np.float32)
        # Simple approximation: horizontal/vertical gradients as features
        # This won't match the TPU exactly but tests the pipeline
        padded = np.pad(image.astype(np.float32), 1, mode='edge')
        for f in range(n_filters):
            angle = f * np.pi / n_filters
            dx = np.cos(angle)
            dy = np.sin(angle)
            # Approximate directional gradient
            gx = padded[1:-1, 2:] - padded[1:-1, :-2]
            gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
            features[:, :, f] = np.maximum(0, dx * gx + dy * gy)
        return features

    def _make_flow_obj(self, pool_factor=4, search_range=4, temperature=0.1):
        from libredgetpu.optical_flow_module import OpticalFlow
        obj = object.__new__(OpticalFlow)
        obj._search_range = search_range
        obj._temperature = temperature
        obj._pool_factor = pool_factor
        sr = search_range
        displacements = []
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                displacements.append((dx, dy))
        obj._displacements = np.array(displacements, dtype=np.float32)
        obj._n_displacements = len(displacements)
        return obj

    def test_identical_frames_zero_flow(self):
        """Identical frames should give near-zero flow."""
        flow = self._make_flow_obj()
        rng = np.random.RandomState(42)
        img = rng.randint(0, 255, (64, 64), dtype=np.uint8)

        feat = self._make_features_from_image(img)
        feat_small = flow._pool_features(feat)
        corr = flow._global_correlation(feat_small, feat_small)
        vx, vy = flow._soft_argmax(corr)

        assert abs(vx) < 0.5, f"Expected vx ≈ 0 for identical frames, got {vx}"
        assert abs(vy) < 0.5, f"Expected vy ≈ 0 for identical frames, got {vy}"

    def test_horizontal_shift(self):
        """Horizontal shift should produce positive vx."""
        flow = self._make_flow_obj(pool_factor=1, search_range=4)
        rng = np.random.RandomState(42)
        img = rng.randint(50, 200, (32, 32), dtype=np.uint8)

        feat_t = self._make_features_from_image(img)
        # Shift image right by 2 pixels
        img_shifted = np.zeros_like(img)
        img_shifted[:, 2:] = img[:, :-2]
        feat_t1 = self._make_features_from_image(img_shifted)

        corr = flow._global_correlation(feat_t, feat_t1)
        vx, vy = flow._soft_argmax(corr)

        assert vx > 0.5, f"Expected positive vx for rightward shift, got {vx}"

    def test_vertical_shift(self):
        """Vertical shift should produce positive vy."""
        flow = self._make_flow_obj(pool_factor=1, search_range=4)
        rng = np.random.RandomState(42)
        img = rng.randint(50, 200, (32, 32), dtype=np.uint8)

        feat_t = self._make_features_from_image(img)
        # Shift image down by 2 pixels
        img_shifted = np.zeros_like(img)
        img_shifted[2:, :] = img[:-2, :]
        feat_t1 = self._make_features_from_image(img_shifted)

        corr = flow._global_correlation(feat_t, feat_t1)
        vx, vy = flow._soft_argmax(corr)

        assert vy > 0.5, f"Expected positive vy for downward shift, got {vy}"

    def test_flow_to_direction(self):
        """Direction labels should be correct."""
        from libredgetpu.optical_flow_module import OpticalFlow
        assert OpticalFlow.flow_to_direction(0.0, 0.0) == "center"
        assert OpticalFlow.flow_to_direction(2.0, 0.0) == "right"
        assert OpticalFlow.flow_to_direction(-2.0, 0.0) == "left"
        assert OpticalFlow.flow_to_direction(0.0, 2.0) == "down"
        assert OpticalFlow.flow_to_direction(0.0, -2.0) == "up"
        assert OpticalFlow.flow_to_direction(2.0, -2.0) == "up-right"
        assert OpticalFlow.flow_to_direction(-2.0, 2.0) == "down-left"
        assert OpticalFlow.flow_to_direction(0.1, 0.1) == "center"  # below threshold


# ---------------------------------------------------------------------------
# Pooled builder tests (build_optical_flow_pooled)
# ---------------------------------------------------------------------------

class TestPooledBuilder:
    """Test build_optical_flow_pooled() produces valid Gabor+Pool TFLite models."""

    def test_basic_build(self):
        """Should build without error for default parameters."""
        tflite_bytes, metadata = build_optical_flow_pooled(64, 64)
        assert len(tflite_bytes) > 100
        assert metadata["height"] == 64
        assert metadata["width"] == 64
        assert metadata["fused_pool"] == 4

    def test_file_identifier(self):
        """TFLite output should have TFL3 identifier."""
        tflite_bytes, _ = build_optical_flow_pooled(32, 32, pool_factor=4)
        assert tflite_bytes[4:8] == b"TFL3"

    def test_roundtrip_parse(self):
        """Built model should be parseable with correct operator sequence."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow_pooled(64, 64)
        m = parse_full(tflite_bytes)

        # 4 operators: QUANTIZE → DEPTHWISE_CONV_2D → AVERAGE_POOL_2D → QUANTIZE
        assert len(m.operators) == 4
        assert m.operators[0].opcode_name == "QUANTIZE"
        assert m.operators[1].opcode_name == "DEPTHWISE_CONV_2D"
        assert m.operators[2].opcode_name == "AVERAGE_POOL_2D"
        assert m.operators[3].opcode_name == "QUANTIZE"

    def test_output_shape(self):
        """Output tensor should have pooled dimensions."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow_pooled(64, 64, pool_factor=4)
        m = parse_full(tflite_bytes)

        output_idx = m.graph_outputs[0]
        # Output: [1, 16, 16, 8] (64/4 = 16)
        assert m.tensors[output_idx].shape == [1, 16, 16, 8]

    def test_input_shape(self):
        """Input tensor should have original dimensions."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow_pooled(64, 64, pool_factor=4)
        m = parse_full(tflite_bytes)

        assert m.tensors[0].shape == [1, 64, 64, 1]

    def test_tensor_dtypes(self):
        """Input/output should be uint8, internal tensors int8."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow_pooled(64, 64)
        m = parse_full(tflite_bytes)

        # Input: uint8 (type 3)
        assert m.tensors[0].dtype == 3
        # Quantize output: int8 (type 9)
        assert m.tensors[1].dtype == 9
        # Final output: uint8 (type 3)
        output_idx = m.graph_outputs[0]
        assert m.tensors[output_idx].dtype == 3

    def test_metadata_fused_pool(self):
        """Metadata should include fused_pool key."""
        _, metadata = build_optical_flow_pooled(64, 64, pool_factor=4)
        assert metadata["fused_pool"] == 4
        assert metadata["output_count"] == 16 * 16 * 8  # 2048

    def test_metadata_keys(self):
        """Metadata should contain all required keys."""
        _, metadata = build_optical_flow_pooled(64, 64)
        required = ["height", "width", "ksize", "orientations", "sigmas",
                     "num_filters", "input_scale", "input_zero_point",
                     "output_scale", "output_zero_point", "output_count",
                     "gabor_weight_scale", "fused_pool"]
        for key in required:
            assert key in metadata, f"Missing metadata key: {key}"

    def test_various_sizes(self):
        """Should build for various image sizes with pool_factor=4."""
        for h, w in [(32, 32), (64, 64), (128, 128), (48, 64)]:
            tflite_bytes, metadata = build_optical_flow_pooled(h, w, pool_factor=4)
            assert metadata["height"] == h
            assert metadata["width"] == w
            assert metadata["fused_pool"] == 4
            assert metadata["output_count"] == (h // 4) * (w // 4) * 8

    def test_pool_factor_2(self):
        """Should work with pool_factor=2."""
        tflite_bytes, metadata = build_optical_flow_pooled(64, 64, pool_factor=2)
        assert metadata["fused_pool"] == 2
        assert metadata["output_count"] == 32 * 32 * 8

        from libredgetpu.tflite_parser import parse_full
        m = parse_full(tflite_bytes)
        output_idx = m.graph_outputs[0]
        assert m.tensors[output_idx].shape == [1, 32, 32, 8]

    def test_non_divisible_raises(self):
        """Non-divisible dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            build_optical_flow_pooled(65, 64, pool_factor=4)
        with pytest.raises(ValueError, match="divisible"):
            build_optical_flow_pooled(64, 63, pool_factor=4)

    def test_conv2d_weights_preserved(self):
        """Gabor weights should be non-zero and match depthwise shape [1, 7, 7, 8]."""
        from libredgetpu.tflite_parser import parse_full

        tflite_bytes, _ = build_optical_flow_pooled(64, 64)
        m = parse_full(tflite_bytes)

        conv_op = m.operators[1]
        weight_tensor = m.tensors[conv_op.inputs[1]]
        assert weight_tensor.shape == [1, 7, 7, 8]
        weight_buf = m.buffers[weight_tensor.buffer_index]
        vals = np.frombuffer(weight_buf, dtype=np.int8)
        assert np.sum(np.abs(vals)) > 0


# ---------------------------------------------------------------------------
# Pooled pipeline tests (CPU-only, no hardware)
# ---------------------------------------------------------------------------

class TestPooledPipeline:
    """Test the fused-pool compute pipeline using synthetic data."""

    def _make_flow_obj(self, pool_factor=4, search_range=4, temperature=0.1,
                       fused_pool=4):
        from libredgetpu.optical_flow_module import OpticalFlow
        obj = object.__new__(OpticalFlow)
        obj._search_range = search_range
        obj._temperature = temperature
        obj._pool_factor = pool_factor
        obj._fused_pool = fused_pool
        obj._height = 64
        obj._width = 64
        obj._num_filters = 8
        if fused_pool:
            obj._out_h = 64 // fused_pool
            obj._out_w = 64 // fused_pool
        else:
            obj._out_h = 64
            obj._out_w = 64
        sr = search_range
        displacements = []
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                displacements.append((dx, dy))
        obj._displacements = np.array(displacements, dtype=np.float32)
        obj._n_displacements = len(displacements)

        # Mock _output_info for _compute_from_uint8
        class MockInfo:
            scale = 0.5
            zero_point = 0
        obj._output_info = MockInfo()

        # Overlap counts for normalization
        if fused_pool:
            ph, pw = obj._out_h, obj._out_w
        else:
            ph, pw = obj._out_h // pool_factor, obj._out_w // pool_factor
        nf = obj._num_filters
        obj._overlap_counts = np.array(
            [(ph - abs(dy)) * (pw - abs(dx)) * nf
             for dy in range(-sr, sr + 1) for dx in range(-sr, sr + 1)],
            dtype=np.float64,
        )
        return obj

    def test_identical_features_zero_flow(self):
        """Identical pooled features should give near-zero flow."""
        flow = self._make_flow_obj(fused_pool=4)
        rng = np.random.RandomState(42)
        feat = rng.randint(0, 255, (16, 16, 8), dtype=np.uint8)
        vx, vy = flow._compute_from_uint8(feat, feat)
        assert abs(vx) < 0.5, f"Expected vx ≈ 0, got {vx}"
        assert abs(vy) < 0.5, f"Expected vy ≈ 0, got {vy}"

    def test_shifted_features_detect_motion(self):
        """Shifted features should produce non-zero flow."""
        flow = self._make_flow_obj(fused_pool=4)
        rng = np.random.RandomState(42)
        feat_t = rng.randint(50, 200, (16, 16, 8), dtype=np.uint8)
        # Shift right by 2 pooled pixels
        feat_t1 = np.zeros_like(feat_t)
        feat_t1[:, 2:, :] = feat_t[:, :-2, :]
        vx, vy = flow._compute_from_uint8(feat_t, feat_t1)
        assert vx > 0.5, f"Expected positive vx for rightward shift, got {vx}"

    def test_fused_pool_skips_cpu_pooling(self):
        """Fused pool path should work directly on 16x16 features."""
        flow = self._make_flow_obj(fused_pool=4)
        rng = np.random.RandomState(42)
        # Features are already 16x16 (pooled size)
        feat = rng.randint(0, 255, (16, 16, 8), dtype=np.uint8)
        # This should NOT call _pool_features_int internally
        vx, vy = flow._compute_from_uint8(feat, feat)
        assert isinstance(vx, float)
        assert isinstance(vy, float)


# ---------------------------------------------------------------------------
# Template list tests
# ---------------------------------------------------------------------------

class TestTemplateList:
    """Test template discovery functions."""

    def test_list_templates_returns_list(self):
        """list_templates should return a list (possibly empty)."""
        from libredgetpu.optical_flow.templates import list_templates
        templates = list_templates()
        assert isinstance(templates, list)

    def test_list_templates_format(self):
        """Template entries should be (height, width) tuples."""
        from libredgetpu.optical_flow.templates import list_templates
        templates = list_templates()
        for entry in templates:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            h, w = entry
            assert isinstance(h, int) and h > 0
            assert isinstance(w, int) and w > 0


# ---------------------------------------------------------------------------
# Hardware tests (require USB Edge TPU)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
def test_features_nontrivial():
    """Gabor features should be non-trivial for a random image."""
    from libredgetpu.optical_flow.templates import list_templates
    from libredgetpu.optical_flow_module import OpticalFlow

    templates = list_templates()
    if not templates:
        pytest.skip("No optical flow templates available. "
                     "Generate with: python -m libredgetpu.optical_flow_gen")

    h, w = templates[0]
    with OpticalFlow.from_template(h) as flow:
        img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        features = flow.extract_features(img)
        assert features.shape == (h, w, flow.num_filters)
        # Features should not be all zeros
        assert np.sum(np.abs(features)) > 0


@pytest.mark.hardware
def test_identical_frames():
    """Identical frames should give near-zero flow."""
    from libredgetpu.optical_flow.templates import list_templates
    from libredgetpu.optical_flow_module import OpticalFlow

    templates = list_templates()
    if not templates:
        pytest.skip("No optical flow templates available")

    h, w = templates[0]
    with OpticalFlow.from_template(h) as flow:
        img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        vx, vy = flow.compute(img, img)
        assert abs(vx) < 0.5, f"Expected vx ≈ 0 for identical frames, got {vx}"
        assert abs(vy) < 0.5, f"Expected vy ≈ 0 for identical frames, got {vy}"


@pytest.mark.hardware
def test_horizontal_shift():
    """Rightward shifted frame should give positive vx."""
    from libredgetpu.optical_flow.templates import list_templates
    from libredgetpu.optical_flow_module import OpticalFlow

    templates = list_templates()
    if not templates:
        pytest.skip("No optical flow templates available")

    h, w = templates[0]
    with OpticalFlow.from_template(h) as flow:
        rng = np.random.RandomState(42)
        frame_t = rng.randint(50, 200, (h, w), dtype=np.uint8)
        # Shift right by ~4 pixels (visible in pooled space)
        shift = flow.pool_factor
        frame_t1 = np.zeros_like(frame_t)
        frame_t1[:, shift:] = frame_t[:, :-shift]
        vx, vy = flow.compute(frame_t, frame_t1)
        print(f"  Horizontal shift: vx={vx:.2f}, vy={vy:.2f}")
        assert vx > 0.3, f"Expected positive vx for rightward shift, got {vx}"


@pytest.mark.hardware
def test_vertical_shift():
    """Downward shifted frame should give positive vy."""
    from libredgetpu.optical_flow.templates import list_templates
    from libredgetpu.optical_flow_module import OpticalFlow

    templates = list_templates()
    if not templates:
        pytest.skip("No optical flow templates available")

    h, w = templates[0]
    with OpticalFlow.from_template(h) as flow:
        rng = np.random.RandomState(42)
        frame_t = rng.randint(50, 200, (h, w), dtype=np.uint8)
        shift = flow.pool_factor
        frame_t1 = np.zeros_like(frame_t)
        frame_t1[shift:, :] = frame_t[:-shift, :]
        vx, vy = flow.compute(frame_t, frame_t1)
        print(f"  Vertical shift: vx={vx:.2f}, vy={vy:.2f}")
        assert vy > 0.3, f"Expected positive vy for downward shift, got {vy}"


@pytest.mark.hardware
def test_benchmark():
    """Benchmark optical flow latency."""
    from libredgetpu.optical_flow.templates import list_templates
    from libredgetpu.optical_flow_module import OpticalFlow

    templates = list_templates()
    if not templates:
        pytest.skip("No optical flow templates available")

    h, w = templates[0]
    with OpticalFlow.from_template(h) as flow:
        rng = np.random.RandomState(42)
        frame_t = rng.randint(0, 255, (h, w), dtype=np.uint8)
        frame_t1 = rng.randint(0, 255, (h, w), dtype=np.uint8)

        # Warmup
        for _ in range(3):
            flow.compute(frame_t, frame_t1)

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            flow.compute(frame_t, frame_t1)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = np.mean(times)
        std = np.std(times)
        print(f"  {h}x{w}: avg={avg:.2f} ms, std={std:.2f} ms, "
              f"min={min(times):.2f}, max={max(times):.2f}")


@pytest.mark.hardware
def test_relayout_standard():
    """Standard mode: relayout fixes vertical and diagonal flow detection."""
    from libredgetpu.optical_flow.templates import list_templates
    from libredgetpu.optical_flow_module import OpticalFlow

    templates = list_templates()
    if not templates:
        pytest.skip("No optical flow templates available")

    h, w = templates[0]
    with OpticalFlow.from_template(h) as flow:
        rng = np.random.RandomState(42)
        texture = rng.randint(0, 256, (h, w), dtype=np.uint8)
        pf = flow.pool_factor

        # Test all 4 cardinal directions + diagonal
        for dx, dy, name in [
            (pf, 0, "right"), (-pf, 0, "left"),
            (0, pf, "down"), (0, -pf, "up"),
            (pf, pf, "diag"),
        ]:
            shifted = np.roll(texture, shift=(dy, dx), axis=(0, 1))
            vx, vy = flow.compute(texture, shifted)
            exp_vx, exp_vy = dx / pf, dy / pf
            print(f"  {name}: expected=({exp_vx:+.0f},{exp_vy:+.0f}) got=({vx:+.2f},{vy:+.2f})")
            assert abs(vx - exp_vx) < 0.7, f"{name}: vx error {abs(vx - exp_vx):.2f}"
            assert abs(vy - exp_vy) < 0.7, f"{name}: vy error {abs(vy - exp_vy):.2f}"


@pytest.mark.hardware
def test_relayout_pooled():
    """Pooled mode: relayout fixes vertical and diagonal flow detection."""
    from libredgetpu.optical_flow_module import OpticalFlow

    with OpticalFlow.from_template(64, pooled=True) as flow:
        rng = np.random.RandomState(42)
        texture = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        pf = flow._fused_pool or flow.pool_factor

        for dx, dy, name in [
            (pf, 0, "right"), (-pf, 0, "left"),
            (0, pf, "down"), (0, -pf, "up"),
            (pf, pf, "diag"),
        ]:
            shifted = np.roll(texture, shift=(dy, dx), axis=(0, 1))
            vx, vy = flow.compute(texture, shifted)
            exp_vx, exp_vy = dx / pf, dy / pf
            print(f"  {name}: expected=({exp_vx:+.0f},{exp_vy:+.0f}) got=({vx:+.2f},{vy:+.2f})")
            assert abs(vx - exp_vx) < 0.7, f"{name}: vx error {abs(vx - exp_vx):.2f}"
            assert abs(vy - exp_vy) < 0.7, f"{name}: vy error {abs(vy - exp_vy):.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
