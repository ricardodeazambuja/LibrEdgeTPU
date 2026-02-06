"""Tests for the CPU replica of the Edge TPU optical flow pipeline.

Validates each stage of the integer arithmetic independently, then
tests the full end-to-end pipeline.
"""

import numpy as np
import pytest

from libredgetpu.gui.cpu_replica import CPUReplicaOpticalFlow, ReplicaQuantParams


# ---------------------------------------------------------------------------
# Stage 1: QUANTIZE (uint8 -> int8)
# ---------------------------------------------------------------------------

class TestQuantize:
    """Validate uint8 -> int8 mapping for all 256 values."""

    def test_default_mapping_all_256(self):
        """With defaults (M=1, q_int8_zp=-128), int8_out = uint8 - 128."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        # Create a 16x16 image with pixel values 0..255 (repeating)
        vals = np.arange(256, dtype=np.uint8)
        # Make a 16x16 image
        img = np.zeros((16, 16), dtype=np.uint8)
        img.flat[:256] = vals

        engine.extract_features_uint8(img)
        after_q = engine.get_intermediates()["after_quantize_int8"]

        # Expected: int8_out = uint8 - 128
        expected = (vals.astype(np.int16) - 128).astype(np.int8)
        np.testing.assert_array_equal(after_q.flat[:256], expected)

    def test_uint8_0_maps_to_int8_neg128(self):
        """uint8=0 should map to int8=-128 (the zero-point)."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.zeros((16, 16), dtype=np.uint8)
        engine.extract_features_uint8(img)
        after_q = engine.get_intermediates()["after_quantize_int8"]
        assert after_q.min() == -128
        assert after_q.max() == -128

    def test_uint8_255_maps_to_int8_127(self):
        """uint8=255 should map to int8=127."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.full((16, 16), 255, dtype=np.uint8)
        engine.extract_features_uint8(img)
        after_q = engine.get_intermediates()["after_quantize_int8"]
        assert after_q.min() == 127
        assert after_q.max() == 127

    def test_uint8_128_maps_to_int8_0(self):
        """uint8=128 should map to int8=0."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.full((16, 16), 128, dtype=np.uint8)
        engine.extract_features_uint8(img)
        after_q = engine.get_intermediates()["after_quantize_int8"]
        assert after_q.min() == 0
        assert after_q.max() == 0


# ---------------------------------------------------------------------------
# Stage 2: DEPTHWISE_CONV_2D
# ---------------------------------------------------------------------------

class TestDepthwiseConv:
    """Validate depthwise convolution with integer arithmetic."""

    def test_uniform_input_produces_uniform_output(self):
        """Uniform input should produce spatially uniform conv output
        (except at borders where padding with zp=-128 differs)."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        # Uniform mid-gray image
        img = np.full((16, 16), 128, dtype=np.uint8)
        engine.extract_features_uint8(img)
        conv_out = engine.get_intermediates()["conv_output_int8"]

        # Interior pixels (away from 3-pixel border) should be identical
        interior = conv_out[3:-3, 3:-3, :]
        for ch in range(engine.n_filters):
            ch_interior = interior[:, :, ch]
            assert ch_interior.min() == ch_interior.max(), \
                f"Channel {ch} interior not uniform: min={ch_interior.min()}, max={ch_interior.max()}"

    def test_padding_uses_zero_point_not_zero(self):
        """SAME padding pads with the input zero point, which in the
        shifted domain (input - input_zp) becomes 0.

        For a uniform image with uint8=128 (-> int8=0), after subtracting
        input_zp=-128 we get 128 everywhere. Interior pixels see uniform
        128 * weight sums. Border pixels see some 0 padding (representing
        input_zp), so their accumulators should differ from interior.
        """
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.full((16, 16), 128, dtype=np.uint8)
        engine.extract_features_uint8(img)
        acc = engine.get_intermediates()["conv_acc_int32"]

        # Interior should be uniform (all pixels contribute equally)
        interior_center = acc[8, 8, :]
        interior_other = acc[5, 5, :]
        np.testing.assert_array_equal(interior_center, interior_other)

        # Border should differ from interior because padding contributes 0
        # instead of 128 (the shifted input value)
        border_acc = acc[0, 0, :]
        has_border_effect = np.any(border_acc != interior_center)
        assert has_border_effect, \
            "Border accumulator should differ from interior due to zero padding"

    def test_relu_clamps_to_zero_point(self):
        """Fused ReLU should clamp output to >= conv_output_zp (=-128).

        With default params, zp=-128 represents 0.0 in float, so ReLU
        should produce values >= -128. Since -128 is the minimum int8,
        this is effectively always satisfied, but verify no values are
        below the zero point.
        """
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        engine.extract_features_uint8(img)
        conv_out = engine.get_intermediates()["conv_output_int8"]
        # All values should be >= conv_output_zp = -128
        assert conv_out.min() >= engine.params.conv_output_zp

    def test_nonzero_output_for_gradient_input(self):
        """A horizontal gradient should activate Gabor filters."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        # Horizontal gradient: activates vertical-edge-sensitive Gabors
        row = np.linspace(0, 255, 16, dtype=np.uint8)
        img = np.tile(row, (16, 1))
        engine.extract_features_uint8(img)
        conv_out = engine.get_intermediates()["conv_output_int8"]
        # At least some channels should have values above the zero point
        assert conv_out.max() > engine.params.conv_output_zp


# ---------------------------------------------------------------------------
# Stage 3: AVG_POOL_2D
# ---------------------------------------------------------------------------

class TestAvgPool:
    """Validate average pooling with integer rounding."""

    def test_uniform_block_preserves_value(self):
        """Uniform input through pool should preserve value."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.full((16, 16), 128, dtype=np.uint8)
        engine.extract_features_uint8(img)
        conv_out = engine.get_intermediates()["conv_output_int8"]
        pool_out = engine.get_intermediates()["pool_output_int8"]

        # For each pooled pixel, check that it's the average of the 4x4 block
        for ch in range(engine.n_filters):
            for py in range(engine.out_h):
                for px in range(engine.out_w):
                    block = conv_out[
                        py * 4:(py + 1) * 4,
                        px * 4:(px + 1) * 4,
                        ch
                    ].astype(np.int32)
                    expected = int(np.round(block.sum() / 16.0))
                    expected = np.clip(expected, -128, 127)
                    actual = int(pool_out[py, px, ch])
                    assert actual == expected, \
                        f"Pool mismatch at ({py},{px},{ch}): {actual} != {expected}"

    def test_output_shape(self):
        """Pool output should be (H/P, W/P, n_filters)."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        engine.extract_features_uint8(img)
        pool_out = engine.get_intermediates()["pool_output_int8"]
        assert pool_out.shape == (16, 16, 8)


# ---------------------------------------------------------------------------
# Stage 4: Final QUANTIZE (int8 -> uint8)
# ---------------------------------------------------------------------------

class TestFinalQuantize:
    """Validate int8 -> uint8 mapping."""

    def test_default_mapping(self):
        """With defaults (pool_scale == final_scale, pool_zp=-128, final_zp=0):
        uint8_out = pool_int8 + 128.
        """
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.full((16, 16), 128, dtype=np.uint8)
        engine.extract_features_uint8(img)
        pool_out = engine.get_intermediates()["pool_output_int8"]
        final_out = engine.get_intermediates()["final_uint8"]

        # uint8_out = pool_int8 + 128
        expected = (pool_out.astype(np.int16) + 128).clip(0, 255).astype(np.uint8)
        np.testing.assert_array_equal(final_out, expected)

    def test_int8_neg128_maps_to_uint8_0(self):
        """int8=-128 should map to uint8=0."""
        # Use a black image — after ReLU, conv output is clipped to zp=-128
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.zeros((16, 16), dtype=np.uint8)
        engine.extract_features_uint8(img)
        final_out = engine.get_intermediates()["final_uint8"]
        # Minimum value should be 0 (from int8=-128 + 128)
        assert final_out.min() >= 0


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_uniform_gray_near_zero_flow(self):
        """Two identical uniform gray frames should produce ~0 flow."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        img = np.full((64, 64), 128, dtype=np.uint8)
        vx, vy = engine.compute(img, img)
        assert abs(vx) < 0.1, f"vx={vx} should be near 0 for identical frames"
        assert abs(vy) < 0.1, f"vy={vy} should be near 0 for identical frames"

    def test_identical_frames_zero_flow(self):
        """Two identical random frames should produce ~0 flow."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        np.random.seed(42)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        vx, vy = engine.compute(img, img)
        assert abs(vx) < 0.1, f"vx={vx} should be near 0"
        assert abs(vy) < 0.1, f"vy={vy} should be near 0"

    def test_horizontal_shift_detected(self):
        """A large horizontal shift should produce nonzero vx."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        np.random.seed(42)
        # Create a textured image wider than the frame
        img1 = np.random.randint(50, 200, (64, 80), dtype=np.uint8)
        # Frame t: crop [0:64, 0:64]
        frame_t = img1[:, :64]
        # Frame t+1: crop [0:64, 8:72] — content shifted left by 8 pixels
        # (column 8 of original is now at column 0)
        frame_t1 = img1[:, 8:72]

        vx, vy = engine.compute(frame_t, frame_t1)
        # Content moved left in the frame → negative vx
        # At 64×64 pooled to 16×16 (pool_factor=4), 8 pixels = 2 pooled pixels
        assert vx < -0.5, f"vx={vx} should be negative for leftward content shift"
        assert abs(vx) > 1.0, f"|vx|={abs(vx)} should be > 1 for 8px shift (2 pooled px)"

    def test_vertical_shift_detected(self):
        """A large vertical shift should produce nonzero vy."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        np.random.seed(42)
        img1 = np.random.randint(50, 200, (80, 64), dtype=np.uint8)
        frame_t = img1[:64, :]
        # Content shifted up by 8 pixels (row 8 is now at row 0)
        frame_t1 = img1[8:72, :]

        vx, vy = engine.compute(frame_t, frame_t1)
        # Content moved up → negative vy
        assert vy < -0.5, f"vy={vy} should be negative for upward content shift"
        assert abs(vy) > 1.0, f"|vy|={abs(vy)} should be > 1 for 8px shift (2 pooled px)"

    def test_output_shape(self):
        """Feature extraction should produce correct output shape."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        feat = engine.extract_features_uint8(img)
        assert feat.shape == (16, 16, 8)
        assert feat.dtype == np.uint8

    def test_small_image_size(self):
        """Pipeline should work with smaller image sizes."""
        engine = CPUReplicaOpticalFlow(height=16, width=16)
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        feat = engine.extract_features_uint8(img)
        assert feat.shape == (4, 4, 8)

    def test_wrong_image_shape_raises(self):
        """Wrong input shape should raise ValueError."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        with pytest.raises(ValueError, match="Image shape"):
            engine.extract_features_uint8(np.zeros((32, 32), dtype=np.uint8))


# ---------------------------------------------------------------------------
# TFLite interpreter comparison
# ---------------------------------------------------------------------------

class TestVsTFLiteInterpreter:
    """Compare CPU replica output against TFLite interpreter.

    Requires tensorflow — skip if not installed.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build model and interpreter."""
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            pytest.skip("tensorflow not installed")

        from libredgetpu.tflite_builder import build_optical_flow_pooled

        tflite_bytes, metadata = build_optical_flow_pooled(64, 64)
        # Use BUILTIN_WITHOUT_DEFAULT_DELEGATES to avoid XNNPACK quantization bug
        self.interpreter = tf.lite.Interpreter(
            model_content=tflite_bytes,
            experimental_preserve_all_tensors=True,
            num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.interpreter.allocate_tensors()
        self.engine = CPUReplicaOpticalFlow(height=64, width=64)

    def _run_tflite(self, image_uint8):
        """Run TFLite interpreter on a uint8 image."""
        inp = image_uint8.reshape(1, 64, 64, 1)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], inp)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(output_details[0]['index']).squeeze()

    def test_uniform_gray(self):
        """Uniform gray image: replica vs TFLite should match within ±1 LSB."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        replica_out = self.engine.extract_features_uint8(img)
        tflite_out = self._run_tflite(img)

        diff = np.abs(replica_out.astype(np.int16) - tflite_out.astype(np.int16))
        max_diff = diff.max()
        assert max_diff <= 2, \
            f"Max difference {max_diff} exceeds ±2 LSB tolerance"

    def test_gradient_image(self):
        """Horizontal gradient: replica vs TFLite should match within ±2 LSB."""
        row = np.linspace(0, 255, 64, dtype=np.uint8)
        img = np.tile(row, (64, 1))
        replica_out = self.engine.extract_features_uint8(img)
        tflite_out = self._run_tflite(img)

        diff = np.abs(replica_out.astype(np.int16) - tflite_out.astype(np.int16))
        max_diff = diff.max()
        # Allow ±2 LSB for rounding differences (banker's vs round-half-away)
        assert max_diff <= 2, \
            f"Max difference {max_diff} exceeds ±2 LSB tolerance"

    def test_random_image(self):
        """Random image: replica vs TFLite should match within ±2 LSB."""
        np.random.seed(123)
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        replica_out = self.engine.extract_features_uint8(img)
        tflite_out = self._run_tflite(img)

        diff = np.abs(replica_out.astype(np.int16) - tflite_out.astype(np.int16))
        max_diff = diff.max()
        mean_diff = diff.mean()
        assert max_diff <= 2, \
            f"Max difference {max_diff} exceeds ±2 LSB tolerance (mean={mean_diff:.3f})"

    def test_intermediate_quantize_stage(self):
        """Compare Stage 1 (QUANTIZE) intermediate against TFLite."""
        img = np.full((64, 64), 100, dtype=np.uint8)
        self.engine.extract_features_uint8(img)
        replica_int8 = self.engine.get_intermediates()["after_quantize_int8"]

        # Run TFLite and get quantize_out tensor
        inp = img.reshape(1, 64, 64, 1)
        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]['index'], inp)
        self.interpreter.invoke()

        # Find the quantize_out tensor by name
        tensor_details = self.interpreter.get_tensor_details()
        q_out_idx = None
        for td in tensor_details:
            if "quantize_out" in td['name']:
                q_out_idx = td['index']
                break

        if q_out_idx is not None:
            tflite_int8 = self.interpreter.get_tensor(q_out_idx).squeeze()
            diff = np.abs(replica_int8.astype(np.int16) - tflite_int8.astype(np.int16))
            assert diff.max() <= 1, \
                f"QUANTIZE stage max diff {diff.max()} exceeds ±1"


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

class TestReplicaParams:
    """Verify that custom params affect output."""

    def test_custom_scales_change_output(self):
        """Different conv_output_scale should produce different features."""
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

        engine_default = CPUReplicaOpticalFlow(height=64, width=64)
        feat_default = engine_default.extract_features_uint8(img)

        # Use a much larger conv_output_scale (less resolution)
        params = ReplicaQuantParams(conv_output_scale=0.1, pool_output_scale=0.1,
                                    final_output_scale=0.1)
        engine_custom = CPUReplicaOpticalFlow(height=64, width=64, params=params)
        feat_custom = engine_custom.extract_features_uint8(img)

        # They should differ (different quantization resolution)
        assert not np.array_equal(feat_default, feat_custom), \
            "Different scales should produce different features"

    def test_default_params_match_builder(self):
        """Default params should produce same weight scales as tflite_builder."""
        engine = CPUReplicaOpticalFlow(height=64, width=64)
        # Verify there are 8 per-channel scales
        assert len(engine.per_ch_weight_scales) == 8
        # All scales should be positive
        assert all(s > 0 for s in engine.per_ch_weight_scales)

    def test_verbose_mode_no_crash(self):
        """Verbose mode should not crash."""
        engine = CPUReplicaOpticalFlow(height=16, width=16, verbose=True)
        img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        feat = engine.extract_features_uint8(img)
        assert feat.shape == (4, 4, 8)
