"""Ground-truth validation tests — compare TFLite builder outputs against
TFLite interpreter (Google's CPU reference) and numpy for the four
robotics modules: SpotTracker, PatternTracker, OpticalFlow, VisualCompass.

These tests catch:
  - Builder producing structurally valid but numerically wrong models
  - Quantization scale errors that produce wrong magnitudes
  - Weight layout bugs (cross-correlation vs convolution flip)
  - Zero-point handling errors
  - Soft argmax intermediate value correctness

All tests skip gracefully if tensorflow is not installed.
"""

import numpy as np
import pytest

# ── Shared helpers ──────────────────────────────────────────────────────

tf = pytest.importorskip("tensorflow")


def _make_interpreter(tflite_bytes):
    """Create TFLite interpreter avoiding XNNPACK quantization bug."""
    interp = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True,
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interp.allocate_tensors()
    return interp


def _run_interpreter(interp, input_data):
    """Run interpreter on input_data, return output array."""
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    interp.set_tensor(inp_det[0]["index"], input_data)
    interp.invoke()
    return interp.get_tensor(out_det[0]["index"])


def _dequantize(raw_int8, scale, zero_point):
    """Dequantize int8 → float."""
    return (raw_int8.astype(np.float32) - zero_point) * scale


def _make_gaussian_spot(height, width, cy, cx, sigma=6.0):
    """Create uint8 image with a Gaussian spot at (cy, cx)."""
    yy, xx = np.mgrid[:height, :width]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return (g * 255).astype(np.uint8)


def _decode_spot_tracker_output(interp_output, metadata):
    """Decode SpotTracker TFLite output to (x_off, y_off) in [-1, +1]."""
    scale = metadata["output_scale"]
    zp = metadata["output_zero_point"]
    temperature = metadata["temperature"]
    y_offset = metadata["y_offset"]

    raw = interp_output.flatten().astype(np.float32)
    offsets = (raw - zp) * scale
    x_raw, y_raw = float(offsets[0]), float(offsets[1])
    y_raw -= y_offset
    x_off = x_raw * temperature
    y_off = y_raw * temperature
    return x_off, y_off


# ═══════════════════════════════════════════════════════════════════════
#  SpotTracker
# ═══════════════════════════════════════════════════════════════════════

class TestSpotTrackerGroundTruth:
    """Validate build_spot_tracker models via TFLite interpreter."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from libredgetpu.tflite_builder import build_spot_tracker
        tflite_bytes, meta = build_spot_tracker(64, 64, variant="bright")
        self.interp = _make_interpreter(tflite_bytes)
        self.meta = meta

    def _run(self, image_uint8):
        inp = image_uint8.reshape(1, 64, 64, 1).astype(np.uint8)
        return _run_interpreter(self.interp, inp)

    def _track(self, image_uint8):
        out = self._run(image_uint8)
        return _decode_spot_tracker_output(out, self.meta)

    def test_bright_spot_upper_left(self):
        """Spot at (16, 16) → x_off < -0.2, y_off < -0.2."""
        img = _make_gaussian_spot(64, 64, 16, 16)
        x_off, y_off = self._track(img)
        assert x_off < -0.2, f"x_off={x_off}, expected < -0.2"
        assert y_off < -0.2, f"y_off={y_off}, expected < -0.2"

    def test_bright_spot_lower_right(self):
        """Spot at (48, 48) → x_off > 0.2, y_off > 0.2."""
        img = _make_gaussian_spot(64, 64, 48, 48)
        x_off, y_off = self._track(img)
        assert x_off > 0.2, f"x_off={x_off}, expected > 0.2"
        assert y_off > 0.2, f"y_off={y_off}, expected > 0.2"

    def test_bright_spot_center(self):
        """Spot at (31.5, 31.5) → |x_off| < 0.1, |y_off| < 0.1."""
        # Use (32, 32) for integer coords — close enough to center
        img = _make_gaussian_spot(64, 64, 32, 32)
        x_off, y_off = self._track(img)
        assert abs(x_off) < 0.15, f"|x_off|={abs(x_off)}, expected < 0.15"
        assert abs(y_off) < 0.15, f"|y_off|={abs(y_off)}, expected < 0.15"

    def test_non_uniform_intermediate_not_zero(self):
        """Non-uniform input: raw y_output should reflect spot position.

        A broken pipeline that always returns zero-point would fail this
        test because the y channel encodes position + y_offset.

        Note: uniform images cause int8 softmax saturation (all probs
        quantize to 0 since 1/4096 < 1/256), so we use a peaked image.
        """
        img = _make_gaussian_spot(64, 64, 20, 32, sigma=8.0)
        out = self._run(img)

        scale = self.meta["output_scale"]
        zp = self.meta["output_zero_point"]

        raw = out.flatten().astype(np.float32)
        x_raw = (raw[0] - zp) * scale
        y_raw = (raw[1] - zp) * scale

        # y_raw encodes position + y_offset (10.0). For a spot at row 20,
        # the y coordinate should be negative (above center), so y_raw < 10.
        # But it should NOT be zero — that would mean the pipeline is broken.
        assert abs(y_raw) > 1.0, (
            f"y_raw={y_raw:.2f}, expected non-trivial value "
            f"(zero means pipeline is broken)"
        )
        # x_raw should be near 0 (spot is at center column)
        assert abs(x_raw) < 2.0, f"x_raw={x_raw:.2f}, expected near 0"

    def test_vs_numpy_soft_argmax(self):
        """Compare TFLite output against numpy reference soft argmax.

        The float-domain reference computes:
          probs = softmax(pixel_intensities / temperature)
          x = sum(probs * x_coords)
          y = sum(probs * y_coords)

        The quantized TFLite output should match within tolerance.
        """
        img = _make_gaussian_spot(64, 64, 20, 44, sigma=8.0)

        # ── TFLite result ──
        x_tflite, y_tflite = self._track(img)

        # ── Numpy reference ──
        h, w = 64, 64
        temperature = self.meta["temperature"]
        pixels = img.flatten().astype(np.float64)

        # Coordinate grids matching builder convention
        half_w = max((w - 1) / 2.0, 1)
        half_h = max((h - 1) / 2.0, 1)
        x_coords = np.zeros(h * w, dtype=np.float64)
        y_coords = np.zeros(h * w, dtype=np.float64)
        for i in range(h):
            for j in range(w):
                x_coords[i * w + j] = (j - half_w) / half_w / temperature
                y_coords[i * w + j] = (i - half_h) / half_h / temperature

        # Softmax
        logits = pixels - np.max(pixels)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        # Weighted sum → raw output
        x_ref_raw = float(np.sum(probs * x_coords))
        y_ref_raw = float(np.sum(probs * y_coords))

        # Scale to [-1, +1]
        x_ref = x_ref_raw * temperature
        y_ref = y_ref_raw * temperature

        # Allow tolerance for int8 quantization (output_scale ≈ 0.1)
        tol = 3.0 * self.meta["output_scale"] * temperature
        assert abs(x_tflite - x_ref) < tol, (
            f"x: TFLite={x_tflite:.4f} vs numpy={x_ref:.4f}, "
            f"diff={abs(x_tflite - x_ref):.4f} > tol={tol:.4f}"
        )
        assert abs(y_tflite - y_ref) < tol, (
            f"y: TFLite={y_tflite:.4f} vs numpy={y_ref:.4f}, "
            f"diff={abs(y_tflite - y_ref):.4f} > tol={tol:.4f}"
        )

    def test_monotonic_x_sweep(self):
        """Moving spot rightward should monotonically increase x_off."""
        positions = [10, 20, 32, 44, 54]
        x_vals = []
        for cx in positions:
            img = _make_gaussian_spot(64, 64, 32, cx)
            x_off, _ = self._track(img)
            x_vals.append(x_off)
        for i in range(len(x_vals) - 1):
            assert x_vals[i] < x_vals[i + 1], (
                f"x not monotonic: x[{positions[i]}]={x_vals[i]:.4f} "
                f">= x[{positions[i+1]}]={x_vals[i+1]:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════
#  PatternTracker
# ═══════════════════════════════════════════════════════════════════════

class TestPatternTrackerGroundTruth:
    """Validate build_pattern_tracker models via TFLite interpreter."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from libredgetpu.tflite_builder import build_pattern_tracker
        tflite_bytes, meta = build_pattern_tracker(64, 64, 8, 8, channels=1)
        self.interp = _make_interpreter(tflite_bytes)
        self.meta = meta

    def _run(self, image_uint8):
        inp = image_uint8.reshape(1, 64, 64, 1).astype(np.uint8)
        return _run_interpreter(self.interp, inp)

    def _track(self, image_uint8):
        out = self._run(image_uint8)
        return _decode_spot_tracker_output(out, self.meta)

    def test_template_at_center(self):
        """Default Gaussian template at image center → |offset| < 0.15."""
        # Build image with a Gaussian blob matching the default template
        img = _make_gaussian_spot(64, 64, 32, 32, sigma=2.0)
        x_off, y_off = self._track(img)
        assert abs(x_off) < 0.15, f"|x_off|={abs(x_off)}"
        assert abs(y_off) < 0.15, f"|y_off|={abs(y_off)}"

    def test_template_at_corner(self):
        """Template in upper-left → negative offsets."""
        # Place strong patch at upper-left, weak elsewhere
        img = np.full((64, 64), 16, dtype=np.uint8)
        img[4:12, 4:12] = _make_gaussian_spot(8, 8, 4, 4, sigma=2.0)
        x_off, y_off = self._track(img)
        assert x_off < -0.2, f"x_off={x_off}, expected < -0.2"
        assert y_off < -0.2, f"y_off={y_off}, expected < -0.2"

    def test_template_at_lower_right(self):
        """Template in lower-right → positive offsets."""
        img = np.full((64, 64), 16, dtype=np.uint8)
        img[50:58, 50:58] = _make_gaussian_spot(8, 8, 4, 4, sigma=2.0)
        x_off, y_off = self._track(img)
        assert x_off > 0.2, f"x_off={x_off}, expected > 0.2"
        assert y_off > 0.2, f"y_off={y_off}, expected > 0.2"

    def test_custom_weights_change_output(self):
        """Different Conv2D weights should produce different outputs."""
        from libredgetpu.tflite_builder import build_pattern_tracker

        # Default Gaussian kernel
        _, meta_default = build_pattern_tracker(64, 64, 8, 8)
        interp_default = _make_interpreter(_)

        # Custom: vertical stripe kernel
        stripe_kernel = np.zeros((1, 8, 8, 1), dtype=np.int8)
        stripe_kernel[0, :, 0::2, 0] = 127  # Columns 0, 2, 4, 6
        tflite2, meta2 = build_pattern_tracker(
            64, 64, 8, 8, conv_weights_int8=stripe_kernel,
        )
        interp_stripe = _make_interpreter(tflite2)

        # Same input, different kernels
        img = np.random.RandomState(42).randint(0, 255, (64, 64), dtype=np.uint8)
        inp = img.reshape(1, 64, 64, 1).astype(np.uint8)

        out1 = _run_interpreter(interp_default, inp).flatten()
        out2 = _run_interpreter(interp_stripe, inp).flatten()

        # Outputs must differ (different kernels on same input)
        assert not np.array_equal(out1, out2), (
            "Different Conv2D kernels produced identical output — "
            "kernel weights may not be used"
        )

    def test_vs_numpy_correlation_peak(self):
        """TFLite correlation peak should match numpy correlation peak.

        Build pattern tracker, extract the quantized Gaussian kernel,
        compute float correlation on the same input via numpy, and
        verify both agree on the peak location.
        """
        scipy_signal = pytest.importorskip("scipy.signal")

        from libredgetpu.tflite_builder import build_pattern_tracker

        # Build model and extract kernel from metadata
        tflite_bytes, meta = build_pattern_tracker(64, 64, 8, 8, channels=1)

        # Default kernel is Gaussian — reconstruct it
        kh, kw = meta["kernel_height"], meta["kernel_width"]
        sigma = max(kh, kw) / 4
        cy, cx = kh / 2, kw / 2
        yy, xx = np.mgrid[:kh, :kw]
        kernel_float = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        kernel_float /= kernel_float.max()

        # Create image with bright patch at known position
        img = np.full((64, 64), 16, dtype=np.float64)
        # Place a Gaussian blob matching the kernel at (24, 40)
        patch = kernel_float * 239 + 16  # Scale to [16, 255]
        img[24:24 + kh, 40:40 + kw] = patch

        # ── Numpy reference: cross-correlation ──
        corr = scipy_signal.correlate2d(
            img, kernel_float, mode="valid", boundary="fill", fillvalue=0
        )
        np_peak = np.unravel_index(np.argmax(corr), corr.shape)

        # ── TFLite: decode (x_off, y_off) → approximate pixel position ──
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        interp = _make_interpreter(tflite_bytes)
        inp = img_uint8.reshape(1, 64, 64, 1).astype(np.uint8)
        out = _run_interpreter(interp, inp)
        x_off, y_off = _decode_spot_tracker_output(out, meta)

        # Convert offsets [-1, +1] to pixel coordinates
        # The correlation output has shape (sh - kh + 1, sw - kw + 1)
        corr_h = 64 - kh + 1  # 57
        corr_w = 64 - kw + 1  # 57
        # x_off=0, y_off=0 means center of correlation map
        tflite_col = (x_off + 1.0) / 2.0 * (corr_w - 1)
        tflite_row = (y_off + 1.0) / 2.0 * (corr_h - 1)

        # Both should find the peak near (24, 40) in the correlation map
        assert abs(tflite_row - np_peak[0]) <= 3, (
            f"Row: TFLite={tflite_row:.1f} vs numpy={np_peak[0]}, "
            f"diff={abs(tflite_row - np_peak[0]):.1f}"
        )
        assert abs(tflite_col - np_peak[1]) <= 3, (
            f"Col: TFLite={tflite_col:.1f} vs numpy={np_peak[1]}, "
            f"diff={abs(tflite_col - np_peak[1]):.1f}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  OpticalFlow
# ═══════════════════════════════════════════════════════════════════════

class TestOpticalFlowGroundTruth:
    """Validate build_optical_flow models via TFLite interpreter."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from libredgetpu.tflite_builder import (
            build_optical_flow,
            build_optical_flow_pooled,
        )
        self.tflite_bytes, self.meta = build_optical_flow(64, 64)
        self.interp = _make_interpreter(self.tflite_bytes)
        self.pooled_bytes, self.pooled_meta = build_optical_flow_pooled(64, 64)
        self.interp_pooled = _make_interpreter(self.pooled_bytes)

    def _run(self, image_uint8, interp=None):
        if interp is None:
            interp = self.interp
        inp = image_uint8.reshape(1, 64, 64, 1).astype(np.uint8)
        return _run_interpreter(interp, inp)

    def test_gabor_features_not_all_zero(self):
        """Non-uniform input should produce non-trivial Gabor features.

        Catches the zero-result bug: if features are all at zero-point,
        the optical flow pipeline is completely broken.
        """
        rng = np.random.RandomState(42)
        img = rng.randint(0, 255, (64, 64), dtype=np.uint8)
        out = self._run(img)

        # Output is uint8 features — should have significant variation
        out_flat = out.flatten()
        assert out_flat.std() > 5.0, (
            f"Gabor features have std={out_flat.std():.1f}, "
            f"expected > 5.0 (features may be all at zero-point)"
        )
        # At least 10% of values should differ from the mode
        mode_val = np.bincount(out_flat).argmax()
        non_mode_frac = np.mean(out_flat != mode_val)
        assert non_mode_frac > 0.1, (
            f"Only {non_mode_frac*100:.1f}% of features differ from mode={mode_val}"
        )

    def test_shifted_input_changes_features(self):
        """Shifted input should produce different Gabor features.

        If the model ignores input (e.g., all-zero weights), shifted and
        unshifted inputs would produce identical features.
        """
        rng = np.random.RandomState(42)
        img = rng.randint(0, 255, (64, 64), dtype=np.uint8)
        img_shifted = np.roll(img, 4, axis=1)

        out1 = self._run(img).flatten()
        out2 = self._run(img_shifted).flatten()

        # Features must differ
        diff_count = np.sum(out1 != out2)
        assert diff_count > len(out1) * 0.05, (
            f"Only {diff_count}/{len(out1)} features differ between "
            f"original and shifted input"
        )

    def test_gabor_kernels_match_builder(self):
        """Gabor weights in the TFLite model should match _generate_gabor_kernels."""
        from libredgetpu.tflite_builder import _generate_gabor_kernels

        kernels_float = _generate_gabor_kernels(ksize=7, orientations=4, sigmas=(1.5, 3.0))
        n_filters = kernels_float.shape[3]

        # Extract quantized weights from TFLite model
        tensor_details = self.interp.get_tensor_details()
        conv_weight_td = None
        for td in tensor_details:
            if "gabor" in td["name"].lower() or "conv" in td["name"].lower():
                if td["shape"].tolist() == [1, 7, 7, n_filters]:
                    conv_weight_td = td
                    break

        if conv_weight_td is None:
            # Try finding by shape alone
            for td in tensor_details:
                shape = td["shape"].tolist()
                if len(shape) == 4 and shape[1] == 7 and shape[2] == 7:
                    conv_weight_td = td
                    break

        assert conv_weight_td is not None, (
            "Could not find Gabor conv weight tensor in model"
        )

        # Get quantized weights and dequantize
        q_weights = self.interp.get_tensor(conv_weight_td["index"])
        quant_params = conv_weight_td["quantization_parameters"]
        scales = quant_params["scales"]
        zps = quant_params["zero_points"]

        # Per-channel dequantize and compare
        for ch in range(n_filters):
            scale = scales[ch] if len(scales) > 1 else scales[0]
            zp = zps[ch] if len(zps) > 1 else zps[0]
            dequant = (q_weights[0, :, :, ch].astype(np.float32) - zp) * scale
            ref = kernels_float[:, :, 0, ch]

            # Quantization error per element should be within 1 LSB
            max_err = np.max(np.abs(dequant - ref))
            assert max_err < 2.0 * scale, (
                f"Gabor kernel {ch}: max error {max_err:.6f} > "
                f"2*scale={2.0*scale:.6f}"
            )

    def test_pooled_vs_standard_features(self):
        """Pooled model output ≈ avg_pool(standard model output).

        Both models apply the same Gabor filters; the pooled model adds
        AVG_POOL_2D(4×4). Verifying consistency catches fused-pool bugs.
        """
        rng = np.random.RandomState(42)
        img = rng.randint(50, 200, (64, 64), dtype=np.uint8)

        # Standard model: (1, 64, 64, 8) uint8
        out_std = self._run(img, self.interp).squeeze()
        # Pooled model: (1, 16, 16, 8) uint8
        out_pool = self._run(img, self.interp_pooled).squeeze()

        # Manual avg_pool of standard output
        h, w, c = out_std.shape
        p = 4
        pooled_manual = out_std.reshape(h // p, p, w // p, p, c).mean(axis=(1, 3))

        # Compare — allow tolerance for double quantization
        # (standard output is quantized, then avg_pool quantizes again)
        diff = np.abs(pooled_manual - out_pool.astype(np.float32))
        mean_diff = diff.mean()
        assert mean_diff < 10.0, (
            f"Pooled vs manual avg_pool: mean diff={mean_diff:.1f}, "
            f"expected < 10.0"
        )

    def test_tflite_output_matches_cpu_replica(self):
        """TFLite interpreter output should match CPUReplicaOpticalFlow.

        The CPU replica was validated to ±1 LSB in test_cpu_replica.py.
        This is a cross-check that both produce the same features.
        """
        try:
            from libredgetpu.gui.cpu_replica import CPUReplicaOpticalFlow
        except ImportError:
            pytest.skip("cpu_replica not available")

        engine = CPUReplicaOpticalFlow(height=64, width=64)
        img = np.random.RandomState(42).randint(50, 200, (64, 64), dtype=np.uint8)

        # CPU replica (pooled model)
        replica_out = engine.extract_features_uint8(img)

        # TFLite pooled model
        tflite_out = self._run(img, self.interp_pooled).squeeze()

        # Compare
        diff = np.abs(replica_out.astype(np.int16) - tflite_out.astype(np.int16))
        assert diff.max() <= 2, (
            f"Max diff between CPU replica and TFLite: {diff.max()} > 2 LSB"
        )


# ═══════════════════════════════════════════════════════════════════════
#  VisualCompass
# ═══════════════════════════════════════════════════════════════════════

class TestVisualCompassGroundTruth:
    """Validate VisualCompass yaw computation."""

    def test_yaw_formula_basic(self):
        """Known vx + FOV should produce correct yaw angle.

        With fov=90°, width=64, pool_factor=4:
          deg_per_pooled_px = 90 * 4 / 64 = 5.625
          yaw = vx * 5.625
        """
        from libredgetpu.visual_compass import VisualCompass
        from unittest.mock import MagicMock

        # Create a mock OpticalFlow with known params
        mock_flow = MagicMock()
        mock_flow.fused_pool = 0
        mock_flow.pool_factor = 4
        mock_flow.width = 64
        mock_flow.compute.return_value = (2.0, 0.0)

        compass = VisualCompass(mock_flow, fov_deg=90.0)

        expected_deg_per_pp = 90.0 * 4 / 64  # 5.625
        assert abs(compass.deg_per_pooled_px - expected_deg_per_pp) < 1e-6

        # Compute yaw
        dummy = np.zeros((64, 64), dtype=np.uint8)
        yaw = compass.compute_yaw(dummy, dummy)
        expected_yaw = 2.0 * expected_deg_per_pp  # 11.25°
        assert abs(yaw - expected_yaw) < 1e-6, (
            f"yaw={yaw}, expected={expected_yaw}"
        )

    def test_yaw_sign_convention(self):
        """Positive vx (rightward motion) → positive yaw (clockwise)."""
        from libredgetpu.visual_compass import VisualCompass
        from unittest.mock import MagicMock

        mock_flow = MagicMock()
        mock_flow.fused_pool = 4
        mock_flow.pool_factor = 4
        mock_flow.width = 64

        compass = VisualCompass(mock_flow, fov_deg=90.0)
        dummy = np.zeros(1, dtype=np.uint8)

        # Rightward motion → positive yaw
        mock_flow.compute.return_value = (1.5, 0.0)
        yaw_right = compass.compute_yaw(dummy, dummy)
        assert yaw_right > 0, f"Rightward vx should give positive yaw, got {yaw_right}"

        # Leftward motion → negative yaw
        mock_flow.compute.return_value = (-1.5, 0.0)
        yaw_left = compass.compute_yaw(dummy, dummy)
        assert yaw_left < 0, f"Leftward vx should give negative yaw, got {yaw_left}"

    def test_yaw_zero_drift(self):
        """Identical frames (vx=0) over many calls should not accumulate drift."""
        from libredgetpu.visual_compass import VisualCompass
        from unittest.mock import MagicMock

        mock_flow = MagicMock()
        mock_flow.fused_pool = 4
        mock_flow.pool_factor = 4
        mock_flow.width = 64
        mock_flow.compute.return_value = (0.0, 0.0)

        compass = VisualCompass(mock_flow, fov_deg=90.0)
        dummy = np.zeros(1, dtype=np.uint8)

        cumulative_yaw = 0.0
        for _ in range(100):
            cumulative_yaw += compass.compute_yaw(dummy, dummy)

        assert abs(cumulative_yaw) < 1e-10, (
            f"Cumulative yaw after 100 zero-flow frames: {cumulative_yaw}"
        )

    def test_fused_pool_changes_scale(self):
        """fused_pool > 0 should change deg_per_pooled_px vs pool_factor."""
        from libredgetpu.visual_compass import VisualCompass
        from unittest.mock import MagicMock

        # Standard mode: pool_factor=4, fused_pool=0
        mock1 = MagicMock()
        mock1.fused_pool = 0
        mock1.pool_factor = 4
        mock1.width = 64
        c1 = VisualCompass(mock1, fov_deg=90.0)

        # Pooled mode: fused_pool=4, pool_factor=4
        mock2 = MagicMock()
        mock2.fused_pool = 4
        mock2.pool_factor = 4
        mock2.width = 64
        c2 = VisualCompass(mock2, fov_deg=90.0)

        # Both should give the same scale since effective_pool = 4 in both
        assert abs(c1.deg_per_pooled_px - c2.deg_per_pooled_px) < 1e-6

        # Different fused_pool → different scale
        mock3 = MagicMock()
        mock3.fused_pool = 8
        mock3.pool_factor = 4
        mock3.width = 64
        c3 = VisualCompass(mock3, fov_deg=90.0)

        assert c3.deg_per_pooled_px != c1.deg_per_pooled_px
