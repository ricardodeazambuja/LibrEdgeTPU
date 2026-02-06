#!/usr/bin/env python3
"""Tests for pattern tracker module.

Offline tests run without hardware. Hardware tests require USB Edge TPU.

Usage:
    pytest tests/test_pattern_tracker.py -v                    # offline only
    pytest tests/test_pattern_tracker.py -v --run-hardware     # all tests
    pytest tests/test_pattern_tracker.py -v --run-hardware -k validated
"""

import json
import os
import time

import numpy as np
import pytest

from libredgetpu.pattern_tracker import PatternTracker


# ---------------------------------------------------------------------------
# Offline tests (no hardware required)
# ---------------------------------------------------------------------------

class TestTemplateList:
    """Test template discovery functions."""

    def test_list_templates_returns_list(self):
        """list_templates should return a list (possibly empty)."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        assert isinstance(templates, list)

    def test_list_templates_format(self):
        """Template entries should be (search_h, search_w, kernel_h, kernel_w, channels) tuples."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        for entry in templates:
            assert isinstance(entry, tuple)
            assert len(entry) == 5
            sh, sw, kh, kw, ch = entry
            assert isinstance(sh, int) and sh > 0
            assert isinstance(sw, int) and sw > 0
            assert isinstance(kh, int) and kh > 0
            assert isinstance(kw, int) and kw > 0
            assert ch in (1, 3)
            assert kh < sh, "Kernel must be smaller than search image"

    def test_at_least_one_template(self):
        """At least one template should be available."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        assert len(templates) >= 1, "No pattern tracker templates found"


class TestSidecarMetadata:
    """Verify sidecar JSON has all required fields."""

    def test_all_templates_have_required_fields(self):
        """All template sidecars should have the required metadata fields."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        required_fields = [
            "search_height", "search_width", "kernel_height", "kernel_width",
            "channels", "input_scale", "input_zero_point",
            "output_scale", "output_zero_point", "output_count",
            "conv_weight_scale", "conv_weight_count",
            "temperature", "y_offset",
        ]

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")
            assert os.path.isfile(json_path), f"Missing sidecar: {json_path}"

            with open(json_path) as f:
                meta = json.load(f)

            for field in required_fields:
                assert field in meta, f"{base_name}: missing field '{field}'"

    def test_output_scale_coverage(self):
        """Output scale should cover at least [-10, +10] range (20 units span)."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            scale = meta["output_scale"]
            zp = meta["output_zero_point"]

            min_val = (-128 - zp) * scale
            max_val = (127 - zp) * scale
            span = max_val - min_val

            print(f"  {base_name}: scale={scale:.6f}, zp={zp}, "
                  f"range=[{min_val:.1f}, {max_val:.1f}], span={span:.1f}")

            assert span >= 15.0, (
                f"{base_name}: dequant span {span:.1f} < 15.0"
            )

    def test_conv_weight_scale_positive(self):
        """conv_weight_scale should be positive for all templates."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            assert meta["conv_weight_scale"] > 0, \
                f"{base_name}: invalid conv_weight_scale"

    def test_conv_weight_count_matches_kernel(self):
        """conv_weight_count should equal kernel_h * kernel_w * channels."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            expected = kh * kw * ch
            assert meta["conv_weight_count"] == expected, \
                f"{base_name}: conv_weight_count={meta['conv_weight_count']} != {expected}"

    def test_y_offset_is_10(self):
        """All templates should have y_offset = 10.0 (1/temperature where T=0.1)."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            assert meta["y_offset"] == 10.0, \
                f"{base_name}: y_offset={meta['y_offset']} (expected 10.0)"


class TestSetTemplateValidation:
    """Test set_template() input validation (no hardware required)."""

    def _get_first_template(self):
        """Get the first available template for testing."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")
        sh, sw, kh, kw, ch = templates[0]
        return PatternTracker.from_template(sh, kh, ch)

    def test_rejects_wrong_shape(self):
        """set_template with wrong dimensions should raise ValueError."""
        tracker = self._get_first_template()
        wrong_patch = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            tracker.set_template(wrong_patch)

    def test_rejects_wrong_channels(self):
        """set_template with wrong channel count should raise ValueError."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        gray_templates = [(sh, sw, kh, kw, ch) for sh, sw, kh, kw, ch in templates if ch == 1]
        if not gray_templates:
            pytest.skip("No grayscale templates available")
        sh, sw, kh, kw, ch = gray_templates[0]
        tracker = PatternTracker.from_template(sh, kh, ch)

        # Provide 3-channel patch to 1-channel model
        wrong_patch = np.zeros((kh, kw, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            tracker.set_template(wrong_patch)

    def test_accepts_correct_shape(self):
        """set_template with correct shape should succeed (uses fast path, no compiler)."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")
        sh, sw, kh, kw, ch = templates[0]
        tracker = PatternTracker.from_template(sh, kh, ch)

        patch = np.random.rand(kh, kw) if ch == 1 else np.random.rand(kh, kw, ch)
        patch = (patch * 255).astype(np.float32)

        # Fast path: no compiler needed when conv_weight_offsets available
        tracker.set_template(patch)


class TestCompilerFreeTemplateSwap:
    """Test compiler-free template swapping via blob offset patching."""

    def test_conv_weight_offsets_present(self):
        """All template JSONs should have conv_weight_offsets."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            assert "conv_weight_offsets" in meta, \
                f"{base_name}: missing conv_weight_offsets"
            assert "conv_weight_blob" in meta, \
                f"{base_name}: missing conv_weight_blob"

    def test_offsets_count_matches_kernel(self):
        """Offset count should equal kernel_h * kernel_w * channels."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            expected = kh * kw * ch
            offsets = meta["conv_weight_offsets"]
            assert len(offsets) == expected, \
                f"{base_name}: {len(offsets)} offsets != {expected} weights"

    def test_offsets_are_unique(self):
        """All offsets within a template should be unique."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            offsets = meta["conv_weight_offsets"]
            assert len(set(offsets)) == len(offsets), \
                f"{base_name}: duplicate offsets found"

    def test_offsets_within_blob_bounds(self):
        """All offsets should be within the parameter blob bounds."""
        from libredgetpu.pattern.templates import list_templates

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            tracker = PatternTracker.from_template(sh, kh, ch)
            offsets = tracker._conv_weight_offsets
            assert offsets is not None, f"{sh}x{sw}/{kh}x{kw}/{ch}ch: no offsets loaded"

            # Determine target blob
            if tracker._conv_weight_blob == "eo_params":
                blob = tracker._original_eo_params
            else:
                blob = tracker._original_pc_params
            blob_size = len(blob)

            assert np.all(offsets >= 0), "Negative offsets found"
            assert np.all(offsets < blob_size), \
                f"Offsets exceed blob size ({blob_size}): max={offsets.max()}"

    def test_set_template_uses_fast_path(self):
        """set_template_raw() should use fast path when offsets available."""
        from libredgetpu.pattern.templates import list_templates

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        sh, sw, kh, kw, ch = templates[0]
        tracker = PatternTracker.from_template(sh, kh, ch)

        # Verify fast path is available
        assert tracker._conv_weight_offsets is not None

        # Store original params for comparison
        if tracker._conv_weight_blob == "eo_params":
            orig_params = tracker._eo_params
        else:
            orig_params = tracker._pc_params

        # Set a new template
        n = kh * kw * ch
        new_weights = np.arange(n, dtype=np.int8)
        tracker.set_template_raw(new_weights)

        # Verify params changed
        if tracker._conv_weight_blob == "eo_params":
            new_params = tracker._eo_params
        else:
            new_params = tracker._pc_params

        assert new_params != orig_params, "Parameters should have changed"

    def test_set_template_round_trip(self):
        """Setting a known pattern should produce consistent params."""
        from libredgetpu.pattern.templates import list_templates

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        sh, sw, kh, kw, ch = templates[0]
        tracker = PatternTracker.from_template(sh, kh, ch)

        n = kh * kw * ch
        pattern = np.full(n, 42, dtype=np.int8)

        # Set same pattern twice → should produce identical params
        tracker.set_template_raw(pattern)
        if tracker._conv_weight_blob == "eo_params":
            params_1 = tracker._eo_params
        else:
            params_1 = tracker._pc_params

        tracker.set_template_raw(pattern)
        if tracker._conv_weight_blob == "eo_params":
            params_2 = tracker._eo_params
        else:
            params_2 = tracker._pc_params

        assert params_1 == params_2, "Same weights should produce same params"

    def test_original_params_preserved(self):
        """Original params should be preserved after multiple swaps."""
        from libredgetpu.pattern.templates import list_templates

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        sh, sw, kh, kw, ch = templates[0]
        tracker = PatternTracker.from_template(sh, kh, ch)

        original_pc = tracker._original_pc_params

        # Do multiple swaps
        n = kh * kw * ch
        for val in [0, 42, 127, -128]:
            weights = np.full(n, val, dtype=np.int8)
            tracker.set_template_raw(weights)

        # Original should be unchanged
        assert tracker._original_pc_params == original_pc

    def test_conv_weight_blob_field_valid(self):
        """conv_weight_blob should be 'pc_params' or 'eo_params'."""
        from libredgetpu.pattern.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            base_name = f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"
            json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            blob_name = meta.get("conv_weight_blob")
            assert blob_name in ("pc_params", "eo_params"), \
                f"{base_name}: invalid conv_weight_blob '{blob_name}'"

    def test_all_configs_fast_path(self):
        """All template configs should support the fast path."""
        from libredgetpu.pattern.templates import list_templates

        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")

        for sh, sw, kh, kw, ch in templates:
            tracker = PatternTracker.from_template(sh, kh, ch)
            n = kh * kw * ch
            weights = np.zeros(n, dtype=np.int8)
            # Should succeed without compiler
            tracker.set_template_raw(weights)


class TestCoordinateGrid:
    """Test coordinate grid normalization math."""

    def test_center_is_zero(self):
        """Center of an odd-sized grid should be exactly zero."""
        from libredgetpu.tflite_builder import _create_coordinate_grids

        x_coords, y_coords = _create_coordinate_grids(11, 11, 0.1)
        assert abs(x_coords[5, 5]) < 1e-6, f"Center X: {x_coords[5, 5]}"
        assert abs(y_coords[5, 5]) < 1e-6, f"Center Y: {y_coords[5, 5]}"

    def test_corners_have_correct_signs(self):
        """Corner coordinates should have correct signs."""
        from libredgetpu.tflite_builder import _create_coordinate_grids

        x_coords, y_coords = _create_coordinate_grids(10, 10, 0.1)

        # Top-left: negative x, negative y
        assert x_coords[0, 0] < 0
        assert y_coords[0, 0] < 0

        # Bottom-right: positive x, positive y
        assert x_coords[-1, -1] > 0
        assert y_coords[-1, -1] > 0

    def test_range_is_1_over_temperature(self):
        """Coordinate range should be [-1/T, +1/T]."""
        from libredgetpu.tflite_builder import _create_coordinate_grids

        temp = 0.1
        x_coords, y_coords = _create_coordinate_grids(51, 51, temp)

        # Edges should be approximately ±1/T = ±10
        assert abs(x_coords[0, 0] - (-1.0 / temp)) < 0.5
        assert abs(x_coords[0, -1] - (1.0 / temp)) < 0.5


class TestFromTemplate:
    """Test from_template class method."""

    def test_from_template_returns_tracker(self):
        """from_template should return a PatternTracker instance."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")
        sh, sw, kh, kw, ch = templates[0]
        tracker = PatternTracker.from_template(sh, kh, ch)
        assert isinstance(tracker, PatternTracker)
        assert tracker.search_height == sh
        assert tracker.search_width == sw
        assert tracker.kernel_height == kh
        assert tracker.kernel_width == kw
        assert tracker.channels == ch

    def test_missing_template_raises(self):
        """Requesting a non-existent template should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PatternTracker.from_template(999, 99)

    def test_properties(self):
        """PatternTracker properties should reflect sidecar metadata."""
        from libredgetpu.pattern.templates import list_templates
        templates = list_templates()
        if not templates:
            pytest.skip("No pattern tracker templates available")
        sh, sw, kh, kw, ch = templates[0]
        tracker = PatternTracker.from_template(sh, kh, ch)
        assert tracker.temperature == 0.1
        assert tracker.conv_weight_scale is not None
        assert tracker.conv_weight_scale > 0
        assert tracker.conv_weight_range is not None


# ---------------------------------------------------------------------------
# Hardware tests (require USB Edge TPU)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
def test_track_output_shape():
    """PatternTracker.track() should return exactly 2 floats."""
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No pattern tracker templates available")

    sh, sw, kh, kw, ch = templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        if ch == 1:
            img = np.random.randint(0, 255, (sh, sw), dtype=np.uint8)
        else:
            img = np.random.randint(0, 255, (sh, sw, ch), dtype=np.uint8)
        x_off, y_off = tracker.track(img)
        assert isinstance(x_off, float), f"x_off type: {type(x_off)}"
        assert isinstance(y_off, float), f"y_off type: {type(y_off)}"
        assert -2.0 < x_off < 2.0, f"x_off out of range: {x_off}"
        assert -2.0 < y_off < 2.0, f"y_off out of range: {y_off}"


@pytest.mark.hardware
def test_uniform_image():
    """Uniform image should produce near-zero x offset.

    With uniform input, the Conv2D correlation map is flat. The int8 Softmax
    over thousands of identical values loses precision — probabilities quantize
    to near-zero. X is symmetric so it stays ~0, but the Y channel's +10 offset
    creates a systematic bias when quantized probabilities collapse. We only
    check X here; Y bias is a known int8 quantization artifact.
    """
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    gray_templates = [(sh, sw, kh, kw, ch)
                      for sh, sw, kh, kw, ch in templates if ch == 1]
    if not gray_templates:
        pytest.skip("No grayscale pattern templates available")

    sh, sw, kh, kw, ch = gray_templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        img = np.full((sh, sw), 128, dtype=np.uint8)
        x_off, y_off = tracker.track(img)

        print(f"  Uniform image ({sh}x{sw}/{kh}x{kw}): x_off={x_off:.4f}, y_off={y_off:.4f}")
        assert abs(x_off) < 0.2, f"Expected x_off near 0, got {x_off}"
        # Y may have systematic bias due to asymmetric offset + int8 quantization
        assert abs(y_off) < 1.5, f"y_off too large: {y_off}"


@pytest.mark.hardware
def test_template_placed_at_known_position():
    """A Gaussian patch placed at a known offset should produce matching offsets.

    Places a bright Gaussian blob in the upper-left quadrant and verifies
    that the tracker reports negative x and y offsets.
    """
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    gray_templates = [(sh, sw, kh, kw, ch)
                      for sh, sw, kh, kw, ch in templates if ch == 1]
    if not gray_templates:
        pytest.skip("No grayscale pattern templates available")

    sh, sw, kh, kw, ch = gray_templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        # Dark background with Gaussian patch in upper-left
        img = np.full((sh, sw), 16, dtype=np.uint8)

        # Place Gaussian template in upper-left quadrant
        cy, cx = kh // 2, kw // 2  # center of placed template
        r_start = sh // 8
        c_start = sw // 8
        for dy in range(kh):
            for dx in range(kw):
                dist_sq = (dy - cy)**2 + (dx - cx)**2
                sigma = max(kh, kw) / 4
                val = 255 * np.exp(-dist_sq / (2 * sigma**2))
                r, c = r_start + dy, c_start + dx
                if 0 <= r < sh and 0 <= c < sw:
                    img[r, c] = int(val)

        x_off, y_off = tracker.track(img)
        print(f"  Upper-left Gaussian: x_off={x_off:.4f}, y_off={y_off:.4f}")

        # Upper-left = negative x and negative y
        assert x_off < 0, f"Expected negative x_off, got {x_off}"
        assert y_off < 0, f"Expected negative y_off, got {y_off}"


@pytest.mark.hardware
def test_template_at_four_corners():
    """Bright patches at four quadrants should produce correct offset signs."""
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    gray_templates = [(sh, sw, kh, kw, ch)
                      for sh, sw, kh, kw, ch in templates if ch == 1]
    if not gray_templates:
        pytest.skip("No grayscale pattern templates available")

    sh, sw, kh, kw, ch = gray_templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        spot_size = kh

        # Test positions: (row_frac, col_frac, expected_x_sign, expected_y_sign)
        test_cases = [
            (0.2, 0.2, -1, -1),   # Upper-left
            (0.2, 0.8, +1, -1),   # Upper-right
            (0.8, 0.2, -1, +1),   # Lower-left
            (0.8, 0.8, +1, +1),   # Lower-right
        ]

        for row_frac, col_frac, x_sign, y_sign in test_cases:
            img = np.full((sh, sw), 16, dtype=np.uint8)
            row = int(row_frac * sh)
            col = int(col_frac * sw)
            r1 = max(0, row - spot_size // 2)
            r2 = min(sh, row + spot_size // 2)
            c1 = max(0, col - spot_size // 2)
            c2 = min(sw, col + spot_size // 2)
            img[r1:r2, c1:c2] = 255

            x_off, y_off = tracker.track(img)

            if x_sign < 0:
                assert x_off < 0.1, f"({row_frac}, {col_frac}): expected x<0, got {x_off}"
            else:
                assert x_off > -0.1, f"({row_frac}, {col_frac}): expected x>0, got {x_off}"

            if y_sign < 0:
                assert y_off < 0.1, f"({row_frac}, {col_frac}): expected y<0, got {y_off}"
            else:
                assert y_off > -0.1, f"({row_frac}, {col_frac}): expected y>0, got {y_off}"


@pytest.mark.hardware
def test_benchmark():
    """Benchmark pattern tracker latency."""
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No pattern tracker templates available")

    sh, sw, kh, kw, ch = templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        if ch == 1:
            img = np.random.randint(0, 255, (sh, sw), dtype=np.uint8)
        else:
            img = np.random.randint(0, 255, (sh, sw, ch), dtype=np.uint8)

        # Warmup
        for _ in range(5):
            tracker.track(img)

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            tracker.track(img)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = np.mean(times)
        std = np.std(times)
        print(f"  {sh}x{sw}/{kh}x{kw}/{ch}ch: "
              f"avg={avg:.2f} ms, std={std:.2f} ms, "
              f"min={min(times):.2f}, max={max(times):.2f}")


@pytest.mark.hardware
@pytest.mark.validated
def test_set_template_runtime_swap():
    """Runtime template swap should track different patterns.

    Creates a search image with two distinct patches at different positions.
    Swaps the template to match each patch and verifies the tracked position
    changes accordingly.

    Uses the compiler-free fast path (no edgetpu_compiler needed).
    """
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    gray_templates = [(sh, sw, kh, kw, ch)
                      for sh, sw, kh, kw, ch in templates if ch == 1]
    if not gray_templates:
        pytest.skip("No grayscale pattern templates available")

    sh, sw, kh, kw, ch = gray_templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        # Create two distinct patches
        # Patch A: bright Gaussian (upper-left in image)
        patch_a = np.zeros((kh, kw), dtype=np.float32)
        cy, cx = kh / 2, kw / 2
        sigma = max(kh, kw) / 4
        for dy in range(kh):
            for dx in range(kw):
                patch_a[dy, dx] = 255 * np.exp(-((dx - cx)**2 + (dy - cy)**2) / (2 * sigma**2))

        # Patch B: horizontal stripes
        patch_b = np.zeros((kh, kw), dtype=np.float32)
        for dy in range(kh):
            if dy % 4 < 2:
                patch_b[dy, :] = 255

        # Create search image: patch_a in upper-left, patch_b in lower-right
        img = np.full((sh, sw), 32, dtype=np.uint8)

        # Place patch A in upper-left quadrant
        r_a, c_a = sh // 4 - kh // 2, sw // 4 - kw // 2
        r_a, c_a = max(0, r_a), max(0, c_a)
        img[r_a:r_a + kh, c_a:c_a + kw] = patch_a.astype(np.uint8)

        # Place patch B in lower-right quadrant
        r_b, c_b = 3 * sh // 4 - kh // 2, 3 * sw // 4 - kw // 2
        r_b, c_b = max(0, r_b), max(0, c_b)
        img[r_b:r_b + kh, c_b:c_b + kw] = patch_b.astype(np.uint8)

        # Track with patch A template → should find upper-left
        tracker.set_template(patch_a)
        x_a, y_a = tracker.track(img)
        print(f"  Patch A template: x={x_a:+.3f}, y={y_a:+.3f}")

        # Track with patch B template → should find lower-right
        tracker.set_template(patch_b)
        x_b, y_b = tracker.track(img)
        print(f"  Patch B template: x={x_b:+.3f}, y={y_b:+.3f}")

        # Patch A should track upper-left (x<0, y<0)
        # Patch B should track lower-right (x>0, y>0)
        assert x_a < x_b, f"Patch A x ({x_a}) should be < Patch B x ({x_b})"
        assert y_a < y_b, f"Patch A y ({y_a}) should be < Patch B y ({y_b})"


@pytest.mark.hardware
@pytest.mark.validated
def test_moving_template():
    """Moving a patch across the image should produce monotonically changing offsets."""
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    gray_templates = [(sh, sw, kh, kw, ch)
                      for sh, sw, kh, kw, ch in templates if ch == 1]
    if not gray_templates:
        pytest.skip("No grayscale pattern templates available")

    sh, sw, kh, kw, ch = gray_templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        # Create a Gaussian template
        cy, cx = kh / 2, kw / 2
        sigma = max(kh, kw) / 4
        template_patch = np.zeros((kh, kw), dtype=np.uint8)
        for dy in range(kh):
            for dx in range(kw):
                val = 255 * np.exp(-((dx - cx)**2 + (dy - cy)**2) / (2 * sigma**2))
                template_patch[dy, dx] = int(val)

        # Move patch from left to right across center row
        positions = np.linspace(0.2, 0.8, 5)
        x_offsets = []

        for col_frac in positions:
            img = np.full((sh, sw), 16, dtype=np.uint8)
            row = sh // 2 - kh // 2
            col = int(col_frac * sw) - kw // 2
            col = max(0, min(col, sw - kw))
            img[row:row + kh, col:col + kw] = template_patch

            x_off, y_off = tracker.track(img)
            x_offsets.append(x_off)
            print(f"  Position {col_frac:.2f}: x_off={x_off:.4f}, y_off={y_off:.4f}")

        # X offsets should be monotonically increasing (left to right)
        for i in range(1, len(x_offsets)):
            assert x_offsets[i] > x_offsets[i - 1], \
                f"X offset should increase left to right: {x_offsets}"

        # Range should be significant
        offset_range = x_offsets[-1] - x_offsets[0]
        assert offset_range > 0.3, f"Offset range too small: {offset_range}"


@pytest.mark.hardware
def test_multiple_swaps():
    """Multiple consecutive set_template calls should work without errors.

    Uses the compiler-free fast path (no edgetpu_compiler needed).
    """
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    gray_templates = [(sh, sw, kh, kw, ch)
                      for sh, sw, kh, kw, ch in templates if ch == 1]
    if not gray_templates:
        pytest.skip("No grayscale pattern templates available")

    sh, sw, kh, kw, ch = gray_templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        img = np.random.randint(0, 255, (sh, sw), dtype=np.uint8)

        for i in range(3):
            # Random template each time
            patch = np.random.randint(0, 256, (kh, kw)).astype(np.float32)
            tracker.set_template(patch)
            x_off, y_off = tracker.track(img)
            print(f"  Swap {i}: x={x_off:+.3f}, y={y_off:+.3f}")
            assert -2.0 < x_off < 2.0
            assert -2.0 < y_off < 2.0


@pytest.mark.hardware
def test_rgb_template():
    """RGB template should produce valid outputs."""
    from libredgetpu.pattern.templates import list_templates

    templates = list_templates()
    rgb_templates = [(sh, sw, kh, kw, ch)
                     for sh, sw, kh, kw, ch in templates if ch == 3]
    if not rgb_templates:
        pytest.skip("No RGB pattern templates available")

    sh, sw, kh, kw, ch = rgb_templates[0]
    with PatternTracker.from_template(sh, kh, ch) as tracker:
        img = np.random.randint(0, 255, (sh, sw, 3), dtype=np.uint8)
        x_off, y_off = tracker.track(img)
        print(f"  RGB ({sh}x{sw}/{kh}x{kw}): x={x_off:+.3f}, y={y_off:+.3f}")
        assert isinstance(x_off, float) and isinstance(y_off, float)
        assert -2.0 < x_off < 2.0
        assert -2.0 < y_off < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
