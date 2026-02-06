#!/usr/bin/env python3
"""Tests for spot tracker module.

Offline tests run without hardware. Hardware tests require USB Edge TPU.

Usage:
    pytest tests/test_spot_tracker.py -v                    # offline only
    pytest tests/test_spot_tracker.py -v --run-hardware     # all tests
    pytest tests/test_spot_tracker.py -v --run-hardware -k validated
"""

import json
import os
import time

import numpy as np
import pytest

from libredgetpu.spot_tracker import SpotTracker


# ---------------------------------------------------------------------------
# Offline tests (no hardware required)
# ---------------------------------------------------------------------------

class TestOffsetToDirection:
    """Test direction conversion logic (pure CPU)."""

    def test_center(self):
        """Small offsets should return 'center'."""
        assert SpotTracker.offset_to_direction(0.0, 0.0) == "center"
        assert SpotTracker.offset_to_direction(0.05, 0.05) == "center"
        assert SpotTracker.offset_to_direction(-0.05, -0.05) == "center"

    def test_cardinal_left(self):
        """Negative x offset should return 'left'."""
        assert SpotTracker.offset_to_direction(-0.5, 0.0) == "left"

    def test_cardinal_right(self):
        """Positive x offset should return 'right'."""
        assert SpotTracker.offset_to_direction(0.5, 0.0) == "right"

    def test_cardinal_up(self):
        """Negative y offset should return 'up'."""
        assert SpotTracker.offset_to_direction(0.0, -0.5) == "up"

    def test_cardinal_down(self):
        """Positive y offset should return 'down'."""
        assert SpotTracker.offset_to_direction(0.0, 0.5) == "down"

    def test_diagonal_up_left(self):
        """Negative x and y should return 'up-left'."""
        assert SpotTracker.offset_to_direction(-0.5, -0.5) == "up-left"

    def test_diagonal_up_right(self):
        """Positive x, negative y should return 'up-right'."""
        assert SpotTracker.offset_to_direction(0.5, -0.5) == "up-right"

    def test_diagonal_down_left(self):
        """Negative x, positive y should return 'down-left'."""
        assert SpotTracker.offset_to_direction(-0.5, 0.5) == "down-left"

    def test_diagonal_down_right(self):
        """Positive x and y should return 'down-right'."""
        assert SpotTracker.offset_to_direction(0.5, 0.5) == "down-right"

    def test_custom_threshold(self):
        """Custom threshold should affect center detection."""
        # With default threshold (0.1), this is not center
        assert SpotTracker.offset_to_direction(0.15, 0.0) == "right"
        # With higher threshold, it is center
        assert SpotTracker.offset_to_direction(0.15, 0.0, threshold=0.2) == "center"


class TestServoError:
    """Test servo error conversion (pure CPU)."""

    def test_zero_offset(self):
        """Zero offset should give zero error."""
        x_err, y_err = SpotTracker.offset_to_servo_error(0.0, 0.0)
        assert x_err == 0.0
        assert y_err == 0.0

    def test_left_offset(self):
        """Left offset (negative x) should give positive x error (move right)."""
        x_err, y_err = SpotTracker.offset_to_servo_error(-0.5, 0.0)
        assert x_err == 0.5  # Positive = move right
        assert y_err == 0.0

    def test_right_offset(self):
        """Right offset (positive x) should give negative x error (move left)."""
        x_err, y_err = SpotTracker.offset_to_servo_error(0.5, 0.0)
        assert x_err == -0.5  # Negative = move left
        assert y_err == 0.0

    def test_up_offset(self):
        """Up offset (negative y) should give positive y error (move down)."""
        x_err, y_err = SpotTracker.offset_to_servo_error(0.0, -0.5)
        assert x_err == 0.0
        assert y_err == 0.5  # Positive = move down

    def test_down_offset(self):
        """Down offset (positive y) should give negative y error (move up)."""
        x_err, y_err = SpotTracker.offset_to_servo_error(0.0, 0.5)
        assert x_err == 0.0
        assert y_err == -0.5  # Negative = move up

    def test_gain_scaling(self):
        """Gain should scale the error."""
        x_err, y_err = SpotTracker.offset_to_servo_error(-1.0, -1.0, gain=0.5)
        assert x_err == 0.5
        assert y_err == 0.5


class TestTemplateList:
    """Test template discovery functions."""

    def test_list_templates_returns_list(self):
        """list_templates should return a list (possibly empty)."""
        from libredgetpu.tracker.templates import list_templates
        templates = list_templates()
        assert isinstance(templates, list)

    def test_list_templates_format(self):
        """Template entries should be (variant, height, width) tuples."""
        from libredgetpu.tracker.templates import list_templates
        templates = list_templates()
        for entry in templates:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            variant, h, w = entry
            assert isinstance(variant, str)
            assert isinstance(h, int) and h > 0
            assert isinstance(w, int) and w > 0


class TestOutputScaleCoverage:
    """Verify sidecar JSON output_scale covers the full [-10, +10] range."""

    def test_all_templates_have_sufficient_output_scale(self):
        """All templates should have output_scale >= 0.078 to cover [-10, +10].

        The X channel uses [-10, +10] and Y uses [0, +20] (with y_offset=10).
        The quantizer picks a range that covers both channels, so the int8
        range may be asymmetric. What matters is output_scale is large enough
        that the dequantized range spans at least 20 units (covering either
        [-10,+10] for X or [0,+20] for Y).
        """
        from libredgetpu.tracker.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No spot tracker templates available")

        for variant, h, w in templates:
            if variant == "bright":
                json_path = os.path.join(_TEMPLATES_DIR, f"bright_{h}x{w}_edgetpu.json")
            else:
                json_path = os.path.join(_TEMPLATES_DIR, f"{variant}_{h}x{w}_edgetpu.json")

            assert os.path.isfile(json_path), f"Missing sidecar: {json_path}"
            with open(json_path) as f:
                meta = json.load(f)

            scale = meta["output_scale"]
            zp = meta["output_zero_point"]

            # Compute dequantized range: (int8_val - zp) * scale
            min_val = (-128 - zp) * scale
            max_val = (127 - zp) * scale
            span = max_val - min_val

            print(f"  {variant}_{h}x{w}: scale={scale:.6f}, zp={zp}, "
                  f"range=[{min_val:.1f}, {max_val:.1f}], span={span:.1f}")

            # output_scale must be large enough to span 20 units
            # (255 int8 levels * scale >= 20 => scale >= 0.078)
            assert scale >= 0.078, (
                f"{variant}_{h}x{w}: output_scale={scale:.6f} too small "
                f"(need >=0.078 for 20-unit span)"
            )
            assert span >= 20.0, (
                f"{variant}_{h}x{w}: dequant span {span:.1f} < 20.0"
            )

    def test_all_templates_have_y_offset(self):
        """All templates should have y_offset in their sidecar JSON."""
        from libredgetpu.tracker.templates import list_templates, _TEMPLATES_DIR

        templates = list_templates()
        if not templates:
            pytest.skip("No spot tracker templates available")

        for variant, h, w in templates:
            if variant == "bright":
                json_path = os.path.join(_TEMPLATES_DIR, f"bright_{h}x{w}_edgetpu.json")
            else:
                json_path = os.path.join(_TEMPLATES_DIR, f"{variant}_{h}x{w}_edgetpu.json")

            with open(json_path) as f:
                meta = json.load(f)

            assert "y_offset" in meta, (
                f"{variant}_{h}x{w}: missing y_offset in sidecar JSON"
            )
            assert meta["y_offset"] == 10.0, (
                f"{variant}_{h}x{w}: y_offset={meta['y_offset']} (expected 10.0)"
            )


# ---------------------------------------------------------------------------
# Color tracking offline tests (no hardware required)
# ---------------------------------------------------------------------------

class TestFindClosestColor:
    """Test closest color matching logic (pure CPU)."""

    def test_exact_match_red(self):
        """Exact preset coefficients should match with distance 0."""
        from libredgetpu.tracker.templates import find_closest_color
        variant, dist = find_closest_color([1.0, -0.5, -0.5], 64)
        assert variant == "color_red"
        assert dist < 1e-6

    def test_exact_match_blue(self):
        """Exact blue preset should match."""
        from libredgetpu.tracker.templates import find_closest_color
        variant, dist = find_closest_color([-0.5, -0.5, 1.0], 64)
        assert variant == "color_blue"
        assert dist < 1e-6

    def test_closest_to_orange(self):
        """Orange-ish [1.0, 0.5, -0.5] should match yellow (nearest preset)."""
        from libredgetpu.tracker.templates import find_closest_color
        variant, dist = find_closest_color([1.0, 0.5, -0.5], 64)
        assert variant == "color_yellow"
        assert dist > 0  # not an exact match

    def test_closest_to_teal(self):
        """Teal-ish [-0.5, 0.8, 0.2] should match cyan."""
        from libredgetpu.tracker.templates import find_closest_color
        variant, dist = find_closest_color([-0.5, 0.8, 0.2], 64)
        assert variant == "color_cyan"

    def test_wrong_length_raises(self):
        """Non-3-element weights should raise ValueError."""
        from libredgetpu.tracker.templates import find_closest_color
        with pytest.raises(ValueError, match="Expected 3"):
            find_closest_color([1.0, 0.5], 64)

    def test_no_templates_raises(self):
        """Missing size should raise FileNotFoundError."""
        from libredgetpu.tracker.templates import find_closest_color
        with pytest.raises(FileNotFoundError):
            find_closest_color([1.0, 0.0, 0.0], 999)


class TestFromColorWeights:
    """Test from_color_weights class method (no hardware)."""

    def test_returns_tracker_with_match_info(self):
        """from_color_weights should set matched_variant and matched_distance."""
        tracker = SpotTracker.from_color_weights(64, weights=[1.0, -0.5, -0.5])
        assert tracker.matched_variant == "color_red"
        assert tracker.matched_distance < 1e-6
        assert tracker.channels == 3

    def test_custom_weights_find_nearest(self):
        """Custom weights should find nearest preset."""
        tracker = SpotTracker.from_color_weights(64, weights=[0.9, 0.1, -0.8])
        assert tracker.matched_variant is not None
        assert tracker.matched_distance > 0

    def test_no_match_info_on_regular_template(self):
        """Regular from_template should have None for match properties."""
        tracker = SpotTracker.from_template(64, variant="bright")
        assert tracker.matched_variant is None
        assert tracker.matched_distance is None


class TestSetColorOffline:
    """Test set_color validation (no hardware, no open())."""

    def test_rejects_bright_tracker(self):
        """set_color on a grayscale tracker should raise RuntimeError."""
        tracker = SpotTracker.from_template(64, variant="bright")
        with pytest.raises(RuntimeError, match="channels=1"):
            tracker.set_color([1.0, 0.0, 0.0])

    def test_rejects_wrong_length(self):
        """set_color with != 3 weights should raise ValueError."""
        tracker = SpotTracker.from_template(64, variant="color_red")
        with pytest.raises(ValueError, match="Expected 3"):
            tracker.set_color([1.0, 0.0])

    def test_rejects_out_of_range(self):
        """Coefficients exceeding quantization range should raise ValueError."""
        tracker = SpotTracker.from_template(64, variant="color_red")
        with pytest.raises(ValueError, match="exceeds quantization range"):
            tracker.set_color([5.0, 0.0, 0.0])  # way beyond [-1, +1]

    def test_patches_blob_bytes(self):
        """set_color should modify bytes [0:3] of the param blob."""
        tracker = SpotTracker.from_template(64, variant="color_red")
        original = tracker._pc_params[:3]
        tracker.set_color([-0.5, 1.0, -0.5])  # switch to green
        patched = tracker._pc_params[:3]
        assert original != patched, "Blob should change after set_color"

    def test_set_color_matches_compiled_template(self):
        """set_color to blue on a red template should produce same blob bytes as the blue template."""
        red = SpotTracker.from_template(64, variant="color_red")
        blue = SpotTracker.from_template(64, variant="color_blue")

        red.set_color([-0.5, -0.5, 1.0])  # switch red template to blue coefficients
        assert list(red._pc_params[:3]) == list(blue._pc_params[:3]), \
            "set_color(blue) on red template should match compiled blue template bytes"


class TestColorTemplateDiscovery:
    """Test color template listing and discovery."""

    def test_color_templates_listed(self):
        """list_templates should include color variants."""
        from libredgetpu.tracker.templates import list_templates
        templates = list_templates()
        color_templates = [v for v, h, w in templates if v.startswith("color_")]
        assert len(color_templates) >= 7, \
            f"Expected at least 7 color templates, got {len(color_templates)}"

    def test_available_colors(self):
        """get_available_colors should return the 7 preset colors."""
        from libredgetpu.tracker.templates import get_available_colors
        colors = get_available_colors()
        for expected in ["red", "green", "blue", "yellow", "cyan", "magenta", "white"]:
            assert expected in colors, f"Missing color: {expected}"

    def test_color_sidecar_has_weight_scale(self):
        """All color template sidecars should have color_weight_scale."""
        from libredgetpu.tracker.templates import list_templates, _TEMPLATES_DIR
        templates = list_templates()
        color_templates = [(v, h, w) for v, h, w in templates if v.startswith("color_")]
        assert len(color_templates) > 0, "No color templates found"

        for variant, h, w in color_templates:
            json_path = os.path.join(_TEMPLATES_DIR, f"{variant}_{h}x{w}_edgetpu.json")
            with open(json_path) as f:
                meta = json.load(f)
            assert "color_weight_scale" in meta, \
                f"{variant}_{h}x{w}: missing color_weight_scale in sidecar"
            assert meta["color_weight_scale"] > 0, \
                f"{variant}_{h}x{w}: invalid color_weight_scale={meta['color_weight_scale']}"


# ---------------------------------------------------------------------------
# Model creation test (requires TensorFlow, optional)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("tensorflow", reason="TensorFlow not installed"),
    reason="TensorFlow not installed"
)
def test_coordinate_grid_creation():
    """Test that coordinate grids are created correctly."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not installed")

    from libredgetpu.tflite_builder import _create_coordinate_grids

    h, w = 64, 64
    temp = 0.1
    x_coords, y_coords = _create_coordinate_grids(h, w, temp)

    # Check shapes
    assert x_coords.shape == (h, w), f"X coords shape: {x_coords.shape}"
    assert y_coords.shape == (h, w), f"Y coords shape: {y_coords.shape}"

    # Check center pixel is near zero
    center_x = x_coords[h // 2, w // 2]
    center_y = y_coords[h // 2, w // 2]
    # For 64x64, center is between pixels 31 and 32, so not exactly zero
    assert abs(center_x) < 0.5, f"Center X: {center_x}"
    assert abs(center_y) < 0.5, f"Center Y: {center_y}"

    # Check corners (accounting for temperature scaling)
    # Top-left: should be (-1/temp, -1/temp)
    assert x_coords[0, 0] < 0, "Top-left X should be negative"
    assert y_coords[0, 0] < 0, "Top-left Y should be negative"

    # Bottom-right: should be (+1/temp, +1/temp)
    assert x_coords[-1, -1] > 0, "Bottom-right X should be positive"
    assert y_coords[-1, -1] > 0, "Bottom-right Y should be positive"


# ---------------------------------------------------------------------------
# Hardware tests (require USB Edge TPU)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
def test_track_output_shape():
    """SpotTracker.track() should return exactly 2 values."""
    from libredgetpu.tracker.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No spot tracker templates available. Generate with: python -m libredgetpu.spot_tracker_gen")

    variant, h, w = templates[0]
    with SpotTracker.from_template(h, variant=variant) as tracker:
        if tracker.channels == 1:
            img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        else:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        x_off, y_off = tracker.track(img)
        assert isinstance(x_off, float), f"x_off type: {type(x_off)}"
        assert isinstance(y_off, float), f"y_off type: {type(y_off)}"


@pytest.mark.hardware
def test_uniform_image():
    """Uniform gray image should produce near-zero offsets.

    With uniform brightness, the softmax gives equal probability to all pixels,
    so the weighted sum of coordinates should be ~0 for both X and Y.
    """
    from libredgetpu.tracker.templates import list_templates

    templates = list_templates()
    # Find a grayscale template
    bright_templates = [(v, h, w) for v, h, w in templates if v == "bright"]
    if not bright_templates:
        pytest.skip("No bright spot templates available")

    variant, h, w = bright_templates[0]
    with SpotTracker.from_template(h, variant=variant) as tracker:
        # Uniform gray image - no clear bright spot
        img = np.full((h, w), 128, dtype=np.uint8)
        x_off, y_off = tracker.track(img)

        print(f"  Uniform image: x_off={x_off:.4f}, y_off={y_off:.4f}")
        # Both should be near zero (center of image)
        assert abs(x_off) < 0.15, f"Expected x_off near 0, got {x_off}"
        assert abs(y_off) < 0.15, f"Expected y_off near 0, got {y_off}"


@pytest.mark.hardware
def test_bright_corner_upper_left():
    """Bright spot in upper-left should give negative offsets."""
    from libredgetpu.tracker.templates import list_templates

    templates = list_templates()
    bright_templates = [(v, h, w) for v, h, w in templates if v == "bright"]
    if not bright_templates:
        pytest.skip("No bright spot templates available")

    variant, h, w = bright_templates[0]
    with SpotTracker.from_template(h, variant=variant) as tracker:
        # Dark image with bright spot in upper-left corner
        img = np.full((h, w), 16, dtype=np.uint8)
        spot_size = h // 8
        img[:spot_size, :spot_size] = 255

        x_off, y_off = tracker.track(img)
        print(f"  Upper-left spot: x_off={x_off:.4f}, y_off={y_off:.4f}")

        # Upper-left = negative x and negative y
        direction = SpotTracker.offset_to_direction(x_off, y_off)
        assert x_off < 0, f"Expected negative x_off, got {x_off}"
        assert y_off < 0, f"Expected negative y_off, got {y_off}"
        print(f"  Direction: {direction}")


@pytest.mark.hardware
def test_bright_corner_lower_right():
    """Bright spot in lower-right should give more positive offsets than upper-left."""
    from libredgetpu.tracker.templates import list_templates

    templates = list_templates()
    bright_templates = [(v, h, w) for v, h, w in templates if v == "bright"]
    if not bright_templates:
        pytest.skip("No bright spot templates available")

    variant, h, w = bright_templates[0]
    with SpotTracker.from_template(h, variant=variant) as tracker:
        # First measure upper-left for comparison
        img_ul = np.full((h, w), 16, dtype=np.uint8)
        spot_size = h // 8
        img_ul[:spot_size, :spot_size] = 255
        x_ul, y_ul = tracker.track(img_ul)

        # Then measure lower-right
        img_lr = np.full((h, w), 16, dtype=np.uint8)
        img_lr[-spot_size:, -spot_size:] = 255
        x_lr, y_lr = tracker.track(img_lr)

        print(f"  Upper-left spot: x_off={x_ul:.4f}, y_off={y_ul:.4f}")
        print(f"  Lower-right spot: x_off={x_lr:.4f}, y_off={y_lr:.4f}")

        # Lower-right should have higher x and y than upper-left
        assert x_lr > x_ul, f"Lower-right x ({x_lr}) should be > upper-left x ({x_ul})"
        assert y_lr > y_ul, f"Lower-right y ({y_lr}) should be > upper-left y ({y_ul})"


@pytest.mark.hardware
def test_bright_spot_positions():
    """Test bright spots at various positions."""
    from libredgetpu.tracker.templates import list_templates

    templates = list_templates()
    bright_templates = [(v, h, w) for v, h, w in templates if v == "bright"]
    if not bright_templates:
        pytest.skip("No bright spot templates available")

    variant, h, w = bright_templates[0]
    with SpotTracker.from_template(h, variant=variant) as tracker:
        # Test positions: (row_frac, col_frac, expected_x_sign, expected_y_sign)
        test_cases = [
            (0.25, 0.25, -1, -1),  # Upper-left quadrant
            (0.25, 0.75, +1, -1),  # Upper-right quadrant
            (0.75, 0.25, -1, +1),  # Lower-left quadrant
            (0.75, 0.75, +1, +1),  # Lower-right quadrant
        ]

        spot_size = max(h // 16, 4)

        for row_frac, col_frac, x_sign, y_sign in test_cases:
            img = np.full((h, w), 16, dtype=np.uint8)
            row = int(row_frac * h)
            col = int(col_frac * w)
            # Draw bright spot
            r1, r2 = max(0, row - spot_size), min(h, row + spot_size)
            c1, c2 = max(0, col - spot_size), min(w, col + spot_size)
            img[r1:r2, c1:c2] = 255

            x_off, y_off = tracker.track(img)

            # Check sign matches expected (threshold excludes zero-crossing)
            if x_sign < 0:
                assert x_off < -0.05, f"Position ({row_frac}, {col_frac}): expected x<-0.05, got {x_off}"
            else:
                assert x_off > 0.05, f"Position ({row_frac}, {col_frac}): expected x>0.05, got {x_off}"

            if y_sign < 0:
                assert y_off < -0.05, f"Position ({row_frac}, {col_frac}): expected y<-0.05, got {y_off}"
            else:
                assert y_off > 0.05, f"Position ({row_frac}, {col_frac}): expected y>0.05, got {y_off}"


@pytest.mark.hardware
def test_benchmark():
    """Benchmark spot tracker latency."""
    from libredgetpu.tracker.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No spot tracker templates available")

    variant, h, w = templates[0]
    with SpotTracker.from_template(h, variant=variant) as tracker:
        if tracker.channels == 1:
            img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        else:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

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
        print(f"  {variant} {h}x{w}: avg={avg:.2f} ms, std={std:.2f} ms, min={min(times):.2f}, max={max(times):.2f}")


# ---------------------------------------------------------------------------
# Validated test (hardware + synthetic moving spot)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.validated
def test_moving_spot():
    """Moving spot should produce monotonically changing offsets.

    This validates the visual servoing concept: as a spot moves across
    the image, the tracker should report monotonically changing offsets.
    """
    from libredgetpu.tracker.templates import list_templates

    templates = list_templates()
    bright_templates = [(v, h, w) for v, h, w in templates if v == "bright"]
    if not bright_templates:
        pytest.skip("No bright spot templates available")

    variant, h, w = bright_templates[0]
    with SpotTracker.from_template(h, variant=variant) as tracker:
        spot_size = max(h // 10, 6)

        # Move spot from left to right
        positions = np.linspace(0.2, 0.8, 5)
        x_offsets = []

        for col_frac in positions:
            img = np.full((h, w), 16, dtype=np.uint8)
            row = h // 2  # Center row
            col = int(col_frac * w)
            r1, r2 = max(0, row - spot_size), min(h, row + spot_size)
            c1, c2 = max(0, col - spot_size), min(w, col + spot_size)
            img[r1:r2, c1:c2] = 255

            x_off, y_off = tracker.track(img)
            x_offsets.append(x_off)
            print(f"  Position {col_frac:.2f}: x_off={x_off:.4f}, y_off={y_off:.4f}")

        # X offsets should be monotonically increasing (left to right)
        for i in range(1, len(x_offsets)):
            assert x_offsets[i] > x_offsets[i-1], \
                f"X offset should increase left to right: {x_offsets}"

        # The range of offsets should be significant
        offset_range = x_offsets[-1] - x_offsets[0]
        assert offset_range > 0.3, f"Offset range too small: {offset_range}"


# ---------------------------------------------------------------------------
# Multi-size hardware tests (parametrized across all template sizes)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.validated
@pytest.mark.parametrize("size", [16, 64, 128])
def test_corner_tracking_multi_size(size):
    """Corner spots should produce correct sign AND magnitude > 0.3 at each size.

    This is the key test for the quantization fix: before the fix, 64x64 and
    128x128 had narrow output_scale that clipped off-center outputs to ~0.
    """
    from libredgetpu.tracker.templates import get_template

    try:
        tflite_path, json_path = get_template(size, "bright")
    except FileNotFoundError:
        pytest.skip(f"No bright template for {size}x{size}")

    with SpotTracker(tflite_path, metadata_path=json_path) as tracker:
        h, w = size, size
        spot_size = max(h // 8, 2)

        # Test all four corners: (row_slice, col_slice, expected_x_sign, expected_y_sign)
        corners = [
            (slice(0, spot_size), slice(0, spot_size), -1, -1, "upper-left"),
            (slice(0, spot_size), slice(w - spot_size, w), +1, -1, "upper-right"),
            (slice(h - spot_size, h), slice(0, spot_size), -1, +1, "lower-left"),
            (slice(h - spot_size, h), slice(w - spot_size, w), +1, +1, "lower-right"),
        ]

        for row_sl, col_sl, x_sign, y_sign, label in corners:
            img = np.full((h, w), 16, dtype=np.uint8)
            img[row_sl, col_sl] = 255

            x_off, y_off = tracker.track(img)
            print(f"  {size}x{size} {label}: x={x_off:.4f}, y={y_off:.4f}")

            # Check sign
            if x_sign < 0:
                assert x_off < 0, f"{size}x{size} {label}: expected x<0, got {x_off}"
            else:
                assert x_off > 0, f"{size}x{size} {label}: expected x>0, got {x_off}"

            if y_sign < 0:
                assert y_off < 0, f"{size}x{size} {label}: expected y<0, got {y_off}"
            else:
                assert y_off > 0, f"{size}x{size} {label}: expected y>0, got {y_off}"

            # Check magnitude — this catches the quantization clipping bug
            assert abs(x_off) > 0.3, (
                f"{size}x{size} {label}: |x_off|={abs(x_off):.4f} too small "
                f"(output_scale clipping?)"
            )
            assert abs(y_off) > 0.3, (
                f"{size}x{size} {label}: |y_off|={abs(y_off):.4f} too small "
                f"(output_scale clipping?)"
            )


# ---------------------------------------------------------------------------
# Color tracking hardware tests
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.parametrize("color,channel", [
    ("color_red", 0),
    ("color_green", 1),
    ("color_blue", 2),
])
def test_color_template_detects_matching_spot(color, channel):
    """Each color template should detect a spot of its own color in the correct position.

    Places a colored spot in the upper-right quadrant and verifies x>0, y<0.
    """
    from libredgetpu.tracker.templates import get_template

    size = 64
    try:
        tflite_path, json_path = get_template(size, color)
    except FileNotFoundError:
        pytest.skip(f"No template for {color} at {size}x{size}")

    with SpotTracker(tflite_path, metadata_path=json_path) as tracker:
        img = np.full((size, size, 3), 32, dtype=np.uint8)
        # Colored spot in upper-right
        spot = slice(size // 8, size // 4)
        img[spot, 3 * size // 4:7 * size // 8, channel] = 255

        x_off, y_off = tracker.track(img)
        print(f"  {color}: x={x_off:+.3f}, y={y_off:+.3f}")
        assert x_off > 0, f"{color}: expected x>0, got {x_off}"
        assert y_off < 0, f"{color}: expected y<0, got {y_off}"


@pytest.mark.hardware
def test_color_template_all_seven():
    """All 7 color templates should produce valid outputs on random RGB input."""
    from libredgetpu.tracker.templates import get_available_colors, get_template

    colors = get_available_colors()
    if not colors:
        pytest.skip("No color templates available")

    size = 64
    for color_name in colors:
        variant = f"color_{color_name}"
        try:
            tflite_path, json_path = get_template(size, variant)
        except FileNotFoundError:
            continue

        with SpotTracker(tflite_path, metadata_path=json_path) as tracker:
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            x_off, y_off = tracker.track(img)
            # Just verify it runs without error and returns valid floats
            assert isinstance(x_off, float) and isinstance(y_off, float), \
                f"{variant}: invalid output types"
            assert -2.0 < x_off < 2.0, f"{variant}: x_off={x_off} out of range"
            assert -2.0 < y_off < 2.0, f"{variant}: y_off={y_off} out of range"
            print(f"  {variant}: x={x_off:+.3f}, y={y_off:+.3f}")


@pytest.mark.hardware
@pytest.mark.validated
def test_set_color_runtime_swap():
    """Runtime color swap should track different colored spots in the same image.

    Creates an image with a red spot upper-left and a green spot lower-right.
    Switching the color filter at runtime should flip which spot is tracked.
    This is the core validation of runtime color swapping.
    """
    size = 64
    with SpotTracker.from_template(size, variant="color_red") as tracker:
        # Image with two colored spots
        img = np.full((size, size, 3), 32, dtype=np.uint8)
        img[5:15, 5:15, 0] = 255       # red spot upper-left
        img[50:60, 50:60, 1] = 255      # green spot lower-right

        # Red filter: should find upper-left (x<0, y<0)
        tracker.set_color([1.0, -0.5, -0.5])
        x_r, y_r = tracker.track(img)
        print(f"  Red filter:   x={x_r:+.3f}, y={y_r:+.3f}")
        assert x_r < 0, f"Red filter should find upper-left (x<0), got x={x_r}"
        assert y_r < 0, f"Red filter should find upper-left (y<0), got y={y_r}"

        # Green filter: should find lower-right (x>0, y>0)
        tracker.set_color([-0.5, 1.0, -0.5])
        x_g, y_g = tracker.track(img)
        print(f"  Green filter: x={x_g:+.3f}, y={y_g:+.3f}")
        assert x_g > 0, f"Green filter should find lower-right (x>0), got x={x_g}"
        assert y_g > 0, f"Green filter should find lower-right (y>0), got y={y_g}"

        # The two tracked positions should be clearly different
        dx = abs(x_g - x_r)
        dy = abs(y_g - y_r)
        assert dx > 0.5, f"X difference between red/green too small: {dx}"
        assert dy > 0.5, f"Y difference between red/green too small: {dy}"


@pytest.mark.hardware
@pytest.mark.validated
def test_set_color_custom_at_runtime():
    """Arbitrary custom [R,G,B] coefficients should work at runtime.

    Tests that a non-preset color (orange-ish) correctly detects a matching
    spot when the scene has both a matching and non-matching colored region.
    """
    size = 64
    with SpotTracker.from_template(size, variant="color_red") as tracker:
        # Orange filter: high R, some G, anti-B
        tracker.set_color([0.8, 0.3, -0.6])

        # Image: orange spot lower-left, blue spot upper-right
        img = np.full((size, size, 3), 32, dtype=np.uint8)
        # Orange spot (R=255, G=180, B=0) in lower-left
        img[48:58, 5:15, 0] = 255
        img[48:58, 5:15, 1] = 180
        img[48:58, 5:15, 2] = 0
        # Blue spot (R=0, G=0, B=255) in upper-right
        img[5:15, 48:58, 0] = 0
        img[5:15, 48:58, 1] = 0
        img[5:15, 48:58, 2] = 255

        x_off, y_off = tracker.track(img)
        print(f"  Orange filter: x={x_off:+.3f}, y={y_off:+.3f}")

        # Orange filter should prefer the orange spot (lower-left: x<0, y>0)
        # and reject the blue spot (negative contribution from B channel)
        assert x_off < 0, f"Orange filter should find lower-left (x<0), got x={x_off}"
        assert y_off > 0, f"Orange filter should find lower-left (y>0), got y={y_off}"


@pytest.mark.hardware
def test_set_color_multiple_swaps():
    """Multiple consecutive set_color calls should all work without errors.

    Each swap re-uploads parameters. Tests that the param cache invalidation
    works correctly across 5 consecutive color changes.
    """
    size = 64
    with SpotTracker.from_template(size, variant="color_red") as tracker:
        # Red spot lower-left, green spot upper-right
        img = np.full((size, size, 3), 32, dtype=np.uint8)
        img[48:58, 5:15, 0] = 255     # red spot lower-left
        img[5:15, 48:58, 1] = 255     # green spot upper-right

        swaps = [
            ([1.0, -0.5, -0.5], "red"),
            ([-0.5, 1.0, -0.5], "green"),
            ([-0.5, -0.5, 1.0], "blue"),
            ([-0.5, 1.0, -0.5], "green again"),
            ([1.0, -0.5, -0.5], "red again"),
        ]

        results = []
        for weights, label in swaps:
            tracker.set_color(weights)
            x_off, y_off = tracker.track(img)
            results.append((label, x_off, y_off))
            print(f"  {label}: x={x_off:+.3f}, y={y_off:+.3f}")
            # Basic validity: all outputs should be in range
            assert -2.0 < x_off < 2.0, f"{label}: x_off={x_off} out of range"
            assert -2.0 < y_off < 2.0, f"{label}: y_off={y_off} out of range"

        # Red filter should find lower-left (x<0), green should find upper-right (x>0)
        assert results[0][1] < 0, f"Red filter should find lower-left (x<0)"
        assert results[1][1] > 0, f"Green filter should find upper-right (x>0)"
        # Return to same filters should give consistent results
        assert abs(results[1][1] - results[3][1]) < 0.1, \
            "Two green filter runs should agree"
        assert abs(results[0][1] - results[4][1]) < 0.1, \
            "Two red filter runs should agree"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
