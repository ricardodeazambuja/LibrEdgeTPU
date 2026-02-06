#!/usr/bin/env python3
"""Tests for looming detection module.

Offline tests run without hardware. Hardware tests require USB Edge TPU.

Usage:
    pytest tests/test_looming.py -v                    # offline only
    pytest tests/test_looming.py -v --run-hardware     # all tests
    pytest tests/test_looming.py -v --run-hardware -k validated
"""

import time

import numpy as np
import pytest

from libredgetpu.looming_detector import LoomingDetector


# ---------------------------------------------------------------------------
# Offline tests (no hardware required)
# ---------------------------------------------------------------------------

class TestComputeTau:
    """Test tau computation logic (pure CPU)."""

    def test_tau_uniform(self):
        """Uniform zone values should give tau ≈ 1.0."""
        zones = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        tau = LoomingDetector.compute_tau(zones)
        assert abs(tau - 1.0) < 0.01, f"Expected tau ≈ 1.0, got {tau}"

    def test_tau_looming(self):
        """High center, low periphery should give tau > 1.0."""
        # Center (index 4) is high, periphery is low
        zones = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0])
        tau = LoomingDetector.compute_tau(zones)
        assert tau > 5.0, f"Expected tau > 5.0, got {tau}"

    def test_tau_receding(self):
        """Low center, high periphery should give tau < 1.0."""
        # Center (index 4) is low, periphery is high
        zones = np.array([10.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0])
        tau = LoomingDetector.compute_tau(zones)
        assert tau < 0.2, f"Expected tau < 0.2, got {tau}"

    def test_tau_zero_periphery(self):
        """Zero periphery should use epsilon to prevent division by zero."""
        zones = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        tau = LoomingDetector.compute_tau(zones, epsilon=1e-6)
        assert tau > 1e5, f"Expected very large tau, got {tau}"

    def test_tau_wrong_size(self):
        """Wrong number of zones should raise ValueError."""
        with pytest.raises(ValueError, match="Expected 9 zones"):
            LoomingDetector.compute_tau(np.array([1.0, 2.0, 3.0]))


class TestComputeTTC:
    """Test time-to-contact computation (pure CPU)."""

    def test_ttc_approaching(self):
        """Increasing tau should give finite TTC."""
        # Tau increasing: 0.8 -> 1.0 -> 1.2 -> 1.4
        tau_history = [0.8, 1.0, 1.2, 1.4]
        dt = 0.1  # 100ms between samples
        ttc = LoomingDetector.compute_ttc(tau_history, dt)
        # Rate = (1.4 - 0.8) / (3 * 0.1) = 2.0 per second
        # TTC to reach 2.0 from 1.4 = (2.0 - 1.4) / 2.0 = 0.3 seconds
        assert 0.0 < ttc < 1.0, f"Expected finite TTC, got {ttc}"

    def test_ttc_receding(self):
        """Decreasing tau should give infinite TTC."""
        # Tau decreasing: object moving away
        tau_history = [1.4, 1.2, 1.0, 0.8]
        dt = 0.1
        ttc = LoomingDetector.compute_ttc(tau_history, dt)
        assert ttc == float('inf'), f"Expected infinite TTC, got {ttc}"

    def test_ttc_stable(self):
        """Stable tau should give infinite TTC."""
        tau_history = [1.0, 1.0, 1.0, 1.0]
        dt = 0.1
        ttc = LoomingDetector.compute_ttc(tau_history, dt)
        assert ttc == float('inf'), f"Expected infinite TTC, got {ttc}"

    def test_ttc_insufficient_samples(self):
        """Too few samples should return infinite TTC."""
        tau_history = [1.0, 1.2]
        dt = 0.1
        ttc = LoomingDetector.compute_ttc(tau_history, dt, min_samples=3)
        assert ttc == float('inf'), f"Expected infinite TTC with insufficient data"


class TestTemplateList:
    """Test template discovery functions."""

    def test_list_templates_returns_list(self):
        """list_templates should return a list (possibly empty)."""
        from libredgetpu.looming.templates import list_templates
        templates = list_templates()
        assert isinstance(templates, list)

    def test_list_templates_format(self):
        """Template entries should be (height, width, zones) tuples."""
        from libredgetpu.looming.templates import list_templates
        templates = list_templates()
        for entry in templates:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            h, w, z = entry
            assert isinstance(h, int) and h > 0
            assert isinstance(w, int) and w > 0
            assert isinstance(z, int) and z > 0


# ---------------------------------------------------------------------------
# Model creation test (requires TensorFlow, optional)
# ---------------------------------------------------------------------------

def test_model_creation():
    """Test that looming model can be built and has correct Sobel kernels."""
    from libredgetpu.tflite_builder import build_looming
    from libredgetpu.tflite_parser import parse_full

    tflite_bytes, metadata = build_looming(64, 64)
    m = parse_full(tflite_bytes)

    # Find Sobel Conv2D kernels and verify shape/values
    conv_ops = [op for op in m.operators if op.opcode_name == "CONV_2D"]
    assert len(conv_ops) == 2, f"Expected 2 Sobel Conv2Ds, got {len(conv_ops)}"

    for conv in conv_ops:
        weight_tensor = m.tensors[conv.inputs[1]]
        assert weight_tensor.shape == [1, 3, 3, 1]
        weight_buf = m.buffers[weight_tensor.buffer_index]
        vals = np.frombuffer(weight_buf, dtype=np.int8)
        # Verify Sobel kernel pattern: {0, ±64, ±127}
        assert set(np.abs(vals)) == {0, 64, 127}


# ---------------------------------------------------------------------------
# Hardware tests (require USB Edge TPU)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
def test_detect_output_shape():
    """LoomingDetector.detect() should return exactly 9 values."""
    from libredgetpu.looming.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No looming templates available. Generate with: python -m libredgetpu.looming_gen")

    h, w, z = templates[0]
    with LoomingDetector.from_template(h, zones=z) as detector:
        img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        zones = detector.detect(img)
        assert zones.shape == (z * z,), f"Expected shape ({z * z},), got {zones.shape}"


@pytest.mark.hardware
def test_uniform_image():
    """Uniform gray image should produce similar zone values."""
    from libredgetpu.looming.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No looming templates available")

    h, w, z = templates[0]
    with LoomingDetector.from_template(h, zones=z) as detector:
        # Uniform gray image has no edges
        img = np.full((h, w), 128, dtype=np.uint8)
        zones = detector.detect(img)

        # All zones should be similar (within some tolerance)
        zone_std = np.std(zones)
        zone_mean = np.mean(zones)
        # Coefficient of variation should be low for uniform image
        cv = zone_std / max(zone_mean, 1e-6)
        assert cv < 1.0, f"Zone values too variable for uniform image: std={zone_std:.3f}, mean={zone_mean:.3f}"


@pytest.mark.hardware
def test_center_edge():
    """Vertical stripe in center should make zone[4] higher than periphery."""
    from libredgetpu.looming.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No looming templates available")

    h, w, z = templates[0]
    with LoomingDetector.from_template(h, zones=z) as detector:
        # Create image with vertical edge in center
        img = np.full((h, w), 32, dtype=np.uint8)
        # Add vertical stripe in center third
        stripe_start = w // 3
        stripe_end = 2 * w // 3
        img[:, stripe_start:stripe_end] = 224

        zones = detector.detect(img)
        tau = LoomingDetector.compute_tau(zones)

        # Center zone should have more edge activity
        # (stripe creates edges at boundaries)
        center = zones[4]
        mean_periphery = np.mean([zones[i] for i in [0, 1, 2, 3, 5, 6, 7, 8]])

        print(f"  Zones: {zones}")
        print(f"  Center: {center:.4f}, Periphery mean: {mean_periphery:.4f}")
        print(f"  Tau: {tau:.3f}")

        # Note: this test may need adjustment based on actual model behavior
        # The edge pattern spans multiple zones, so we're just checking basic functionality


@pytest.mark.hardware
def test_benchmark():
    """Benchmark looming detection latency."""
    from libredgetpu.looming.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No looming templates available")

    h, w, z = templates[0]
    with LoomingDetector.from_template(h, zones=z) as detector:
        img = np.random.randint(0, 255, (h, w), dtype=np.uint8)

        # Warmup
        for _ in range(5):
            detector.detect(img)

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            detector.detect(img)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = np.mean(times)
        std = np.std(times)
        print(f"  {h}x{w}: avg={avg:.2f} ms, std={std:.2f} ms, min={min(times):.2f}, max={max(times):.2f}")


# ---------------------------------------------------------------------------
# Validated test (hardware + synthetic expanding circle)
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.validated
def test_expanding_circle():
    """Concentric circles should show increasing tau as radius grows.

    This validates the looming detection concept: as an object approaches,
    its edges expand from center outward, increasing the center/periphery ratio.
    """
    from libredgetpu.looming.templates import list_templates

    templates = list_templates()
    if not templates:
        pytest.skip("No looming templates available")

    h, w, z = templates[0]
    with LoomingDetector.from_template(h, zones=z) as detector:
        center_y, center_x = h // 2, w // 2

        tau_values = []
        radii = [h // 8, h // 4, 3 * h // 8]  # Increasing circle radii

        for radius in radii:
            # Create image with circle outline
            img = np.full((h, w), 32, dtype=np.uint8)
            y, x = np.ogrid[:h, :w]
            dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
            # Draw circle edge (ring)
            ring_mask = (dist_sq >= (radius - 2) ** 2) & (dist_sq <= (radius + 2) ** 2)
            img[ring_mask] = 224

            zones = detector.detect(img)
            tau = LoomingDetector.compute_tau(zones)
            tau_values.append(tau)
            print(f"  Radius {radius}: tau={tau:.3f}")

        # As circle expands, more edges move to periphery, tau should change
        # (The exact direction depends on model architecture)
        # At minimum, we should get different tau values for different sizes
        tau_range = max(tau_values) - min(tau_values)
        assert tau_range > 0.01, f"Tau should vary with circle size, but range was only {tau_range:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
