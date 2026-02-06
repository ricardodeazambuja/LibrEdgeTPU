#!/usr/bin/env python3
"""Tests for VisualCompass — yaw estimation wrapper around OpticalFlow.

All tests run offline using a lightweight MockFlow. No hardware required.

Usage:
    pytest tests/test_visual_compass.py -v
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from libredgetpu.visual_compass import VisualCompass


# ---------------------------------------------------------------------------
# MockFlow — minimal stand-in for OpticalFlow
# ---------------------------------------------------------------------------

class MockFlow:
    """Lightweight OpticalFlow substitute for offline testing."""

    def __init__(self, width=64, height=64, fused_pool=4, pool_factor=4):
        self._width = width
        self._height = height
        self._fused_pool = fused_pool
        self._pool_factor = pool_factor
        self._opened = False
        self._closed = False
        # Settable return value for compute()
        self._vx = 0.0
        self._vy = 0.0

    # Properties that VisualCompass reads
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fused_pool(self):
        return self._fused_pool

    @property
    def pool_factor(self):
        return self._pool_factor

    def compute(self, frame_t, frame_t1):
        return self._vx, self._vy

    def open(self):
        self._opened = True

    def close(self):
        self._closed = True


# ---------------------------------------------------------------------------
# TestInit — constructor and property validation
# ---------------------------------------------------------------------------

class TestInit:
    """Test VisualCompass.__init__ and property accessors."""

    def test_basic_properties(self):
        """Properties expose flow dimensions and fov."""
        flow = MockFlow(width=64, height=64, fused_pool=4)
        compass = VisualCompass(flow, fov_deg=90.0)
        assert compass.fov_deg == 90.0
        assert compass.width == 64
        assert compass.height == 64
        assert compass.flow is flow

    def test_deg_per_pooled_px_fused(self):
        """With fused_pool > 0, effective_pool = fused_pool."""
        flow = MockFlow(width=64, fused_pool=4, pool_factor=8)
        compass = VisualCompass(flow, fov_deg=90.0)
        # deg_per_pooled_px = 90 * 4 / 64 = 5.625
        assert compass.deg_per_pooled_px == pytest.approx(5.625)

    def test_deg_per_pooled_px_standard(self):
        """With fused_pool = 0, effective_pool = pool_factor."""
        flow = MockFlow(width=64, fused_pool=0, pool_factor=4)
        compass = VisualCompass(flow, fov_deg=90.0)
        # deg_per_pooled_px = 90 * 4 / 64 = 5.625
        assert compass.deg_per_pooled_px == pytest.approx(5.625)

    def test_deg_per_pooled_px_different_fov(self):
        """Different FOV produces proportionally different scaling."""
        flow = MockFlow(width=128, fused_pool=4)
        compass = VisualCompass(flow, fov_deg=180.0)
        # deg_per_pooled_px = 180 * 4 / 128 = 5.625
        assert compass.deg_per_pooled_px == pytest.approx(5.625)

    def test_fov_zero_raises(self):
        """fov_deg = 0 is rejected."""
        flow = MockFlow()
        with pytest.raises(ValueError, match="fov_deg must be in"):
            VisualCompass(flow, fov_deg=0.0)

    def test_fov_negative_raises(self):
        """Negative fov_deg is rejected."""
        flow = MockFlow()
        with pytest.raises(ValueError, match="fov_deg must be in"):
            VisualCompass(flow, fov_deg=-10.0)

    def test_fov_above_360_raises(self):
        """fov_deg > 360 is rejected."""
        flow = MockFlow()
        with pytest.raises(ValueError, match="fov_deg must be in"):
            VisualCompass(flow, fov_deg=361.0)

    def test_fov_360_accepted(self):
        """fov_deg = 360 (fisheye) is valid."""
        flow = MockFlow(width=64, fused_pool=4)
        compass = VisualCompass(flow, fov_deg=360.0)
        assert compass.fov_deg == 360.0
        # deg_per_pooled_px = 360 * 4 / 64 = 22.5
        assert compass.deg_per_pooled_px == pytest.approx(22.5)


# ---------------------------------------------------------------------------
# TestCompute — yaw computation
# ---------------------------------------------------------------------------

class TestCompute:
    """Test compute_yaw and compute methods."""

    def test_zero_flow(self):
        """Zero displacement yields zero yaw."""
        flow = MockFlow(width=64, fused_pool=4)
        flow._vx, flow._vy = 0.0, 0.0
        compass = VisualCompass(flow, fov_deg=90.0)
        assert compass.compute_yaw(np.zeros(1), np.zeros(1)) == pytest.approx(0.0)

    def test_positive_vx(self):
        """Positive vx (rightward scene motion) yields positive yaw."""
        flow = MockFlow(width=64, fused_pool=4)
        flow._vx, flow._vy = 2.0, 0.0
        compass = VisualCompass(flow, fov_deg=90.0)
        # yaw = 2.0 * 5.625 = 11.25
        yaw = compass.compute_yaw(np.zeros(1), np.zeros(1))
        assert yaw == pytest.approx(11.25)

    def test_negative_vx(self):
        """Negative vx yields negative yaw (leftward rotation)."""
        flow = MockFlow(width=64, fused_pool=4)
        flow._vx, flow._vy = -3.0, 0.0
        compass = VisualCompass(flow, fov_deg=90.0)
        # yaw = -3.0 * 5.625 = -16.875
        yaw = compass.compute_yaw(np.zeros(1), np.zeros(1))
        assert yaw == pytest.approx(-16.875)

    def test_magnitude_scaling(self):
        """Larger vx yields proportionally larger yaw."""
        flow = MockFlow(width=64, fused_pool=4)
        compass = VisualCompass(flow, fov_deg=90.0)

        flow._vx = 1.0
        yaw1 = compass.compute_yaw(np.zeros(1), np.zeros(1))
        flow._vx = 4.0
        yaw4 = compass.compute_yaw(np.zeros(1), np.zeros(1))
        assert yaw4 == pytest.approx(4.0 * yaw1)

    def test_vy_ignored_in_yaw(self):
        """Vertical displacement doesn't affect yaw."""
        flow = MockFlow(width=64, fused_pool=4)
        flow._vx, flow._vy = 1.0, 3.5
        compass = VisualCompass(flow, fov_deg=90.0)
        yaw = compass.compute_yaw(np.zeros(1), np.zeros(1))
        # Should equal vx * deg_per_px regardless of vy
        assert yaw == pytest.approx(1.0 * 5.625)

    def test_compute_returns_three(self):
        """compute() returns (yaw_deg, vx, vy)."""
        flow = MockFlow(width=64, fused_pool=4)
        flow._vx, flow._vy = 2.0, -1.0
        compass = VisualCompass(flow, fov_deg=90.0)
        yaw, vx, vy = compass.compute(np.zeros(1), np.zeros(1))
        assert yaw == pytest.approx(2.0 * 5.625)
        assert vx == pytest.approx(2.0)
        assert vy == pytest.approx(-1.0)

    def test_different_fov(self):
        """Different FOV changes the yaw magnitude."""
        flow = MockFlow(width=64, fused_pool=4)
        flow._vx = 2.0

        c60 = VisualCompass(flow, fov_deg=60.0)
        c120 = VisualCompass(flow, fov_deg=120.0)

        y60 = c60.compute_yaw(np.zeros(1), np.zeros(1))
        y120 = c120.compute_yaw(np.zeros(1), np.zeros(1))
        assert y120 == pytest.approx(2.0 * y60)


# ---------------------------------------------------------------------------
# TestYawToDirection — static direction classifier
# ---------------------------------------------------------------------------

class TestYawToDirection:
    """Test yaw_to_direction static method."""

    def test_center(self):
        """Small yaw is classified as center."""
        assert VisualCompass.yaw_to_direction(0.0) == "center"
        assert VisualCompass.yaw_to_direction(0.5) == "center"
        assert VisualCompass.yaw_to_direction(-0.5) == "center"

    def test_right(self):
        """Positive yaw above threshold is right."""
        assert VisualCompass.yaw_to_direction(2.0) == "right"

    def test_left(self):
        """Negative yaw below threshold is left."""
        assert VisualCompass.yaw_to_direction(-2.0) == "left"

    def test_threshold_boundary(self):
        """Exactly at threshold is still center."""
        assert VisualCompass.yaw_to_direction(1.0) == "center"
        assert VisualCompass.yaw_to_direction(-1.0) == "center"
        assert VisualCompass.yaw_to_direction(1.01) == "right"
        assert VisualCompass.yaw_to_direction(-1.01) == "left"

    def test_custom_threshold(self):
        """Custom threshold changes classification boundary."""
        assert VisualCompass.yaw_to_direction(3.0, threshold_deg=5.0) == "center"
        assert VisualCompass.yaw_to_direction(6.0, threshold_deg=5.0) == "right"
        assert VisualCompass.yaw_to_direction(-6.0, threshold_deg=5.0) == "left"


# ---------------------------------------------------------------------------
# TestLifecycle — context manager and open/close delegation
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Test open/close/context manager delegation."""

    def test_open_delegates(self):
        """open() calls flow.open()."""
        flow = MockFlow()
        compass = VisualCompass(flow, fov_deg=90.0)
        compass.open()
        assert flow._opened

    def test_close_delegates(self):
        """close() calls flow.close()."""
        flow = MockFlow()
        compass = VisualCompass(flow, fov_deg=90.0)
        compass.close()
        assert flow._closed

    def test_context_manager(self):
        """Context manager calls open on entry and close on exit."""
        flow = MockFlow()
        compass = VisualCompass(flow, fov_deg=90.0)
        with compass as c:
            assert c is compass
            assert flow._opened
        assert flow._closed


# ---------------------------------------------------------------------------
# TestFromTemplate — factory classmethod
# ---------------------------------------------------------------------------

class TestFromTemplate:
    """Test from_template factory."""

    @patch("libredgetpu.visual_compass.OpticalFlow.from_template")
    def test_creates_with_pooled(self, mock_from_template):
        """from_template passes pooled=True and creates VisualCompass."""
        mock_flow = MockFlow(width=64, fused_pool=4)
        mock_from_template.return_value = mock_flow

        compass = VisualCompass.from_template(64, fov_deg=90.0, pooled=True)

        mock_from_template.assert_called_once_with(
            64,
            search_range=4,
            temperature=0.1,
            pool_factor=4,
            firmware_path=None,
            pooled=True,
        )
        assert compass.fov_deg == 90.0
        assert compass._owns_flow is True

    @patch("libredgetpu.visual_compass.OpticalFlow.from_template")
    def test_creates_without_pooled(self, mock_from_template):
        """from_template can pass pooled=False."""
        mock_flow = MockFlow(width=128, fused_pool=0, pool_factor=4)
        mock_from_template.return_value = mock_flow

        compass = VisualCompass.from_template(128, fov_deg=120.0, pooled=False)

        mock_from_template.assert_called_once_with(
            128,
            search_range=4,
            temperature=0.1,
            pool_factor=4,
            firmware_path=None,
            pooled=False,
        )
        assert compass.fov_deg == 120.0
        assert compass.width == 128
