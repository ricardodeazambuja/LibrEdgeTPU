"""Visual servo tracking for Edge TPU.

This module provides:
- SpotTracker: Edge TPU-accelerated bright spot and color tracking
- get_template, list_templates: Template discovery functions

Visual servoing works by computing the (x_offset, y_offset) of a target
from the image center using soft argmax (differentiable centroid).

Output interpretation:
- (0, 0): Target at image center
- (-1, 0): Target at left edge
- (+1, 0): Target at right edge
- (0, -1): Target at top edge
- (0, +1): Target at bottom edge

For servo control, negate the offset to get the error signal:
- x_off < 0: Move camera right (target is left)
- y_off < 0: Move camera down (target is above)

Usage:
    from libredgetpu.tracker import SpotTracker

    with SpotTracker.from_template(64) as tracker:
        x_off, y_off = tracker.track(grayscale_image)
        direction = SpotTracker.offset_to_direction(x_off, y_off)
        if direction != "center":
            print(f"Target is {direction}")
"""

from ..spot_tracker import SpotTracker
from .templates import get_template, list_templates

__all__ = ["SpotTracker", "get_template", "list_templates"]
