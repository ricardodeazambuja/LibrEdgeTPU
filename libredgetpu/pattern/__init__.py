"""Template matching for Edge TPU.

This module provides:
- PatternTracker: Edge TPU-accelerated template matching via Conv2D correlation
- get_template, list_templates: Template discovery functions

Template matching works by sliding a reference patch across a search image
using Conv2D cross-correlation, then locating the peak via soft argmax.

Output interpretation:
- (0, 0): Template found at image center
- (-1, 0): Template found at left edge
- (+1, 0): Template found at right edge
- (0, -1): Template found at top edge
- (0, +1): Template found at bottom edge

Usage:
    from libredgetpu.pattern import PatternTracker

    with PatternTracker.from_template(128, 16) as tracker:
        x_off, y_off = tracker.track(search_image)
        tracker.set_template(new_patch)  # swap template at runtime
        x_off, y_off = tracker.track(search_image)
"""

from ..pattern_tracker import PatternTracker
from .templates import get_template, list_templates

__all__ = ["PatternTracker", "get_template", "list_templates"]
