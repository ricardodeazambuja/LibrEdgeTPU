"""Looming detection for collision avoidance on Edge TPU.

This module provides:
- LoomingDetector: Edge TPU-accelerated edge density computation
- get_template, list_templates: Template discovery functions

Looming detection works by computing edge density in 3x3 spatial zones.
The ratio of center zone to periphery (tau) indicates approaching objects:
- tau > 1.0: Object approaching (center edges expanding)
- tau < 1.0: Object receding (center edges contracting)
- tau ≈ 1.0: Stable or lateral motion

Usage:
    from libredgetpu.looming import LoomingDetector

    with LoomingDetector.from_template(64) as detector:
        zones = detector.detect(grayscale_image)
        tau = LoomingDetector.compute_tau(zones)
        if tau > 1.2:
            print("Collision warning!")
"""

from ..looming_detector import LoomingDetector
from .templates import get_template, list_templates

__all__ = ["LoomingDetector", "get_template", "list_templates"]
