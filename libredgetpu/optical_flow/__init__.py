"""Optical flow for ego-motion estimation on Edge TPU.

This module provides:
- OpticalFlow: Edge TPU-accelerated Gabor feature extraction + CPU correlation
- get_template, list_templates: Template discovery functions

Optical flow works by extracting multi-orientation Gabor features on the Edge TPU,
then computing global displacement via CPU-side correlation and soft argmax.

Usage:
    from libredgetpu.optical_flow import OpticalFlow

    with OpticalFlow.from_template(64) as flow:
        vx, vy = flow.compute(frame_t, frame_t1)
        direction = OpticalFlow.flow_to_direction(vx, vy)
"""

from ..optical_flow_module import OpticalFlow
from .templates import get_template, list_templates

__all__ = ["OpticalFlow", "get_template", "list_templates"]
