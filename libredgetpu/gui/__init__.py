"""Interactive GUI testing tool for LibrEdgeTPU algorithms.

This module provides a Flask-based web interface for testing all 9 LibrEdgeTPU
algorithms with live webcam input, mouse interaction, and real-time visualization.

Usage:
    python -m libredgetpu.gui [--port 5000] [--camera 0] [--synthetic]

The GUI supports:
- Live webcam capture with algorithm-specific overlays
- Hardware mode (Edge TPU) and synthetic mode (CPU fallback)
- Interactive mouse controls (click, drag for algorithm-specific actions)
- Real-time performance metrics (FPS, latency)
- Screenshot capture with annotations
"""

def create_app(*args, **kwargs):
    """Lazy import to avoid requiring cv2/Flask at import time."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)

__all__ = ["create_app"]
