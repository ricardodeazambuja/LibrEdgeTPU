"""Shared utilities for standalone robotics example scripts.

Provides webcam loop, FPS tracking, display, and resize helpers so each
example script stays focused on its algorithm.
"""

import argparse
import signal
import sys
import time
from collections import deque

import cv2
import numpy as np


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add webcam and display arguments shared by all examples."""
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera frame height (default: 480)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — no OpenCV window")


class WebcamLoop:
    """Iterator yielding BGR frames with FPS tracking and clean shutdown.

    Usage::

        loop = WebcamLoop(args)
        for frame in loop:
            # ... process frame ...
            loop.show(annotated_frame)
            loop.print_metrics({"vx": 1.2}, latency_ms=0.5)
        loop.cleanup()
    """

    def __init__(self, args: argparse.Namespace):
        self._no_display = args.no_display
        self._cap = cv2.VideoCapture(args.camera)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {args.camera}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        # Warmup: flush stale frames (matches gui/camera.py pattern)
        for _ in range(5):
            self._cap.read()
            time.sleep(0.1)

        # Rolling FPS tracker
        self._timestamps = deque(maxlen=30)
        self._running = True

        # Clean shutdown on Ctrl+C (critical: USB device must be released)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        self._running = False

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if not self._running:
            raise StopIteration

        ret, frame = self._cap.read()
        if not ret:
            # Retry a few times on failure
            for _ in range(3):
                time.sleep(0.1)
                ret, frame = self._cap.read()
                if ret:
                    break
            if not ret:
                raise StopIteration

        self._timestamps.append(time.perf_counter())
        return frame

    @property
    def fps(self) -> float:
        """Rolling FPS over the last 30 frames."""
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / dt if dt > 0 else 0.0

    def show(self, frame: np.ndarray, window_name: str = "LibrEdgeTPU") -> None:
        """Display frame in an OpenCV window (skipped in headless mode).

        Returns False if the user pressed 'q' to quit.
        """
        if self._no_display:
            return
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._running = False

    def get_key(self) -> int:
        """Return the last key pressed (0xFF if none), or -1 in headless mode."""
        if self._no_display:
            return -1
        return cv2.waitKey(1) & 0xFF

    def print_metrics(self, metrics: dict, latency_ms: float) -> None:
        """Print single-line metrics to terminal with carriage return."""
        parts = [f"FPS: {self.fps:.1f}", f"latency: {latency_ms:.2f} ms"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.3f}")
            else:
                parts.append(f"{k}: {v}")
        line = " | ".join(parts)
        sys.stdout.write(f"\r{line}    ")
        sys.stdout.flush()

    def cleanup(self) -> None:
        """Release camera and destroy windows."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if not self._no_display:
            cv2.destroyAllWindows()
        # Final newline after \r output
        print()


def resize_gray(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """Convert BGR frame to grayscale and resize with anti-aliasing.

    Uses INTER_AREA to avoid aliasing artifacts on large downscale
    (e.g., 640->64 = 10x).

    Args:
        frame_bgr: BGR uint8 frame from webcam.
        size: Target square dimension.

    Returns:
        Grayscale uint8 array of shape (size, size).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def resize_for_tracker(frame_bgr: np.ndarray, size: int, channels: int) -> np.ndarray:
    """Resize frame for a tracker, converting to grayscale only if needed.

    Args:
        frame_bgr: BGR uint8 frame from webcam.
        size: Target square dimension.
        channels: Expected channel count (1 for grayscale, 3 for color).

    Returns:
        uint8 array of shape (size, size) or (size, size, 3).
    """
    if channels == 1:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = frame_bgr
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def draw_text(img: np.ndarray, text: str, pos: tuple,
              color: tuple = (255, 255, 255), scale: float = 0.5) -> None:
    """Draw text with black background for visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, 1)
