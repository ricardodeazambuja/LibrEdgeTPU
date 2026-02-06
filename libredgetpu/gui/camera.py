"""Webcam capture and synthetic frame generators for GUI testing."""

import cv2
import numpy as np
from typing import Optional, Tuple
import time


class Camera:
    """Base class for camera/frame sources."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame.

        Returns:
            (success, frame) tuple. Frame is BGR uint8 numpy array if success=True.
        """
        raise NotImplementedError

    def release(self):
        """Release camera resources."""
        pass

    def get_resolution(self) -> Tuple[int, int]:
        """Get current frame resolution (width, height)."""
        raise NotImplementedError


class RealCamera(Camera):
    """Wrapper around OpenCV VideoCapture for real webcam."""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """Initialize webcam capture.

        Args:
            camera_id: Camera device index
            width: Requested frame width
            height: Requested frame height
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Verify actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Warmup: Some cameras need time to initialize and flush buffers
        # Read and discard a few frames to ensure camera is ready
        for i in range(5):
            self.cap.read()
            time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from webcam."""
        ret, frame = self.cap.read()

        # If read fails, camera might have gone stale - retry with delays
        if not ret:
            for i in range(3):
                time.sleep(0.1)
                ret, frame = self.cap.read()
                if ret:
                    break

        return ret, frame

    def release(self):
        """Release webcam."""
        if self.cap is not None:
            self.cap.release()

    def get_resolution(self) -> Tuple[int, int]:
        """Get webcam resolution."""
        return (self.width, self.height)


class SyntheticCamera(Camera):
    """Generates synthetic test patterns for offline testing."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        pattern: str = "noise",
        fps: float = 30.0
    ):
        """Initialize synthetic frame generator.

        Args:
            width: Frame width
            height: Frame height
            pattern: Pattern type ("noise", "checkerboard", "rotating", "panning", "wandering_dot")
            fps: Target frame rate (for time-based animations)
        """
        self.width = width
        self.height = height
        self.pattern = pattern
        self.fps = fps
        self.frame_count = 0
        self.start_time = time.time()

        # For wandering_dot: create static textured background
        if pattern == "wandering_dot":
            self._static_background = np.random.randint(
                30, 90, (height, width, 3), dtype=np.uint8
            )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Generate a synthetic frame.

        Each read() call advances the animation by one frame-time (1/fps),
        ensuring motion is visible even with rapid consecutive reads.
        """
        if self.pattern == "noise":
            frame = self._generate_noise()
        elif self.pattern == "checkerboard":
            frame = self._generate_checkerboard()
        elif self.pattern == "rotating":
            frame = self._generate_rotating()
        elif self.pattern == "panning":
            frame = self._generate_panning()
        elif self.pattern == "wandering_dot":
            frame = self._generate_wandering_dot()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

        self.frame_count += 1
        # Advance start_time to ensure animation progresses even with rapid reads
        self.start_time -= 1.0 / self.fps
        return True, frame

    def get_resolution(self) -> Tuple[int, int]:
        """Get frame resolution."""
        return (self.width, self.height)

    def _get_animation_time(self) -> float:
        """Get current animation time in seconds."""
        return time.time() - self.start_time

    def _generate_noise(self) -> np.ndarray:
        """Generate random noise."""
        return np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)

    def _generate_checkerboard(self) -> np.ndarray:
        """Generate static checkerboard pattern."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        square_size = 40

        for i in range(0, self.height, square_size):
            for j in range(0, self.width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    frame[i:i+square_size, j:j+square_size] = 255

        return frame

    def _generate_rotating(self) -> np.ndarray:
        """Generate rotating pattern for optical flow testing."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        t = self._get_animation_time()
        angle = (t * 180) % 360  # 180 degrees per second (fast for optical flow)

        # Draw rotating line pattern
        cx, cy = self.width // 2, self.height // 2
        for offset_angle in range(0, 360, 30):
            total_angle = angle + offset_angle
            rad = np.deg2rad(total_angle)
            x1 = int(cx + 200 * np.cos(rad))
            y1 = int(cy + 200 * np.sin(rad))
            cv2.line(frame, (cx, cy), (x1, y1), (255, 255, 255), 2)

        return frame

    def _generate_panning(self) -> np.ndarray:
        """Generate horizontally panning pattern for optical flow testing.

        Uses a wider non-repeating textured pattern (3x wider than viewport)
        that pans across the visible window. This avoids correspondence
        ambiguity from periodic patterns.

        Stripe widths are 40-120 pixels so features survive 10× downscale
        (640→64) without aliasing.  Each output pixel at 64×64 covers ~10
        input pixels, so stripes must be ≥20 pixels to stay above Nyquist.
        """
        t = self._get_animation_time()

        # Create a wide canvas (3x viewport width) with non-repeating structure
        canvas_width = self.width * 3
        canvas = np.zeros((self.height, canvas_width, 3), dtype=np.uint8)

        # Add varied vertical stripes wide enough to survive 10× downscale.
        # Width 40-120 pixels → 4-12 pixels at 64×64 → 1-3 pooled pixels.
        np.random.seed(42)  # Deterministic pattern
        x_pos = 0
        while x_pos < canvas_width:
            stripe_width = np.random.randint(40, 120)
            # Varied intensities for richer features after downscale
            intensity = np.random.randint(20, 240)
            cv2.rectangle(canvas, (x_pos, 0), (x_pos + stripe_width, self.height),
                         (intensity, intensity, intensity), -1)
            x_pos += stripe_width

        # Add non-periodic texture: thick horizontal bars at varied Y positions.
        # Thickness 6 so they survive vertical downscale (480→64, ~7.5× factor).
        for x in range(0, canvas_width, 120):
            for y_offset in [40, 120, 240, 360]:
                y = (y_offset + (x // 200) * 37) % self.height
                bar_length = 60 + (x % 80)
                cv2.rectangle(canvas, (x, y), (min(x + bar_length, canvas_width), y + 6),
                             (200, 200, 200), -1)

        # Pan across the canvas (wraps after full traverse)
        # Speed: 600 pixels/sec at 640px wide.
        # After 10× downscale (640→64): 600/10/30 = 2 px/frame at 64×64.
        # With pool_factor=2: 2/2 = 1 pooled pixel/frame → detectable.
        # Use negative to make scene move rightward (viewport pans leftward over canvas)
        offset = int((-t * 600) % canvas_width)

        # Extract viewport window
        x_start = offset % canvas_width
        if x_start + self.width <= canvas_width:
            frame = canvas[:, x_start:x_start + self.width].copy()
        else:
            # Wrap around at edge
            part1_width = canvas_width - x_start
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :part1_width] = canvas[:, x_start:]
            frame[:, part1_width:] = canvas[:, :self.width - part1_width]

        return frame

    def _generate_wandering_dot(self) -> np.ndarray:
        """Generate wandering Gaussian dot for spot tracker testing.

        Uses a static textured background to provide features for optical flow.
        The bright dot dominates for spot tracking, but the background texture
        allows optical flow and compass to track the scene motion.
        """
        # Use static textured background (consistent across frames for optical flow)
        frame = self._static_background.copy()

        t = self._get_animation_time()
        # Lissajous curve (faster motion for optical flow)
        x = int(self.width / 2 + (self.width / 4) * np.sin(t * 2.0))
        y = int(self.height / 2 + (self.height / 4) * np.cos(t * 2.8))

        # Draw Gaussian dot (bright, dominates for spot tracking)
        dot = self._make_gaussian_dot(x, y, sigma=20, intensity=255)
        frame[:, :, 1] = np.maximum(frame[:, :, 1], dot)  # Green channel, overlay

        return frame

    def _make_gaussian_dot(
        self,
        cx: int,
        cy: int,
        sigma: float = 20,
        intensity: int = 255
    ) -> np.ndarray:
        """Generate a Gaussian dot at (cx, cy).

        Args:
            cx: Center X coordinate
            cy: Center Y coordinate
            sigma: Gaussian standard deviation
            intensity: Peak intensity (0-255)

        Returns:
            Single-channel uint8 image with Gaussian dot
        """
        y, x = np.ogrid[:self.height, :self.width]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        return (gaussian * intensity).astype(np.uint8)


def create_camera(
    camera_id: Optional[int] = None,
    synthetic: bool = False,
    pattern: str = "noise",
    width: int = 640,
    height: int = 480
) -> Camera:
    """Factory function to create a camera instance.

    Args:
        camera_id: Webcam device ID (0, 1, ...). If None, auto-detect.
        synthetic: Force synthetic mode even if webcam available
        pattern: Synthetic pattern type
        width: Frame width
        height: Frame height

    Returns:
        Camera instance (RealCamera or SyntheticCamera)
    """
    if synthetic:
        return SyntheticCamera(width, height, pattern)

    # Try to open real camera
    if camera_id is None:
        camera_id = 0

    try:
        return RealCamera(camera_id, width, height)
    except RuntimeError:
        # Fallback to synthetic if webcam fails
        print(f"Warning: Failed to open camera {camera_id}, using synthetic mode")
        return SyntheticCamera(width, height, pattern)
