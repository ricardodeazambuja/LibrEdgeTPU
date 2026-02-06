"""SpotTracker — Edge TPU-accelerated visual servoing for bright spots and colors.

Computes (x_offset, y_offset) from image center using soft argmax.
Output is directly usable as control error for visual servoing.

Usage:
    with SpotTracker.from_template(64) as tracker:
        x_off, y_off = tracker.track(grayscale_image)
        if abs(x_off) > 0.1 or abs(y_off) > 0.1:
            # Target not centered - steer to compensate
            steer_x = -x_off  # Negative because offset is where target IS
            steer_y = -y_off
"""

from typing import Optional, Tuple, List

import numpy as np

__all__ = ["SpotTracker"]

from ._base import EdgeTPUModelBase
from ._constants import SIGN_BIT_FLIP


class SpotTracker(EdgeTPUModelBase):
    """Edge TPU-accelerated visual servoing using soft argmax centroid."""

    def __init__(self, tflite_path: str, metadata_path: Optional[str] = None,
                 firmware_path: Optional[str] = None):
        """Initialize SpotTracker from a compiled Edge TPU model.

        Args:
            tflite_path: Path to compiled *_edgetpu.tflite model.
            metadata_path: Path to JSON sidecar with quantization metadata.
                          If None, looks for {tflite_path}.json or infers from TFLite.
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.
        """
        super().__init__(tflite_path, metadata_path=metadata_path,
                         firmware_path=firmware_path)

        # Extract tracker-specific metadata
        self._height = self._metadata.get("height", self._input_info.shape[1])
        self._width = self._metadata.get("width", self._input_info.shape[2])
        self._channels = self._metadata.get("channels", self._input_info.shape[3])
        self._variant = self._metadata.get("variant", "bright")
        self._temperature = self._metadata.get("temperature", 0.1)
        # Y offset baked into model to force different quantization from X
        self._y_offset = self._metadata.get("y_offset", 1.0 / self._temperature)

        # Color weight scale for runtime color swapping (color models only).
        self._color_weight_scale = self._metadata.get("color_weight_scale", None)

    def _default_output_size(self) -> int:
        return 2

    @classmethod
    def from_template(cls, size: int, variant: str = "bright",
                      firmware_path: Optional[str] = None) -> "SpotTracker":
        """Create a SpotTracker from a pre-compiled template.

        Args:
            size: Square image dimension (e.g., 64 for 64x64).
            variant: "bright" for grayscale, or "color_red", "color_green", etc.
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.

        Returns:
            SpotTracker instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the specified configuration.
        """
        from .tracker.templates import get_template
        tflite_path, json_path = get_template(size, variant)
        return cls(tflite_path, metadata_path=json_path, firmware_path=firmware_path)

    @classmethod
    def from_color_weights(cls, size: int, weights: List[float],
                           firmware_path: Optional[str] = None) -> "SpotTracker":
        """Create a SpotTracker using the closest pre-compiled color template.

        Matches the given [R, G, B] coefficients against all available color
        templates at the requested size using Euclidean distance.

        Args:
            size: Square image dimension (e.g., 64 for 64x64).
            weights: [R, G, B] coefficients describing the target color.
                     e.g., [1.0, 0.5, -0.5] for orange-ish.
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.

        Returns:
            SpotTracker instance (not yet opened). Use .matched_variant to see
            which color was selected.

        Raises:
            FileNotFoundError: If no color templates exist at the requested size.
        """
        from .tracker.templates import find_closest_color, get_template
        variant, distance = find_closest_color(weights, size)
        tflite_path, json_path = get_template(size, variant)
        instance = cls(tflite_path, metadata_path=json_path, firmware_path=firmware_path)
        instance._matched_variant = variant
        instance._matched_distance = distance
        return instance

    @property
    def matched_variant(self) -> Optional[str]:
        """The color variant selected by from_color_weights(), or None."""
        return getattr(self, "_matched_variant", None)

    @property
    def matched_distance(self) -> Optional[float]:
        """Euclidean distance to the matched color (0.0 = exact), or None."""
        return getattr(self, "_matched_distance", None)

    def track_raw(self, image_bytes: bytes) -> bytes:
        """Run tracking with raw uint8 input bytes. Returns raw output bytes.

        Args:
            image_bytes: Image as flat uint8 bytes (H*W*C bytes).

        Returns:
            Raw output bytes (2 bytes: x_off, y_off).
        """
        return self._execute_raw(image_bytes)

    def track(self, image: np.ndarray, resize: bool = True) -> Tuple[float, float]:
        """Track bright spot or color in image.

        Args:
            image: Input image as numpy array.
                   For bright spot (channels=1): (H, W) or (H, W, 1) or (1, H, W, 1)
                   For color (channels=3): (H, W, 3) or (1, H, W, 3)
                   dtype: uint8 (preferred) or float32 [0, 255].
            resize: If True (default), resize image to model's expected dimensions.

        Returns:
            (x_offset, y_offset) tuple where:
            - x_offset: -1.0 (left) to +1.0 (right), 0.0 = center
            - y_offset: -1.0 (top) to +1.0 (bottom), 0.0 = center
        """
        image = self._normalize_tracker_input(
            image, resize, self._height, self._width, self._channels)
        quantized = self._quantize_input(image)
        raw_output = self.track_raw(quantized.tobytes())
        return self._decode_tracker_output(raw_output, self._y_offset, self._temperature)

    # ── Runtime color swapping ────────────────────────────────────────────

    def set_color(self, weights: List[float]) -> None:
        """Change the color filter at runtime without recompilation.

        Patches the Conv2D color filter weights (3 bytes) in the DarwiNN
        parameter blob and re-uploads to the Edge TPU. Only works with
        color tracker templates (channels=3).

        Args:
            weights: [R, G, B] filter coefficients, typically in [-1, +1].
                     Positive = attract, negative = repel.

        Raises:
            RuntimeError: If not opened, or if this is not a color model.
            ValueError: If weights length != 3 or values exceed quantization range.
        """
        if self._channels != 3:
            raise RuntimeError(
                f"set_color() only works with color tracker templates (channels=3). "
                f"This tracker has channels={self._channels}."
            )
        if not self._cached_mode:
            raise RuntimeError("set_color() requires a cached-mode color template")
        if len(weights) != 3:
            raise ValueError(f"Expected 3 color weights [R, G, B], got {len(weights)}")

        # Determine Conv2D weight scale
        scale = self._color_weight_scale
        if scale is None:
            from .tracker.templates import COLOR_FILTER_WEIGHTS
            variant = self._variant
            if variant.startswith("color_"):
                color_name = variant.split("_", 1)[1]
                preset = COLOR_FILTER_WEIGHTS.get(color_name)
                if preset:
                    max_abs = max(abs(v) for v in preset)
                    scale = max_abs / 127.0
            if scale is None:
                scale = 1.0 / 127.0  # fallback: range [-1, +1]

        # Quantize [R, G, B] to int8
        int8_vals = []
        for i, w in enumerate(weights):
            q = round(w / scale)
            if q < -128 or q > 127:
                raise ValueError(
                    f"Weight[{i}]={w} exceeds quantization range "
                    f"[{-128*scale:.4f}, {127*scale:.4f}] for this template. "
                    f"Use a template with a wider weight_scale."
                )
            int8_vals.append(max(-128, min(127, q)))

        # Patch bytes [0, 1, 2] in the PC param blob: int8 XOR 0x80
        if len(self._pc_params) < 3:
            raise RuntimeError(
                "Invalid color template: parameter blob too small "
                f"({len(self._pc_params)} bytes, need at least 3)"
            )
        blob = bytearray(self._pc_params)
        for i, val in enumerate(int8_vals):
            blob[i] = (val ^ SIGN_BIT_FLIP) & 0xFF
        self._pc_params = bytes(blob)

        # Force re-upload of parameters on next track() call
        if self._driver is not None:
            self._driver._cached_token = 0

    # ── Utility methods ───────────────────────────────────────────────────

    @staticmethod
    def offset_to_direction(x_off: float, y_off: float, threshold: float = 0.1) -> str:
        """Convert offset to a cardinal/ordinal direction string.

        Args:
            x_off: X offset from center (-1 to +1).
            y_off: Y offset from center (-1 to +1).
            threshold: Dead zone threshold for "center" classification.

        Returns:
            Direction string: "center", "left", "right", "up", "down",
            "up-left", "up-right", "down-left", "down-right".
        """
        if abs(x_off) < threshold and abs(y_off) < threshold:
            return "center"

        if abs(x_off) >= threshold:
            x_dir = "right" if x_off > 0 else "left"
        else:
            x_dir = ""

        if abs(y_off) >= threshold:
            y_dir = "down" if y_off > 0 else "up"
        else:
            y_dir = ""

        if y_dir and x_dir:
            return f"{y_dir}-{x_dir}"
        elif y_dir:
            return y_dir
        elif x_dir:
            return x_dir
        else:
            return "center"

    @staticmethod
    def offset_to_servo_error(x_off: float, y_off: float,
                               gain: float = 1.0) -> Tuple[float, float]:
        """Convert offset to servo error signal.

        Args:
            x_off: X offset from center (-1 to +1).
            y_off: Y offset from center (-1 to +1).
            gain: Scaling factor for the error signal.

        Returns:
            (x_error, y_error) tuple for use with PID controller.
        """
        return (-x_off * gain, -y_off * gain)

    @property
    def height(self) -> int:
        """Input image height."""
        return self._height

    @property
    def width(self) -> int:
        """Input image width."""
        return self._width

    @property
    def channels(self) -> int:
        """Input image channels (1 for grayscale, 3 for RGB)."""
        return self._channels

    @property
    def variant(self) -> str:
        """Tracker variant ('bright' or 'color_{color}')."""
        return self._variant

    @property
    def temperature(self) -> float:
        """Softmax temperature used for soft argmax."""
        return self._temperature
