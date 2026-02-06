"""LoomingDetector — Edge TPU-accelerated looming detection for collision avoidance.

Computes edge density in 3x3 spatial zones using Sobel filters and pooling.
The ratio of center zone to periphery (tau) indicates approaching objects.

Usage:
    with LoomingDetector.from_template(64) as detector:
        zones = detector.detect(grayscale_image)  # returns [9] float32
        tau = LoomingDetector.compute_tau(zones)
        if tau > 1.2:
            print("Collision warning!")
"""

from typing import List, Optional

import numpy as np

__all__ = ["LoomingDetector"]

from ._base import EdgeTPUModelBase
from ._quantize import dequantize


class LoomingDetector(EdgeTPUModelBase):
    """Edge TPU-accelerated looming detection using edge density in spatial zones."""

    def __init__(self, tflite_path: str, metadata_path: Optional[str] = None,
                 firmware_path: Optional[str] = None):
        """Initialize LoomingDetector from a compiled Edge TPU model.

        Args:
            tflite_path: Path to compiled *_edgetpu.tflite model.
            metadata_path: Path to JSON sidecar with quantization metadata.
                          If None, looks for {tflite_path}.json or infers from TFLite.
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.
        """
        super().__init__(tflite_path, metadata_path=metadata_path,
                         firmware_path=firmware_path)

        # Extract detector-specific metadata
        self._height = self._metadata.get("height", self._input_info.shape[1])
        self._width = self._metadata.get("width", self._input_info.shape[2])
        self._zones = self._metadata.get("zones", 3)
        self._output_count = self._zones * self._zones

    def _default_output_size(self) -> int:
        # Called by base class before our __init__ sets _output_count,
        # so we compute it on the fly from metadata or default.
        zones = 3
        if hasattr(self, '_metadata'):
            zones = self._metadata.get("zones", 3)
        return zones * zones

    @classmethod
    def from_template(cls, size: int, zones: int = 3,
                      firmware_path: Optional[str] = None) -> "LoomingDetector":
        """Create a LoomingDetector from a pre-compiled template.

        Args:
            size: Square image dimension (e.g., 64 for 64x64).
            zones: Number of zones per dimension (default 3 for 3x3=9 zones).
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.

        Returns:
            LoomingDetector instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the specified configuration.
        """
        from .looming.templates import get_template
        tflite_path, json_path = get_template(size, zones)
        return cls(tflite_path, metadata_path=json_path, firmware_path=firmware_path)

    def detect_raw(self, image_bytes: bytes) -> bytes:
        """Run detection with raw uint8 input bytes. Returns raw uint8 output bytes.

        Args:
            image_bytes: Grayscale image as flat uint8 bytes (H*W bytes).

        Returns:
            Raw uint8 output bytes (zones*zones bytes).
        """
        return self._execute_raw(image_bytes)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect edge density in spatial zones.

        Args:
            image: Grayscale image as numpy array.
                   Shape: (H, W) or (H, W, 1) or (1, H, W, 1).
                   dtype: uint8 (preferred) or float32 [0, 255].

        Returns:
            Zone densities as float32 array of shape (zones*zones,).
            For 3x3 zones: [0,1,2,3,4,5,6,7,8] where zone 4 is center.

        Zone layout (3x3):
            [0][1][2]  <- top
            [3][4][5]  <- middle (4 = center)
            [6][7][8]  <- bottom
        """
        image = np.asarray(image)

        # Handle different input shapes
        if image.ndim == 4:
            image = image.squeeze()
        elif image.ndim == 3:
            image = image.squeeze(axis=-1) if image.shape[-1] == 1 else image

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Validate shape
        if image.shape != (self._height, self._width):
            raise ValueError(
                f"Image shape {image.shape} does not match "
                f"expected ({self._height}, {self._width})"
            )

        # Quantize input
        quantized = self._quantize_input(image)

        raw_output = self.detect_raw(quantized.tobytes())

        # Dequantize: uint8 -> float
        out_info = self._output_info
        out_uint8 = np.frombuffer(raw_output, dtype=np.uint8)[:self._output_count]
        zones = dequantize(out_uint8, out_info.scale, out_info.zero_point)

        return zones

    @staticmethod
    def compute_tau(zones: np.ndarray, epsilon: float = 1e-6) -> float:
        """Compute tau (looming ratio) from zone densities.

        Tau = center_density / mean(periphery_densities)

        Args:
            zones: Zone densities array of shape (9,) for 3x3 zones.
            epsilon: Small value to prevent division by zero.

        Returns:
            Tau value:
            - tau > 1.0: Object approaching (center expanding)
            - tau < 1.0: Object receding (center contracting)
            - tau ~ 1.0: Stable or lateral motion
        """
        zones = np.asarray(zones).flatten()
        if len(zones) != 9:
            raise ValueError(f"Expected 9 zones, got {len(zones)}")

        center = zones[4]
        periphery_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        periphery_mean = np.mean(zones[periphery_indices])

        return float(center / max(periphery_mean, epsilon))

    @staticmethod
    def compute_ttc(tau_history: List[float], dt: float,
                    min_samples: int = 3) -> float:
        """Compute time-to-contact from tau history.

        Uses linear regression on tau over time to estimate when contact occurs.

        Args:
            tau_history: List of recent tau values (oldest first).
            dt: Time interval between samples in seconds.
            min_samples: Minimum samples needed for valid estimate.

        Returns:
            Estimated time-to-contact in seconds.
            Returns float('inf') if object is not approaching or insufficient data.
            Returns 0.0 if object has already passed the contact threshold
            (current_tau > 2.0) — treat as "contact imminent or occurred".
        """
        if dt <= 0 or len(tau_history) < min_samples:
            return float('inf')

        tau_arr = np.array(tau_history)
        n = len(tau_arr)

        # d(tau)/dt as the approach rate
        tau_rate = (tau_arr[-1] - tau_arr[0]) / ((n - 1) * dt) if n > 1 else 0

        if tau_rate <= 0:
            return float('inf')

        current_tau = tau_arr[-1]
        if current_tau > 1.0:
            ttc = (2.0 - current_tau) / tau_rate if tau_rate > 0 else float('inf')
            # Negative TTC means object already past contact threshold;
            # clamp to 0.0 to signal "contact imminent or occurred".
            return max(0.0, ttc)
        else:
            return float('inf')

    @property
    def height(self) -> int:
        """Input image height."""
        return self._height

    @property
    def width(self) -> int:
        """Input image width."""
        return self._width

    @property
    def zones_per_dim(self) -> int:
        """Number of zones per dimension."""
        return self._zones
