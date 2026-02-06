"""VisualCompass — Yaw estimation via Edge TPU optical flow.

Thin wrapper around :class:`OpticalFlow` that converts horizontal
displacement (vx) into a yaw angle in degrees using the camera's
field-of-view.

The entire Gabor + correlation pipeline is reused from OpticalFlow;
this module adds only the FOV-based angular conversion and a
domain-specific API for heading estimation.

Usage::

    with VisualCompass.from_template(64, fov_deg=90, pooled=True) as compass:
        yaw = compass.compute_yaw(frame_t, frame_t1)        # degrees
        yaw, vx, vy = compass.compute(frame_t, frame_t1)    # full output

Sign convention: positive yaw = rightward camera rotation
(positive vx = rightward scene motion).
"""

from typing import Optional, Tuple

import numpy as np

from .optical_flow_module import OpticalFlow

__all__ = ["VisualCompass"]


class VisualCompass:
    """Yaw estimation wrapper around :class:`OpticalFlow`.

    Converts the horizontal displacement ``vx`` (in pooled pixels) into
    a yaw angle via::

        deg_per_pooled_px = fov_deg * effective_pool / image_width
        yaw_deg = vx * deg_per_pooled_px

    Terminology:

    * ``pool_factor`` — CPU-side block-sum downsampling factor applied to
      full-resolution Gabor features (standard mode only, default 4).
    * ``fused_pool`` — AVG_POOL_2D factor baked into the Edge TPU model
      (pooled mode).  When > 0, the TPU outputs already-downsampled
      features and no CPU pooling is needed.
    * ``effective_pool`` — the active downsampling factor: ``fused_pool``
      when > 0, otherwise ``pool_factor``.  Used to convert pixel
      displacement to degrees.
    """

    def __init__(self, flow: OpticalFlow, fov_deg: float) -> None:
        """Wrap an existing OpticalFlow instance with FOV-based yaw conversion.

        Args:
            flow: An :class:`OpticalFlow` instance (opened or unopened).
            fov_deg: Horizontal field-of-view of the camera in degrees.
                     Must be in (0, 360].

        Raises:
            ValueError: If *fov_deg* is not in (0, 360].
        """
        if fov_deg <= 0 or fov_deg > 360:
            raise ValueError(
                f"fov_deg must be in (0, 360], got {fov_deg}"
            )

        self._flow = flow
        self._fov_deg = float(fov_deg)
        self._owns_flow = False  # True when created via from_template

        effective_pool = flow.fused_pool if flow.fused_pool > 0 else flow.pool_factor
        self._deg_per_pooled_px = self._fov_deg * effective_pool / flow.width

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_template(cls, size: int, fov_deg: float, *,
                      pooled: bool = True,
                      search_range: int = 4,
                      temperature: float = 0.1,
                      pool_factor: int = 4,
                      firmware_path: Optional[str] = None) -> "VisualCompass":
        """Create a VisualCompass from a pre-compiled OpticalFlow template.

        Args:
            size: Square image dimension (e.g. 64).
            fov_deg: Horizontal field-of-view in degrees (0, 360].
            pooled: Use Gabor+Pool template for reduced USB transfer (default True).
            search_range: Maximum displacement in pooled pixels (default 4).
            temperature: Softmax temperature (default 0.1).
            pool_factor: CPU downsampling factor (default 4).
            firmware_path: Edge TPU firmware path; auto-downloaded if None.

        Returns:
            VisualCompass instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the specified size.
        """
        flow = OpticalFlow.from_template(
            size,
            search_range=search_range,
            temperature=temperature,
            pool_factor=pool_factor,
            firmware_path=firmware_path,
            pooled=pooled,
        )
        obj = cls(flow, fov_deg)
        obj._owns_flow = True
        return obj

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_yaw(self, frame_t: np.ndarray, frame_t1: np.ndarray) -> float:
        """Compute yaw angle between two frames.

        Args:
            frame_t: Reference frame (time t). Shape (H, W), dtype uint8.
            frame_t1: Current frame (time t+1). Shape (H, W), dtype uint8.

        Returns:
            Yaw angle in degrees.  Positive = rightward camera rotation.
        """
        vx, _ = self._flow.compute(frame_t, frame_t1)
        return vx * self._deg_per_pooled_px

    def compute(self, frame_t: np.ndarray,
                frame_t1: np.ndarray) -> Tuple[float, float, float]:
        """Compute yaw angle and raw flow displacements.

        Args:
            frame_t: Reference frame. Shape (H, W), dtype uint8.
            frame_t1: Current frame. Shape (H, W), dtype uint8.

        Returns:
            ``(yaw_deg, vx, vy)`` — yaw in degrees, displacements in pooled pixels.
        """
        vx, vy = self._flow.compute(frame_t, frame_t1)
        return vx * self._deg_per_pooled_px, vx, vy

    # ------------------------------------------------------------------
    # Lifecycle (delegates to flow)
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the underlying OpticalFlow device."""
        self._flow.open()

    def close(self) -> None:
        """Close the underlying OpticalFlow device."""
        self._flow.close()

    def __enter__(self) -> "VisualCompass":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fov_deg(self) -> float:
        """Horizontal field-of-view in degrees."""
        return self._fov_deg

    @property
    def deg_per_pooled_px(self) -> float:
        """Degrees of yaw per pooled-pixel displacement."""
        return self._deg_per_pooled_px

    @property
    def flow(self) -> OpticalFlow:
        """Underlying OpticalFlow instance."""
        return self._flow

    @property
    def width(self) -> int:
        """Input image width."""
        return self._flow.width

    @property
    def height(self) -> int:
        """Input image height."""
        return self._flow.height

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def yaw_to_direction(yaw_deg: float, threshold_deg: float = 1.0) -> str:
        """Classify a yaw angle into a direction label.

        Args:
            yaw_deg: Yaw angle in degrees (positive = right).
            threshold_deg: Minimum magnitude to register a direction.

        Returns:
            ``"left"``, ``"right"``, or ``"center"``.
        """
        if yaw_deg < -threshold_deg:
            return "left"
        elif yaw_deg > threshold_deg:
            return "right"
        return "center"
