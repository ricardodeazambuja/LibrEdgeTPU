"""OpticalFlow — Edge TPU-accelerated global optical flow via Gabor features.

Computes a single (vx, vy) displacement vector between two grayscale frames
using Gabor feature extraction on the Edge TPU and CPU-side global correlation
with soft argmax.

Pipeline (standard mode):
    1. Edge TPU: Conv2D with 8 fixed Gabor kernels → feat [H,W,8] (×2 frames)
    2. CPU: Downsample features via block mean (4× by default)
    3. CPU: 81 global correlation scores for ±4 pixel displacements
    4. CPU: Softmax + weighted sum → sub-pixel (vx, vy)

Pipeline (pooled mode — fused_pool):
    1. Edge TPU: Conv2D + AVG_POOL_2D → feat [H/P,W/P,8] (×2 frames)
    2. CPU: 81 global correlation scores (no CPU pooling needed)
    3. CPU: Softmax + weighted sum → sub-pixel (vx, vy)
    USB transfer is P²× smaller (e.g. 4KB vs 64KB for P=4, 64×64 input).

Usage::

    with OpticalFlow.from_template(64) as flow:
        vx, vy = flow.compute(frame_t, frame_t1)
        direction = OpticalFlow.flow_to_direction(vx, vy)

    # Pooled mode (reduced USB, faster on RPi):
    with OpticalFlow.from_template(64, pooled=True) as flow:
        vx, vy = flow.compute(frame_t, frame_t1)
"""

from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided

__all__ = ["OpticalFlow"]

from ._base import EdgeTPUModelBase
from ._quantize import dequantize
from .delegate import relayout_output


class OpticalFlow(EdgeTPUModelBase):
    """Edge TPU-accelerated global optical flow using Gabor features + CPU correlation.

    Computes a single ``(vx, vy)`` displacement vector between two grayscale
    frames.  The Edge TPU extracts 8-channel Gabor features (4 orientations x
    2 scales, 7x7 kernels, ReLU).  The CPU then performs block-sum pooling,
    overlap-normalized global cross-correlation, and soft argmax to produce
    sub-pixel displacement.

    Two modes are supported:

    * **Standard**: Edge TPU outputs full-resolution features ``[H, W, 8]``.
      CPU pools 4x before correlation.
    * **Pooled** (``fused_pool``): AVG_POOL_2D is fused into the Edge TPU
      model, outputting ``[H/P, W/P, 8]`` directly — P²x smaller USB
      transfer.

    Key constructor args: ``search_range`` (displacement grid),
    ``temperature`` (softmax sharpness), ``pool_factor`` (CPU downsampling).
    See module docstring or ``from_template()`` for a usage example.
    """

    def __init__(self, tflite_path: str, metadata_path: Optional[str] = None,
                 firmware_path: Optional[str] = None,
                 search_range: int = 4, temperature: float = 0.1,
                 pool_factor: int = 4):
        """Initialize OpticalFlow from a compiled Edge TPU model.

        Args:
            tflite_path: Path to compiled *_edgetpu.tflite model.
            metadata_path: Path to JSON sidecar with quantization metadata.
                          If None, looks for {tflite_path}.json or infers from TFLite.
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.
            search_range: Maximum displacement in pooled pixels (default 4, → 81 scores).
            temperature: Softmax temperature for soft argmax (default 0.1).
            pool_factor: Spatial downsampling factor for features (default 4).
        """
        super().__init__(tflite_path, metadata_path=metadata_path,
                         firmware_path=firmware_path)

        # Extract metadata
        self._height = self._metadata.get("height", self._input_info.shape[1])
        self._width = self._metadata.get("width", self._input_info.shape[2])
        self._num_filters = self._metadata.get("num_filters", 8)
        self._search_range = search_range
        self._temperature = temperature
        self._pool_factor = pool_factor

        # Detect fused pooling (Gabor+Pool model)
        self._fused_pool = self._metadata.get("fused_pool", 0)
        if self._fused_pool:
            self._out_h = self._height // self._fused_pool
            self._out_w = self._width // self._fused_pool
        else:
            self._out_h = self._height
            self._out_w = self._width

        # Cache the output layer reference for tile relayout.
        # The Edge TPU stores output activations in a tiled memory layout;
        # relayout_output() uses the DarwiNN TileLayout tables to de-scatter
        # the raw bytes into standard YXZ order.
        if self._cached_mode:
            self._output_layer = self._eo_exe.output_layers[0]
        elif self._sa_exe is not None:
            self._output_layer = self._sa_exe.output_layers[0]
        else:
            self._output_layer = None

        # Pre-compute displacement grid: all (dx, dy) pairs in ±search_range
        sr = self._search_range
        displacements = []
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                displacements.append((dx, dy))
        self._displacements = np.array(displacements, dtype=np.float32)  # [81, 2]
        self._n_displacements = len(displacements)

        # Pre-compute overlap pixel counts for each displacement.
        # Used to normalize correlation scores so larger overlaps don't
        # dominate (critical for non-negative ReLU features).
        if self._fused_pool:
            ph, pw = self._out_h, self._out_w
        else:
            ph = self._out_h // self._pool_factor
            pw = self._out_w // self._pool_factor
        nf = self._num_filters
        self._overlap_counts = np.array(
            [(ph - abs(dy)) * (pw - abs(dx)) * nf
             for dy in range(-sr, sr + 1) for dx in range(-sr, sr + 1)],
            dtype=np.float64,
        )

    def _default_output_size(self) -> int:
        # Output is [1, H, W, num_filters] or [1, H/P, W/P, num_filters] uint8
        h = 64
        w = 64
        nf = 8
        if hasattr(self, '_metadata'):
            h = self._metadata.get("height", h)
            w = self._metadata.get("width", w)
            nf = self._metadata.get("num_filters", nf)
            fused_pool = self._metadata.get("fused_pool", 0)
            if fused_pool:
                h = h // fused_pool
                w = w // fused_pool
        return h * w * nf

    @classmethod
    def from_template(cls, size: int, search_range: int = 4,
                      temperature: float = 0.1, pool_factor: int = 4,
                      firmware_path: Optional[str] = None,
                      pooled: bool = False) -> "OpticalFlow":
        """Create an OpticalFlow instance from a pre-compiled template.

        Args:
            size: Square image dimension (e.g., 64 for 64×64).
            search_range: Maximum displacement in pooled pixels (default 4).
            temperature: Softmax temperature for soft argmax (default 0.1).
            pool_factor: Spatial downsampling factor (default 4).
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.
            pooled: If True, use the Gabor+Pool template (reduced USB transfer).

        Returns:
            OpticalFlow instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the specified size.
        """
        if pooled:
            from .optical_flow.templates import get_pooled_template
            tflite_path, json_path = get_pooled_template(size, pool_factor)
        else:
            from .optical_flow.templates import get_template
            tflite_path, json_path = get_template(size)
        return cls(tflite_path, metadata_path=json_path,
                   firmware_path=firmware_path,
                   search_range=search_range, temperature=temperature,
                   pool_factor=pool_factor)

    def _extract_features_uint8(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor features as raw uint8 from the Edge TPU (no dequantize).

        Args:
            image: Grayscale image as numpy array.
                   Shape: (H, W) or (H, W, 1) or (1, H, W, 1).
                   dtype: uint8 (preferred) or float32 [0, 255].

        Returns:
            Feature map as uint8 array of shape (H, W, num_filters).
        """
        image = np.asarray(image)

        # Handle different input shapes
        if image.ndim == 4:
            image = image.squeeze(axis=0)
        if image.ndim == 3:
            image = image.squeeze(axis=-1) if image.shape[-1] == 1 else image
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Validate shape
        if image.shape != (self._height, self._width):
            raise ValueError(
                f"Image shape {image.shape} does not match "
                f"expected ({self._height}, {self._width})"
            )

        # Normalize uint8 [0,255] to float32 [0,1] before quantization
        # The model expects input_scale=1/255, which maps [0,1] float to int8
        image_normalized = image.astype(np.float32) / 255.0
        quantized = self._quantize_input(image_normalized)
        raw_output = self._execute_raw(quantized.tobytes())

        # De-scatter from Edge TPU tiled layout to standard YXZ order
        if self._output_layer is not None and self._output_layer.tile_layout is not None:
            return relayout_output(raw_output, self._output_layer)
        n = self._out_h * self._out_w * self._num_filters
        return np.frombuffer(raw_output, dtype=np.uint8)[:n].reshape(
            self._out_h, self._out_w, self._num_filters)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor features from a grayscale image using the Edge TPU.

        Args:
            image: Grayscale image as numpy array.
                   Shape: (H, W) or (H, W, 1) or (1, H, W, 1).
                   dtype: uint8 (preferred) or float32 [0, 255].

        Returns:
            Feature map as float32 array of shape (H, W, num_filters).
        """
        feat_uint8 = self._extract_features_uint8(image)
        out_info = self._output_info
        return dequantize(
            feat_uint8.ravel(), out_info.scale, out_info.zero_point
        ).reshape(self._out_h, self._out_w, self._num_filters)

    def extract_features_raw(self, image_bytes: bytes) -> bytes:
        """Extract features with raw uint8 input bytes.

        Args:
            image_bytes: Grayscale image as flat uint8 bytes (H*W bytes).

        Returns:
            Raw uint8 output bytes (H*W*num_filters bytes).
        """
        return self._execute_raw(image_bytes)

    def compute(self, frame_t: np.ndarray, frame_t1: np.ndarray) -> Tuple[float, float]:
        """Compute global optical flow between two grayscale frames.

        Uses integer arithmetic internally (no float conversion until
        soft argmax) for maximum speed on weak CPUs.

        Args:
            frame_t: Reference frame (time t). Shape: (H, W), dtype: uint8.
            frame_t1: Current frame (time t+1). Shape: (H, W), dtype: uint8.

        Returns:
            (vx, vy) displacement in pooled pixels (sub-pixel via soft argmax).
            Positive vx = rightward motion, positive vy = downward motion.
        """
        # Get raw uint8 features — no dequantize
        feat_t_u8 = self._extract_features_uint8(frame_t)
        feat_t1_u8 = self._extract_features_uint8(frame_t1)

        return self._compute_from_uint8(feat_t_u8, feat_t1_u8)

    def compute_raw(self, frame_t_bytes: bytes,
                    frame_t1_bytes: bytes) -> Tuple[float, float]:
        """Compute flow from raw uint8 bytes.

        Uses integer arithmetic internally for speed.

        Args:
            frame_t_bytes: Reference frame as flat uint8 bytes (H*W).
            frame_t1_bytes: Current frame as flat uint8 bytes (H*W).

        Returns:
            (vx, vy) displacement in pooled pixels.
        """
        raw_t = self._execute_raw(frame_t_bytes)
        raw_t1 = self._execute_raw(frame_t1_bytes)

        # De-scatter from Edge TPU tiled layout to standard YXZ order
        if self._output_layer is not None and self._output_layer.tile_layout is not None:
            feat_t_u8 = relayout_output(raw_t, self._output_layer)
            feat_t1_u8 = relayout_output(raw_t1, self._output_layer)
        else:
            n = self._out_h * self._out_w * self._num_filters
            shape = (self._out_h, self._out_w, self._num_filters)
            feat_t_u8 = np.frombuffer(raw_t, dtype=np.uint8)[:n].reshape(shape)
            feat_t1_u8 = np.frombuffer(raw_t1, dtype=np.uint8)[:n].reshape(shape)

        return self._compute_from_uint8(feat_t_u8, feat_t1_u8)

    def _compute_from_uint8(self, feat_t_u8: np.ndarray,
                            feat_t1_u8: np.ndarray) -> Tuple[float, float]:
        """Flow computation from uint8 feature maps.

        Pools if needed, correlates, and normalizes by overlap area so that
        larger overlaps don't dominate the score.  Mean subtraction is NOT
        used: for all-positive ReLU features the raw cross-correlation
        already peaks at the true displacement after overlap normalization
        (the matched position gives E[X²] per pixel, any other gives the
        smaller E[X·X_shifted]).
        """
        zp = np.int16(self._output_info.zero_point)

        # uint8 → int16 (subtract zero point)
        feat_t_int = feat_t_u8.astype(np.int16) - zp
        feat_t1_int = feat_t1_u8.astype(np.int16) - zp

        if self._fused_pool:
            # Features already pooled by the TPU
            feat_t_f = feat_t_int.astype(np.float32)
            feat_t1_f = feat_t1_int.astype(np.float32)
        else:
            # Block-sum pooling (int32 → float32)
            feat_t_f = self._pool_features_int(feat_t_int).astype(np.float32)
            feat_t1_f = self._pool_features_int(feat_t1_int).astype(np.float32)

        # Correlate and normalize by overlap pixel count
        corr = self._global_correlation(feat_t_f, feat_t1_f).astype(np.float64)
        corr /= self._overlap_counts

        return self._soft_argmax(corr.astype(np.float32))

    def _pool_features(self, feat: np.ndarray) -> np.ndarray:
        """Downsample feature map via block mean (float path).

        Args:
            feat: Feature map of shape (H, W, C), float32.

        Returns:
            Downsampled features of shape (H/P, W/P, C) where P = pool_factor.
        """
        h, w, c = feat.shape
        p = self._pool_factor

        if h % p == 0 and w % p == 0:
            return feat.reshape(h // p, p, w // p, p, c).mean(axis=(1, 3))
        else:
            h_trunc = (h // p) * p
            w_trunc = (w // p) * p
            cropped = feat[:h_trunc, :w_trunc, :]
            return cropped.reshape(h_trunc // p, p, w_trunc // p, p, c).mean(axis=(1, 3))

    def _pool_features_int(self, feat: np.ndarray) -> np.ndarray:
        """Downsample feature map via block-sum (integer path, no float).

        Args:
            feat: Feature map of shape (H, W, C), int16.

        Returns:
            Downsampled features of shape (H/P, W/P, C) as int32.
        """
        h, w, c = feat.shape
        p = self._pool_factor

        if h % p == 0 and w % p == 0:
            return feat.reshape(h // p, p, w // p, p, c).sum(axis=(1, 3), dtype=np.int32)
        else:
            h_trunc = (h // p) * p
            w_trunc = (w // p) * p
            cropped = feat[:h_trunc, :w_trunc, :]
            return cropped.reshape(
                h_trunc // p, p, w_trunc // p, p, c
            ).sum(axis=(1, 3), dtype=np.int32)

    def _global_correlation(self, feat_t: np.ndarray,
                            feat_t1: np.ndarray) -> np.ndarray:
        """Compute global correlation scores for all displacement candidates.

        For each (dx, dy) in ±search_range, shift feat_t by (dx, dy) and
        compute the sum of element-wise products with feat_t1 over the
        overlapping region.  A peak at (dx, dy) means the scene content
        moved by (dx, dy) pixels between frame_t and frame_t1.

        Uses vectorized pad + stride-tricks + einsum instead of a Python
        loop — ~10-50× faster on CPUs without SIMD (e.g. RPi Zero).

        Args:
            feat_t: Reference features (H', W', C).
            feat_t1: Current features (H', W', C).

        Returns:
            Correlation scores array of shape (n_displacements,).
        """
        h, w, c = feat_t.shape
        sr = self._search_range
        side = 2 * sr + 1

        # Pad feat_t so shifted overlaps auto-mask boundary with zeros
        padded = np.pad(feat_t, ((sr, sr), (sr, sr), (0, 0)), mode='constant')

        # Sliding window view: shape (side, side, h, w, c)
        # view[j, i] = padded[j:j+h, i:i+w, :] — zero-copy
        s = padded.strides
        view = as_strided(padded,
                          shape=(side, side, h, w, c),
                          strides=(s[0], s[1], s[0], s[1], s[2]))

        # Batched dot products: corr_map[j, i] = sum(view[j,i] * feat_t1)
        # Use int64 accumulation for integer inputs to prevent overflow
        if np.issubdtype(feat_t1.dtype, np.integer):
            corr_map = np.einsum('ijhwc,hwc->ij', view, feat_t1,
                                 dtype=np.int64)
        else:
            corr_map = np.einsum('ijhwc,hwc->ij', view, feat_t1)

        # corr_map[j, i] corresponds to displacement (dx=i-sr, dy=j-sr).
        # Flip both axes so that displacement ordering matches the array.
        return corr_map[::-1, ::-1].ravel()

    def _soft_argmax(self, corr: np.ndarray) -> Tuple[float, float]:
        """Compute sub-pixel displacement via softmax-weighted sum.

        Args:
            corr: Correlation scores of shape (n_displacements,).

        Returns:
            (vx, vy) sub-pixel displacement.
        """
        # Numerical stability: subtract max before exp
        corr_shifted = corr - np.max(corr)
        weights = np.exp(corr_shifted / max(self._temperature, 1e-6))
        total = np.sum(weights)
        if total < 1e-12:
            return 0.0, 0.0
        weights /= total

        # Weighted sum of displacements
        vx = float(np.sum(weights * self._displacements[:, 0]))
        vy = float(np.sum(weights * self._displacements[:, 1]))
        return vx, vy

    @staticmethod
    def flow_to_direction(vx: float, vy: float, threshold: float = 0.3) -> str:
        """Classify flow vector into a direction label.

        Args:
            vx: Horizontal flow (positive = rightward).
            vy: Vertical flow (positive = downward).
            threshold: Minimum magnitude to register a direction.

        Returns:
            Direction string: "left", "right", "up", "down",
            "up-left", "up-right", "down-left", "down-right", or "center".
        """
        mag = (vx ** 2 + vy ** 2) ** 0.5
        if mag < threshold:
            return "center"

        parts = []
        if vy < -threshold:
            parts.append("up")
        elif vy > threshold:
            parts.append("down")
        if vx < -threshold:
            parts.append("left")
        elif vx > threshold:
            parts.append("right")

        return "-".join(parts) if parts else "center"

    @property
    def height(self) -> int:
        """Input image height."""
        return self._height

    @property
    def width(self) -> int:
        """Input image width."""
        return self._width

    @property
    def search_range(self) -> int:
        """Maximum displacement search range in pooled pixels."""
        return self._search_range

    @property
    def num_filters(self) -> int:
        """Number of Gabor filter channels."""
        return self._num_filters

    @property
    def pool_factor(self) -> int:
        """Spatial downsampling factor."""
        return self._pool_factor

    @property
    def fused_pool(self) -> int:
        """Fused pool factor (0 if pooling is done on CPU, >0 if on TPU)."""
        return self._fused_pool
