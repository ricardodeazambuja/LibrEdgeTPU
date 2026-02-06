"""CPU replica of the Edge TPU optical flow pipeline.

Faithfully reproduces the Edge TPU's integer arithmetic on CPU, step by step:

1. QUANTIZE (uint8 -> int8)
2. DEPTHWISE_CONV_2D (int8 input x int8 weights -> int32 acc -> int8 output, fused ReLU)
3. AVG_POOL_2D (int8, 4x4 stride 4)
4. QUANTIZE (int8 -> uint8)

Then reuses OpticalFlow's CPU-side correlation + soft argmax for the final
(vx, vy) output.

This mode exists to isolate hardware/toolchain issues: if the CPU replica
produces different output from the Edge TPU, the bug is in hardware
execution or output relayout; if they match, the bug is in the shared
post-processing or quantization parameters.

Usage::

    engine = CPUReplicaOpticalFlow(height=64, width=64)
    vx, vy = engine.compute(prev_gray_uint8, curr_gray_uint8)
    intermediates = engine.get_intermediates()
"""

import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["CPUReplicaOpticalFlow", "ReplicaQuantParams"]


@dataclass
class ReplicaQuantParams:
    """Tweakable quantization parameters for the CPU replica.

    Defaults match ``build_optical_flow_pooled(64, 64)`` from tflite_builder.py.
    Set any field to override the default for debugging.
    """
    input_scale: float = 1.0 / 255.0
    input_zp: int = 0
    q_int8_scale: float = 1.0 / 255.0
    q_int8_zp: int = -128
    # Auto-computed from Gabor kernels if left at 0.0
    conv_output_scale: float = 0.0
    conv_output_zp: int = -128
    pool_output_scale: float = 0.0
    pool_output_zp: int = -128
    final_output_scale: float = 0.0
    final_output_zp: int = 0
    ksize: int = 7
    orientations: int = 4
    sigmas: Tuple[float, ...] = (1.5, 3.0)
    pool_factor: int = 4
    search_range: int = 4
    temperature: float = 0.1


class CPUReplicaOpticalFlow:
    """CPU replica of the Edge TPU Gabor + Pool optical flow pipeline.

    Performs the full integer pipeline:
    uint8 image -> QUANTIZE -> DEPTHWISE_CONV_2D -> AVG_POOL_2D -> QUANTIZE -> uint8 features.
    Then runs the same correlation + soft argmax as OpticalFlow._compute_from_uint8().
    """

    def __init__(self, height: int = 64, width: int = 64,
                 params: Optional[ReplicaQuantParams] = None,
                 verbose: bool = False):
        """Initialize the CPU replica pipeline.

        Args:
            height: Input image height.
            width: Input image width.
            params: Quantization parameters. Uses defaults if None.
            verbose: Print stage-by-stage min/max/mean to stderr.
        """
        if params is None:
            params = ReplicaQuantParams()
        self._params = params
        self._height = height
        self._width = width
        self._verbose = verbose

        # Import and generate Gabor kernels (same as tflite_builder)
        from ..tflite_builder import _generate_gabor_kernels
        self._n_filters = params.orientations * len(params.sigmas)

        gabor_hwio = _generate_gabor_kernels(
            params.ksize, params.orientations, params.sigmas
        )
        # Shape: [ksize, ksize, 1, n_filters] -> [1, ksize, ksize, n_filters]
        gabor_float = np.transpose(gabor_hwio, (2, 0, 1, 3))

        # Quantize weights per-channel (same as tflite_builder)
        self._per_ch_weight_scales = []
        self._weights_int8 = np.zeros_like(gabor_float, dtype=np.int8)

        for ch in range(self._n_filters):
            kernel_ch = gabor_float[0, :, :, ch]
            ch_abs_max = float(np.max(np.abs(kernel_ch)))
            ch_scale = max(ch_abs_max, 1e-6) / 127.0
            self._per_ch_weight_scales.append(ch_scale)
            self._weights_int8[0, :, :, ch] = np.clip(
                np.round(kernel_ch / ch_scale), -127, 127
            ).astype(np.int8)

        # Bias: zero int32
        self._bias_int32 = np.zeros(self._n_filters, dtype=np.int32)

        # Auto-compute output scales if left at 0
        mean_weight_scale = float(np.mean(self._per_ch_weight_scales))
        ksize = params.ksize
        worst_case_acc = ksize * ksize * 127.0 * 1.5
        auto_conv_output_max = worst_case_acc * params.q_int8_scale * mean_weight_scale
        auto_conv_output_scale = auto_conv_output_max / 127.0

        if params.conv_output_scale == 0.0:
            params.conv_output_scale = auto_conv_output_scale
        if params.pool_output_scale == 0.0:
            params.pool_output_scale = params.conv_output_scale
        if params.final_output_scale == 0.0:
            params.final_output_scale = params.pool_output_scale

        # Compute per-channel requantization multipliers (float64 for precision)
        self._conv_requant_M = np.array([
            np.float64(params.q_int8_scale) * np.float64(ws)
            / np.float64(params.conv_output_scale)
            for ws in self._per_ch_weight_scales
        ], dtype=np.float64)

        # Pool output -> final uint8 requantization
        self._final_requant_M = np.float64(
            params.pool_output_scale / params.final_output_scale
        )

        # Pooled output dimensions
        self._out_h = height // params.pool_factor
        self._out_w = width // params.pool_factor

        # Pre-compute displacement grid and overlap counts
        # (mirrors OpticalFlow.__init__)
        sr = params.search_range
        displacements = []
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                displacements.append((dx, dy))
        self._displacements = np.array(displacements, dtype=np.float32)

        ph, pw = self._out_h, self._out_w
        nf = self._n_filters
        self._overlap_counts = np.array(
            [(ph - abs(dy)) * (pw - abs(dx)) * nf
             for dy in range(-sr, sr + 1) for dx in range(-sr, sr + 1)],
            dtype=np.float64,
        )

        # Intermediates storage
        self._intermediates: Dict[str, np.ndarray] = {}

    def extract_features_uint8(self, image_uint8: np.ndarray) -> np.ndarray:
        """Full pipeline: uint8 (H,W) -> uint8 (H/P, W/P, n_filters).

        Args:
            image_uint8: Grayscale uint8 image of shape (H, W).

        Returns:
            Feature map as uint8 array of shape (H/P, W/P, n_filters).
        """
        p = self._params
        H, W = self._height, self._width
        image_uint8 = np.asarray(image_uint8, dtype=np.uint8)

        if image_uint8.shape != (H, W):
            raise ValueError(
                f"Image shape {image_uint8.shape} != expected ({H}, {W})"
            )

        self._intermediates["input_uint8"] = image_uint8.copy()

        # Stage 1: QUANTIZE (uint8 -> int8)
        # M = input_scale / q_int8_scale = 1.0 for defaults
        M_quant = np.float64(p.input_scale) / np.float64(p.q_int8_scale)
        int8_out = np.clip(
            np.round(
                image_uint8.astype(np.float64) * M_quant
                + p.q_int8_zp
                - p.input_zp * M_quant
            ),
            -128, 127
        ).astype(np.int8)
        self._intermediates["after_quantize_int8"] = int8_out.copy()

        if self._verbose:
            print(f"[replica] Stage 1 QUANTIZE: min={int8_out.min()} max={int8_out.max()} "
                  f"mean={int8_out.astype(float).mean():.2f}", file=sys.stderr)

        # Stage 2: DEPTHWISE_CONV_2D (SAME padding, fused ReLU)
        # TFLite quantized conv accumulates:
        #   acc = bias + sum((input[i] - input_zp) * (weight[j] - weight_zp))
        # Since weight_zp = 0 for symmetric weights:
        #   acc = bias + sum((input[i] - input_zp) * weight[j])
        # The input_zp subtraction happens BEFORE multiplication.
        # Padding fills with input_zp, so (pad_val - input_zp) = 0.
        ksize = p.ksize
        pad = ksize // 2  # 3 for ksize=7
        input_zp_conv = int(p.q_int8_zp)  # -128

        # Subtract input_zp first, then pad with 0 (since pad represents input_zp)
        input_shifted = int8_out.astype(np.int32) - input_zp_conv
        padded = np.pad(
            input_shifted,
            ((pad, pad), (pad, pad)),
            mode='constant',
            constant_values=0  # (input_zp - input_zp) = 0
        )

        conv_int8 = np.zeros((H, W, self._n_filters), dtype=np.int8)
        conv_acc_debug = np.zeros((H, W, self._n_filters), dtype=np.int32)

        weights_int32 = self._weights_int8.astype(np.int32)  # [1, ksize, ksize, n_filters]

        for ch in range(self._n_filters):
            # Accumulate int32: sum of (input - input_zp) * weight
            acc = np.full((H, W), self._bias_int32[ch], dtype=np.int32)
            for ky in range(ksize):
                for kx in range(ksize):
                    w_val = weights_int32[0, ky, kx, ch]
                    acc += padded[ky:ky + H, kx:kx + W] * w_val

            conv_acc_debug[:, :, ch] = acc

            # Requantize: M_ch = q_int8_scale * weight_scale[ch] / conv_output_scale
            M_ch = self._conv_requant_M[ch]
            requantized = np.round(acc.astype(np.float64) * M_ch + p.conv_output_zp)
            clipped = np.clip(requantized, -128, 127).astype(np.int8)

            # Fused ReLU: max(val, conv_output_zp)
            # conv_output_zp = -128 represents 0.0 in the quantized domain
            clipped = np.maximum(clipped, np.int8(p.conv_output_zp))
            conv_int8[:, :, ch] = clipped

        self._intermediates["conv_acc_int32"] = conv_acc_debug.copy()
        self._intermediates["conv_output_int8"] = conv_int8.copy()

        if self._verbose:
            print(f"[replica] Stage 2 CONV: acc min={conv_acc_debug.min()} max={conv_acc_debug.max()} "
                  f"out min={conv_int8.min()} max={conv_int8.max()}", file=sys.stderr)

        # Stage 3: AVG_POOL_2D (pool_factor x pool_factor, stride pool_factor, VALID)
        pf = p.pool_factor
        reshaped = conv_int8.reshape(
            self._out_h, pf, self._out_w, pf, self._n_filters
        ).astype(np.int32)
        block_sums = reshaped.sum(axis=(1, 3))
        # TFLite AVG_POOL divides by count (pool_factor²)
        pool_count = pf * pf
        pool_int8 = np.clip(
            np.round(block_sums.astype(np.float64) / pool_count),
            -128, 127
        ).astype(np.int8)

        self._intermediates["pool_output_int8"] = pool_int8.copy()

        if self._verbose:
            print(f"[replica] Stage 3 POOL: min={pool_int8.min()} max={pool_int8.max()} "
                  f"mean={pool_int8.astype(float).mean():.2f}", file=sys.stderr)

        # Stage 4: QUANTIZE (int8 -> uint8)
        M_final = self._final_requant_M
        uint8_out = np.clip(
            np.round(
                (pool_int8.astype(np.float64) - p.pool_output_zp) * M_final
                + p.final_output_zp
            ),
            0, 255
        ).astype(np.uint8)

        self._intermediates["final_uint8"] = uint8_out.copy()

        if self._verbose:
            print(f"[replica] Stage 4 QUANTIZE: min={uint8_out.min()} max={uint8_out.max()} "
                  f"mean={uint8_out.mean():.2f}", file=sys.stderr)

        return uint8_out

    def compute(self, frame_t: np.ndarray, frame_t1: np.ndarray) -> Tuple[float, float]:
        """Compute optical flow between two uint8 grayscale frames.

        Args:
            frame_t: Reference frame (time t). Shape: (H, W), dtype: uint8.
            frame_t1: Current frame (time t+1). Shape: (H, W), dtype: uint8.

        Returns:
            (vx, vy) displacement in pooled pixels (sub-pixel via soft argmax).
        """
        feat_t = self.extract_features_uint8(frame_t)
        feat_t1 = self.extract_features_uint8(frame_t1)

        # Reuse the same post-processing as OpticalFlow._compute_from_uint8
        # by constructing a lightweight proxy object
        return self._compute_from_uint8(feat_t, feat_t1)

    def _compute_from_uint8(self, feat_t_u8: np.ndarray,
                            feat_t1_u8: np.ndarray) -> Tuple[float, float]:
        """Flow computation from uint8 feature maps.

        Mirrors OpticalFlow._compute_from_uint8() exactly:
        subtract zero point, cast to float, correlate, normalize, soft argmax.
        """
        p = self._params

        # The final output has zp = final_output_zp = 0, so subtracting zp is a no-op
        # for default params. But keep it general.
        zp = np.int16(p.final_output_zp)
        feat_t_int = feat_t_u8.astype(np.int16) - zp
        feat_t1_int = feat_t1_u8.astype(np.int16) - zp

        feat_t_f = feat_t_int.astype(np.float32)
        feat_t1_f = feat_t1_int.astype(np.float32)

        corr = self._global_correlation(feat_t_f, feat_t1_f).astype(np.float64)
        corr /= self._overlap_counts

        return self._soft_argmax(corr.astype(np.float32))

    def _global_correlation(self, feat_t: np.ndarray,
                            feat_t1: np.ndarray) -> np.ndarray:
        """Global correlation scores for all displacement candidates.

        Mirrors OpticalFlow._global_correlation() exactly.
        """
        from numpy.lib.stride_tricks import as_strided

        h, w, c = feat_t.shape
        sr = self._params.search_range
        side = 2 * sr + 1

        padded = np.pad(feat_t, ((sr, sr), (sr, sr), (0, 0)), mode='constant')
        s = padded.strides
        view = as_strided(padded,
                          shape=(side, side, h, w, c),
                          strides=(s[0], s[1], s[0], s[1], s[2]))

        if np.issubdtype(feat_t1.dtype, np.integer):
            corr_map = np.einsum('ijhwc,hwc->ij', view, feat_t1, dtype=np.int64)
        else:
            corr_map = np.einsum('ijhwc,hwc->ij', view, feat_t1)

        return corr_map[::-1, ::-1].ravel()

    def _soft_argmax(self, corr: np.ndarray) -> Tuple[float, float]:
        """Sub-pixel displacement via softmax-weighted sum.

        Mirrors OpticalFlow._soft_argmax() exactly.
        """
        corr_shifted = corr - np.max(corr)
        weights = np.exp(corr_shifted / max(self._params.temperature, 1e-6))
        total = np.sum(weights)
        if total < 1e-12:
            return 0.0, 0.0
        weights /= total

        vx = float(np.sum(weights * self._displacements[:, 0]))
        vy = float(np.sum(weights * self._displacements[:, 1]))
        return vx, vy

    def get_intermediates(self) -> Dict[str, np.ndarray]:
        """Return intermediates from the last extract_features_uint8() call.

        Keys:
            input_uint8: Original input image.
            after_quantize_int8: After Stage 1 (QUANTIZE).
            conv_acc_int32: Raw int32 accumulator from Stage 2 (before requantize).
            conv_output_int8: After Stage 2 (DEPTHWISE_CONV_2D + ReLU).
            pool_output_int8: After Stage 3 (AVG_POOL_2D).
            final_uint8: After Stage 4 (QUANTIZE to uint8).
        """
        return dict(self._intermediates)

    @property
    def params(self) -> ReplicaQuantParams:
        """Current quantization parameters."""
        return self._params

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def out_h(self) -> int:
        return self._out_h

    @property
    def out_w(self) -> int:
        return self._out_w

    @property
    def n_filters(self) -> int:
        return self._n_filters

    @property
    def weights_int8(self) -> np.ndarray:
        """Int8 Gabor weights [1, ksize, ksize, n_filters]."""
        return self._weights_int8.copy()

    @property
    def per_ch_weight_scales(self):
        """Per-channel weight scales."""
        return list(self._per_ch_weight_scales)
