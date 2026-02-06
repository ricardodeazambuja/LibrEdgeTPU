"""Shared quantization/dequantization utilities.

Centralizes the quantization formula used across all modules to prevent
inconsistencies (e.g., missing epsilon guards).
"""

import numpy as np

from ._constants import QUANT_EPSILON


def _safe_scale(scale):
    """Return a safe positive scale value, raising on negative scales."""
    if scale < 0:
        raise ValueError(f"Invalid quantization scale: {scale} (must be >= 0)")
    return max(scale, QUANT_EPSILON)


def quantize_uint8(array, scale, zero_point):
    """Quantize float32 array to uint8 using TFLite affine quantization.

    q = clamp(round(value / scale + zero_point), 0, 255)

    Args:
        array: Input data (will be cast to float32).
        scale: Quantization scale (float). Must be >= 0; zero is replaced by epsilon.
        zero_point: Quantization zero point (int).

    Returns:
        numpy uint8 array.

    Raises:
        ValueError: If scale is negative.
    """
    return np.clip(
        np.round(np.asarray(array, dtype=np.float32) / _safe_scale(scale) + zero_point),
        0, 255,
    ).astype(np.uint8)


def quantize_int8(array, scale, zero_point=0):
    """Quantize float32 array to int8 (weight quantization).

    q = clamp(round(value / scale + zero_point), -128, 127)

    Args:
        array: Input data (will be cast to float32).
        scale: Quantization scale (float). Must be >= 0; zero is replaced by epsilon.
        zero_point: Quantization zero point (int, usually 0 for weights).

    Returns:
        numpy int8 array.

    Raises:
        ValueError: If scale is negative.
    """
    return np.clip(
        np.round(np.asarray(array, dtype=np.float32) / _safe_scale(scale) + zero_point),
        -128, 127,
    ).astype(np.int8)


def dequantize(array, scale, zero_point):
    """Dequantize uint8 or int8 array to float32.

    value = (q - zero_point) * scale

    Args:
        array: Quantized data (uint8 or int8).
        scale: Quantization scale (float).
        zero_point: Quantization zero point (int).

    Returns:
        numpy float32 array.
    """
    return (np.asarray(array).astype(np.float32) - zero_point) * scale
