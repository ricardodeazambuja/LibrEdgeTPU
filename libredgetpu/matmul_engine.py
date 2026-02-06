"""MatMulEngine — runtime weight-swapping matrix multiplication on Edge TPU.

Pre-compiled Dense(N×N) template models serve as the execution skeleton.
Users provide float32 weight matrices; the engine quantizes, generates
DarwiNN parameter blobs directly (no compiler needed), and uploads to
the Edge TPU.

The parameter blob format was empirically determined (Experiment 3):
  - 64-row groups, each with [overhead: 64*8 bytes][weights: 64*N bytes]
  - Weight values: int8 XOR 0x80 (sign-bit flip)
  - Weight layout: 4-column tiles within each group

When param_overhead is available in the sidecar JSON, set_weights() uses
this fast path (pure NumPy, ~microseconds, works on ARM).
Falls back to edgetpu_compiler recompilation if overhead is unavailable.

Usage::

    from libredgetpu import MatMulEngine

    with MatMulEngine.from_template(256) as engine:
        engine.set_weights(my_weights)      # float32 [256, 256]
        result = engine.matmul(x)            # float32 [256] → float32 [256]

    # Or with a custom compiled model + sidecar JSON:
    with MatMulEngine("my_model_edgetpu.tflite", "my_model_edgetpu.json") as engine:
        engine.set_weights(W)
        y = engine.matmul(x)
"""

import base64
import json
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np

__all__ = ["MatMulEngine"]

from .tflite_parser import parse as parse_tflite
from .delegate import parse_darwinn, TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY, TYPE_STAND_ALONE
from .transport import USBTransport, TAG_INSTRUCTIONS, TAG_PARAMETERS
from .driver import EdgeTPUDriver
from ._constants import SIGN_BIT_FLIP, QUANT_EPSILON, MAC_ARRAY_ROWS, WIDE_BUS_WIDTH
from ._quantize import quantize_uint8, quantize_int8, dequantize

# DarwiNN parameter blob layout aliases (Experiment 3)
_ROW_TILE = MAC_ARRAY_ROWS   # rows per group (matches 64x64 MAC array)
_COL_TILE = WIDE_BUS_WIDTH   # columns per tile block (wide memory bus width)


def _generate_param_blob(int8_weights, n, overhead_bytes):
    """Generate a DarwiNN parameter blob from int8 weights + template overhead.

    This eliminates the need for edgetpu_compiler at runtime.
    Format: 64-row groups, each = [overhead: 64*8][weights: 64*N in 4-col tiles].
    Values: int8 XOR 0x80 (sign-bit flip to uint8).

    Args:
        int8_weights: numpy array shape (n, n), dtype int8.
        n: square matrix dimension.
        overhead_bytes: bytes, the per-group overhead extracted from the template.
            Length must be ceil(n/64) * 64 * 8.

    Returns:
        bytes: DarwiNN parameter blob ready for USB upload.
    """
    num_groups = (n + _ROW_TILE - 1) // _ROW_TILE
    group_overhead = _ROW_TILE * 8
    group_weight_bytes = _ROW_TILE * n
    group_size = group_overhead + group_weight_bytes
    total_size = num_groups * group_size

    blob = np.zeros(total_size, dtype=np.uint8)

    # Copy overhead bytes into group headers
    overhead = np.frombuffer(overhead_bytes, dtype=np.uint8)
    for g in range(num_groups):
        src = g * group_overhead
        dst = g * group_size
        blob[dst:dst + group_overhead] = overhead[src:src + group_overhead]

    # Build vectorized index mapping: (row, col) -> blob offset
    rows = np.arange(n, dtype=np.int64).repeat(n)
    cols = np.tile(np.arange(n, dtype=np.int64), n)
    rg = rows // _ROW_TILE
    rl = rows % _ROW_TILE
    offsets = (rg * group_size + group_overhead
               + (cols // _COL_TILE) * (_ROW_TILE * _COL_TILE)
               + rl * _COL_TILE + (cols % _COL_TILE))

    # XOR 0x80 and place into blob
    values = int8_weights.flatten().view(np.uint8) ^ SIGN_BIT_FLIP
    blob[offsets] = values

    return bytes(blob)


def _extract_overhead(param_blob, n):
    """Extract the overhead bytes from a DarwiNN parameter blob.

    Returns the concatenated overhead headers from all 64-row groups.
    """
    num_groups = (n + _ROW_TILE - 1) // _ROW_TILE
    group_overhead = _ROW_TILE * 8
    group_size = group_overhead + _ROW_TILE * n

    overhead = bytearray()
    for g in range(num_groups):
        start = g * group_size
        overhead.extend(param_blob[start:start + group_overhead])
    return bytes(overhead)


def _patch_tflite_weights(tflite_bytes, new_int8_weights):
    """Patch the weight buffer in an uncompiled TFLite model with new int8 values.

    Uses structural FlatBuffer offsets (not content search) to locate the
    weight buffer precisely.  Returns the patched TFLite bytes.
    """
    buf = bytearray(tflite_bytes)
    n = new_int8_weights.shape[0]
    expected_size = n * n
    weight_bytes = new_int8_weights.astype(np.int8).tobytes()

    from .tflite_parser import parse_full
    full = parse_full(bytes(buf))

    for i, buffer_data in enumerate(full.buffers):
        if buffer_data is not None and len(buffer_data) == expected_size:
            offset = full.buffer_offsets[i]
            if offset >= 0:
                buf[offset:offset + expected_size] = weight_bytes
                return bytes(buf)

    raise ValueError(f"Could not find {expected_size}-byte weight buffer in TFLite model")


def _recompile_and_extract_params(tflite_bytes):
    """Compile a TFLite model with edgetpu_compiler and extract DarwiNN params.

    Returns the PARAMETER_CACHING executable's parameter blob.
    """
    if shutil.which("edgetpu_compiler") is None:
        raise FileNotFoundError(
            "edgetpu_compiler not found on PATH. "
            "Install from https://coral.ai/docs/edgetpu/compiler/"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tflite")
        with open(model_path, "wb") as f:
            f.write(tflite_bytes)

        try:
            result = subprocess.run(
                ["edgetpu_compiler", "-s", "-o", tmpdir, model_path],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "edgetpu_compiler timed out after 120 seconds"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"edgetpu_compiler failed: {result.stderr[:500]}"
            )

        compiled_path = os.path.join(tmpdir, "model_edgetpu.tflite")
        with open(compiled_path, "rb") as f:
            compiled_bytes = f.read()

        model = parse_tflite(compiled_bytes)
        exes = parse_darwinn(model.custom_op_data)
        for exe in exes:
            if exe.exec_type == TYPE_PARAMETER_CACHING and exe.parameters:
                return exe.parameters

    raise RuntimeError("Recompiled model has no PARAMETER_CACHING parameters")


class MatMulEngine:
    """Runtime weight-swapping matrix multiplication engine for Edge TPU.

    Wraps a pre-compiled Dense(N×N) template model and allows swapping
    the weight matrix at runtime.  The primary (fast) path generates
    DarwiNN parameter blobs directly from the empirically determined format
    — pure NumPy, no compiler, works on ARM.  A fallback path using
    ``edgetpu_compiler`` is available for templates that lack the
    ``param_overhead`` sidecar field.
    """

    def __init__(self, tflite_path: str, metadata_path: Optional[str] = None,
                 firmware_path: Optional[str] = None) -> None:
        """Initialize from a compiled Edge TPU model.

        Args:
            tflite_path: Path to *_edgetpu.tflite compiled model.
            metadata_path: Path to sidecar JSON with quantization metadata.
                If None, looks for a .json file alongside the .tflite.
            firmware_path: Optional path to Edge TPU firmware binary.
        """
        self._tflite_path = tflite_path
        self._firmware_path = firmware_path

        # Parse TFLite model
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()
        model = parse_tflite(tflite_bytes)
        self._input_info = model.input_tensor
        self._output_info = model.output_tensor

        # Parse DarwiNN executables
        self._executables = parse_darwinn(model.custom_op_data)
        self._pc_exe = None
        self._eo_exe = None
        self._sa_exe = None

        for exe in self._executables:
            if exe.exec_type == TYPE_PARAMETER_CACHING:
                self._pc_exe = exe
            elif exe.exec_type == TYPE_EXECUTION_ONLY:
                self._eo_exe = exe
            elif exe.exec_type == TYPE_STAND_ALONE:
                self._sa_exe = exe

        if self._pc_exe is None or self._eo_exe is None:
            raise ValueError(
                "MatMulEngine requires a cached-mode model "
                "(PARAMETER_CACHING + EXECUTION_ONLY executables). "
                "Standalone/streamed models are not supported."
            )

        # Store original parameters for reset
        self._original_params = self._pc_exe.parameters

        # Output size from EO executable
        layers = self._eo_exe.output_layers
        self._output_size = sum(l.size_bytes for l in layers) if layers else 256

        # Load sidecar metadata
        if metadata_path is None:
            base = os.path.splitext(tflite_path)[0]
            candidate = base + ".json"
            if os.path.isfile(candidate):
                metadata_path = candidate

        self._metadata = {}
        self._weight_scale = None
        self._weight_zero_point = None
        self._param_overhead = None  # overhead bytes for compiler-free blob gen

        if metadata_path is not None:
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)
            self._weight_scale = self._metadata.get("weight_scale")
            self._weight_zero_point = self._metadata.get("weight_zero_point")

            # Load param_overhead if available (enables compiler-free set_weights)
            overhead_b64 = self._metadata.get("param_overhead")
            if overhead_b64 is not None:
                self._param_overhead = base64.b64decode(overhead_b64)

        # Derive matrix size from input shape
        input_shape = self._input_info.shape
        self._matrix_size = input_shape[-1] if input_shape else None
        if self._metadata.get("matrix_size"):
            self._matrix_size = self._metadata["matrix_size"]

        # Expected param blob size
        self._param_size = self._metadata.get("param_size")
        if self._param_size is None and self._original_params is not None:
            self._param_size = len(self._original_params)

        # Locate uncompiled TFLite for recompilation
        self._uncompiled_tflite_path = None
        base = os.path.splitext(tflite_path)[0]
        # Template pattern: dense_256_edgetpu.tflite → dense_256.tflite
        if base.endswith("_edgetpu"):
            candidate = base[:-len("_edgetpu")] + ".tflite"
            if os.path.isfile(candidate):
                self._uncompiled_tflite_path = candidate

        # Transport + driver
        self._transport = USBTransport(firmware_path=firmware_path)
        self._driver = None
        self._hw_initialized = False
        self._weights_dirty = True

    @classmethod
    def from_template(cls, n: int, firmware_path: Optional[str] = None) -> "MatMulEngine":
        """Create a MatMulEngine from a pre-compiled template.

        Args:
            n: Square matrix dimension (e.g. 256, 512, 1024).
            firmware_path: Optional path to Edge TPU firmware.

        Returns:
            MatMulEngine instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the specified size.
        """
        from .templates import get_template
        tflite_path, json_path = get_template(n)
        return cls(tflite_path, metadata_path=json_path, firmware_path=firmware_path)

    def open(self) -> None:
        """Open USB connection and initialize Edge TPU hardware."""
        self._transport.open()
        self._driver = EdgeTPUDriver(self._transport)
        self._driver.init_hardware()
        self._hw_initialized = True
        self._weights_dirty = True

    def close(self) -> None:
        """Release hardware resources."""
        self._transport.close()
        self._driver = None
        self._hw_initialized = False

    def __enter__(self) -> "MatMulEngine":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Weight management ──────────────────────────────────────────────────

    def quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Quantize a float32 weight matrix to int8.

        Args:
            weights: numpy array of shape [N, N], float32.

        Returns:
            numpy array of shape [N, N], int8.

        Raises:
            ValueError: If weight_scale is not available or shape is wrong.
        """
        weights = np.asarray(weights, dtype=np.float32)
        n = self._matrix_size
        if weights.shape != (n, n):
            raise ValueError(
                f"Weight shape {weights.shape} does not match "
                f"template matrix size ({n}, {n})"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Weight matrix contains NaN or Inf values")
        if self._weight_scale is None:
            raise ValueError(
                "weight_scale not available. Provide a sidecar JSON."
            )

        scale = self._weight_scale
        zp = self._weight_zero_point if self._weight_zero_point is not None else 0

        return quantize_int8(weights, scale, zp)

    def set_weights(self, weights: np.ndarray) -> None:
        """Set new weights from a float32 matrix.

        Quantizes weights to int8 and generates the DarwiNN parameter blob.

        Fast path (default): When param_overhead is in the sidecar JSON,
        generates the blob directly using the empirically determined format.
        Pure NumPy, works on any platform including ARM.

        Fallback path: When param_overhead is unavailable, patches the
        uncompiled TFLite and recompiles with edgetpu_compiler (x86-only).

        Args:
            weights: numpy array of shape [N, N], float32.
        """
        quantized = self.quantize_weights(weights)
        n = self._matrix_size

        if self._param_overhead is not None:
            # Fast path: generate blob directly (no compiler needed)
            new_params = _generate_param_blob(quantized, n, self._param_overhead)
        elif self._uncompiled_tflite_path is not None:
            # Fallback: recompile via edgetpu_compiler
            with open(self._uncompiled_tflite_path, "rb") as f:
                uncompiled_bytes = f.read()
            patched_bytes = _patch_tflite_weights(uncompiled_bytes, quantized)
            new_params = _recompile_and_extract_params(patched_bytes)
        else:
            raise RuntimeError(
                "Cannot set weights: no param_overhead in sidecar JSON "
                "and no uncompiled TFLite model found for recompilation. "
                "Regenerate templates with template_gen.py to add param_overhead."
            )

        # Upload to hardware
        self.set_weights_raw(new_params)

    def set_weights_raw(self, param_bytes: bytes) -> None:
        """Set new weights from a raw DarwiNN parameter blob.

        This is the low-level path — the param_bytes must be in the
        compiler's internal format (not flat int8). Use set_weights()
        for the high-level float32 API.

        Args:
            param_bytes: DarwiNN parameter blob bytes.
        """
        if not self._hw_initialized:
            raise RuntimeError("Call open() or use as context manager first")

        if isinstance(param_bytes, (bytearray, memoryview)):
            param_bytes = bytes(param_bytes)

        # Invalidate cache and upload new parameters
        self._driver.reset_cached_parameters()

        pc = self._pc_exe
        if pc.dma_steps:
            self._driver.execute_dma_hints(
                pc.dma_steps,
                [bs.data for bs in pc.bitstreams],
                b"", params=param_bytes,
            )
        else:
            pc_instr = pc.bitstreams[0].data
            self._driver.send_raw(pc_instr, TAG_INSTRUCTIONS)
            self._driver.send_raw(param_bytes, TAG_PARAMETERS)
            self._driver.read_status_packet()

        self._driver._cached_token = -1  # sentinel: "custom weights loaded"
        self._current_params = param_bytes
        self._weights_dirty = False

    def reset_weights(self) -> None:
        """Restore the template's original compiled weights."""
        if self._original_params is None:
            raise RuntimeError("Template has no original parameters to restore")
        self.set_weights_raw(self._original_params)

    # ── Inference ──────────────────────────────────────────────────────────

    def matmul_raw(self, input_bytes: bytes) -> bytes:
        """Run matrix multiplication with raw uint8 input bytes.

        Args:
            input_bytes: Raw uint8 input vector (N bytes).

        Returns:
            bytes: Raw uint8 output vector.
        """
        if not self._hw_initialized:
            raise RuntimeError("Call open() or use as context manager first")

        if isinstance(input_bytes, (bytearray, memoryview)):
            input_bytes = bytes(input_bytes)

        if self._weights_dirty:
            raise RuntimeError(
                "No weights loaded. Call set_weights() or set_weights_raw() first."
            )

        eo = self._eo_exe
        if eo.dma_steps:
            return self._driver.execute_dma_hints(
                eo.dma_steps,
                [bs.data for bs in eo.bitstreams],
                input_bytes, params=eo.parameters or b"",
                output_size=self._output_size,
            )
        else:
            eo_instr = eo.bitstreams[0].data
            return self._driver.execute_inference_only(
                eo_instr, input_bytes, self._output_size
            )

    def matmul(self, x: np.ndarray) -> np.ndarray:
        """Run matrix multiplication: y = W @ x.

        Args:
            x: numpy array, float32. Shape [N] or [1, N].

        Returns:
            numpy array, float32. Shape matching input dimensionality.
        """
        x = np.asarray(x, dtype=np.float32)
        orig_shape = x.shape

        # Quantize input: float → uint8
        in_info = self._input_info
        quantized = quantize_uint8(x, in_info.scale, in_info.zero_point)

        raw_output = self.matmul_raw(quantized.tobytes())

        # Dequantize output: uint8 → float
        out_info = self._output_info
        out_uint8 = np.frombuffer(raw_output, dtype=np.uint8)
        result = dequantize(out_uint8, out_info.scale, out_info.zero_point)

        if len(orig_shape) == 1:
            return result[:self._matrix_size]
        return result[:self._matrix_size].reshape(1, -1)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def matrix_size(self) -> Optional[int]:
        """Square matrix dimension N."""
        return self._matrix_size

    @property
    def weight_scale(self) -> Optional[float]:
        """Quantization scale for weights (float32 = (int8 - zp) * scale)."""
        return self._weight_scale

    @property
    def weight_zero_point(self) -> Optional[int]:
        """Quantization zero point for weights."""
        return self._weight_zero_point

    @property
    def weight_range(self) -> Optional[Tuple[float, float]]:
        """Representable float32 weight range as (min, max) tuple."""
        if self._weight_scale is None:
            return None
        zp = self._weight_zero_point if self._weight_zero_point is not None else 0
        return (
            (-128 - zp) * self._weight_scale,
            (127 - zp) * self._weight_scale,
        )

    @property
    def input_scale(self) -> float:
        """Input tensor quantization scale."""
        return self._input_info.scale

    @property
    def input_zero_point(self) -> int:
        """Input tensor quantization zero point."""
        return self._input_info.zero_point

    @property
    def output_scale(self) -> float:
        """Output tensor quantization scale."""
        return self._output_info.scale

    @property
    def output_zero_point(self) -> int:
        """Output tensor quantization zero point."""
        return self._output_info.zero_point

    @property
    def param_size(self) -> Optional[int]:
        """Expected parameter blob size in bytes."""
        return self._param_size
