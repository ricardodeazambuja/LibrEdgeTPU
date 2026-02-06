"""Base class for Edge TPU model wrappers.

Consolidates the shared TFLite parsing, DarwiNN extraction, executable
classification, USB transport management, and cached/standalone execution
protocol used by SimpleInvoker, SpotTracker, and LoomingDetector.
"""

import json
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tflite_parser import parse as parse_tflite, TFLiteModel
from .delegate import (
    parse_darwinn, DarwiNNExecutable,
    TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY, TYPE_STAND_ALONE,
)
from .transport import USBTransport, TAG_INSTRUCTIONS, TAG_PARAMETERS
from .driver import EdgeTPUDriver
from ._constants import SIGN_BIT_FLIP
from ._quantize import quantize_uint8


class EdgeTPUModelBase:
    """Base class for Edge TPU model wrappers.

    Handles TFLite parsing, DarwiNN extraction, executable classification,
    USB transport lifecycle, and the cached/standalone execution protocol.

    Subclasses should call ``super().__init__(...)`` and then set any
    additional attributes.  Override ``_default_output_size`` to control
    the fallback when DarwiNN output layers are unavailable.
    """

    def __init__(self, tflite_path: str, metadata_path: Optional[str] = None,
                 firmware_path: Optional[str] = None) -> None:
        self._tflite_path = tflite_path
        self._firmware_path = firmware_path

        # ── Load metadata from JSON sidecar ───────────────────────────────
        self._metadata: Dict = {}
        if metadata_path is None:
            candidate = tflite_path.replace("_edgetpu.tflite", "_edgetpu.json")
            if os.path.isfile(candidate):
                metadata_path = candidate
        if metadata_path and os.path.isfile(metadata_path):
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)

        # ── Parse TFLite model ────────────────────────────────────────────
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()
        model = parse_tflite(tflite_bytes)
        self._input_info = model.input_tensor
        self._output_info = model.output_tensor

        # ── Parse and classify DarwiNN executables ────────────────────────
        self._executables: List[DarwiNNExecutable] = parse_darwinn(model.custom_op_data)
        self._pc_exe: Optional[DarwiNNExecutable] = None
        self._eo_exe: Optional[DarwiNNExecutable] = None
        self._sa_exe: Optional[DarwiNNExecutable] = None

        for exe in self._executables:
            if exe.exec_type == TYPE_PARAMETER_CACHING:
                self._pc_exe = exe
            elif exe.exec_type == TYPE_EXECUTION_ONLY:
                self._eo_exe = exe
            elif exe.exec_type == TYPE_STAND_ALONE:
                self._sa_exe = exe

        self._cached_mode = self._pc_exe is not None and self._eo_exe is not None

        # ── Determine output size ─────────────────────────────────────────
        if self._cached_mode:
            layers = self._eo_exe.output_layers
        elif self._sa_exe is not None:
            layers = self._sa_exe.output_layers
        else:
            layers = []
        self._output_size = (
            sum(l.size_bytes for l in layers) if layers
            else self._default_output_size()
        )

        # ── Pre-cache invariant bitstreams and parameters ─────────────────
        if self._cached_mode:
            pc = self._pc_exe
            eo = self._eo_exe
            self._pc_token = pc.parameter_caching_token
            self._pc_bitstreams = [bs.data for bs in pc.bitstreams]
            self._pc_params = pc.parameters or b""
            self._pc_has_dma = bool(pc.dma_steps)
            self._eo_bitstreams = [bs.data for bs in eo.bitstreams]
            self._eo_params = eo.parameters or b""
            self._eo_has_dma = bool(eo.dma_steps)
        elif self._sa_exe is not None:
            sa = self._sa_exe
            self._sa_bitstreams = [bs.data for bs in sa.bitstreams]
            self._sa_params = sa.parameters or b""
            self._sa_has_dma = bool(sa.dma_steps)

        # ── USB transport + driver (opened lazily) ────────────────────────
        self._transport = USBTransport(firmware_path=firmware_path)
        self._driver: Optional[EdgeTPUDriver] = None
        self._hw_initialized = False
        self._cache_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open USB connection and initialize hardware."""
        self._transport.open()
        self._driver = EdgeTPUDriver(self._transport)
        self._driver.init_hardware()
        self._hw_initialized = True

    def close(self) -> None:
        """Release hardware."""
        self._hw_initialized = False
        self._transport.close()
        self._driver = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Execution protocol ────────────────────────────────────────────────

    def _execute_raw(self, input_bytes: bytes) -> bytes:
        """Execute the model with raw input bytes.

        Handles the cached-mode parameter caching protocol and the
        standalone execution path.  This is the common implementation
        shared by invoke_raw / track_raw / detect_raw.
        """
        if not self._hw_initialized or self._driver is None:
            raise RuntimeError("Call open() or use as context manager first")

        if isinstance(input_bytes, (bytearray, memoryview)):
            input_bytes = bytes(input_bytes)

        if self._cached_mode:
            self._ensure_params_cached()

            if self._eo_has_dma:
                return self._driver.execute_dma_hints(
                    self._eo_exe.dma_steps,
                    self._eo_bitstreams,
                    input_bytes, params=self._eo_params,
                    output_size=self._output_size,
                )
            else:
                return self._driver.execute_inference_only(
                    self._eo_bitstreams[0], input_bytes, self._output_size
                )
        else:
            if self._sa_has_dma:
                return self._driver.execute_dma_hints(
                    self._sa_exe.dma_steps,
                    self._sa_bitstreams,
                    input_bytes, params=self._sa_params,
                    output_size=self._output_size,
                )
            else:
                return self._driver.execute_standalone(
                    self._sa_bitstreams[0], self._sa_params,
                    input_bytes, self._output_size
                )

    def _ensure_params_cached(self) -> None:
        """Upload parameters if not already cached on the device."""
        with self._cache_lock:
            token = self._pc_token
            if token == 0 or token != self._driver._cached_token:
                self._driver.reset_cached_parameters()
                if self._pc_has_dma:
                    self._driver.execute_dma_hints(
                        self._pc_exe.dma_steps,
                        self._pc_bitstreams,
                        b"", params=self._pc_params,
                    )
                else:
                    self._driver.send_raw(self._pc_bitstreams[0], TAG_INSTRUCTIONS)
                    self._driver.send_raw(self._pc_params, TAG_PARAMETERS)
                    self._driver.read_status_packet()
                self._driver._cached_token = token

    # ── Overridable hooks ─────────────────────────────────────────────────

    def _default_output_size(self) -> int:
        """Fallback output size when DarwiNN output layers are unavailable.

        Override in subclasses to provide a sensible default.
        """
        return 256

    # ── Common properties ─────────────────────────────────────────────────

    @property
    def input_shape(self):
        """Input tensor shape from TFLite model."""
        return self._input_info.shape

    @property
    def output_shape(self):
        """Output tensor shape from TFLite model."""
        return self._output_info.shape

    @property
    def tflite_path(self) -> str:
        """Path to the loaded TFLite model."""
        return self._tflite_path

    # ── Shared tracker helpers ─────────────────────────────────────────────
    # Used by SpotTracker and PatternTracker to avoid code duplication.
    # Dimension-specific values are passed as arguments.

    def _quantize_input(self, image: np.ndarray) -> np.ndarray:
        """Quantize image to uint8 using model's input quantization parameters."""
        in_info = self._input_info
        return quantize_uint8(image, in_info.scale, in_info.zero_point)

    def _normalize_tracker_input(self, image: np.ndarray, resize: bool,
                                  h_out: int, w_out: int, channels: int) -> np.ndarray:
        """Normalize tracker input shape, dtype, and size.

        Args:
            image: Raw input image.
            resize: Whether to resize if dimensions don't match.
            h_out: Expected output height.
            w_out: Expected output width.
            channels: Expected number of channels.

        Returns:
            Normalized image array of shape (h_out, w_out, channels).
        """
        image = np.asarray(image)

        # Handle batch dimension
        if image.ndim == 4:
            image = image.squeeze(axis=0)
        if image.ndim == 2 and channels == 1:
            image = image[:, :, np.newaxis]
        elif image.ndim == 3 and image.shape[-1] == 1 and channels == 1:
            pass  # already (H, W, 1)

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Resize if requested and needed
        if resize and image.shape[:2] != (h_out, w_out):
            image = self._resize_tracker_image(image, h_out, w_out, channels)

        # Validate shape
        expected_shape = (h_out, w_out, channels)
        if image.shape != expected_shape:
            if channels == 1 and image.shape == (h_out, w_out):
                image = image[:, :, np.newaxis]
            elif not resize:
                raise ValueError(
                    f"Image shape {image.shape} does not match expected {expected_shape}. "
                    f"Consider setting resize=True."
                )
            else:
                raise ValueError(
                    f"Image shape {image.shape} does not match expected {expected_shape}"
                )

        return image

    def _resize_tracker_image(self, image: np.ndarray,
                               h_out: int, w_out: int, channels: int) -> np.ndarray:
        """Resize image to target dimensions using block mean or area averaging.

        Args:
            image: Input image.
            h_out: Target height.
            w_out: Target width.
            channels: Number of channels.

        Returns:
            Resized uint8 image of shape (h_out, w_out, channels).
        """
        h_in, w_in = image.shape[:2]

        if h_in % h_out == 0 and w_in % w_out == 0:
            # Fast path: exact integer ratio block mean
            bh, bw = h_in // h_out, w_in // w_out
            if channels == 1 or (image.ndim == 3 and image.shape[-1] == 1):
                img_2d = image.squeeze() if image.ndim == 3 else image
                resized = img_2d.reshape(h_out, bh, w_out, bw).mean(axis=(1, 3)).astype(np.uint8)
                return resized[:, :, np.newaxis]
            else:
                resized = image.reshape(h_out, bh, w_out, bw, channels).mean(axis=(1, 3)).astype(np.uint8)
                return resized
        else:
            # Slow path: area averaging for non-integer ratios
            h_step = h_in / h_out
            w_step = w_in / w_out
            resized = np.zeros((h_out, w_out, channels), dtype=np.float32)
            for i in range(h_out):
                for j in range(w_out):
                    y_start = int(i * h_step)
                    y_end = int((i + 1) * h_step) if i < h_out - 1 else h_in
                    x_start = int(j * w_step)
                    x_end = int((j + 1) * w_step) if j < w_out - 1 else w_in
                    if image.ndim == 2:
                        resized[i, j, 0] = image[y_start:y_end, x_start:x_end].mean()
                    else:
                        resized[i, j] = image[y_start:y_end, x_start:x_end].mean(axis=(0, 1))
            return resized.astype(np.uint8)

    def _decode_tracker_output(self, raw_output: bytes,
                                y_offset: float, temperature: float) -> Tuple[float, float]:
        """Dequantize raw tracker output bytes to (x_offset, y_offset).

        Args:
            raw_output: Raw output bytes from the Edge TPU (2 bytes).
            y_offset: Y offset baked into the model.
            temperature: Softmax temperature used for scaling.

        Returns:
            (x_offset, y_offset) in [-1, +1] range.
        """
        out_info = self._output_info
        if len(raw_output) < 2:
            raise ValueError(
                f"Tracker output too short: expected >= 2 bytes, got {len(raw_output)}"
            )
        out_uint8 = np.frombuffer(raw_output, dtype=np.uint8)[:2]

        if out_info.dtype == 9:
            # INT8 output: XOR sign bit to convert uint8 -> int8
            out_vals = (out_uint8 ^ SIGN_BIT_FLIP).view(np.int8).astype(np.float32)
        else:
            out_vals = out_uint8.astype(np.float32)

        offsets = (out_vals - out_info.zero_point) * out_info.scale

        x_raw = float(offsets[0])
        y_raw = float(offsets[1])

        # Remove the Y offset baked into the model
        y_raw = y_raw - y_offset

        # Scale from model range back to [-1, +1]
        x_off = x_raw * temperature
        y_off = y_raw * temperature

        return x_off, y_off
