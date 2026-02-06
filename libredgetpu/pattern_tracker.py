"""PatternTracker — Edge TPU-accelerated template matching via sliding correlation.

Locates a reference patch within a larger search image using Conv2D
cross-correlation + soft argmax peak detection. The user can swap the
reference template at runtime.

Usage:
    with PatternTracker.from_template(128, 16) as tracker:
        x_off, y_off = tracker.track(search_image)  # [-1,+1]
        tracker.set_template(new_patch)              # swap template
        x_off, y_off = tracker.track(search_image)   # tracks new patch
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np

__all__ = ["PatternTracker"]

from ._base import EdgeTPUModelBase
from ._constants import SIGN_BIT_FLIP
from ._quantize import quantize_int8
from .tflite_parser import parse as parse_tflite, parse_full
from .delegate import parse_darwinn, TYPE_PARAMETER_CACHING


class PatternTracker(EdgeTPUModelBase):
    """Edge TPU-accelerated template matching using Conv2D sliding correlation."""

    def __init__(self, tflite_path: str, metadata_path: Optional[str] = None,
                 firmware_path: Optional[str] = None):
        """Initialize PatternTracker from a compiled Edge TPU model.

        Args:
            tflite_path: Path to compiled *_edgetpu.tflite model.
            metadata_path: Path to JSON sidecar with quantization metadata.
                          If None, looks for {tflite_path}.json or infers from TFLite.
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.
        """
        super().__init__(tflite_path, metadata_path=metadata_path,
                         firmware_path=firmware_path)

        # Extract tracker-specific metadata
        self._search_height = self._metadata.get("search_height", self._input_info.shape[1])
        self._search_width = self._metadata.get("search_width", self._input_info.shape[2])
        self._channels = self._metadata.get("channels", self._input_info.shape[3])
        self._kernel_height = self._metadata.get("kernel_height", 16)
        self._kernel_width = self._metadata.get("kernel_width", 16)
        self._temperature = self._metadata.get("temperature", 0.1)
        self._y_offset = self._metadata.get("y_offset", 1.0 / self._temperature)

        # Conv2D weight quantization for runtime template swapping
        self._conv_weight_scale = self._metadata.get("conv_weight_scale", None)
        self._conv_weight_zero_point = self._metadata.get("conv_weight_zero_point", 0)
        self._conv_weight_count = self._metadata.get(
            "conv_weight_count",
            self._kernel_height * self._kernel_width * self._channels
        )

        # Conv2D blob offsets for compiler-free template swapping (fast path).
        # The offsets may refer to the PC or EO parameter blob depending on
        # the compiler's caching strategy (stored in conv_weight_blob field).
        raw_offsets = self._metadata.get("conv_weight_offsets", None)
        if raw_offsets is not None:
            self._conv_weight_offsets = np.array(raw_offsets, dtype=np.intp)
        else:
            self._conv_weight_offsets = None
        self._conv_weight_blob = self._metadata.get("conv_weight_blob", "pc_params")

        # Snapshot original params for repeated patching (fast path clones these)
        if self._cached_mode:
            self._original_pc_params = self._pc_params
            self._original_eo_params = self._eo_params

        # Locate uncompiled TFLite for recompilation fallback
        self._uncompiled_tflite_path = None
        base = os.path.splitext(tflite_path)[0]
        if base.endswith("_edgetpu"):
            candidate = base[:-len("_edgetpu")] + ".tflite"
            if os.path.isfile(candidate):
                self._uncompiled_tflite_path = candidate

    def _default_output_size(self) -> int:
        return 2

    @classmethod
    def from_template(cls, search_size: int, kernel_size: int,
                      channels: int = 1,
                      firmware_path: Optional[str] = None) -> "PatternTracker":
        """Create a PatternTracker from a pre-compiled template.

        Args:
            search_size: Square search image dimension (e.g., 128 for 128x128).
            kernel_size: Square template/kernel dimension (e.g., 16 for 16x16).
            channels: Input channels (1=grayscale, 3=RGB).
            firmware_path: Path to Edge TPU firmware. Auto-downloaded if None.

        Returns:
            PatternTracker instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the configuration.
        """
        from .pattern.templates import get_template
        tflite_path, json_path = get_template(search_size, kernel_size, channels)
        return cls(tflite_path, metadata_path=json_path, firmware_path=firmware_path)

    def track_raw(self, image_bytes: bytes) -> bytes:
        """Run tracking with raw uint8 input bytes. Returns raw output bytes.

        Args:
            image_bytes: Image as flat uint8 bytes (H*W*C bytes).

        Returns:
            Raw output bytes (2 bytes: x_off, y_off).
        """
        return self._execute_raw(image_bytes)

    def track(self, image: np.ndarray, resize: bool = True) -> Tuple[float, float]:
        """Track template in search image.

        Args:
            image: Input search image as numpy array.
                   For grayscale: (H, W) or (H, W, 1) or (1, H, W, 1)
                   For RGB: (H, W, 3) or (1, H, W, 3)
                   dtype: uint8 (preferred) or float32 [0, 255].
            resize: If True (default), resize image to model's expected dimensions.

        Returns:
            (x_offset, y_offset) tuple where:
            - x_offset: -1.0 (left) to +1.0 (right), 0.0 = center
            - y_offset: -1.0 (top) to +1.0 (bottom), 0.0 = center
        """
        image = self._normalize_tracker_input(
            image, resize, self._search_height, self._search_width, self._channels)
        quantized = self._quantize_input(image)
        raw_output = self.track_raw(quantized.tobytes())
        return self._decode_tracker_output(raw_output, self._y_offset, self._temperature)

    # ── Runtime template swapping ─────────────────────────────────────

    def set_template(self, patch: np.ndarray) -> None:
        """Change the Conv2D template at runtime.

        Quantizes the patch, patches the Conv2D weights in the parameter blob,
        and forces re-upload to the Edge TPU on the next track() call.

        Args:
            patch: Template image as numpy array.
                   Shape: (kernel_h, kernel_w) for grayscale,
                          (kernel_h, kernel_w, channels) for RGB.
                   dtype: uint8 [0, 255] or float32 [0, 255].

        Raises:
            ValueError: If patch shape/channels don't match, or values out of range.
            RuntimeError: If not a cached-mode model.
        """
        if not self._cached_mode:
            raise RuntimeError("set_template() requires a cached-mode template")

        patch = np.asarray(patch, dtype=np.float32)

        # Normalize shape
        if patch.ndim == 2 and self._channels == 1:
            patch = patch[:, :, np.newaxis]

        expected_shape = (self._kernel_height, self._kernel_width, self._channels)
        if patch.shape != expected_shape:
            raise ValueError(
                f"Patch shape {patch.shape} does not match expected {expected_shape}"
            )

        # Detect likely [0, 1] normalized images and warn
        if patch.max() <= 1.0 and patch.min() >= 0.0 and patch.max() > 0.0:
            import warnings
            warnings.warn(
                f"Patch values in [0, {patch.max():.2f}] look normalized. "
                f"Expected [0, 255] range. Rescaling to [0, 255].",
                UserWarning,
                stacklevel=2,
            )
            patch = patch * 255.0

        # Quantize patch to int8
        scale = self._conv_weight_scale
        if scale is None:
            raise RuntimeError(
                "conv_weight_scale not available in sidecar JSON. "
                "Regenerate template with pattern_tracker_gen.py."
            )

        zp = self._conv_weight_zero_point
        int8_weights = quantize_int8(patch.flatten(), scale, zp)

        self.set_template_raw(int8_weights)

    def set_template_raw(self, int8_weights: np.ndarray) -> None:
        """Set template from pre-quantized int8 weights.

        Uses a 2-tier strategy:
        1. **Fast path** (default): If ``conv_weight_offsets`` are in the
           sidecar JSON, patches the DarwiNN parameter blob directly using
           ``_patch_conv2d_blob()``.  Pure NumPy, ~microseconds, works on ARM.
        2. **Fallback**: If offsets are unavailable but the uncompiled
           ``.tflite`` exists, recompiles with ``edgetpu_compiler`` (~seconds,
           x86-only).
        3. **Error**: Neither available — raises RuntimeError with instructions.

        Args:
            int8_weights: Flat int8 array of length conv_weight_count.
        """
        int8_weights = np.asarray(int8_weights, dtype=np.int8).flatten()

        if len(int8_weights) != self._conv_weight_count:
            raise ValueError(
                f"Expected {self._conv_weight_count} weights, "
                f"got {len(int8_weights)}"
            )

        if self._conv_weight_offsets is not None:
            # Fast path: patch blob directly (no compiler needed)
            if self._conv_weight_blob == "eo_params":
                new_params = _patch_conv2d_blob(
                    self._original_eo_params, int8_weights,
                    self._conv_weight_offsets,
                )
            else:
                new_params = _patch_conv2d_blob(
                    self._original_pc_params, int8_weights,
                    self._conv_weight_offsets,
                )
        elif self._uncompiled_tflite_path is not None:
            # Fallback: recompile via edgetpu_compiler
            with open(self._uncompiled_tflite_path, "rb") as f:
                uncompiled_bytes = f.read()

            patched_bytes = _patch_conv2d_weights(
                uncompiled_bytes, int8_weights,
                self._kernel_height, self._kernel_width, self._channels
            )
            new_params = _recompile_and_extract_params(patched_bytes)
        else:
            raise RuntimeError(
                "Cannot set template: no conv_weight_offsets in sidecar JSON "
                "and no uncompiled TFLite model for recompilation. "
                "Regenerate templates with: "
                "python -m libredgetpu.pattern_tracker_gen --standard"
            )

        # Update the appropriate parameter blob
        if (self._conv_weight_offsets is not None
                and self._conv_weight_blob == "eo_params"):
            self._eo_params = new_params
        else:
            self._pc_params = new_params

        # Force re-upload on next track() call
        if self._driver is not None:
            self._driver._cached_token = 0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def search_height(self) -> int:
        """Search image height."""
        return self._search_height

    @property
    def search_width(self) -> int:
        """Search image width."""
        return self._search_width

    @property
    def kernel_height(self) -> int:
        """Template kernel height."""
        return self._kernel_height

    @property
    def kernel_width(self) -> int:
        """Template kernel width."""
        return self._kernel_width

    @property
    def channels(self) -> int:
        """Input image channels (1=grayscale, 3=RGB)."""
        return self._channels

    @property
    def temperature(self) -> float:
        """Softmax temperature."""
        return self._temperature

    @property
    def conv_weight_scale(self) -> Optional[float]:
        """Conv2D weight quantization scale."""
        return self._conv_weight_scale

    @property
    def conv_weight_range(self) -> Optional[Tuple[float, float]]:
        """Representable float32 range for Conv2D weights as (min, max)."""
        if self._conv_weight_scale is None:
            return None
        zp = self._conv_weight_zero_point
        return (
            (-128 - zp) * self._conv_weight_scale,
            (127 - zp) * self._conv_weight_scale,
        )


def _patch_conv2d_blob(base_params: bytes, int8_weights: np.ndarray,
                       offsets: np.ndarray) -> bytes:
    """Patch Conv2D weights into a DarwiNN parameter blob using pre-computed offsets.

    This is the fast path for compiler-free template swapping.  Each weight's
    blob position was discovered at template-generation time and stored in the
    sidecar JSON.

    Args:
        base_params: Original (template) parameter blob bytes.
        int8_weights: Flat int8 weight array (length N).
        offsets: Array of N blob byte offsets (one per weight).

    Returns:
        New parameter blob bytes with patched weights.
    """
    blob = np.frombuffer(base_params, dtype=np.uint8).copy()
    blob[offsets] = int8_weights.view(np.uint8) ^ SIGN_BIT_FLIP
    return bytes(blob)


def _patch_conv2d_weights(tflite_bytes: bytes, new_int8_weights: np.ndarray,
                           kernel_h: int, kernel_w: int, channels: int) -> bytes:
    """Patch Conv2D weight buffer in an uncompiled TFLite model.

    Uses structural FlatBuffer offsets (not content search) to locate the
    weight buffer precisely.
    """
    full = parse_full(tflite_bytes)
    expected_size = kernel_h * kernel_w * channels
    buf = bytearray(tflite_bytes)

    for op in full.operators:
        if op.opcode_name == "CONV_2D":
            weight_idx = op.inputs[1]
            weight_tensor = full.tensors[weight_idx]
            buffer_data = full.buffers[weight_tensor.buffer_index]
            if buffer_data is not None and len(buffer_data) == expected_size:
                offset = full.buffer_offsets[weight_tensor.buffer_index]
                if offset >= 0:
                    weight_bytes = new_int8_weights.astype(np.int8).tobytes()
                    buf[offset:offset + expected_size] = weight_bytes
                    return bytes(buf)

    raise ValueError(
        f"Could not find {expected_size}-byte Conv2D weight buffer in TFLite model"
    )


def _recompile_and_extract_params(tflite_bytes: bytes) -> bytes:
    """Compile a TFLite model with edgetpu_compiler and extract PC params."""
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
