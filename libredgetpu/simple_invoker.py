"""SimpleInvoker — user-facing API for Edge TPU inference without libedgetpu.

Usage:
    with SimpleInvoker("model_edgetpu.tflite") as model:
        output = model.invoke(input_array)       # float32 in -> float32 out
        raw = model.invoke_raw(input_bytes)       # uint8 in -> uint8 out
"""

import numpy as np

__all__ = ["SimpleInvoker"]

from ._base import EdgeTPUModelBase
from ._constants import SIGN_BIT_FLIP
from ._quantize import quantize_uint8, dequantize
from .delegate import TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY, TYPE_STAND_ALONE


class SimpleInvoker(EdgeTPUModelBase):
    """Run fully-mapped Edge TPU models via direct USB — no libedgetpu required."""

    def __init__(self, tflite_path, firmware_path=None):
        super().__init__(tflite_path, firmware_path=firmware_path)

    def invoke_raw(self, input_bytes):
        """Run inference with raw uint8 input bytes. Returns raw uint8 output bytes.

        This is the zero-overhead path — no quantization/dequantization.
        """
        return self._execute_raw(input_bytes)

    def invoke_raw_outputs(self, input_bytes):
        """Run inference and return a list of per-output-layer raw byte arrays.

        Unlike invoke_raw() which returns a single concatenated blob, this
        splits the output using DarwiNN output_layer sizes so post-processing
        modules can work with individual tensors.
        """
        raw = self.invoke_raw(input_bytes)

        if self._cached_mode:
            layers = self._eo_exe.output_layers
        elif self._sa_exe is not None:
            layers = self._sa_exe.output_layers
        else:
            return [raw]

        if not layers or len(layers) <= 1:
            return [raw]

        expected_total = sum(layer.size_bytes for layer in layers)
        if len(raw) < expected_total:
            import warnings
            warnings.warn(
                f"Output size mismatch: expected {expected_total} bytes "
                f"({len(layers)} layers), got {len(raw)} bytes",
                RuntimeWarning,
                stacklevel=2,
            )

        parts = []
        offset = 0
        for layer in layers:
            parts.append(raw[offset:offset + layer.size_bytes])
            offset += layer.size_bytes
        return parts

    def invoke(self, input_array):
        """Run inference with float32 input. Returns float32 output.

        Handles quantization (float->uint8) and dequantization (uint8->float)
        using the TFLite model's quantization parameters.

        Note: Edge TPU hardware always outputs uint8 bytes. For models with
        int8 output type (TFLite dtype 9), the raw bytes are XOR'd with 0x80
        to convert from the TPU's unsigned representation to signed int8.
        """
        input_array = np.asarray(input_array, dtype=np.float32)

        # Quantize: float -> uint8
        in_info = self._input_info
        quantized = quantize_uint8(input_array, in_info.scale, in_info.zero_point)

        raw_output = self.invoke_raw(quantized.tobytes())

        # Dequantize output
        out_info = self._output_info
        out_uint8 = np.frombuffer(raw_output, dtype=np.uint8)

        if out_info.dtype == 9:
            # INT8 output: Edge TPU outputs uint8, need XOR 0x80 to get int8
            out_int8 = (out_uint8 ^ SIGN_BIT_FLIP).view(np.int8)
            return dequantize(out_int8, out_info.scale, out_info.zero_point)
        else:
            # UINT8 output (most models): use directly
            return dequantize(out_uint8, out_info.scale, out_info.zero_point)

    @property
    def input_scale(self):
        return self._input_info.scale

    @property
    def input_zero_point(self):
        return self._input_info.zero_point

    @property
    def output_scale(self):
        return self._output_info.scale

    @property
    def output_zero_point(self):
        return self._output_info.zero_point

    @property
    def output_layers(self):
        """Return list of LayerInfo for each output layer (from DarwiNN executable)."""
        if self._cached_mode:
            return self._eo_exe.output_layers
        elif self._sa_exe is not None:
            return self._sa_exe.output_layers
        return []
