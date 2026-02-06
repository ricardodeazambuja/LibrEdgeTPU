"""Pre-compiled Dense(N×N) Edge TPU templates for MatMulEngine.

Templates are generated at dev time using template_gen.py and shipped
with the package. Each template consists of:
  - dense_{n}_edgetpu.tflite  (compiled Edge TPU model — instructions + params)
  - dense_{n}.tflite           (uncompiled quantized TFLite — for runtime recompilation)
  - dense_{n}_edgetpu.json    (quantization metadata sidecar — scales, zero points, param size)

The uncompiled TFLite is used by MatMulEngine.set_weights() to patch in new
int8 weight values while preserving the quantization metadata, then recompile
with edgetpu_compiler. This produces identical instructions with a new parameter
blob (experimentally verified).
"""

import os
from typing import Optional, Tuple

__all__ = ["get_template", "list_templates"]

_TEMPLATES_DIR = os.path.dirname(__file__)


def get_template(n: int) -> Tuple[str, str]:
    """Locate a pre-compiled template for matrix size N.

    Args:
        n: Square matrix dimension.

    Returns:
        (tflite_path, json_path) tuple with absolute paths.

    Raises:
        FileNotFoundError: If no template exists for size N.
    """
    tflite_path = os.path.join(_TEMPLATES_DIR, f"dense_{n}_edgetpu.tflite")
    json_path = os.path.join(_TEMPLATES_DIR, f"dense_{n}_edgetpu.json")

    if not os.path.isfile(tflite_path):
        available = list_templates()
        raise FileNotFoundError(
            f"No template found for matrix size {n}. "
            f"Available sizes: {available}. "
            f"Generate with: python -m libredgetpu.template_gen --sizes {n}"
        )
    return tflite_path, json_path


def list_templates():
    """Return sorted list of available template sizes."""
    sizes = []
    if os.path.isdir(_TEMPLATES_DIR):
        for fname in os.listdir(_TEMPLATES_DIR):
            if fname.startswith("dense_") and fname.endswith("_edgetpu.tflite"):
                try:
                    n = int(fname.split("_")[1])
                    sizes.append(n)
                except (ValueError, IndexError):
                    pass
    return sorted(sizes)
