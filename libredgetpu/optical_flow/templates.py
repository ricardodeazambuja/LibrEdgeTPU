"""Pre-compiled optical flow Gabor Edge TPU templates.

Templates are generated at dev time using optical_flow_gen.py and shipped
with the package. Each template consists of:
  - gabor_{h}x{w}_7k_4o_2s_edgetpu.tflite  (compiled Edge TPU model)
  - gabor_{h}x{w}_7k_4o_2s.tflite           (uncompiled quantized TFLite)
  - gabor_{h}x{w}_7k_4o_2s_edgetpu.json    (quantization metadata sidecar)
"""

import os
from typing import Tuple, List

__all__ = ["get_template", "get_pooled_template", "list_templates"]

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def get_template(size: int) -> Tuple[str, str]:
    """Locate a pre-compiled Gabor template for given image size.

    Args:
        size: Square image dimension (height = width).

    Returns:
        (tflite_path, json_path) tuple with absolute paths.

    Raises:
        FileNotFoundError: If no template exists for the specified size.
    """
    base_name = f"gabor_{size}x{size}_7k_4o_2s"
    tflite_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.tflite")
    json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

    if not os.path.isfile(tflite_path):
        available = list_templates()
        raise FileNotFoundError(
            f"No template found for size {size}x{size}. "
            f"Available templates: {available}. "
            f"Generate with: python -m libredgetpu.optical_flow_gen --sizes {size}"
        )
    return tflite_path, json_path


def get_pooled_template(size: int, pool_factor: int = 4) -> Tuple[str, str]:
    """Locate a pre-compiled Gabor+Pool template for given image size.

    Args:
        size: Square image dimension (height = width).
        pool_factor: Spatial pooling factor (default 4).

    Returns:
        (tflite_path, json_path) tuple with absolute paths.

    Raises:
        FileNotFoundError: If no pooled template exists for the specified size.
    """
    base_name = f"gabor_{size}x{size}_p{pool_factor}"
    tflite_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.tflite")
    json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

    if not os.path.isfile(tflite_path):
        available = list_templates()
        raise FileNotFoundError(
            f"No pooled template found for size {size}x{size} pool={pool_factor}. "
            f"Available templates: {available}. "
            f"Generate with: python -m libredgetpu.optical_flow_gen "
            f"--pooled --pool-factor {pool_factor} --sizes {size}"
        )
    return tflite_path, json_path


def list_templates() -> List[Tuple[int, int]]:
    """Return sorted list of available template configurations.

    Returns:
        List of (height, width) tuples.
    """
    templates = []
    if os.path.isdir(_TEMPLATES_DIR):
        for fname in os.listdir(_TEMPLATES_DIR):
            if fname.startswith("gabor_") and fname.endswith("_edgetpu.tflite"):
                # Parse gabor_{h}x{w}_7k_4o_2s_edgetpu.tflite
                try:
                    parts = fname.replace("_edgetpu.tflite", "").split("_")
                    # parts = ["gabor", "{h}x{w}", "7k", "4o", "2s"]
                    hw = parts[1].split("x")
                    h, w = int(hw[0]), int(hw[1])
                    templates.append((h, w))
                except (ValueError, IndexError):
                    pass
    return sorted(templates)
