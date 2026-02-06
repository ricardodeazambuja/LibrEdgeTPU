"""Pre-compiled looming detection Edge TPU templates.

Templates are generated at dev time using looming_gen.py and shipped
with the package. Each template consists of:
  - looming_{h}x{w}_{z}x{z}_edgetpu.tflite  (compiled Edge TPU model)
  - looming_{h}x{w}_{z}x{z}.tflite           (uncompiled quantized TFLite)
  - looming_{h}x{w}_{z}x{z}_edgetpu.json    (quantization metadata sidecar)
"""

import os
from typing import Tuple, List

__all__ = ["get_template", "list_templates"]

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def get_template(size: int, zones: int = 3) -> Tuple[str, str]:
    """Locate a pre-compiled template for given image size and zone count.

    Args:
        size: Square image dimension (height = width).
        zones: Number of zones per dimension (default 3 for 3x3=9 zones).

    Returns:
        (tflite_path, json_path) tuple with absolute paths.

    Raises:
        FileNotFoundError: If no template exists for the specified configuration.
    """
    base_name = f"looming_{size}x{size}_{zones}x{zones}"
    tflite_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.tflite")
    json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

    if not os.path.isfile(tflite_path):
        available = list_templates()
        raise FileNotFoundError(
            f"No template found for size {size}x{size} with {zones}x{zones} zones. "
            f"Available templates: {available}. "
            f"Generate with: python -m libredgetpu.looming_gen --sizes {size} --zones {zones}"
        )
    return tflite_path, json_path


def list_templates() -> List[Tuple[int, int, int]]:
    """Return sorted list of available template configurations.

    Returns:
        List of (height, width, zones) tuples.
    """
    templates = []
    if os.path.isdir(_TEMPLATES_DIR):
        for fname in os.listdir(_TEMPLATES_DIR):
            if fname.startswith("looming_") and fname.endswith("_edgetpu.tflite"):
                # Parse looming_{h}x{w}_{z}x{z}_edgetpu.tflite
                try:
                    parts = fname.replace("_edgetpu.tflite", "").split("_")
                    # parts = ["looming", "{h}x{w}", "{z}x{z}"]
                    hw = parts[1].split("x")
                    h, w = int(hw[0]), int(hw[1])
                    zz = parts[2].split("x")
                    z = int(zz[0])
                    templates.append((h, w, z))
                except (ValueError, IndexError):
                    pass
    return sorted(templates)
