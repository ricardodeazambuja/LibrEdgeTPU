"""Pre-compiled pattern tracker Edge TPU templates.

Templates are generated at dev time using pattern_tracker_gen.py and shipped
with the package. Each template consists of:
  - pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch_edgetpu.tflite  (compiled Edge TPU model)
  - pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch.tflite           (uncompiled quantized TFLite)
  - pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch_edgetpu.json    (quantization metadata sidecar)

Naming convention: pattern_{search_h}x{search_w}_{kernel_h}x{kernel_w}_{channels}ch
"""

import os
from typing import Tuple, List

__all__ = ["get_template", "list_templates"]

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def get_template(search_size: int, kernel_size: int,
                 channels: int = 1) -> Tuple[str, str]:
    """Locate a pre-compiled template for given configuration.

    Args:
        search_size: Square search image dimension (height = width).
        kernel_size: Square template/kernel dimension.
        channels: Input channels (1=grayscale, 3=RGB).

    Returns:
        (tflite_path, json_path) tuple with absolute paths.

    Raises:
        FileNotFoundError: If no template exists for the specified configuration.
    """
    base_name = f"pattern_{search_size}x{search_size}_{kernel_size}x{kernel_size}_{channels}ch"
    tflite_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.tflite")
    json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

    if not os.path.isfile(tflite_path):
        available = list_templates()
        available_str = ", ".join(
            f"{sh}x{sw}/{kh}x{kw}/{ch}ch" for sh, sw, kh, kw, ch in available
        ) if available else "none"
        raise FileNotFoundError(
            f"No template found for search={search_size}x{search_size}, "
            f"kernel={kernel_size}x{kernel_size}, channels={channels}. "
            f"Available templates: {available_str}. "
            f"Generate with: python -m libredgetpu.pattern_tracker_gen "
            f"--search-sizes {search_size} --template-sizes {kernel_size} "
            f"--channels {channels}"
        )
    return tflite_path, json_path


def list_templates() -> List[Tuple[int, int, int, int, int]]:
    """Return sorted list of available template configurations.

    Returns:
        List of (search_h, search_w, kernel_h, kernel_w, channels) tuples.
    """
    templates = []
    if os.path.isdir(_TEMPLATES_DIR):
        for fname in os.listdir(_TEMPLATES_DIR):
            if fname.startswith("pattern_") and fname.endswith("_edgetpu.tflite"):
                try:
                    # Parse: pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch_edgetpu.tflite
                    base = fname.replace("_edgetpu.tflite", "")
                    parts = base.split("_")
                    # parts = ["pattern", "{sh}x{sw}", "{kh}x{kw}", "{ch}ch"]
                    if len(parts) != 4:
                        continue
                    search_hw = parts[1].split("x")
                    kernel_hw = parts[2].split("x")
                    ch_str = parts[3].replace("ch", "")

                    sh, sw = int(search_hw[0]), int(search_hw[1])
                    kh, kw = int(kernel_hw[0]), int(kernel_hw[1])
                    ch = int(ch_str)
                    templates.append((sh, sw, kh, kw, ch))
                except (ValueError, IndexError):
                    pass
    return sorted(templates)


def get_available_search_sizes(kernel_size: int = None,
                                channels: int = None) -> List[int]:
    """Return list of available search image sizes.

    Args:
        kernel_size: Filter to this kernel size (optional).
        channels: Filter to this channel count (optional).

    Returns:
        List of available square search sizes.
    """
    sizes = set()
    for sh, sw, kh, kw, ch in list_templates():
        if sh != sw:
            continue
        if kernel_size is not None and kh != kernel_size:
            continue
        if channels is not None and ch != channels:
            continue
        sizes.add(sh)
    return sorted(sizes)


def get_available_kernel_sizes(search_size: int = None,
                                channels: int = None) -> List[int]:
    """Return list of available kernel/template sizes.

    Args:
        search_size: Filter to this search size (optional).
        channels: Filter to this channel count (optional).

    Returns:
        List of available square kernel sizes.
    """
    sizes = set()
    for sh, sw, kh, kw, ch in list_templates():
        if kh != kw:
            continue
        if search_size is not None and sh != search_size:
            continue
        if channels is not None and ch != channels:
            continue
        sizes.add(kh)
    return sorted(sizes)
