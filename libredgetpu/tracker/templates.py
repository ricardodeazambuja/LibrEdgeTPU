"""Pre-compiled spot tracker Edge TPU templates.

Templates are generated at dev time using spot_tracker_gen.py and shipped
with the package. Each template consists of:
  - {variant}_{h}x{w}_edgetpu.tflite  (compiled Edge TPU model)
  - {variant}_{h}x{w}.tflite           (uncompiled quantized TFLite)
  - {variant}_{h}x{w}_edgetpu.json    (quantization metadata sidecar)

Variants:
  - bright_{h}x{w}: Grayscale bright spot tracking
  - color_{color}_{h}x{w}: RGB color tracking (red, green, blue, etc.)
"""

import math
import os
from typing import Tuple, List, Optional, Sequence

__all__ = ["get_template", "list_templates", "find_closest_color"]

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Color filter presets: [R, G, B] coefficients for the 1x1 Conv2D color filter.
# Must stay in sync with spot_tracker_gen.COLOR_FILTERS.
COLOR_FILTER_WEIGHTS = {
    "red": [1.0, -0.5, -0.5],
    "green": [-0.5, 1.0, -0.5],
    "blue": [-0.5, -0.5, 1.0],
    "yellow": [0.5, 0.5, -1.0],
    "white": [0.33, 0.33, 0.33],
    "cyan": [-0.5, 0.5, 0.5],
    "magenta": [0.5, -0.5, 0.5],
}


def get_template(size: int, variant: str = "bright") -> Tuple[str, str]:
    """Locate a pre-compiled template for given image size and variant.

    Args:
        size: Square image dimension (height = width).
        variant: "bright" for grayscale, or "color_red", "color_green", etc.

    Returns:
        (tflite_path, json_path) tuple with absolute paths.

    Raises:
        FileNotFoundError: If no template exists for the specified configuration.
    """
    if variant == "bright":
        base_name = f"bright_{size}x{size}"
    elif variant.startswith("color_"):
        color_name = variant.split("_", 1)[1]
        base_name = f"color_{color_name}_{size}x{size}"
    else:
        # Assume it's a direct base name
        base_name = f"{variant}_{size}x{size}"

    tflite_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.tflite")
    json_path = os.path.join(_TEMPLATES_DIR, f"{base_name}_edgetpu.json")

    if not os.path.isfile(tflite_path):
        available = list_templates()
        available_str = ", ".join(f"{v}@{s}x{s}" for v, s, _ in available) if available else "none"
        raise FileNotFoundError(
            f"No template found for variant '{variant}' at size {size}x{size}. "
            f"Available templates: {available_str}. "
            f"Generate with: python -m libredgetpu.spot_tracker_gen --sizes {size} --variant {variant}"
        )
    return tflite_path, json_path


def list_templates() -> List[Tuple[str, int, int]]:
    """Return sorted list of available template configurations.

    Returns:
        List of (variant, height, width) tuples.
        variant is "bright" or "color_{color}".
    """
    templates = []
    if os.path.isdir(_TEMPLATES_DIR):
        for fname in os.listdir(_TEMPLATES_DIR):
            if fname.endswith("_edgetpu.tflite"):
                # Parse {variant}_{h}x{w}_edgetpu.tflite
                try:
                    base = fname.replace("_edgetpu.tflite", "")
                    # Split from the right to find dimensions
                    # e.g., "bright_64x64" or "color_red_64x64"
                    parts = base.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    variant_part, dims = parts
                    hw = dims.split("x")
                    if len(hw) != 2:
                        continue
                    h, w = int(hw[0]), int(hw[1])
                    templates.append((variant_part, h, w))
                except (ValueError, IndexError):
                    pass
    return sorted(templates)


def get_available_colors() -> List[str]:
    """Return list of available color variants.

    Returns:
        List of color names (e.g., ["red", "green", "blue"]).
    """
    colors = set()
    for variant, _, _ in list_templates():
        if variant.startswith("color_"):
            colors.add(variant.split("_", 1)[1])
    return sorted(colors)


def get_available_sizes(variant: str = "bright") -> List[int]:
    """Return list of available sizes for a given variant.

    Args:
        variant: "bright" or "color_{color}".

    Returns:
        List of available square sizes (e.g., [64, 128]).
    """
    sizes = set()
    for v, h, w in list_templates():
        if v == variant or (variant.startswith("color_") and v == variant):
            if h == w:  # Only square templates
                sizes.add(h)
    return sorted(sizes)


def find_closest_color(weights: Sequence[float], size: int) -> Tuple[str, float]:
    """Find the pre-compiled color template closest to the given RGB weights.

    Compares against all color templates available at the requested size
    using Euclidean distance between the [R, G, B] coefficient vectors.

    Args:
        weights: [R, G, B] coefficients describing the target color filter.
        size: Square image dimension (must match an available template size).

    Returns:
        (variant, distance) tuple. variant is e.g. "color_red".
        distance is the Euclidean distance (0.0 = exact match).

    Raises:
        FileNotFoundError: If no color templates exist at the requested size.
    """
    if len(weights) != 3:
        raise ValueError(f"Expected 3 color weights [R, G, B], got {len(weights)}")

    available = list_templates()
    # Filter to color templates at the requested size
    candidates = [
        v for v, h, w in available
        if v.startswith("color_") and h == size and w == size
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No color templates available at size {size}x{size}. "
            f"Generate with: python -m libredgetpu.spot_tracker_gen --sizes {size} --all-colors"
        )

    best_variant = None
    best_dist = float("inf")

    for variant in candidates:
        color_name = variant.split("_", 1)[1]
        preset = COLOR_FILTER_WEIGHTS.get(color_name)
        if preset is None:
            continue
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(weights, preset)))
        if dist < best_dist:
            best_dist = dist
            best_variant = variant

    if best_variant is None:
        raise FileNotFoundError("No matching color presets found in templates")

    return best_variant, best_dist
