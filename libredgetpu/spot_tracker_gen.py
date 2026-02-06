"""Dev-time template generator for SpotTracker.

Creates quantized visual servoing models for Edge TPU using soft argmax.
The model computes (x_offset, y_offset) from image center using a soft centroid.

Requires edgetpu_compiler for compilation (never imported at runtime).
Model construction uses the analytical TFLite builder (no TensorFlow needed).

Usage:
    python -m libredgetpu.spot_tracker_gen --sizes 64 128
    python -m libredgetpu.spot_tracker_gen --sizes 64 --variant color_red
    python -m libredgetpu.spot_tracker_gen --sizes 64 --output-dir /tmp/templates
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .tflite_builder import build_spot_tracker


# Color filter presets for RGB tracking
# Each filter is [R, G, B] coefficients for a 1x1 convolution
COLOR_FILTERS = {
    "red": [1.0, -0.5, -0.5],     # High R, low G, low B
    "green": [-0.5, 1.0, -0.5],   # Low R, high G, low B
    "blue": [-0.5, -0.5, 1.0],    # Low R, low G, high B
    "yellow": [0.5, 0.5, -1.0],   # High R+G, low B
    "white": [0.33, 0.33, 0.33],  # Average of all channels
    "cyan": [-0.5, 0.5, 0.5],     # Low R, high G+B
    "magenta": [0.5, -0.5, 0.5],  # High R+B, low G
}


def _compile_for_edgetpu(tflite_path: str, output_dir: str):
    """Run edgetpu_compiler on a TFLite model. Returns path to compiled model."""
    if shutil.which("edgetpu_compiler") is None:
        raise FileNotFoundError(
            "edgetpu_compiler not found on PATH. "
            "Install from https://coral.ai/docs/edgetpu/compiler/"
        )
    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", output_dir, tflite_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"edgetpu_compiler stderr:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"edgetpu_compiler failed with exit code {result.returncode}")

    if "Model compiled successfully" not in result.stdout:
        print(f"Warning: compilation output:\n{result.stdout}", file=sys.stderr)

    base = os.path.splitext(os.path.basename(tflite_path))[0]
    compiled_path = os.path.join(output_dir, f"{base}_edgetpu.tflite")
    if not os.path.isfile(compiled_path):
        raise FileNotFoundError(f"Expected compiled model at {compiled_path}")
    return compiled_path


def generate_template(height: int, width: int, variant: str = "bright",
                      temperature: float = 0.1, output_dir: str = None):
    """Generate a spot tracker Edge TPU template with sidecar metadata.

    Creates:
        {variant}_{h}x{w}_edgetpu.tflite  — compiled Edge TPU model
        {variant}_{h}x{w}_edgetpu.json    — quantization metadata sidecar
        {variant}_{h}x{w}.tflite          — uncompiled quantized TFLite

    Args:
        height: Input image height.
        width: Input image width.
        variant: "bright" for grayscale, or "color_{color}" for color tracking.
        temperature: Softmax temperature (default 0.1 for sharp peaks).
        output_dir: Directory to write output files.

    Returns:
        (tflite_path, json_path) tuple.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "tracker", "templates")
    os.makedirs(output_dir, exist_ok=True)

    # Determine variant type and build model
    if variant == "bright":
        base_name = f"bright_{height}x{width}"
        print(f"  Creating bright spot tracker {height}x{width}...")
        tflite_bytes, metadata = build_spot_tracker(
            height, width, variant="bright", temperature=temperature
        )
    elif variant.startswith("color_"):
        color_name = variant.split("_", 1)[1]
        if color_name not in COLOR_FILTERS:
            raise ValueError(f"Unknown color: {color_name}. Available: {list(COLOR_FILTERS.keys())}")
        base_name = f"color_{color_name}_{height}x{width}"
        print(f"  Creating color tracker ({color_name}) {height}x{width}...")
        tflite_bytes, metadata = build_spot_tracker(
            height, width, variant=variant, temperature=temperature,
            color_weights=COLOR_FILTERS[color_name],
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'bright' or 'color_{{color}}'")

    with tempfile.TemporaryDirectory() as tmpdir:

        uncompiled_path = os.path.join(tmpdir, f"{base_name}.tflite")
        with open(uncompiled_path, "wb") as f:
            f.write(tflite_bytes)

        # Compile for Edge TPU
        print(f"  Compiling for Edge TPU...")
        compiled_path = _compile_for_edgetpu(uncompiled_path, tmpdir)

        # Copy to output directory
        final_tflite = os.path.join(output_dir, f"{base_name}_edgetpu.tflite")
        shutil.copy2(compiled_path, final_tflite)

        final_uncompiled = os.path.join(output_dir, f"{base_name}.tflite")
        shutil.copy2(uncompiled_path, final_uncompiled)

        # Copy compiler log if present
        log_path = os.path.join(tmpdir, f"{base_name}_edgetpu.log")
        if os.path.isfile(log_path):
            shutil.copy2(log_path, os.path.join(output_dir, f"{base_name}_edgetpu.log"))

    # Write sidecar JSON
    final_json = os.path.join(output_dir, f"{base_name}_edgetpu.json")
    with open(final_json, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"  Saved: {final_tflite}")
    print(f"  Saved: {final_json}")
    return final_tflite, final_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate spot tracker Edge TPU templates",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128],
        help="Image sizes to generate (square, default: 64 128)",
    )
    parser.add_argument(
        "--variant", type=str, default="bright",
        help="Tracker variant: 'bright' (grayscale) or 'color_red', 'color_green', etc.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Softmax temperature (default: 0.1 for sharp peaks)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: libredgetpu/tracker/templates/)",
    )
    parser.add_argument(
        "--all-colors", action="store_true",
        help="Generate templates for all color filters",
    )
    parser.add_argument(
        "--color-weights", type=float, nargs=3, metavar=("R", "G", "B"),
        help="Custom [R, G, B] filter coefficients. Registers them as the "
             "variant name given by --variant (e.g., --variant color_orange "
             "--color-weights 1.0 0.5 -0.5)",
    )
    args = parser.parse_args()

    # Register custom color weights if provided
    if args.color_weights is not None:
        if not args.variant.startswith("color_"):
            parser.error("--color-weights requires --variant color_<name> "
                         "(e.g., --variant color_orange)")
        color_name = args.variant.split("_", 1)[1]
        COLOR_FILTERS[color_name] = args.color_weights
        print(f"Registered custom color '{color_name}': {args.color_weights}")

    variants = [args.variant]
    if args.all_colors:
        variants = ["bright"] + [f"color_{c}" for c in COLOR_FILTERS.keys()]

    for size in args.sizes:
        for variant in variants:
            print(f"Generating {variant} template for {size}x{size}...")
            generate_template(size, size, variant=variant,
                            temperature=args.temperature, output_dir=args.output_dir)
            print()

    print("Done.")


if __name__ == "__main__":
    main()
