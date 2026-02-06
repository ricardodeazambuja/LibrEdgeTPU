"""Dev-time template generator for LoomingDetector.

Creates quantized looming detection models for Edge TPU with fixed Sobel kernels.
The model computes edge density in 3x3 spatial zones for collision avoidance.

Requires edgetpu_compiler for compilation (never imported at runtime).
Model construction uses the analytical TFLite builder (no TensorFlow needed).

Usage:
    python -m libredgetpu.looming_gen --sizes 64 128
    python -m libredgetpu.looming_gen --sizes 64 --output-dir /tmp/templates
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .tflite_builder import build_looming


def _compile_for_edgetpu(tflite_path, output_dir):
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

    # Check for "fully mapped" in output
    if "Model compiled successfully" not in result.stdout:
        print(f"Warning: compilation output:\n{result.stdout}", file=sys.stderr)

    base = os.path.splitext(os.path.basename(tflite_path))[0]
    compiled_path = os.path.join(output_dir, f"{base}_edgetpu.tflite")
    if not os.path.isfile(compiled_path):
        raise FileNotFoundError(f"Expected compiled model at {compiled_path}")
    return compiled_path


def generate_template(height, width, zones=3, output_dir=None):
    """Generate a looming detection Edge TPU template with sidecar metadata.

    Creates:
        looming_{h}x{w}_{z}x{z}_edgetpu.tflite  — compiled Edge TPU model
        looming_{h}x{w}_{z}x{z}_edgetpu.json    — quantization metadata sidecar
        looming_{h}x{w}_{z}x{z}.tflite          — uncompiled quantized TFLite

    Args:
        height: Input image height.
        width: Input image width.
        zones: Number of zones per dimension (default 3 for 3x3=9 zones).
        output_dir: Directory to write output files.

    Returns:
        (tflite_path, json_path) tuple.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "looming", "templates")
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"looming_{height}x{width}_{zones}x{zones}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Build quantized model analytically (no TF needed)
        print(f"  Creating looming model {height}x{width} with {zones}x{zones} zones...")
        tflite_bytes, metadata = build_looming(height, width, zones=zones)

        uncompiled_path = os.path.join(tmpdir, f"{base_name}.tflite")
        with open(uncompiled_path, "wb") as f:
            f.write(tflite_bytes)

        # Step 2: Compile for Edge TPU
        print(f"  Compiling for Edge TPU...")
        compiled_path = _compile_for_edgetpu(uncompiled_path, tmpdir)

        # Step 3: Copy to output directory
        final_tflite = os.path.join(output_dir, f"{base_name}_edgetpu.tflite")
        shutil.copy2(compiled_path, final_tflite)

        # Step 3b: Save uncompiled TFLite for reference
        final_uncompiled = os.path.join(output_dir, f"{base_name}.tflite")
        shutil.copy2(uncompiled_path, final_uncompiled)

        # Also copy the compiler log if present
        log_path = os.path.join(tmpdir, f"{base_name}_edgetpu.log")
        if os.path.isfile(log_path):
            shutil.copy2(log_path, os.path.join(output_dir, f"{base_name}_edgetpu.log"))

    # Step 4: Write sidecar JSON
    final_json = os.path.join(output_dir, f"{base_name}_edgetpu.json")
    with open(final_json, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"  Saved: {final_tflite}")
    print(f"  Saved: {final_json}")
    return final_tflite, final_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate looming detection Edge TPU templates",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128],
        help="Image sizes to generate (square, default: 64 128)",
    )
    parser.add_argument(
        "--zones", type=int, default=3,
        help="Number of zones per dimension (default: 3 for 3x3=9 zones)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: libredgetpu/looming/templates/)",
    )
    args = parser.parse_args()

    for size in args.sizes:
        print(f"Generating template for looming {size}x{size}...")
        generate_template(size, size, zones=args.zones, output_dir=args.output_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
