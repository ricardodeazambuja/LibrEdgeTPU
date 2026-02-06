"""Dev-time template generator for OpticalFlow.

Creates quantized Gabor feature extraction models for Edge TPU with fixed kernels.
The model extracts multi-orientation, multi-scale edge features for optical flow.

Requires edgetpu_compiler for compilation (never imported at runtime).
Model construction uses the analytical TFLite builder (no TensorFlow needed).

Usage:
    python -m libredgetpu.optical_flow_gen --sizes 64 128
    python -m libredgetpu.optical_flow_gen --sizes 64 --output-dir /tmp/templates
    python -m libredgetpu.optical_flow_gen --pooled --pool-factor 4 --sizes 64
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .tflite_builder import build_optical_flow, build_optical_flow_pooled


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

    if "Model compiled successfully" not in result.stdout:
        print(f"Warning: compilation output:\n{result.stdout}", file=sys.stderr)

    base = os.path.splitext(os.path.basename(tflite_path))[0]
    compiled_path = os.path.join(output_dir, f"{base}_edgetpu.tflite")
    if not os.path.isfile(compiled_path):
        raise FileNotFoundError(f"Expected compiled model at {compiled_path}")
    return compiled_path


def generate_template(height, width, ksize=7, orientations=4,
                      sigmas=(1.5, 3.0), output_dir=None):
    """Generate an optical flow Gabor Edge TPU template with sidecar metadata.

    Creates:
        gabor_{h}x{w}_7k_4o_2s_edgetpu.tflite  — compiled Edge TPU model
        gabor_{h}x{w}_7k_4o_2s_edgetpu.json    — quantization metadata sidecar
        gabor_{h}x{w}_7k_4o_2s.tflite          — uncompiled quantized TFLite

    Args:
        height: Input image height.
        width: Input image width.
        ksize: Gabor kernel size (default 7).
        orientations: Number of orientations (default 4).
        sigmas: Gaussian envelope scales (default (1.5, 3.0)).
        output_dir: Directory to write output files.

    Returns:
        (tflite_path, json_path) tuple.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "optical_flow", "templates")
    os.makedirs(output_dir, exist_ok=True)

    n_scales = len(sigmas)
    base_name = f"gabor_{height}x{width}_{ksize}k_{orientations}o_{n_scales}s"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Build quantized model analytically (no TF needed)
        print(f"  Creating Gabor model {height}x{width} "
              f"(k={ksize}, o={orientations}, s={n_scales})...")
        tflite_bytes, metadata = build_optical_flow(
            height, width, ksize=ksize, orientations=orientations, sigmas=sigmas)

        uncompiled_path = os.path.join(tmpdir, f"{base_name}.tflite")
        with open(uncompiled_path, "wb") as f:
            f.write(tflite_bytes)

        # Step 2: Compile for Edge TPU
        print(f"  Compiling for Edge TPU...")
        compiled_path = _compile_for_edgetpu(uncompiled_path, tmpdir)

        # Step 3: Copy to output directory
        final_tflite = os.path.join(output_dir, f"{base_name}_edgetpu.tflite")
        shutil.copy2(compiled_path, final_tflite)

        # Save uncompiled TFLite for reference
        final_uncompiled = os.path.join(output_dir, f"{base_name}.tflite")
        shutil.copy2(uncompiled_path, final_uncompiled)

        # Copy the compiler log if present
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


def generate_pooled_template(height, width, ksize=7, orientations=4,
                             sigmas=(1.5, 3.0), pool_factor=4, output_dir=None):
    """Generate an optical flow Gabor+Pool Edge TPU template with sidecar metadata.

    Creates:
        gabor_{h}x{w}_p{pool}_edgetpu.tflite  — compiled Edge TPU model
        gabor_{h}x{w}_p{pool}_edgetpu.json    — quantization metadata sidecar
        gabor_{h}x{w}_p{pool}.tflite          — uncompiled quantized TFLite

    Args:
        height: Input image height (must be divisible by pool_factor).
        width: Input image width (must be divisible by pool_factor).
        ksize: Gabor kernel size (default 7).
        orientations: Number of orientations (default 4).
        sigmas: Gaussian envelope scales (default (1.5, 3.0)).
        pool_factor: Spatial pooling factor (default 4).
        output_dir: Directory to write output files.

    Returns:
        (tflite_path, json_path) tuple.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "optical_flow", "templates")
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"gabor_{height}x{width}_p{pool_factor}"

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Creating Gabor+Pool model {height}x{width} pool={pool_factor} "
              f"(k={ksize}, o={orientations}, s={len(sigmas)})...")
        tflite_bytes, metadata = build_optical_flow_pooled(
            height, width, ksize=ksize, orientations=orientations,
            sigmas=sigmas, pool_factor=pool_factor)

        uncompiled_path = os.path.join(tmpdir, f"{base_name}.tflite")
        with open(uncompiled_path, "wb") as f:
            f.write(tflite_bytes)

        print(f"  Compiling for Edge TPU...")
        compiled_path = _compile_for_edgetpu(uncompiled_path, tmpdir)

        final_tflite = os.path.join(output_dir, f"{base_name}_edgetpu.tflite")
        shutil.copy2(compiled_path, final_tflite)

        final_uncompiled = os.path.join(output_dir, f"{base_name}.tflite")
        shutil.copy2(uncompiled_path, final_uncompiled)

        log_path = os.path.join(tmpdir, f"{base_name}_edgetpu.log")
        if os.path.isfile(log_path):
            shutil.copy2(log_path, os.path.join(output_dir, f"{base_name}_edgetpu.log"))

    final_json = os.path.join(output_dir, f"{base_name}_edgetpu.json")
    with open(final_json, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"  Saved: {final_tflite}")
    print(f"  Saved: {final_json}")
    return final_tflite, final_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate optical flow Gabor Edge TPU templates",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64, 128],
        help="Image sizes to generate (square, default: 64 128)",
    )
    parser.add_argument(
        "--ksize", type=int, default=7,
        help="Gabor kernel size (default: 7)",
    )
    parser.add_argument(
        "--orientations", type=int, default=4,
        help="Number of orientations (default: 4)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: libredgetpu/optical_flow/templates/)",
    )
    parser.add_argument(
        "--pooled", action="store_true",
        help="Generate pooled (Gabor+AVG_POOL) templates instead of Gabor-only",
    )
    parser.add_argument(
        "--pool-factor", type=int, default=4,
        help="Pool factor for --pooled mode (default: 4)",
    )
    args = parser.parse_args()

    for size in args.sizes:
        if args.pooled:
            print(f"Generating pooled template for Gabor+Pool {size}x{size} "
                  f"pool={args.pool_factor}...")
            generate_pooled_template(size, size, ksize=args.ksize,
                                     orientations=args.orientations,
                                     pool_factor=args.pool_factor,
                                     output_dir=args.output_dir)
        else:
            print(f"Generating template for Gabor {size}x{size}...")
            generate_template(size, size, ksize=args.ksize,
                              orientations=args.orientations,
                              output_dir=args.output_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
