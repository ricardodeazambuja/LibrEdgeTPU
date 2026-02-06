"""Dev-time template generator for MatMulEngine.

Creates quantized Dense(N×N) TFLite models, compiles them for Edge TPU,
and saves sidecar JSON with quantization metadata.

Requires edgetpu_compiler (never imported at runtime).  TFLite models are
built directly via tflite_builder — no TensorFlow needed.

Legacy TF workflow (for custom architectures not covered by the builder):
    pip install tensorflow>=2.15  (or tensorflow-cpu)
    # Tested with TF 2.15, 2.16, 2.17

Usage:
    python -m libredgetpu.template_gen --sizes 256 512 1024
    python -m libredgetpu.template_gen --sizes 256 --output-dir /tmp/templates
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _create_and_quantize_model(n, output_dir):
    """Create a Dense(N, N) quantized TFLite model, return (tflite_bytes, metadata)."""
    from .tflite_builder import build_dense
    return build_dense(n)


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


def generate_template(n, output_dir):
    """Generate a Dense(N×N) Edge TPU template with sidecar metadata.

    Creates:
        dense_{n}_edgetpu.tflite  — compiled Edge TPU model
        dense_{n}_edgetpu.json    — quantization metadata sidecar
        dense_{n}.tflite          — uncompiled quantized TFLite (for runtime recompilation)

    Args:
        n: Matrix size (square NxN).
        output_dir: Directory to write output files.

    Returns:
        (tflite_path, json_path) tuple.
    """
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Create and quantize model
        print(f"  Creating Dense({n}, {n}) model...")
        tflite_bytes, metadata = _create_and_quantize_model(n, tmpdir)

        uncompiled_path = os.path.join(tmpdir, f"dense_{n}.tflite")
        with open(uncompiled_path, "wb") as f:
            f.write(tflite_bytes)

        # Step 2: Compile for Edge TPU
        print(f"  Compiling for Edge TPU...")
        compiled_path = _compile_for_edgetpu(uncompiled_path, tmpdir)

        # Step 3: Verify param_size matches DarwiNN parameter blob
        # (The compiler may add padding — detect and record actual size)
        from .tflite_parser import parse as parse_tflite
        from .delegate import parse_darwinn, TYPE_PARAMETER_CACHING

        with open(compiled_path, "rb") as f:
            compiled_bytes = f.read()
        model = parse_tflite(compiled_bytes)
        executables = parse_darwinn(model.custom_op_data)
        for exe in executables:
            if exe.exec_type == TYPE_PARAMETER_CACHING and exe.parameters:
                metadata["param_size"] = len(exe.parameters)
                # Extract overhead bytes for compiler-free blob generation
                from .matmul_engine import _extract_overhead
                overhead = _extract_overhead(exe.parameters, n)
                metadata["param_overhead"] = base64.b64encode(overhead).decode("ascii")
                break

        # Step 4: Copy to output directory
        final_tflite = os.path.join(output_dir, f"dense_{n}_edgetpu.tflite")
        shutil.copy2(compiled_path, final_tflite)

        # Step 4b: Save uncompiled TFLite for runtime recompilation
        final_uncompiled = os.path.join(output_dir, f"dense_{n}.tflite")
        shutil.copy2(uncompiled_path, final_uncompiled)

        # Also copy the compiler log if present
        log_path = os.path.join(tmpdir, f"dense_{n}_edgetpu.log")
        if os.path.isfile(log_path):
            shutil.copy2(log_path, os.path.join(output_dir, f"dense_{n}_edgetpu.log"))

    # Step 5: Write sidecar JSON
    final_json = os.path.join(output_dir, f"dense_{n}_edgetpu.json")
    with open(final_json, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"  Saved: {final_tflite}")
    print(f"  Saved: {final_json}")
    return final_tflite, final_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate Dense(N×N) Edge TPU templates for MatMulEngine",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[256, 512, 1024],
        help="Matrix sizes to generate (default: 256 512 1024)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: libredgetpu/templates/)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "templates")

    for n in args.sizes:
        print(f"Generating template for Dense({n}, {n})...")
        generate_template(n, args.output_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
