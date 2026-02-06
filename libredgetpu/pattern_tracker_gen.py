"""Dev-time template generator for PatternTracker.

Creates quantized template-matching models for Edge TPU using Conv2D sliding
correlation + soft argmax peak detection. The model locates a reference patch
within a larger search image.

Requires edgetpu_compiler for compilation (never imported at runtime).
Model construction uses the analytical TFLite builder (no TensorFlow needed).

Usage:
    python -m libredgetpu.pattern_tracker_gen --search-sizes 128 --template-sizes 16
    python -m libredgetpu.pattern_tracker_gen --search-sizes 64 128 256 --template-sizes 8 16 32 --channels 1
    python -m libredgetpu.pattern_tracker_gen --search-sizes 128 --template-sizes 16 --channels 3
    python -m libredgetpu.pattern_tracker_gen --output-dir /tmp/templates
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .tflite_builder import build_pattern_tracker


def _compile_for_edgetpu(tflite_path: str, output_dir: str):
    """Run edgetpu_compiler on a TFLite model. Returns path to compiled model."""
    if shutil.which("edgetpu_compiler") is None:
        raise FileNotFoundError(
            "edgetpu_compiler not found on PATH. "
            "Install from https://coral.ai/docs/edgetpu/compiler/"
        )
    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-t", "300", "-o", output_dir, tflite_path],
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


def _extract_conv2d_offsets(tflite_bytes, kernel_h, kernel_w, channels,
                            compile_fn):
    """Discover Conv2D weight-to-blob-offset mapping via bulk probing.

    Uses unique int8 probe values (1..127 per batch) to map each Conv2D
    weight to its byte position in the DarwiNN parameter blob. This enables
    compiler-free template swapping at runtime.

    The compile_fn returns a dict of {blob_name: blob_bytes} for all
    parameter blobs in the compiled model. The function automatically
    detects which blob contains the Conv2D weights (may be PC params
    or EO params depending on the model's caching strategy).

    Args:
        tflite_bytes: Uncompiled quantized TFLite model bytes.
        kernel_h, kernel_w: Conv2D kernel dimensions.
        channels: Number of input channels.
        compile_fn: Callable(tflite_bytes) -> dict[str, bytes].
                    Compiles a TFLite model and returns all parameter blobs
                    as {"pc_params": bytes, "eo_params": bytes}.

    Returns:
        (blob_name, offsets) tuple where:
            blob_name: Which blob contains the weights ("pc_params" or "eo_params").
            offsets: List of ints, blob byte offset per Conv2D weight (length = N).

    Raises:
        RuntimeError: If offset extraction fails.
    """
    from .tflite_parser import parse_full
    from ._constants import SIGN_BIT_FLIP

    n_weights = kernel_h * kernel_w * channels
    batch_size = 127  # unique probe values 1..127

    # Find Conv2D weight buffer location in the TFLite flatbuffer
    full = parse_full(tflite_bytes)
    fb_offset = None
    for op in full.operators:
        if op.opcode_name == "CONV_2D":
            weight_idx = op.inputs[1]
            weight_tensor = full.tensors[weight_idx]
            buf_data = full.buffers[weight_tensor.buffer_index]
            if buf_data is not None and len(buf_data) == n_weights:
                fb_offset = full.buffer_offsets[weight_tensor.buffer_index]
                break

    if fb_offset is None or fb_offset < 0:
        raise RuntimeError(
            f"Could not find {n_weights}-byte Conv2D weight buffer in TFLite model"
        )

    def _patch(weights_int8):
        buf = bytearray(tflite_bytes)
        buf[fb_offset:fb_offset + n_weights] = np.asarray(
            weights_int8, dtype=np.int8
        ).tobytes()
        return bytes(buf)

    # Compile base (all -128) and first probe to detect which blob
    # contains the Conv2D weights.
    base_val = -128  # maps to blob byte 0x00 after XOR 0x80
    base_weights = np.full(n_weights, base_val, dtype=np.int8)
    base_blobs = compile_fn(_patch(base_weights))

    # Probe: change weight[0] from -128 to -1 (big change)
    detect_weights = np.full(n_weights, base_val, dtype=np.int8)
    detect_weights[0] = -1
    detect_blobs = compile_fn(_patch(detect_weights))

    # Find which blob changed
    target_blob_name = None
    for name in base_blobs:
        base_arr = np.frombuffer(base_blobs[name], dtype=np.uint8)
        det_arr = np.frombuffer(detect_blobs[name], dtype=np.uint8)
        if len(base_arr) != len(det_arr):
            continue
        n_diffs = np.sum(base_arr != det_arr)
        if n_diffs > 0:
            target_blob_name = name
            break

    if target_blob_name is None:
        raise RuntimeError(
            "Could not detect which parameter blob contains Conv2D weights. "
            "Neither PC nor EO blobs changed when weights were modified."
        )

    base_arr = np.frombuffer(base_blobs[target_blob_name], dtype=np.uint8)

    # Bulk probe: batches of up to 127 unique values
    n_batches = math.ceil(n_weights / batch_size)
    mapping = {}

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_weights)
        batch_count = end - start

        weights = np.full(n_weights, base_val, dtype=np.int8)
        for i in range(batch_count):
            weights[start + i] = np.int8(base_val + i + 1)

        probe_blobs = compile_fn(_patch(weights))
        probe_arr = np.frombuffer(
            probe_blobs[target_blob_name], dtype=np.uint8
        )

        diff_mask = probe_arr != base_arr

        for i in range(batch_count):
            expected_byte = ((base_val + i + 1) ^ SIGN_BIT_FLIP) & 0xFF
            candidates = np.where(diff_mask & (probe_arr == expected_byte))[0]

            if len(candidates) == 1:
                mapping[start + i] = int(candidates[0])
            elif len(candidates) > 1:
                xor_diff = probe_arr ^ base_arr
                expected_xor = (i + 1) & 0xFF
                refined = [c for c in candidates if xor_diff[c] == expected_xor]
                if len(refined) == 1:
                    mapping[start + i] = int(refined[0])
                else:
                    raise RuntimeError(
                        f"Ambiguous mapping for weight[{start + i}]: "
                        f"{len(candidates)} candidates, {len(refined)} after filter"
                    )
            else:
                raise RuntimeError(
                    f"No blob offset found for weight[{start + i}] "
                    f"(expected byte 0x{expected_byte:02x} in {target_blob_name})"
                )

    if len(mapping) != n_weights:
        raise RuntimeError(
            f"Incomplete mapping: {len(mapping)}/{n_weights} weights mapped"
        )

    offsets = [mapping[i] for i in range(n_weights)]

    if len(set(offsets)) != n_weights:
        raise RuntimeError(
            f"Non-unique offsets: {len(set(offsets))} unique out of {n_weights}"
        )

    return target_blob_name, offsets


def generate_template(search_h: int, search_w: int,
                      kernel_h: int, kernel_w: int,
                      channels: int = 1,
                      temperature: float = 0.1,
                      output_dir: str = None):
    """Generate a pattern tracker Edge TPU template with sidecar metadata.

    Creates:
        pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch_edgetpu.tflite  — compiled model
        pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch_edgetpu.json    — metadata sidecar
        pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch.tflite          — uncompiled model

    Args:
        search_h, search_w: Search image dimensions.
        kernel_h, kernel_w: Template/kernel dimensions.
        channels: Input channels (1=grayscale, 3=RGB).
        temperature: Softmax temperature (default 0.1).
        output_dir: Directory to write output files.

    Returns:
        (tflite_path, json_path) tuple.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "pattern", "templates")
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"pattern_{search_h}x{search_w}_{kernel_h}x{kernel_w}_{channels}ch"

    print(f"  Creating pattern tracker model "
          f"(search={search_h}x{search_w}, kernel={kernel_h}x{kernel_w}, "
          f"channels={channels})...")

    tflite_bytes, metadata = build_pattern_tracker(
        search_h, search_w, kernel_h, kernel_w,
        channels=channels, temperature=temperature,
    )

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

        # Extract Conv2D weight-to-blob offsets for compiler-free template swapping
        print(f"  Extracting Conv2D blob offsets...")
        n_weights = kernel_h * kernel_w * channels
        n_batches = math.ceil(n_weights / 127)
        print(f"    {n_weights} weights, {n_batches + 1} compilations needed...")

        def _compile_and_get_all_params(patched_tflite_bytes):
            """Compile a patched TFLite and return all parameter blobs."""
            probe_path = os.path.join(tmpdir, "probe.tflite")
            with open(probe_path, "wb") as pf:
                pf.write(patched_tflite_bytes)
            probe_compiled = _compile_for_edgetpu(probe_path, tmpdir)
            with open(probe_compiled, "rb") as cf:
                compiled_bytes = cf.read()
            from .tflite_parser import parse as parse_tflite
            from .delegate import (
                parse_darwinn, TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY,
            )
            model = parse_tflite(compiled_bytes)
            exes = parse_darwinn(model.custom_op_data)
            result = {}
            for exe in exes:
                if exe.exec_type == TYPE_PARAMETER_CACHING and exe.parameters:
                    result["pc_params"] = exe.parameters
                elif exe.exec_type == TYPE_EXECUTION_ONLY and exe.parameters:
                    result["eo_params"] = exe.parameters
            if not result:
                raise RuntimeError("No parameter blobs in compiled model")
            return result

        blob_name, conv_offsets = _extract_conv2d_offsets(
            tflite_bytes, kernel_h, kernel_w, channels,
            _compile_and_get_all_params,
        )
        metadata["conv_weight_offsets"] = conv_offsets
        metadata["conv_weight_blob"] = blob_name
        print(f"    Mapped {len(conv_offsets)} offsets in {blob_name} "
              f"(range [{min(conv_offsets)}, {max(conv_offsets)}])")

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
        description="Generate pattern tracker Edge TPU templates",
    )
    parser.add_argument(
        "--search-sizes", type=int, nargs="+", default=[64, 128, 256],
        help="Search image sizes (square, default: 64 128 256)",
    )
    parser.add_argument(
        "--template-sizes", type=int, nargs="+", default=[8, 16, 32],
        help="Template/kernel sizes (square, default: 8 16 32)",
    )
    parser.add_argument(
        "--channels", type=int, default=1, choices=[1, 3],
        help="Input channels (1=grayscale, 3=RGB, default: 1)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Softmax temperature (default: 0.1 for sharp peaks)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: libredgetpu/pattern/templates/)",
    )
    parser.add_argument(
        "--standard", action="store_true",
        help="Generate the 6 standard template combinations shipped with the package",
    )
    args = parser.parse_args()

    if args.standard:
        # Generate the standard combinations shipped with the package.
        # Correlation map size = (search - kernel + 1)^2 determines model
        # complexity. Maps >~13000 positions may time out in the compiler.
        combos = [
            (64, 64, 8, 8, 1),      # corr 57x57=3249, fast
            (64, 64, 16, 16, 1),     # corr 49x49=2401, larger kernel
            (128, 128, 16, 16, 1),   # corr 113x113=12769, balanced
            (128, 128, 32, 32, 1),   # corr 97x97=9409, large kernel
            (64, 64, 8, 8, 3),       # corr 57x57=3249, RGB fast
            (128, 128, 16, 16, 3),   # corr 113x113=12769, RGB balanced
        ]
        for sh, sw, kh, kw, ch in combos:
            print(f"Generating pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch...")
            generate_template(sh, sw, kh, kw, ch,
                            temperature=args.temperature,
                            output_dir=args.output_dir)
            print()
    else:
        for search_size in args.search_sizes:
            for kernel_size in args.template_sizes:
                if kernel_size >= search_size:
                    print(f"Skipping kernel {kernel_size} >= search {search_size}")
                    continue
                print(f"Generating pattern_{search_size}x{search_size}_{kernel_size}x{kernel_size}_{args.channels}ch...")
                generate_template(
                    search_size, search_size, kernel_size, kernel_size,
                    channels=args.channels,
                    temperature=args.temperature,
                    output_dir=args.output_dir,
                )
                print()

    print("Done.")


if __name__ == "__main__":
    main()
