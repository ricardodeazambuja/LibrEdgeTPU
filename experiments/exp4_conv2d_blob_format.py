"""Experiment 4: Determine Conv2D weight layout in DarwiNN parameter blob.

Goal: Determine the mapping from Conv2D kernel weights (h*w*C*1) to the
DarwiNN parameter blob, so that set_template() can patch weights directly
without recompilation (like Experiment 3 did for Dense layers).

This version uses the **real PatternTracker model architecture** (Conv2D +
ReLU + Softmax + Dense x2 + Concat) via pattern_tracker_gen, ensuring blob
layouts match the shipped templates exactly.

Methodology:
  Phase A: Compile with known constant weights, verify XOR 0x80 transform
  Phase B: Zero weights -> extract overhead bytes (requant multipliers)
  Phase C: Bulk offset mapping using unique int8 probe values (1..127 per
           batch), reducing compilations from N to ceil(N/127)+1 per config
  Phase D: Cross-validate with 5 weight patterns (byte-perfect match)

Compilation count per config:
  64x64/8x8/1ch    (64 weights)  -> 2 compilations
  64x64/16x16/1ch  (256 weights) -> 3 compilations
  128x128/16x16/1ch (256 weights) -> 3 compilations
  128x128/32x32/1ch (1024 weights) -> 10 compilations
  64x64/8x8/3ch    (192 weights) -> 3 compilations
  128x128/16x16/3ch (768 weights) -> 8 compilations

Usage:
    python -m experiments.exp4_conv2d_blob_format
    python -m experiments.exp4_conv2d_blob_format --phase A
    python -m experiments.exp4_conv2d_blob_format --configs 64:8:1 128:16:1
    python -m experiments.exp4_conv2d_blob_format --phase D --configs 64:8:1
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libredgetpu.tflite_parser import parse as parse_tflite, parse_full
from libredgetpu.delegate import parse_darwinn, TYPE_PARAMETER_CACHING
from libredgetpu._constants import SIGN_BIT_FLIP

# Where results are persisted across runs
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "exp4_conv2d")

# Standard 6 configurations matching shipped templates
STANDARD_CONFIGS = [
    (64, 64, 8, 8, 1),
    (64, 64, 16, 16, 1),
    (128, 128, 16, 16, 1),
    (128, 128, 32, 32, 1),
    (64, 64, 8, 8, 3),
    (128, 128, 16, 16, 3),
]


def _check_compiler():
    if shutil.which("edgetpu_compiler") is None:
        sys.exit("ERROR: edgetpu_compiler not found on PATH")


def _config_name(sh, sw, kh, kw, ch):
    return f"pattern_{sh}x{sw}_{kh}x{kw}_{ch}ch"


def _create_and_quantize(sh, sw, kh, kw, ch, kernel_values=None):
    """Create real PatternTracker model, quantize it. Returns tflite_bytes.

    Uses the actual architecture from pattern_tracker_gen to ensure blob
    layouts match the shipped templates.
    """
    from libredgetpu.pattern_tracker_gen import (
        _create_pattern_tracker_model, _quantize_model,
    )

    if kernel_values is not None:
        # Temporarily monkey-patch the kernel initializer
        import tensorflow as tf
        kernel_values = np.asarray(kernel_values, dtype=np.float32)
        kernel_init_val = kernel_values.reshape(kh, kw, ch, 1)

        # Build model with custom kernel
        model = _create_pattern_tracker_model(sh, sw, kh, kw, ch, temperature=0.1)

        # Override the pattern_conv layer weights
        for layer in model.layers:
            if layer.name == "pattern_conv":
                layer.set_weights([kernel_init_val])
                break

        tflite_bytes, _metadata = _quantize_model(model, sh, sw, kh, kw, ch)
    else:
        model = _create_pattern_tracker_model(sh, sw, kh, kw, ch, temperature=0.1)
        tflite_bytes, _metadata = _quantize_model(model, sh, sw, kh, kw, ch)

    return tflite_bytes


def _find_conv_weight_buffer(tflite_bytes, kh, kw, ch):
    """Find Conv2D weight buffer in TFLite flatbuffer.

    Returns (offset, buffer_data, scale, zero_point) or (None, None, None, None).
    """
    full = parse_full(tflite_bytes)
    expected_size = kh * kw * ch

    for op in full.operators:
        if op.opcode_name == "CONV_2D":
            weight_idx = op.inputs[1]
            weight_tensor = full.tensors[weight_idx]
            buf_data = full.buffers[weight_tensor.buffer_index]
            if buf_data is not None and len(buf_data) == expected_size:
                offset = full.buffer_offsets[weight_tensor.buffer_index]
                if offset >= 0:
                    return offset, buf_data, weight_tensor.scale, weight_tensor.zero_point
    return None, None, None, None


def _patch_conv_weights(tflite_bytes, offset, new_int8_weights):
    """Patch Conv2D weight bytes at the given flatbuffer offset."""
    buf = bytearray(tflite_bytes)
    raw = np.asarray(new_int8_weights, dtype=np.int8).tobytes()
    buf[offset:offset + len(raw)] = raw
    return bytes(buf)


def _compile_and_extract_params(tflite_bytes, name, tmpdir):
    """Compile TFLite, return PC parameter blob bytes (or None on failure)."""
    model_path = os.path.join(tmpdir, f"{name}.tflite")
    with open(model_path, "wb") as f:
        f.write(tflite_bytes)

    out_dir = os.path.join(tmpdir, f"out_{name}")
    os.makedirs(out_dir, exist_ok=True)

    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", out_dir, model_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  COMPILER FAILED: {result.stderr[:200]}")
        return None

    compiled_path = os.path.join(out_dir, f"{name}_edgetpu.tflite")
    if not os.path.isfile(compiled_path):
        return None

    with open(compiled_path, "rb") as f:
        compiled_bytes = f.read()

    model = parse_tflite(compiled_bytes)
    exes = parse_darwinn(model.custom_op_data)
    for exe in exes:
        if exe.exec_type == TYPE_PARAMETER_CACHING and exe.parameters:
            return exe.parameters
    return None


def _save_result(config_name, phase, data):
    """Save phase results to persistent JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{config_name}_{phase}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def _load_result(config_name, phase):
    """Load phase results from persistent JSON file, or None if not found."""
    path = os.path.join(RESULTS_DIR, f"{config_name}_{phase}.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Phase A: Constant weights — verify XOR 0x80 transform ───────────────

def phase_a(sh, sw, kh, kw, ch):
    """Compile with known constant weights, verify XOR 0x80 transform."""
    name = _config_name(sh, sw, kh, kw, ch)
    print(f"\n=== Phase A: Constant weights ({name}) ===")
    _check_compiler()

    n_weights = kh * kw * ch
    test_values = [1, 42, 127, -1, -128]

    # Create base model once
    tflite = _create_and_quantize(sh, sw, kh, kw, ch)
    offset, buf, scale, zp = _find_conv_weight_buffer(tflite, kh, kw, ch)
    if offset is None:
        print("  ERROR: could not find weight buffer")
        return

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for val in test_values:
            int8_weights = np.full(n_weights, val, dtype=np.int8)
            patched = _patch_conv_weights(tflite, offset, int8_weights)
            blob = _compile_and_extract_params(patched, f"const_{val}", tmpdir)
            if blob is None:
                print(f"  val={val}: compilation failed")
                continue

            expected_byte = (val ^ SIGN_BIT_FLIP) & 0xFF
            blob_arr = np.frombuffer(blob, dtype=np.uint8)
            matches = np.where(blob_arr == expected_byte)[0]
            print(f"  val={val:4d} -> expected byte 0x{expected_byte:02x}, "
                  f"found {len(matches)} occurrences in {len(blob)}-byte blob "
                  f"(expect ~{n_weights})")
            results[str(val)] = {
                "expected_byte": expected_byte,
                "occurrences": int(len(matches)),
                "blob_size": len(blob),
            }

    _save_result(name, "phase_a", results)


# ── Phase B: Zero weights — extract overhead ─────────────────────────────

def phase_b(sh, sw, kh, kw, ch):
    """Zero weights -> extract overhead bytes."""
    name = _config_name(sh, sw, kh, kw, ch)
    print(f"\n=== Phase B: Zero weights ({name}) ===")
    _check_compiler()

    n_weights = kh * kw * ch

    tflite = _create_and_quantize(sh, sw, kh, kw, ch)
    offset, buf, scale, zp = _find_conv_weight_buffer(tflite, kh, kw, ch)
    if offset is None:
        print("  ERROR: could not find weight buffer")
        return None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        int8_zero = np.zeros(n_weights, dtype=np.int8)
        patched = _patch_conv_weights(tflite, offset, int8_zero)
        blob = _compile_and_extract_params(patched, "zero", tmpdir)
        if blob is None:
            print("  ERROR: compilation failed")
            return None, None

    blob_arr = np.frombuffer(blob, dtype=np.uint8)
    zero_byte = SIGN_BIT_FLIP  # 0x80
    non_zero_mask = blob_arr != zero_byte
    overhead_positions = np.where(non_zero_mask)[0]

    print(f"  Blob size: {len(blob)} bytes")
    print(f"  Non-0x80 (overhead) bytes: {len(overhead_positions)}")
    print(f"  0x80 (zero-weight) bytes: {len(blob) - len(overhead_positions)}")

    _save_result(name, "phase_b", {
        "blob_size": len(blob),
        "overhead_count": int(len(overhead_positions)),
        "overhead_positions": overhead_positions.tolist()[:50],
    })

    return blob, tflite


# ── Phase C: Bulk offset mapping ─────────────────────────────────────────

def phase_c(sh, sw, kh, kw, ch):
    """Bulk offset mapping using unique int8 probe values.

    Instead of probing one weight at a time (N compilations), we fill batches
    of up to 127 weights with unique values 1..127, compile once per batch,
    and XOR the blob to find each weight's offset. This reduces compilations
    from N to ceil(N/127) + 1.
    """
    name = _config_name(sh, sw, kh, kw, ch)
    print(f"\n=== Phase C: Bulk offset mapping ({name}) ===")
    _check_compiler()

    n_weights = kh * kw * ch
    batch_size = 127  # unique values 1..127 per batch
    n_batches = math.ceil(n_weights / batch_size)

    print(f"  {n_weights} weights, {n_batches} batch(es) + 1 zero ref = {n_batches + 1} compilations")

    tflite = _create_and_quantize(sh, sw, kh, kw, ch)
    offset, buf, scale, zp = _find_conv_weight_buffer(tflite, kh, kw, ch)
    if offset is None:
        print("  ERROR: could not find weight buffer")
        return None, None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Compile zero-weight reference
        int8_zero = np.zeros(n_weights, dtype=np.int8)
        patched_zero = _patch_conv_weights(tflite, offset, int8_zero)
        zero_blob = _compile_and_extract_params(patched_zero, "zero_ref", tmpdir)
        if zero_blob is None:
            print("  ERROR: zero blob compilation failed")
            return None, None, None

        zero_arr = np.frombuffer(zero_blob, dtype=np.uint8)

        # 2. Batch probing
        mapping = {}  # weight_index -> blob_offset

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_weights)
            batch_count = end - start

            # Fill weights: positions [start..end) get unique values 1..batch_count
            weights = np.zeros(n_weights, dtype=np.int8)
            for i in range(batch_count):
                weights[start + i] = i + 1  # values 1..127

            patched = _patch_conv_weights(tflite, offset, weights)
            probe_blob = _compile_and_extract_params(
                patched, f"batch_{batch_idx}", tmpdir
            )
            if probe_blob is None:
                print(f"  Batch {batch_idx}: compilation failed")
                continue

            probe_arr = np.frombuffer(probe_blob, dtype=np.uint8)

            # XOR with zero blob to isolate weight contributions
            diff_arr = probe_arr ^ zero_arr

            # For each probe value v (1..batch_count), the expected XOR diff is:
            # (v XOR 0x80) XOR (0 XOR 0x80) = v XOR 0 = v
            # So diff_arr at the weight's blob position should equal v.
            for i in range(batch_count):
                probe_val = i + 1
                candidates = np.where(diff_arr == probe_val)[0]

                if len(candidates) == 1:
                    mapping[start + i] = int(candidates[0])
                elif len(candidates) == 0:
                    print(f"  weight[{start + i}] (val={probe_val}): NO MATCH in diff")
                else:
                    # Multiple matches — shouldn't happen with unique values,
                    # but filter by checking the actual byte value
                    expected_byte = (probe_val ^ SIGN_BIT_FLIP) & 0xFF
                    refined = [c for c in candidates if probe_arr[c] == expected_byte]
                    if len(refined) == 1:
                        mapping[start + i] = int(refined[0])
                    else:
                        print(f"  weight[{start + i}] (val={probe_val}): "
                              f"{len(candidates)} candidates, {len(refined)} after filter")

            mapped_so_far = sum(1 for k in mapping if start <= k < end)
            print(f"  Batch {batch_idx}: mapped {mapped_so_far}/{batch_count} weights "
                  f"(positions [{start}, {end}))")

    print(f"\n  Total mapped: {len(mapping)}/{n_weights}")

    if len(mapping) == n_weights:
        offsets = [mapping[i] for i in range(n_weights)]
        print(f"  Offset range: [{min(offsets)}, {max(offsets)}]")
        print(f"  First 16 offsets: {offsets[:16]}")

        # Check uniqueness
        unique_offsets = set(offsets)
        if len(unique_offsets) != n_weights:
            print(f"  WARNING: only {len(unique_offsets)} unique offsets (expected {n_weights})")
        else:
            print(f"  All offsets unique: YES")

        _save_result(name, "phase_c", {
            "n_weights": n_weights,
            "n_batches": n_batches,
            "blob_size": len(zero_blob),
            "offsets": offsets,
            "all_unique": len(unique_offsets) == n_weights,
        })

        return mapping, len(zero_blob), zero_blob, tflite

    _save_result(name, "phase_c", {
        "n_weights": n_weights,
        "mapped": len(mapping),
        "blob_size": len(zero_blob),
        "partial_mapping": {str(k): v for k, v in mapping.items()},
    })

    return None, None, None, None


# ── Phase D: Cross-validation ────────────────────────────────────────────

def phase_d(sh, sw, kh, kw, ch):
    """Cross-validate discovered offsets with 5 weight patterns."""
    name = _config_name(sh, sw, kh, kw, ch)
    print(f"\n=== Phase D: Cross-validation ({name}) ===")
    _check_compiler()

    # Try to load Phase C results first
    result_c = _load_result(name, "phase_c")
    if result_c and "offsets" in result_c:
        offsets = result_c["offsets"]
        n_weights = result_c["n_weights"]
        blob_size = result_c["blob_size"]
        print(f"  Loaded Phase C results ({n_weights} offsets, blob size {blob_size})")
    else:
        print("  Running Phase C first...")
        ret = phase_c(sh, sw, kh, kw, ch)
        if ret[0] is None:
            print("  FAILED: Phase C could not determine mapping")
            return False
        mapping, blob_size, zero_blob, tflite = ret
        offsets = [mapping[i] for i in range(kh * kw * ch)]
        n_weights = len(offsets)

    # Create base model and get zero blob
    tflite = _create_and_quantize(sh, sw, kh, kw, ch)
    offset, buf, scale, zp = _find_conv_weight_buffer(tflite, kh, kw, ch)
    if offset is None:
        print("  ERROR: could not find weight buffer")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        # Get zero blob for this model
        int8_zero = np.zeros(n_weights, dtype=np.int8)
        patched_zero = _patch_conv_weights(tflite, offset, int8_zero)
        zero_blob = _compile_and_extract_params(patched_zero, "zero_ref_d", tmpdir)
        if zero_blob is None:
            print("  ERROR: zero blob compilation failed")
            return False

        zero_arr = np.frombuffer(zero_blob, dtype=np.uint8).copy()

        # Generate 5 test weight patterns
        rng = np.random.default_rng(12345)
        patterns = {
            "random_uniform": rng.integers(-128, 128, n_weights, dtype=np.int8),
            "gradient": np.linspace(-128, 127, n_weights).astype(np.int8),
            "checkerboard": np.array(
                [127 if (i % 2) == 0 else -128 for i in range(n_weights)], dtype=np.int8
            ),
            "near_zero": rng.integers(-5, 6, n_weights, dtype=np.int8),
            "identity_like": np.eye(
                min(kh, kw), dtype=np.float32
            ).flatten()[:n_weights].astype(np.int8) * 100,
        }

        all_match = True
        results = {}

        for pat_name, int8_weights in patterns.items():
            patched = _patch_conv_weights(tflite, offset, int8_weights)
            compiler_blob = _compile_and_extract_params(
                patched, f"val_{pat_name}", tmpdir
            )
            if compiler_blob is None:
                print(f"  {pat_name}: compilation failed")
                all_match = False
                results[pat_name] = "compilation_failed"
                continue

            # Build predicted blob using discovered offsets
            predicted = zero_arr.copy()
            for wi in range(n_weights):
                blob_off = offsets[wi]
                val = int(int8_weights[wi])
                predicted[blob_off] = (val ^ SIGN_BIT_FLIP) & 0xFF

            compiler_arr = np.frombuffer(compiler_blob, dtype=np.uint8)
            if np.array_equal(predicted, compiler_arr):
                print(f"  {pat_name}: BYTE-PERFECT MATCH")
                results[pat_name] = "perfect"
            else:
                diffs = np.where(predicted != compiler_arr)[0]
                print(f"  {pat_name}: MISMATCH at {len(diffs)} positions")
                for d in diffs[:5]:
                    print(f"    blob[{d}]: predicted=0x{predicted[d]:02x}, "
                          f"compiler=0x{compiler_arr[d]:02x}")
                all_match = False
                results[pat_name] = f"mismatch_{len(diffs)}"

    _save_result(name, "phase_d", {
        "all_match": all_match,
        "patterns": results,
    })

    return all_match


# ── Main ─────────────────────────────────────────────────────────────────

def _parse_config(s):
    """Parse 'SH:KH:CH' config string. Returns (sh, sw, kh, kw, ch)."""
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Config must be SH:KH:CH (e.g. 64:8:1), got '{s}'"
        )
    sh, kh, ch = int(parts[0]), int(parts[1]), int(parts[2])
    return (sh, sh, kh, kh, ch)


def main():
    parser = argparse.ArgumentParser(
        description="Determine Conv2D weight layout in DarwiNN parameter blob",
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["A", "B", "C", "D", "all"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--configs", type=_parse_config, nargs="*", default=None,
                        help="Configs as SH:KH:CH (e.g. 64:8:1 128:16:3). "
                             "Default: all 6 standard configs")
    args = parser.parse_args()

    configs = args.configs if args.configs else STANDARD_CONFIGS

    for sh, sw, kh, kw, ch in configs:
        print(f"\n{'='*60}")
        print(f"Config: {_config_name(sh, sw, kh, kw, ch)}")
        print(f"  weights: {kh*kw*ch}, batches: {math.ceil(kh*kw*ch/127)}")
        print(f"{'='*60}")

        if args.phase in ("A", "all"):
            phase_a(sh, sw, kh, kw, ch)
        if args.phase in ("B", "all"):
            phase_b(sh, sw, kh, kw, ch)
        if args.phase in ("C", "all"):
            phase_c(sh, sw, kh, kw, ch)
        if args.phase in ("D", "all"):
            success = phase_d(sh, sw, kh, kw, ch)
            if success:
                print(f"\n  >>> Conv2D blob format VERIFIED for {_config_name(sh, sw, kh, kw, ch)} <<<")

    print("\nDone. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
