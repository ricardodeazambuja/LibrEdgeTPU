"""Experiment 3b: Validate compiler-free parameter blob generation.

Uses the mapping discovered in Exp 3:
  - Value transform: int8 XOR 0x80
  - Layout: 64-row groups, each with [overhead:64*8][weights:64*N in 4-col tiles]
  - Overhead bytes are weight-independent (copy from template)

Method:
  1. Extract template blob from compiled Dense(N)
  2. Create test weights, compile with edgetpu_compiler, get reference blob
  3. Generate predicted blob using our formula (template overhead + reordered weights)
  4. Compare byte-for-byte

Usage:
    python -m experiments.exp3b_validate_blob_gen
    python -m experiments.exp3b_validate_blob_gen --sizes 64 128 256
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from libredgetpu.tflite_parser import parse as parse_tflite, parse_full
from libredgetpu.delegate import parse_darwinn, TYPE_PARAMETER_CACHING

ROW_TILE = 64
COL_TILE = 4


def weight_to_blob_offset(row, col, n):
    """Map weight matrix position to byte offset in DarwiNN parameter blob."""
    rg = row // ROW_TILE
    rl = row % ROW_TILE
    group_size = ROW_TILE * 8 + ROW_TILE * n
    group_start = rg * group_size
    weight_start = group_start + ROW_TILE * 8
    cb = col // COL_TILE
    cl = col % COL_TILE
    return weight_start + cb * (ROW_TILE * COL_TILE) + rl * COL_TILE + cl


def generate_blob(template_blob, new_weights_int8, n):
    """Generate a DarwiNN parameter blob from template overhead + new int8 weights.

    Args:
        template_blob: bytes, the reference blob (provides overhead bytes)
        new_weights_int8: np.ndarray shape (n, n), dtype int8
        n: matrix size

    Returns:
        bytes, the predicted parameter blob
    """
    blob = bytearray(template_blob)  # start with template (gets overhead right)

    # Overwrite weight positions with new values (XOR 0x80)
    for row in range(n):
        for col in range(n):
            offset = weight_to_blob_offset(row, col, n)
            blob[offset] = int(new_weights_int8[row, col]) ^ 0x80 & 0xFF

    return bytes(blob)


def generate_blob_fast(template_blob, new_weights_int8, n):
    """Vectorized version of generate_blob using precomputed index array."""
    blob = bytearray(template_blob)

    # Build index mapping: weight_flat_idx -> blob_offset
    rows = np.arange(n).repeat(n)
    cols = np.tile(np.arange(n), n)

    rg = rows // ROW_TILE
    rl = rows % ROW_TILE
    group_size = ROW_TILE * 8 + ROW_TILE * n
    group_start = rg * group_size
    weight_start = group_start + ROW_TILE * 8
    cb = cols // COL_TILE
    cl = cols % COL_TILE
    offsets = weight_start + cb * (ROW_TILE * COL_TILE) + rl * COL_TILE + cl

    # XOR 0x80 and place
    values = (new_weights_int8.flatten().astype(np.uint8) ^ 0x80).astype(np.uint8)
    blob_arr = np.frombuffer(bytes(blob), dtype=np.uint8).copy()
    blob_arr[offsets] = values

    return bytes(blob_arr)


def _create_dense_tflite(n):
    """Create quantized Dense(N,N) TFLite. Returns (bytes, weight_buffer_offset)."""
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=n, input_shape=[n], use_bias=False),
    ])

    def representative_dataset():
        rng = np.random.default_rng(0)
        for _ in range(256):
            yield [rng.uniform(-1.0, 1.0, [1, n]).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_bytes = converter.convert()

    full = parse_full(tflite_bytes)
    expected_size = n * n
    buf_array = bytearray(tflite_bytes)
    for buffer_data in full.buffers:
        if buffer_data is not None and len(buffer_data) == expected_size:
            offset = buf_array.find(buffer_data)
            if offset >= 0:
                return tflite_bytes, offset

    raise RuntimeError(f"Could not find weight buffer in Dense({n})")


def _patch_and_compile(tflite_bytes, w_offset, weights_int8, name, tmpdir):
    """Patch weights into TFLite, compile, return PC param blob."""
    patched = bytearray(tflite_bytes)
    raw = weights_int8.astype(np.int8).tobytes()
    patched[w_offset:w_offset + len(raw)] = raw

    model_path = os.path.join(tmpdir, f"{name}.tflite")
    with open(model_path, "wb") as f:
        f.write(bytes(patched))

    out_dir = os.path.join(tmpdir, f"out_{name}")
    os.makedirs(out_dir, exist_ok=True)

    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", out_dir, model_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compile failed: {result.stderr[:200]}")

    compiled_path = os.path.join(out_dir, f"{name}_edgetpu.tflite")
    with open(compiled_path, "rb") as f:
        compiled_data = f.read()

    model = parse_tflite(compiled_data)
    exes = parse_darwinn(model.custom_op_data)
    for exe in exes:
        if exe.exec_type == TYPE_PARAMETER_CACHING and exe.parameters:
            return bytes(exe.parameters)
    raise RuntimeError("No PC executable found")


def main():
    if shutil.which("edgetpu_compiler") is None:
        sys.exit("ERROR: edgetpu_compiler not found on PATH")

    parser = argparse.ArgumentParser(description="Exp3b: Validate blob generation")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    test_cases = [
        ("random_A", lambda n: rng.integers(-128, 128, (n, n), dtype=np.int8)),
        ("random_B", lambda n: rng.integers(-128, 128, (n, n), dtype=np.int8)),
        ("identity", lambda n: np.clip(np.eye(n) * 50, -128, 127).astype(np.int8)),
        ("checkerboard", lambda n: np.fromfunction(
            lambda i, j: ((i + j) % 2) * 127 - 64, (n, n)).astype(np.int8)),
        ("gradient", lambda n: np.fromfunction(
            lambda i, j: (i * n + j) % 256 - 128, (n, n)).astype(np.int8)),
    ]

    all_pass = True

    for n in args.sizes:
        print(f"\n{'='*60}")
        print(f"Dense({n})")
        print(f"{'='*60}")

        tflite_bytes, w_offset = _create_dense_tflite(n)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Get template blob (all-zero weights)
            template_blob = _patch_and_compile(
                tflite_bytes, w_offset,
                np.zeros((n, n), dtype=np.int8),
                f"template_{n}", tmpdir)

            print(f"  Template blob size: {len(template_blob)} bytes")
            expected = (n // ROW_TILE) * (ROW_TILE * 8 + ROW_TILE * n)
            # Handle case where N < ROW_TILE
            if n <= ROW_TILE:
                expected = ROW_TILE * 8 + ROW_TILE * n
            print(f"  Expected size: {expected} bytes  "
                  f"{'OK' if len(template_blob) == expected else 'MISMATCH!'}")

            for test_name, weight_fn in test_cases:
                weights = weight_fn(n)

                # Get compiler-generated reference blob
                ref_blob = _patch_and_compile(
                    tflite_bytes, w_offset, weights,
                    f"ref_{n}_{test_name}", tmpdir)

                # Generate predicted blob using our formula
                pred_blob = generate_blob_fast(template_blob, weights, n)

                # Compare
                match = ref_blob == pred_blob
                if match:
                    print(f"  {test_name:15s}: PERFECT MATCH")
                else:
                    all_pass = False
                    ref_arr = np.frombuffer(ref_blob, dtype=np.uint8)
                    pred_arr = np.frombuffer(pred_blob, dtype=np.uint8)
                    diff_idx = np.where(ref_arr != pred_arr)[0]
                    print(f"  {test_name:15s}: MISMATCH at {len(diff_idx)} bytes")
                    for di in diff_idx[:10]:
                        print(f"    blob[{di}]: ref=0x{ref_arr[di]:02x} "
                              f"pred=0x{pred_arr[di]:02x}")

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED - blob generation is compiler-free!")
    else:
        print("SOME TESTS FAILED - formula needs refinement")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
