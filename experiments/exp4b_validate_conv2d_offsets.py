"""Experiment 4b: Validate Conv2D blob offset mapping.

Loads the offset tables discovered by exp4 Phase C, generates 5 weight
patterns per config, predicts blobs using offset-based patching, and
compares byte-for-byte against edgetpu_compiler output.

Target: 30/30 byte-perfect matches (6 configs x 5 patterns).

Usage:
    python -m experiments.exp4b_validate_conv2d_offsets
    python -m experiments.exp4b_validate_conv2d_offsets --configs 64:8:1 128:16:1
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libredgetpu._constants import SIGN_BIT_FLIP

# Import helpers from exp4
from experiments.exp4_conv2d_blob_format import (
    RESULTS_DIR, STANDARD_CONFIGS,
    _check_compiler, _config_name,
    _create_and_quantize, _find_conv_weight_buffer,
    _patch_conv_weights, _compile_and_extract_params,
    _load_result,
)


def validate_config(sh, sw, kh, kw, ch):
    """Validate offsets for one config using 5 weight patterns.

    Returns (n_passed, n_total) tuple.
    """
    name = _config_name(sh, sw, kh, kw, ch)
    n_weights = kh * kw * ch

    # Load Phase C offsets
    result_c = _load_result(name, "phase_c")
    if result_c is None or "offsets" not in result_c:
        print(f"  {name}: No Phase C offsets found. Run exp4 Phase C first.")
        return 0, 5

    offsets = result_c["offsets"]
    assert len(offsets) == n_weights, \
        f"Offset count mismatch: {len(offsets)} vs {n_weights}"

    # Create base model
    tflite = _create_and_quantize(sh, sw, kh, kw, ch)
    offset, buf, scale, zp = _find_conv_weight_buffer(tflite, kh, kw, ch)
    if offset is None:
        print(f"  {name}: Could not find Conv2D weight buffer")
        return 0, 5

    # Generate 5 test patterns
    rng = np.random.default_rng(99999)  # Different seed from exp4
    patterns = {
        "random_a": rng.integers(-128, 128, n_weights, dtype=np.int8),
        "random_b": rng.integers(-128, 128, n_weights, dtype=np.int8),
        "gradient_rev": np.linspace(127, -128, n_weights).astype(np.int8),
        "sparse_high": np.where(
            np.arange(n_weights) % 7 == 0, 120, 0
        ).astype(np.int8),
        "alternating": np.array(
            [(-1)**i * min(i + 1, 127) for i in range(n_weights)], dtype=np.int8
        ),
    }

    passed = 0
    total = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # Get zero blob
        int8_zero = np.zeros(n_weights, dtype=np.int8)
        patched_zero = _patch_conv_weights(tflite, offset, int8_zero)
        zero_blob = _compile_and_extract_params(patched_zero, "zero_val", tmpdir)
        if zero_blob is None:
            print(f"  {name}: Zero blob compilation failed")
            return 0, 5

        zero_arr = np.frombuffer(zero_blob, dtype=np.uint8).copy()

        for pat_name, int8_weights in patterns.items():
            total += 1

            # Compile with actual weights
            patched = _patch_conv_weights(tflite, offset, int8_weights)
            compiler_blob = _compile_and_extract_params(
                patched, f"v_{pat_name}", tmpdir
            )
            if compiler_blob is None:
                print(f"  {name}/{pat_name}: compilation failed")
                continue

            # Predict blob using offsets
            predicted = zero_arr.copy()
            for wi in range(n_weights):
                blob_off = offsets[wi]
                val = int(int8_weights[wi])
                predicted[blob_off] = (val ^ SIGN_BIT_FLIP) & 0xFF

            compiler_arr = np.frombuffer(compiler_blob, dtype=np.uint8)
            if np.array_equal(predicted, compiler_arr):
                print(f"  {name}/{pat_name}: PASS")
                passed += 1
            else:
                diffs = np.where(predicted != compiler_arr)[0]
                print(f"  {name}/{pat_name}: FAIL ({len(diffs)} byte mismatches)")

    return passed, total


def main():
    parser = argparse.ArgumentParser(
        description="Validate Conv2D blob offset mapping against compiler output",
    )
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="Configs as SH:KH:CH. Default: all 6 standard configs")
    args = parser.parse_args()

    _check_compiler()

    if args.configs:
        configs = []
        for s in args.configs:
            parts = s.split(":")
            sh, kh, ch = int(parts[0]), int(parts[1]), int(parts[2])
            configs.append((sh, sh, kh, kh, ch))
    else:
        configs = STANDARD_CONFIGS

    total_passed = 0
    total_tests = 0

    for sh, sw, kh, kw, ch in configs:
        name = _config_name(sh, sw, kh, kw, ch)
        print(f"\nValidating {name}...")
        passed, total = validate_config(sh, sw, kh, kw, ch)
        total_passed += passed
        total_tests += total
        print(f"  Result: {passed}/{total}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed}/{total_tests} byte-perfect matches")
    if total_passed == total_tests:
        print("ALL VALIDATIONS PASSED")
    else:
        print(f"FAILURES: {total_tests - total_passed}")


if __name__ == "__main__":
    main()
