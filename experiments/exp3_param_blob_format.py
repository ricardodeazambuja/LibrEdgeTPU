"""Experiment 3: Determine DarwiNN parameter blob format.

Goal: Determine the mapping from int8 weight matrix to the parameter blob
that edgetpu_compiler produces. If we can predict the blob from weights alone,
we eliminate the x86-only compiler dependency entirely.

Phase 1 tests:
  A) All-constant weights — checks if values are stored unchanged (no transform)
  B) All-zero weights — reveals overhead structure (non-zero bytes = metadata)
  C) Overhead scaling — verifies N*8 overhead pattern across sizes
  D) Single-probe mapping — one non-zero weight at (i,j), find it in the blob

Produces a single JSON report at the end.

Usage:
    python -m experiments.exp3_param_blob_format
    python -m experiments.exp3_param_blob_format --sizes 64 128 256
    python -m experiments.exp3_param_blob_format --phase D --sizes 256
"""

import argparse
import json
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


def _check_compiler():
    if shutil.which("edgetpu_compiler") is None:
        sys.exit("ERROR: edgetpu_compiler not found on PATH")


def _create_dense_tflite(n):
    """Create a quantized Dense(N,N) TFLite with uniform(-1,1) representative data.
    Returns (tflite_bytes, weight_buffer_offset, weight_tensor_int8)."""
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

    # Find weight buffer offset in the raw flatbuffer bytes
    full = parse_full(tflite_bytes)
    expected_size = n * n
    buf_array = bytearray(tflite_bytes)
    weight_offset = None
    for buffer_data in full.buffers:
        if buffer_data is not None and len(buffer_data) == expected_size:
            offset = buf_array.find(buffer_data)
            if offset >= 0:
                weight_offset = offset
                break

    if weight_offset is None:
        raise RuntimeError(f"Could not find weight buffer in Dense({n}) TFLite")

    return tflite_bytes, weight_offset


def _patch_weights(tflite_bytes, weight_offset, new_int8_weights):
    """Replace weight bytes in TFLite. new_int8_weights is a flat int8 array."""
    patched = bytearray(tflite_bytes)
    raw = new_int8_weights.astype(np.int8).tobytes()
    patched[weight_offset:weight_offset + len(raw)] = raw
    return bytes(patched)


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
        print(f"  COMPILE FAILED: {result.stderr[:200]}", file=sys.stderr)
        return None

    compiled_path = os.path.join(out_dir, f"{name}_edgetpu.tflite")
    if not os.path.isfile(compiled_path):
        return None

    with open(compiled_path, "rb") as f:
        compiled_data = f.read()

    model = parse_tflite(compiled_data)
    exes = parse_darwinn(model.custom_op_data)
    for exe in exes:
        if exe.exec_type == TYPE_PARAMETER_CACHING and exe.parameters:
            return bytes(exe.parameters)
    return None


# ── Phase A: All-constant weights ──────────────────────────────────────────

def phase_a(sizes, tmpdir):
    """Test if constant int8 weight values appear unchanged in the param blob."""
    print("\n=== Phase A: Value transformation test ===")
    test_values = [0, 1, 42, 127, -1, -128]
    results = {}

    for n in sizes:
        print(f"\n  Dense({n}):")
        tflite_bytes, w_offset = _create_dense_tflite(n)
        size_results = {}

        for val in test_values:
            weights = np.full(n * n, val, dtype=np.int8)
            patched = _patch_weights(tflite_bytes, w_offset, weights)
            blob = _compile_and_extract_params(patched, f"const_{n}_{val}", tmpdir)
            if blob is None:
                size_results[str(val)] = {"error": "compile failed"}
                continue

            blob_arr = np.frombuffer(blob, dtype=np.uint8)
            val_u8 = np.uint8(val)  # how val looks as unsigned byte

            count_exact = int(np.sum(blob_arr == val_u8))
            count_xor80 = int(np.sum(blob_arr == (val_u8 ^ 0x80)))
            count_neg = int(np.sum(blob_arr == np.uint8(-val)))
            unique_vals = int(np.unique(blob_arr).size)

            size_results[str(val)] = {
                "blob_size": len(blob),
                "expected_weight_bytes": n * n,
                "overhead": len(blob) - n * n,
                "count_exact_match": count_exact,
                "count_xor_0x80": count_xor80,
                "count_negated": count_neg,
                "unique_byte_values": unique_vals,
            }

            print(f"    val={val:4d} (0x{val_u8:02x}): blob={len(blob)}, "
                  f"exact={count_exact}/{n*n}, "
                  f"xor80={count_xor80}, neg={count_neg}, "
                  f"unique={unique_vals}")

        results[str(n)] = size_results
    return results


# ── Phase B: All-zero weights (reveal overhead) ───────────────────────────

def phase_b(sizes, tmpdir):
    """With all-zero weights, non-zero bytes in the blob are overhead/metadata."""
    print("\n=== Phase B: Overhead structure (zero weights) ===")
    results = {}

    for n in sizes:
        tflite_bytes, w_offset = _create_dense_tflite(n)
        weights = np.zeros(n * n, dtype=np.int8)
        patched = _patch_weights(tflite_bytes, w_offset, weights)
        blob = _compile_and_extract_params(patched, f"zero_{n}", tmpdir)
        if blob is None:
            results[str(n)] = {"error": "compile failed"}
            continue

        blob_arr = np.frombuffer(blob, dtype=np.uint8)
        nonzero_indices = np.nonzero(blob_arr)[0]
        zero_count = int(np.sum(blob_arr == 0))
        nonzero_count = len(blob_arr) - zero_count

        # Find overhead byte positions and group into contiguous runs
        overhead_runs = []
        if len(nonzero_indices) > 0:
            run_start = int(nonzero_indices[0])
            run_end = run_start
            for idx in nonzero_indices[1:]:
                if idx == run_end + 1:
                    run_end = int(idx)
                else:
                    overhead_runs.append({
                        "start": run_start,
                        "end": run_end + 1,
                        "length": run_end - run_start + 1,
                        "hex": blob[run_start:run_end + 1].hex(),
                    })
                    run_start = int(idx)
                    run_end = run_start
            overhead_runs.append({
                "start": run_start,
                "end": run_end + 1,
                "length": run_end - run_start + 1,
                "hex": blob[run_start:run_end + 1].hex(),
            })

        # Also try interpreting overhead as int32 arrays
        overhead_as_int32 = []
        for run in overhead_runs:
            chunk = blob[run["start"]:run["end"]]
            if len(chunk) % 4 == 0:
                vals = np.frombuffer(chunk, dtype=np.int32)
                overhead_as_int32.append({
                    "offset": run["start"],
                    "int32_values": vals.tolist()[:32],  # cap for readability
                    "unique_int32": int(np.unique(vals).size),
                })

        results[str(n)] = {
            "blob_size": len(blob),
            "weight_bytes": n * n,
            "overhead": len(blob) - n * n,
            "overhead_ratio": f"N*{(len(blob) - n * n) / n:.1f}" if n > 0 else "?",
            "zero_bytes": zero_count,
            "nonzero_bytes": nonzero_count,
            "nonzero_runs": overhead_runs[:20],  # cap for readability
            "overhead_as_int32": overhead_as_int32[:10],
        }

        print(f"  Dense({n}): blob={len(blob)}, overhead={len(blob) - n*n} "
              f"(N*{(len(blob) - n*n)/n:.1f}), "
              f"nonzero_bytes={nonzero_count}, runs={len(overhead_runs)}")
        for run in overhead_runs[:5]:
            print(f"    [{run['start']:6d}:{run['end']:6d}] "
                  f"len={run['length']:4d}  {run['hex'][:80]}...")

    return results


# ── Phase C: Overhead scaling ──────────────────────────────────────────────

def phase_c(sizes, tmpdir):
    """Verify that overhead = N * constant across sizes."""
    print("\n=== Phase C: Overhead scaling ===")
    results = {}

    for n in sizes:
        tflite_bytes, w_offset = _create_dense_tflite(n)
        # Use a known constant (42) so we can also count weight bytes
        weights = np.full(n * n, 42, dtype=np.int8)
        patched = _patch_weights(tflite_bytes, w_offset, weights)
        blob = _compile_and_extract_params(patched, f"scale_{n}", tmpdir)
        if blob is None:
            results[str(n)] = {"error": "compile failed"}
            continue

        overhead = len(blob) - n * n
        per_row = overhead / n if n > 0 else 0

        results[str(n)] = {
            "blob_size": len(blob),
            "weight_bytes": n * n,
            "overhead": overhead,
            "overhead_per_row": per_row,
        }
        print(f"  Dense({n:4d}): blob={len(blob):8d}, "
              f"weights={n*n:8d}, overhead={overhead:6d} (N*{per_row:.1f})")

    return results


# ── Phase D: Single-probe mapping ──────────────────────────────────────────

def phase_d(sizes, tmpdir):
    """Place a single non-zero weight at (row, col), find it in the blob via diff.
    Phase A proved values are XOR 0x80, so baseline(0) = 0x80, probe(42) = 0xAA.
    We use byte-diff from baseline to locate the probe position."""
    print("\n=== Phase D: Single-probe weight mapping ===")
    PROBE_VAL = 42  # int8 value; appears as 42^0x80 = 0xAA in blob
    results = {}

    for n in sizes:
        print(f"\n  Dense({n}):")
        tflite_bytes, w_offset = _create_dense_tflite(n)

        # Baseline: all-zero weights (blob will have 0x80 at weight positions)
        zero_weights = np.zeros(n * n, dtype=np.int8)
        patched_zero = _patch_weights(tflite_bytes, w_offset, zero_weights)
        baseline_blob = _compile_and_extract_params(
            patched_zero, f"probe_baseline_{n}", tmpdir)
        if baseline_blob is None:
            results[str(n)] = {"error": "baseline compile failed"}
            continue

        baseline_arr = np.frombuffer(baseline_blob, dtype=np.uint8)

        # Probe positions: corners, tile boundaries (multiples of 64), center
        boundary_vals = sorted(set([0, 1, 63, 64, 65, n // 2, n - 2, n - 1])
                               & set(range(n)))
        probe_positions = []
        for r in boundary_vals:
            for c in boundary_vals:
                probe_positions.append((r, c))

        mapping = []  # list of {row, col, blob_offset, ...}

        for idx, (row, col) in enumerate(probe_positions):
            weights = np.zeros((n, n), dtype=np.int8)
            weights[row, col] = PROBE_VAL
            patched = _patch_weights(tflite_bytes, w_offset, weights.flatten())
            blob = _compile_and_extract_params(
                patched, f"probe_{n}_{row}_{col}", tmpdir)
            if blob is None:
                mapping.append({"row": row, "col": col, "error": "compile failed"})
                continue

            blob_arr = np.frombuffer(blob, dtype=np.uint8)

            # Find ALL positions where blob differs from baseline
            diff_indices = np.where(blob_arr != baseline_arr)[0]
            diff_details = []
            for di in diff_indices:
                diff_details.append({
                    "pos": int(di),
                    "baseline": int(baseline_arr[di]),
                    "probe": int(blob_arr[di]),
                })

            # The weight byte should be the one that changed from 0x80 to 0xAA
            weight_diffs = [d for d in diff_details
                            if d["baseline"] == 0x80
                            and d["probe"] == (PROBE_VAL & 0xFF) ^ 0x80]
            # Other diffs are overhead/metadata changes
            meta_diffs = [d for d in diff_details if d not in weight_diffs]

            blob_offset = weight_diffs[0]["pos"] if len(weight_diffs) == 1 else None

            entry = {
                "row": row,
                "col": col,
                "weight_linear_idx": row * n + col,
                "blob_offset": blob_offset,
                "diff_count": len(diff_details),
                "weight_diffs": weight_diffs,
                "meta_diffs": meta_diffs,
            }
            mapping.append(entry)

            status = "OK" if blob_offset is not None else f"AMBIG({len(diff_details)})"
            if idx < 10 or status != "OK":
                meta_str = ""
                if meta_diffs:
                    meta_str = f"  meta: {meta_diffs}"
                print(f"    [{row:3d},{col:3d}] -> blob[{blob_offset}]  {status}{meta_str}")

        # Extract clean mapping pairs: (row, col) -> blob_offset
        clean = [(m["row"], m["col"], m["blob_offset"])
                 for m in mapping if m.get("blob_offset") is not None]

        formula_checks = {}
        if clean:
            rows, cols, offsets = zip(*clean)
            rows, cols, offsets = np.array(rows), np.array(cols), np.array(offsets)

            # Test common reordering hypotheses
            RT = 64   # row tile size (MAC array dim)
            CT = 4    # col tile size (wide memory bus width?)
            def _grouped_tile(r, c):
                """64-row groups, each with [overhead][4-col-tiled weights]."""
                rg = r // RT
                rl = r % RT
                group_size = RT * 8 + RT * n
                group_start = rg * group_size
                weight_start = group_start + RT * 8
                cb = c // CT
                cl = c % CT
                return weight_start + cb * (RT * CT) + rl * CT + cl

            formulas = [
                ("grouped_tile64x4", _grouped_tile),
                ("row_major", lambda r, c: r * n + c),
                ("col_major", lambda r, c: c * n + r),
                # Flat 4-column tiling (no row groups) — only works for N<=64
                ("tile4c_flat", lambda r, c: n * 8 + (c // 4) * (n * 4) + r * 4 + (c % 4)),
            ]

            for name, formula in formulas:
                try:
                    predicted = np.array([formula(r, c) for r, c in zip(rows, cols)])
                    matches = int(np.sum(predicted == offsets))
                except Exception:
                    matches = 0
                formula_checks[name] = {
                    "matches": matches,
                    "total": len(clean),
                    "perfect": matches == len(clean),
                }
                if matches > 0:
                    print(f"    Formula '{name}': {matches}/{len(clean)} matches"
                          f"{' *** PERFECT ***' if matches == len(clean) else ''}")

            # If no formula is perfect, dump raw mapping for analysis
            if not any(v["perfect"] for v in formula_checks.values()):
                print(f"    No perfect formula. Raw mapping ({len(clean)} probes):")
                for r, c, off in clean[:30]:
                    print(f"      w[{r:3d},{c:3d}] (linear={r*n+c:6d}) -> blob[{off:6d}]"
                          f"  delta={off - (r*n+c):+d}")
                # Try to detect pattern in deltas
                if len(clean) >= 2:
                    deltas = [off - (r * n + c) for r, c, off in clean]
                    print(f"    Deltas from row_major: {deltas[:20]}")

        results[str(n)] = {
            "probe_value": PROBE_VAL,
            "baseline_blob_size": len(baseline_blob),
            "num_probes": len(probe_positions),
            "clean_probes": len(clean),
            "formula_checks": formula_checks,
            "mapping": mapping,
        }

    return results


def main():
    _check_compiler()

    parser = argparse.ArgumentParser(description="Exp3: Parameter blob format")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256],
                        help="Matrix sizes to test (default: 64 128 256)")
    parser.add_argument("--phase", type=str, default="ABCD",
                        help="Which phases to run (default: ABCD)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: exp3_results.json in experiments/)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "exp3_results.json")

    report = {"sizes": args.sizes, "phases": {}}

    with tempfile.TemporaryDirectory() as tmpdir:
        if "A" in args.phase.upper():
            report["phases"]["A_value_transform"] = phase_a(args.sizes, tmpdir)

        if "B" in args.phase.upper():
            report["phases"]["B_overhead_structure"] = phase_b(args.sizes, tmpdir)

        if "C" in args.phase.upper():
            report["phases"]["C_overhead_scaling"] = phase_c(args.sizes, tmpdir)

        if "D" in args.phase.upper():
            report["phases"]["D_probe_mapping"] = phase_d(args.sizes, tmpdir)

    # Write JSON report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "A_value_transform" in report["phases"]:
        print("\nPhase A (value transform):")
        for n_str, vals in report["phases"]["A_value_transform"].items():
            for v_str, info in vals.items():
                if "error" in info:
                    continue
                n = int(n_str)
                exact = info["count_exact_match"]
                expected = n * n
                pct = exact / expected * 100 if expected else 0
                xor80 = info["count_xor_0x80"]
                xpct = xor80 / expected * 100 if expected else 0
                if pct > 95:
                    print(f"  Dense({n}) val={v_str}: values UNCHANGED ({pct:.0f}% exact)")
                elif xpct > 95:
                    print(f"  Dense({n}) val={v_str}: values XOR 0x80 ({xpct:.0f}%)")
                else:
                    print(f"  Dense({n}) val={v_str}: TRANSFORMED "
                          f"(exact={pct:.0f}%, xor80={xpct:.0f}%)")

    if "C_overhead_scaling" in report["phases"]:
        print("\nPhase C (overhead scaling):")
        for n_str, info in report["phases"]["C_overhead_scaling"].items():
            if "error" in info:
                continue
            print(f"  Dense({n_str}): overhead = {info['overhead']} = "
                  f"N * {info['overhead_per_row']:.1f}")

    if "D_probe_mapping" in report["phases"]:
        print("\nPhase D (probe mapping):")
        for n_str, info in report["phases"]["D_probe_mapping"].items():
            if "error" in info:
                continue
            for fname, finfo in info.get("formula_checks", {}).items():
                if finfo["perfect"]:
                    print(f"  Dense({n_str}): FORMULA = {fname} "
                          f"({finfo['matches']}/{finfo['total']})")


if __name__ == "__main__":
    main()
