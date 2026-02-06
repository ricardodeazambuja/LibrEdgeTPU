"""Benchmark libredgetpu vs tflite_runtime inference speed on Edge TPU.

Standalone script (not a pytest test) — requires hardware.
tflite_runtime is optional; if missing, only libredgetpu is benchmarked.

IMPORTANT: tflite_runtime (libedgetpu) and libredgetpu use incompatible USB
firmware. tflite_runtime runs FIRST (libedgetpu loads its own firmware from
bootloader mode). Then the user must replug the device so libredgetpu can load
its single-endpoint firmware. Use --libredgetpu-only or --tflite-only to skip
the replug step.

NOTE: The tflite_runtime backend requires ABI-compatible versions of
tflite_runtime and libedgetpu. Google's official libedgetpu 16.0 Debian
package (2021) is incompatible with tflite_runtime >= 2.6 on Python >= 3.10.
Use feranick's rebuilt libedgetpu from github.com/feranick/libedgetpu
(matched to your tflite_runtime version) and pass --delegate /path/to/libedgetpu.so.1.0

Usage:
    python -m tests.benchmark_vs_tflite
    python -m tests.benchmark_vs_tflite --models mobilenet_v1 mobilenet_v2
    python -m tests.benchmark_vs_tflite --warmup 5 --iterations 50
    python -m tests.benchmark_vs_tflite --json results.json
    python -m tests.benchmark_vs_tflite --libredgetpu-only
    python -m tests.benchmark_vs_tflite --tflite-only
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from tests.model_zoo import MODELS, get_model
from libredgetpu.simple_invoker import SimpleInvoker
from libredgetpu.tflite_parser import parse as parse_tflite

# Graceful tflite_runtime detection
_HAS_TFLITE_RUNTIME = False
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    _HAS_TFLITE_RUNTIME = True
except ImportError:
    pass


def _get_input_size(model_path):
    """Return the raw input size in bytes from a compiled TFLite model."""
    with open(model_path, "rb") as f:
        model = parse_tflite(f.read())
    return int(np.prod(model.input_tensor.shape))


def _stats(latencies):
    """Compute summary statistics from a list of latency values (ms)."""
    arr = sorted(latencies)
    n = len(arr)
    return {
        "avg": sum(arr) / n,
        "std": (sum((x - sum(arr) / n) ** 2 for x in arr) / n) ** 0.5,
        "min": arr[0],
        "max": arr[-1],
        "p50": arr[n // 2],
        "p95": arr[int(n * 0.95)],
    }


def run_libredgetpu(model_path, input_bytes, n_warmup, n_iter):
    """Benchmark libredgetpu backend. Returns list of latencies in ms."""
    with SimpleInvoker(model_path) as model:
        # Warmup
        for _ in range(n_warmup):
            model.invoke_raw(input_bytes)

        # Timed iterations
        latencies = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            model.invoke_raw(input_bytes)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return latencies


def run_tflite_runtime(model_path, input_bytes, n_warmup, n_iter,
                       delegate_path=None):
    """Benchmark tflite_runtime backend. Returns list of latencies in ms."""
    # Try loading the Edge TPU delegate
    delegate = None
    candidates = ([delegate_path] if delegate_path
                  else ["libedgetpu.so.1", "libedgetpu.1.dylib"])
    for lib in candidates:
        try:
            delegate = load_delegate(lib)
            break
        except (ValueError, OSError):
            continue
    if delegate is None:
        raise RuntimeError(
            f"Could not load Edge TPU delegate. Tried: {candidates}"
        )

    interp = Interpreter(
        model_path=model_path,
        experimental_delegates=[delegate],
    )
    interp.allocate_tensors()

    input_detail = interp.get_input_details()[0]
    input_shape = input_detail["shape"]
    input_array = np.frombuffer(input_bytes, dtype=np.uint8)[
        : int(np.prod(input_shape))
    ].reshape(input_shape)

    # Warmup
    for _ in range(n_warmup):
        interp.set_tensor(input_detail["index"], input_array)
        interp.invoke()

    # Timed iterations
    latencies = []
    for _ in range(n_iter):
        interp.set_tensor(input_detail["index"], input_array)
        t0 = time.perf_counter()
        interp.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return latencies


def _fmt_stats(s):
    """Format avg +/- std as a short string."""
    return f"{s['avg']:6.1f} +/- {s['std']:<4.1f}"


def _print_table(results, show_tflite):
    """Print a formatted results table."""
    if show_tflite:
        hdr = (
            f"{'Model':<24s} | {'libredgetpu (ms)':>18s} | "
            f"{'tflite_rt (ms)':>18s} | {'Ratio':>6s}"
        )
        sep = "-" * 24 + "-|-" + "-" * 18 + "-|-" + "-" * 18 + "-|-" + "-" * 6
    else:
        hdr = f"{'Model':<24s} | {'libredgetpu (ms)':>18s}"
        sep = "-" * 24 + "-|-" + "-" * 18

    print()
    print(hdr)
    print(sep)

    for name, r in results.items():
        has_py = "libredgetpu" in r
        has_tf = "tflite_runtime" in r
        py_s = _fmt_stats(r["libredgetpu"]) if has_py else "      —         "
        if show_tflite:
            tf_s = _fmt_stats(r["tflite_runtime"]) if has_tf else "      —         "
            if has_py and has_tf:
                ratio = r["libredgetpu"]["avg"] / r["tflite_runtime"]["avg"]
                print(f"{name:<24s} | {py_s:>18s} | {tf_s:>18s} | {ratio:5.1f}x")
            else:
                print(f"{name:<24s} | {py_s:>18s} | {tf_s:>18s} |      —")
        else:
            print(f"{name:<24s} | {py_s:>18s}")

    print()


def _wait_for_replug():
    """Wait for user to replug the Edge TPU device."""
    print()
    print("=" * 60)
    print("  Please REPLUG the Edge TPU USB device now.")
    print("  (libredgetpu uses different firmware than libedgetpu)")
    print("=" * 60)
    input("  Press Enter after replugging... ")
    # Give the device a moment to enumerate
    time.sleep(2.0)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark libredgetpu vs tflite_runtime inference speed"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to benchmark (default: all in model_zoo)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iterations", type=int, default=30, help="Timed iterations (default: 30)"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Save results to JSON file"
    )
    parser.add_argument(
        "--libredgetpu-only",
        action="store_true",
        help="Only benchmark libredgetpu (skip tflite_runtime)",
    )
    parser.add_argument(
        "--tflite-only",
        action="store_true",
        help="Only benchmark tflite_runtime (skip libredgetpu)",
    )
    parser.add_argument(
        "--delegate",
        type=str,
        default=None,
        help="Path to libedgetpu shared library (default: system libedgetpu.so.1)",
    )
    args = parser.parse_args()

    model_names = args.models if args.models else sorted(MODELS.keys())

    run_tflite = (
        _HAS_TFLITE_RUNTIME and not args.libredgetpu_only
    )
    run_py = not args.tflite_only

    if args.tflite_only and not _HAS_TFLITE_RUNTIME:
        print("ERROR: --tflite-only requires tflite_runtime to be installed.")
        print("       Install with: pip install tflite-runtime")
        sys.exit(1)

    if not _HAS_TFLITE_RUNTIME and not args.libredgetpu_only:
        print("NOTE: tflite_runtime not installed — benchmarking libredgetpu only.")
        print("      Install with: pip install tflite-runtime")
        print()

    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"Models: {', '.join(model_names)}")
    backends = []
    if run_tflite:
        backends.append("tflite_runtime")
    if run_py:
        backends.append("libredgetpu")
    print(f"Backends: {', '.join(backends)}")
    print()

    # Download all models first
    model_paths = {}
    for name in model_names:
        try:
            model_paths[name] = get_model(name)
        except Exception as e:
            print(f"WARNING: skipping {name}: {e}")

    results = {name: {} for name in model_names if name in model_paths}

    # --- Phase 1: tflite_runtime (runs FIRST — needs bootloader/libedgetpu firmware) ---
    if run_tflite:
        print("=== tflite_runtime (libedgetpu) ===")
        for name in model_names:
            if name not in model_paths:
                continue
            path = model_paths[name]
            input_size = _get_input_size(path)
            input_bytes = np.random.randint(
                0, 256, size=input_size, dtype=np.uint8
            ).tobytes()

            print(f"  {name}...", end="", flush=True)
            try:
                latencies = run_tflite_runtime(
                    path, input_bytes, args.warmup, args.iterations,
                    delegate_path=args.delegate,
                )
                s = _stats(latencies)
                results[name]["tflite_runtime"] = s
                print(f" {s['avg']:.1f} +/- {s['std']:.1f} ms")
            except Exception as e:
                print(f" FAILED: {e}")

    # --- Replug between backends ---
    if run_tflite and run_py:
        _wait_for_replug()

    # --- Phase 2: libredgetpu (loads single-endpoint firmware) ---
    if run_py:
        print("=== libredgetpu ===")
        for name in model_names:
            if name not in model_paths:
                continue
            path = model_paths[name]
            input_size = _get_input_size(path)
            input_bytes = np.random.randint(
                0, 256, size=input_size, dtype=np.uint8
            ).tobytes()

            print(f"  {name}...", end="", flush=True)
            try:
                latencies = run_libredgetpu(
                    path, input_bytes, args.warmup, args.iterations
                )
                s = _stats(latencies)
                results[name]["libredgetpu"] = s
                print(f" {s['avg']:.1f} +/- {s['std']:.1f} ms")
            except Exception as e:
                print(f" FAILED: {e}")

    # --- Results ---
    show_tflite = run_tflite and any("tflite_runtime" in r for r in results.values())
    _print_table(results, show_tflite)

    # Optional JSON output
    if args.json and results:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
