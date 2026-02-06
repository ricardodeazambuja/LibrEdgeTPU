#!/usr/bin/env python3
"""MatMul engine example — Edge TPU matrix multiply benchmark.

Benchmarks matrix-vector multiplication y = W @ x on the Edge TPU.
No webcam needed — runs a tight benchmark loop and reports latency
statistics and optional accuracy verification against NumPy.

Requirements: Edge TPU USB accelerator
"""

import argparse
import time

import numpy as np

from libredgetpu import MatMulEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="MatMulEngine — matrix multiply benchmark")
    parser.add_argument("--dim", type=int, default=256,
                        choices=[256, 512, 1024],
                        help="Matrix dimension NxN (default: 256)")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of benchmark iterations (default: 1000)")
    parser.add_argument("--verify", action="store_true",
                        help="Compare results with NumPy and report max error")
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.dim
    iters = args.iterations

    print(f"MatMulEngine benchmark: {n}x{n}, {iters} iterations")

    # Generate random weights
    np.random.seed(42)
    W = np.random.randn(n, n).astype(np.float32) * 0.01

    with MatMulEngine.from_template(n) as engine:
        engine.set_weights(W)

        # Warmup
        x_warmup = np.random.randn(n).astype(np.float32) * 0.1
        for _ in range(10):
            engine.matmul(x_warmup)

        # Benchmark
        latencies = []
        max_error = 0.0

        for i in range(iters):
            x = np.random.randn(n).astype(np.float32) * 0.1

            t0 = time.perf_counter()
            y_hw = engine.matmul(x)
            latencies.append((time.perf_counter() - t0) * 1000)

            if args.verify:
                y_np = W @ x
                err = np.max(np.abs(y_hw - y_np))
                max_error = max(max_error, err)

            if (i + 1) % 100 == 0:
                med = np.median(latencies[-100:])
                print(f"  [{i+1}/{iters}] median: {med:.3f} ms", end="")
                if args.verify:
                    print(f"  max_err: {max_error:.4f}", end="")
                print()

        latencies = np.array(latencies)
        print(f"\nResults ({n}x{n}, {iters} iterations):")
        print(f"  Mean:   {latencies.mean():.3f} ms")
        print(f"  Median: {np.median(latencies):.3f} ms")
        print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
        print(f"  Min:    {latencies.min():.3f} ms")
        print(f"  Max:    {latencies.max():.3f} ms")
        throughput = 1000.0 / latencies.mean()
        print(f"  Throughput: {throughput:.0f} matmuls/sec")

        if args.verify:
            print(f"  Max quantization error: {max_error:.4f}")


if __name__ == "__main__":
    main()
