#!/usr/bin/env python3
"""Reservoir computer example — echo state network with webcam sensory input.

Feeds real-time webcam features (mean R, G, B, brightness) into an
Edge TPU-accelerated echo state network.  Visualizes the reservoir state
as a 16x16 heatmap overlaid on the camera feed.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time

import cv2
import numpy as np

from libredgetpu import ReservoirComputer
from _common import add_common_args, WebcamLoop, draw_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReservoirComputer — echo state network")
    add_common_args(parser)
    parser.add_argument("--dim", type=int, default=256,
                        choices=[256, 512, 1024],
                        help="Reservoir state dimension (default: 256)")
    parser.add_argument("--input-dim", type=int, default=4,
                        help="Input signal dimension (default: 4)")
    parser.add_argument("--spectral-radius", type=float, default=0.95,
                        help="Reservoir spectral radius (default: 0.95)")
    parser.add_argument("--leak-rate", type=float, default=1.0,
                        help="Leaky integrator rate (default: 1.0)")
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["tanh", "relu", "identity"],
                        help="Activation function (default: tanh)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--input-scaling", type=float, default=0.1,
                        help="Input weight scaling (default: 0.1)")
    return parser.parse_args()


def frame_to_features(frame, input_dim):
    """Extract simple features from webcam frame.

    Returns [input_dim] float32 features:
    - mean R, G, B (if input_dim >= 3)
    - mean brightness (if input_dim >= 4)
    - additional spatial stats for higher dims
    """
    features = np.zeros(input_dim, dtype=np.float32)
    h, w = frame.shape[:2]

    # Mean color channels (normalized to [-1, 1])
    means = frame.mean(axis=(0, 1)) / 127.5 - 1.0
    n = min(3, input_dim)
    features[:n] = means[:n]

    # Mean brightness
    if input_dim >= 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features[3] = gray.mean() / 127.5 - 1.0

    # Spatial variance for higher dimensions
    if input_dim > 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Divide into quadrants
        quads = [
            gray[:h//2, :w//2], gray[:h//2, w//2:],
            gray[h//2:, :w//2], gray[h//2:, w//2:],
        ]
        for i, q in enumerate(quads):
            if 4 + i < input_dim:
                features[4 + i] = q.std() / 128.0

    return features


def draw_state_heatmap(frame, state, alpha=0.4):
    """Draw reservoir state as 16x16 heatmap overlay."""
    grid = state[:256].reshape(16, 16)
    vmin, vmax = grid.min(), grid.max()
    vrange = vmax - vmin + 1e-8
    normalized = ((grid - vmin) / vrange * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    h, w = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h),
                                 interpolation=cv2.INTER_NEAREST)
    cv2.addWeighted(heatmap_resized, alpha, frame, 1 - alpha, 0, frame)


def main():
    args = parse_args()
    loop = WebcamLoop(args)

    with ReservoirComputer.from_template(
        args.dim,
        args.input_dim,
        spectral_radius=args.spectral_radius,
        input_scaling=args.input_scaling,
        leak_rate=args.leak_rate,
        activation=args.activation,
        seed=args.seed,
    ) as rc:
        for frame in loop:
            u = frame_to_features(frame, args.input_dim)

            t0 = time.perf_counter()
            state = rc.step(u)
            latency_ms = (time.perf_counter() - t0) * 1000

            norm = float(np.linalg.norm(state))

            draw_state_heatmap(frame, state)

            h = frame.shape[0]
            draw_text(frame, f"State norm: {norm:.3f} | dim={args.dim}",
                      (10, 30))
            draw_text(frame,
                      f"Input: [{', '.join(f'{v:.2f}' for v in u[:4])}]",
                      (10, 55))
            draw_text(frame, f"{latency_ms:.1f} ms", (10, 80))

            loop.show(frame)
            loop.print_metrics({"norm": norm}, latency_ms)

    loop.cleanup()


if __name__ == "__main__":
    main()
