#!/usr/bin/env python3
"""Looming detector example — collision avoidance with time-to-contact.

Detects approaching objects via edge density in 3x3 spatial zones.
Computes tau (center/periphery ratio) and TTC (time-to-contact) using
a sliding window of tau values.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from libredgetpu import LoomingDetector
from _common import add_common_args, WebcamLoop, resize_gray, draw_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoomingDetector — collision avoidance + TTC")
    add_common_args(parser)
    parser.add_argument("--image-size", type=int, default=64, choices=[64, 128],
                        help="Input image resolution (default: 64)")
    parser.add_argument("--tau-threshold", type=float, default=1.2,
                        help="Tau warning threshold (default: 1.2)")
    parser.add_argument("--ttc-window", type=int, default=10,
                        help="Frames for TTC estimation (default: 10)")
    return parser.parse_args()


def draw_heatmap_grid(frame, densities):
    """Draw 3x3 heatmap overlay on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    vmin, vmax = densities.min(), densities.max()
    vrange = vmax - vmin + 1e-8

    for i in range(3):
        for j in range(3):
            val = (densities[i, j] - vmin) / vrange
            # Blue -> Green -> Red colormap
            if val < 0.5:
                b = int(255 * (1 - val * 2))
                g = int(255 * val * 2)
                r = 0
            else:
                b = 0
                g = int(255 * (1 - (val - 0.5) * 2))
                r = int(255 * (val - 0.5) * 2)

            y1, y2 = i * h // 3, (i + 1) * h // 3
            x1, x2 = j * w // 3, (j + 1) * w // 3
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (b, g, r), -1)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


def main():
    args = parse_args()
    loop = WebcamLoop(args)

    tau_history = deque(maxlen=args.ttc_window)
    last_time = time.perf_counter()

    with LoomingDetector.from_template(args.image_size) as detector:
        for frame in loop:
            gray = resize_gray(frame, args.image_size)

            t0 = time.perf_counter()
            zones = detector.detect(gray)
            latency_ms = (time.perf_counter() - t0) * 1000

            densities = zones.reshape(3, 3)
            tau = LoomingDetector.compute_tau(zones)

            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            tau_history.append(tau)

            ttc = LoomingDetector.compute_ttc(list(tau_history), dt)

            # Draw heatmap
            draw_heatmap_grid(frame, densities)

            # Zone values
            h, w = frame.shape[:2]
            for i in range(3):
                for j in range(3):
                    cx = j * w // 3 + w // 6
                    cy = i * h // 3 + h // 6
                    draw_text(frame, f"{densities[i, j]:.2f}", (cx - 20, cy),
                              scale=0.4)

            # Status line
            warning = "COLLISION!" if tau > args.tau_threshold else ""
            color = (0, 0, 255) if warning else (255, 255, 255)
            draw_text(frame, f"Tau: {tau:.3f}  TTC: {ttc:.2f}s  {warning}",
                      (10, h - 20), color=color)

            loop.show(frame)
            loop.print_metrics({"tau": tau, "ttc": ttc}, latency_ms)

    loop.cleanup()


if __name__ == "__main__":
    main()
