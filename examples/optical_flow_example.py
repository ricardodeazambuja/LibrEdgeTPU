#!/usr/bin/env python3
"""Optical flow example — global ego-motion estimation via Gabor features.

Computes a single (vx, vy) displacement vector between consecutive frames
using Edge TPU-accelerated Gabor feature extraction and CPU-side correlation.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time

import cv2
import numpy as np

from libredgetpu import OpticalFlow
from _common import add_common_args, WebcamLoop, resize_gray, draw_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpticalFlow — ego-motion estimation")
    add_common_args(parser)
    parser.add_argument("--image-size", type=int, default=64, choices=[64, 128],
                        help="Input image resolution (default: 64)")
    parser.add_argument("--pooled", action="store_true",
                        help="Use fused Gabor+Pool model (16x less USB)")
    parser.add_argument("--pool-factor", type=int, default=2,
                        choices=[1, 2, 4, 8],
                        help="Spatial pooling factor (default: 2)")
    parser.add_argument("--search-range", type=int, default=4,
                        help="Max displacement to search in pooled pixels (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Soft argmax temperature (default: 0.1)")
    return parser.parse_args()


def flow_to_direction(vx, vy, threshold=0.3):
    """Classify flow vector into a direction."""
    if abs(vx) < threshold and abs(vy) < threshold:
        return "still"
    if abs(vx) > abs(vy):
        return "right" if vx > 0 else "left"
    return "down" if vy > 0 else "up"


def main():
    args = parse_args()
    loop = WebcamLoop(args)

    prev_gray = None

    with OpticalFlow.from_template(
        args.image_size,
        pooled=args.pooled,
        pool_factor=args.pool_factor,
        search_range=args.search_range,
        temperature=args.temperature,
    ) as flow:
        for frame in loop:
            gray = resize_gray(frame, args.image_size)

            if prev_gray is None:
                prev_gray = gray.copy()
                draw_text(frame, "Initializing...", (10, 30))
                loop.show(frame)
                continue

            t0 = time.perf_counter()
            vx, vy = flow.compute(prev_gray, gray)
            latency_ms = (time.perf_counter() - t0) * 1000

            magnitude = np.sqrt(vx**2 + vy**2)
            direction = flow_to_direction(vx, vy)

            # Draw flow arrow from center
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            scale = 10.0
            ex = int(cx + vx * scale)
            ey = int(cy + vy * scale)
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 0, 255), 2,
                            tipLength=0.3)

            draw_text(frame, f"Flow: ({vx:.2f}, {vy:.2f}) mag={magnitude:.2f}",
                      (10, 30))
            draw_text(frame, f"Dir: {direction} | {latency_ms:.1f} ms",
                      (10, 55))

            loop.show(frame)
            loop.print_metrics({"vx": vx, "vy": vy, "dir": direction},
                               latency_ms)

            prev_gray = gray.copy()

    loop.cleanup()


if __name__ == "__main__":
    main()
