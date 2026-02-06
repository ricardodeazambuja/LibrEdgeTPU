#!/usr/bin/env python3
"""Visual compass example — yaw/heading estimation from optical flow.

Wraps OpticalFlow to convert horizontal displacement into yaw angle (degrees).
Accumulates yaw over time for dead-reckoning heading estimation.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time

import cv2
import numpy as np

from libredgetpu import VisualCompass
from _common import add_common_args, WebcamLoop, resize_gray, draw_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisualCompass — yaw/heading estimation")
    add_common_args(parser)
    parser.add_argument("--image-size", type=int, default=64, choices=[64, 128],
                        help="Input image resolution (default: 64)")
    parser.add_argument("--fov-deg", type=float, default=90.0,
                        help="Camera horizontal FOV in degrees (default: 90)")
    parser.add_argument("--pooled", action="store_true", default=True,
                        help="Use fused Gabor+Pool model (default: True)")
    parser.add_argument("--no-pooled", dest="pooled", action="store_false",
                        help="Disable pooled mode")
    parser.add_argument("--pool-factor", type=int, default=4,
                        choices=[1, 2, 4, 8],
                        help="Spatial pooling factor (default: 4)")
    parser.add_argument("--search-range", type=int, default=4,
                        help="Max displacement to search (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Soft argmax temperature (default: 0.1)")
    return parser.parse_args()


def draw_compass(frame, yaw_deg, center=None, radius=50):
    """Draw a compass needle showing cumulative yaw."""
    h, w = frame.shape[:2]
    if center is None:
        center = (w - radius - 20, h - radius - 20)
    cx, cy = center

    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)
    cv2.putText(frame, "N", (cx - 5, cy - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    yaw_rad = np.deg2rad(yaw_deg - 90)
    ex = int(cx + radius * np.cos(yaw_rad))
    ey = int(cy + radius * np.sin(yaw_rad))
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (255, 255, 0), 3,
                    tipLength=0.2)

    draw_text(frame, f"{yaw_deg:.1f} deg", (cx - 30, cy + radius + 20))


def main():
    args = parse_args()
    loop = WebcamLoop(args)

    prev_gray = None
    cumulative_yaw = 0.0

    with VisualCompass.from_template(
        args.image_size,
        args.fov_deg,
        pooled=args.pooled,
        pool_factor=args.pool_factor,
        search_range=args.search_range,
        temperature=args.temperature,
    ) as compass:
        for frame in loop:
            gray = resize_gray(frame, args.image_size)

            if prev_gray is None:
                prev_gray = gray.copy()
                draw_text(frame, "Initializing... (press 'r' to reset yaw)",
                          (10, 30))
                loop.show(frame)
                continue

            t0 = time.perf_counter()
            delta_yaw, vx, vy = compass.compute(prev_gray, gray)
            latency_ms = (time.perf_counter() - t0) * 1000

            cumulative_yaw += delta_yaw
            direction = VisualCompass.yaw_to_direction(delta_yaw)

            # Draw compass
            draw_compass(frame, cumulative_yaw)

            h = frame.shape[0]
            draw_text(frame, f"Yaw: {cumulative_yaw:.1f} deg  delta: {delta_yaw:.2f}",
                      (10, 30))
            draw_text(frame,
                      f"Flow: ({vx:.2f}, {vy:.2f}) | {direction} | {latency_ms:.1f} ms",
                      (10, 55))
            draw_text(frame, "Press 'r' to reset yaw", (10, h - 20))

            loop.show(frame)

            # Check for reset key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                cumulative_yaw = 0.0

            loop.print_metrics({"yaw": cumulative_yaw, "delta": delta_yaw},
                               latency_ms)

            prev_gray = gray.copy()

    loop.cleanup()


if __name__ == "__main__":
    main()
