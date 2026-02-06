#!/usr/bin/env python3
"""Spot tracker example — visual servoing via soft argmax centroid.

Tracks the brightest point (or a specific color) in the camera frame and
outputs (x_offset, y_offset) in [-1, +1], ready for PID-based servo control.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time

import cv2
import numpy as np

from libredgetpu import SpotTracker
from _common import add_common_args, WebcamLoop, resize_for_tracker, draw_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpotTracker — visual servoing via soft argmax")
    add_common_args(parser)
    parser.add_argument("--image-size", type=int, default=64, choices=[64, 128],
                        help="Input image resolution (default: 64)")
    parser.add_argument("--variant", type=str, default="bright",
                        choices=["bright", "color_red", "color_green",
                                 "color_blue", "color_yellow", "color_cyan",
                                 "color_magenta", "color_white"],
                        help="Tracking variant (default: bright)")
    parser.add_argument("--servo-gain", type=float, default=1.0,
                        help="Gain applied to servo error output (default: 1.0)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Dead-zone threshold for direction (default: 0.1)")
    return parser.parse_args()


def offset_to_direction(x_off, y_off, threshold):
    """Classify offset into a human-readable direction."""
    if abs(x_off) < threshold and abs(y_off) < threshold:
        return "centered"
    if abs(x_off) > abs(y_off):
        return "right" if x_off > 0 else "left"
    return "down" if y_off > 0 else "up"


def main():
    args = parse_args()
    loop = WebcamLoop(args)

    with SpotTracker.from_template(args.image_size, variant=args.variant) as tracker:
        for frame in loop:
            img = resize_for_tracker(frame, args.image_size, tracker._channels)

            t0 = time.perf_counter()
            x_off, y_off = tracker.track(img)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Apply servo gain
            servo_x = -x_off * args.servo_gain
            servo_y = -y_off * args.servo_gain

            direction = offset_to_direction(x_off, y_off, args.threshold)

            # Draw crosshair at tracked position
            h, w = frame.shape[:2]
            px = int((x_off / 2 + 0.5) * w)
            py = int((y_off / 2 + 0.5) * h)
            cv2.line(frame, (px - 20, py), (px + 20, py), (0, 255, 0), 2)
            cv2.line(frame, (px, py - 20), (px, py + 20), (0, 255, 0), 2)

            draw_text(frame, f"Offset: ({x_off:+.3f}, {y_off:+.3f})", (10, 30))
            draw_text(frame, f"Servo:  ({servo_x:+.3f}, {servo_y:+.3f})", (10, 55))
            draw_text(frame, f"Dir: {direction} | {latency_ms:.1f} ms", (10, 80))

            loop.show(frame)
            loop.print_metrics({"x": x_off, "y": y_off, "dir": direction},
                               latency_ms)

    loop.cleanup()


if __name__ == "__main__":
    main()
