#!/usr/bin/env python3
"""Pattern tracker example — template matching via Conv2D correlation.

Tracks a template pattern across frames using Edge TPU-accelerated
convolution. Templates can be loaded from a file or auto-cropped from
the center of the first frame.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time

import cv2
import numpy as np

from libredgetpu import PatternTracker
from _common import add_common_args, WebcamLoop, resize_for_tracker, draw_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="PatternTracker — template matching via Conv2D")
    add_common_args(parser)
    parser.add_argument("--search-size", type=int, default=64,
                        choices=[64, 128],
                        help="Search window resolution (default: 64)")
    parser.add_argument("--kernel-size", type=int, default=16,
                        choices=[8, 16, 32],
                        help="Template kernel size (default: 16)")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3],
                        help="Input channels: 1=grayscale, 3=color (default: 1)")
    parser.add_argument("--template", type=str, default=None,
                        help="Path to template image file (default: auto-crop center)")
    return parser.parse_args()


def load_template(path, kernel_size, channels):
    """Load and resize template from file."""
    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot read template image: {path}")
    return cv2.resize(img, (kernel_size, kernel_size),
                      interpolation=cv2.INTER_AREA)


def auto_crop_template(img, kernel_size):
    """Crop center of frame as template."""
    h, w = img.shape[:2]
    y0 = (h - kernel_size) // 2
    x0 = (w - kernel_size) // 2
    return img[y0:y0 + kernel_size, x0:x0 + kernel_size].copy()


def main():
    args = parse_args()
    loop = WebcamLoop(args)

    template_set = False
    template_display = None

    with PatternTracker.from_template(
        args.search_size, args.kernel_size, channels=args.channels
    ) as tracker:
        for frame in loop:
            img = resize_for_tracker(frame, args.search_size, args.channels)

            # Set template on first frame
            if not template_set:
                if args.template:
                    patch = load_template(args.template, args.kernel_size,
                                          args.channels)
                else:
                    patch = auto_crop_template(img, args.kernel_size)
                tracker.set_template(patch)
                template_display = cv2.resize(patch, (64, 64),
                                              interpolation=cv2.INTER_NEAREST)
                template_set = True
                print(f"Template set: {args.kernel_size}x{args.kernel_size}")

            t0 = time.perf_counter()
            x_off, y_off = tracker.track(img)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Convert offset [-1, +1] to pixel coords in original frame
            h, w = frame.shape[:2]
            match_x = int((x_off / 2 + 0.5) * w)
            match_y = int((y_off / 2 + 0.5) * h)

            # Draw bounding box at matched position
            half_k = args.kernel_size * w // args.search_size // 2
            cv2.rectangle(frame,
                          (match_x - half_k, match_y - half_k),
                          (match_x + half_k, match_y + half_k),
                          (0, 0, 255), 2)
            cv2.circle(frame, (match_x, match_y), 3, (0, 255, 0), -1)

            # Show template thumbnail in corner
            if template_display is not None:
                if template_display.ndim == 2:
                    thumb = cv2.cvtColor(template_display, cv2.COLOR_GRAY2BGR)
                else:
                    thumb = template_display
                frame[5:69, 5:69] = thumb
                cv2.rectangle(frame, (4, 4), (70, 70), (0, 255, 255), 1)

            draw_text(frame,
                      f"Match: ({match_x}, {match_y}) off=({x_off:.3f}, {y_off:.3f})",
                      (80, 30))
            draw_text(frame, f"{latency_ms:.1f} ms", (80, 55))
            draw_text(frame, "Press 't' to re-capture template from center",
                      (10, h - 20))

            loop.show(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("t"):
                patch = auto_crop_template(img, args.kernel_size)
                tracker.set_template(patch)
                template_display = cv2.resize(patch, (64, 64),
                                              interpolation=cv2.INTER_NEAREST)
                print("Template re-captured from center")

            loop.print_metrics({"x": x_off, "y": y_off}, latency_ms)

    loop.cleanup()


if __name__ == "__main__":
    main()
