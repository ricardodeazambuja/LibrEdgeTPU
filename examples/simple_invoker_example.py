#!/usr/bin/env python3
"""Simple invoker example — standard ML inference with 5 model types.

Runs pre-trained Edge TPU models with full post-processing:
classification, detection, segmentation, posenet, multipose.

Models are auto-downloaded from the EdgeTPUModelZoo on first run.

Requirements: Edge TPU USB accelerator, opencv-python
"""

import argparse
import time

import cv2
import numpy as np

from libredgetpu import SimpleInvoker
from libredgetpu.gui.model_zoo import (
    MODEL_REGISTRY, download_model, download_labels, get_model_names,
)
from _common import add_common_args, WebcamLoop, draw_text


# COCO skeleton connections for pose drawing
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Per-person colors (BGR)
_PERSON_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 100, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]

# PASCAL VOC colormap (21 classes)
_PASCAL_COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


# Map short names to full registry names
_MODEL_SHORTCUTS = {
    "classification": "Classification (MobileNet V1)",
    "detection": "Detection (SSD MobileDet)",
    "segmentation": "Segmentation (DeepLabV3)",
    "posenet": "Pose (PoseNet)",
    "multipose": "MultiPose (Multi-Person)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SimpleInvoker — ML inference with post-processing")
    add_common_args(parser)
    parser.add_argument("--model", type=str, default="classification",
                        choices=list(_MODEL_SHORTCUTS.keys()),
                        help="Model type (default: classification)")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Confidence threshold for detection/pose (default: 0.5)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Custom .tflite model path (overrides --model)")
    return parser.parse_args()


def draw_classification(frame, raw_outputs, labels):
    """Draw top-5 classification labels."""
    scores = np.frombuffer(raw_outputs[0], dtype=np.uint8).astype(np.float32)
    top5 = np.argsort(scores)[-5:][::-1]
    for i, idx in enumerate(top5):
        label = labels[idx] if idx < len(labels) else f"class {idx}"
        draw_text(frame, f"{i+1}. {label} ({scores[idx]:.0f})",
                  (10, 110 + i * 22))


def draw_detection(frame, raw_outputs, output_layers, tflite_bytes,
                   labels, threshold):
    """Draw bounding boxes for SSD detection."""
    from libredgetpu.postprocess.ssd_decoder import postprocess_ssd
    detections = postprocess_ssd(raw_outputs, output_layers,
                                 tflite_bytes, score_threshold=threshold)
    h, w = frame.shape[:2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
              (255, 0, 255), (255, 255, 0)]

    for cls_id, score, ymin, xmin, ymax, xmax in detections[:10]:
        color = colors[cls_id % len(colors)]
        label_idx = cls_id - 1
        label = labels[label_idx] if 0 <= label_idx < len(labels) else f"class {cls_id}"
        x0, y0 = max(0, int(xmin * w)), max(0, int(ymin * h))
        x1, y1 = min(w, int(xmax * w)), min(h, int(ymax * h))
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        draw_text(frame, f"{label} {score:.0%}", (x0, y0 - 5), color=color)


def draw_segmentation(frame, raw_outputs, output_layers, tflite_bytes):
    """Draw segmentation colormap overlay."""
    from libredgetpu.postprocess.deeplabv3 import postprocess_deeplabv3
    seg_map = postprocess_deeplabv3(raw_outputs, output_layers, tflite_bytes)
    h, w = frame.shape[:2]
    color_rgb = _PASCAL_COLORS[seg_map]
    color_resized = cv2.resize(color_rgb, (w, h),
                               interpolation=cv2.INTER_NEAREST)
    color_bgr = color_resized[:, :, ::-1]
    cv2.addWeighted(color_bgr, 0.4, frame, 0.6, 0, frame)


def draw_pose(frame, raw_outputs, output_layers, tflite_bytes,
              input_size, threshold):
    """Draw PoseNet skeleton."""
    from libredgetpu.postprocess.posenet_decoder import postprocess_posenet
    poses = postprocess_posenet(raw_outputs, output_layers, tflite_bytes)
    _draw_skeletons(frame, poses, input_size, threshold)


def draw_multipose(frame, raw_outputs, output_layers, tflite_bytes,
                   input_size, threshold):
    """Draw MultiPose skeletons (filtered by keypoint count)."""
    from libredgetpu.postprocess.multipose_decoder import postprocess_multipose
    poses = postprocess_multipose(raw_outputs, output_layers, tflite_bytes)
    # Keep poses with at least 3 confident keypoints
    good = [p for p in poses if int(np.sum(p.keypoint_scores > 0.3)) >= 3]
    _draw_skeletons(frame, good, input_size, threshold)


def _draw_skeletons(frame, poses, input_size, threshold):
    """Draw keypoints and skeleton edges."""
    h, w = frame.shape[:2]
    input_w, input_h = input_size
    sx, sy = w / input_w, h / input_h
    r = max(3, min(w, h) // 100)

    for pi, pose in enumerate(poses):
        color = _PERSON_COLORS[pi % len(_PERSON_COLORS)]
        kps = pose.keypoints  # [17, 2] as (y, x)
        scores = pose.keypoint_scores

        for i, j in _SKELETON:
            if scores[i] > threshold and scores[j] > threshold:
                p1 = (int(kps[i, 1] * sx), int(kps[i, 0] * sy))
                p2 = (int(kps[j, 1] * sx), int(kps[j, 0] * sy))
                cv2.line(frame, p1, p2, color, max(1, r // 2))

        for k in range(17):
            if scores[k] > threshold:
                px = int(kps[k, 1] * sx)
                py = int(kps[k, 0] * sy)
                cv2.circle(frame, (px, py), r, color, -1)
                cv2.circle(frame, (px, py), r, (255, 255, 255), 1)


def main():
    args = parse_args()

    if args.model_path:
        # Custom model — no post-processing
        model_name = None
        model_path = args.model_path
        meta = None
    else:
        model_name = _MODEL_SHORTCUTS[args.model]
        meta = MODEL_REGISTRY[model_name]
        print(f"Downloading model: {model_name}...")
        model_path = download_model(model_name)

    labels = []
    if meta and meta.get("labels"):
        labels = download_labels(meta["labels"])

    tflite_bytes = None
    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    loop = WebcamLoop(args)

    with SimpleInvoker(model_path) as invoker:
        for frame in loop:
            if meta:
                input_w, input_h = meta["input_size"]
                resized = cv2.resize(frame, (input_w, input_h),
                                     interpolation=cv2.INTER_AREA)

                if meta["input_type"] == "int8":
                    int8_vals = (resized.astype(np.float32) - 127).astype(np.int8)
                    input_bytes = (int8_vals.view(np.uint8) ^ 0x80).tobytes()
                else:
                    input_bytes = resized.tobytes()
            else:
                # Custom model: send raw frame bytes
                input_bytes = frame.tobytes()
                input_w, input_h = frame.shape[1], frame.shape[0]

            t0 = time.perf_counter()
            raw_outputs = invoker.invoke_raw_outputs(input_bytes)
            output_layers = invoker.output_layers
            latency_ms = (time.perf_counter() - t0) * 1000

            # Post-processing dispatch
            if meta:
                pp = meta["postprocessor"]
                if pp == "classification":
                    draw_classification(frame, raw_outputs, labels)
                elif pp == "ssd_detection":
                    draw_detection(frame, raw_outputs, output_layers,
                                   tflite_bytes, labels, args.score_threshold)
                elif pp == "deeplabv3":
                    draw_segmentation(frame, raw_outputs, output_layers,
                                      tflite_bytes)
                elif pp == "posenet":
                    draw_pose(frame, raw_outputs, output_layers, tflite_bytes,
                              (input_w, input_h), args.score_threshold)
                elif pp == "multipose":
                    draw_multipose(frame, raw_outputs, output_layers,
                                   tflite_bytes, (input_w, input_h),
                                   args.score_threshold)

            h = frame.shape[0]
            model_label = args.model if not args.model_path else "custom"
            draw_text(frame, f"{model_label} | {latency_ms:.1f} ms", (10, 30))

            loop.show(frame)
            loop.print_metrics({"model": model_label}, latency_ms)

    loop.cleanup()


if __name__ == "__main__":
    main()
