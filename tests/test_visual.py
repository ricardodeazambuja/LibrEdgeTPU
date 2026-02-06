#!/usr/bin/env python3
"""Visual proof tests: run real images through 4 model types on Edge TPU.

Produces annotated output images proving correct inference for:
  1. Classification (MobileNet V1) — top-5 labels
  2. Object Detection (SSD MobileDet) — bounding boxes + labels
  3. Semantic Segmentation (DeepLabV3) — feature activation heatmap
  4. Pose Estimation (PoseNet) — keypoints + skeleton
  5. Multi-Person Pose (MultiPose) — per-person skeletons

Usage:
    pytest tests/test_visual.py -v --run-hardware   # all 5 via pytest
    python -m tests.test_visual                     # all 5 via CLI
    python -m tests.test_visual --image photo.jpg   # custom image
    python -m tests.test_visual --models classification detection
"""

import argparse
import os
import sys

import numpy as np
import pytest

pytest.importorskip("PIL", reason="Pillow is required: pip install Pillow")
from PIL import Image, ImageDraw, ImageFont

from libredgetpu.simple_invoker import SimpleInvoker
from libredgetpu.tflite_parser import parse_full
from libredgetpu.postprocess.ssd_decoder import postprocess_ssd
from tests.model_zoo import get_model, get_labels, get_sample_image, get_multipose_image

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_and_resize(image_path, width, height):
    """Load image, resize to (width, height), return (PIL Image, np uint8 array)."""
    img = Image.open(image_path).convert("RGB")
    resized = img.resize((width, height), Image.BILINEAR)
    return img, resized, np.array(resized, dtype=np.uint8)



def _save(img, name):
    """Save image to results directory."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    path = os.path.join(_RESULTS_DIR, name)
    img.save(path)
    print(f"  Saved: {path}")
    return path


def _get_font(size=14):
    """Try to get a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


# ── 1. Classification ────────────────────────────────────────────────────────

def classify_image(image_path):
    """Run MobileNet V1 classification and return annotated image."""
    print("\n[Classification] MobileNet V1")
    model_path = get_model("mobilenet_v1")
    labels = get_labels("imagenet")

    orig, resized, input_arr = _load_and_resize(image_path, 224, 224)

    with SimpleInvoker(model_path) as model:
        raw = model.invoke_raw(input_arr.tobytes())

    scores = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    # Dequantize using TFLite output quant params (but for top-k ranking,
    # raw uint8 ordering is sufficient)
    top5_idx = np.argsort(scores)[-5:][::-1]

    # Draw on original image
    result = orig.copy()
    draw = ImageDraw.Draw(result)
    font = _get_font(max(16, result.height // 25))

    y = 10
    for i, idx in enumerate(top5_idx):
        score = scores[idx]
        label = labels[idx] if idx < len(labels) else f"class {idx}"
        text = f"{i+1}. {label} ({score:.0f})"
        print(f"  {text}")
        # Draw text with background
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([8, y - 2, 12 + tw, y + th + 2], fill=(0, 0, 0, 180))
        draw.text((10, y), text, fill=(0, 255, 0), font=font)
        y += th + 6

    return result


# ── 2. Object Detection ──────────────────────────────────────────────────────

def detect_objects(image_path):
    """Run SSD MobileDet detection and return annotated image."""
    print("\n[Detection] SSD MobileDet")
    model_path = get_model("ssd_mobiledet")
    coco_labels = get_labels("coco")

    orig, resized, input_arr = _load_and_resize(image_path, 320, 320)

    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    with SimpleInvoker(model_path) as model:
        raw_outputs = model.invoke_raw_outputs(input_arr.tobytes())
        output_layers = model.output_layers

    detections = postprocess_ssd(raw_outputs, output_layers, tflite_bytes,
                                 score_threshold=0.6)

    print(f"  Found {len(detections)} detections")

    # Draw on original image
    result = orig.copy()
    draw = ImageDraw.Draw(result)
    font = _get_font(max(14, result.height // 30))
    w, h = result.size

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]

    for i, (cls_id, score, ymin, xmin, ymax, xmax) in enumerate(detections[:20]):
        color = colors[cls_id % len(colors)]
        label_idx = cls_id - 1
        label = coco_labels[label_idx] if 0 <= label_idx < len(coco_labels) else f"class {cls_id}"
        text = f"{label} {score:.0%}"
        print(f"  {text} [{ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f}]")

        x0, y0 = int(xmin * w), int(ymin * h)
        x1, y1 = int(xmax * w), int(ymax * h)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x0, y0 - th - 4, x0 + tw + 4, y0], fill=color)
        draw.text((x0 + 2, y0 - th - 2), text, fill=(255, 255, 255), font=font)

    return result


# ── 3. Semantic Segmentation ─────────────────────────────────────────────────

# PASCAL VOC class names and colors
PASCAL_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

PASCAL_COLORS = np.array([
    [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128],
    [128,0,128], [0,128,128], [128,128,128], [64,0,0], [192,0,0],
    [64,128,0], [192,128,0], [64,0,128], [192,0,128], [64,128,128],
    [192,128,128], [0,64,0], [128,64,0], [0,192,0], [128,192,0],
    [0,64,128],
], dtype=np.uint8)


def segment_image(image_path):
    """Run DeepLabV3 and return semantic segmentation overlay."""
    print("\n[Segmentation] DeepLabV3")
    from libredgetpu.postprocess.deeplabv3 import postprocess_deeplabv3

    model_path = get_model("deeplabv3")

    orig, resized, input_arr = _load_and_resize(image_path, 513, 513)

    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    with SimpleInvoker(model_path) as model:
        raw_outputs = model.invoke_raw_outputs(input_arr.tobytes())
        output_layers = model.output_layers

    seg_map = postprocess_deeplabv3(raw_outputs, output_layers, tflite_bytes)

    classes_found = np.unique(seg_map)
    print(f"  Classes found:")
    for c in classes_found:
        pct = 100.0 * np.sum(seg_map == c) / seg_map.size
        print(f"    {PASCAL_CLASSES[c]:15s}: {pct:.1f}%")

    # Colorize and resize to original image size
    color_map = PASCAL_COLORS[seg_map]
    seg_img = Image.fromarray(color_map).resize(orig.size, Image.NEAREST)

    # Blend with original image
    result = Image.blend(orig, seg_img, alpha=0.5)

    # Add legend
    draw = ImageDraw.Draw(result)
    font = _get_font(max(14, result.height // 30))
    y = 10
    for c in classes_found:
        color = tuple(PASCAL_COLORS[c])
        label = PASCAL_CLASSES[c]
        pct = 100.0 * np.sum(seg_map == c) / seg_map.size
        text = f"{label} ({pct:.0f}%)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([8, y - 1, 14 + tw, y + th + 1], fill=(0, 0, 0, 180))
        draw.rectangle([10, y + 2, 10 + th - 4, y + th - 2], fill=color, outline=(255, 255, 255))
        draw.text((16 + th - 4, y), text, fill=(255, 255, 255), font=font)
        y += th + 6

    return result


# ── 4. Pose Estimation ───────────────────────────────────────────────────────

# COCO keypoint names and skeleton connections
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12),                           # torso
    (11, 12),                                   # hips
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

KEYPOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]


def estimate_pose(image_path):
    """Run PoseNet and return annotated image with keypoints + skeleton."""
    print("\n[Pose] PoseNet")
    from libredgetpu.postprocess.posenet_decoder import postprocess_posenet

    model_path = get_model("posenet")

    # PoseNet input: 641x481 (width x height)
    orig, resized, input_arr = _load_and_resize(image_path, 641, 481)
    input_h, input_w = 481, 641

    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    with SimpleInvoker(model_path) as model:
        raw_outputs = model.invoke_raw_outputs(input_arr.tobytes())
        output_layers = model.output_layers

    poses = postprocess_posenet(raw_outputs, output_layers, tflite_bytes)

    print(f"  Found {len(poses)} pose(s)")

    # Use the top pose
    keypoints = []
    if poses:
        pose = poses[0]
        print(f"  Top pose score: {pose.score:.2f}")
        for k in range(17):
            # PoseNet returns keypoints as (y, x) in pixel coords
            y_px = pose.keypoints[k, 0]
            x_px = pose.keypoints[k, 1]
            conf = pose.keypoint_scores[k]
            keypoints.append((x_px, y_px, conf))
    else:
        keypoints = [(0, 0, 0)] * 17

    # Scale to original image size
    orig_w, orig_h = orig.size
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    print(f"  Keypoints (confidence > 0.3):")
    for i, (x, y, c) in enumerate(keypoints):
        if c > 0.3:
            print(f"    {KEYPOINT_NAMES[i]:16s}: ({x:.0f}, {y:.0f}) conf={c:.2f}")

    # Draw on original image
    result = orig.copy()
    draw = ImageDraw.Draw(result)
    radius = max(4, min(orig_w, orig_h) // 80)
    line_width = max(2, radius // 2)
    confidence_threshold = 0.5

    # Draw skeleton lines
    for (i, j) in SKELETON:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci > confidence_threshold and cj > confidence_threshold:
            draw.line(
                [(xi * scale_x, yi * scale_y), (xj * scale_x, yj * scale_y)],
                fill=(255, 255, 255), width=line_width,
            )

    # Draw keypoints
    for k, (x, y, c) in enumerate(keypoints):
        if c > confidence_threshold:
            px, py = x * scale_x, y * scale_y
            color = KEYPOINT_COLORS[k]
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color, outline=(255, 255, 255), width=1,
            )

    # Legend
    font = _get_font(max(12, result.height // 35))
    draw.text((10, 10), "PoseNet Keypoints", fill=(255, 255, 255), font=font)

    return result


# ── 5. Multi-Person Pose Estimation ─────────────────────────────────────────

# Per-person colors for multi-pose visualization
_PERSON_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 100, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def estimate_multipose(image_path):
    """Run MultiPose PoseNet and return annotated image with per-person skeletons.

    Uses a multi-person test image (test_couple.jpg) instead of the provided
    image_path, since multipose needs multiple people to be visually meaningful.
    """
    print("\n[MultiPose] PoseNet Multi-Person")
    from libredgetpu.postprocess.multipose_decoder import postprocess_multipose

    model_path = get_model("posenet_multipose")

    # Use a dedicated multi-person image for meaningful visual output
    multipose_image = get_multipose_image()
    print(f"  Using multi-person image: {multipose_image}")

    # MultiPose input: 257x257
    orig, resized, input_arr = _load_and_resize(multipose_image, 257, 257)
    input_h, input_w = 257, 257

    # Int8 preprocessing: quantize to int8, then XOR 0x80 for Edge TPU
    input_int8 = (input_arr.astype(np.float32) - 127).astype(np.int8)
    input_bytes = (input_int8.view(np.uint8) ^ 0x80).tobytes()

    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    with SimpleInvoker(model_path) as model:
        raw_outputs = model.invoke_raw_outputs(input_bytes)
        output_layers = model.output_layers

    poses = postprocess_multipose(raw_outputs, output_layers, tflite_bytes)

    # Filter: keep only poses with at least 3 confident keypoints
    kp_threshold = 0.3
    min_valid_kps = 3
    good_poses = []
    for pose in poses:
        n_valid = int(np.sum(pose.keypoint_scores > kp_threshold))
        if n_valid >= min_valid_kps:
            good_poses.append(pose)

    print(f"  Raw: {len(poses)} pose(s), after filtering: {len(good_poses)} with >= {min_valid_kps} keypoints")

    # Scale to original image size
    orig_w, orig_h = orig.size
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    # Draw on original image
    result = orig.copy()
    draw = ImageDraw.Draw(result)
    radius = max(4, min(orig_w, orig_h) // 80)
    line_width = max(2, radius // 2)

    for p_idx, pose in enumerate(good_poses):
        color = _PERSON_COLORS[p_idx % len(_PERSON_COLORS)]
        print(f"  Person {p_idx + 1}: score={pose.score:.2f}, "
              f"keypoints={int(np.sum(pose.keypoint_scores > kp_threshold))}")

        # Extract keypoints for this pose
        keypoints = []
        for k in range(17):
            y_px = pose.keypoints[k, 0]
            x_px = pose.keypoints[k, 1]
            conf = pose.keypoint_scores[k]
            keypoints.append((x_px, y_px, conf))
            if conf > kp_threshold:
                print(f"    {KEYPOINT_NAMES[k]:16s}: ({x_px:.0f}, {y_px:.0f}) conf={conf:.2f}")

        # Draw skeleton lines
        for (i, j) in SKELETON:
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > kp_threshold and cj > kp_threshold:
                draw.line(
                    [(xi * scale_x, yi * scale_y), (xj * scale_x, yj * scale_y)],
                    fill=color, width=line_width,
                )

        # Draw keypoints
        for k, (x, y, c) in enumerate(keypoints):
            if c > kp_threshold:
                px, py = x * scale_x, y * scale_y
                draw.ellipse(
                    [px - radius, py - radius, px + radius, py + radius],
                    fill=color, outline=(255, 255, 255), width=1,
                )

    # Legend
    font = _get_font(max(12, result.height // 35))
    draw.text((10, 10), f"MultiPose: {len(good_poses)} person(s)", fill=(255, 255, 255), font=font)

    return result


# ── Pytest test wrappers ─────────────────────────────────────────────────────

def _default_image():
    """Return default test image path (Grace Hopper)."""
    return get_sample_image()


def _posenet_image():
    """Return full-body image for pose estimation, fallback to default."""
    path = os.path.join(os.path.dirname(__file__), "models", "squat.bmp")
    return path if os.path.isfile(path) else _default_image()


@pytest.mark.hardware
def test_classification():
    """Visual test: MobileNet V1 classification with top-5 labels."""
    result = classify_image(_default_image())
    _save(result, "classification_mobilenet_v1.png")


@pytest.mark.hardware
def test_detection():
    """Visual test: SSD MobileDet object detection with bounding boxes."""
    result = detect_objects(_default_image())
    _save(result, "detection_ssd_mobiledet.png")


@pytest.mark.hardware
def test_segmentation():
    """Visual test: DeepLabV3 semantic segmentation overlay."""
    result = segment_image(_default_image())
    _save(result, "segmentation_deeplabv3.png")


@pytest.mark.hardware
def test_pose():
    """Visual test: PoseNet single-person keypoints + skeleton."""
    result = estimate_pose(_posenet_image())
    _save(result, "pose_posenet.png")


@pytest.mark.hardware
def test_multipose():
    """Visual test: MultiPose multi-person skeletons."""
    result = estimate_multipose(_default_image())
    _save(result, "multipose_posenet.png")


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_TASKS = {
    "classification": ("classification_mobilenet_v1.png", classify_image),
    "detection": ("detection_ssd_mobiledet.png", detect_objects),
    "segmentation": ("segmentation_deeplabv3.png", segment_image),
    "pose": ("pose_posenet.png", estimate_pose),
    "multipose": ("multipose_posenet.png", estimate_multipose),
}


def main():
    parser = argparse.ArgumentParser(description="Visual proof tests for libredgetpu")
    parser.add_argument("--image", type=str, help="Path to input image (default: sample)")
    parser.add_argument("--models", nargs="+", choices=list(ALL_TASKS.keys()),
                        help="Which models to run (default: all)")
    args = parser.parse_args()

    # Get image
    if args.image:
        image_path = args.image
        if not os.path.isfile(image_path):
            print(f"Image not found: {image_path}")
            sys.exit(1)
    else:
        image_path = get_sample_image()
    print(f"Input image: {image_path}")

    # Use a full-body single-person image for pose estimation if available
    posenet_image = os.path.join(os.path.dirname(__file__), "models", "squat.bmp")
    if not os.path.isfile(posenet_image):
        posenet_image = image_path

    tasks = args.models or list(ALL_TASKS.keys())
    passed = 0
    failed = 0

    for task_name in tasks:
        filename, func = ALL_TASKS[task_name]
        try:
            task_image = posenet_image if task_name == "pose" else image_path
            result_img = func(task_image)
            _save(result_img, filename)
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            err_str = str(e)
            if "No such device" in err_str or "No Google Coral" in err_str:
                print("\n  Device lost — stopping (physical replug required)")
                break

    print(f"\nVisual tests: {passed} passed, {failed} failed")
    print(f"Results in: {_RESULTS_DIR}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
