"""OpenCV overlay rendering helpers for GUI visualization.

Extracted and adapted from test_visual_robotics.py to ensure consistent,
battle-tested rendering across the interactive GUI.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


# Color scheme constants (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def draw_crosshair(
    img: np.ndarray,
    x: float,
    y: float,
    color: Tuple[int, int, int] = COLOR_GREEN,
    size: int = 20,
    thickness: int = 2
) -> None:
    """Draw a crosshair at the specified position.

    Args:
        img: Image to draw on (modified in-place)
        x: X coordinate (pixel)
        y: Y coordinate (pixel)
        color: BGR color tuple
        size: Crosshair arm length in pixels
        thickness: Line thickness
    """
    cx, cy = int(x), int(y)
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)


def draw_text_with_background(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
    text_color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_BLACK,
    padding: int = 4
) -> None:
    """Draw text with a background rectangle for better visibility.

    Args:
        img: Image to draw on (modified in-place)
        text: Text string to render
        position: (x, y) bottom-left corner of text
        font_scale: OpenCV font scale
        thickness: Text line thickness
        text_color: BGR color for text
        bg_color: BGR color for background rectangle
        padding: Padding around text in pixels
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Draw background rectangle
    cv2.rectangle(
        img,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1
    )
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)


def value_to_heatmap_color(value: float, vmin: float = 0.0, vmax: float = 1.0) -> Tuple[int, int, int]:
    """Convert a scalar value to a heatmap color (blue → green → red).

    Args:
        value: Input value
        vmin: Minimum value (maps to blue)
        vmax: Maximum value (maps to red)

    Returns:
        BGR color tuple
    """
    # Normalize to [0, 1]
    normalized = np.clip((value - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Simple blue → green → red colormap
    if normalized < 0.5:
        # Blue → Green
        r = 0
        g = int(255 * (normalized * 2))
        b = int(255 * (1 - normalized * 2))
    else:
        # Green → Red
        r = int(255 * ((normalized - 0.5) * 2))
        g = int(255 * (1 - (normalized - 0.5) * 2))
        b = 0

    return (b, g, r)  # BGR format


def draw_heatmap_grid(
    img: np.ndarray,
    values: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    alpha: float = 0.5
) -> None:
    """Overlay a heatmap grid on the image.

    Args:
        img: Image to draw on (modified in-place)
        values: 2D array of values (grid_rows × grid_cols)
        grid_rows: Number of grid rows
        grid_cols: Number of grid columns
        alpha: Overlay transparency (0=transparent, 1=opaque)
    """
    h, w = img.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    overlay = img.copy()
    vmin, vmax = values.min(), values.max()

    for i in range(grid_rows):
        for j in range(grid_cols):
            color = value_to_heatmap_color(values[i, j], vmin, vmax)
            y1 = i * cell_h
            x1 = j * cell_w
            y2 = min(y1 + cell_h, h)
            x2 = min(x1 + cell_w, w)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_flow_arrow(
    img: np.ndarray,
    dx: float,
    dy: float,
    scale: float = 10.0,
    color: Tuple[int, int, int] = COLOR_RED,
    thickness: int = 2
) -> None:
    """Draw a flow vector arrow from center of image.

    Args:
        img: Image to draw on (modified in-place)
        dx: Horizontal displacement
        dy: Vertical displacement
        scale: Arrow length multiplier
        color: BGR color
        thickness: Arrow line thickness
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    end_x = int(cx + dx * scale)
    end_y = int(cy + dy * scale)

    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), color, thickness, tipLength=0.3)


def draw_compass_needle(
    img: np.ndarray,
    yaw_deg: float,
    radius: int = 50,
    center: Optional[Tuple[int, int]] = None,
    color: Tuple[int, int, int] = COLOR_CYAN,
    thickness: int = 3
) -> None:
    """Draw a compass needle showing yaw angle.

    Args:
        img: Image to draw on (modified in-place)
        yaw_deg: Yaw angle in degrees (0=north, clockwise positive)
        radius: Needle length in pixels
        center: (x, y) center position, defaults to bottom-right corner
        color: BGR color
        thickness: Line thickness
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w - radius - 20, h - radius - 20)

    cx, cy = center

    # Draw compass circle
    cv2.circle(img, (cx, cy), radius, COLOR_WHITE, 1)

    # Draw cardinal directions
    cv2.putText(img, "N", (cx - 5, cy - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

    # Draw needle (0 degrees = up, clockwise positive)
    yaw_rad = np.deg2rad(yaw_deg - 90)  # Convert to standard math coords
    end_x = int(cx + radius * np.cos(yaw_rad))
    end_y = int(cy + radius * np.sin(yaw_rad))

    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), color, thickness, tipLength=0.2)

    # Draw angle text
    text = f"{yaw_deg:.1f}°"
    draw_text_with_background(img, text, (cx - 20, cy + radius + 20))


def draw_scatter_plot(
    img: np.ndarray,
    x_data: np.ndarray,
    y_data: np.ndarray,
    position: Tuple[int, int],
    size: Tuple[int, int] = (150, 150),
    title: str = "",
    color: Tuple[int, int, int] = COLOR_CYAN
) -> None:
    """Draw a small scatter plot overlay.

    Args:
        img: Image to draw on (modified in-place)
        x_data: X coordinates (will be auto-scaled)
        y_data: Y coordinates (will be auto-scaled)
        position: (x, y) top-left corner of plot
        size: (width, height) of plot area
        title: Plot title
        color: Point color (BGR)
    """
    px, py = position
    pw, ph = size

    # Draw background
    cv2.rectangle(img, (px, py), (px + pw, py + ph), COLOR_BLACK, -1)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), COLOR_WHITE, 1)

    if len(x_data) == 0:
        return

    # Normalize data to plot area
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()

    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1

    x_scaled = ((x_data - x_min) / x_range * (pw - 20) + 10).astype(int)
    y_scaled = ((y_data - y_min) / y_range * (ph - 20) + 10).astype(int)

    # Draw points
    for x, y in zip(x_scaled, y_scaled):
        cv2.circle(img, (px + x, py + ph - y), 2, color, -1)

    # Draw title
    if title:
        draw_text_with_background(img, title, (px + 5, py + 15), font_scale=0.4)


def draw_histogram(
    img: np.ndarray,
    data: np.ndarray,
    position: Tuple[int, int],
    size: Tuple[int, int] = (150, 100),
    bins: int = 20,
    title: str = "",
    color: Tuple[int, int, int] = COLOR_GREEN
) -> None:
    """Draw a histogram overlay.

    Args:
        img: Image to draw on (modified in-place)
        data: 1D array of values
        position: (x, y) top-left corner
        size: (width, height) of plot area
        bins: Number of histogram bins
        title: Plot title
        color: Bar color (BGR)
    """
    px, py = position
    pw, ph = size

    # Draw background
    cv2.rectangle(img, (px, py), (px + pw, py + ph), COLOR_BLACK, -1)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), COLOR_WHITE, 1)

    if len(data) == 0:
        return

    # Compute histogram
    counts, _ = np.histogram(data, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1

    # Draw bars
    bar_width = pw // bins
    for i, count in enumerate(counts):
        bar_height = int((count / max_count) * (ph - 10))
        x1 = px + i * bar_width
        y1 = py + ph - bar_height
        x2 = px + (i + 1) * bar_width - 1
        y2 = py + ph
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    # Draw title
    if title:
        draw_text_with_background(img, title, (px + 5, py + 15), font_scale=0.4)


def draw_bottom_message(
    img: np.ndarray,
    text: str,
    font_scale: float = 0.5,
    thickness: int = 1,
    text_color: Tuple[int, int, int] = COLOR_RED,
    bg_color: Tuple[int, int, int] = COLOR_BLACK,
    padding: int = 4,
    margin: int = 10
) -> None:
    """Draw a word-wrapped message at the bottom of the image.

    Long text is split into multiple lines that fit within the image width.
    Lines are stacked upward from the bottom margin so they never overlap
    the performance HUD in the top-left corner.

    Args:
        img: Image to draw on (modified in-place)
        text: Text string to render (may be long)
        font_scale: OpenCV font scale
        thickness: Text line thickness
        text_color: BGR color for text
        bg_color: BGR color for background rectangle
        padding: Padding around text in pixels
        margin: Margin from image edges in pixels
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = w - 2 * margin - 2 * padding

    # Word-wrap: split text into lines that fit within max_width
    words = text.split()
    lines: List[str] = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_width or not current_line:
            current_line = test
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    if not lines:
        return

    # Measure line height once
    (_, line_h), baseline = cv2.getTextSize("Ay", font, font_scale, thickness)
    line_step = line_h + baseline + 2 * padding

    # Draw lines bottom-up from the bottom margin
    y_cursor = h - margin
    for line in reversed(lines):
        draw_text_with_background(
            img, line, (margin, y_cursor),
            font_scale=font_scale, thickness=thickness,
            text_color=text_color, bg_color=bg_color, padding=padding
        )
        y_cursor -= line_step


def draw_performance_hud(
    img: np.ndarray,
    fps: float,
    latency_ms: float,
    mode: str = "HARDWARE"
) -> None:
    """Draw performance metrics HUD in top-left corner.

    Args:
        img: Image to draw on (modified in-place)
        fps: Frames per second
        latency_ms: Algorithm latency in milliseconds
        mode: "HARDWARE" or "SYNTHETIC"
    """
    if "REPLICA" in mode:
        mode_color = COLOR_CYAN
    elif mode.startswith("HARDWARE"):
        mode_color = COLOR_GREEN
    else:
        mode_color = COLOR_YELLOW

    draw_text_with_background(img, f"FPS: {fps:.1f}", (10, 30), font_scale=0.6, thickness=2)
    draw_text_with_background(img, f"Latency: {latency_ms:.2f} ms", (10, 55), font_scale=0.6, thickness=2)
    draw_text_with_background(img, mode, (10, 80), font_scale=0.6, thickness=2, text_color=mode_color)


# ── SimpleInvoker overlay helpers ────────────────────────────────────────────

# Per-class detection colors (BGR)
_DETECTION_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (0, 255, 128),
]

# PASCAL VOC colormap for segmentation (21 classes)
_PASCAL_COLORS = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

# COCO skeleton connections for pose drawing
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Per-person colors for multi-pose
_PERSON_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 100, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def draw_classification_labels(
    img: np.ndarray,
    top_labels: List[Tuple[str, float]],
    position: Tuple[int, int] = (10, 110)
) -> None:
    """Draw ranked classification labels on the image.

    Args:
        img: Image to draw on (modified in-place)
        top_labels: List of (label, score) tuples, ranked by score
        position: (x, y) of the first label row
    """
    x, y = position
    for i, (label, score) in enumerate(top_labels):
        text = f"{i+1}. {label} ({score:.0f})"
        draw_text_with_background(img, text, (x, y + i * 22),
                                  font_scale=0.45, text_color=COLOR_GREEN)


def draw_bounding_boxes(
    img: np.ndarray,
    detections: list,
    labels: List[str],
    max_boxes: int = 10
) -> None:
    """Draw bounding boxes with class labels and scores.

    Args:
        img: Image to draw on (modified in-place)
        detections: List of (class_id, score, ymin, xmin, ymax, xmax) tuples.
            Coordinates are normalized [0, 1].
        labels: List of label strings (index-aligned, class 0 = background).
        max_boxes: Maximum number of boxes to draw.
    """
    h, w = img.shape[:2]

    for i, (cls_id, score, ymin, xmin, ymax, xmax) in enumerate(detections[:max_boxes]):
        color = _DETECTION_COLORS[cls_id % len(_DETECTION_COLORS)]
        # COCO labels are 1-indexed (class 0 = background, labels start at 1)
        label_idx = cls_id - 1
        label = labels[label_idx] if 0 <= label_idx < len(labels) else f"class {cls_id}"
        text = f"{label} {score:.0%}"

        x0 = max(0, int(xmin * w))
        y0 = max(0, int(ymin * h))
        x1 = min(w, int(xmax * w))
        y1 = min(h, int(ymax * h))

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        draw_text_with_background(img, text, (x0 + 2, y0 - 4),
                                  font_scale=0.4, text_color=COLOR_WHITE,
                                  bg_color=color)


def draw_segmentation_overlay(
    img: np.ndarray,
    seg_map: np.ndarray,
    alpha: float = 0.4
) -> None:
    """Blend a segmentation colormap onto the image.

    Args:
        img: Image to draw on (modified in-place). BGR uint8.
        seg_map: [H, W] int array of class indices (0-20 for PASCAL VOC).
        alpha: Blend strength (0 = invisible, 1 = fully opaque).
    """
    h, w = img.shape[:2]
    # Colorize: class index → RGB
    color_map_rgb = _PASCAL_COLORS[seg_map]
    # Resize to image dimensions (nearest to preserve class boundaries)
    color_map_resized = cv2.resize(color_map_rgb, (w, h),
                                   interpolation=cv2.INTER_NEAREST)
    # Convert RGB → BGR for OpenCV
    color_map_bgr = color_map_resized[:, :, ::-1]
    cv2.addWeighted(color_map_bgr, alpha, img, 1 - alpha, 0, img)


def draw_skeleton(
    img: np.ndarray,
    poses: list,
    input_size: Tuple[int, int],
    confidence_threshold: float = 0.3
) -> None:
    """Draw keypoints and skeleton edges for one or more poses.

    Args:
        img: Image to draw on (modified in-place). BGR uint8.
        poses: List of Pose objects (from posenet/multipose decoder).
            Each has .keypoints [17, 2] (y, x) and .keypoint_scores [17].
        input_size: (width, height) of the model input, for scaling.
        confidence_threshold: Minimum keypoint score to draw.
    """
    h, w = img.shape[:2]
    input_w, input_h = input_size
    scale_x = w / input_w
    scale_y = h / input_h
    radius = max(3, min(w, h) // 100)
    line_width = max(1, radius // 2)

    for p_idx, pose in enumerate(poses):
        color = _PERSON_COLORS[p_idx % len(_PERSON_COLORS)]
        kps = pose.keypoints  # [17, 2] as (y, x) in model-input pixels
        scores = pose.keypoint_scores  # [17]

        # Draw skeleton edges
        for (i, j) in _SKELETON:
            if scores[i] > confidence_threshold and scores[j] > confidence_threshold:
                pt1 = (int(kps[i, 1] * scale_x), int(kps[i, 0] * scale_y))
                pt2 = (int(kps[j, 1] * scale_x), int(kps[j, 0] * scale_y))
                cv2.line(img, pt1, pt2, color, line_width)

        # Draw keypoints
        for k in range(17):
            if scores[k] > confidence_threshold:
                px = int(kps[k, 1] * scale_x)
                py = int(kps[k, 0] * scale_y)
                cv2.circle(img, (px, py), radius, color, -1)
                cv2.circle(img, (px, py), radius, COLOR_WHITE, 1)
