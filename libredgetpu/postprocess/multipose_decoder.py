"""Multi-person pose estimation decoder for libredgetpu.

Adapted from the JavaScript-based multi-pose PoseNet decoder:
  https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5

Works with the MobileNet V1 0.50 stride-16 multi-pose model (4 raw output
tensors, no PosenetDecoderOp custom op):
  0: heatmaps           [1, H, W, 17]   — keypoint score logits
  1: short_offsets       [1, H, W, 34]   — sub-pixel refinement (y,x per keypoint)
  2: displacement_fwd    [1, H, W, 32]   — forward displacement (16 edges)
  3: displacement_bwd    [1, H, W, 32]   — backward displacement (16 edges)

Requires scipy (optional dependency: ``pip install libredgetpu[multipose]``).
"""

import hashlib
import os
import ssl
import urllib.request
from typing import List

import numpy as np

from .posenet_decoder import Pose, NUM_KEYPOINTS, NUM_EDGES, SKELETON  # noqa: F401

# Model download constants
_MODEL_FILENAME = "downloadedModels_mobilenet_float_050_model-stride16_edgetpu.tflite"
_MODEL_URL = (
    "https://raw.githubusercontent.com/ricardodeazambuja/MultiPose-EdgeTPU-RPI0/main/"
    + _MODEL_FILENAME
)
_MODEL_SHA256 = "2eee7b608187fb9211606f55e913ae60589bc8bf27892108cb19de020d7524b1"

# Pose chain: parent → child edges for traversal
_POSE_CHAIN = [
    ("nose", "leftEye"), ("leftEye", "leftEar"),
    ("nose", "rightEye"), ("rightEye", "rightEar"),
    ("nose", "leftShoulder"), ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"), ("leftShoulder", "leftHip"),
    ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"),
    ("nose", "rightShoulder"), ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"), ("rightShoulder", "rightHip"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
]

# Map camelCase part names to keypoint indices
_PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar",
    "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftHip", "rightHip",
    "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
]
_PART_IDS = {name: i for i, name in enumerate(_PART_NAMES)}

# Pre-compute parent/child edge arrays
_PARENT_CHILD_TUPLES = [
    (_PART_IDS[parent], _PART_IDS[child]) for parent, child in _POSE_CHAIN
]
_PARENT_TO_CHILD_EDGES = [t[1] for t in _PARENT_CHILD_TUPLES]
_CHILD_TO_PARENT_EDGES = [t[0] for t in _PARENT_CHILD_TUPLES]


def get_multipose_model() -> str:
    """Return path to the multi-pose Edge TPU model, downloading if needed.

    The model (~292 KB) is cached in the ``multipose_model/`` subdirectory
    alongside this module.  SHA256 is verified on download.

    Returns:
        Absolute path to the ``*_edgetpu.tflite`` model file.

    Raises:
        FileNotFoundError: If download fails (includes URL and expected path).
    """
    pkg_dir = os.path.dirname(__file__)
    model_dir = os.path.join(pkg_dir, "multipose_model")
    model_path = os.path.join(model_dir, _MODEL_FILENAME)

    if os.path.isfile(model_path):
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    tmp_path = model_path + ".tmp"

    print(f"MultiPose model not found. Downloading from GitHub...")
    try:
        ctx = ssl.create_default_context()
        resp = urllib.request.urlopen(_MODEL_URL, timeout=30, context=ctx)
        data = resp.read()

        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != _MODEL_SHA256:
            raise RuntimeError(
                f"SHA256 mismatch: expected {_MODEL_SHA256}, got {actual_hash}"
            )

        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, model_path)
        size_kb = len(data) / 1024
        print(f"  Saved to {model_path} ({size_kb:.0f} KB)")
        return model_path

    except Exception as e:
        # Clean up partial download
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise FileNotFoundError(
            f"Failed to download MultiPose model: {e}\n"
            f"  URL: {_MODEL_URL}\n"
            f"  Expected path: {model_path}"
        ) from e


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _build_part_score_queue(scores, threshold, radius=1):
    """Find local maxima in each keypoint heatmap using scipy max filter.

    Args:
        scores: [H, W, 17] float32, post-sigmoid score maps.
        threshold: Minimum score for a candidate.
        radius: Local maximum filter radius (default 1).

    Returns:
        List of (score, heatmap_y, heatmap_x, keypoint_id) sorted by score
        descending.
    """
    try:
        from scipy.ndimage import maximum_filter
    except ImportError:
        raise ImportError(
            "scipy is required for multi-pose estimation.\n"
            "Install it with: pip install scipy\n"
            "Or: pip install libredgetpu[multipose]"
        )

    parts = []
    lmd = 2 * radius + 1

    for kid in range(NUM_KEYPOINTS):
        kp_scores = scores[:, :, kid].copy()
        kp_scores[kp_scores < threshold] = 0.0

        max_vals = maximum_filter(kp_scores, size=lmd, mode="constant")

        max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)
        ys, xs = max_loc.nonzero()
        for y, x in zip(ys, xs):
            parts.append((scores[y, x, kid], y, x, kid))

    # Sort by score descending
    parts.sort(key=lambda p: p[0], reverse=True)
    return parts


def _get_image_coords(hy, hx, kid, stride, offsets):
    """Convert heatmap coordinates to image coordinates using offsets.

    Offset channel layout is grouped: offsets[y, x, k] for y-offset,
    offsets[y, x, k + 17] for x-offset.
    """
    return (
        hy * stride + offsets[hy, hx, kid],
        hx * stride + offsets[hy, hx, kid + NUM_KEYPOINTS],
    )


def _get_strided_index(y, x, stride, height, width):
    """Convert image coordinates to clamped heatmap indices."""
    return (
        int(max(0, min(y / stride, height - 1))),
        int(max(0, min(x / stride, width - 1))),
    )


def _traverse_to_target_keypoint(edge_id, source_pos, target_id, scores,
                                  offsets, stride, displacements,
                                  height, width, refine_steps=2):
    """Follow displacement from source keypoint to target keypoint.

    Displacement layout is grouped: disp[y, x, edge_id] for y-displacement,
    disp[y, x, num_edges + edge_id] for x-displacement.

    Args:
        edge_id: Index into the displacement tensor (0..15).
        source_pos: (y, x) position of the source keypoint in image coords.
        target_id: Keypoint index of the target.
        scores: [H, W, 17] score maps (post-sigmoid).
        offsets: [H, W, 34] offset maps.
        stride: Output stride of the model (16).
        displacements: [H, W, 32] displacement maps (fwd or bwd).
        height, width: Heatmap spatial dimensions.
        refine_steps: Number of offset refinement iterations (default 2).

    Returns:
        (target_pos_y, target_pos_x, target_score)
    """
    src_y, src_x = source_pos
    num_edges = displacements.shape[2] // 2

    # Get strided index of source position
    sy, sx = _get_strided_index(src_y, src_x, stride, height, width)

    # Follow displacement
    dy = displacements[sy, sx, edge_id]
    dx = displacements[sy, sx, num_edges + edge_id]

    target_y = src_y + dy
    target_x = src_x + dx

    # Refine with offsets
    for _ in range(refine_steps):
        ty, tx = _get_strided_index(target_y, target_x, stride, height, width)
        off_y = offsets[ty, tx, target_id]
        off_x = offsets[ty, tx, target_id + NUM_KEYPOINTS]
        target_y = ty * stride + off_y
        target_x = tx * stride + off_x

    # Score at final position
    fy, fx = _get_strided_index(target_y, target_x, stride, height, width)
    score = scores[fy, fx, target_id]

    return target_y, target_x, score


def _decode_pose(root_score, root_y, root_x, root_kid, scores, offsets,
                 stride, disp_fwd, disp_bwd):
    """Decode a full pose by traversing edges from a root keypoint.

    Uses explicit backward-then-forward traversal (not priority queue)
    matching the original JS implementation.

    Returns:
        keypoints: [17, 2] float32 (y, x) in image pixel coordinates.
        keypoint_scores: [17] float32 in [0, 1].
    """
    height, width = scores.shape[:2]
    num_edges = len(_PARENT_TO_CHILD_EDGES)

    keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    keypoint_scores = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    decoded = [False] * NUM_KEYPOINTS

    # Set root keypoint
    keypoints[root_kid] = [root_y, root_x]
    keypoint_scores[root_kid] = root_score
    decoded[root_kid] = True

    # Backward edges: traverse from child → parent
    for edge in range(num_edges - 1, -1, -1):
        source_kid = _PARENT_TO_CHILD_EDGES[edge]
        target_kid = _CHILD_TO_PARENT_EDGES[edge]
        if decoded[source_kid] and not decoded[target_kid]:
            src_pos = (keypoints[source_kid, 0], keypoints[source_kid, 1])
            ty, tx, tscore = _traverse_to_target_keypoint(
                edge, src_pos, target_kid, scores, offsets, stride,
                disp_bwd, height, width,
            )
            keypoints[target_kid] = [ty, tx]
            keypoint_scores[target_kid] = tscore
            decoded[target_kid] = True

    # Forward edges: traverse from parent → child
    for edge in range(num_edges):
        source_kid = _CHILD_TO_PARENT_EDGES[edge]
        target_kid = _PARENT_TO_CHILD_EDGES[edge]
        if decoded[source_kid] and not decoded[target_kid]:
            src_pos = (keypoints[source_kid, 0], keypoints[source_kid, 1])
            ty, tx, tscore = _traverse_to_target_keypoint(
                edge, src_pos, target_kid, scores, offsets, stride,
                disp_fwd, height, width,
            )
            keypoints[target_kid] = [ty, tx]
            keypoint_scores[target_kid] = tscore
            decoded[target_kid] = True

    return keypoints, keypoint_scores


def _within_nms_radius(poses, sq_radius, y, x, kid):
    """Check if (y, x) is within NMS radius of the same keypoint in any pose."""
    for kps, _ in poses:
        dy = y - kps[kid, 0]
        dx = x - kps[kid, 1]
        if dy * dy + dx * dx <= sq_radius:
            return True
    return False


def decode_multiple_poses(heatmaps, offsets, disp_fwd, disp_bwd,
                          stride=16, max_pose_detections=10,
                          score_threshold=0.25, nms_radius=20):
    """Decode multiple poses from raw model output tensors.

    All four input tensors should be dequantized float32 arrays with the
    batch dimension squeezed:
      heatmaps:  [H, W, 17]  — keypoint score logits (pre-sigmoid)
      offsets:   [H, W, 34]  — sub-pixel refinement
      disp_fwd:  [H, W, 32]  — forward displacements (16 edges)
      disp_bwd:  [H, W, 32]  — backward displacements (16 edges)

    Args:
        stride: Network output stride (16 for this model).
        max_pose_detections: Maximum number of poses to return.
        score_threshold: Minimum keypoint score (post-sigmoid).
        nms_radius: NMS radius in pixels.

    Returns:
        List of Pose objects sorted by decreasing instance score.
    """
    # Apply sigmoid to heatmaps
    scores = _sigmoid(heatmaps)

    # Build candidate queue from local maxima
    queue = _build_part_score_queue(scores, score_threshold)

    sq_nms_radius = nms_radius * nms_radius
    poses = []  # list of (keypoints, keypoint_scores)

    while len(poses) < max_pose_detections and queue:
        root_score, hy, hx, kid = queue.pop(0)

        # Get image coordinates for the root keypoint
        root_y, root_x = _get_image_coords(hy, hx, kid, stride, offsets)

        # NMS: skip if too close to same keypoint in an existing pose
        if _within_nms_radius(poses, sq_nms_radius, root_y, root_x, kid):
            continue

        # Decode full pose from this root
        kps, kp_scores = _decode_pose(
            root_score, root_y, root_x, kid,
            scores, offsets, stride, disp_fwd, disp_bwd,
        )

        # Instance score = mean of all keypoint scores
        instance_score = float(np.mean(kp_scores))
        poses.append((kps, kp_scores))

    # Convert to Pose objects
    result = []
    for kps, kp_scores in poses:
        instance_score = float(np.mean(kp_scores))
        result.append(Pose(
            keypoints=kps,
            keypoint_scores=kp_scores,
            score=instance_score,
        ))

    # Sort by score descending
    result.sort(key=lambda p: p.score, reverse=True)
    return result


def postprocess_multipose(raw_outputs, output_layers, tflite_bytes,
                          stride=16, max_pose_detections=10,
                          score_threshold=0.25, nms_radius=20) -> List[Pose]:
    """Full multi-pose post-processing pipeline.

    Dequantizes the 4 raw Edge TPU output tensors and runs the multi-pose
    decoder.  Tensors are identified by DarwiNN layer name (``heatmap``,
    ``offset``, ``displacement_fwd``, ``displacement_bwd``).
    Falls back to channel-count identification if names are unavailable.

    **Input preprocessing for INT8 models**: This model uses INT8 input.
    Convert uint8 pixel data to the Edge TPU's uint8 domain::

        int8_vals = (pixels.astype(np.float32) - 127).astype(np.int8)
        input_bytes = (int8_vals.view(np.uint8) ^ 0x80).tobytes()

    Args:
        raw_outputs: List of 4 raw byte arrays from Edge TPU.
        output_layers: List of LayerInfo from DarwiNN executable.
        tflite_bytes: Raw bytes of the *_edgetpu.tflite model file.
        stride: Network output stride (default 16).
        max_pose_detections: Maximum poses (default 10).
        score_threshold: Minimum keypoint score (default 0.25).
        nms_radius: NMS radius in pixels (default 20).

    Returns:
        List of Pose objects with keypoints in pixel coordinates.
    """
    from ..delegate import relayout_output

    # Dequantize all output tensors and identify by DarwiNN layer name.
    # The output_layers order matches the physical DMA byte order (not
    # necessarily the TFLite graph_outputs order).
    heatmaps = None
    offsets = None
    disp_fwd = None
    disp_bwd = None

    for i, layer in enumerate(output_layers):
        arr = relayout_output(raw_outputs[i], layer).astype(np.float32)
        arr = (arr - layer.zero_point) * layer.dequant_factor

        lname = layer.name.lower() if layer.name else ""
        if "heatmap" in lname:
            heatmaps = arr
        elif "displacement_fwd" in lname:
            disp_fwd = arr
        elif "displacement_bwd" in lname:
            disp_bwd = arr
        elif "offset" in lname:
            offsets = arr

    # Fallback: identify by channel count if names are unavailable
    if any(t is None for t in [heatmaps, offsets, disp_fwd, disp_bwd]):
        disp_32 = []
        for i, layer in enumerate(output_layers):
            arr = relayout_output(raw_outputs[i], layer).astype(np.float32)
            arr = (arr - layer.zero_point) * layer.dequant_factor
            channels = arr.shape[-1] if arr.ndim >= 2 else 0
            if channels == NUM_KEYPOINTS and heatmaps is None:
                heatmaps = arr
            elif channels == 2 * NUM_KEYPOINTS and offsets is None:
                offsets = arr
            elif channels == 2 * NUM_EDGES:
                disp_32.append(arr)
        if disp_fwd is None and len(disp_32) >= 1:
            disp_fwd = disp_32[0]
        if disp_bwd is None and len(disp_32) >= 2:
            disp_bwd = disp_32[1]

    if heatmaps is None:
        raise ValueError("Could not find heatmap tensor in outputs")
    if offsets is None:
        raise ValueError("Could not find offset tensor in outputs")
    if disp_fwd is None or disp_bwd is None:
        raise ValueError("Could not find both displacement tensors in outputs")

    return decode_multiple_poses(
        heatmaps, offsets, disp_fwd, disp_bwd,
        stride=stride,
        max_pose_detections=max_pose_detections,
        score_threshold=score_threshold,
        nms_radius=nms_radius,
    )
