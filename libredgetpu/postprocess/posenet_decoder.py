"""PoseNet PersonLab decoder for libredgetpu.

Pure NumPy port of the C++ decoder from google-coral/edgetpu:
  edgetpu/src/cpp/posenet/posenet_decoder.cc
  edgetpu/src/cpp/posenet/posenet_decoder_op.cc

Decodes raw Edge TPU heatmap/offset tensors into pose keypoints using the
PersonLab algorithm (Papandreou et al., arXiv:1803.08225).  The decoder
algorithm supports multi-person detection, but the shipped PoseNet model
(posenet_mobilenet_v1_075_*_decoder) is a **single-person** model and will
reliably detect only one pose per image.

The PoseNet model outputs 3 tensors from the Edge TPU:
  0: heatmaps    [1, H, W, 17]   — keypoint score logits
  1: short_offsets [1, H, W, 34]  — sub-pixel refinement (y,x per keypoint)
  2: mid_offsets   [1, H, W, 64]  — inter-keypoint displacements

The PosenetDecoderOp custom op parameters (max_detections, score_threshold,
stride, nms_radius) are stored as FlexBuffers in the TFLite custom_options.
"""

import heapq
from dataclasses import dataclass
from typing import List

import numpy as np


NUM_KEYPOINTS = 17
NUM_EDGES = 16

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 32 edges: 16 forward + 16 backward (matches C++ kEdgeList)
EDGE_LIST = [
    # Forward edges
    (0, 1), (1, 3), (0, 2), (2, 4),
    (0, 5), (5, 7), (7, 9), (5, 11),
    (11, 13), (13, 15), (0, 6), (6, 8),
    (8, 10), (6, 12), (12, 14), (14, 16),
    # Backward edges
    (1, 0), (3, 1), (2, 0), (4, 2),
    (5, 0), (7, 5), (9, 7), (11, 5),
    (13, 11), (15, 13), (6, 0), (8, 6),
    (10, 8), (12, 6), (14, 12), (16, 14),
]

# Skeleton connections for visualization
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class Pose:
    """A detected pose with keypoint positions and scores."""
    keypoints: np.ndarray    # [17, 2] float32 (y, x) in pixel coordinates
    keypoint_scores: np.ndarray  # [17] float32 in [0, 1]
    score: float             # instance-level score


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _logodds(x):
    return -np.log(1.0 / (x + 1e-6) - 1.0)


def _build_adjacency_list():
    """Build adjacency list from edge list. Returns (child_ids, edge_ids)."""
    child_ids = [[] for _ in range(NUM_KEYPOINTS)]
    edge_ids = [[] for _ in range(NUM_KEYPOINTS)]
    for k, (parent, child) in enumerate(EDGE_LIST):
        child_ids[parent].append(child)
        edge_ids[parent].append(k)
    return child_ids, edge_ids


def _sample_bilinear(tensor, y, x, channels):
    """Bilinear interpolation of tensor[y, x, channels].

    tensor: [H, W, C]
    y, x: float coordinates
    channels: list of channel indices to sample

    Returns: array of sampled values (one per channel).
    """
    h, w, _ = tensor.shape
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)

    y_floor = int(np.floor(y))
    y_ceil = min(int(np.ceil(y)), h - 1)
    x_floor = int(np.floor(x))
    x_ceil = min(int(np.ceil(x)), w - 1)
    y_lerp = y - y_floor
    x_lerp = x - x_floor

    result = np.empty(len(channels), dtype=np.float32)
    for i, c in enumerate(channels):
        top_left = tensor[y_floor, x_floor, c]
        top_right = tensor[y_floor, x_ceil, c]
        bottom_left = tensor[y_ceil, x_floor, c]
        bottom_right = tensor[y_ceil, x_ceil, c]
        top = (1 - x_lerp) * top_left + x_lerp * top_right
        bottom = (1 - x_lerp) * bottom_left + x_lerp * bottom_right
        result[i] = (1 - y_lerp) * top + y_lerp * bottom
    return result


def _find_displaced_position(short_offsets, mid_offsets, height, width,
                              source_y, source_x, edge_id, target_id,
                              refinement_steps):
    """Follow mid-range offsets, then refine with short-range offsets."""
    y, x = source_y, source_x

    # Follow mid-range offset
    # mid_offsets layout: [H, W, 4*NUM_EDGES] = [fwd_y, fwd_x, bwd_y, bwd_x]
    offsets = _sample_bilinear(mid_offsets, y, x,
                                [edge_id, NUM_EDGES + edge_id])
    y = np.clip(y + offsets[0], 0, height - 1)
    x = np.clip(x + offsets[1], 0, width - 1)

    # Refine with short-range offsets
    for _ in range(refinement_steps):
        offsets = _sample_bilinear(short_offsets, y, x,
                                    [target_id, NUM_KEYPOINTS + target_id])
        y = np.clip(y + offsets[0], 0, height - 1)
        x = np.clip(x + offsets[1], 0, width - 1)

    return y, x


def _build_keypoint_queue(scores, short_offsets, height, width,
                           score_threshold, local_max_radius=1):
    """Find local maxima in score maps and build a max-heap of candidates.

    Returns a max-heap of (-score, y_refined, x_refined, keypoint_id).
    (Negated scores because heapq is a min-heap.)
    """
    queue = []
    for y in range(height):
        for x in range(width):
            for k in range(NUM_KEYPOINTS):
                score = scores[y, x, k]
                if score < score_threshold:
                    continue

                # Check local maximum
                is_max = True
                y_start = max(y - local_max_radius, 0)
                y_end = min(y + local_max_radius + 1, height)
                x_start = max(x - local_max_radius, 0)
                x_end = min(x + local_max_radius + 1, width)
                for yy in range(y_start, y_end):
                    for xx in range(x_start, x_end):
                        if scores[yy, xx, k] > score:
                            is_max = False
                            break
                    if not is_max:
                        break

                if is_max:
                    dy = short_offsets[y, x, k]
                    dx = short_offsets[y, x, NUM_KEYPOINTS + k]
                    y_ref = np.clip(y + dy, 0, height - 1)
                    x_ref = np.clip(x + dx, 0, width - 1)
                    heapq.heappush(queue, (-score, y_ref, x_ref, k))

    return queue


def _backtrack_decode_pose(scores, short_offsets, mid_offsets, height, width,
                            root_y, root_x, root_id, adjacency,
                            refinement_steps):
    """Decode a full pose starting from a root keypoint via graph traversal."""
    child_ids, edge_ids = adjacency

    keypoints = np.full((NUM_KEYPOINTS, 2), -1.0, dtype=np.float32)
    keypoint_scores = np.full(NUM_KEYPOINTS, -1e5, dtype=np.float32)
    decoded = [False] * NUM_KEYPOINTS

    # Priority queue: (-score, y, x, keypoint_id)
    root_score = _sample_bilinear(scores, root_y, root_x, [root_id])[0]
    pq = [(-root_score, root_y, root_x, root_id)]

    while pq:
        neg_score, y, x, kid = heapq.heappop(pq)
        if decoded[kid]:
            continue

        keypoints[kid] = [y, x]
        keypoint_scores[kid] = -neg_score
        decoded[kid] = True

        # Traverse to children
        for child_id, eid in zip(child_ids[kid], edge_ids[kid]):
            if decoded[child_id]:
                continue

            # Adjust edge_id for backward edges (>= NUM_EDGES -> add NUM_EDGES)
            actual_eid = eid
            if eid >= NUM_EDGES:
                actual_eid += NUM_EDGES

            cy, cx = _find_displaced_position(
                short_offsets, mid_offsets, height, width,
                y, x, actual_eid, child_id, refinement_steps,
            )
            child_score = _sample_bilinear(scores, cy, cx, [child_id])[0]
            heapq.heappush(pq, (-child_score, cy, cx, child_id))

    return keypoints, keypoint_scores


def _pass_keypoint_nms(poses, keypoint, kid, squared_nms_radius):
    """Check if a keypoint passes NMS against all existing poses."""
    for pose_kps in poses:
        dy = keypoint[0] - pose_kps[kid, 0]
        dx = keypoint[1] - pose_kps[kid, 1]
        if dy * dy + dx * dx <= squared_nms_radius:
            return False
    return True


def _perform_soft_keypoint_nms(all_keypoints, all_keypoint_scores,
                                 decreasing_indices, squared_nms_radius,
                                 topk):
    """Soft keypoint NMS: reduce instance scores for overlapping keypoints."""
    n = len(decreasing_indices)
    instance_scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ci = decreasing_indices[i]
        # Find occluded keypoints (overlap with higher-scoring instances)
        occluded = np.zeros(NUM_KEYPOINTS, dtype=bool)
        for j in range(i):
            cj = decreasing_indices[j]
            for k in range(NUM_KEYPOINTS):
                dy = all_keypoints[ci, k, 0] - all_keypoints[cj, k, 0]
                dx = all_keypoints[ci, k, 1] - all_keypoints[cj, k, 1]
                if dy * dy + dx * dx <= squared_nms_radius:
                    occluded[k] = True

        # Score = average of top-k non-occluded keypoint scores
        sorted_indices = np.argsort(all_keypoint_scores[ci])[::-1]
        total = 0.0
        for ki in range(topk):
            idx = sorted_indices[ki]
            if not occluded[idx]:
                total += all_keypoint_scores[ci, idx]
        instance_scores[i] = total / max(topk, 1)

    return instance_scores


def decode_poses(heatmaps, short_offsets, mid_offsets,
                 max_detections=10, score_threshold=0.5,
                 nms_radius=20.0, stride=16,
                 mid_short_offset_refinement_steps=5) -> List[Pose]:
    """Decode poses from raw Edge TPU output tensors.

    All three input tensors should be dequantized float32 arrays with the
    batch dimension squeezed:
      heatmaps:      [H, W, 17]  — logit scores (pre-sigmoid)
      short_offsets: [H, W, 34]  — in block space (divided by stride)
      mid_offsets:   [H, W, 64]  — in block space (divided by stride)

    Args:
        max_detections: Maximum number of poses to detect.
        score_threshold: Minimum pose score (post-sigmoid).
        nms_radius: NMS radius in pixels (will be scaled to block space).
        stride: Network stride (typically 16).
        mid_short_offset_refinement_steps: Refinement iterations (typically 5).

    Returns:
        List of Pose objects sorted by decreasing score.
    """
    height, width, _ = heatmaps.shape
    nms_radius_block = nms_radius / stride
    squared_nms = nms_radius_block * nms_radius_block

    # Build keypoint candidate queue (logit threshold)
    min_score_logit = _logodds(score_threshold)
    queue = _build_keypoint_queue(
        heatmaps, short_offsets, height, width,
        min_score_logit, local_max_radius=1,
    )

    adjacency = _build_adjacency_list()

    all_keypoints = []
    all_keypoint_scores = []
    all_instance_scores = []

    while len(all_keypoints) < max_detections and queue:
        neg_score, ry, rx, rid = heapq.heappop(queue)

        # NMS: reject if too close to same keypoint in existing pose
        if not _pass_keypoint_nms(all_keypoints, np.array([ry, rx]), rid,
                                   squared_nms):
            continue

        # Decode full pose from this root
        kps, kp_scores = _backtrack_decode_pose(
            heatmaps, short_offsets, mid_offsets, height, width,
            ry, rx, rid, adjacency, mid_short_offset_refinement_steps,
        )

        # Convert scores from logits to probabilities
        kp_scores_prob = _sigmoid(kp_scores)

        # Instance score = average of all keypoint scores
        topk = NUM_KEYPOINTS
        sorted_idx = np.argsort(kp_scores_prob)[::-1]
        instance_score = np.mean(kp_scores_prob[sorted_idx[:topk]])

        if instance_score >= score_threshold:
            all_keypoints.append(kps)
            all_keypoint_scores.append(kp_scores_prob)
            all_instance_scores.append(instance_score)

    if not all_keypoints:
        return []

    all_keypoints = np.array(all_keypoints)
    all_keypoint_scores = np.array(all_keypoint_scores)

    # Sort by decreasing instance score
    decreasing_indices = np.argsort(all_instance_scores)[::-1].tolist()

    # Soft keypoint NMS and rescoring
    final_scores = _perform_soft_keypoint_nms(
        all_keypoints, all_keypoint_scores,
        decreasing_indices, squared_nms, NUM_KEYPOINTS,
    )

    # Re-sort by final scores
    order = np.argsort(final_scores)[::-1]

    poses = []
    for idx in order:
        if final_scores[idx] < score_threshold:
            break
        orig_idx = decreasing_indices[idx]
        # Scale keypoints from block space to pixel space
        pixel_kps = all_keypoints[orig_idx].copy()
        pixel_kps[:, 0] *= stride  # y
        pixel_kps[:, 1] *= stride  # x
        poses.append(Pose(
            keypoints=pixel_kps,
            keypoint_scores=all_keypoint_scores[orig_idx],
            score=float(final_scores[idx]),
        ))

    return poses


def _parse_flexbuffer_params(custom_options):
    """Parse PosenetDecoderOp parameters from FlexBuffer custom_options.

    Falls back to default values if parsing fails.

    Returns dict with keys: max_detections, score_threshold, stride, nms_radius.
    """
    defaults = {
        "max_detections": 10,
        "score_threshold": 0.5,
        "stride": 16,
        "nms_radius": 20.0,
    }

    if custom_options is None:
        return defaults

    try:
        from flatbuffers.flexbuffers import GetRoot
        root = GetRoot(custom_options)
        if not root.IsMap:
            return defaults
        fb_map = root.AsMap
        keys = fb_map.Keys
        vals = fb_map.Values
        parsed = {}
        for i in range(len(keys)):
            k = keys[i]
            key_str = k.AsKey if k.IsKey else (k.AsString if k.IsString else None)
            if key_str is None:
                continue
            v = vals[i]
            if v.IsFloat:
                parsed[key_str] = v.AsFloat
            elif v.IsInt:
                parsed[key_str] = v.AsInt
        for key in defaults:
            if key in parsed:
                defaults[key] = parsed[key]
    except (ImportError, Exception):
        pass

    return defaults


def postprocess_posenet(raw_outputs, output_layers, tflite_bytes) -> List[Pose]:
    """Full PoseNet post-processing pipeline.

    Args:
        raw_outputs: List of 3 raw byte arrays from Edge TPU:
            [0]: heatmaps    [H, W, 17]
            [1]: short_offsets [H, W, 34]
            [2]: mid_offsets   [H, W, 64]
        output_layers: List of LayerInfo from DarwiNN executable.
        tflite_bytes: Raw bytes of the *_edgetpu.tflite model file.

    Returns:
        List of Pose objects with keypoints in pixel coordinates.
    """
    from ..delegate import relayout_output
    from ..tflite_parser import parse_full

    model = parse_full(tflite_bytes)

    # Find PosenetDecoderOp to get parameters
    params = None
    for op in model.operators:
        if op.opcode_name == "PosenetDecoderOp":
            params = _parse_flexbuffer_params(op.custom_options)
            break

    if params is None:
        # Fallback: use defaults
        params = _parse_flexbuffer_params(None)

    stride = params["stride"]

    # De-scatter and dequantize the 3 output tensors.
    # Identify by channel count (robust to DMA reordering):
    #   17 channels = heatmaps, 34 = short_offsets, 64 = mid_offsets
    heatmaps = None       # [H, W, 17]
    short_offsets = None   # [H, W, 34]
    mid_offsets = None     # [H, W, 64]

    for i, layer in enumerate(output_layers[:3]):
        arr = relayout_output(raw_outputs[i], layer).astype(np.float32)
        arr = (arr - layer.zero_point) * layer.dequant_factor
        if layer.z_dim == 17:
            heatmaps = arr
        elif layer.z_dim == 34:
            short_offsets = arr
        elif layer.z_dim == 64:
            mid_offsets = arr

    if heatmaps is None or short_offsets is None or mid_offsets is None:
        raise ValueError(
            "Could not identify PoseNet output tensors by channel count. "
            f"Got z_dims: {[l.z_dim for l in output_layers[:3]]}"
        )

    # Scale offsets from pixel space to block space (matches C++ DequantizeTensor
    # with extra_scale = 1.0/stride)
    short_offsets = short_offsets / stride
    mid_offsets = mid_offsets / stride

    return decode_poses(
        heatmaps, short_offsets, mid_offsets,
        max_detections=params["max_detections"],
        score_threshold=params["score_threshold"],
        nms_radius=params["nms_radius"],
        stride=stride,
        mid_short_offset_refinement_steps=5,
    )
