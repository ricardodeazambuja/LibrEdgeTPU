"""SSD post-processing for libredgetpu.

Decodes raw Edge TPU outputs from SSD-style detection models (e.g.
SSDLite MobileDet) into bounding boxes with class IDs and scores.

Reads anchor boxes and NMS parameters from the TFLite model's
TFLite_Detection_PostProcess custom op, then applies box decoding,
sigmoid scoring, and per-class NMS entirely in NumPy.
"""

import numpy as np

from ..tflite_parser import parse_full


def postprocess_ssd(raw_outputs, output_layers, tflite_bytes,
                    score_threshold=0.5):
    """SSD post-processing using model's own anchors and DetectionPostProcess params.

    Reads anchors and NMS parameters from the TFLite model, decodes Edge TPU
    outputs into bounding boxes with class IDs and scores.

    Args:
        raw_outputs: List of 2 raw byte arrays from Edge TPU:
            [0]: bounding box encodings (z_dim=4)
            [1]: class score logits (z_dim>4)
        output_layers: List of LayerInfo from DarwiNN executable.
        tflite_bytes: Raw bytes of the *_edgetpu.tflite model file.
        score_threshold: Minimum score for a detection (default 0.5).

    Returns:
        List of (class_id, score, ymin, xmin, ymax, xmax) tuples,
        sorted by score descending. Coordinates are normalized [0, 1].
    """
    import flatbuffers.flexbuffers as fb

    model = parse_full(tflite_bytes)

    # Extract DetectionPostProcess parameters
    params = {}
    anchor_tensor_idx = None
    for op in model.operators:
        if op.opcode_name == "TFLite_Detection_PostProcess":
            root = fb.GetRoot(op.custom_options)
            m = root.AsMap
            for k in m.Keys:
                key = k.AsKey
                try:
                    params[key] = m[key].AsFloat
                except Exception:
                    try:
                        params[key] = m[key].AsInt
                    except Exception:
                        pass
            # Anchor tensor is the 3rd input to DetectionPostProcess
            anchor_tensor_idx = op.inputs[2]
            break

    y_scale = params.get("y_scale", 10.0)
    x_scale = params.get("x_scale", 10.0)
    h_scale = params.get("h_scale", 5.0)
    w_scale = params.get("w_scale", 5.0)
    nms_iou = params.get("nms_iou_threshold", 0.6)
    nms_score = max(params.get("nms_score_threshold", 1e-8), score_threshold)
    max_dets = int(params.get("max_detections", 100))

    # Read pre-computed anchors from model buffer [N, 4] as [cy, cx, h, w]
    anchor_t = model.tensors[anchor_tensor_idx]
    anchor_buf = model.buffers[anchor_t.buffer_index]
    anchors = np.frombuffer(anchor_buf, dtype=np.float32).reshape(-1, 4)
    a_cy = anchors[:, 0]
    a_cx = anchors[:, 1]
    a_h = anchors[:, 2]
    a_w = anchors[:, 3]

    # Dequantize Edge TPU outputs.
    # Identify by z_dim (robust to DMA reordering): boxes have z=4, scores have z>4.
    box_layer, box_raw_bytes = None, None
    score_layer, score_raw_bytes = None, None
    for i, layer in enumerate(output_layers[:2]):
        if layer.z_dim == 4:
            box_layer, box_raw_bytes = layer, raw_outputs[i]
        else:
            score_layer, score_raw_bytes = layer, raw_outputs[i]

    boxes_uint8 = np.frombuffer(box_raw_bytes, dtype=np.uint8).astype(np.float32)
    boxes_raw = (boxes_uint8 - box_layer.zero_point) * box_layer.dequant_factor
    n_anchors = box_layer.y_dim * box_layer.x_dim
    box_padded_z = box_layer.size_bytes // n_anchors
    boxes_raw = boxes_raw.reshape(n_anchors, box_padded_z)[:, :4]

    scores_uint8 = np.frombuffer(score_raw_bytes, dtype=np.uint8).astype(np.float32)
    scores_raw = (scores_uint8 - score_layer.zero_point) * score_layer.dequant_factor
    n_classes = score_layer.z_dim
    score_padded_z = score_layer.size_bytes // n_anchors
    scores_raw = scores_raw.reshape(n_anchors, score_padded_z)[:, :n_classes]

    # Decode boxes: raw [dy, dx, dh, dw] -> [ymin, xmin, ymax, xmax]
    dy = boxes_raw[:, 0] / y_scale
    dx = boxes_raw[:, 1] / x_scale
    dh = boxes_raw[:, 2] / h_scale
    dw = boxes_raw[:, 3] / w_scale
    cy = dy * a_h + a_cy
    cx = dx * a_w + a_cx
    h = np.exp(dh) * a_h
    w = np.exp(dw) * a_w
    decoded = np.stack([cy - h/2, cx - w/2, cy + h/2, cx + w/2], axis=1)

    # Sigmoid scores
    probs = 1.0 / (1.0 + np.exp(-scores_raw))

    # Per-class NMS (skip background class 0)
    detections = []
    for cls in range(1, n_classes):
        cls_scores = probs[:, cls]
        mask = cls_scores > nms_score
        if not np.any(mask):
            continue
        cls_boxes = decoded[mask]
        cls_sc = cls_scores[mask]
        # NMS
        order = np.argsort(cls_sc)[::-1]
        keep = []
        while len(order) > 0 and len(keep) < max_dets:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            yy1 = np.maximum(cls_boxes[i, 0], cls_boxes[rest, 0])
            xx1 = np.maximum(cls_boxes[i, 1], cls_boxes[rest, 1])
            yy2 = np.minimum(cls_boxes[i, 2], cls_boxes[rest, 2])
            xx2 = np.minimum(cls_boxes[i, 3], cls_boxes[rest, 3])
            inter = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)
            area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
            area_r = (cls_boxes[rest, 2] - cls_boxes[rest, 0]) * (cls_boxes[rest, 3] - cls_boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-8)
            order = rest[iou < nms_iou]
        for k in keep:
            detections.append((cls, cls_sc[k], *cls_boxes[k]))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections[:max_dets]
