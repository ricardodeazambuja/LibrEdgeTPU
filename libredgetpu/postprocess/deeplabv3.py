"""DeepLabV3 CPU post-processing for libredgetpu.

The Edge TPU runs the MobileNetV2 backbone + ASPP, producing two intermediate
tensors.  The remaining 8 TFLite ops (resize, quantize, concat, two 1x1 convs,
two resizes, argmax) run on the CPU to produce the final segmentation map.

This module extracts the necessary weights and quant params from the TFLite
model and implements those ops in NumPy.
"""

import numpy as np

from ..tflite_parser import parse_full, TFLiteModelFull


# PASCAL VOC 2012 class names (21 classes)
PASCAL_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]


def _get_buffer(model: TFLiteModelFull, tensor_idx: int) -> bytes:
    """Return raw buffer bytes for a tensor."""
    buf_idx = model.tensors[tensor_idx].buffer_index
    data = model.buffers[buf_idx]
    if data is None:
        return b""
    return data


def _extract_weights(model: TFLiteModelFull):
    """Extract the two 1x1 conv weight/bias sets from the TFLite graph.

    Walks the operator list to find the two CONV_2D ops that run on CPU
    (after the edgetpu-custom-op), then reads their weights and biases
    from the model buffers.

    Returns (w1, b1, w1_in_scale, w2, b2, w2_in_scale) where:
      w1: [out_ch1, in_ch1] float32 weight matrix for concat_projection
      b1: [out_ch1] float32 bias
      w1_in_scale: input scale for the first conv (ASPP output scale)
      w2: [out_ch2, in_ch2] float32 weight matrix for logits
      b2: [out_ch2] float32 bias
      w2_in_scale: input scale for the second conv (projection output scale)
    """
    conv_ops = []
    for op in model.operators:
        if op.opcode_name == "CONV_2D":
            conv_ops.append(op)

    if len(conv_ops) < 2:
        raise ValueError(
            f"Expected at least 2 CONV_2D ops in DeepLabV3, found {len(conv_ops)}"
        )

    results = []
    for conv in conv_ops[:2]:
        # CONV_2D inputs: [input, weights, bias]
        input_idx = conv.inputs[0]
        weight_idx = conv.inputs[1]
        bias_idx = conv.inputs[2]
        output_idx = conv.outputs[0]

        # Weight tensor
        wt = model.tensors[weight_idx]
        w_raw = np.frombuffer(_get_buffer(model, weight_idx), dtype=np.uint8)
        w_shape = wt.shape  # [out_ch, 1, 1, in_ch] for 1x1 conv
        out_ch = w_shape[0]
        in_ch = w_shape[-1]
        w = (w_raw.reshape(out_ch, in_ch).astype(np.float32) - wt.zero_point) * wt.scale

        # Bias tensor (int32)
        bt = model.tensors[bias_idx]
        b_raw = np.frombuffer(_get_buffer(model, bias_idx), dtype=np.int32)
        # Bias scale = input_scale * weight_scale
        inp_t = model.tensors[input_idx]
        b = b_raw.astype(np.float32) * (inp_t.scale * wt.scale)

        # Output scale/zp for requantization
        out_t = model.tensors[output_idx]

        results.append((w, b, inp_t.scale, out_t.scale, out_t.zero_point))

    w1, b1, w1_in_sc, _, _ = results[0]
    w2, b2, w2_in_sc, _, _ = results[1]
    return w1, b1, w1_in_sc, w2, b2, w2_in_sc


def postprocess_deeplabv3(raw_outputs, output_layers, tflite_bytes):
    """Run CPU post-processing for DeepLabV3.

    Args:
        raw_outputs: List of 2 raw byte arrays from Edge TPU:
            [0]: ASPP features (e.g. [33, 33, 256])
            [1]: Image pooling (e.g. [1, 1, 256])
        output_layers: List of LayerInfo from DarwiNN executable.
        tflite_bytes: Raw bytes of the *_edgetpu.tflite model file.

    Returns:
        seg_map: int64 numpy array of shape [H, H] with class indices (0-20).
                 H is the ASPP spatial size (typically 33).
    """
    from ..delegate import relayout_output

    model = parse_full(tflite_bytes)

    # De-scatter and dequantize output tensors.
    # Identify by spatial size (robust to DMA reordering):
    #   ASPP has large spatial dims (e.g. 33x33), pooling has 1x1.
    aspp = None
    pool = None
    for i, layer in enumerate(output_layers[:2]):
        arr = relayout_output(raw_outputs[i], layer).astype(np.float32)
        arr = (arr - layer.zero_point) * layer.dequant_factor
        if layer.y_dim == 1 and layer.x_dim == 1:
            pool = arr
        else:
            aspp = arr

    if aspp is None or pool is None:
        raise ValueError(
            "Could not identify DeepLabV3 output tensors by spatial dims. "
            f"Got dims: {[(l.y_dim, l.x_dim) for l in output_layers[:2]]}"
        )

    h, w, _ = aspp.shape  # typically 33, 33

    # 1. Resize image pooling from 1x1 to HxW (nearest/tile)
    pool_resized = np.tile(pool, (h, w, 1))

    # 2. Concat: [pool_resized, ASPP] -> [H, W, 512]
    concat = np.concatenate([pool_resized, aspp], axis=2)

    # 3-4. Extract and apply CPU conv weights
    w1, b1, _, w2, b2, _ = _extract_weights(model)

    # 1x1 conv: concat_projection [H, W, 512] -> [H, W, 256] + ReLU6
    in_ch1 = w1.shape[1]
    proj = concat.reshape(-1, in_ch1) @ w1.T + b1
    proj = np.clip(proj, 0, 6)  # ReLU6

    # 1x1 conv: logits [H, W, 256] -> [H, W, 21]
    in_ch2 = w2.shape[1]
    logits = proj.reshape(-1, in_ch2) @ w2.T + b2
    logits = logits.reshape(h, w, -1)

    # 5. Argmax -> class map
    seg_map = np.argmax(logits, axis=2)

    return seg_map
