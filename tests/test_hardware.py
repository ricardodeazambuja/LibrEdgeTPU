#!/usr/bin/env python3
"""Hardware tests: run inference on the Edge TPU.

Requires a USB-connected Edge TPU device.

Usage:
    pytest tests/test_hardware.py --run-hardware         # all hardware tests
    pytest tests/test_hardware.py --run-hardware -k mobilenet_v1
    pytest tests/test_hardware.py --run-hardware -k validated
"""

import time

import numpy as np
import pytest

from libredgetpu.simple_invoker import SimpleInvoker
from tests.model_zoo import get_model, get_sample_image, MODELS


# ---------------------------------------------------------------------------
# Classification / detection / segmentation / pose models
# ---------------------------------------------------------------------------

SIMPLE_MODELS = [
    "mobilenet_v1",
    "mobilenet_v2",
    "inception_v1",
    "efficientnet_s",
]

DMA_HINT_MODELS = [
    "ssd_mobiledet",
    "ssd_mobilenet_v1",
    "ssd_mobilenet_v2",
    "deeplabv3",
    "posenet",
]

ALL_MODELS = SIMPLE_MODELS + DMA_HINT_MODELS


@pytest.mark.hardware
@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_inference(model_name, n_warmup=2, n_bench=10):
    """Run inference on a model and verify non-empty output."""
    path = get_model(model_name)

    with SimpleInvoker(path) as model:
        inp = np.random.uniform(-1, 1, model.input_shape).astype(np.float32)

        # Warm up (includes param caching on first call)
        for _ in range(n_warmup):
            out = model.invoke(inp)

        assert out.size > 0, "Output is empty"

        # Verify raw path works too
        raw = model.invoke_raw(
            np.zeros(np.prod(model.input_shape), dtype=np.uint8).tobytes()
        )
        assert len(raw) > 0, "Raw output is empty"

        # Benchmark
        times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            model.invoke(inp)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = sum(times) / len(times)
        print(f"  {model_name}: avg={avg:.2f} ms  "
              f"(min={min(times):.2f}, max={max(times):.2f})")


# ---------------------------------------------------------------------------
# Validated post-processing tests
# ---------------------------------------------------------------------------

def _load_image(width, height):
    """Load the sample test image, resize, return uint8 array."""
    from PIL import Image
    img_path = get_sample_image()
    img = Image.open(img_path).convert("RGB").resize((width, height), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


@pytest.mark.hardware
@pytest.mark.validated
def test_posenet_validated():
    """PoseNet with PersonLab decoder: detect poses in Grace Hopper image."""
    from libredgetpu.postprocess.posenet_decoder import postprocess_posenet

    model_path = get_model("posenet")
    input_arr = _load_image(641, 481)

    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    with SimpleInvoker(model_path) as model:
        model.invoke_raw(input_arr.tobytes())  # warm up
        raw_outputs = model.invoke_raw_outputs(input_arr.tobytes())
        output_layers = model.output_layers

    poses = postprocess_posenet(raw_outputs, output_layers, tflite_bytes)

    assert len(poses) >= 1, "Expected at least 1 pose in Grace Hopper image"
    assert poses[0].score > 0.2, f"Top pose score too low: {poses[0].score}"

    # Check keypoints within image bounds
    valid_kps = sum(
        1 for k in range(17)
        if poses[0].keypoint_scores[k] > 0.3
        and 0 <= poses[0].keypoints[k, 0] <= 481
        and 0 <= poses[0].keypoints[k, 1] <= 641
    )
    assert valid_kps >= 5, f"Expected at least 5 valid keypoints, got {valid_kps}"


@pytest.mark.hardware
@pytest.mark.validated
def test_multipose_validated():
    """MultiPose: detect multiple people in Grace Hopper image."""
    from libredgetpu.postprocess.multipose_decoder import postprocess_multipose

    model_path = get_model("posenet_multipose")
    input_arr = _load_image(257, 257)

    # MultiPose model uses int8 input: quantize to int8, then XOR 0x80
    # for the Edge TPU's uint8 domain (the compiler adjusts the first
    # layer's zero_point by +128).
    input_int8 = (input_arr.astype(np.float32) - 127).astype(np.int8)
    input_bytes = (input_int8.view(np.uint8) ^ 0x80).tobytes()

    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    with SimpleInvoker(model_path) as model:
        model.invoke_raw(input_bytes)  # warm up
        raw_outputs = model.invoke_raw_outputs(input_bytes)
        output_layers = model.output_layers

    poses = postprocess_multipose(raw_outputs, output_layers, tflite_bytes)

    assert len(poses) >= 1, "Expected at least 1 pose in Grace Hopper image"
    assert poses[0].score > 0.1, f"Top pose score too low: {poses[0].score}"

    # Check keypoints within image bounds
    valid_kps = sum(
        1 for k in range(17)
        if poses[0].keypoint_scores[k] > 0.1
        and 0 <= poses[0].keypoints[k, 0] <= 257
        and 0 <= poses[0].keypoints[k, 1] <= 257
    )
    assert valid_kps >= 3, f"Expected at least 3 valid keypoints, got {valid_kps}"


@pytest.mark.hardware
@pytest.mark.validated
def test_deeplabv3_validated():
    """DeepLabV3 with CPU post-processing: segment Grace Hopper image."""
    from libredgetpu.postprocess.deeplabv3 import postprocess_deeplabv3

    model_path = get_model("deeplabv3")
    input_arr = _load_image(513, 513)

    with open(model_path, "rb") as f:
        tflite_bytes = f.read()

    with SimpleInvoker(model_path) as model:
        model.invoke_raw(input_arr.tobytes())  # warm up
        raw_outputs = model.invoke_raw_outputs(input_arr.tobytes())
        output_layers = model.output_layers

    seg_map = postprocess_deeplabv3(raw_outputs, output_layers, tflite_bytes)

    assert seg_map.shape == (33, 33), f"Expected (33, 33), got {seg_map.shape}"
    assert 15 in np.unique(seg_map), "Expected 'person' class (15) in segmentation"

    person_pct = 100.0 * np.sum(seg_map == 15) / seg_map.size
    assert person_pct > 5.0, f"Person coverage too low: {person_pct:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-hardware"])
