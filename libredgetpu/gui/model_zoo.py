"""Model registry and download helper for the SimpleInvoker GUI mode.

Provides a curated set of Edge TPU models with known post-processing pipelines.
Models and labels are downloaded from the EdgeTPUModelZoo and cached in
``~/.cache/libredgetpu/models/``.
"""

import os
import urllib.request

_BASE_URL = (
    "https://raw.githubusercontent.com/ricardodeazambuja/EdgeTPUModelZoo/master"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "libredgetpu", "models")

# Registry: display name -> metadata dict
MODEL_REGISTRY = {
    "Classification (MobileNet V1)": {
        "zoo_dir": "mobilenet_v1_1.0_224_quant",
        "zoo_file": "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
        "input_size": (224, 224),
        "input_type": "uint8",
        "labels": "imagenet",
        "postprocessor": "classification",
    },
    "Detection (SSD MobileDet)": {
        "zoo_dir": "ssdlite_mobiledet_coco_qat_postprocess",
        "zoo_file": "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
        "input_size": (320, 320),
        "input_type": "uint8",
        "labels": "coco",
        "postprocessor": "ssd_detection",
    },
    "Segmentation (DeepLabV3)": {
        "zoo_dir": "deeplabv3_mnv2_pascal_quant",
        "zoo_file": "deeplabv3_mnv2_pascal_quant_edgetpu.tflite",
        "input_size": (513, 513),
        "input_type": "uint8",
        "labels": None,
        "postprocessor": "deeplabv3",
    },
    "Pose (PoseNet)": {
        "zoo_dir": "posenet_mobilenet_v1_075_481_641_16_quant_decoder",
        "zoo_file": "posenet_mobilenet_v1_075_481_641_16_quant_decoder_edgetpu.tflite",
        "input_size": (641, 481),
        "input_type": "uint8",
        "labels": None,
        "postprocessor": "posenet",
    },
    "MultiPose (Multi-Person)": {
        "zoo_dir": None,  # special URL
        "zoo_file": "downloadedModels_mobilenet_float_050_model-stride16_edgetpu.tflite",
        "zoo_url": (
            "https://raw.githubusercontent.com/ricardodeazambuja/"
            "MultiPose-EdgeTPU-RPI0/main/"
            "downloadedModels_mobilenet_float_050_model-stride16_edgetpu.tflite"
        ),
        "input_size": (257, 257),
        "input_type": "int8",
        "labels": None,
        "postprocessor": "multipose",
    },
}

# Label files (directory, filename) — reuses EdgeTPUModelZoo structure
_LABEL_REGISTRY = {
    "imagenet": (
        "mobilenet_v1_1.0_224_quant",
        "imagenet_labels.txt",
    ),
    "coco": (
        "ssdlite_mobiledet_coco_qat_postprocess",
        "coco_labels.txt",
    ),
}


def get_model_names():
    """Return list of available model display names."""
    return list(MODEL_REGISTRY.keys())


def download_model(name):
    """Download (if needed) and return the absolute path to a model file.

    Args:
        name: Display name from MODEL_REGISTRY.

    Returns:
        Absolute path to the *_edgetpu.tflite file.

    Raises:
        ValueError: If name is not in MODEL_REGISTRY.
        OSError: If download fails.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {get_model_names()}")

    meta = MODEL_REGISTRY[name]
    local_path = os.path.join(_CACHE_DIR, meta["zoo_file"])

    if os.path.isfile(local_path):
        return local_path

    os.makedirs(_CACHE_DIR, exist_ok=True)

    if meta.get("zoo_url"):
        url = meta["zoo_url"]
    else:
        url = f"{_BASE_URL}/{meta['zoo_dir']}/{meta['zoo_file']}"

    print(f"Downloading model: {name}")
    print(f"  URL: {url}")
    urllib.request.urlretrieve(url, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  Saved to {local_path} ({size_mb:.1f} MB)")

    return local_path


def download_labels(name):
    """Download (if needed) and return a list of label strings.

    Args:
        name: Label set name (e.g. "imagenet", "coco"), or None.

    Returns:
        List of label strings, or empty list if name is None.

    Raises:
        ValueError: If name is not in _LABEL_REGISTRY.
    """
    if name is None:
        return []

    if name not in _LABEL_REGISTRY:
        raise ValueError(f"Unknown label set {name!r}")

    directory, filename = _LABEL_REGISTRY[name]
    local_path = os.path.join(_CACHE_DIR, filename)

    if not os.path.isfile(local_path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        url = f"{_BASE_URL}/{directory}/{filename}"
        print(f"Downloading labels: {name}")
        urllib.request.urlretrieve(url, local_path)

    with open(local_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
