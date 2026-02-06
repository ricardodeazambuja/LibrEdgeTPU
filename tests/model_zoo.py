"""Download and cache Edge TPU models from EdgeTPUModelZoo.

Models are downloaded from:
    https://github.com/ricardodeazambuja/EdgeTPUModelZoo

Files are cached in tests/models/ (git-ignored).
"""

import os
import urllib.request

_BASE_URL = (
    "https://raw.githubusercontent.com/ricardodeazambuja/EdgeTPUModelZoo/master"
)
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# Registry: short name -> (directory, filename)
MODELS = {
    "mobilenet_v1": (
        "mobilenet_v1_1.0_224_quant",
        "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
    ),
    "mobilenet_v2": (
        "mobilenet_v2_1.0_224_quant",
        "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
    ),
    "inception_v1": (
        "inception_v1_224_quant",
        "inception_v1_224_quant_edgetpu.tflite",
    ),
    "inception_v2": (
        "inception_v2_224_quant",
        "inception_v2_224_quant_edgetpu.tflite",
    ),
    "ssd_mobiledet": (
        "ssdlite_mobiledet_coco_qat_postprocess",
        "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
    ),
    "efficientnet_s": (
        "efficientnet-edgetpu-S_quant",
        "efficientnet-edgetpu-S_quant_edgetpu.tflite",
    ),
    "deeplabv3": (
        "deeplabv3_mnv2_pascal_quant",
        "deeplabv3_mnv2_pascal_quant_edgetpu.tflite",
    ),
    "posenet": (
        "posenet_mobilenet_v1_075_481_641_16_quant_decoder",
        "posenet_mobilenet_v1_075_481_641_16_quant_decoder_edgetpu.tflite",
    ),
    "ssd_mobilenet_v1": (
        "ssd_mobilenet_v1_coco_quant_postprocess",
        "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
    ),
    "ssd_mobilenet_v2": (
        "ssd_mobilenet_v2_coco_quant_postprocess",
        "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
    ),
}

# Models hosted outside the standard EdgeTPUModelZoo (full URL, filename)
_SPECIAL_URLS = {
    "posenet_multipose": (
        "https://raw.githubusercontent.com/ricardodeazambuja/MultiPose-EdgeTPU-RPI0/main/"
        "downloadedModels_mobilenet_float_050_model-stride16_edgetpu.tflite",
        "downloadedModels_mobilenet_float_050_model-stride16_edgetpu.tflite",
    ),
}


def get_model(name):
    """Return the local path to a model, downloading it if necessary.

    Args:
        name: Short model name (key in MODELS or _SPECIAL_URLS dict), or a
              full path to a local .tflite file (returned as-is).

    Returns:
        Absolute path to the *_edgetpu.tflite file.
    """
    # If it's already an absolute path, just return it
    if os.path.isabs(name) and os.path.isfile(name):
        return name

    # Check special URLs first
    if name in _SPECIAL_URLS:
        url, filename = _SPECIAL_URLS[name]
        local_path = os.path.join(_MODELS_DIR, filename)
        if os.path.isfile(local_path):
            return local_path
        os.makedirs(_MODELS_DIR, exist_ok=True)
        print(f"Downloading {name}: {url}")
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Saved to {local_path} ({size_mb:.1f} MB)")
        return local_path

    if name not in MODELS:
        available = ", ".join(sorted(list(MODELS.keys()) + list(_SPECIAL_URLS.keys())))
        raise ValueError(f"Unknown model {name!r}. Available: {available}")

    directory, filename = MODELS[name]
    local_path = os.path.join(_MODELS_DIR, filename)

    if os.path.isfile(local_path):
        return local_path

    os.makedirs(_MODELS_DIR, exist_ok=True)

    url = f"{_BASE_URL}/{directory}/{filename}"
    print(f"Downloading {name}: {url}")
    urllib.request.urlretrieve(url, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  Saved to {local_path} ({size_mb:.1f} MB)")

    return local_path


# Label file names within each model directory
LABELS = {
    "imagenet": (
        "mobilenet_v1_1.0_224_quant",
        "imagenet_labels.txt",
    ),
    "coco": (
        "ssdlite_mobiledet_coco_qat_postprocess",
        "coco_labels.txt",
    ),
}

# Sample test image (Grace Hopper — person, good for classification + pose)
_SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/google-coral/test_data/master/"
    "grace_hopper.bmp"
)
_SAMPLE_IMAGE_FILE = "grace_hopper.png"


# Multi-person test image (couple — 2 people, good for multipose)
_MULTIPOSE_IMAGE_URL = (
    "https://raw.githubusercontent.com/google-coral/project-posenet/master/"
    "test_data/test_couple.jpg"
)
_MULTIPOSE_IMAGE_FILE = "test_couple.jpg"


def get_multipose_image():
    """Return path to a multi-person test image, downloading if necessary."""
    local_path = os.path.join(_MODELS_DIR, _MULTIPOSE_IMAGE_FILE)
    if not os.path.isfile(local_path):
        os.makedirs(_MODELS_DIR, exist_ok=True)
        print(f"Downloading multi-person image: {_MULTIPOSE_IMAGE_URL}")
        urllib.request.urlretrieve(_MULTIPOSE_IMAGE_URL, local_path)
    return local_path


def get_labels(name):
    """Return a list of label strings, downloading the file if necessary.

    Args:
        name: Label set name (key in LABELS dict).

    Returns:
        List of label strings (index-aligned with model output classes).
    """
    if name not in LABELS:
        available = ", ".join(sorted(LABELS.keys()))
        raise ValueError(f"Unknown label set {name!r}. Available: {available}")

    directory, filename = LABELS[name]
    local_path = os.path.join(_MODELS_DIR, filename)

    if not os.path.isfile(local_path):
        os.makedirs(_MODELS_DIR, exist_ok=True)
        url = f"{_BASE_URL}/{directory}/{filename}"
        print(f"Downloading labels {name}: {url}")
        urllib.request.urlretrieve(url, local_path)

    with open(local_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_sample_image():
    """Return the local path to a sample test image (PNG), downloading if necessary.

    The upstream image is BMP; we convert to PNG on download to save space.
    """
    local_path = os.path.join(_MODELS_DIR, _SAMPLE_IMAGE_FILE)
    if not os.path.isfile(local_path):
        os.makedirs(_MODELS_DIR, exist_ok=True)
        print(f"Downloading sample image: {_SAMPLE_IMAGE_URL}")
        tmp_path = local_path + ".tmp"
        urllib.request.urlretrieve(_SAMPLE_IMAGE_URL, tmp_path)
        # Convert BMP → PNG
        from PIL import Image
        Image.open(tmp_path).save(local_path, "PNG")
        os.remove(tmp_path)
    return local_path


def list_models():
    """Print all available models and their download status."""
    for name, (directory, filename) in sorted(MODELS.items()):
        local_path = os.path.join(_MODELS_DIR, filename)
        status = "cached" if os.path.isfile(local_path) else "not downloaded"
        print(f"  {name:20s} [{status}]  {filename}")
