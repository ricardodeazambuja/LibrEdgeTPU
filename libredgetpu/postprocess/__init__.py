"""Post-processing modules for Edge TPU models.

Provides CPU-side post-processing for models whose TFLite graphs include
operations that cannot run on the Edge TPU (custom ops, unsupported builtins).

Modules:
    deeplabv3       — Semantic segmentation (DeepLabV3 MobileNetV2)
    posenet_decoder — Single-person pose estimation (PoseNet PersonLab)
    multipose_decoder — Multi-person pose estimation (MobileNet V1 0.50)
    ssd_decoder     — Object detection (SSD MobileDet / SSD MobileNet)
"""
