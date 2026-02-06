# Standalone Robotics Examples

Copy-paste-ready Python scripts for integrating libredgetpu modules into
robotics platforms (Raspberry Pi, Jetson, etc.).  Each script demonstrates
a real-time webcam loop with full parameter control via argparse.

## Requirements

```bash
pip install libredgetpu opencv-python
# Edge TPU USB accelerator required for all examples except --help
```

## Quick Start

```bash
cd examples/

# Visual servoing — track brightest spot
python spot_tracker_example.py --image-size 64 --variant bright

# Collision avoidance with time-to-contact
python looming_detector_example.py --tau-threshold 1.2 --ttc-window 10

# Global optical flow
python optical_flow_example.py --pooled --pool-factor 2

# Yaw/heading estimation
python visual_compass_example.py --fov-deg 90 --pooled

# Template matching
python pattern_tracker_example.py --kernel-size 16 --template logo.png

# Matrix multiply benchmark (no webcam)
python matmul_engine_example.py --dim 256 --iterations 1000 --verify

# Echo state network with webcam input
python reservoir_computer_example.py --dim 256 --spectral-radius 0.95

# Place recognition / similarity search
python embedding_similarity_example.py --dim 256 --top-k 3

# ML inference (classification, detection, segmentation, pose)
python simple_invoker_example.py --model detection --score-threshold 0.3
```

## Overview

| Script | Module | Use Case | Webcam | Latency |
|--------|--------|----------|--------|---------|
| `spot_tracker_example.py` | SpotTracker | Visual servoing | Yes | ~1 ms |
| `looming_detector_example.py` | LoomingDetector | Collision avoidance + TTC | Yes | ~1.5 ms |
| `optical_flow_example.py` | OpticalFlow | Ego-motion estimation | Yes | ~2 ms |
| `visual_compass_example.py` | VisualCompass | Yaw/heading estimation | Yes | ~2 ms |
| `pattern_tracker_example.py` | PatternTracker | Template matching | Yes | ~5 ms |
| `matmul_engine_example.py` | MatMulEngine | Matrix multiply benchmark | No | ~0.28 ms |
| `reservoir_computer_example.py` | ReservoirComputer | Echo state network | Yes | ~0.6 ms |
| `embedding_similarity_example.py` | EmbeddingSimilarity | Place recognition | Yes | ~0.28 ms |
| `simple_invoker_example.py` | SimpleInvoker | ML inference (5 models) | Yes | 4-15 ms |

## Common Arguments

All webcam-based scripts share these arguments (via `_common.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | 0 | Camera device index |
| `--width` | 640 | Camera frame width |
| `--height` | 480 | Camera frame height |
| `--no-display` | off | Headless mode (no OpenCV window) |

## Headless Mode

For embedded systems without a display:

```bash
python spot_tracker_example.py --no-display
# Metrics printed to terminal via single-line \r updates
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset yaw (visual_compass) |
| `t` | Re-capture template from center (pattern_tracker) |
| `s` | Snapshot to gallery (embedding_similarity) |
| `c` | Clear gallery (embedding_similarity) |

## Adapting for Your Robot

Each script follows the same pattern:

```python
with Module.from_template(...) as module:
    for frame in webcam_loop:
        result = module.compute(frame)
        # result → your robot's control loop
```

Replace the webcam loop with your camera driver and the display calls
with your robot's actuator commands.
