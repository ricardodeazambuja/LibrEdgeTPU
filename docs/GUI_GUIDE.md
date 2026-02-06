# LibrEdgeTPU Interactive GUI Guide

## Overview

The LibrEdgeTPU Interactive GUI is a web-based testing tool that allows real-time visual validation of all 9 LibrEdgeTPU algorithms with live webcam input and mouse interaction.

## Installation

Install with GUI dependencies:

```bash
pip install -e ".[gui]"
```

This installs:
- `flask>=2.0.0` - Web server
- `opencv-python>=4.5.0` - Webcam capture and image processing

## Quick Start

### Basic Launch

```bash
python -m libredgetpu.gui
```

This will:
- Auto-detect USB Edge TPU (hardware mode) or fallback to CPU (synthetic mode)
- Auto-detect webcam (device 0)
- Start server at `http://0.0.0.0:5000`
- Default resolution: 640×480

### Command-Line Options

```bash
python -m libredgetpu.gui --help
```

Options:
- `--port 5000` - Server port (default: 5000)
- `--camera 0` - Camera device ID (default: auto-detect)
- `--synthetic` - Force CPU mode (no Edge TPU)
- `--pattern wandering_dot` - Synthetic pattern type
- `--width 640` - Frame width
- `--height 480` - Frame height
- `--host 0.0.0.0` - Server host (default: 0.0.0.0 for network access)

### Examples

**Force synthetic mode with rotating pattern:**
```bash
python -m libredgetpu.gui --synthetic --pattern rotating
```

**Use second camera with higher resolution:**
```bash
python -m libredgetpu.gui --camera 1 --width 1280 --height 720
```

**Run on custom port for remote access:**
```bash
python -m libredgetpu.gui --port 8080 --host 0.0.0.0
```

## Accessing the GUI

### Local Access
Open browser: `http://localhost:5000`

### Remote Access
Find your IP address and access from another device:
```bash
hostname -I  # Get IP address
# Then open: http://<IP>:5000
```

Example: `http://192.168.1.100:5000`

## GUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  LibrEdgeTPU Interactive Tester          [HARDWARE MODE]    │
├───────────────────────────────┬─────────────────────────────┤
│                               │  Algorithm: [SpotTracker ▼] │
│                               │                             │
│                               │  Parameters:                │
│     Live Video Canvas         │    Image Size: [64 ▼]       │
│     (interactive)             │    Variant: [bright ▼]      │
│                               │    [Apply]                  │
│                               │                             │
│  • Click for interaction      │  Performance:               │
│  • Drag for ROI selection     │    FPS: 28.3                │
│  • Spacebar to pause/play     │    Latency: 1.2 ms          │
│                               │                             │
│                               │  Output:                    │
│                               │    (algorithm metrics)      │
│                               │                             │
│                               │  Actions:                   │
│                               │   [⏸️ Pause] [📷 Screenshot]│
└───────────────────────────────┴─────────────────────────────┘

**Controls:**
- **Pause/Play**: Click button or press **Spacebar** to freeze/resume video
- **Mouse**: Click or drag on canvas for algorithm-specific interaction
- **Parameters**: Adjust algorithm-specific parameters and click **Apply** to re-initialize
- **Screenshot**: Capture current frame with annotations
```

## Per-Algorithm Interactions

### 1. SpotTracker
**Purpose:** Visual servoing via soft argmax

**Interaction:**
- **Click canvas** → Places synthetic Gaussian spot at click position
- Algorithm tracks brightest spot in frame

**Overlay:**
- Green crosshair at tracked position
- Offset values (normalized coordinates from center)

**Use case:** Test visual servoing, target tracking

### 2. PatternTracker
**Purpose:** Template matching via Conv2D correlation

**Interaction:**
1. **Press Pause (or Spacebar)** → Freeze frame for precise selection
2. **Click + drag canvas** → Select template ROI (red rectangle)
3. Release to capture template (auto-resized to square 8×8 to 32×32)
4. **Press Play (or Spacebar)** → Resume tracking

**Overlay:**
- Red box around matched template position
- Match coordinates

**Template constraints:**
- Forced to square aspect ratio (uses min(width, height))
- Size clamped to [8×8, 32×32] pixels
- Template auto-resized to fit constraints

**Use case:** Test object tracking, template matching

### 3. LoomingDetector
**Purpose:** Collision avoidance via edge density

**Interaction:**
- Passive observation (no click needed)
- Move objects toward camera to trigger looming

**Overlay:**
- 3×3 heatmap grid (blue=low density, red=high density)
- Max density value

**Use case:** Test collision avoidance, time-to-contact estimation

### 4. OpticalFlow
**Purpose:** Global ego-motion estimation

**Interaction:**
- Passive observation
- Move camera (pan, tilt, rotate) to generate flow

**Overlay:**
- Red arrow from center showing flow direction
- Flow magnitude (dx, dy)

**Use case:** Test visual odometry, motion estimation

### 5. VisualCompass
**Purpose:** Yaw angle estimation from optical flow

**Interaction:**
- Passive observation
- Rotate camera left/right to change yaw

**Overlay:**
- Compass needle in bottom-right corner
- Current yaw angle in degrees (0° = north)

**Use case:** Test heading estimation, navigation

### 6. MatMulEngine
**Purpose:** Runtime weight-swapping matrix multiply

**Interaction:**
- **Click canvas** → Randomize weights and reset plot

**Overlay:**
- Scatter plot: x[0] vs y[0] (input-output correlation)
- Histogram: output distribution

**Use case:** Test matrix operations, weight swapping

### 7. ReservoirComputer
**Purpose:** Echo state network (recurrent neural network)

**Interaction:**
- **Click canvas** → Inject spike into first neuron
- Watch state evolution over time

**Overlay:**
- 16×16 heatmap of reservoir state
- State norm

**Use case:** Test temporal processing, recurrent dynamics

### 8. EmbeddingSimilarity
**Purpose:** Cosine similarity search

**Interaction:**
- **Click canvas** → Add current frame to gallery (max 10)
- Algorithm computes similarity to all gallery items

**Overlay:**
- Top-3 most similar thumbnails with scores
- Gallery size

**Use case:** Test similarity search, image retrieval

### 9. SimpleInvoker
**Purpose:** Standard ML inference with model selection and post-processing

**Interaction:**
- Select a model from the dropdown (5 options) and click **Apply**
- Models are downloaded automatically on first use (cached in `~/.cache/libredgetpu/models/`)
- Each model has a dedicated post-processing pipeline and overlay visualization

**Available Models:**
| Model | Input Size | Post-processing | Overlay |
|-------|-----------|----------------|---------|
| Classification (MobileNet V1) | 224×224 | Top-5 argsort | Ranked labels with scores |
| Detection (SSD MobileDet) | 320×320 | SSD anchor decoding + NMS | Bounding boxes + class labels |
| Segmentation (DeepLabV3) | 513×513 | CPU conv + argmax | PASCAL VOC colormap blend |
| Pose (PoseNet) | 641×481 | PersonLab decoder | Keypoints + skeleton |
| MultiPose (Multi-Person) | 257×257 | Multi-pose decoder | Per-person colored skeletons |

**Notes:**
- MultiPose requires `scipy`: `pip install scipy`
- INT8 models (MultiPose) handle input preprocessing automatically
- Requires Edge TPU hardware — shows informative message in synthetic mode

**Use case:** Test standard ML inference, validate post-processing pipelines

## Tunable Parameters

Each algorithm exposes tunable parameters via a control panel that appears below the algorithm
dropdown. Adjust values and click **Apply** to re-initialize the algorithm with the new settings.

**How it works:**
1. Select an algorithm from the dropdown — its parameters appear automatically
2. Adjust values (dropdowns, number inputs, sliders, checkboxes)
3. Click **Apply** — the algorithm re-initializes with the new parameters
4. A "Applied successfully" message confirms the change; errors are shown in red

**Important:** Applying parameters creates a fresh algorithm instance. Any accumulated state
(PatternTracker template, EmbeddingSimilarity gallery, VisualCompass cumulative yaw) is reset.

### Available Parameters

| Algorithm | Parameters |
|-----------|-----------|
| **SpotTracker** | Image Size (64/128), Variant (bright/color_red/.../color_white) |
| **PatternTracker** | Search Size (64/128), Kernel Size (8/16/32), Channels (1/3) |
| **LoomingDetector** | Image Size (64/128) |
| **OpticalFlow** | Image Size (64/128), Pooled Mode (on/off), Pool Factor (1/2/4/8), Search Range (1-8), Temperature (0.01-1.0) |
| **VisualCompass** | Same as OpticalFlow + FOV in degrees (1-360) |
| **MatMulEngine** | Dimension (256/512/1024) |
| **ReservoirComputer** | Dimension (256/512/1024), Spectral Radius (0.1-2.0), Leak Rate (0.01-1.0), Activation (tanh/relu/identity) |
| **EmbeddingSimilarity** | Dimension (256/512/1024) |
| **SimpleInvoker** | Model (Classification/Detection/Segmentation/PoseNet/MultiPose) |

**Spacebar note:** The Spacebar shortcut for pause/play is disabled when a parameter input
or dropdown is focused, so you can type values without accidentally toggling pause.

## Synthetic Patterns

When running in synthetic mode (`--synthetic`), choose from:

1. **noise** - Random noise (stress test)
2. **checkerboard** - Static checkerboard (template matching test)
3. **rotating** - Rotating line pattern (optical flow rotation test)
4. **panning** - Horizontally scrolling stripes (optical flow translation test)
5. **wandering_dot** - Lissajous curve Gaussian dot (spot tracker test) ⭐ **Default**

## Performance Metrics

The GUI displays:
- **FPS** - Frames per second (end-to-end including MJPEG encoding)
- **Latency** - Algorithm processing time (excludes webcam capture and MJPEG encoding)
- **Mode** - HARDWARE (Edge TPU) or SYNTHETIC (CPU)

**Expected latencies** (hardware mode):
- SpotTracker: ~1 ms
- PatternTracker: ~5 ms
- LoomingDetector: ~1.5 ms
- OpticalFlow: ~2 ms
- VisualCompass: ~2 ms
- MatMulEngine: ~0.28 ms
- ReservoirComputer: ~0.6 ms
- EmbeddingSimilarity: ~0.28 ms

**Note:** Total FPS includes webcam capture (~30 FPS) and MJPEG encoding (~50-100 ms overhead).

## Screenshots

Click **📷 Screenshot** to download annotated frame as JPEG with timestamp:
```
libredgetpu_screenshot_20260207_143052.jpg
```

## Troubleshooting

### No webcam detected
**Symptom:** Black screen or "Failed to open camera" warning

**Solution:**
- Check webcam is plugged in: `ls /dev/video*`
- Try different camera ID: `--camera 1`
- Use synthetic mode: `--synthetic`

### No Edge TPU detected
**Symptom:** Mode badge shows "SYNTHETIC" instead of "HARDWARE"

**Solution:**
- Check USB TPU: `lsusb | grep -iE "1a6e|18d1"`
- `1a6e:089a` = bootloader (needs firmware)
- `18d1:9302` = runtime (ready)
- Use synthetic mode for offline testing

### Low FPS
**Symptom:** FPS < 10

**Solution:**
- Reduce resolution: `--width 320 --height 240`
- Use pooled optical flow (already default)
- Check CPU usage
- Network latency if accessing remotely

### Browser shows blank page
**Symptom:** GUI loads but no video

**Solution:**
- Check server console for errors
- Verify `/video_feed` endpoint responds: `curl http://localhost:5000/video_feed`
- Check browser console (F12) for JavaScript errors

### Algorithm switch doesn't respond
**Symptom:** Dropdown change has no effect

**Solution:**
- Check browser console for POST errors
- Verify backend is running
- Reload page (Ctrl+R)

## Development

### Adding Custom Algorithms

1. Create new `AlgorithmMode` class in `algorithm_modes.py`:
```python
class MyAlgorithmMode(AlgorithmMode):
    @classmethod
    def get_param_schema(cls):
        return [
            {"name": "threshold", "label": "Threshold", "type": "range",
             "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
             "description": "Detection threshold"},
        ]

    def __init__(self, synthetic=False, threshold=0.5):
        super().__init__(synthetic)
        self._active_params = {"threshold": threshold}
        # ... init with threshold ...

    def process(self, frame, mouse_state):
        # Your processing logic
        annotated = frame.copy()
        # ... draw overlays ...
        return annotated
```

2. Register in `ALGORITHM_MODES`:
```python
ALGORITHM_MODES["MyAlgorithm"] = MyAlgorithmMode
```

3. Restart server — the parameter panel will appear automatically

### Architecture

```
Frontend (browser)
  ↓ SSE (metrics)
  ↓ POST (click/drag)
  ↓ GET (MJPEG stream)
Flask server (app.py)
  ↓
VideoStream (manages camera + algorithm)
  ↓
AlgorithmMode.process(frame, mouse_state)
  ↓
Hardware (Edge TPU) or Synthetic (NumPy CPU)
```

## Limitations

- **Latency:** MJPEG adds ~50-100 ms overhead (acceptable for testing/demos, not closed-loop control)
- **Bandwidth:** MJPEG is ~10-20× larger than H.264 (use LAN, not internet)
- **Security:** No authentication (run on trusted network only)
- **Browser compatibility:** Tested on Chrome/Firefox (Safari may have issues)

## See Also

- [ROBOTICS_STATUS.md](ROBOTICS_STATUS.md) - Full algorithm specifications
- [ADVANCED_README.md](ADVANCED_README.md) - Architecture internals
- [VISUAL_TESTS.md](VISUAL_TESTS.md) - Static visual test suite
