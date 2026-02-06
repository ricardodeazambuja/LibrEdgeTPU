# libredgetpu

[![Offline Tests](https://github.com/ricardodeazambuja/libredgetpu/actions/workflows/test-offline.yml/badge.svg)](https://github.com/ricardodeazambuja/libredgetpu/actions/workflows/test-offline.yml)

Pure-Python Edge TPU inference engine. No libedgetpu. No tflite_runtime.

Run compiled Edge TPU models (`*_edgetpu.tflite`) directly via USB using only three
dependencies: `pyusb`, `numpy`, and `flatbuffers`. Works on any Linux system with USB support
(x86-64, ARM, RISC-V, etc.) — the entire stack is pure Python, so there are no
architecture-specific binaries. An optional C extension (`_usb_accel.c`) for faster USB
transfers compiles on any platform with `libusb-1.0-dev` and a C compiler; if unavailable,
the pure-Python fallback is used automatically.

## Background

libredgetpu was built from scratch through independent analysis of the Google Coral Edge TPU's USB
protocol, hardware registers, and DarwiNN executable format. It replaces Google's
proprietary `libedgetpu` + `tflite_runtime` stack with a pure-Python implementation that
gives full control over the hardware — enabling runtime weight swapping, custom computation
templates, and direct parameter blob manipulation.

The Edge TPU is an int8 MAC engine (64x64 systolic array at 480 MHz = 4 TOPS). libredgetpu
exploits it as a general-purpose accelerator for matrix multiplication, visual servoing,
collision avoidance, and standard ML inference.

See [`docs/HARDWARE_ANALYSIS.md`](docs/HARDWARE_ANALYSIS.md) for the full hardware
findings (~1400 lines covering ISA, registers, DMA, and patent cross-references).

## Install

```bash
pip install -e .
```

On Linux, you need libusb and udev rules for non-root USB access:

```bash
sudo apt install libusb-1.0-0-dev
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1a6e", MODE="0666"' | sudo tee /etc/udev/rules.d/99-coral.rules
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666"' | sudo tee -a /etc/udev/rules.d/99-coral.rules
sudo udevadm control --reload-rules
```

**Firmware**: The Edge TPU firmware (`apex_latest_single_ep.bin`, ~36 MB) is auto-downloaded
on first use with SHA256 verification over TLS. Upload to the device retries up to 3 times
on USB errors before failing. For offline/air-gapped environments, pre-download it:

```bash
python -c "from libredgetpu.transport import USBTransport; USBTransport._ensure_firmware()"
```

An optional C extension (`_usb_accel`) compiles automatically if `libusb-1.0-dev` is present,
giving ~2x USB transfer speedup. Without it, the package works as pure Python via pyusb.

```bash
# Check if the C extension is active
python -c "from libredgetpu.transport import _HAS_C_ACCEL; print(_HAS_C_ACCEL)"
```

## Quick Start: Run Any Edge TPU Model

```python
from libredgetpu import SimpleInvoker
import numpy as np

with SimpleInvoker("mobilenet_v1_edgetpu.tflite") as model:
    output = model.invoke(np.zeros(150528, dtype=np.float32))  # float in/out
    raw = model.invoke_raw(input_bytes)                         # uint8 in/out
```

`SimpleInvoker` handles firmware download, hardware init, quantization, parameter caching,
and DMA hint-driven execution automatically.

## Robotics Modules

Beyond standard inference (``SimpleInvoker``), libredgetpu ships eight robotics modules
built on ``MatMulEngine`` and pre-compiled templates. All run entirely on the Edge TPU
(CPU only does final scalar math). No TensorFlow or `edgetpu_compiler` needed at runtime.

### MatMulEngine: Custom Matrix Multiply

Runtime weight-swapping `y = Wx` on the Edge TPU. Useful for adaptive controllers,
online learning, dynamic transforms, sensor pre-processing (FIR filters, DCT, PCA).

```python
from libredgetpu import MatMulEngine
import numpy as np

with MatMulEngine.from_template(256) as engine:
    # Set custom weights (float32 -> int8 internally)
    W = np.random.randn(256, 256).astype(np.float32) * 0.05
    engine.set_weights(W)       # microseconds (compiler-free, pure NumPy)

    # Run y = W @ x
    x = np.random.randn(256).astype(np.float32)
    y = engine.matmul(x)        # ~0.28 ms

    # Swap to different weights instantly
    engine.set_weights(np.eye(256, dtype=np.float32) * 0.04)
    y2 = engine.matmul(x)
```

**Templates included**: 256, 512, 1024. Generate more: `python -m libredgetpu.template_gen --sizes 2048`

**Key properties**:
- `set_weights()` generates DarwiNN parameter blobs directly in NumPy (no `edgetpu_compiler` needed at runtime)
- Inference: ~0.28 ms regardless of matrix size (USB-limited)
- Weight upload: 1.3 ms (256x256) to 25 ms (~2896x2896, max cached)
- Weight range: `engine.weight_range` shows the allowed float32 range (values are clipped)
- 8-bit precision: output has 256 distinct values per element

### SpotTracker: Visual Servoing

Tracks the brightest region (or a specific color) in an image. Returns (x, y) offset
from image center, directly usable as servo control error.

```python
from libredgetpu import SpotTracker
import numpy as np

# Grayscale bright spot tracking
with SpotTracker.from_template(64) as tracker:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[10:20, 50:60] = 255  # bright spot in upper-right

    x_off, y_off = tracker.track(image)
    # x_off > 0 (right of center), y_off < 0 (above center)

    # Convert to servo error (inverted for closed-loop control)
    x_err, y_err = SpotTracker.offset_to_servo_error(x_off, y_off, gain=0.5)

    # Or get a direction string
    direction = SpotTracker.offset_to_direction(x_off, y_off)  # "up-right"
```

**Color tracking** is also supported. The model adds a 1x1 convolution color filter
before the soft argmax:

```python
# Track a red object in an RGB image
with SpotTracker.from_template(64, variant="color_red") as tracker:
    rgb_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    x_off, y_off = tracker.track(rgb_image)
```

Available color variants: `color_red`, `color_green`, `color_blue`, `color_yellow`,
`color_cyan`, `color_magenta`, `color_white`.

**Runtime color swapping** — switch target color instantly without recompilation:

```python
with SpotTracker.from_template(64, variant="color_red") as tracker:
    tracker.set_color([1.0, -0.5, -0.5])    # red (original)
    x, y = tracker.track(rgb_image)

    tracker.set_color([-0.5, 1.0, -0.5])    # switch to green instantly
    x, y = tracker.track(rgb_image)

    tracker.set_color([0.8, 0.4, -0.6])     # any custom color
    x, y = tracker.track(rgb_image)
```

Coefficients are `[R, G, B]`: positive = attract, negative = repel. Use a template with
scale=0.007874 (red, green, blue, or yellow) as the base for the widest range ([-1, +1]).

**Closest-match API** — find the nearest pre-compiled color for custom RGB weights:

```python
tracker = SpotTracker.from_color_weights(64, weights=[0.9, 0.1, -0.8])
print(tracker.matched_variant)    # "color_yellow"
print(tracker.matched_distance)   # 0.707 (Euclidean distance)
```

**Templates included**: grayscale (16x16, 64x64, 128x128) and all 7 colors (64x64, 128x128).
Generate custom sizes or colors:

```bash
python -m libredgetpu.spot_tracker_gen --sizes 64 128 --all-colors
python -m libredgetpu.spot_tracker_gen --sizes 64 --variant color_orange \
    --color-weights 1.0 0.5 -0.5
```

**How it works**: Input pixels are converted to a probability distribution via Softmax
(brighter = higher probability). Two Dense layers compute the expected X and Y coordinates
as weighted sums (soft argmax). Output is in [-1, +1] range. Larger input images are
automatically resized to the template dimensions.

### PatternTracker: Template Matching

Locates a reference patch within a larger search image using Conv2D sliding
cross-correlation + soft argmax peak detection. Swap the template at runtime
to track different patterns.

```python
from libredgetpu import PatternTracker
import numpy as np

with PatternTracker.from_template(search_size=64, kernel_size=8) as tracker:
    search_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    x_off, y_off = tracker.track(search_image)
    # x_off, y_off in [-1, +1]: where the template was found

    # Swap to a new template at runtime (requires edgetpu_compiler)
    new_patch = np.random.randint(0, 256, (8, 8)).astype(np.float32)
    tracker.set_template(new_patch)
    x_off, y_off = tracker.track(search_image)
```

**RGB support**:

```python
with PatternTracker.from_template(64, 8, channels=3) as tracker:
    rgb_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    x_off, y_off = tracker.track(rgb_image)
```

**Templates included**: 6 combinations of search/kernel/channels:
- `64x64/8x8/1ch`, `64x64/16x16/1ch`, `128x128/16x16/1ch`, `128x128/32x32/1ch`
- `64x64/8x8/3ch`, `128x128/16x16/3ch`

Generate more: `python -m libredgetpu.pattern_tracker_gen --search-sizes 64 --template-sizes 16 --channels 1`

**How it works**: A Conv2D with `padding='valid'` slides the kernel across the search image,
producing a correlation map. ReLU thresholds negative correlations. Softmax converts the map
to a probability distribution, and two Dense layers compute expected (x, y) via weighted sum
(soft argmax). Template swapping patches the Conv2D weights in the DarwiNN parameter blob.

### LoomingDetector: Collision Avoidance

Detects approaching objects by measuring edge density in a 3x3 grid of spatial zones.
An expanding object fills the center zone with edges faster than the periphery.

```python
from libredgetpu import LoomingDetector
import numpy as np

with LoomingDetector.from_template(64) as detector:
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    # Get edge density for 9 zones
    zones = detector.detect(image)  # [9] float32

    # Compute tau: center / mean(periphery)
    tau = LoomingDetector.compute_tau(zones)
    # tau > 1.0: approaching, tau < 1.0: receding, tau ~ 1.0: stable

    # Estimate time-to-contact from tau history
    tau_history = [0.9, 1.0, 1.1, 1.2]  # collected over time
    ttc = LoomingDetector.compute_ttc(tau_history, dt=0.033)  # at 30 FPS
```

**Templates included**: 64x64, 128x128. Generate more: `python -m libredgetpu.looming_gen --sizes 256`

**Zone layout**:
```
[0][1][2]  top
[3][4][5]  middle (4 = center)
[6][7][8]  bottom
```

**How it works**: Two Conv2D layers with fixed Sobel kernels detect edges. The magnitude
is squared (since Abs isn't supported on Edge TPU), then AvgPool divides the image into
9 zones. All ops run on the Edge TPU; CPU only computes the ratio.

**Performance**: 64x64: 1.0 ms (~1000 Hz), 128x128: ~1.5 ms.

### OpticalFlow: Global Ego-Motion Estimation

Computes a single (vx, vy) displacement vector between two grayscale frames using
Gabor feature extraction on the Edge TPU and CPU-side global correlation with soft argmax.
Useful for ego-motion estimation, heading, and speed sensing.

```python
from libredgetpu import OpticalFlow
import numpy as np

with OpticalFlow.from_template(64) as flow:
    frame_t = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    frame_t1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    # Global optical flow
    vx, vy = flow.compute(frame_t, frame_t1)
    # vx > 0: rightward motion, vy > 0: downward motion

    # Direction classification
    direction = OpticalFlow.flow_to_direction(vx, vy)  # "left", "up-right", etc.
```

**Templates included**: 64x64, 128x128. Generate more: `python -m libredgetpu.optical_flow_gen --sizes 256`

**How it works**: The Edge TPU applies 8 fixed Gabor filters (4 orientations x 2 scales,
7x7 kernels, SAME padding, ReLU) to extract multi-orientation edge features. The CPU
downsamples features 4x via block sum, computes 81 global correlation scores for
+/-4 pixel displacements (normalized by overlap area), and applies softmax-weighted sum
(soft argmax) for sub-pixel (vx, vy) output. Two Edge TPU calls per frame pair (~0.5 ms
each) + trivial CPU (~1 ms).

**Tunable parameters**: `search_range` (default 4, sets ±N pixel displacement grid →
(2N+1)² scores), `temperature` (default 0.1, softmax sharpness), `pool_factor` (default 4,
CPU downsampling factor for standard mode).

**Pooled mode** (`pooled=True`): Fuses AVG_POOL_2D into the Edge TPU model, reducing USB
transfer from 64KB to 4KB per frame (16x smaller). Use `OpticalFlow.from_template(64, pooled=True)`.

**Performance**: 64x64: ~1.5-2.5 ms total (two Edge TPU calls + CPU correlation).

### VisualCompass: Yaw Estimation

Thin wrapper around OpticalFlow that converts horizontal pixel displacement into
a yaw angle using the camera's field-of-view. No additional Edge TPU model needed —
it reuses the OpticalFlow pipeline and adds only the angular conversion.

```python
from libredgetpu import VisualCompass
import numpy as np

with VisualCompass.from_template(64, fov_deg=90.0, pooled=True) as compass:
    frame_t = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    frame_t1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)

    # Yaw angle in degrees (positive = rightward rotation)
    yaw = compass.compute_yaw(frame_t, frame_t1)

    # Full output: yaw + raw flow
    yaw, vx, vy = compass.compute(frame_t, frame_t1)

    # Direction classification
    direction = VisualCompass.yaw_to_direction(yaw)  # "left", "right", "center"
```

**Why a wrapper?** OpticalFlow already computes horizontal displacement (`vx`), which
is the yaw signal. Duplicating the Gabor+correlation pipeline would add code with no
benefit. VisualCompass adds only the `fov_deg / image_width` conversion and a
domain-specific API for heading estimation.

**Key conversion**: `yaw_deg = vx * fov_deg * effective_pool / image_width`, where
`effective_pool` is the fused pool factor (if > 0) or the CPU pool factor.

**How it works**: Delegates to OpticalFlow for Gabor feature extraction and correlation,
then converts the horizontal displacement `vx` into degrees via
`yaw = vx * fov_deg * effective_pool / image_width`.

**Performance**: Same as OpticalFlow (~1.5-2.5 ms for 64x64).

### ReservoirComputer: Echo State Network

Fixed random recurrent network (Echo State Network) on the Edge TPU. The reservoir
weight matrix `W_res` runs as a MatMulEngine matmul; only the linear readout layer
is trained (ridge regression on CPU). Useful for always-on sensor monitoring:
vibration analysis, anomaly detection, time-series classification.

```python
from libredgetpu import ReservoirComputer
import numpy as np

with ReservoirComputer.from_template(256, input_dim=4, seed=42) as rc:
    # Train readout on labelled time-series
    rc.fit(train_inputs, train_targets, warmup=100)

    # Predict on new data
    predictions = rc.predict(test_inputs)

    # Or step one sample at a time for online use
    rc.reset_state()
    for u in streaming_inputs:
        state = rc.step(u)                   # ~0.6 ms
        y = rc.readout @ state               # CPU readout
```

**Performance**: ~1.7 kHz update rate at N=256 (~0.6 ms per step: 0.28 ms Edge TPU +
CPU input projection and leaky integration).

**Key properties**:
- Reuses Dense(N) templates from MatMulEngine (no new Edge TPU model)
- `spectral_radius` controls reservoir dynamics (default 0.95)
- `leak_rate` for leaky integration (default 1.0 = no leak)
- Activations: `tanh` (default), `relu`, `identity`
- Deterministic via `seed` parameter
- Optional `readout_engine` for large output dimensions on Edge TPU

**How it works**: Each timestep, the Edge TPU computes `W_res @ x(t-1)` via MatMulEngine
(0.28 ms). The CPU adds the input projection `W_in @ u(t)`, applies leaky integration
and activation, then multiplies the state by a trained readout matrix `W_out`.

### EmbeddingSimilarity: Cosine Similarity Search

Stores a database of L2-normalized embeddings as MatMulEngine weight rows. A single
matmul computes cosine similarity against all entries. Useful for visual place
recognition (pair with a MobileNet backbone for the embedding extraction step).

```python
from libredgetpu import EmbeddingSimilarity
import numpy as np

with EmbeddingSimilarity.from_template(256) as sim:
    # Build a database of known places (embeddings from e.g. MobileNet backbone)
    rng = np.random.default_rng(0)
    for name in ["kitchen", "hallway", "lab"]:
        sim.add(name, rng.standard_normal(256).astype(np.float32))

    # Query: find the closest match
    query = rng.standard_normal(256).astype(np.float32)
    results = sim.query(query, top_k=3)
    for label, score in results:
        print(f"{label}: {score:.3f}")

    # Persistence
    sim.save("places.npz")
    sim.load("places.npz")  # portable across engines with different weight ranges
```

**Performance**: ~0.28 ms per query (single matmul, USB-limited).

**Precision note**: Int8 quantization gives ~5-6 similarity levels across [0, 1].
Rankings are reliable; absolute scores are coarse. Sufficient for place recognition
and nearest-neighbor retrieval.

**Key properties**:
- Reuses Dense(N) templates from MatMulEngine (no new Edge TPU model)
- L2 normalization + scale factor ensures embeddings fit in the int8 weight range
- Save/load stores pre-scale normalized embeddings for engine portability
- Capacity = matrix_size (e.g., 256 entries for Dense(256))

**How it works**: L2-normalized embeddings are scaled to fit the int8 weight range and
stored as rows of the MatMulEngine weight matrix. A query is L2-normalized and multiplied
by the weight matrix in a single matmul; the output is unscaled to recover cosine similarity.

## Standalone Robotics Examples

Copy-paste-ready Python scripts for integrating libredgetpu into robotics platforms
(Raspberry Pi, Jetson, etc.). Each script is a self-contained webcam loop with full
argparse parameter control.

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
python pattern_tracker_example.py --kernel-size 16

# Matrix multiply benchmark (no webcam)
python matmul_engine_example.py --dim 256 --iterations 1000 --verify

# Echo state network with webcam input
python reservoir_computer_example.py --dim 256 --spectral-radius 0.95

# Place recognition / similarity search
python embedding_similarity_example.py --dim 256 --top-k 3

# ML inference (classification, detection, segmentation, pose)
python simple_invoker_example.py --model detection --score-threshold 0.3
```

All webcam scripts support `--no-display` for headless operation on embedded systems.
See [`examples/README.md`](examples/README.md) for full parameter reference.

## Post-Processing for Complex Models

Some models have CPU-side operations after the Edge TPU portion (custom ops, NMS,
argmax, etc.). libredgetpu includes pure-NumPy reimplementations for four model families:

| Post-processor | Model family | Task | Import |
|---|---|---|---|
| `ssd_decoder` | SSD MobileDet / SSD MobileNet | Object detection | `postprocess_ssd` |
| `posenet_decoder` | PoseNet PersonLab | Single-person pose (17 kpts) | `postprocess_posenet` |
| `multipose_decoder` | MobileNet V1 0.50 | Multi-person pose (17 kpts) | `postprocess_multipose` |
| `deeplabv3` | DeepLabV3 MobileNetV2 | Semantic segmentation (21 classes) | `postprocess_deeplabv3` |

```python
from libredgetpu import SimpleInvoker

# SSD: object detection (COCO 90 classes, bounding boxes + scores)
from libredgetpu.postprocess.ssd_decoder import postprocess_ssd
with open("ssd_mobildet_edgetpu.tflite", "rb") as f:
    tflite_bytes = f.read()
with SimpleInvoker("ssd_mobildet_edgetpu.tflite") as model:
    raw_outputs = model.invoke_raw_outputs(input_bytes)
    detections = postprocess_ssd(raw_outputs, model.output_layers, tflite_bytes,
                                 score_threshold=0.3)
    for class_id, score, ymin, xmin, ymax, xmax in detections:
        print(f"Class {class_id}: {score:.2f}  box=[{ymin:.2f},{xmin:.2f},{ymax:.2f},{xmax:.2f}]")

# PoseNet: single-person pose estimation (17 keypoints)
from libredgetpu.postprocess.posenet_decoder import postprocess_posenet
with open("posenet_edgetpu.tflite", "rb") as f:
    tflite_bytes = f.read()
with SimpleInvoker("posenet_edgetpu.tflite") as model:
    raw_outputs = model.invoke_raw_outputs(input_bytes)
    poses = postprocess_posenet(raw_outputs, model.output_layers, tflite_bytes)
    for pose in poses:
        print(f"Score: {pose.score:.2f}, keypoints: {pose.keypoints.shape}")

# MultiPose: multi-person pose estimation (17 keypoints per person)
# Requires scipy: pip install libredgetpu[multipose]
from libredgetpu.postprocess.multipose_decoder import (
    get_multipose_model, postprocess_multipose,
)
model_path = get_multipose_model()  # auto-downloads ~805 KB model
with open(model_path, "rb") as f:
    tflite_bytes = f.read()
with SimpleInvoker(model_path) as model:
    # Int8 input: quantize to int8 then XOR 0x80 for Edge TPU uint8 domain
    input_int8 = (image_257x257.astype(np.float32) - 127).astype(np.int8)
    input_bytes = (input_int8.view(np.uint8) ^ 0x80).tobytes()
    raw_outputs = model.invoke_raw_outputs(input_bytes)
    poses = postprocess_multipose(raw_outputs, model.output_layers, tflite_bytes)
    for pose in poses:
        print(f"Score: {pose.score:.2f}, keypoints: {pose.keypoints.shape}")

# DeepLabV3: semantic segmentation (21 PASCAL VOC classes)
from libredgetpu.postprocess.deeplabv3 import postprocess_deeplabv3
with open("deeplabv3_edgetpu.tflite", "rb") as f:
    tflite_bytes = f.read()
with SimpleInvoker("deeplabv3_edgetpu.tflite") as model:
    raw_outputs = model.invoke_raw_outputs(input_bytes)
    seg_map = postprocess_deeplabv3(raw_outputs, model.output_layers, tflite_bytes)
    # seg_map: [33, 33] array of class indices (0-20)
```

For implementation details (anchor extraction, NMS parameters, output tensor identification,
relayout handling), see the [Advanced Documentation](docs/ADVANCED_README.md).

## Testing

```bash
pip install -e ".[dev]"

# All offline tests (no hardware needed)
pytest tests/ -v --ignore=tests/test_hardware.py -k "not hardware"

# Individual test suites (offline)
pytest tests/test_parse_models.py -v       # TFLite/DarwiNN parsing
pytest tests/test_matmul_engine.py -v -k "not hardware"   # MatMulEngine
pytest tests/test_spot_tracker.py -v -k "not hardware"   # SpotTracker
pytest tests/test_looming.py -v -k "not hardware"        # LoomingDetector
pytest tests/test_pattern_tracker.py -v -k "not hardware" # PatternTracker
pytest tests/test_transport_guards.py -v     # USB safety checks
pytest tests/test_error_handling.py -v       # Error handling edge cases
pytest tests/test_tflite_builder.py -v       # TFLite builder
pytest tests/test_optical_flow.py -v -k "not hardware"  # OpticalFlow
pytest tests/test_visual_compass.py -v     # VisualCompass
pytest tests/test_reservoir.py -v          # ReservoirComputer
pytest tests/test_embedding_similarity.py -v  # EmbeddingSimilarity

# Hardware tests (requires USB Edge TPU plugged in)
pytest tests/ -v --run-hardware

# Benchmark vs tflite_runtime
python -m tests.benchmark_vs_tflite

# Visual proof tests (hardware + Pillow required, generates annotated images)
python -m tests.test_visual
python -m tests.test_visual --models classification detection
python -m tests.test_visual --image photo.jpg

# Visual proof tests for robotics modules (hardware + Pillow required)
python -m tests.test_visual_robotics
python -m tests.test_visual_robotics --models spot_tracker spot_tracker_color looming
```

Produces annotated output images in `tests/results/`:
- **Classification**: top-5 labels on Grace Hopper (MobileNet V1)
- **Detection**: bounding boxes with class labels (SSD MobileDet)
- **Segmentation**: PASCAL VOC class overlay (DeepLabV3)
- **Pose**: skeleton with 17 keypoints (PoseNet, single-person model)
- **MultiPose**: per-person colored skeletons (MultiPose PoseNet, multi-person)
- **SpotTracker**: 3x3 grid of Gaussian dots with expected/detected crosshairs
- **LoomingDetector**: 3 synthetic scenes with zone heatmaps and tau values
- **PatternTracker**: checkerboard template tracked at 3 positions
- **MatMulEngine**: reversal transform, correlation scatter, error histogram

**Test Suite:** 517 total tests (449 offline + 68 hardware) — all pass ✅

## Interactive GUI

For real-time visual testing with live webcam and mouse interaction, libredgetpu includes
a web-based GUI that supports all 9 algorithms:

```bash
pip install -e ".[gui]"          # Install Flask + OpenCV dependencies
python -m libredgetpu.gui        # Launch web server (http://localhost:5000)
python -m libredgetpu.gui --cpu-replica --synthetic --pattern panning  # CPU integer-replica mode
```

Features:
- **Live webcam input** with algorithm-specific overlays (crosshairs, heatmaps, flow vectors)
- **Tunable parameter controls** — each algorithm exposes its parameters (image size, pool factor,
  temperature, FOV, spectral radius, etc.) via a schema-driven panel with an Apply button.
  Change parameters at runtime without restarting the server.
- **Pause/Play controls** (freeze frame with Spacebar for precise template selection)
- **Mouse interaction** (click to place targets, drag to select templates, etc.)
- **Hardware, synthetic, and CPU replica modes** (works offline without Edge TPU)
- **CPU replica mode** (`--cpu-replica`): faithfully reproduces the Edge TPU's integer arithmetic
  on CPU for OpticalFlow/VisualCompass — useful for debugging quantization and pipeline issues
- **Real-time metrics** (FPS, latency, algorithm-specific outputs)
- **Screenshot capture** with annotations

Algorithm interactions:
- **SpotTracker**: Click to place synthetic spot
- **PatternTracker**: Pause, then drag to select square template (8×8 to 32×32 auto-resize)
- **LoomingDetector**: Watch zone heatmap update
- **OpticalFlow**: Move camera to see flow vectors
- **VisualCompass**: Rotate camera for yaw angle
- **MatMulEngine**: Click to randomize weights, see scatter plot
- **ReservoirComputer**: Click to inject spike, watch state evolution
- **EmbeddingSimilarity**: Click to add frames to gallery, see top-3 matches
- **SimpleInvoker**: Select from 5 pre-trained models (Classification, Detection, Segmentation, PoseNet, MultiPose) with automatic post-processing and overlay visualization

See [`docs/GUI_GUIDE.md`](docs/GUI_GUIDE.md) for full usage guide and [`libredgetpu/gui/README.md`](libredgetpu/gui/README.md) for developer docs.

## Requirements

- Google Coral USB Accelerator
- Python >= 3.9
- Linux (any architecture — tested on Ubuntu 22.04 x86-64, Raspberry Pi OS ARM)
- `pyusb >= 1.2.0`, `numpy >= 1.21.0`, `flatbuffers >= 2.0`

**Not required at runtime**: TensorFlow, tflite_runtime, libedgetpu, edgetpu_compiler.

**Template generation** (all model types): Only `edgetpu_compiler` is needed (tested with version 16.0.384591198).
TFLite files are built directly using the `flatbuffers` library — no TensorFlow install required.

<details>
<summary><b>Legacy TF Workflow</b> (for custom model architectures)</summary>

For model architectures not covered by the built-in TFLite builder (e.g., custom Conv2D
topologies, novel quantization schemes), the original TensorFlow workflow is still available:

```bash
pip install tensorflow>=2.15  # or tensorflow-cpu (~300 MB vs ~2 GB)
```

Tested with TF 2.15, 2.16, 2.17.  `edgetpu_compiler` is still required regardless.
</details>

## Key Constraints

- **8-bit precision**: The Edge TPU outputs uint8 (256 distinct values per element).
  Good for feature extraction, approximate compute, and learned policies. Not suitable
  for high-precision accumulation (EKF covariance, PID integrals).
- **Weight cache**: ~8 MB on-chip SRAM. Max cached square matrix: ~2896x2896.
- **Latency floor**: ~0.28 ms per inference (USB 2.0 round-trip).
- **Power**: ~2W per USB accelerator.
- **Device recovery**: A failed USB transfer can make the device disappear. Physical
  replug is required to reset.

## Acknowledgments

This project was built through independent analysis of the Edge TPU hardware.
The following projects and resources were invaluable references:

- [geohot/edgetpuxray](https://github.com/geohot/edgetpuxray) — Hardware-level Edge TPU exploration: assembler/disassembler, register maps, DarwiNN FlatBuffer schema, and direct USB inference without libedgetpu
- [OliVis/coral](https://github.com/OliVis/coral) — Hardware-accelerated FFT on the Edge TPU, demonstrating custom computation beyond ML inference
- [libedgetpu](https://github.com/google-coral/libedgetpu) (Google) — Official Edge TPU runtime, used as reference for the USB protocol and execution model
- [pycoral](https://github.com/google-coral/pycoral) (Google) — Reference for post-processing implementations (PoseNet, DeepLabV3)
- [feranick/libedgetpu](https://github.com/feranick/libedgetpu) — Rebuilt libedgetpu binaries used for benchmarking
- [Q-Engineering](https://qengineering.eu/google-corals-tpu-explained.html) — Architectural overview of the Edge TPU systolic array and tiling
- Google Patents: [US20190050717A1](https://patents.google.com/patent/US20190050717A1/en), [US20180197068A1](https://patents.google.com/patent/US20180197068A1/en), [GB2558980A](https://patents.google.com/patent/GB2558980A/en), [US20210373895A1](https://patents.google.com/patent/US20210373895A1/en) — Tile architecture, ISA, TTU registers, and memory access patterns

## License

Apache License 2.0
