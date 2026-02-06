# Edge TPU for Robotics: Status, Findings, and Roadmap

This document consolidates all research and implementation work on using the
Google Coral Edge TPU for robotics applications beyond standard ML inference.
It replaces `EDGETPU4ROBOTICS.md`, `opticalflow_edgetpu.md`, and
`robotics_vision_edgetpu.md`.

---

## What We Built (Working, Hardware-Verified)

### libredgetpu — Pure-Python Edge TPU Driver
No libedgetpu, no tflite_runtime. Three deps: `pyusb`, `numpy`, `flatbuffers`.
Runs on RPi4 (aarch64) with no compilation.

- Full hardware init (libedgetpu-style 8-phase sequence)
- DMA hint-driven execution for complex multi-subgraph models
- Smart parameter caching via `parameter_caching_token`
- Optional C extension for USB transfer speed (matches libedgetpu latency)
- 517 tests (449 offline + 68 hardware) — all pass ✅

**Tested models**: MobileNet V1/V2, Inception V1, EfficientNet-S, SSD MobileDet,
SSD MobileNet V1/V2, DeepLabV3, PoseNet — all with CPU post-processing where needed.

### MatMulEngine — Runtime Weight-Swapping Matrix Multiply
Pre-compiled Dense(N) templates with compiler-free parameter blob generation.

```python
from libredgetpu import MatMulEngine
import numpy as np

with MatMulEngine.from_template(256) as engine:
    engine.set_weights(my_matrix)  # microseconds, pure NumPy
    y = engine.matmul(x)           # ~0.28 ms, float32 in/out
```

- Templates: 256, 512, 1024 (pre-compiled, ship with package)
- `set_weights()` generates DarwiNN param blobs directly — no `edgetpu_compiler`
  needed at runtime (Experiment 3: 25/25 byte-perfect matches vs compiler)
- Works on ARM (RPi4) — no x86 dependency at runtime
- Weight constraint: `[-128 * weight_scale, 127 * weight_scale]`

### LoomingDetector — Collision Avoidance
Edge density in 3x3 spatial zones using fixed Sobel kernels.

```
Image → Sobel_X(/8) + Sobel_Y(/8) → Square → Add → AvgPool(H/3,W/3) → [9] zones
```

- Tau = center / mean(periphery): `>1` = approaching, `<1` = receding
- 64x64: 1.0 ms, 128x128: ~1.5 ms
- All ops fully on Edge TPU, CPU only computes tau from 9 scalars
- 17 tests (12 offline + 5 hardware)

### SpotTracker — Visual Servoing
Soft argmax centroid for (x_offset, y_offset) from image center.

```
Image [H,W,1] → Reshape [H*W] → Softmax → Dense(1) x2 → Concat [x_off, y_off]
```

- Templates: 16x16, 64x64, 128x128, all fully on Edge TPU
- Uses Dense layers (FULLY_CONNECTED), not Conv2D — avoids compiler kernel limits
- Strategic representative dataset forces full output range at all sizes
- Color variant supported (1x1 Conv2D color filter → soft centroid)
- Runtime color swapping via `set_color()` — patches 3 bytes in DarwiNN blob
- 55 tests (38 offline + 17 hardware)

### PatternTracker — Template Matching
Locates a reference patch within a larger search image using Conv2D sliding
cross-correlation + soft argmax peak detection.

```
Image [1,H,W,C] → Conv2D(kernel, valid) → ReLU → Reshape → Softmax → Dense(1) x2 → Concat [x_off, y_off]
```

- Templates: 64x64/8x8, 64x64/16x16, 128x128/16x16, 128x128/32x32 (grayscale + RGB)
- Runtime template swapping via `set_template()` — patches Conv2D weights
- Supports grayscale (1-channel) and RGB (3-channel) inputs
- All ops fully on Edge TPU, CPU only does final coordinate extraction
- 35 tests (26 offline + 9 hardware)

### OpticalFlow — Global Ego-Motion Estimation
Edge TPU Gabor feature extraction + CPU global correlation with soft argmax.

```
Standard: Frame [H,W,1] → Conv2D(8 Gabor, SAME, ReLU) → [H,W,8] (×2) → CPU pool+corr → (vx,vy)
Pooled:   Frame [H,W,1] → Conv2D(8 Gabor) → AVG_POOL(P×P) → [H/P,W/P,8] (×2) → CPU corr → (vx,vy)
```

- Gabor kernels: 4 orientations (0/45/90/135) × 2 scales (sigma=1.5, 3.0), 7×7
- Templates: 64×64, 128×128 (Gabor feature extraction model)
- **Pooled mode** (Gabor+Pool): fuses AVG_POOL_2D into Edge TPU model
  - USB transfer: 4KB vs 64KB per frame (16× reduction for 64×64, P=4)
  - Use `OpticalFlow.from_template(64, pooled=True)`
- CPU correlation: 81 displacements (±4 pooled pixels), overlap-normalized, sub-pixel via soft argmax
- Tunable: `search_range` (±N grid), `temperature` (softmax), `pool_factor` (CPU downsample)
- Two Edge TPU calls per frame pair (~0.5 ms each) + CPU (~1 ms)
- Total latency: ~1.5-2.5 ms standard; faster with pooled mode on bandwidth-limited systems
- All Gabor weights fixed — no training data needed
- 61 tests (54 offline + 7 hardware)

### VisualCompass — Yaw Estimation (Wrapper)
Thin wrapper around OpticalFlow that converts horizontal displacement (vx) into
a yaw angle using the camera's field-of-view.

```
OpticalFlow.compute() → (vx, vy) → yaw_deg = vx × fov_deg × effective_pool / image_width
```

- Reuses OpticalFlow's Gabor + correlation pipeline entirely — no additional Edge TPU model
- `from_template(size, fov_deg, pooled=True)` creates both OpticalFlow and compass
- `compute_yaw(frame_t, frame_t1)` returns yaw in degrees
- `compute(frame_t, frame_t1)` returns `(yaw_deg, vx, vy)`
- `yaw_to_direction()` classifies as "left" / "right" / "center"
- Performance: same as OpticalFlow (~1.5-2.5 ms for 64×64)
- 25 offline tests (all mocked, no hardware needed)

### ReservoirComputer — Echo State Network
Fixed random recurrent network (Echo State Network) on the Edge TPU via
MatMulEngine composition.

```
Each timestep:
  Edge TPU:  h1 = W_res @ x(t-1)               # 0.28 ms via MatMulEngine
  CPU:       h2 = W_in  @ u(t)                  # microseconds
  CPU:       x(t) = (1-a)*x(t-1) + a*act(h1+h2) # leaky integration
  CPU:       y(t) = W_out @ x(t)                # ridge regression readout
```

- Reuses Dense(N) templates — no new Edge TPU model needed
- Reservoir weights generated with target spectral radius, clipped to engine range
- `fit()` trains readout via ridge regression (numpy lstsq)
- Optional `readout_engine` for large output dimensions on Edge TPU
- Activations: tanh (default), relu, identity — all on CPU
- ~0.6 ms per step at N=256 (~1.7 kHz update rate)
- 32 offline tests (all mocked, no hardware needed)

### EmbeddingSimilarity — Cosine Similarity Search
Stores L2-normalized embeddings as MatMulEngine weight rows.  A single matmul
computes cosine similarity against all database entries.  Pair with a MobileNet
backbone (via SimpleInvoker) for visual place recognition.

```
Database:  store scaled L2-normalized embeddings as weight rows
Query:     similarity = (W @ normalize(query)) / scale_factor   # 0.28 ms
Result:    argsort → top-K labels with cosine similarity scores
```

- Reuses Dense(N) templates — no new Edge TPU model needed
- Scale factor maps [-1, 1] normalized embeddings to engine's int8 weight range
- ~5-6 similarity levels across [0, 1] (int8 quantization); reliable for ranking
- Save/load stores pre-scale normalized embeddings for engine portability
- Capacity = matrix_size (e.g., 256 entries for Dense(256))
- ~0.28 ms per query (single matmul, USB-limited)
- 29 offline tests (all mocked, no hardware needed)

### Interactive GUI — Real-Time Visual Testing Tool
Web-based (Flask + OpenCV) interactive testing tool for all 9 algorithms.
Not a robotics module — a development/demo utility.

```bash
pip install -e ".[gui]"
python -m libredgetpu.gui --synthetic   # CPU fallback (no Edge TPU needed)
python -m libredgetpu.gui               # Auto-detect hardware
```

- Live webcam + algorithm-specific overlays (crosshairs, heatmaps, flow arrows)
- Mouse interaction (click to place targets, drag to select templates)
- Hardware and synthetic modes (works offline without Edge TPU)
- MJPEG streaming to any browser (desktop, mobile, over SSH)
- Real-time FPS/latency metrics via Server-Sent Events
- See `docs/GUI_GUIDE.md` for full usage guide

### Standalone Robotics Examples
Copy-paste-ready Python scripts in `examples/` for integration into robotics
platforms (Raspberry Pi, Jetson, etc.). One script per module with full argparse
parameter control and real-time webcam loop.

| Script | Module | Extras beyond GUI |
|--------|--------|-------------------|
| `spot_tracker_example.py` | SpotTracker | servo gain, dead-zone threshold |
| `looming_detector_example.py` | LoomingDetector | sliding-window TTC |
| `optical_flow_example.py` | OpticalFlow | all flow params exposed |
| `visual_compass_example.py` | VisualCompass | cumulative yaw, 'r' reset |
| `pattern_tracker_example.py` | PatternTracker | file template loading |
| `matmul_engine_example.py` | MatMulEngine | benchmark + NumPy verify |
| `reservoir_computer_example.py` | ReservoirComputer | seed, input scaling |
| `embedding_similarity_example.py` | EmbeddingSimilarity | save/load .npz |
| `simple_invoker_example.py` | SimpleInvoker | 5 model types, threshold |

All webcam scripts support `--no-display` for headless operation.
See `examples/README.md` for full parameter reference.

---

## The 8-Bit Output Wall (Critical Constraint)

```
int8_input × int8_weight → int32_accumulator → requantize → uint8_output
```

The MAC array accumulates in 32-bit internally, but output is always requantized
to uint8 (256 distinct values). The int32 accumulator is never exposed.

### What We Proved Doesn't Work

**Byte decomposition** — Split 16-bit values into byte halves, run 4 matmuls,
recombine. Fails because each intermediate result is requantized to uint8,
losing all but 8 bits before recombination.

**Multi-output tiling** — Multiple output branches with different quantization
scales to capture different accumulator bit-ranges. Experimentally tested two
variants (scaled weights, residual decomposition). Results for Dense(256):
```
Single Dense:        MAE = 3.591  (2.5 effective bits)
Coarse+Fine combo:   MAE = 3.592  (2.5 effective bits)  ← no improvement
```
Root cause: for N>=64, the bottleneck is accumulated int8 weight x input
quantization noise across N MACs, not the output requantization. Adding more
branches measures the same noisy accumulator from different angles.

**Hardware confirmation**: DarwiNN executables have single `zero_point` +
`dequantization_factor` per output layer. No registers, DMA params, or fields
to access wider accumulator bits. `DataType` supports FIXED_POINT16 in schema
but outputs are always FIXED_POINT8 in practice.

### What This Means

- Operations producing results naturally in 8 bits: perfect fit
- Approximate results acceptable (learned policies, feature extraction): good fit
- High-precision accumulation (covariance updates, PID integrals): CPU only
- Hybrid designs work: Edge TPU for bulk compute, CPU for precision-sensitive parts

---

## What We Discarded and Why

### Image Compression — NOT RECOMMENDED
Entropy coding (Huffman/RLE) is 30-50% of JPEG encoding and fundamentally
incompatible with systolic arrays. USB overhead (150ms) exceeds CPU time
(2-30ms) for DCT/DWT transforms. Realistic speedup vs CPU: 0.1x-0.5x (slower).

### Classical Optical Flow — NOT VIABLE

| Method | Blocking Issues |
|--------|-----------------|
| Lucas-Kanade | Per-pixel 2x2 matrix solve, needs 16+ bit precision, division |
| Horn-Schunck | Iterative (USB round-trip per iteration), division required |
| Block Matching | ArgMax not on Edge TPU, Abs not supported |
| Phase Correlation | FFT works, but normalization needs division |

Core problems: 8-bit output wall kills gradient products (need 15+ bits),
per-pixel 2x2 inverse can't batch efficiently, iterative methods need
50+ USB round-trips, no division/ArgMax on hardware.

### Virtual Optical Mouse (Raw Patch Correlation) — NOT VIABLE AS-IS
Using Frame(t-1) patch as conv weights requires recompilation every frame
(~50ms → 20 Hz max). Also ArgMax not supported. Fixable with fixed feature
extractors + soft argmax (see Optical Flow below).

### edgetpu_compiler Debug Output — DEAD END
73 MLIR passes, debug files (`*-instructions-lst.txt`, `*-chips-debug.txt`)
exist but are gated by two-level check: MLIR pass option + internal
compiler_options proto field. Binary patching attempted (flag redirect, boolean
flip, env vars) — all failed. Our own disassembler is more productive.

---

## What Remains Possible (Not Yet Implemented)

### Learned Controllers at High Frequency
Small MLP policy from sim-to-real RL: 64→64→64→action_dim.
~0.28 ms inference = 1+ kHz control loops at ~2W. Int8 is natural for RL
policies trained end-to-end. Best power/performance for battery-powered robots.

### Batch MPC with Learned Dynamics
Neural network forward model on Edge TPU. Stack N candidate state-action pairs
as rows → one matmul evaluates all rollouts. CPU handles cost evaluation and
trajectory selection (the cheap part).

### Other Linear Algebra
- **Kalman filter** state prediction (12x12 matrices, covariance stays on CPU)
- **Coordinate transforms** (batch point cloud rotation, ~1.4° resolution)
- **Custom linear transforms** (DCT, PCA, wavelets — any fixed `y = Wx`)
- **Sensor pre-processing** (FIR/IIR filters, matched filtering, spectral analysis)

---

## Hardware Performance Reference

### Inference Latency (cached mode)

| Model | Latency | Type |
|---|---|---|
| Dense 256-2048 | 0.28 ms | Matrix multiply (USB-limited) |
| Looming 64x64 | 1.0 ms | Collision avoidance |
| Looming 128x128 | ~1.5 ms | Collision avoidance |
| SpotTracker (all sizes) | ~1 ms | Visual servoing |
| PatternTracker 64x64 | ~5 ms | Template matching |
| OpticalFlow 64x64 | ~2 ms | Ego-motion estimation |
| ReservoirComputer 256 | ~0.6 ms | Echo State Network |
| EmbeddingSimilarity | ~0.28 ms | Cosine similarity search |
| MobileNet V1 | 4.0 ms | Classification |
| MobileNet V2 | 4.3 ms | Classification |
| Inception V1 | 5.4 ms | Classification |
| EfficientNet-S | 8.6 ms | Classification |
| SSD MobileDet | 11.4 ms | Detection |
| SSD MobileNet V1 | 10.1 ms | Detection |
| PoseNet | 17.0 ms | Pose estimation |
| DeepLabV3 | 26.3 ms | Segmentation |

### Weight Update Overhead

| Operation | Time |
|---|---|
| `set_weights()` fast path (compiler-free) | ~microseconds |
| Parameter upload 64 KB (Dense 256) | 1.3 ms |
| Parameter upload 1 MB (Dense 1024) | 5.6 ms |
| Parameter upload 8 MB (Dense ~2896) | ~25 ms |

### Key Constraints

- **Precision**: int8 in × int8 weight → uint8 out (8 bits, no accumulator access)
- **Weight cache**: ~8 MB on-chip SRAM (max cached square matrix ~2896x2896)
- **Latency floor**: ~0.28 ms per inference (USB 2.0 round-trip)
- **Power**: ~2W per USB accelerator
- **Runtime deps**: pyusb, numpy, flatbuffers only (no TF, no libedgetpu, no compiler)

---

## Edge TPU Supported Operations (v16)

**Supported**: Conv2D, DepthwiseConv2D, TransposeConv, FullyConnected, Add, Sub,
Mul, SquaredDifference, MaxPool, AvgPool, Concat, Split, Slice, Pad, Pack,
Maximum, Minimum, ReduceMax, ReduceMin, Sum, ReLU, ReLU6, Tanh, Logistic,
Softmax, L2Norm, ResizeBilinear, ResizeNearestNeighbor, SpaceToDepth, Reshape,
Transpose, Rsqrt, LSTM

**NOT supported**: ArgMax/ArgMin, Abs, Exp/Log/Pow, Gather/GatherND, Where/Select,
TopK, any custom ops (except edgetpu-custom-op)

**Key workarounds**:
- ArgMax → Soft argmax (Softmax + Mul + Sum)
- Abs → ReLU(x) + ReLU(-x), or x² via Mul(x,x)
- Shift/warp → Pad + Slice (pre-computed offsets)

---

## Applications Requiring Higher Precision (CPU-Only or Hybrid)

These need >8 bits and cannot run solely on the Edge TPU:

- **EKF covariance updates**: 16-32 bit for numerical stability
- **Precision coordinate transforms**: Sub-degree rotation needs ~12+ bits
- **PID integral terms**: Small error accumulation over time
- **Sensor fusion tight loops**: Combining noisy data with precise state estimates
- **Path planning cost matrices**: Fine-granularity distance computations
- **Sub-pixel optical flow**: Classical methods need 15+ bit gradient products

Hybrid approach: Edge TPU for bulk matmul / feature extraction, CPU for
precision-sensitive accumulation and decision-making.

---

## Hardware Analysis Highlights

- Full 128-bit ISA decoded (scalar ops, vector ops, DMA, branch/TTU)
- DarwiNN parameter blob format empirically determined (64-row groups, 4-column
  tiling, XOR 0x80 value transform) — enables compiler-free weight swapping
- Operations encoded in weights + quantization metadata, NOT in instructions
  (mul2/div2/add1 produce nearly identical instruction streams)
- Edge TPU always outputs uint8; int8 models need XOR 0x80 correction
- Hardware register map fully documented (1804 registers from beagle CSR)

See `HARDWARE_ANALYSIS.md` for the complete technical reference.

---

*Based on independent hardware analysis from the CustomEdgeTPU project.
Hardware-verified with libredgetpu (517 tests: 449 offline + 68 hardware, all pass ✅).
See `HARDWARE_ANALYSIS.md` for full ISA/hardware details.*
