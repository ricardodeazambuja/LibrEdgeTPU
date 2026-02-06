# libredgetpu Advanced Guide

Internal architecture, hardware analysis findings, template generation, and
development reference. Read [README.md](README.md) first for user-facing API.

---

## Module Architecture

```
libredgetpu/
  __init__.py              SimpleInvoker, MatMulEngine, LoomingDetector, SpotTracker, PatternTracker, OpticalFlow, VisualCompass, ReservoirComputer, EmbeddingSimilarity
  _base.py                 EdgeTPUModelBase: shared TFLite parsing, DarwiNN extraction,
                           USB lifecycle, cached/standalone execution protocol
  _constants.py            Hardware-determined constants: SIGN_BIT_FLIP, QUANT_EPSILON,
                           MAC_ARRAY_ROWS, WIDE_BUS_WIDTH
  simple_invoker.py        User API: invoke(), invoke_raw(), invoke_raw_outputs()
  matmul_engine.py         Runtime weight-swapping matrix multiply
  looming_detector.py      Collision avoidance via edge density zones
  spot_tracker.py          Visual servoing via soft argmax centroid
  pattern_tracker.py       Template matching via Conv2D correlation + soft argmax
  optical_flow_module.py   Global optical flow via Gabor features + CPU correlation
  visual_compass.py        Yaw estimation wrapper around OpticalFlow
  reservoir.py             Echo State Network via MatMulEngine composition
  embedding_similarity.py  Cosine similarity search via MatMulEngine composition
  tflite_parser.py         Manual TFLite flatbuffer reader (no TF dependency)
  tflite_builder.py        TFLite FlatBuffer builder (no TF dependency)
  delegate.py              DarwiNN Package > MultiExecutable > Executable extraction
  driver.py                Hardware init (8-phase) + execution orchestration
  transport.py             USB layer: firmware download, registers, bulk transfer
  _usb_accel.c             Optional C extension for direct libusb-1.0 transfers
  setup.py                 Build helper for the C extension
  registers.py             Hardware register constants (from beagle CSR)
  darwinn/                 Generated FlatBuffer bindings (from executable.fbs)
  templates/               Pre-compiled Dense(N) templates + sidecar JSON
  looming/                 Looming detection package + templates
  tracker/                 Spot tracker package + templates
  pattern/                 Pattern tracker package + templates
  optical_flow/            Optical flow package + templates
  postprocess/             CPU post-processing (SSD decoder, PoseNet decoder, MultiPose decoder, DeepLabV3)
  gui/                     Flask web GUI with all 9 algorithm modes + CPU replica
examples/                    Standalone robotics scripts — one per module, argparse, webcam loop
  _common.py               Shared webcam loop, FPS tracking, display, resize helpers
  spot_tracker_example.py  Visual servoing with servo error output
  looming_detector_example.py  Collision avoidance + TTC sliding window
  optical_flow_example.py  Ego-motion estimation with flow arrow overlay
  visual_compass_example.py  Yaw/heading accumulation with compass display
  pattern_tracker_example.py  Template matching with file/auto-crop loading
  matmul_engine_example.py  Benchmark (no webcam) with optional NumPy verification
  reservoir_computer_example.py  ESN with webcam sensory input heatmap
  embedding_similarity_example.py  Place recognition with gallery save/load
  simple_invoker_example.py  5 model types with full post-processing
experiments/                 Validation experiment scripts (not shipped)
tests/                       pytest suite (449 offline + 68 hardware tests, all pass ✅)
```

## Data Flow (Inference Pipeline)

```
User: model.invoke(float_input)
        |
        v
[1] tflite_parser.parse()
    Reads compiled *_edgetpu.tflite, finds edgetpu-custom-op,
    extracts customOptions blob + I/O quantization params.
        |
        v
[2] delegate.parse_darwinn()
    Unpacks DarwiNN Package > MultiExecutable > Executable.
    Extracts instruction bitstreams, parameters, field offsets,
    I/O layer info, DmaHints (step-by-step USB transfer order),
    and TileLayout tables (output de-scattering maps).
        |
        v
[3] EdgeTPUModelBase.__init__() [_base.py]
    Classifies executables (PC/EO/SA), determines cached vs standalone
    mode, pre-caches invariant bitstreams and parameters, creates
    USBTransport and EdgeTPUDriver. Subclasses (SimpleInvoker,
    SpotTracker, LoomingDetector) inherit this initialization.
        |
        v
[4] transport.USBTransport
    Opens USB device (vendor IDs 1a6e bootloader / 18d1 runtime).
    Auto-downloads firmware (apex_latest_single_ep.bin, SHA256-verified).
    Provides send(), read_output(), read_status(), write_register(),
    read_register(), poll_register().
        |
        v
[5] driver.EdgeTPUDriver
    8-phase hardware init (mirrors libedgetpu):
      Open > EnableReset > QuitReset > EnableHardwareClockGate >
      InitializeChip > DoRunControl > Interrupts > Misc
    Orchestrates execution: execute_dma_hints() follows DMA hint
    sequence for correct interleaved transfers.
    Delegation methods send_raw() and read_status_packet() wrap
    transport calls so invokers never access transport directly.
        |
        v
[6] EdgeTPUModelBase._execute_raw()
    Common execution protocol: handles parameter caching (via
    _ensure_params_cached()), dispatches to driver for both
    cached and standalone execution paths.
        |
        v
[7] Subclass-specific processing
    SimpleInvoker: quantize float->uint8, dequantize uint8->float
    SpotTracker: _normalize_input(), _resize_image(),
      _quantize_input(), _execute_raw(), _decode_output()
    LoomingDetector: quantize, _execute_raw(), dequantize zones
    OpticalFlow: extract_features() x2 (relayout_output() to
      de-scatter tiled Edge TPU output), _pool_features(),
      _global_correlation(), _soft_argmax() → (vx, vy)

[7b] Composition modules (wrap another engine, no new Edge TPU model)
    VisualCompass: wraps OpticalFlow, converts vx → yaw via FOV
    ReservoirComputer: wraps MatMulEngine for W_res @ x(t), CPU
      handles W_in, leaky integration, and readout (W_out)
    EmbeddingSimilarity: wraps MatMulEngine, stores L2-normalized
      embeddings as weight rows, unscales matmul output to cosine
    (See "Composition Modules" section below for full details)
```

## Algorithmic Foundations and References

This section provides the mathematical grounding for the algorithms used in each
module, with intuition and citations so anyone can understand, extend, or debug
the pipeline.

### Affine Quantization

Every module passes through int8/uint8 quantization. Understanding the mapping is
essential for debugging silent precision issues.

**Core formula.** A real value `r` maps to quantized value `q` via:

```
r = S * (q - Z)
q = clamp(round(r / S) + Z, q_min, q_max)
```

where `S` = scale (float), `Z` = zero_point (integer). This is an affine
(linear + offset) mapping that covers an arbitrary real range with uniform spacing.

**Asymmetric (uint8) vs Symmetric (int8):**

| Variant | Range | Zero Point | Used For |
|---------|-------|------------|----------|
| uint8 (asymmetric) | [0, 255] | typically ~127 | inputs, outputs |
| int8 (symmetric) | [-128, 127] | 0 | weights |

Weights use symmetric quantization (`Z_w = 0`, `S_w = max(|w|) / 127`) because
it eliminates the cross-term with `Z_w` during MAC accumulation, simplifying
the hardware datapath.

**MAC accumulation in int8 x int8 → int32.** For `y = W · x`:

```
y_real = S_w * S_x * Σᵢ q_w[i] * (q_x[i] - Z_x)
```

The hardware accumulates `Σ q_w[i] * q_x[i]` in a 32-bit integer — exact for
up to N=2^15 MACs with 8-bit operands (no overflow). The final requantization
to output uint8 is:

```
q_out = round(M * acc) + Z_out
M = (S_w * S_x) / S_out
```

The multiplier `M` is baked into EO (execution-only) instructions at compile time.

**The 8-bit output wall.** Only 256 distinct values survive requantization, regardless
of accumulator precision. The output spacing equals `S_out`. Practical impact:

- **MatMulEngine**: ~8-bit output ≈ 0.4% relative error for well-scaled inputs
- **SpotTracker**: soft argmax concentrates precision near the peak — sub-pixel
  accuracy despite 8-bit output
- **LoomingDetector**: zone averages smooth quantization noise — tau ratios are robust
- **EmbeddingSimilarity**: ~5-6 distinct similarity levels across [0, 1] — rankings
  reliable, absolute scores coarse

Precision is NOT lost in the MAC (int32 accumulator is exact for N ≤ 256); it is
lost in the final requantization clamp to [0, 255].

**Sign-bit flip (XOR 0x80).** The Edge TPU MAC array operates in unsigned arithmetic.
DarwiNN parameter blobs encode int8 weights as `blob_byte = int8_value XOR 0x80`,
mapping [-128, 127] → [0, 255] while preserving ordering. Outputs are flipped back.

**Representative dataset.** TFLite post-training quantization observes real data to
set scale/zero_point. Wrong range = silent garbage: e.g., feeding [0, 255] integers
gives `scale=1.0`, collapsing float inputs to {0, 1}. Each module's generator
constructs representative data covering the expected dynamic range.

> **References:** Jacob et al. (2018) "Quantization and Training of Neural Networks
> for Efficient Integer-Arithmetic-Only Inference" [[Jacob2018]]; Krishnamoorthi (2018)
> "Quantizing deep convolutional networks for efficient inference" [[Krishnamoorthi2018]]

### Soft Argmax

Used by SpotTracker, PatternTracker, and OpticalFlow's CPU correlation stage.

**Definition:**

```
pos = Σᵢ softmax(x / T)[i] * coords[i]
```

where `T` is temperature. This computes a weighted centroid of the score map,
replacing `ArgMax` (not supported on Edge TPU) with a differentiable, fully
on-chip alternative.

**Temperature effect:** `T → 0` sharpens the softmax toward hard argmax (winner
takes all). `T → ∞` flattens it toward the spatial mean (all positions equal).
SpotTracker uses `T ≈ 1` (default from model); OpticalFlow uses `T = 0.1` for
sharper peaks.

**On Edge TPU:** The Softmax → Dense(1) pattern implements soft argmax in two ops,
both fully mapped. Dense weights encode coordinate values (`coords[i]`).

> **References:** Chapelle & Wu (2010) "Gradient descent optimization of smoothed
> information retrieval metrics" [[Chapelle2010]]; Luvizon et al. (2019) "Human
> pose regression by combining indirect part detection and contextual information"
> [[Luvizon2019]]

### Gabor Filters

Used by OpticalFlow for orientation- and scale-selective texture feature extraction.

**2D Gabor function** (as implemented in `tflite_builder.py:1540-1544`):

```
g(x, y) = exp(-(x'² + y'²) / (2σ²)) * cos(2π y' / λ)
```

where `x' = x cos θ + y sin θ`, `y' = -x sin θ + y cos θ` are coordinates rotated
by orientation `θ`, `σ` controls the Gaussian envelope width, and `λ` the sinusoidal
wavelength.  The envelope is circular (γ=1, no aspect ratio parameter) because rotation
is handled by the orientation bank.  The carrier uses `y'` (not `x'`), so at θ=0
the carrier oscillates along Y, detecting horizontal edges.

For full derivation see [`OPTICAL_FLOW_THEORY.md`](OPTICAL_FLOW_THEORY.md).
For debugging history see [`OPTICAL_FLOW_ENGINEERING_LOG.md`](OPTICAL_FLOW_ENGINEERING_LOG.md).

**Multi-orientation + multi-scale:** 8 kernels = 4 orientations (0/45/90/135°) × 2
scales (σ = 1.5, 3.0) capture texture information invariant to local phase. This
forms a compact texture descriptor.

**Half-wave rectification (ReLU):** Gabor responses are bipolar, but ReLU retains
only the positive phase. This is Edge TPU-friendly (ReLU is natively supported)
and preserves energy information: the negative phase carries redundant orientation
data.

**Quantized Gabor kernels:** 7×7 kernels are normalized to [-1, 1] per filter,
then quantized to int8. The quantization noise is small relative to kernel structure
(~0.8% of peak amplitude).

> **References:** Gabor (1946) "Theory of communication" [[Gabor1946]]; Daugman
> (1985) "Uncertainty relation for resolution in space, spatial frequency, and
> orientation" [[Daugman1985]]; Jain & Farrokhnia (1991) "Unsupervised texture
> segmentation using Gabor filters" [[Jain1991]]

### Cross-Correlation for Template Matching

Used by PatternTracker (on Edge TPU via Conv2D) and OpticalFlow (on CPU for
displacement search).

**Sliding dot product:**

```
C(dx, dy) = Σᵢⱼ T(i, j) * I(i + dx, j + dy)
```

This measures how well template `T` matches image `I` at each displacement `(dx, dy)`.
Implemented as Conv2D with the template as kernel — mathematically identical to
cross-correlation (Conv2D in TFLite uses correlation convention, not convolution).

**ReLU thresholding:** Negative correlation values indicate anti-matches. ReLU
discards them, improving soft argmax peak detection.

**Overlap normalization:** Border displacements have fewer overlapping pixels.
Dividing by the overlap area `(h - |dy|) * (w - |dx|)` prevents bias toward
the center. Used in both PatternTracker and OpticalFlow.

> **Reference:** Lewis (1995) "Fast Normalized Cross-Correlation" [[Lewis1995]]

### OpticalFlow Implementation Details

**Pipeline:**
1. **Edge TPU**: Extract Gabor features from both frames via DEPTHWISE_CONV_2D
2. **CPU**: `relayout_output()` — de-scatter tiled Edge TPU output to standard YXZ order
3. **CPU**: Downsample via block-sum pooling (or skip if `pooled=True` — already fused)
4. **CPU**: Compute global cross-correlation for 81 displacements (±4 pixels in pooled space)
5. **CPU**: Apply overlap normalization to prevent border bias
6. **CPU**: Soft argmax to get sub-pixel `(vx, vy)` displacement

**Output tiling and relayout (critical):** The Edge TPU stores convolution output in a
tiled memory layout (TYXZ format), not standard row-major YXZ. For a 64×64×8 output,
the TPU arranges data in a 4×4 grid of 16×16 tiles (16 tiles total). Without de-scattering,
the raw bytes appear spatially scrambled — horizontal structure partially survives
(16-column tiles) but vertical structure is destroyed (4-row tile boundaries break
vertical continuity).

`relayout_output()` in `delegate.py` uses six lookup tables extracted from the DarwiNN
executable's `TileLayout` to map each `(y, x)` position to the correct byte offset in
the raw buffer:
```
dest[y, x, :] = src[tile_byte_offsets[y_tile + x_tile]
                     + y_local * row_size[x]
                     + x_local_byte_offset[x] : +z_dim]
```

This is a pure permutation (no padding: `size_bytes == H*W*C` exactly). All models with
spatial convolution output need it — SpotTracker/PatternTracker don't because they
collapse spatial dimensions on-TPU via SOFTMAX→FC→CONCAT. The postprocessors (PoseNet,
DeepLabV3, MultiPose) already called `relayout_output()`; OpticalFlow was the only module
that was missing it.

**Pooled mode optimization:** Standard mode outputs full-resolution features (H×W×8)
requiring `H*W*8` bytes USB transfer. Pooled mode fuses AVG_POOL_2D into the Edge
TPU model, outputting (H/P, W/P, 8) for P²× smaller USB transfer (e.g., 64×64×8 =
32 KB → 16×16×8 = 2 KB for P=4). Both standard and pooled outputs require relayout.

**Why no mean subtraction:** For ReLU features (all non-negative), raw cross-correlation
already peaks at the true displacement. Mean subtraction converts it into a zero-mean
autocorrelation that peaks at lag 0, masking motion. Explanation: at the matched
displacement, `E[X²]` per pixel > `E[X·X_shifted]` at any other displacement.

**Pattern requirements for optical flow testing:**
- OpticalFlow estimates **global ego-motion** (camera/scene movement), not object tracking
- Requires textured patterns with GLOBAL translation: `panning`, `rotating`, or `checkerboard`
- **DO NOT use** `wandering_dot`: smooth Gaussian blob lacks texture; only the blob
  moves while background is static (optical flow cannot detect local object motion)
- For GUI testing: use `--pattern panning` for horizontal motion, `--pattern rotating`
  for rotational motion, or use a real webcam
- **GUI sensitivity**: The GUI uses `pooled=False, pool_factor=2` (not the library default
  of pool_factor=4) for better motion sensitivity at webcam resolution. At 640→64 downscale,
  the minimum detectable displacement is ~10px at 640×480 (vs ~20px with pool_factor=4).
  The panning pattern scrolls at 600 px/sec to produce ~1 pooled pixel/frame.

> **References:** Fleet & Weiss (2006) "Optical flow estimation" [[Fleet2006]];
> Barron et al. (1994) "Performance of optical flow techniques" [[Barron1994]]

### Sobel Edge Detection and Looming

Used by LoomingDetector for collision avoidance.

**Sobel operators:**

```
Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
      [-2, 0, 2],             [ 0,  0,  0],
      [-1, 0, 1]]             [ 1,  2,  1]]
```

**Edge magnitude:** Ideally `sqrt(Gx² + Gy²)`, but `Abs` is not supported on
Edge TPU. Instead: `Mul(Gx, Gx) + Mul(Gy, Gy)` — the squared magnitude, which
preserves the relative ordering of edge strengths.

**Tau (τ) theory:** An approaching object grows in the visual field. The ratio
of edge density between center and periphery zones encodes expansion:

```
τ = edge_density_center / edge_density_periphery
```

Time-to-contact is approximated from consecutive tau readings:

```
TTC ≈ Δt / (1 - τ(t) / τ(t-1))
```

When `τ(t) > τ(t-1)`, the object is approaching (looming); when decreasing, it
is receding.

**Sobel scaling by 1/8:** Prevents int8 overflow after squaring. Raw Sobel can
produce values up to ±4 × 255 = ±1020; squared = ~1M. Scaling by 1/8 gives
max ±127.5, squared ≤ 16256, safely within int8 × int8 → int32 range.

> **References:** Lee (1976) "A theory of visual control of braking based on
> information about time-to-collision" [[Lee1976]]; Nelson & Aloimonos (1989)
> "Obstacle avoidance using flow field divergence" [[Nelson1989]]; Sobel & Feldman
> (1968) "A 3x3 isotropic gradient operator for image processing" [[Sobel1968]]

### Echo State Networks

Used by ReservoirComputer.

**State update equation:**

```
x(t) = (1 - α) * x(t-1) + α * f(W_in · u(t) + W_res · x(t-1))
```

where `α` = leak_rate (controls memory decay), `f` = activation function (tanh,
relu, or identity), `W_in` = input projection (N × M, CPU), `W_res` = reservoir
matrix (N × N, Edge TPU), and `u(t)` = input signal.

**Readout:** Only the output layer is trained:

```
y(t) = W_out · x(t)
```

Training uses ridge regression: `W_out = (X^T X + αI)^{-1} X^T Y`, which has a
closed-form solution (no backpropagation through the reservoir).

**Echo state property:** For the reservoir to have fading memory (inputs eventually
forgotten), the spectral radius of `W_res` must satisfy `ρ(W_res) < 1`. The default
0.95 provides a balance between memory capacity and stability. Values closer to 1.0
give longer memory but risk instability; lower values give faster forgetting.

**Why int8 works:** Reservoir dynamics are inherently noise-tolerant. The random
reservoir already introduces stochasticity by design; quantization noise acts as
beneficial regularization, similar to dropout or weight noise injection.

> **References:** Jaeger (2001) "The echo state approach to analysing and training
> recurrent neural networks" (GMD Report 148) [[Jaeger2001]]; Lukoševičius & Jaeger
> (2009) "Reservoir computing approaches to recurrent neural network training"
> [[Lukosevicius2009]]

### Cosine Similarity Search

Used by EmbeddingSimilarity.

**Definition:**

```
cos(a, b) = (a · b) / (||a|| · ||b||)
```

With L2-normalized vectors (`||a|| = ||b|| = 1`), this simplifies to a dot product:
`cos(a, b) = a · b`. This is exactly a matrix-vector multiply — one MatMulEngine
call computes similarities against all database entries simultaneously.

**Int8 resolution:** With 256 output values spread across the similarity range,
the effective resolution is ~0.18 per quantization step (about 5-6 distinct levels
in [0, 1]). Rankings are reliable; absolute similarity scores are coarse.

> **Reference:** Salton, Wong & Yang (1975) "A vector space model for automatic
> indexing" [[Salton1975]]

### Visual Compass (Pixel Displacement to Yaw)

Used by VisualCompass.

**Conversion formula:**

```
deg_per_pooled_px = fov_deg * effective_pool / image_width
yaw_deg = vx * deg_per_pooled_px
```

where `effective_pool` is the active downsampling factor (fused_pool from the
Edge TPU model if > 0, otherwise the CPU pool_factor). This linear mapping
assumes pure rotation (no translation) — valid for small `dt` and distant scenes.

**Sign convention:** Positive `vx` (rightward pixel motion) corresponds to
positive yaw (rightward camera rotation viewed from above).

> **Reference:** Zufferey & Floreano (2006) "Fly-inspired visual steering of an
> ultralight indoor aircraft" [[Zufferey2006]]

### Base Class (`_base.py`)

`EdgeTPUModelBase` consolidates the shared infrastructure:
- **TFLite parsing + DarwiNN extraction**: `__init__` loads the model, classifies
  executables (PC/EO/SA), determines cached vs standalone mode
- **USB lifecycle**: `open()`, `close()`, context manager (`__enter__`/`__exit__`)
- **Execution protocol**: `_execute_raw(input_bytes)` handles parameter caching
  (via `_ensure_params_cached()`) and dispatches to the correct driver method
  based on cached/standalone mode and DMA hint availability
- **Overridable hooks**: `_default_output_size()` for subclass-specific fallback

Subclasses no longer access `_driver._t` directly. The driver exposes
`send_raw(data, tag)` and `read_status_packet()` as delegation methods that
wrap the transport calls.

### Shared Constants (`_constants.py`)

Hardware-determined values used across modules:
- `SIGN_BIT_FLIP = 0x80` — XOR mask for int8/uint8 conversion (param blobs and output)
- `QUANT_EPSILON = 1e-9` — prevents division by zero during quantization
- `MAC_ARRAY_ROWS = 64` — systolic array row count; param blobs use 64-row groups
- `WIDE_BUS_WIDTH = 4` — wide memory bus width; param blobs use 4-column tiling

### Shared Quantization Utilities (`_quantize.py`)

Centralized quantization/dequantization functions used by all modules. This eliminates
5 duplicated inline quantization formulas that previously had inconsistent epsilon guards.

- `quantize_uint8(array, scale, zero_point)` — float32 → uint8 input quantization
  with `max(scale, QUANT_EPSILON)` guard against zero/negative scale
- `quantize_int8(array, scale, zero_point)` — float32 → int8 weight quantization
  with the same epsilon guard
- `dequantize(array, scale, zero_point)` — uint8/int8 → float32 output dequantization

### Shared Tracker Methods (`_base.py`)

`EdgeTPUModelBase` provides shared tracker helper methods used by SpotTracker and
PatternTracker to avoid ~170 lines of code duplication:

- `_quantize_input(image)` — uses model's input quantization params
- `_normalize_tracker_input(image, resize, h, w, channels)` — shape/dtype/resize normalization
- `_resize_tracker_image(image, h, w, channels)` — block mean or area averaging resize
- `_decode_tracker_output(raw, y_offset, temperature)` — dequantize + offset correction

Subclasses pass their specific dimension values as arguments (e.g., SpotTracker passes
`self._height`/`self._width`, PatternTracker passes `self._search_height`/`self._search_width`).

## Composition Modules

Three modules use **composition** (wrapping another engine instance) rather than
inheritance from `EdgeTPUModelBase`. They do not introduce new Edge TPU models —
they reuse existing templates and add CPU-side logic.

### VisualCompass

Wraps `OpticalFlow` to convert pixel displacement into a yaw angle.

**Architecture:**

```
OpticalFlow.compute(frame_t, frame_t1) → (vx, vy)
VisualCompass: yaw_deg = vx * deg_per_pooled_px
```

**Conversion derivation:** The horizontal displacement `vx` is in units of pooled
pixels. One pooled pixel corresponds to `effective_pool` raw pixels, and the camera's
horizontal FOV spans `image_width` raw pixels. Therefore:

```
deg_per_pooled_px = fov_deg * effective_pool / image_width
```

where `effective_pool = fused_pool` if the template uses on-chip pooling (> 0),
otherwise `pool_factor` (CPU downsampling).

**Sign convention:** Positive `vx` = rightward scene motion = rightward camera
rotation (positive yaw). This matches the common robotics convention where positive
yaw is clockwise viewed from above.

**Pooled vs standard mode:** With `pooled=True` (default), the Edge TPU outputs
already-downsampled features — no CPU pooling needed, 16× less USB bandwidth.
Standard mode applies CPU block-sum pooling to full-resolution features.

**`from_template()` parameters:**
- `size` — square image dimension (e.g., 64)
- `fov_deg` — horizontal FOV in degrees (0, 360]
- `pooled` — use Gabor+Pool template (default True)
- `search_range` — max displacement in pooled pixels (default 4)
- `temperature` — softmax sharpness (default 0.1)
- `pool_factor` — CPU pooling factor for standard mode (default 4)

**Lifecycle:** Delegates to the wrapped `OpticalFlow` instance. `open()`, `close()`,
and context manager all pass through.

### ReservoirComputer

Wraps `MatMulEngine` to implement an Echo State Network.

**Architecture:**

```
Per timestep (~0.6 ms total):
  Edge TPU:  h1 = W_res @ x(t-1)                  [0.28 ms via MatMulEngine]
  CPU:       h2 = W_in  @ u(t)                     [microseconds, W_in is NxM]
  CPU:       x(t) = (1-α)*x(t-1) + α*f(h1 + h2)   [leaky integration]
  CPU:       y(t) = W_out @ x(t)                   [readout, trained via ridge regression]
```

**Why W_res on Edge TPU:** The reservoir matrix `W_res` is N × N (e.g., 256 × 256),
making it the dominant compute cost — O(N²) per timestep. `W_in` (N × M) is small
(M is typically 1-10), and `W_out` (K × N) is applied once. Offloading `W_res` to
the Edge TPU cuts per-step latency from ~1 ms (CPU) to ~0.3 ms.

**`fit()` training:**
1. Reset state to zeros
2. Drive reservoir with training inputs, collecting states `X ∈ R^{T×N}`
3. Discard first `warmup` steps (transient)
4. Solve `W_out = argmin ||X·W_out^T - Y||² + α||W_out||²` via augmented least squares
5. Optionally load `W_out` onto a second `readout_engine` for large K

**`step()` inference:** Single `engine.matmul(state)` call + CPU scalar ops.

**Spectral radius default 0.95:** Chosen as a safe default that provides long memory
without instability. The reservoir weight matrix is generated from random normal
entries, scaled so its largest eigenvalue magnitude equals the target spectral radius.

**Weight clipping:** `W_res` values are clipped to the engine's representable weight
range (typically ~[-0.109, +0.107]). For a 256×256 random matrix with spectral
radius 0.95, most entries are already within this range (~0.06 magnitude), but
outliers are clipped. This can slightly reduce the effective spectral radius.

**Optional readout engine:** For large output dimensions K, a second `MatMulEngine`
can accelerate the readout multiply. `fit()` zero-pads `W_out` to [N, N] and loads
it onto this engine. `predict()` then uses Edge TPU for both reservoir and readout.

### EmbeddingSimilarity

Wraps `MatMulEngine` to perform cosine similarity search.

**Architecture:**

```
Setup: L2-normalize embeddings → scale to weight range → store as weight rows
Query: L2-normalize query → matmul (Edge TPU) → unscale → argsort → top-k
```

**Scaling strategy:** L2-normalized embeddings have values in [-1, 1], but the
engine's weight range is much narrower (e.g., [-0.109, +0.107] for Dense(256)).
The scale factor is `min(|w_min|, |w_max|)`:

```
stored_row = normalized_embedding * scale_factor    # fits in weight range
raw_score  = engine.matmul(query)                   # int8 matmul
cosine_sim = raw_score / scale_factor               # undo scaling
```

This works because for L2-normalized vectors `a` and `b`:
`(s·a) · (b) = s · (a · b) = s · cos(a, b)`.

**Resolution analysis:** With int8 quantization and the ~0.1 scale factor,
the output has ~256 possible values spanning the similarity range. After unscaling,
the effective cosine resolution is approximately `output_scale / scale_factor ≈ 0.18`.
This yields ~5-6 distinct similarity levels across [0, 1]. Rankings are preserved
(higher similarity always maps to higher output); absolute scores are coarse.

**Database operations:**
- `add(label, embedding)` — L2-normalize, scale, store in next available row
- `remove(label)` — shift subsequent rows up, zero-pad last row
- `set_database(labels, embeddings)` — bulk replace
- `query(embedding, top_k)` — lazy weight upload, matmul, unscale, argsort
- `query_batch(embeddings, top_k)` — sequential queries
- `save(path)` / `load(path)` — persist pre-scale normalized embeddings (portable across engines)

**Capacity:** Maximum database size = `matrix_size` (e.g., 256 for Dense(256)).
Each embedding occupies one row of the N × N weight matrix.

**Lazy weight upload:** The weight matrix is only uploaded to the Edge TPU when a
query is issued and the database has changed since the last upload. This avoids
unnecessary USB transfers during batch `add()` operations.

## DarwiNN Executable Types

The Edge TPU compiler produces DarwiNN executables embedded in the `edgetpu-custom-op`.
Models with weights <= ~8 MB use **cached mode** (two executables); larger models use
**streamed mode** (one executable).

| Type | Value | Purpose | Contains |
|------|-------|---------|----------|
| STAND_ALONE | 0 | Streamed mode (full model) | Instructions + parameters + I/O layers |
| PARAMETER_CACHING | 1 | Cached mode (load weights) | Instructions + parameters |
| EXECUTION_ONLY | 2 | Cached mode (run inference) | Instructions + I/O layers (+ sometimes params) |

**Cached mode protocol**:
1. Send PC instructions (tag=0) + PC parameters (tag=2) -> read status (0x82)
2. For each inference: send EO instructions (tag=0) + input (tag=1) [+ EO params if present] -> read output (0x81) -> read status (0x82)

The `parameter_caching_token` (uint64) identifies the cached weight set. When the token
matches a previous load, libredgetpu skips step 1 (5x-50x speedup).

## Output Tiling (TileLayout)

The Edge TPU stores convolution output in a tiled memory layout (TYXZ), not standard
row-major YXZ. Each output layer in the DarwiNN executable may contain a `TileLayout`
with six lookup tables that define the de-scattering permutation:

| Table | Indexed by | Purpose |
|-------|-----------|---------|
| `y_tile_id_map` | y coordinate | Maps row to tile row ID |
| `x_tile_id_map` | x coordinate | Maps column to tile column ID |
| `tile_byte_offsets` | tile ID (y+x) | Byte offset of each tile in raw buffer |
| `x_local_byte_offset` | x coordinate | Byte offset within tile row for this column |
| `y_local_y_offset` | y coordinate | Local row index within tile |
| `x_local_y_row_size` | x coordinate | Stride between rows within this tile column |

The de-scatter formula (`relayout_output()` in `delegate.py`):
```python
for y, x in product(range(Y), range(X)):
    tile_id = y_tile_id_map[y] + x_tile_id_map[x]
    base = tile_byte_offsets[tile_id] + y_local_y_offset[y] * x_local_y_row_size[x] + x_local_byte_offset[x]
    dest[y, x, :] = src[base : base + Z]
```

**Typical layout for 64×64×8 output**: 4×4 grid of 16×16-pixel tiles, 16 tiles total.
`size_bytes == H*W*C` exactly (no padding — pure permutation).

**Which modules need relayout:**
- **OpticalFlow** — spatial feature maps [H, W, 8] exposed to CPU for correlation. **Must** call `relayout_output()`.
- **Postprocessors** (PoseNet, DeepLabV3, MultiPose) — spatial outputs. Already call `relayout_output()`.
- **SpotTracker / PatternTracker** — spatial features are internal; SOFTMAX→FC→CONCAT collapses them on-TPU. No relayout needed.
- **MatMulEngine / SimpleInvoker** — 1D output (no spatial dimensions). No relayout needed.

When `tile_layout is None` (e.g., simple dense outputs), `relayout_output()` falls back
to a plain `np.frombuffer().reshape()`.

## DarwiNN Parameter Blob Format (Empirically Determined)

This is the key breakthrough enabling compiler-free `set_weights()`. The parameter
blob stores quantized weights in a specific layout that the MAC array reads directly.

### Structure: 64-Row Groups

For a Dense(N) model (N inputs, N outputs), the param blob is organized as:

```
Group 0: [overhead: 64*8 bytes][weights: 64*N bytes]
Group 1: [overhead: 64*8 bytes][weights: 64*N bytes]
...
Group K: [overhead: 64*8 bytes][weights: 64*N bytes]
```

where K = ceil(N / 64) groups. Total blob size = K * (64*8 + 64*N) bytes.

### Overhead Bytes (Weight-Independent)

The 64*8 = 512 bytes of overhead per group contain per-channel float32 requantization
multipliers. These are determined by the model architecture and quantization scales,
NOT by the weight values. For a given template, the overhead bytes are constant regardless
of what weights are loaded.

This means we can copy them from the original template blob and never recompute them.

### Weight Value Transform

```
blob_byte = int8_weight XOR 0x80
```

This is the same sign-bit flip used for Edge TPU output (int8 <-> uint8). The MAC array
operates in unsigned arithmetic internally.

### Weight Layout: 4-Column Tiling

Within each 64-row group, weights are arranged in 4-column tiles:

```python
def weight_offset(row, col, N):
    """Offset of weight[row][col] within the parameter blob."""
    group = row // 64
    group_start = group * (64 * 8 + 64 * N)  # skip previous groups
    overhead = 64 * 8                          # skip this group's overhead
    local_row = row % 64
    col_block = col // 4
    local_col = col % 4
    return group_start + overhead + col_block * (64 * 4) + local_row * 4 + local_col
```

The 4-column tiling reflects the MAC array's data bus width (4 bytes = 32 bits per cycle).

### Generating a Param Blob (Pure NumPy)

```python
def generate_param_blob(int8_weights, template_overhead, N):
    """Generate a DarwiNN parameter blob from int8 weight matrix.

    Args:
        int8_weights: [N, N] int8 numpy array
        template_overhead: bytes from template blob (512 bytes per 64-row group)
        N: matrix dimension
    """
    n_groups = (N + 63) // 64
    group_size = 64 * 8 + 64 * N
    blob = bytearray(n_groups * group_size)

    for g in range(n_groups):
        # Copy overhead from template
        src_off = g * 512
        dst_off = g * group_size
        blob[dst_off:dst_off + 512] = template_overhead[src_off:src_off + 512]

        # Place weights with XOR 0x80 and 4-column tiling
        for r in range(64):
            row = g * 64 + r
            if row >= N:
                break
            for c in range(N):
                val = int(int8_weights[row, c]) ^ 0x80
                off = dst_off + 512 + (c // 4) * (64 * 4) + r * 4 + (c % 4)
                blob[off] = val & 0xFF

    return bytes(blob)
```

This runs in microseconds vs ~50ms for the `edgetpu_compiler` fallback.

### Validation

25/25 byte-perfect matches across 5 sizes (64, 128, 256, 512, 1024) and 5 weight
patterns (zeros, identity, random, near-zero, extreme). See `experiments/exp3_param_blob_format.py`
and `experiments/exp3b_validate_blob_gen.py`.

## Weight Swapping Protocol (MatMulEngine)

### Fast Path (default, compiler-free)

Used when `param_overhead` is present in the sidecar JSON (all shipped templates have it).

1. User calls `engine.set_weights(float32_matrix)`
2. `quantize_weights()` converts to int8 using template's fixed `weight_scale`
3. `_generate_param_blob()` creates DarwiNN blob in NumPy:
   - Copies template overhead bytes (weight-independent requant multipliers)
   - Places int8 weights XOR 0x80 into 64-row groups with 4-column tiling
4. `set_weights_raw()` sends new PC params via USB tag=2

### Fallback Path (requires edgetpu_compiler)

Used when `param_overhead` is unavailable (old templates without the field).

1-2. Same as above
3. Patches int8 bytes into the **uncompiled** TFLite at the weight buffer offset
4. Runs `edgetpu_compiler` on the patched TFLite (~50ms)
5. Extracts new PC params from the compiled output

### Why Instructions Don't Change

The Edge TPU compiler bakes the **requantization multiplier** into EXECUTION_ONLY
instructions:

```
requant_multiplier = (input_scale * weight_scale) / output_scale
```

As long as these three scales are preserved (which they are, since we only change int8
values and keep the TFLite quantization metadata intact), the instructions are
deterministic. Experimentally verified: 5 different weight distributions produce
**byte-identical** EO and PC instructions. Only the PC parameter blob changes.

See `experiments/exp1c_fixed_scale.py` and `MATMUL_ENGINE_NOTES.md` for the full
experiment log.

## TFLite Builder

`tflite_builder.py` constructs valid TFLite FlatBuffer files directly using the
`flatbuffers` library, eliminating the ~2 GB TensorFlow dependency for template
generation.

### Supported Model Types

| Model | Builder function | Status |
|-------|-----------------|--------|
| Dense(N,N) | `build_dense(n)` | Implemented |
| SpotTracker (bright/color) | `build_spot_tracker(h, w, ...)` | Implemented |
| LoomingDetector | `build_looming(h, w, zones)` | Implemented |
| PatternTracker | `build_pattern_tracker(sh, sw, kh, kw, ...)` | Implemented |
| OpticalFlow (Gabor) | `build_optical_flow(h, w, ...)` | Implemented |

### How It Works

`build_dense(n)` constructs a 3-operator graph:
`QUANTIZE(uint8→int8) → FULLY_CONNECTED(int8) → QUANTIZE(int8→uint8)`

Quantization parameters are computed analytically (not via calibration):
- Input: `scale=2/255, zp=127` maps float [-1,+1] to full uint8 range
- Weights: symmetric int8 (`zp=0`), scale from `max(abs(w))/127`
- Output: `scale=input_scale*weight_scale*N`, `zp=fc_zp+128`

Returns `(tflite_bytes, metadata_dict)` where metadata matches the sidecar JSON
schema used by `MatMulEngine`.

### How to Extend

To add a new model type:
1. Add an option builder (e.g., `_build_conv2d_options` — stubs already exist)
2. Create a `build_<model>()` function following `build_dense()` as template
3. Wire it into the corresponding `*_gen.py` script

### Reference

The builder uses TFLite schema v3 field indices. Key references:
- FlatBuffer schema: `tensorflow/lite/schema/schema.fbs`
- Operator enum values: `BuiltinOperator` (e.g., QUANTIZE=114, FC=9)
- Options union: `BuiltinOptions` (e.g., FullyConnectedOptions=8)

## Template Generation (Dev-Time)

Templates are pre-compiled models that ship with the package. All model generation
requires only `edgetpu_compiler` (x86 only) — no TensorFlow needed.
Users never need any of these at runtime.

### Dense(N) Templates (MatMulEngine)

```bash
python -m libredgetpu.template_gen --sizes 256 512 1024 2048
```

Creates three files per size:
- `dense_{n}_edgetpu.tflite` — compiled Edge TPU model
- `dense_{n}.tflite` — uncompiled quantized TFLite (for fallback recompilation)
- `dense_{n}_edgetpu.json` — sidecar with `weight_scale`, `input_scale`, `output_scale`,
  `param_size`, and `param_overhead` (base64-encoded group headers for compiler-free blob gen)

**Representative dataset**: `uniform(-1, 1)` giving `input_scale ~ 0.00784, zp ~ 127`.
This ensures float [-1, 1] maps to the full uint8 [0, 254] range. An earlier bug used
[0, 255] integers, giving `input_scale = 1.0` which made float inputs useless (all
mapped to 0 or 1).

### SpotTracker Templates

```bash
# Grayscale (bright spot tracking)
python -m libredgetpu.spot_tracker_gen --sizes 16 64 128

# Color tracking (7 presets)
python -m libredgetpu.spot_tracker_gen --sizes 64 128 --variant color_red
python -m libredgetpu.spot_tracker_gen --sizes 64 128 --all-colors
```

**Architecture (grayscale)**: `Input [H,W,1] -> Reshape [H*W] -> Softmax -> Dense(1) x2 -> Concat [x, y]`

**Architecture (color)**: `Input [H,W,3] -> Conv2D 1x1 (color filter) -> ReLU -> Reshape -> Softmax -> Dense(1) x2 -> Concat [x, y]`

Key design decisions:
- Uses **Dense layers** (FULLY_CONNECTED in TFLite), not Conv2D. TF 2.20 no longer
  auto-folds large Conv2D to FULLY_CONNECTED, causing compiler crashes at >= 64x64.
  Dense layers always produce FULLY_CONNECTED and work at all sizes.
- **Y offset**: Y coordinates are shifted by +1/temperature to force different quantization
  from X, preventing the Edge TPU compiler from merging the two Dense computations.
- **Strategic representative dataset**: Includes bright-spot images at corners/edges to
  force the quantizer to observe the full [-10, +10] output range. Without these, large
  images produce near-zero outputs from random noise (softmax spreads uniformly), causing
  output_scale clipping.

Color filter presets (RGB coefficients for 1x1 Conv2D):

| Color | R | G | B | weight_scale |
|-------|---|---|---|---|
| red | 1.0 | -0.5 | -0.5 | 0.007874 |
| green | -0.5 | 1.0 | -0.5 | 0.007874 |
| blue | -0.5 | -0.5 | 1.0 | 0.007874 |
| yellow | 0.5 | 0.5 | -1.0 | 0.007874 |
| cyan | -0.5 | 0.5 | 0.5 | 0.003937 |
| magenta | 0.5 | -0.5 | 0.5 | 0.003937 |
| white | 0.33 | 0.33 | 0.33 | 0.002598 |

Custom colors via CLI: `--color-weights R G B` registers custom coefficients under the
variant name (e.g., `--variant color_orange --color-weights 1.0 0.5 -0.5`).

### Runtime Color Swapping (set_color)

The Conv2D color filter weights occupy exactly **bytes [0, 1, 2]** of the DarwiNN
parameter blob (verified by diffing all 7 compiled color templates — only these 3 bytes
differ). This enables runtime color swapping without recompilation.

**Encoding**: `blob_byte = round(coefficient / color_weight_scale) XOR 0x80`

**`set_color([R, G, B])`** patches those 3 bytes and forces param re-upload:

1. Quantize `[R, G, B]` to int8 using the template's `color_weight_scale`
2. XOR 0x80 each value (same sign-bit flip as all DarwiNN weight encoding)
3. Write to blob offsets [0, 1, 2]
4. Invalidate `_cached_token` so next `track()` re-uploads the param blob

**Template choice for `set_color()`**: Use red/green/blue/yellow (scale=0.007874)
as the base template — they cover coefficients in [-1.0, +1.0] with ~128 levels per
channel. The cyan/magenta templates have a narrower scale (0.003937, range [-0.5, +0.5])
and white is even narrower (0.002598, range [-0.33, +0.33]).

**`from_color_weights([R, G, B])`**: Finds the closest pre-compiled color template
by Euclidean distance in coefficient space. Returns the matched variant and distance.
No compiler needed.

**Sidecar JSON field**: `color_weight_scale` (float) — the Conv2D per-channel
quantization scale. Stored during template generation. Used by `set_color()` to quantize
new coefficients. If missing, inferred from the preset that compiled the template.

### Looming Templates

```bash
python -m libredgetpu.looming_gen --sizes 64 128 256
```

**Architecture**: `Input [H,W,1] -> Conv2D(Sobel_X, /8) + Conv2D(Sobel_Y, /8) -> Mul(self) -> Add -> AvgPool(H/3, W/3) -> Reshape [9]`

Key design decisions:
- Sobel kernels scaled by 1/8 to prevent int8 overflow after squaring
- Squared magnitude (Gx^2 + Gy^2) instead of |Gx| + |Gy| because Abs isn't on Edge TPU
- AvgPool kernel = (H/3, W/3) divides image into exactly 3x3 zones
- All ops fully mapped to Edge TPU

### PatternTracker Templates

```bash
python -m libredgetpu.pattern_tracker_gen --standard
python -m libredgetpu.pattern_tracker_gen --search-sizes 64 --template-sizes 16 --channels 3
```

**Architecture**:
```
Input [1, H, W, C]
    |
Conv2D(kernel=[h,w,C,1], valid, no bias) -- sliding correlation
    |
ReLU -- threshold negative correlations
    |
Reshape [(H-h+1)*(W-w+1)] -- flatten correlation map
    |
Softmax -- peak detection (replaces unsupported ArgMax)
    |
+-- Dense(1) × X_coords --> x position
+-- Dense(1) × Y_coords --> y position (+ 1/T offset)
    |
Concat --> [x_off, y_off] in [-1, +1]
```

Key design decisions:
- Conv2D sliding correlation is mathematically identical to cross-correlation (template matching)
- ReLU thresholds negative correlations: a good match produces positive values
- Softmax + Dense replaces ArgMax (not supported on Edge TPU) with differentiable soft argmax
- Y coordinate offset by +1/temperature forces different quantization from X (prevents Edge TPU from merging the two Dense computations)
- Correlation map size = (search_h - kernel_h + 1) × (search_w - kernel_w + 1) determines model complexity; maps >~13K positions may time out in the compiler

**Template swapping**: `set_template()` patches Conv2D weights in the uncompiled TFLite, recompiles with `edgetpu_compiler`, and extracts the new parameter blob. This requires `edgetpu_compiler` on the PATH (~50ms per swap on x86). Future optimization: determine the Conv2D blob layout (Experiment 4) for compiler-free swapping.

**Shipped templates**: 6 combinations covering grayscale and RGB:
- `64x64/8x8/1ch` (corr 57×57, ~7 MB compiled)
- `64x64/16x16/1ch` (corr 49×49, ~5 MB)
- `128x128/16x16/1ch` (corr 113×113, ~27 MB)
- `128x128/32x32/1ch` (corr 97×97, ~20 MB)
- `64x64/8x8/3ch` (corr 57×57, ~7 MB)
- `128x128/16x16/3ch` (corr 113×113, ~27 MB)

### OpticalFlow (Gabor) Templates

```bash
python -m libredgetpu.optical_flow_gen --sizes 64 128
python -m libredgetpu.optical_flow_gen --pooled --pool-factor 4 --sizes 64
```

**Standard architecture**: `Input [H,W,1] → QUANTIZE(uint8→int8) → DEPTHWISE_CONV_2D(8 Gabor, SAME, ReLU) → QUANTIZE(int8→uint8) → Output [H,W,8]`

**Pooled architecture** (Gabor+Pool): `Input [H,W,1] → QUANTIZE → DEPTHWISE_CONV_2D(8 Gabor, SAME, ReLU) → AVG_POOL_2D(P×P, stride P) → QUANTIZE → Output [H/P,W/P,8]`

Key design decisions:
- Uses DEPTHWISE_CONV_2D (weight shape `[1, 7, 7, 8]`, depth_multiplier=8) with per-channel quantization
- 8 fixed Gabor kernels: 4 orientations (0/45/90/135) × 2 scales (sigma=1.5, 3.0)
- 7×7 kernels with SAME padding preserve spatial dimensions
- ReLU activation (half-wave rectified Gabor responses)
- Output is uint8 feature maps in **tiled TYXZ format** — `relayout_output()` must be called
  to de-scatter into standard YXZ before CPU correlation (see "Output tiling and relayout" above)
- No training data needed — all weights are analytically computed Gabor functions
- CPU cost is trivial: 81 correlations on 16×16×8 features ≈ 166K ops
- **Pooled mode** fuses AVG_POOL_2D into the model, reducing output from H×W×8 to (H/P)×(W/P)×8.
  For 64×64 with P=4: output shrinks from 32KB to 2KB per frame — **16× USB reduction**.
- **Overlap normalization**: correlation scores are divided by the overlap pixel count
  `(ph - |dy|) × (pw - |dx|) × nf` for each displacement, so border positions with smaller
  overlap don't get penalized. Pre-computed in `__init__` as `_overlap_counts`.
- **No mean subtraction**: for ReLU features (zp=0, all positive), mean subtraction converts
  the cross-correlation into an autocorrelation that always peaks at lag 0, masking the true
  displacement. Without it, the matched position gives E[X²] per pixel > E[X·X_shifted].
- **Tunable parameters** (passed to `__init__` or `from_template`):
  - `search_range` (default 4): ±N pixel displacement grid → (2N+1)² scores (default 81)
  - `temperature` (default 0.1): softmax sharpness for soft argmax
  - `pool_factor` (default 4): CPU block-sum downsampling factor (standard mode only)
- **Sidecar JSON fields**: `fused_pool` (int, 0 = standard, >0 = fused pool factor),
  plus standard keys (`height`, `width`, `num_filters`, `input_scale`, `output_scale`, etc.)

**Shipped templates**: 64×64, 128×128 (Gabor features only; CPU does flow computation).
Pooled templates are generated on demand with `--pooled`.

## USB Protocol Details

### Endpoints
| Endpoint | Direction | Purpose |
|----------|-----------|---------|
| EP1 | OUT (write) | Instructions, input activations, parameters |
| EP 0x82 | IN (read) | Execution completion status |
| EP 0x81 | IN (read) | Output activations |

### USB Framing
Every USB transfer has a header: `[uint32 length][uint32 tag]` followed by data.

| Tag | Content |
|-----|---------|
| 0 | Instructions (DarwiNN bitstream) |
| 1 | Input activations |
| 2 | Parameters (weights) |
| 3 | Output activations |
| 4-7 | Interrupts |

### DMA Hints

Complex models (PoseNet, DeepLabV3) require transfers in a specific interleaved order.
DarwiNN executables contain a `DmaHints` table that specifies this order. Each hint step
has a type (instruction, input, output, parameter, interrupt, fence) and byte offsets/sizes.

Without following hints, these models hang silently. The `execute_dma_hints()` method in
`driver.py` handles this correctly.

**Split-input DMAs**: PoseNet sends input as 2 overlapping DMAs (~490KB + ~471KB with
~36KB overlap). The overlap ensures the pipeline doesn't stall at narrow memory boundaries.
`execute_dma_hints()` zero-pads input to cover `max(offset + size)` across all input steps.

### Read Output Gotchas

- `read_output(max_size)` can return MORE or FEWER bytes than `max_size`. USB data
  doesn't align with DMA hint output step sizes.
- Chunked reads must always request 32768 bytes per chunk. Smaller requests trigger
  `LIBUSB_ERROR_OVERFLOW` if the device sends a full 512-byte USB packet.
- The C extension allocates `max_size + 32768` headroom for this reason.

## Hardware Init Sequence (8 Phases)

Mirrors libedgetpu's initialization. Key register: `scu_ctrl_3` at offset 0x1A300C.

1. **Open**: Clear PHY error bits
2. **EnableReset**: Force sleep (bits [23:22] = 3), poll power state (bits [9:8] = 3),
   pulse `gcbb_credit0` register
3. **QuitReset**: Max clock divider (bits [21:20] = 0), poll `scalarCoreRunControl` = 0,
   poll `tileconfig0`
4. **EnableHardwareClockGate**: Set clock gating bits
5. **InitializeChip**: Configure tile memory capacity, instruction memory, etc.
6. **DoRunControl**: Enable run control register
7. **Interrupts**: Configure interrupt registers
8. **Misc**: Final configuration

Register map: 1804 named registers in `beagle_csr_offsets.h`.

## Edge TPU Output Byte Convention

The Edge TPU hardware **always outputs uint8 bytes** (0-255), regardless of the TFLite
model's declared output type.

For models with **uint8 output** (most models: MobileNet, SSD, etc.): use bytes directly.

For models with **int8 output** (e.g., SpotTracker): the raw byte has its sign bit flipped
relative to what CPU TFLite would produce:

```python
# Edge TPU byte vs CPU TFLite byte for int8 models:
hw_byte = cpu_byte ^ 0x80

# To convert Edge TPU output to correct int8:
int8_values = (uint8_bytes ^ 0x80).view(np.int8)
```

This is handled automatically in `simple_invoker.py` and `spot_tracker.py`.

## The 8-Bit Output Wall

```
int8_input x int8_weight -> int32_accumulator -> requantize -> uint8_output
```

The MAC array accumulates in 32-bit internally, but the output is always requantized to
uint8 (256 distinct values). The int32 accumulator is never exposed.

**What doesn't work**:
- **Byte decomposition**: Split 16-bit values into halves, run 4 matmuls, recombine.
  Fails because each intermediate is requantized to uint8.
- **Multi-output tiling**: Multiple outputs with different scales to capture different
  accumulator bit-ranges. Tested and showed no improvement (MAE 3.59 for both single
  and dual-branch).

**Root cause**: For N >= 64, the bottleneck is accumulated int8 weight x input quantization
noise across N MACs, not the output requantization step.

**What works well at 8-bit**: Feature extraction, learned policies, approximate compute,
classification, detection. **What needs CPU**: Covariance updates, PID integrals,
sub-degree rotation, sub-pixel flow.

## Edge TPU Supported Operations (Compiler v16)

**Supported**: Conv2D, DepthwiseConv2D, TransposeConv, FullyConnected, Add, Sub, Mul,
SquaredDifference, MaxPool, AvgPool, Concat, Split, Slice, Pad, Pack, Maximum, Minimum,
ReduceMax, ReduceMin, Sum, ReLU, ReLU6, Tanh, Logistic, Softmax, L2Norm,
ResizeBilinear, ResizeNearestNeighbor, SpaceToDepth, Reshape, Transpose, Rsqrt, LSTM

**NOT supported**: ArgMax/ArgMin, Abs, Exp/Log/Pow, Gather/GatherND, Where/Select, TopK

**Workarounds**:
- ArgMax -> Softmax + Mul + Sum (soft argmax, used in SpotTracker)
- Abs -> ReLU(x) + ReLU(-x), or x^2 via Mul(x, x) (used in LoomingDetector)
- Shift/warp -> Pad + Slice with pre-computed offsets

**Compiler behavior**: The compiler creates at most ONE contiguous Edge TPU subgraph
per model. Any unsupported op breaks the chain permanently — all subsequent ops run on CPU.

## Systolic Array Architecture

The Edge TPU has a 64x64 MAC array at 480 MHz (4 TOPS peak). Key implications:

- Input vectors wider than 64 elements are split into 64-element chunks (tiling passes)
- Each pass reloads weights, so weight-loading is bandwidth-limited
- Dense(256): 4 tiling passes. Dense(1024): 16 passes.
- Post-MAC activation is hardwired ROM (likely ReLU); compiler controls bypass
- **Max cached matrix**: ~2896x2896 (~8 MB in wide SRAM). Beyond this, weights stream
  over USB every inference (10-100x slower).

## Testing Infrastructure

### Test Markers
- Default: offline tests (no hardware)
- `@pytest.mark.hardware`: requires USB Edge TPU (enabled via `--run-hardware`)
- `@pytest.mark.validated`: hardware test with semantic validation (real images, correct outputs)

### Test Suites

| File | Offline | Hardware | What it tests |
|------|---------|----------|---------------|
| `test_parse_models.py` | 12 | 0 | TFLite parsing, DarwiNN extraction, field offsets |
| `test_matmul_engine.py` | 18 | 7 | Quantization, blob generation, weight swap, CPU comparison |
| `test_spot_tracker.py` | 38 | 17 | Direction, servo error, corner tracking, moving spot, color swapping |
| `test_looming.py` | 12 | 5 | Tau computation, TTC, expanding circle validation |
| `test_pattern_tracker.py` | 17 | 9 | Template discovery, sidecar validation, corner tracking, runtime swap |
| `test_transport_guards.py` | 5 | 0 | `_ensure_open()` checks on USB methods |
| `test_error_handling.py` | 27 | 0 | TFLite/DarwiNN parser errors, DMA bounds, quantize edge cases |
| `test_tflite_builder.py` | 121 | 0 | TFLite builder: Dense, SpotTracker, Looming, PatternTracker structure/quant/comparison |
| `test_optical_flow.py` | 54 | 7 | Gabor kernels, builder roundtrip, correlation, soft argmax, pipeline, pooled builder, pooled pipeline, templates, relayout validation |
| `test_visual_compass.py` | 25 | 0 | Init/properties, yaw computation, direction, lifecycle, from_template |
| `test_reservoir.py` | 32 | 0 | Init, weight generation, step, run, fit/predict, readout engine, lifecycle |
| `test_embedding_similarity.py` | 29 | 0 | Init, database CRUD, query/batch, normalization, save/load, lifecycle |
| `test_hardware.py` | 0 | 12 | All zoo models + post-processing validation (PoseNet, MultiPose, DeepLabV3) |
| `test_visual.py` | N/A (script) | 5 | Annotated output images (classification, detection, segmentation, pose, multipose) |
| `test_visual_robotics.py` | N/A (script) | 6 | Annotated output images (spot tracker bright + color, looming, pattern tracker, matmul, optical flow) |
| **Total** | **449** (offline) | **68** (hardware) | **All pass ✅** |

### Visual Proof Tests

`test_visual.py` is a standalone script (not pytest) that generates annotated output images
proving correct inference across all 5 model types. Run via `python -m tests.test_visual`.

Uses validated postprocess modules (not hand-rolled decoders):
- **Detection**: uses `postprocess_ssd()` — reads model's pre-computed anchors and
  `TFLite_Detection_PostProcess` FlexBuffer params (scale factors, NMS thresholds),
  applies box decoding, sigmoid scoring, and per-class NMS — no hand-generated anchor grids
- **Segmentation**: uses `postprocess_deeplabv3()` (weight extraction via `parse_full()`,
  no TensorFlow dependency)
- **Pose**: uses `postprocess_posenet()` (PersonLab decoder with mid/short offsets).
  The shipped PoseNet model is **single-person only**; the decoder algorithm supports
  multi-person but the model produces reliable results for one person.
- **MultiPose**: uses `postprocess_multipose()` (multi-person decoder with forward/backward
  displacements and scipy-based local maxima detection). The MultiPose model (~805 KB,
  MobileNet V1 0.50, 257×257 int8 input) outputs 4 raw tensors — different color per person.
  Requires `scipy` (`pip install libredgetpu[multipose]`).
- **Pose image**: uses `squat.bmp` (single full-body person) for clean skeleton output

Output images saved to `tests/results/` (git-ignored).

See [`VISUAL_TESTS.md`](VISUAL_TESTS.md) for the full guide with reference images and troubleshooting.

### Visual Proof Tests — Robotics Modules

`test_visual_robotics.py` is a standalone script (not pytest) that generates annotated output
images proving correct behavior for the 6 robotics module tests. Run via
`python -m tests.test_visual_robotics`. No model zoo downloads required — uses synthetic scenes.

- **SpotTracker (bright)**: 3x3 grid of grayscale Gaussian dots at 9 positions, tested on
  both flat (uniform dark) and textured (random noise) backgrounds. Green `+` expected vs
  red `x` detected crosshair overlay.
- **SpotTracker (color_red)**: Same 3x3 grid but with red Gaussian dots on RGB images,
  tested on flat and textured backgrounds. Validates the color-selective Conv2D filter.
- **LoomingDetector**: Synthetic scenes (uniform, center circle, border frame) plus real
  photographs (portrait, cat, bird) with 3x3 zone heatmap, prominent tau value, and
  approach/recede interpretation per scene.
- **PatternTracker**: 8x8 checkerboard template placed at 3 positions in 64x64 search image,
  tested on flat and textured backgrounds, with bounding box and detection crosshair overlay.
- **MatMulEngine**: Anti-diagonal reversal (sine wave flip), random matrix correlation scatter
  with R-squared, and error distribution histogram.
- **OpticalFlow**: 5 synthetic shift scenarios (right/down/left/up/none) with frame pair
  visualization, flow vector arrow, and direction/status labels.

Output images saved to `tests/results/` (git-ignored).

See [`VISUAL_TESTS.md`](VISUAL_TESTS.md) for the full guide with reference images and troubleshooting.

### Interactive GUI (Real-Time Testing)

`libredgetpu.gui` is a Flask + OpenCV web interface for live interactive testing of all 9
algorithms. Useful for validating real-world behavior beyond unit tests, quick demos, and
debugging edge cases with a webcam.

```bash
pip install -e ".[gui]"              # adds flask, opencv-python
python -m libredgetpu.gui            # auto-detect Edge TPU + webcam
python -m libredgetpu.gui --synthetic --pattern wandering_dot   # CPU-only, no hardware
python -m libredgetpu.gui --cpu-replica --synthetic --pattern panning  # CPU integer-replica
```

**CPU replica mode** (`--cpu-replica`): Faithfully reproduces the Edge TPU's integer arithmetic
on CPU for OpticalFlow and VisualCompass. This mode executes the full quantized pipeline
(QUANTIZE → DEPTHWISE_CONV_2D → AVG_POOL_2D → QUANTIZE) with int8/int32 integer accumulation,
per-channel requantization, and fused ReLU — matching the Edge TPU within ±1 LSB (validated
against TFLite interpreter). Useful for isolating hardware vs. pipeline issues: if the CPU
replica and Edge TPU disagree, the bug is in hardware execution or output relayout.
Non-flow algorithms fall back to synthetic mode when `--cpu-replica` is active.
The `--cpu` and `--cpu-replica` flags are mutually exclusive.

**Architecture** (`libredgetpu/gui/`):

| File | Role |
|------|------|
| `app.py` | Flask server: MJPEG stream (`/video_feed`), SSE metrics (`/metrics`), `/pause` toggle, click/drag endpoints, parameter schema/apply endpoints |
| `algorithm_modes.py` | 9 `AlgorithmMode` subclasses (one per module), each with `process(frame, mouse_state)` and `get_param_schema()` for tunable parameters |
| `camera.py` | `RealCamera` (cv2.VideoCapture) + `SyntheticCamera` (5 pattern generators) |
| `cpu_replica.py` | CPU integer-replica of Edge TPU optical flow pipeline (QUANTIZE → CONV → POOL → QUANTIZE) |
| `overlay.py` | OpenCV drawing helpers: crosshair, heatmap grid, flow arrow, compass, scatter, histogram |
| `templates/index.html` | Single-page UI: `<img>` for MJPEG stream, dropdown, parameter controls, metrics panel, pause/play button |
| `static/app.js` | Mouse events (click vs drag with 5px threshold), pause toggle (Spacebar shortcut), SSE listener, screenshot download, schema-driven parameter rendering + Apply handler |
| `static/style.css` | Responsive layout (video left, controls right; stacks on mobile), button styles, parameter control styles |

**Pause/Play**: Press the Pause button (or **Spacebar**) to freeze the video stream on the current
frame. Useful for PatternTracker template selection (pause → drag rectangle → play to track) and
inspecting frames. When paused, the frozen frame is preserved and displayed with `[PAUSED]` in the HUD.

**Hardware lifecycle**: Each `AlgorithmMode` calls `_open_hw(resource)` which calls
`resource.open()` and tracks it for `cleanup()`. When switching algorithms via the dropdown,
`VideoStream._switch_algorithm()` calls `cleanup()` on the old mode before creating the new one.

**Tunable parameter controls**: Each `AlgorithmMode` subclass declares tunable parameters via
`get_param_schema()` (classmethod returning a list of descriptors with name, label, type,
default, options/min/max/step, and description). The frontend dynamically renders controls
(select, number, range, checkbox) from the schema. An Apply button POSTs values to
`/apply_params`, which validates against the schema and re-initializes the algorithm with the
new parameters. Active parameter values are tracked via `_active_params` and returned by
`get_active_params()`. Applying parameters always creates a fresh algorithm instance (no
state preservation) since parameters like dimension, pool_factor, or kernel_size require
model re-creation.

| Algorithm | Tunable Parameters |
|-----------|-------------------|
| SpotTracker | image_size (64/128), variant (bright/color_*) |
| PatternTracker | search_size (64/128), kernel_size (8/16/32), channels (1/3) |
| LoomingDetector | image_size (64/128) |
| OpticalFlow | image_size (64/128), pooled, pool_factor (1/2/4/8), search_range (1-8), temperature (0.01-1.0) |
| VisualCompass | same as OpticalFlow + fov_deg (1-360) |
| MatMulEngine | dim (256/512/1024) |
| ReservoirComputer | dim (256/512/1024), spectral_radius (0.1-2.0), leak_rate (0.01-1.0), activation (tanh/relu/identity) |
| EmbeddingSimilarity | dim (256/512/1024) |
| SimpleInvoker | model (Classification/Detection/Segmentation/PoseNet/MultiPose) |

**Adding a new algorithm mode**:
1. Subclass `AlgorithmMode` in `algorithm_modes.py`
2. Implement `process(frame, mouse_state) -> annotated_frame` (both synthetic and hardware paths)
3. Register in `ALGORITHM_MODES` dict
4. Restart server

**Synthetic camera patterns**: `noise`, `checkerboard`, `rotating` (radial lines),
`panning` (scrolling stripes at 600 px/sec), `wandering_dot` (Lissajous Gaussian, default).

**GUI optical flow tuning**: Hardware mode uses `pooled=False, pool_factor=2` instead of the
library default (`pooled=True, pool_factor=4`) for better sensitivity at webcam resolution.
The 640→64 downscale means a 1px shift at 64×64 requires 10px of camera motion at 640×480.
With `pool_factor=2`, the minimum detectable displacement is ~2px at 64×64 = ~20px at 640×480.
The `panning` pattern's 600 px/sec speed produces 2 px/frame at 64×64 (1 pooled pixel/frame).

See [`GUI_GUIDE.md`](GUI_GUIDE.md) for end-user documentation (launch options, per-algorithm
interactions, troubleshooting).

### Model Zoo
Tests auto-download from [EdgeTPUModelZoo](https://github.com/ricardodeazambuja/EdgeTPUModelZoo)
(branch: **master**, not main). Models cache in `tests/models/` (git-ignored).

### Key Validated Tests
- **Moving spot** (SpotTracker): spot slides left-to-right, x_offsets are monotonically
  increasing. Core visual servoing validation.
- **Corner tracking at 16/64/128** (SpotTracker): bright spots at corners have correct
  signs AND magnitude > 0.3. Catches the quantization clipping bug that was found and fixed.
- **Expanding circle** (LoomingDetector): concentric rings at 3 radii produce different
  tau values. Proves spatial sensitivity.
- **PoseNet validated**: Grace Hopper image -> at least 1 pose with >= 5 keypoints
  (single-person model; decoder supports multi-person but shipped model detects one).
- **MultiPose validated**: Grace Hopper image (257×257) -> at least 1 pose with score > 0.1
  and >= 3 valid keypoints. Uses int8 input preprocessing and 4-tensor output decoding.
- **DeepLabV3 validated**: Grace Hopper image -> person class covers > 5% of segmentation map.

## Experiments

Self-contained scripts in `experiments/` that validate Edge TPU behavior.

| Experiment | Question | Finding |
|------------|----------|---------|
| exp1 | Do different weights produce same DarwiNN instructions? | PC instructions are architecture-determined. EO instructions encode requant multiplier. |
| exp1b | Detailed instruction bitstream comparison. | Confirmed exp1 with finer granularity. |
| exp1c | Patching int8 weights while preserving quant metadata? | EO + PC instructions identical. Only PC param blob changes. Validates set_weights(). |
| exp2 | How do Edge TPU outputs differ from CPU TFLite? | Edge TPU always outputs uint8. int8 models: hw_byte = cpu_byte XOR 0x80. |
| exp3 | Can we predict the DarwiNN param blob format? | YES. 64-row groups, 4-column tiling, XOR 0x80. 25/25 byte-perfect matches. |
| exp3b | Validate blob generation at scale. | All sizes and patterns match compiler output. |

## Known Gotchas

### Input quantization determines everything
The representative dataset used during template generation determines input_scale,
output_scale, and the effective dynamic range. Using the wrong range (e.g., [0, 255]
instead of [-1, 1]) silently produces garbage outputs that pass tests with loose
tolerances.

### EO-phase parameters
Some cached models (Inception V1) have a parameter chunk in the EXECUTION_ONLY executable
(not just PARAMETER_CACHING). The 393 KB Inception V1 EO params must be sent during
inference. Always pass `eo.parameters` to `execute_dma_hints()`.

### TFLite flatbuffer field ordering
`Model.buffers` is field index **4** (not 3; field 3 = description string). Getting this
wrong silently returns garbage. Full map:
- Model: version=0, operator_codes=1, subgraphs=2, description=3, buffers=4
- Tensor: shape=0, type=1, buffer=2, name=3, quantization=4
- OperatorCode: deprecated_builtin_code=0(int8), custom_code=1(old)/4(new), builtin_code=3(int32)

### DarwiNN FlatBuffer API
Use `GetRootAsPackage` (not `GetRootAs`). Field offsets for Description enum:
0=OUTPUT, 1=INPUT, 2=PARAMETER, 3=SCRATCH.

### Firmware upload retry
Firmware upload to the bootloader device retries up to 3 times on `USBError`. Between
retries, the transport re-discovers the bootloader device. If the device disappears
during retry, the error is raised immediately. The chunk counter is also validated
against the USB 16-bit `wValue` limit (max 65535 chunks).

### USB device disconnection during test runs
The Edge TPU can spontaneously disappear from the USB bus during test runs, especially
when many tests open and close the device in rapid succession. Each test's context
manager (`with engine:` / `with tracker:`) triggers a full open → USB reset →
re-enumerate → close cycle. The transport layer mitigates this with:

- **Post-reset settle delay** (`_RESET_SETTLE_S = 0.6s`): waits for the device to
  re-enumerate after `dev.reset()`. 0.5s is usually enough; 0.6s adds margin.
- **Re-open retries** (`_REOPEN_MAX_RETRIES = 3`, `_REOPEN_RETRY_S = 1.0s`): retries
  device discovery after reset, since re-enumeration can take longer after many
  consecutive open/close cycles.
- **Firmware upload retries** (`_FW_UPLOAD_MAX_RETRIES = 3`): retries firmware download
  on USB glitches.

**Root cause is not fully understood.** Likely contributing factors:
1. **USB hub/controller timing** — some USB controllers or hubs don't re-enumerate
   the device reliably under rapid reset cycling. Powered USB 3.0 hubs seem more
   reliable than direct motherboard ports or unpowered hubs.
2. **Linux kernel USB subsystem race** — detaching/reattaching the kernel driver
   between tests may race with udev rules or other USB enumeration machinery.
3. **Edge TPU firmware limitations** — Google's own libedgetpu also resets the device
   on open, but typically only opens once per process lifetime. The firmware may not
   be designed for rapid reset cycling.
4. **xHCI controller quirks** — some xHCI host controllers have known issues with
   device re-enumeration after reset (seen in Linux kernel bug trackers).

**When it happens**: the device vanishes from `lsusb` output entirely. No amount of
software retry can recover it — a physical USB replug is required, which puts the
device back in bootloader mode (`1a6e:089a`). The next `open()` will re-upload firmware.

**Workaround**: if running the full test suite and hitting disconnects, try running
hardware test files individually rather than all at once, or add a short delay between
test files.

### Thread safety
Edge TPU requires sequential command/response. No concurrent USB operations.
`EdgeTPUModelBase` uses a `threading.Lock` around parameter cache uploads to
prevent races when `invoke()` / `track()` is called from multiple threads, but
the underlying USB transport is **not** thread-safe. A single thread should own
the device for I/O; the lock only protects the cache-check-then-upload sequence.

## TFLite Parser Internals

`tflite_parser.py` reads TFLite flatbuffers without TensorFlow. Two APIs:

- `parse(tflite_bytes)` — extracts `edgetpu-custom-op` data + I/O quant params. Used by
  all modules for basic model loading.
- `parse_full(tflite_bytes)` — returns all tensors (with buffer indices and names),
  operators (with opcode names), buffers, and graph I/O. Used by post-processing modules
  to extract Conv2D weights/biases from TFLite buffers.

### Key opcode enum values
CONCATENATION=2, CONV_2D=3, MUL=18, RESHAPE=22, RESIZE_BILINEAR=23, SOFTMAX=25,
ARG_MAX=56, QUANTIZE=114, HARD_SWISH=117.

## C Extension Details

`_usb_accel.c` (~400 lines) provides a `UsbDevice` class wrapping direct libusb-1.0 calls:
- `send()`: header coalescing at C level (single `libusb_bulk_transfer`)
- `read_output()`: chunked reads with 32768-byte requests
- `read_status()`: single bulk read from EP 0x82
- `ctrl_transfer()`, `reset()`, `close()`

GIL is released around blocking I/O. Built by `setup.py` with `OptionalBuildExt` — if
`libusb-1.0-dev` is missing, install proceeds as pure Python.

4 GB bounds check on `send()` before uint32 cast to prevent overflow.
100 MB sanity check on `read_output(max_size)` to prevent integer overflow in allocation.
`ctrl_transfer` IN length validated to [0, 65535] range (USB control transfer limit).

## Hardware Analysis Reference

For the full 128-bit ISA decode, hardware register map, patent cross-references, and
DMA/TTU analysis, see [`docs/HARDWARE_ANALYSIS.md`](docs/HARDWARE_ANALYSIS.md)
(~1400 lines).

### edgetpu_compiler Summary

The `edgetpu_compiler` binary (v16.0, 26.5 MB stripped ELF x86-64) has 11 CLI flags.
Useful flags: `-m 10` forces standalone/streamed mode, `-a` enables multi-subgraph
(experimental), `-i` controls tensor boundaries, `-d -k` searches for maximal
delegatable op set. Internally runs 73 MLIR passes in 3 dialects (dwg/dwl/dwgt).
Debug output files exist but are gated by internal-only proto fields — not extractable.

## References

Collected bibliography for the algorithms and techniques referenced throughout this
document. Citation tags (e.g., `[Jacob2018]`) appear in the "Algorithmic Foundations"
section above.

- **[Chapelle2010]** Chapelle, O. & Wu, M. (2010). "Gradient descent optimization of smoothed information retrieval metrics." *Information Retrieval*, 13(3), 216-235.
- **[Daugman1985]** Daugman, J. (1985). "Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters." *Journal of the Optical Society of America A*, 2(7), 1160-1169.
- **[Gabor1946]** Gabor, D. (1946). "Theory of communication." *Journal of the IEE*, 93(26), 429-457.
- **[Jacob2018]** Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H. & Kalenichenko, D. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *Proc. IEEE/CVF CVPR*, 2704-2713.
- **[Jaeger2001]** Jaeger, H. (2001). "The 'echo state' approach to analysing and training recurrent neural networks." *GMD Report 148*, German National Research Center for Information Technology.
- **[Jain1991]** Jain, A. K. & Farrokhnia, F. (1991). "Unsupervised texture segmentation using Gabor filters." *Pattern Recognition*, 24(12), 1167-1186.
- **[Krishnamoorthi2018]** Krishnamoorthi, R. (2018). "Quantizing deep convolutional networks for efficient inference: A whitepaper." *arXiv:1806.08342*, Google.
- **[Lee1976]** Lee, D. N. (1976). "A theory of visual control of braking based on information about time-to-collision." *Perception*, 5(4), 437-459.
- **[Lewis1995]** Lewis, J. P. (1995). "Fast normalized cross-correlation." *Vision Interface*, 120-123.
- **[Lukosevicius2009]** Lukoševičius, M. & Jaeger, H. (2009). "Reservoir computing approaches to recurrent neural network training." *Computer Science Review*, 3(3), 127-149.
- **[Luvizon2019]** Luvizon, D., Tabia, H. & Picard, D. (2019). "Human pose regression by combining indirect part detection and contextual information." *Computers & Graphics*, 85, 15-22.
- **[Nelson1989]** Nelson, R. C. & Aloimonos, J. (1989). "Obstacle avoidance using flow field divergence." *IEEE Trans. PAMI*, 11(10), 1102-1106.
- **[Salton1975]** Salton, G., Wong, A. & Yang, C. S. (1975). "A vector space model for automatic indexing." *Communications of the ACM*, 18(11), 613-620.
- **[Sobel1968]** Sobel, I. & Feldman, G. (1968). "A 3x3 isotropic gradient operator for image processing." Presented at the Stanford Artificial Intelligence Project.
- **[Zufferey2006]** Zufferey, J.-C. & Floreano, D. (2006). "Fly-inspired visual steering of an ultralight indoor aircraft." *IEEE Trans. Robotics*, 22(1), 137-146.

## Remaining Open Questions

1. **unk_3 sub-fields** (18 bits in instruction): which bits = stride vs limit vs mask
   in TTU config. Requires systematic experimentation.
2. **v_cmd mapping**: 20 observed values need mapping to DMA channel/TTU register selectors.
3. **Requantization multiplier location**: possibly the `v_op_2=7` constant `0xfc056600`.
4. **Vector MAC pipeline**: matmul_256 and dense_1_8_mul programs need analysis.
5. **USB disconnection under rapid cycling**: the Edge TPU occasionally disappears from
   the USB bus when tests repeatedly open/reset/close the device. Mitigated with retries
   and settle delays but root cause is unknown — could be host controller, kernel driver,
   or Edge TPU firmware. See "USB device disconnection during test runs" in Known Gotchas.

## Possible Future Robotics Applications

All designed but not yet implemented:

- **Learned Controllers**: Small MLP policies at 1+ kHz. 0.28 ms inference.
- **Batch MPC**: Neural dynamics model evaluating N rollouts in one matmul.

See [`docs/ROBOTICS_STATUS.md`](docs/ROBOTICS_STATUS.md) for full design specifications,
including the 8-bit wall analysis, discarded approaches, and supported operations list.
