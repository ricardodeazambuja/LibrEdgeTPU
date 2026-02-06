# GUI Optical Flow Pipeline — End-to-End Diagnostic

This document traces every transform, function call, and shape/dtype change
from webcam frame capture to the displayed `(vx, vy)` flow vector in the
LibrEdgeTPU GUI.  Its purpose is to provide a single reference for debugging
the optical flow pipeline.

---

## 1. Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  camera.py                                                                  │
│  SyntheticCamera._generate_panning() or RealCamera.read()                   │
│  → BGR uint8 (480, 640, 3)                                                  │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│  app.py:109  —  algorithm.process(frame, mouse_state)                       │
│  (VideoStream.get_frame → OpticalFlowMode.process)                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│  algorithm_modes.py:381  —  cv2.cvtColor(frame, COLOR_BGR2GRAY)             │
│  → grayscale uint8 (480, 640)                                               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│  algorithm_modes.py:398-401                                                 │
│  cv2.resize(gray, (64,64), interpolation=cv2.INTER_AREA)                    │
│  → prev_resized, curr_resized: uint8 (64, 64)                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────────┐
│  optical_flow_module.py:254  —  flow_engine.compute(prev, curr)             │
│  → calls _extract_features_uint8() on each frame                            │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          │ (×2, once per frame)                     │
          │                                          │
┌─────────▼─────────────────────────────────────────────────────────────────┐
│  optical_flow_module.py:181-224  — _extract_features_uint8(image)          │
│                                                                            │
│  1. Squeeze to (64, 64) if needed                    lines 195-200         │
│  2. image / 255.0 → float32 [0, 1]                  line 215              │
│  3. _quantize_input() → quantize_uint8(img, s=1/255, zp=0) → uint8        │
│     _base.py:223-226, _quantize.py:19-38                                   │
│  4. _execute_raw(bytes) → raw bytes from Edge TPU    line 217              │
│     _base.py:134-173                                                       │
│  5. relayout_output(raw, output_layer) → (16,16,8) uint8                   │
│     delegate.py:381-445                                                    │
└─────────┬─────────────────────────────────────────────────────────────────┘
          │
          │ feat_t_u8, feat_t1_u8: uint8 (16, 16, 8)
          │
┌─────────▼─────────────────────────────────────────────────────────────────┐
│  optical_flow_module.py:302  — _compute_from_uint8(feat_t, feat_t1)        │
│                                                                            │
│  1. uint8 → int16 (subtract zero_point)              lines 316-317        │
│  2. int16 → float32 (fused_pool → skip CPU pooling)  lines 319-322        │
│  3. _global_correlation(feat_t_f, feat_t1_f)         line 329             │
│     → float64 (81,) correlation scores                                     │
│  4. Divide by _overlap_counts                        line 330             │
│  5. _soft_argmax(corr) → (vx, vy)                   line 332             │
└─────────┬─────────────────────────────────────────────────────────────────┘
          │
          │ (vx, vy) as float
          │
┌─────────▼─────────────────────────────────────────────────────────────────┐
│  algorithm_modes.py:402-420                                                │
│  Display: draw_flow_arrow, draw text with magnitude/direction              │
│  Store prev_gray = gray.copy()  (full 640×480 resolution)                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Frame Capture (`camera.py`)

### 2.1 SyntheticCamera — Panning Pattern

**`SyntheticCamera._generate_panning()`** — `camera.py:178-232`

1. **Canvas**: 3× viewport width = 1920 × 480 pixels, all zeros (black).
2. **Vertical stripes** (lines 197-205):
   - Deterministic RNG (`seed(42)`), stripe widths 40-120 px.
   - Random intensity 20-240 per stripe (grayscale).
   - Wide stripes survive 10× downscale (640→64) without aliasing.
3. **Horizontal bars** (lines 209-214):
   - Every 120 px in X, bars at varied Y positions, thickness 6 px.
   - Provides non-periodic texture.
4. **Panning offset** (line 219):
   ```python
   offset = int((-t * 120) % canvas_width)
   ```
   Speed = 120 px/sec in the 640 px viewport. At 30 fps, that's
   **4 px/frame** at full resolution, or **0.4 px/frame at 64×64**,
   or **0.1 pooled-px/frame at 16×16**.
5. **Viewport extraction** (lines 222-230):
   Slice `canvas[:, x_start:x_start+640]` with wrap-around.

**Timing trap** (line 133):
```python
self.start_time -= 1.0 / self.fps
```
Each `read()` call rewinds `start_time`, so `_get_animation_time()` returns
`wall_clock + n_reads / fps` — animation always progresses even under rapid
reads.

### 2.2 RealCamera

**`RealCamera.__init__()`** — `camera.py:32-55`

- Opens `cv2.VideoCapture(camera_id)`, sets 640×480.
- **Warmup**: reads and discards 5 frames with `time.sleep(0.1)` each
  (lines 53-55). Essential: some cameras return stale/black frames initially.
- `read()` retries up to 3 times with 0.1s delays on failure (lines 62-67).

---

## 3. GUI Loop (`app.py`)

### 3.1 VideoStream.get_frame()

**`app.py:90-146`**

```
camera.read() → frame (BGR uint8, 640×480×3)
    ↓
algorithm.process(frame, mouse_state)  [line 109]
    ↓
overlay.draw_performance_hud()         [line 124]
    ↓
cv2.imencode('.jpg', quality=85)       [line 134]
    ↓
JPEG bytes
```

### 3.2 MJPEG Streaming

**`generate_mjpeg()`** — `app.py:215-234`

- Calls `stream.get_frame()` in a loop.
- Yields multipart MJPEG boundaries:
  `b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'`
- FPS limiter: `time.sleep(max(0, 1/target_fps - elapsed))`.

### 3.3 Threading

`app.run(threaded=True)` at line 394 — Flask serves each HTTP request in a
separate thread. A `threading.Lock` (line 31) protects algorithm access during
`process()` and mouse state updates.

---

## 4. Algorithm Mode (`algorithm_modes.py`)

### 4.1 OpticalFlowMode.__init__()

**Lines 355-371**

```python
self.flow_engine = OpticalFlow.from_template(image_size, pooled=False, pool_factor=2)
self._open_hw(self.flow_engine)   # calls flow_engine.open()
```

- `image_size` defaults to 64.
- `pooled=False, pool_factor=2` → uses standard Gabor template with CPU-side 2× pooling.
  This gives better motion sensitivity than `pooled=True` (pool_factor=4), halving the
  minimum detectable displacement from ~20px to ~10px at 640×480 webcam resolution.

### 4.2 OpticalFlowMode.process()

**Lines 373-421** — The full per-frame pipeline:

```python
# Step 1: BGR → grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # line 381
# → uint8 (480, 640)

# Step 2: First frame — store and return
if self.prev_gray is None:                             # line 383
    self.prev_gray = gray.copy()
    return annotated

# Step 3: Resize BOTH frames to 64×64
prev_resized = cv2.resize(self.prev_gray,              # line 398
    (self.image_size, self.image_size),
    interpolation=cv2.INTER_AREA)
curr_resized = cv2.resize(gray,                        # line 400
    (self.image_size, self.image_size),
    interpolation=cv2.INTER_AREA)
# → uint8 (64, 64) for each

# Step 4: Compute flow
dx, dy = self.flow_engine.compute(prev_resized, curr_resized)  # line 402

# Step 5: Update prev for next frame
self.prev_gray = gray.copy()                           # line 420
# NOTE: stores FULL resolution (640×480), not resized
```

**Key detail**: `prev_gray` is stored at full resolution (640×480). Both
`prev_gray` and the new `gray` are resized to 64×64 fresh each frame. This
means the downscale quality of `INTER_AREA` matters every frame.

### 4.3 VisualCompassMode

**Lines 430-500** — Nearly identical flow, but calls:
```python
delta_yaw = self.compass.compute_yaw(prev_resized, curr_resized)  # line 484
```

`VisualCompass.compute_yaw()` (`visual_compass.py:118-129`) delegates to
`OpticalFlow.compute()` then multiplies by:
```python
deg_per_pooled_px = fov_deg * effective_pool / width
# = 90 * 4 / 64 = 5.625 degrees per pooled pixel
```

---

## 5. Feature Extraction (`optical_flow_module.py`)

### 5.1 _extract_features_uint8()

**Lines 181-224** — Called once per frame (twice per `compute()` call):

```python
# 1. Input validation — shape must be (64, 64)
#    Squeezes batch/channel dims if present         lines 195-200
#    Converts non-uint8 via np.clip(0,255).astype   lines 203-204

# 2. Normalize to [0, 1]
image_normalized = image.astype(np.float32) / 255.0   # line 215
# → float32 (64, 64) in [0.0, 1.0]

# 3. Quantize to uint8
quantized = self._quantize_input(image_normalized)     # line 216
# _base.py:223-226 → quantize_uint8(img, scale=1/255, zp=0)
# _quantize.py:35-38: q = clip(round(value / scale + zp), 0, 255)
#   = clip(round(value / (1/255) + 0), 0, 255)
#   = clip(round(value * 255), 0, 255)
# → uint8 (64, 64) — essentially the original pixel values restored

# 4. Execute on Edge TPU
raw_output = self._execute_raw(quantized.tobytes())    # line 217
# _base.py:134-173 → sends bytes via USB, gets raw output

# 5. De-scatter tiled output
relayout_output(raw_output, self._output_layer)        # line 221
# delegate.py:381-445
# → uint8 (16, 16, 8)  [for pooled mode: H/4 × W/4 × 8 filters]
```

### 5.2 Quantization Round-Trip

The normalize-then-quantize sequence looks like a no-op but is intentional:

```
uint8 pixel → / 255.0 → float32 [0,1] → * 255 → round → clip → uint8
```

This exists because the model's input quantization has `scale=1/255, zp=0`.
The `_quantize_input()` method is generic (reads scale/zp from the TFLite
model), so normalizing to [0,1] first makes it work correctly for any model's
input quantization parameters.

---

## 6. Edge TPU Model

### 6.1 Model Architecture

Built by `tflite_builder.py:build_optical_flow_pooled()` (line 1758):

```
Input: uint8 [1, 64, 64, 1]
  ↓
QUANTIZE (uint8 → int8)
  scale=1/255, zp_out=-128
  Maps: uint8=0 → int8=-128, uint8=255 → int8=127
  ↓
DEPTHWISE_CONV_2D
  Weight: int8 [1, 7, 7, 8] (8 Gabor kernels)
  Per-channel quantization (each kernel has its own scale)
  Padding: SAME
  Activation: ReLU
  Stride: 1
  depth_multiplier: 8
  ↓
  Output: int8 [1, 64, 64, 8]
  ↓
AVG_POOL_2D
  Filter: 4×4, Stride: 4
  Padding: VALID
  ↓
  Output: int8 [1, 16, 16, 8]
  ↓
QUANTIZE (int8 → uint8)
  Maps back to uint8 for USB transfer
  ↓
Output: uint8 [1, 16, 16, 8]  = 2048 bytes
```

### 6.2 Gabor Kernels

Generated by `_generate_gabor_kernels()`:
- **4 orientations**: 0°, 45°, 90°, 135° (`y_theta`, not `x_theta`)
- **2 sigma values**: 1.5 and 3.0
- **Wavelength**: `lambda = 2 * sigma`
- **Kernel size**: 7×7
- **Total**: 8 filters

### 6.3 Output Quantization

```
conv_output_max = ksize² × 127 × 1.5 × (1/255) × mean(weight_scales) / 127
conv_output_scale = conv_output_max / 127
conv_output_zp = -128   (ReLU → non-negative outputs)
```

The 1.5× safety margin (line 1857) prevents saturation of output channels.

---

## 7. Output Relayout (`delegate.py:381-445`)

### 7.1 The Problem

The Edge TPU stores convolution outputs in a **tiled TYXZ memory layout** —
a 4×4 grid of 16 tiles. The raw USB output bytes are NOT in standard row-major
YXZ order. Without relayout, spatial correlations are destroyed.

### 7.2 The Solution

`relayout_output(raw_bytes, layer)` uses lookup tables from the DarwiNN
executable:

| Table | Indexed by | Maps to |
|-------|-----------|---------|
| `y_tile_id_map[y]` | output row y | partial tile ID |
| `x_tile_id_map[x]` | output col x | added to y's tile ID |
| `tile_byte_offsets[tile_id]` | combined tile ID | base byte offset |
| `y_local_y_offset[y]` | output row y | row within tile |
| `x_local_byte_offset[x]` | output col x | byte offset within tile row |
| `x_local_y_row_size[x]` | output col x | stride between tile rows |

For each `(y, x)` in the output:
```python
tile_id = y_tile_id_map[y] + x_tile_id_map[x]
base = (tile_byte_offsets[tile_id]
        + y_local_y_offset[y] * x_local_y_row_size[x]
        + x_local_byte_offset[x])
dest[y, x, :] = src[base : base + z_dim]
```

### 7.3 Impact

| | Without relayout | With relayout |
|---|---|---|
| Horizontal correlation | Partially survives | Perfect (1.000) |
| Vertical correlation | Destroyed | Perfect (1.000) |

See [`docs/OPTICAL_FLOW_ENGINEERING_LOG.md`](OPTICAL_FLOW_ENGINEERING_LOG.md) for the full investigation.

---

## 8. Correlation & Soft Argmax

### 8.1 _compute_from_uint8()

**`optical_flow_module.py:302-332`**

```python
# 1. Remove zero point: uint8 → int16
zp = int16(output_info.zero_point)
feat_t_int = feat_t_u8.astype(int16) - zp        # line 316
feat_t1_int = feat_t1_u8.astype(int16) - zp      # line 317
# → int16 (16, 16, 8)

# 2. Cast to float32 (fused_pool: skip CPU pooling)
feat_t_f = feat_t_int.astype(float32)             # line 321
feat_t1_f = feat_t1_int.astype(float32)           # line 322
# → float32 (16, 16, 8)

# 3. Global correlation
corr = _global_correlation(feat_t_f, feat_t1_f)   # line 329
# → float64 (81,)

# 4. Normalize by overlap area
corr /= self._overlap_counts                      # line 330
# _overlap_counts[k] = (ph - |dy|) * (pw - |dx|) * nf
# where (dx, dy) is the k-th displacement

# 5. Soft argmax
return _soft_argmax(corr.astype(float32))          # line 332
```

### 8.2 _global_correlation()

**`optical_flow_module.py:376-419`**

For search_range=4, evaluates 9×9 = 81 displacement candidates:

```python
# 1. Pad feat_t with zeros: (16,16,8) → (24,24,8)
padded = np.pad(feat_t, ((4,4),(4,4),(0,0)))       # line 400

# 2. Sliding window view (zero-copy via stride tricks)
view = as_strided(padded,
    shape=(9, 9, 16, 16, 8),                       # lines 405-407
    strides=(s[0], s[1], s[0], s[1], s[2]))
# view[j, i] = padded[j:j+16, i:i+16, :]

# 3. Batch dot product via einsum
corr_map = einsum('ijhwc,hwc->ij', view, feat_t1)  # line 415
# → float64 (9, 9)

# 4. Flip and flatten (displacement sign convention)
return corr_map[::-1, ::-1].ravel()                 # line 419
# → float64 (81,)
```

The flip ensures `corr[k]` corresponds to scene displacement `(dx[k], dy[k])`
matching the `_displacements` array.

### 8.3 _soft_argmax()

**`optical_flow_module.py:421-441`**

```python
# temperature = 0.1
corr_shifted = corr - max(corr)                     # line 431
weights = exp(corr_shifted / 0.1)                    # line 432
weights /= sum(weights)                              # line 436

vx = sum(weights * displacements[:, 0])              # line 439
vy = sum(weights * displacements[:, 1])              # line 440
```

With temperature=0.1, the softmax is very sharp — the strongest correlation
peak dominates. Sub-pixel precision comes from weighted interpolation between
neighboring displacement bins.

### 8.4 Overlap Normalization

**`optical_flow_module.py:128-132`**

```python
_overlap_counts[k] = (ph - |dy|) * (pw - |dx|) * nf
# ph=16, pw=16, nf=8
# At (0,0): 16 * 16 * 8 = 2048
# At (±4,0): 12 * 16 * 8 = 1536
# At (±4,±4): 12 * 12 * 8 = 1152
```

Without this normalization, zero displacement (0,0) always wins because it
has the most overlapping pixels. With normalization, the correlation is
"per overlapping element" and the true displacement peak emerges.

---

## 9. Known Aliasing Bug & Fix

### 9.1 The Problem

Prior to the fix, `cv2.resize()` used the default `INTER_LINEAR` interpolation.
At 10× downscale (640→64), bilinear interpolation **skips most input pixels**
— it samples only the 4 nearest neighbors of each output pixel center.

For the synthetic panning pattern with narrow stripes (8-32 px):
- 10× downscale aliases stripes into a **moiré pattern** at 64×64
- The moiré shifts frame-to-frame by a pattern unrelated to the true pan
- After Gabor filtering + correlation, this produces **±3 pooled px
  oscillation** in vx instead of the expected ~0.1 pooled px/frame

### 9.2 The Fix

**`algorithm_modes.py:398-401`** — Changed to `interpolation=cv2.INTER_AREA`:

```python
prev_resized = cv2.resize(self.prev_gray,
    (self.image_size, self.image_size),
    interpolation=cv2.INTER_AREA)          # ← was default INTER_LINEAR
```

`INTER_AREA` averages all pixels in each source region, acting as a proper
anti-aliasing low-pass filter. This matches the resampling behavior assumed by
the Gabor feature pipeline.

### 9.3 Impact

| Scenario | INTER_LINEAR | INTER_AREA |
|----------|-------------|-----------|
| Narrow stripes (8-32 px) | ±3 vx oscillation | Correct ~0 vx |
| Wide stripes (40-120 px) | OK | OK |
| Real webcam | OK (natural images lack extreme high-freq) | OK |

The synthetic panning pattern was also updated to use wider stripes (40-120 px)
to stay above the Nyquist limit even with bilinear interpolation.

See `experiments/OPTICAL_FLOW_GUI_FIX_LOG.md` for the full investigation.

---

## 10. What Could Still Go Wrong

### 10.1 Panning Speed and Detection Threshold

The panning pattern moves at 600 px/sec at 640 px resolution.
After downscale and pooling:

```
600 px/sec ÷ 10 (640→64) = 60 px/sec at 64×64
60 px/sec ÷ 2  (pool)    = 30 pooled-px/sec at 32×32
30 / 30 fps              = 1.0 pooled-px/frame → detectable
```

The previous speed (120 px/sec) was 10× too slow — only 0.1 pooled-px/frame with
pool_factor=4, which was below the minimum detectable displacement (~0.5 pooled px).
The fix involved both increasing speed (120→600 px/sec) and reducing pool_factor (4→2).

**To validate**: use a faster pan (e.g., 1200 px/sec → 1 pooled-px/frame)
or test with known non-zero expected flow.

### 10.2 Full-Resolution prev_gray Storage

`prev_gray` is stored at 640×480 (line 420), then re-resized each frame
(line 398). This means the "previous" features come from a separately
downscaled image. If the interpolation kernel introduces any asymmetry
between two independent resize operations on slightly shifted images,
that asymmetry becomes a systematic bias. Using `INTER_AREA` makes this
negligible, but it's worth noting.

### 10.3 Integer Zero Point Interaction

The output zero point from the Edge TPU model determines the "zero level"
for features. If the zero point is incorrectly extracted or doesn't match
the model's actual output, all correlation values shift, potentially
biasing the soft argmax. The pipeline reads it from the TFLite metadata
(`_output_info.zero_point`), which should be reliable.

### 10.4 Why CPU Mode "Works" but Edge TPU Doesn't

The `--cpu` (synthetic) path and the Edge TPU path are **completely different
algorithms operating at different resolutions**. Seeing correct flow in CPU
mode does NOT validate the Edge TPU pipeline.

**CPU mode** (`algorithm_modes.py:390-394`):
```python
shift, _ = cv2.phaseCorrelate(
    self.prev_gray.astype(np.float32), gray.astype(np.float32)
)
```
- Uses OpenCV's `phaseCorrelate` (FFT-based sub-pixel phase correlation).
- Operates on **full 640×480** frames (no downscale).
- Works directly on raw pixel intensities (no Gabor features, no pooling).
- Phase correlation is inherently robust to intensity variations and has
  sub-pixel precision via the cross-power spectrum peak.
- At 120 px/sec and 30 fps, sees **4 px/frame shift** — easily detectable.

**Edge TPU mode** (`algorithm_modes.py:396-402`):
```python
prev_resized = cv2.resize(self.prev_gray, (64, 64), INTER_AREA)
curr_resized = cv2.resize(gray, (64, 64), INTER_AREA)
dx, dy = self.flow_engine.compute(prev_resized, curr_resized)
```
- Downscales 10× to 64×64 first → the 4 px/frame becomes 0.4 px/frame.
- Extracts 8-channel Gabor features via Edge TPU.
- AVG_POOL_2D further reduces to 16×16 → 0.1 pooled-px/frame.
- Cross-correlation over 81 displacement bins (±4 pooled pixels).
- Soft argmax with temperature=0.1 to get sub-pixel result.

**Key differences that explain the discrepancy**:

| | CPU mode | Edge TPU mode |
|---|---|---|
| Algorithm | FFT phase correlation | Gabor features + spatial correlation |
| Resolution | 640×480 | 64×64 → 16×16 (pooled) |
| Motion per frame | 4 px | 0.1 pooled px |
| Quantization | float32 throughout | uint8 → int8 → uint8 → int16 → float32 |
| Sub-pixel method | Cross-power spectrum peak | Softmax-weighted argmax |

The CPU mode succeeding only proves `cv2.phaseCorrelate` can detect 4 px
shifts at 640×480 — it says nothing about whether the Gabor + correlation
pipeline can detect 0.1 pooled-pixel shifts through int8 quantization.

**To make a fair comparison**, the CPU mode would need to:
1. Downscale to 64×64 with `INTER_AREA`
2. Apply the same Gabor kernels (via `scipy.ndimage.convolve` or similar)
3. Pool to 16×16
4. Run the same 81-bin correlation + soft argmax

Without that, "CPU works, Edge TPU doesn't" is comparing apples to oranges.

**This is now implemented** as the `--cpu-replica` mode (see Section 11 below).

### 10.5 Temperature Sensitivity

With `temperature=0.1`, the softmax is extremely sharp. If the true peak
is at (0.1, 0), but quantization noise creates a slightly higher bin at
(0, 0), the result snaps to (0, 0) rather than (0.1, 0). A higher
temperature (e.g., 0.5) would give smoother sub-pixel estimates but
noisier results. The current value is a reasonable default but may need
tuning for specific use cases.

---

## 11. CPU Replica Mode (`--cpu-replica`)

### 11.1 Purpose

The `--cpu` mode uses `cv2.phaseCorrelate` — a completely different algorithm at full
640×480 resolution. It can't serve as ground truth for debugging the Edge TPU pipeline.
The `--cpu-replica` mode faithfully reproduces the Edge TPU's integer arithmetic on CPU,
step by step, so outputs can be compared directly.

### 11.2 Integer Pipeline (`cpu_replica.py`)

```
Input: uint8 (H, W) grayscale
  ↓
Stage 1: QUANTIZE (uint8 → int8)
  int8_out = clip(round(uint8 * M + q_int8_zp - input_zp * M), -128, 127)
  Default: int8 = uint8 - 128
  ↓
Stage 2: DEPTHWISE_CONV_2D (int8 × int8 → int32 → int8, fused ReLU)
  acc = bias + sum((input_int8 - input_zp) * weight_int8)     ← KEY: subtract input_zp BEFORE multiply
  Padding: SAME with pad_value = input_zp (→ 0 after subtraction)
  Requantize: clip(round(acc * M_ch + conv_output_zp), -128, 127)
  ReLU: max(val, conv_output_zp)
  ↓
Stage 3: AVG_POOL_2D (4×4, stride 4, VALID)
  pool_int8 = clip(round(block_sum / 16), -128, 127)
  ↓
Stage 4: QUANTIZE (int8 → uint8)
  uint8_out = clip(round((pool_int8 - pool_zp) * M_final + final_zp), 0, 255)
  Default: uint8 = pool_int8 + 128
  ↓
Output: uint8 (H/4, W/4, 8) features
```

Then: same correlation + soft argmax as `OpticalFlow._compute_from_uint8()`.

### 11.3 Key Implementation Details

- **Input zero-point subtraction**: TFLite's quantized conv computes
  `acc = sum((input - input_zp) * weight)`, NOT `sum(input * weight)`.
  Missing this causes the center pixel of a uniform image to incorrectly
  read as zero instead of the correct non-zero value.
- **Per-channel requantization**: Each Gabor filter channel has its own
  weight scale. The multiplier `M_ch = q_int8_scale * weight_scale[ch] / conv_output_scale`
  must use float64 to avoid precision loss.
- **AVG_POOL rounding**: TFLite uses fixed-point multiply-and-shift internally,
  not true float division. Expect ±1 LSB differences vs. the replica.
- **Validated**: max ±1 LSB vs. TFLite interpreter (`BUILTIN_WITHOUT_DEFAULT_DELEGATES`
  to avoid XNNPACK quantization bug).

### 11.4 Usage

```bash
# Synthetic panning pattern (speed: 600 px/sec)
python -m libredgetpu.gui --cpu-replica --synthetic --pattern panning

# With webcam
python -m libredgetpu.gui --cpu-replica
```

The HUD shows `CPU_REPLICA` in cyan. Non-flow algorithms (SpotTracker, etc.)
fall back to synthetic mode when `--cpu-replica` is active.

### 11.5 Debugging with Intermediates

```python
from libredgetpu.gui.cpu_replica import CPUReplicaOpticalFlow
engine = CPUReplicaOpticalFlow(height=64, width=64, verbose=True)
vx, vy = engine.compute(frame_t, frame_t1)
intermediates = engine.get_intermediates()
# Keys: input_uint8, after_quantize_int8, conv_acc_int32,
#        conv_output_int8, pool_output_int8, final_uint8
```

---

## Appendix: File Reference Summary

| Stage | File | Key Lines |
|-------|------|-----------|
| Frame capture (synthetic) | `camera.py` | 178-232 |
| Frame capture (webcam) | `camera.py` | 29-78 |
| GUI loop | `app.py` | 90-146 |
| MJPEG streaming | `app.py` | 215-234 |
| OpticalFlow mode init | `algorithm_modes.py` | 355-371 |
| OpticalFlow process | `algorithm_modes.py` | 373-421 |
| VisualCompass mode | `algorithm_modes.py` | 430-500 |
| Feature extraction | `optical_flow_module.py` | 181-224 |
| Flow computation | `optical_flow_module.py` | 254-272 |
| uint8 flow core | `optical_flow_module.py` | 302-332 |
| Global correlation | `optical_flow_module.py` | 376-419 |
| Soft argmax | `optical_flow_module.py` | 421-441 |
| Input quantization | `_base.py:223-226`, `_quantize.py:19-38` | |
| Edge TPU execution | `_base.py` | 134-173 |
| Output relayout | `delegate.py` | 381-445 |
| TileLayout dataclass | `delegate.py` | 60-68 |
| Model builder (pooled) | `tflite_builder.py` | 1758-1906 |
| VisualCompass yaw | `visual_compass.py` | 51-72, 118-129 |
| CPU replica pipeline | `gui/cpu_replica.py` | Full file |
| CPU replica tests | `tests/test_cpu_replica.py` | Full file |
