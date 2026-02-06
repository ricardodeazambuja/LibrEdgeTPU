# Optical Flow Engineering Log

Chronological narrative of getting optical flow working on the Edge TPU.
Five bugs formed a dependency chain: each fix was necessary but not sufficient
until all five were resolved.  A sixth issue (XNNPACK delegate bug) was a red
herring that complicated debugging.

---

## Timeline

| Date | Bug # | Discovery | Status |
|------|-------|-----------|--------|
| Feb 7 | — | Initial OpticalFlow implementation | Broken: always returns (0, 0) or random signs |
| Feb 8 AM | 1 | Input not normalized | Fixed |
| Feb 8 AM | 2 | Conv output scale saturated | Fixed |
| Feb 8 midday | — | XNNPACK delegate bug found | Red herring (doesn't affect Edge TPU) |
| Feb 8 PM | 3 | Wrong filter orientations + broken channels | Fixed |
| Feb 8 PM | 4 | Vertical flow destroyed by output tiling | Fixed |
| Feb 9 | 5 | GUI shows zeros (aliasing + tuning) | Fixed |
| Feb 9 | — | CPU replica pipeline built, Conv2D scale bug root-caused | Validated |

---

## Bug 1: Input Not Normalized

### Symptoms

All Gabor features saturated or near-zero.  Correlation gave random peaks.

### Root cause

Raw uint8 pixel values `[0, 255]` were passed directly to `_quantize_input()`,
which expected float values in `[0, 1]` (since the model uses
`input_scale = 1/255, input_zp = 0`).

Without normalization, `quantize(128) = clamp(round(128 / (1/255)) + 0, 0, 255)`
overflows massively.  The quantized values saturated at 255 for almost all
pixels, producing a flat input to the convolution.

### Fix

```python
image_normalized = image.astype(np.float32) / 255.0
quantized = self._quantize_input(image_normalized)
```

Added in `_extract_features_uint8()` in `optical_flow_module.py`.

### Validation

After fix: features showed meaningful spatial structure per channel, but
correlation still broken (Bugs 2-4 remained).

---

## Bug 2: Conv Output Scale Too Small (Saturation)

### Symptoms

Even with correct input normalization, most Gabor feature channels saturated
at 127 (the int8 maximum).  The convolution output had only 2-3 unique values
per channel.  Soft argmax on a near-uniform distribution returns the center
(0, 0).

### Wrong hypothesis explored

Initially suspected the XNNPACK delegate bug (see below) was causing the
saturation.  Spent ~40 minutes isolating that bug before realizing it only
affects TFLite CPU inference, not the Edge TPU.

### Root cause

The output scale was estimated using an RMS-based formula:

```python
rms_factor = 127.0 / sqrt(3.0)    # ≈ 73.3
typical_acc = sqrt(ksize²) * rms_factor² / 127.0
```

This gave `scale ≈ 7.2e-05`, but the Edge TPU produced values that needed
`scale ≈ 0.002-0.003` — **30-40× larger**.

The RMS estimate assumed inputs centered at zero (the int8 range `[-128, 127]`).
But TFLite quantized convolution subtracts `input_zp` before multiplication:

```
acc = Σ (input[i] - input_zp) × weight[i]
```

For `input_zp = -128`, the effective input range is `[0, 255]`, not `[-128, 127]`.
This roughly doubles the maximum accumulator value, and the RMS estimate was
already too small for the worst case.

### Fix

Replaced RMS estimate with 150% worst-case:

```python
worst_case_acc = ksize * ksize * 127.0
conservative_acc = worst_case_acc * 1.5    # 150% safety margin
conv_output_scale = conservative_acc * q_int8_scale * gabor_weight_scale / 127.0
```

The 150% factor was determined iteratively:
- 100% (exact worst-case): channels 5 and 7 still saturated
- 150%: all 8 channels have meaningful dynamic range

### Validation

After fix: all channels showed >10 unique values.  But correlation still
returned wrong peaks (Bugs 3-4 remained).

---

## Bug 3: Wrong Filter Orientations and Broken Channels

### Symptoms

Per-channel feature comparison between CPU reference and Edge TPU:
- Channels 0, 7: MAE < 0.3 (nearly perfect)
- Channels 1-6: MAE 11-78 (terrible)

Channels 5 and 7 showed a ~128 offset (suggesting int8/uint8 confusion).

### Wrong hypotheses explored

1. **Weight format mismatch (HWIO vs OHWI)**: The Gabor kernels were generated
   as `[7, 7, 1, 8]` (HWIO) and stored in a tensor declared as
   `[8, 7, 7, 1]` (OHWI).  Transposing `(3, 0, 1, 2)` was tried — it helped
   channels 1 and 3 but made channels 5 and 7 worse (MAE 128-138).

2. **Multi-channel Conv2D ordering**: Investigated whether Conv2D applied
   filters in a different order than expected.

### Root cause

Two intertwined issues:

1. **x_theta carrier convention**: The original Gabor generator used
   `cos(2π x'/λ)` as the carrier.  At θ=0, this produces vertical stripes
   (carrier oscillates along X), which detects vertical edges.  But the
   standard Gabor convention is that θ=0 detects horizontal edges.  Switching
   to `y_theta` (`cos(2π y'/λ)`) fixed the orientation convention.

2. **Conv2D weight format mismatch**: Standard Conv2D expects weights in OHWI
   format `[out_channels, H, W, in_channels]`, but the Gabor kernels were
   generated as HWIO `[H, W, 1, n_filters]`.  Switching to DEPTHWISE_CONV_2D
   with `depth_multiplier=N` and weight shape `[1, H, W, N]` eliminated the
   format confusion entirely — depthwise convolution treats the last dimension
   as independent filters, matching the Gabor bank semantics.

### Fix

1. Changed carrier from `x_theta` to `y_theta` in `_generate_gabor_kernels()`.
2. Changed from Conv2D to DEPTHWISE_CONV_2D in `build_optical_flow()` and
   `build_optical_flow_pooled()`.
3. Added per-channel quantization (each Gabor filter gets its own scale).

### Validation

After fix: all 8 channels matched CPU reference (MAE < 1.0 per channel).
But full pipeline still failed vertical shifts (Bug 4 remained).

---

## Bug 4: Vertical Flow Destroyed by Output Tiling

### Symptoms

With correct features per-channel, the full flow pipeline showed:
- Horizontal shifts: partially detected (flow sign correct but magnitude off)
- Vertical shifts: always returned vx ≈ -4, vy ≈ +4 (random-looking)
- Identical frames: sometimes returned non-zero flow

### Wrong hypotheses explored

1. **Correlation implementation bug**: Re-implemented correlation with scipy
   and manual loops — same wrong results.
2. **Quantization noise**: Added noise to synthetic frames — no change.
3. **Weight transpose residual**: Tried all 24 permutations of weight axes —
   none fixed vertical flow.

### Key experiment: shift equivariance test

Created a simple test: convolve an image, shift it by (dx, dy) pixels, convolve
again, and check if the features are related by the same shift.

```python
# Feature from shifted input should equal shifted feature
corr = Σ feat(image)[y+dy, x+dx] * feat(shift(image, dx, dy))[y, x]
# Should be 1.000 for all (dx, dy)
```

Results **without** relayout:
- Horizontal shifts: correlation 0.757 (partial survival, 16-column tile width)
- Vertical shifts: correlation 0.275 (destroyed, 4-row tile height)

Results **with** `relayout_output()`:
- All shifts: correlation **1.000** (perfect equivariance)

### Root cause

The Edge TPU stores convolution output in a **tiled memory layout** (TYXZ
format).  For a 64×64×8 output, the hardware uses a 4×4 grid of 16×16-pixel
tiles (16 tiles total).

The raw USB bytes are in tile order, not raster order.  Without de-scattering:
- Within each 16-column tile, horizontal relationships are preserved
- Across 4-row tile boundaries, vertical relationships are scrambled

Other modules (SpotTracker, PatternTracker) don't need relayout because
SOFTMAX→FC→CONCAT collapses spatial dimensions on-chip.  The postprocessors
(PoseNet, DeepLabV3) already called `relayout_output()`.  OpticalFlow was the
only module that read raw spatial features and was missing the relayout step.

### Fix

```python
from .delegate import relayout_output

# In _extract_features_uint8():
if self._output_layer is not None and self._output_layer.tile_layout is not None:
    return relayout_output(raw_output, self._output_layer)
```

Also added to `compute_raw()` for the raw-bytes code path.

### Validation

After fix:
- 8/8 direction tests pass (left/right/up/down/diagonals)
- Shift equivariance: 1.000 for all displacements
- Pooled mode: 72% of raw bytes differed from relayouted bytes; all shifts
  correct after relayout

---

## Bug 5: GUI Shows Zeros

### Symptoms

After fixing the library pipeline, the GUI's OpticalFlow mode still showed
`vx ≈ 0, vy ≈ 0` for the synthetic panning pattern, or oscillated between
±3 on vx.

### Wrong hypotheses explored

1. **Hardware/relayout broken for pooled mode**: Tested with identical frames
   on hardware — peak at (0, 0), correct. Rejected.
2. **Noise sensitivity**: Added noise std 0-50 to identical frames — peak
   stayed at (0, 0). Rejected.
3. **Webcam frame issues**: Captured 12 real webcam frames, ran through full
   pipeline — all vx = 0 for static camera. Rejected.
4. **Race condition in GUI loop**: Simulated exact GUI processing loop —
   all vx = 0. Rejected.

### Root cause

**Two independent issues:**

1. **`cv2.resize` aliasing at large downscale (640→64 = 10×).**
   The GUI resizes webcam frames from 640×480 to 64×64 using the OpenCV
   default interpolation (`INTER_LINEAR` = bilinear).  Bilinear interpolation
   samples only nearby source pixels, skipping most of the image.  For the
   synthetic panning pattern (narrow vertical stripes), this creates severe
   moiré artifacts.  Small physical shifts (4px at 640) appear as large jumps
   (±3 pooled px) due to aliasing.

   Fix: use `cv2.INTER_AREA` which properly averages all source pixels in
   each destination pixel's footprint.

2. **Panning pattern stripe widths below Nyquist frequency.**
   The original stripes were 8-32px wide at 640 resolution, which maps to
   0.8-3.2px at 64×64 — below the Nyquist limit.  Even with INTER_AREA,
   stripes near the Nyquist limit can alias.

   Fix: use wider stripes (40-120px at 640 resolution = 4-12px at 64×64).

3. **Pool factor too coarse for GUI sensitivity.**
   The library default `pool_factor=4` means the minimum detectable motion is
   ~4 raw pixels at 64×64 = ~40px at 640×480 (very large camera motion).

   Fix: GUI uses `pool_factor=2` for 2× better sensitivity, and the panning
   pattern scrolls at 600 px/sec (producing ~1 pooled pixel/frame at 30 FPS).

### Validation

After all three fixes:
- Panning + OpticalFlow: 15 frames, all vx = 0.000 vy = 0.000 (for static
  scene; correct non-zero values for actual panning)
- Panning + VisualCompass: 10 frames, all yaw = 0.000° (static)
- All other synthetic patterns: zeros
- Full test suite: all pass

---

## Red Herring: XNNPACK Delegate Bug

### Discovery

During Bug 2 investigation, tested the quantized Gabor model on TFLite CPU
to validate features.  A simple test — constant uint8=128 input through
`QUANTIZE → DEPTHWISE_CONV_2D` with a 3×3 identity kernel — produced
**non-constant output**:

```
Expected: all 0 (constant input through identity kernel)
Actual:   {-105, -30, 100} in a position-dependent TYXZ pattern
```

### Investigation (40 minutes)

1. Tested 11 input values: only `uint8=0` produced correct constant output.
   All others showed errors of 30-183.
2. Isolated QUANTIZE alone: worked correctly.
3. Isolated the chain QUANTIZE → DEPTHWISE_CONV_2D: broken.
4. Hypothesis: operator fusion bug in TFLite.

### Root cause

TensorFlow Lite's **XNNPACK delegate** has a bug in quantized convolution.
Confirmed by testing with and without XNNPACK:

```python
# WITH XNNPACK (default): WRONG
interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
# → uint8=128 produces {-105, -30, 100}

# WITHOUT XNNPACK: CORRECT
interpreter = tf.lite.Interpreter(
    model_content=tflite_bytes,
    experimental_preserve_all_tensors=True  # disables XNNPACK
)
# → uint8=128 produces 0 (constant)
```

### Why it was a red herring

The Edge TPU does **not** use the XNNPACK delegate — it has its own hardware
execution path.  The XNNPACK bug only affects TFLite CPU inference used for
testing.  The optical flow saturation (Bug 2) was caused by the output scale
estimate, not by XNNPACK.

However, the XNNPACK bug made it impossible to validate Edge TPU output against
TFLite CPU output, which complicated debugging of Bugs 2-4.

### Permanent workaround

All test files now use `experimental_preserve_all_tensors=True` when creating
TFLite interpreters for quantized model validation.

---

## Bonus: Conv2D Output Scale Root Cause (Feb 9)

### Context

While building the CPU replica pipeline (`gui/cpu_replica.py`), the same
output scale saturation bug was found in `build_pattern_tracker` and
`build_spot_tracker` (color variant).

### Root cause (generalized)

The formula used for output scale estimation was:

```python
conv_output_max = q_int8_scale * conv_weight_scale * n_weights * 127
```

This assumed the maximum input value is 127 (the int8 maximum).  But TFLite
quantized convolution computes:

```
acc = Σ (input[i] - input_zp) × weight[i]
```

For `input_zp = -128`, the effective input range is `[0, 255]`, not `[-128, 127]`.
The correct maximum effective input is `127 - (-128) = 255`, making the true
worst-case accumulator value roughly **2× larger** than estimated.

When the output scale is 2× too small, the requantization maps all real-world
accumulator values to 127 (saturated).  For softmax over a uniform distribution,
all probabilities are equal → soft argmax returns the center → output is zero.

### Fix

```python
max_effective_input = (127 - q_int8_zp) * q_int8_scale  # 255 * scale
weight_abs_sum = sum(abs(weights)) * scale
conv_output_max = max_effective_input * weight_abs_sum * safety_factor
```

Applied to `build_pattern_tracker`, `build_spot_tracker` (color), and
`build_optical_flow` / `build_optical_flow_pooled`.

---

## Lessons Learned

1. **The input_zp subtraction is the most common source of quantization bugs.**
   Three different bugs (output scale, CPU replica, PatternTracker) all had
   the same root cause: forgetting that `input_zp = -128` shifts the effective
   input range from `[-128, 127]` to `[0, 255]`.

2. **Test against the hardware, not the simulator.**  The XNNPACK bug in TFLite
   CPU meant that simulator-based validation was unreliable.  The Edge TPU
   produces correct results for computations that TFLite CPU gets wrong.

3. **Shift equivariance is the key diagnostic.**  When features look correct
   per-channel but flow is wrong, testing whether `feat(shift(image))` equals
   `shift(feat(image))` immediately reveals tiling/layout bugs.

4. **Output scale estimation should be worst-case, not average-case.**  The
   Edge TPU's int8 output has limited dynamic range.  A few saturated samples
   (clipped to 127) are less harmful than a uniformly saturated output (all 127).
   It is better to waste some precision headroom than to clip.

5. **GUI issues are often independent of library bugs.**  The aliasing bug
   (Bug 5) was purely a GUI preprocessing issue — the library pipeline was
   correct.  Always test the library API directly before debugging the GUI.

6. **Document each bug as you fix it.**  Without the investigation trail, the
   connection between input_zp subtraction and three separate bugs would have
   been invisible.  The scattered docs (now consolidated here) were essential
   for the final root-cause synthesis.

---

## Files Changed (Complete List)

| File | Change | Bug # |
|------|--------|-------|
| `libredgetpu/optical_flow_module.py` | Add `/255.0` normalization, add `relayout_output()` calls | 1, 4 |
| `libredgetpu/tflite_builder.py` | Switch to y_theta carrier, DEPTHWISE_CONV_2D, per-channel quant, 150% worst-case scale | 2, 3 |
| `libredgetpu/delegate.py` | Implement `relayout_output()` with TileLayout lookup tables | 4 |
| `libredgetpu/gui/algorithm_modes.py` | `cv2.INTER_AREA` on all resize calls, `pool_factor=2` | 5 |
| `libredgetpu/gui/camera.py` | Wider panning stripes (40-120px), 600 px/sec speed | 5 |
| `libredgetpu/gui/cpu_replica.py` | Full integer-faithful CPU replica pipeline | Validation |
| `tests/test_optical_flow.py` | Shift equivariance tests, relayout validation, builder tests | 2, 3, 4 |
| `tests/test_cpu_replica.py` | Per-stage validation, TFLite comparison, end-to-end flow | Validation |
| `tests/test_ground_truth.py` | Ground-truth validation against known shifts | Validation |

---

## Original Investigation Documents

This log consolidates content from the following files, which have been deleted:

- `docs/OPTICAL_FLOW_BUG_INVESTIGATION.md` — Initial multi-channel MAE analysis (pre-relayout)
- `docs/OPTICAL_FLOW_DEBUG.md` — Tiling discovery and relayout fix (the breakthrough)
- `docs/OPTICAL_FLOW_DEBUG_SUMMARY.md` — Output scale analysis (partial progress)
- `docs/OPTICAL_FLOW_FINDINGS.md` — Weight format investigation (HWIO vs OHWI)
- `docs/PHASE1_FINDINGS.md` — XNNPACK anomaly first observed
- `docs/QUANTIZE_BUG_ROOT_CAUSE.md` — XNNPACK bug confirmed
- `docs/QUANTIZE_INVESTIGATION_SUMMARY.md` — XNNPACK systematic testing
- `docs/INVESTIGATION_COMPLETE.md` — XNNPACK phase wrap-up

Additional reference: `experiments/OPTICAL_FLOW_GUI_FIX_LOG.md` — GUI aliasing investigation trail (retained in experiments/).
