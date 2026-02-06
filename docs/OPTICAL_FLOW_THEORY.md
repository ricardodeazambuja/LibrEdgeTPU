# Optical Flow on the Edge TPU: Theory and Implementation

A methods-section document covering the full pipeline for computing global
optical flow on a quantized int8 MAC engine.

---

## 1. Introduction

The Google Coral Edge TPU is a 64x64 systolic array clocked at 480 MHz that
performs int8 multiply-accumulate (MAC) operations.  Its native workload is
neural network inference, but the hardware is fundamentally a matrix engine:
any computation expressible as quantized convolutions or matrix multiplies can
be offloaded to it.

**Global optical flow** — estimating a single (vx, vy) displacement vector
between two frames — is exactly such a computation.  The feature extraction
step (Gabor filtering) is a depthwise convolution, and the correlation step is
a sum of element-wise products.

### Why not classical methods?

Classical optical flow algorithms are a poor fit for the Edge TPU:

| Method | Why it doesn't map |
|--------|--------------------|
| **Lucas-Kanade** | Requires spatial gradient computation (Ix, Iy, It), then solving a 2x2 linear system per pixel. The gradient→solve chain has no direct convolution equivalent. |
| **Horn-Schunck** | Iterative PDE solver with a Laplacian smoothness constraint. Each iteration requires a full-image pass — no batch-MAC equivalent. |
| **Phase correlation** | Requires 2D FFT, which is not an Edge TPU operation. (FFT can be decomposed into butterfly matmuls, but the per-stage requantization to 8 bits destroys precision after ~3 stages.) |
| **Dense flow (FlowNet, RAFT)** | Too large for the 8 MB on-chip SRAM cache. Would require streaming mode (10-100x slower) and still produce 8-bit output at each decoder stage. |

### The design choice: Gabor feature correlation

The approach implemented in libredgetpu splits the pipeline into two stages:

1. **Edge TPU** — Extract multi-orientation, multi-scale Gabor features via a
   single depthwise convolution (all 8 filters in one model call).
2. **CPU** — Downsample, cross-correlate over a displacement grid, normalize
   by overlap area, and apply soft argmax for sub-pixel output.

This split exploits the Edge TPU for the compute-intensive convolution
(49 MACs per pixel per filter × 8 filters × 4096 pixels ≈ 1.6M MACs) while
keeping the lightweight correlation (81 displacements × 16×16×8 ≈ 166K
multiply-adds) on the CPU where float arithmetic avoids the 8-bit output wall.

---

## 2. Gabor Feature Extraction

### 2.1 The Gabor function

A 2D Gabor filter is a Gaussian envelope modulated by a sinusoidal carrier:

```
g(x, y) = exp(-(x'² + y'²) / (2σ²)) · cos(2π y' / λ)
```

where the rotated coordinates are:

```
x' =  x cos θ + y sin θ
y' = -x sin θ + y cos θ
```

and:

- **θ** is the orientation angle (the edge direction the filter responds to)
- **σ** is the Gaussian envelope scale (controls spatial extent)
- **λ** is the carrier wavelength (controls frequency selectivity)

Note: the carrier uses **y'** (not x'), following the convention where at θ=0
the carrier oscillates along the Y axis.  This means θ=0 detects **horizontal
edges** (the carrier is perpendicular to the edge orientation).

### 2.2 Parameter choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **λ = 2σ** | One cycle per Gaussian envelope | Optimal bandwidth: the envelope contains exactly one full cycle of the carrier, maximizing the joint space-frequency localization. |
| **γ = 1** (circular envelope) | No aspect ratio parameter | Rotation is handled by the orientation bank — each θ sees a symmetric envelope. An elongated envelope (γ < 1) would be redundant. |
| **4 orientations** | θ ∈ {0°, 45°, 90°, 135°} | Covers the four principal edge directions. Finer sampling (e.g., 8 orientations) would double the output channels and USB transfer with diminishing returns for global flow. |
| **2 scales** | σ ∈ {1.5, 3.0} | Fine scale (σ=1.5, λ=3.0) captures high-frequency texture; coarse scale (σ=3.0, λ=6.0) captures larger structures. Together they provide robustness across spatial frequencies. |
| **7×7 kernel** | 7 = 2·⌊3σ_max⌋ + 1 | For σ=3.0, the kernel extends to ±3.0 pixels (99.7% of Gaussian energy). Larger kernels would increase Edge TPU compute with negligible information gain. |

This produces **8 filters** (4 orientations × 2 scales) stored as a single
depthwise convolution kernel of shape `[1, 7, 7, 8]`.

### 2.3 Orientation convention

The orientation parameter θ specifies the **edge direction** the filter responds
to, not the carrier direction:

| θ | Carrier oscillates along | Detects |
|---|--------------------------|---------|
| 0° (0) | Y axis (vertical stripes in kernel) | Horizontal edges |
| 45° (π/4) | NE-SW diagonal | 45° diagonal edges |
| 90° (π/2) | X axis (horizontal stripes in kernel) | Vertical edges |
| 135° (3π/4) | NW-SE diagonal | 135° diagonal edges |

This `y_theta` convention (carrier along y' at θ=0) is used consistently in
both the Gabor kernel generator (`_generate_gabor_kernels` in `tflite_builder.py`)
and the CPU replica (`cpu_replica.py`).

### 2.4 Normalization and quantization

Each 7×7 kernel is normalized to `[-1, 1]` by dividing by its peak absolute
value:

```python
gabor = gabor / max(|gabor|)    # per-filter, max(|g|) ≈ 1.0
```

The kernels are then quantized to int8 with **per-channel** scales:

```python
scale_ch = max(|kernel_ch|, 1e-6) / 127.0
int8_ch = clip(round(kernel_ch / scale_ch), -127, 127)
```

Per-channel quantization is critical: each Gabor filter has a different peak
magnitude (the fine-scale filters have smaller support and lower peak values).
A single global scale would waste precision on the smaller kernels.

### 2.5 ReLU half-wave rectification

The depthwise convolution uses fused ReLU activation.  Gabor responses are
bipolar (positive where the carrier aligns with the input, negative where it
anti-aligns), but ReLU retains only the positive phase.  This is acceptable
because:

1. **Redundancy**: The positive and negative phases carry the same orientation
   and scale information — they differ only in carrier phase (0 vs π).
2. **Edge TPU native**: ReLU is a zero-cost fused activation on the Edge TPU.
3. **Non-negative features**: All-positive features simplify the correlation
   stage (see Section 7 on why mean subtraction is harmful).

---

## 3. Quantized Convolution on Edge TPU

### 3.1 Input quantization

The input image (uint8, [0, 255]) is first normalized to float [0, 1]:

```python
image_float = image_uint8 / 255.0
```

then quantized to int8 via the QUANTIZE operator:

```
int8_val = clamp(round(float_val / scale + zp), -128, 127)
```

with `scale = 1/255`, `zp = -128`.  This maps:

| uint8 | float | int8 |
|-------|-------|------|
| 0 | 0.0 | -128 |
| 128 | 0.502 | 0 |
| 255 | 1.0 | 127 |

### 3.2 Depthwise convolution accumulation

The DEPTHWISE_CONV_2D operator computes, for each output channel `c` and
spatial position `(y, x)`:

```
acc[y, x, c] = bias[c] + Σ_{ky, kx} (input[y+ky, x+kx] - input_zp) × weight[ky, kx, c]
```

Key points:

- **input_zp subtraction happens before multiplication**.  For `input_zp = -128`,
  the effective input value for int8=0 is `0 - (-128) = 128`, not 0.  This
  means the effective input range is `[0, 255]` (uint8 range), not `[-128, 127]`.
- **SAME padding** fills border pixels with `input_zp`.  After subtraction,
  `(input_zp - input_zp) = 0`, so padding contributes zero to the accumulator.
  This is mathematically equivalent to zero-padding the subtracted input.
- **Bias is zero** (we use unbiased Gabor filters).
- **Accumulation is in int32** — exact for up to 2^15 MACs with 8-bit operands.

### 3.3 Per-channel requantization

The int32 accumulator is requantized to int8 output via a per-channel multiplier:

```
M_ch = (input_scale × weight_scale_ch) / output_scale
output_int8[y, x, c] = clamp(round(acc[y, x, c] × M_ch + output_zp), -128, 127)
```

The multiplier `M_ch` is baked into the Edge TPU's execution-only instructions
at compile time.

### 3.4 Output scale estimation

The output scale determines the dynamic range of the requantized output.  If
too small, the output saturates at 127 for all spatial positions; if too large,
precision is wasted.

**The RMS estimate fails**.  An early attempt used:

```
typical_acc = sqrt(ksize²) × rms_factor² / 127
```

This gave `scale ≈ 7.2e-05`, but the Edge TPU empirically produced values
30-40× larger.  The root cause: the RMS estimate assumed inputs centered at
zero (the int8 range `[-128, 127]`), but after the `input_zp` subtraction the
effective inputs are in `[0, 255]` — roughly 2× the assumed range.

**The correct estimate** uses a 150% worst-case:

```python
worst_case_acc = ksize² × 127.0          # max input × max weight × num elements
conservative_acc = worst_case_acc × 1.5  # 150% safety margin
conv_output_scale = conservative_acc × input_scale × mean_weight_scale / 127.0
```

The 150% factor was determined empirically: 100% still saturated channels 5 and
7 (the coarse-scale filters with higher response magnitudes).

### 3.5 Output zero point

With fused ReLU, all outputs are non-negative, so `output_zp = -128` maps the
quantized zero to the bottom of the int8 range.  This gives the full `[0, 255]`
uint8 range for positive values.

---

## 4. Block Pooling

The full-resolution Gabor features (`[H, W, 8]`) are spatially downsampled
before correlation.  Two modes are supported:

### 4.1 CPU block-sum pooling (standard mode)

The CPU reshapes the feature map into blocks and sums:

```python
# feat: [H, W, C] int16 (after subtracting output zero point)
pooled = feat.reshape(H//P, P, W//P, P, C).sum(axis=(1, 3))  # [H/P, W/P, C] int32
```

This is a **sum** (not mean) — integer division is avoided.  The sum pools
are larger by a factor of P² compared to averages, but since both correlation
operands are pooled identically, the ratio is preserved.

### 4.2 Fused AVG_POOL_2D (pooled mode)

When `pooled=True`, the model includes an `AVG_POOL_2D` operator after the
convolution, producing `[H/P, W/P, 8]` directly on the Edge TPU.  This
reduces USB transfer by a factor of **P²** (e.g., 64×64×8 = 32 KB →
16×16×8 = 2 KB for P=4).

The Edge TPU's AVG_POOL_2D divides by the pool count (P²) internally and
requantizes the result to int8.  This introduces a small quantization loss
compared to CPU block-sum, but the 16× USB bandwidth reduction is a net win
for bandwidth-constrained deployments (e.g., Raspberry Pi).

---

## 5. Output Tiling and Relayout

### 5.1 The problem

The Edge TPU stores convolution output activations in a **tiled memory layout**
(TYXZ format), not standard row-major YXZ.  For a 64×64×8 output, the hardware
arranges data in a 4×4 grid of 16×16-pixel tiles (16 tiles total).

Without de-scattering, the raw bytes appear spatially scrambled:

- **Horizontal structure partially survives** (16-column tile width preserves
  left-right relationships within each tile)
- **Vertical structure is destroyed** (4-row tile height breaks vertical
  continuity across tile boundaries)

This manifests as broken vertical correlations: horizontal flow detection works
roughly, but vertical flow gives nonsense.

### 5.2 The relayout algorithm

`relayout_output()` in `delegate.py` uses six lookup tables extracted from the
DarwiNN executable's `TileLayout` structure:

```python
for y in range(Y):
    for x in range(X):
        tile_id = y_tile_id_map[y] + x_tile_id_map[x]
        base = (tile_byte_offsets[tile_id]
                + y_local_y_offset[y] * x_local_y_row_size[x]
                + x_local_byte_offset[x])
        dest[y, x, :] = src[base : base + Z]
```

| Table | Indexed by | Purpose |
|-------|-----------|---------|
| `y_tile_id_map[y]` | row | Maps row to tile row ID |
| `x_tile_id_map[x]` | column | Maps column to tile column ID |
| `tile_byte_offsets[id]` | tile ID | Byte offset of each tile in raw buffer |
| `x_local_byte_offset[x]` | column | Byte offset within tile row |
| `y_local_y_offset[y]` | row | Local row index within tile |
| `x_local_y_row_size[x]` | column | Row stride within this tile column |

This is a **pure permutation** — `size_bytes == H × W × C` exactly, with no
padding.

### 5.3 Which modules need relayout

| Module | Needs relayout? | Why |
|--------|----------------|-----|
| **OpticalFlow** | Yes | Spatial feature maps `[H, W, 8]` are exposed to CPU for correlation |
| **Postprocessors** (PoseNet, DeepLabV3, MultiPose) | Yes | Spatial output maps |
| **SpotTracker / PatternTracker** | No | SOFTMAX→FC→CONCAT collapses spatial dimensions on-chip |
| **MatMulEngine / SimpleInvoker** | No | 1D output (no spatial dimensions) |

---

## 6. Cross-Correlation

### 6.1 Definition

For each candidate displacement (dx, dy) in a ±R grid (default R=4, giving
81 candidates), the global correlation score is:

```
C(dx, dy) = Σ_{y,x,c} F_t(y + dy, x + dx, c) × F_{t+1}(y, x, c)
```

where F_t and F_{t+1} are the pooled feature maps from the reference and
current frames.  The peak of C indicates the displacement that best aligns
the two feature maps.

### 6.2 Efficient implementation

Rather than a Python loop over 81 displacements, the implementation uses
`numpy` stride tricks and `einsum`:

```python
# Pad reference features with zeros (out-of-bounds = no contribution)
padded = np.pad(feat_t, ((R, R), (R, R), (0, 0)), mode='constant')

# Create a sliding window view: shape [2R+1, 2R+1, H, W, C]
view = as_strided(padded,
                  shape=(side, side, H, W, C),
                  strides=(s[0], s[1], s[0], s[1], s[2]))

# Batched dot products: corr_map[j, i] = Σ view[j,i,:,:,:] * feat_t1
corr_map = np.einsum('ijhwc,hwc->ij', view, feat_t1)
```

The `as_strided` call creates a zero-copy view of all 81 shifted versions
simultaneously.  The `einsum` then computes all 81 dot products in a single
vectorized call.  For 16×16×8 features, this is ~10-50× faster than a Python
loop (important on weak CPUs like the RPi Zero).

Integer features use `dtype=np.int64` accumulation to prevent overflow
(16-bit × 16-bit × 2048 elements can exceed int32).

---

## 7. Overlap Normalization

### 7.1 The problem

For a displacement (dx, dy), only `(ph - |dy|) × (pw - |dx|)` spatial
positions overlap between the shifted and unshifted feature maps.  Without
normalization, the center displacement (0, 0) has the maximum overlap and
thus the highest raw correlation, biasing the result toward zero displacement.

### 7.2 The formula

```
C_norm(dx, dy) = C(dx, dy) / ((ph - |dy|) × (pw - |dx|) × n_f)
```

where `ph`, `pw` are the pooled feature map dimensions and `n_f` is the number
of filter channels (8).  This divides by the total number of element-wise
products contributing to each score.

The overlap counts are pre-computed in `__init__` as `_overlap_counts` (an
array of 81 values) and applied via vectorized division.

### 7.3 Why it works for ReLU features

After ReLU, all feature values are non-negative.  For the true displacement
(dx*, dy*), the overlapping region computes `Σ f(y)²` (each element multiplied
by itself), which equals `n × E[X²]` per pixel.  For any other displacement,
the overlapping region computes `Σ f(y) × f(y + δ)`, which equals `n × E[X × X_shifted]`.

By the Cauchy-Schwarz inequality, `E[X²] ≥ E[X × X_shifted]` for any non-trivial
shift δ, with equality only when the features are spatially constant.  Since
Gabor features are not constant (they respond to edges), the true displacement
always has the highest normalized score.

### 7.4 Why mean subtraction is harmful

For all-positive ReLU features, mean subtraction converts the cross-correlation
into a zero-mean centered correlation:

```
C_centered(dx, dy) = Σ (f_t - μ_t)(f_{t+1} - μ_{t+1})
```

This is mathematically the sample covariance.  For stationary textures with
small displacements, `C_centered` peaks at **lag (0, 0)** regardless of the
true displacement, because the autocorrelation of any signal peaks at zero lag.
The mean subtraction removes the information that distinguishes different
displacement hypotheses.

Without mean subtraction, the raw cross-correlation exploits the fact that
`E[X²] > E[X · X_shifted]` for non-constant signals, which is exactly the
signal we need.

---

## 8. Soft Argmax

### 8.1 Definition

The peak displacement is computed as a weighted average over all candidates:

```
pos = Σ_i softmax(C_i / T) × coords_i
```

where `T = 0.1` is the temperature, `C_i` are the normalized correlation scores,
and `coords_i` are the (dx, dy) displacement coordinates.

### 8.2 Temperature effect

- `T → 0`: Approaches hard argmax (winner takes all).  Gives integer-valued output.
- `T → ∞`: All weights equal, output approaches the mean of all coordinates (0, 0).
- `T = 0.1` (default): Sharp enough to produce a clear peak, soft enough for
  sub-pixel interpolation.  A correlation score difference of 1.0 between adjacent
  displacements gives a weight ratio of `exp(1/0.1) ≈ 22026`, so the peak
  dominates heavily.

### 8.3 Numerical stability

The softmax is computed with the standard max-subtraction trick:

```python
corr_shifted = corr - max(corr)          # prevent overflow in exp
weights = exp(corr_shifted / T)
weights /= sum(weights)
vx = Σ weights × dx_coords
vy = Σ weights × dy_coords
```

If `sum(weights) < 1e-12` (uniform zero features), the output defaults to
(0.0, 0.0) — indicating no detectable motion.

---

## 9. Visual Compass

The `VisualCompass` module wraps `OpticalFlow` to convert horizontal pixel
displacement into a yaw angle.

### 9.1 Conversion formula

```
deg_per_pooled_px = fov_deg × effective_pool / image_width
yaw_deg = vx × deg_per_pooled_px
```

where `effective_pool` is the active downsampling factor:
- `fused_pool` (if > 0) — the AVG_POOL_2D factor baked into the Edge TPU model
- `pool_factor` (otherwise) — the CPU-side block-sum downsampling factor

### 9.2 Derivation

One pooled pixel spans `effective_pool` raw pixels in the input image.
The camera's horizontal FOV of `fov_deg` degrees spans `image_width` raw
pixels.  Therefore:

```
1 pooled pixel = effective_pool raw pixels
               = effective_pool × (fov_deg / image_width) degrees
```

A displacement of `vx` pooled pixels thus corresponds to
`vx × effective_pool × fov_deg / image_width` degrees of yaw.

### 9.3 Sign convention

Positive `vx` means rightward scene motion (content moves right between
frame_t and frame_t+1).  This corresponds to rightward camera rotation
(positive yaw viewed from above), matching the standard robotics convention.

### 9.4 Pure-rotation assumption

The linear mapping from pixel displacement to angle assumes pure rotation
(no translation).  This is valid when:
- The scene is distant (parallax from translation is negligible)
- The time between frames is small (large rotations break the small-angle
  approximation)

For a 64×64 image with `pool_factor=4` and `search_range=4`, the maximum
detectable displacement is ±4 pooled pixels = ±16 raw pixels, or
±(16/64) × FOV.  For a 90° FOV camera, this is ±22.5° per frame pair.

---

## 10. Design Decisions

### Why Gabor over learned features?

1. **No training data needed.** Gabor kernels are analytically defined.
   The entire pipeline works without any dataset or training step.
2. **Interpretable.** Each filter has a known orientation and scale.
   When a bug causes wrong flow, you can inspect per-channel responses.
3. **Compact.** 8 filters with 7×7 kernels = 392 weights.  A learned
   feature extractor would need a training pipeline and might overfit to
   specific scenes.
4. **Gabor filters are near-optimal** for joint space-frequency
   localization (Daugman 1985), making them a principled choice for
   texture feature extraction.

### Why global flow over dense flow?

Global flow (one (vx, vy) per frame pair) is appropriate for ego-motion
estimation — the primary robotics use case (heading, speed, yaw).  Dense
flow (one vector per pixel) would require:
- Much larger output tensors (H×W×2 vs scalar)
- Multi-scale pyramids (too large for 8 MB SRAM)
- Iterative refinement (8-bit requantization kills precision per iteration)

### Why DEPTHWISE_CONV_2D?

Depthwise convolution applies each filter independently to the (single) input
channel.  This is semantically correct for Gabor filtering: each orientation/scale
filter operates independently, with no cross-channel mixing.  It also enables
per-channel quantization (each filter gets its own scale), which is critical
for preventing saturation in filters with different dynamic ranges.

Standard Conv2D would create cross-channel weights (input_channels ×
output_channels), which is unnecessary for a single-channel grayscale input
with independent filters.

### Why CPU correlation at 16×16 is fast enough

After pooling (P=4), the feature maps are 16×16×8.  The correlation
computation is:

```
81 displacements × 16 × 16 × 8 = 166,912 multiply-adds
```

At ~1 GHz effective throughput on a modern CPU (even an RPi 4 at ~1.5 GHz),
this takes < 1 ms.  Offloading to the Edge TPU would require building a
specialized correlation model, managing two additional USB transfers, and
dealing with the 8-bit output wall — all for a computation that is already
fast on CPU.

### Why not FFT-based correlation?

Phase correlation via FFT would give sub-pixel accuracy via the Fourier shift
theorem.  However:
1. FFT is not an Edge TPU operation.
2. The 16×16 feature maps are small enough that direct spatial correlation
   (O(N² × D²) where D = displacement range) is competitive with FFT-based
   correlation (O(N² log N²)).
3. FFT-based methods would still need the Gabor feature extraction step, so
   the Edge TPU workload is unchanged.

---

## 11. End-to-End Pipeline Summary

```
Input: two uint8 grayscale frames [H, W]

For each frame:
  1. Normalize:        uint8 [0, 255] → float [0, 1]        (CPU)
  2. Quantize:         float [0, 1] → int8 [-128, 127]      (Edge TPU, QUANTIZE op)
  3. Gabor conv:       int8 → int32 acc → int8 + ReLU        (Edge TPU, DEPTHWISE_CONV_2D)
  4. (Optional) Pool:  int8 [H, W, 8] → int8 [H/P, W/P, 8]  (Edge TPU, AVG_POOL_2D)
  5. Requantize:       int8 → uint8                           (Edge TPU, QUANTIZE op)
  6. Relayout:         TYXZ tiles → YXZ standard order        (CPU, lookup-table permutation)
  7. (If no fused pool) Block-sum pool: uint8 → int32         (CPU)

Then:
  8. Correlate:        81 dot products on pooled features     (CPU, einsum)
  9. Normalize:        divide by overlap counts               (CPU)
  10. Soft argmax:     softmax → weighted sum → (vx, vy)      (CPU)

Output: (vx, vy) displacement in pooled pixels (sub-pixel precision)
```

Total latency: ~2 ms for 64×64 input (two Edge TPU calls × ~0.5 ms each + ~1 ms CPU).

---

## 12. References

- **[Gabor1946]** Gabor, D. (1946). "Theory of communication." *Journal of the IEE*, 93(26), 429-457.
- **[Daugman1985]** Daugman, J. (1985). "Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters." *JOSA A*, 2(7), 1160-1169.
- **[Jain1991]** Jain, A. K. & Farrokhnia, F. (1991). "Unsupervised texture segmentation using Gabor filters." *Pattern Recognition*, 24(12), 1167-1186.
- **[Lewis1995]** Lewis, J. P. (1995). "Fast normalized cross-correlation." *Vision Interface*, 120-123.
- **[Barron1994]** Barron, J. L., Fleet, D. J. & Beauchemin, S. S. (1994). "Performance of optical flow techniques." *IJCV*, 12(1), 43-77.
- **[Fleet2006]** Fleet, D. J. & Weiss, Y. (2006). "Optical flow estimation." In *Handbook of Mathematical Models in Computer Vision*, 237-257.
- **[Jacob2018]** Jacob, B. et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *Proc. IEEE/CVF CVPR*, 2704-2713.
- **[Chapelle2010]** Chapelle, O. & Wu, M. (2010). "Gradient descent optimization of smoothed information retrieval metrics." *Information Retrieval*, 13(3), 216-235.
- **[Zufferey2006]** Zufferey, J.-C. & Floreano, D. (2006). "Fly-inspired visual steering of an ultralight indoor aircraft." *IEEE Trans. Robotics*, 22(1), 137-146.
