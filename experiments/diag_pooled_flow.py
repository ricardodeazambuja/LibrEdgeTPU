#!/usr/bin/env python3
"""Diagnostic: pooled optical flow correlation profile on hardware.

Hypotheses to test:
H1: Pooled mode relayout is incorrect → features scrambled → false peak
H2: Overlap normalization inflates edge displacements for smooth features
H3: Temperature too low → winner-take-all on flat correlation

Experiment:
1. Identical frames → expect peak at (0,0)
2. Print full correlation profile (9×9 grid)
3. Compare pooled vs standard mode
4. Test with textured vs smooth images
"""

import numpy as np
from libredgetpu.optical_flow_module import OpticalFlow

def print_corr_grid(corr, search_range=4):
    """Print correlation as a 9×9 grid with displacement labels."""
    side = 2 * search_range + 1
    grid = corr.reshape(side, side)

    # Find peak
    peak_idx = np.argmax(corr)
    peak_dy = peak_idx // side - search_range
    peak_dx = peak_idx % side - search_range

    print(f"\n  Correlation grid (peak at dx={peak_dx}, dy={peak_dy}, val={corr[peak_idx]:.4f}):")
    print(f"  {'':>6}", end="")
    for dx in range(-search_range, search_range+1):
        print(f"  dx={dx:+d}", end="")
    print()

    for j, dy in enumerate(range(-search_range, search_range+1)):
        print(f"  dy={dy:+d}", end="")
        for i in range(side):
            val = grid[j, i]
            marker = " *" if (j * side + i) == peak_idx else "  "
            print(f" {val:7.2f}{marker}", end="")
        print()

def run_experiment(flow, label, img_t, img_t1):
    """Run one experiment and print diagnostics."""
    print(f"\n{'='*70}")
    print(f"Experiment: {label}")
    print(f"  Mode: {'pooled' if flow.fused_pool else 'standard'}")
    print(f"  Image size: {img_t.shape}")
    print(f"  Feature output: {flow._out_h}×{flow._out_w}×{flow._num_filters}")

    # Extract features
    feat_t = flow._extract_features_uint8(img_t)
    feat_t1 = flow._extract_features_uint8(img_t1)

    print(f"  feat_t shape: {feat_t.shape}, dtype: {feat_t.dtype}")
    print(f"  feat_t range: [{feat_t.min()}, {feat_t.max()}], mean: {feat_t.mean():.1f}")
    print(f"  feat_t1 range: [{feat_t1.min()}, {feat_t1.max()}], mean: {feat_t1.mean():.1f}")

    # Check if features differ between frames
    if np.array_equal(img_t, img_t1):
        feat_diff = np.abs(feat_t.astype(int) - feat_t1.astype(int))
        print(f"  Feature diff (identical input): max={feat_diff.max()}, mean={feat_diff.mean():.3f}")

    # Run correlation pipeline manually
    zp = np.int16(flow._output_info.zero_point)
    feat_t_int = feat_t.astype(np.int16) - zp
    feat_t1_int = feat_t1.astype(np.int16) - zp

    if flow._fused_pool:
        feat_t_f = feat_t_int.astype(np.float32)
        feat_t1_f = feat_t1_int.astype(np.float32)
    else:
        feat_t_f = flow._pool_features_int(feat_t_int).astype(np.float32)
        feat_t1_f = flow._pool_features_int(feat_t1_int).astype(np.float32)

    print(f"  Pooled feat shape: {feat_t_f.shape}")
    print(f"  Pooled feat range: [{feat_t_f.min():.1f}, {feat_t_f.max():.1f}]")

    # Raw correlation (before normalization)
    corr_raw = flow._global_correlation(feat_t_f, feat_t1_f).astype(np.float64)
    print(f"\n  Raw correlation (before overlap norm):")
    print_corr_grid(corr_raw, flow._search_range)

    # Normalized correlation
    corr_norm = corr_raw / flow._overlap_counts
    print(f"\n  Normalized correlation (after overlap norm):")
    print_corr_grid(corr_norm, flow._search_range)

    # Soft argmax
    vx, vy = flow._soft_argmax(corr_norm.astype(np.float32))
    print(f"\n  Result: vx={vx:.4f}, vy={vy:.4f}")

    # Also try with higher temperature
    old_temp = flow._temperature
    for temp in [0.5, 1.0, 5.0]:
        flow._temperature = temp
        vx_t, vy_t = flow._soft_argmax(corr_norm.astype(np.float32))
        print(f"  With temp={temp}: vx={vx_t:.4f}, vy={vy_t:.4f}")
    flow._temperature = old_temp

    return vx, vy


# ─── Main ─────────────────────────────────────────────────────────────
rng = np.random.RandomState(42)

# Test images
textured = rng.randint(0, 256, (64, 64), dtype=np.uint8)
smooth = np.full((64, 64), 128, dtype=np.uint8)
gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))

# Also simulate webcam-like frame: smooth with some edges
webcam_like = cv2.GaussianBlur(textured, (15, 15), 5) if False else textured

print("=" * 70)
print("DIAGNOSTIC: Pooled Optical Flow Correlation Profile")
print("=" * 70)

# ─── Experiment 1: Pooled mode, identical textured frames ─────────
with OpticalFlow.from_template(64, pooled=True) as flow:
    run_experiment(flow, "Pooled: identical textured frames", textured, textured)

# ─── Experiment 2: Pooled mode, identical smooth frames ───────────
with OpticalFlow.from_template(64, pooled=True) as flow:
    run_experiment(flow, "Pooled: identical smooth frames", smooth, smooth)

# ─── Experiment 3: Pooled mode, identical gradient frames ─────────
with OpticalFlow.from_template(64, pooled=True) as flow:
    run_experiment(flow, "Pooled: identical gradient frames", gradient, gradient)

# ─── Experiment 4: Standard mode, identical textured frames ───────
with OpticalFlow.from_template(64, pooled=False) as flow:
    run_experiment(flow, "Standard: identical textured frames", textured, textured)

# ─── Experiment 5: Pooled mode, shifted textured frames ───────────
with OpticalFlow.from_template(64, pooled=True) as flow:
    shifted = np.roll(textured, shift=4, axis=1)  # shift right by 4
    run_experiment(flow, "Pooled: textured, shift right by 4px", textured, shifted)

print("\n" + "=" * 70)
print("DONE")
