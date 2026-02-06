#!/usr/bin/env python3
"""Diagnostic: why does the synthetic panning pattern produce ±3 oscillation?

Investigate the panning pattern's feature structure and correlation profile.
"""

import cv2
import numpy as np
import time
from libredgetpu.gui.camera import SyntheticCamera
from libredgetpu.optical_flow_module import OpticalFlow

cam = SyntheticCamera(640, 480, pattern="panning")

# Read consecutive frames
frames = []
for i in range(5):
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

print("=" * 70)
print("PANNING PATTERN ANALYSIS")
print("=" * 70)

# Check frame differences
for i in range(1, len(frames)):
    diff = np.abs(frames[i].astype(int) - frames[i-1].astype(int))
    resized_prev = cv2.resize(frames[i-1], (64, 64))
    resized_curr = cv2.resize(frames[i], (64, 64))
    diff_r = np.abs(resized_curr.astype(int) - resized_prev.astype(int))
    print(f"\nFrame {i-1}→{i}:")
    print(f"  640x480 diff: mean={diff.mean():.1f} max={diff.max()} "
          f"non-zero pixels: {np.count_nonzero(diff)}/{diff.size}")
    print(f"  64x64  diff: mean={diff_r.mean():.1f} max={diff_r.max()} "
          f"non-zero pixels: {np.count_nonzero(diff_r)}/{diff_r.size}")

# Run correlation analysis on frame pair 0→1
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS: Frame 0→1")
print("=" * 70)

with OpticalFlow.from_template(64, pooled=True) as flow:
    prev = cv2.resize(frames[0], (64, 64))
    curr = cv2.resize(frames[1], (64, 64))

    # Extract features
    feat_prev = flow._extract_features_uint8(prev)
    feat_curr = flow._extract_features_uint8(curr)

    print(f"\nFeatures prev: shape={feat_prev.shape} range=[{feat_prev.min()},{feat_prev.max()}] mean={feat_prev.mean():.1f}")
    print(f"Features curr: shape={feat_curr.shape} range=[{feat_curr.min()},{feat_curr.max()}] mean={feat_curr.mean():.1f}")

    feat_diff = np.abs(feat_prev.astype(int) - feat_curr.astype(int))
    print(f"Feature diff: mean={feat_diff.mean():.1f} max={feat_diff.max()}")

    # Compute correlation
    zp = np.int16(flow._output_info.zero_point)
    ft = (feat_prev.astype(np.int16) - zp).astype(np.float32)
    f1 = (feat_curr.astype(np.int16) - zp).astype(np.float32)

    corr_raw = flow._global_correlation(ft, f1).astype(np.float64)
    corr_norm = corr_raw / flow._overlap_counts

    sr = flow._search_range
    side = 2 * sr + 1
    grid = corr_norm.reshape(side, side)

    # Print center row (dy=0)
    print(f"\nCorrelation dy=0 row (normalized):")
    for i, dx in enumerate(range(-sr, sr+1)):
        v = grid[sr, i]
        peak = " ←PEAK" if v == grid[sr, :].max() else ""
        print(f"  dx={dx:+d}: {v:.2f}{peak}")

    # Check: how flat is the correlation?
    center_row = grid[sr, :]
    row_max = center_row.max()
    row_min = center_row.min()
    row_range = row_max - row_min
    print(f"\n  dy=0 row: max={row_max:.2f} min={row_min:.2f} range={row_range:.2f}")
    print(f"  Full grid: max={grid.max():.2f} min={grid.min():.2f} range={grid.max()-grid.min():.2f}")

    # Check feature spatial structure (autocorrelation of features)
    print(f"\nFeature spatial autocorrelation (column-wise, dy=0):")
    ph, pw, nf = ft.shape
    for dx in range(-4, 5):
        if dx >= 0:
            overlap = ft[:, :pw-dx, :].ravel() @ ft[:, dx:, :].ravel()
            n_elem = ph * (pw - dx) * nf
        else:
            adx = -dx
            overlap = ft[:, adx:, :].ravel() @ ft[:, :pw-adx, :].ravel()
            n_elem = ph * (pw - adx) * nf
        print(f"  dx={dx:+d}: autocorr/elem = {overlap/n_elem:.2f}")

    # Check: does the feature map have periodicity?
    print(f"\nFeature map column means (first channel):")
    col_means = ft[:, :, 0].mean(axis=0)
    print(f"  {' '.join(f'{v:.0f}' for v in col_means)}")

    # Soft argmax at different temperatures
    print(f"\nSoft argmax results at different temperatures:")
    for temp in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        old_temp = flow._temperature
        flow._temperature = temp
        vx, vy = flow._soft_argmax(corr_norm.astype(np.float32))
        flow._temperature = old_temp
        print(f"  temp={temp:6.2f}: vx={vx:+.3f} vy={vy:+.3f}")

print("\n" + "=" * 70)
print("DONE")
