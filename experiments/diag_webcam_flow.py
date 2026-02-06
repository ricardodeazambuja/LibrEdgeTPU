#!/usr/bin/env python3
"""Diagnostic: simulate webcam-like conditions and capture real webcam frames.

Experiments:
E1: Add varying noise levels to identical frames → find threshold where peak moves
E2: Small sub-pixel shifts (simulating camera vibration) → check correlation profile
E3: Capture actual webcam frames → run optical flow on consecutive frames
E4: Compare pooled vs standard mode on webcam frames
"""

import numpy as np
import time
import sys
from libredgetpu.optical_flow_module import OpticalFlow

def corr_peak(flow, img_t, img_t1):
    """Run correlation and return (vx, vy, peak_dx, peak_dy, corr_grid)."""
    feat_t = flow._extract_features_uint8(img_t)
    feat_t1 = flow._extract_features_uint8(img_t1)
    zp = np.int16(flow._output_info.zero_point)
    ft = feat_t.astype(np.int16) - zp
    f1 = feat_t1.astype(np.int16) - zp
    if flow._fused_pool:
        ft = ft.astype(np.float32)
        f1 = f1.astype(np.float32)
    else:
        ft = flow._pool_features_int(ft).astype(np.float32)
        f1 = flow._pool_features_int(f1).astype(np.float32)
    corr_raw = flow._global_correlation(ft, f1).astype(np.float64)
    corr_norm = corr_raw / flow._overlap_counts
    sr = flow._search_range
    side = 2 * sr + 1
    grid = corr_norm.reshape(side, side)
    peak_idx = np.argmax(corr_norm)
    peak_dy = peak_idx // side - sr
    peak_dx = peak_idx % side - sr
    vx, vy = flow._soft_argmax(corr_norm.astype(np.float32))
    return vx, vy, peak_dx, peak_dy, grid


rng = np.random.RandomState(42)
textured = rng.randint(0, 256, (64, 64), dtype=np.uint8)

print("=" * 70)
print("E1: NOISE SENSITIVITY — adding noise to identical frames")
print("=" * 70)

with OpticalFlow.from_template(64, pooled=True) as flow:
    for noise_std in [0, 1, 2, 5, 10, 20, 30, 50]:
        results = []
        for trial in range(10):
            noisy = np.clip(
                textured.astype(np.int16) + rng.normal(0, noise_std, textured.shape).astype(np.int16),
                0, 255
            ).astype(np.uint8)
            vx, vy, pdx, pdy, _ = corr_peak(flow, textured, noisy)
            results.append((vx, vy, pdx, pdy))
        vxs = [r[0] for r in results]
        vys = [r[1] for r in results]
        pdxs = [r[2] for r in results]
        print(f"  noise_std={noise_std:3d}: vx mean={np.mean(vxs):+.2f} std={np.std(vxs):.2f}  "
              f"peak_dx values: {sorted(set(pdxs))}")

print("\n" + "=" * 70)
print("E2: SUB-PIXEL SHIFTS — simulating camera vibration via resize")
print("=" * 70)

with OpticalFlow.from_template(64, pooled=True) as flow:
    # Create a 640×480 source, resize to 64×64 with sub-pixel shift
    big_img = rng.randint(0, 256, (480, 640), dtype=np.uint8)
    import cv2
    frame_t = cv2.resize(big_img, (64, 64))

    for shift_x in [-5, -2, -1, 0, 1, 2, 5, 10, 20]:
        # Simulate camera shift by translating the big image
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        shifted_big = cv2.warpAffine(big_img, M, (640, 480))
        frame_t1 = cv2.resize(shifted_big, (64, 64))

        vx, vy, pdx, pdy, grid = corr_peak(flow, frame_t, frame_t1)
        # Show the center row of correlation
        sr = flow._search_range
        center_row = grid[sr, :]  # dy=0 row
        print(f"  shift_x={shift_x:+3d}px(640): vx={vx:+.2f} peak_dx={pdx:+d}  "
              f"corr dy=0: {' '.join(f'{v:.0f}' for v in center_row)}")

print("\n" + "=" * 70)
print("E3: REAL WEBCAM — capture consecutive frames and test")
print("=" * 70)

try:
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam")

    # Warmup
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    frames = []
    for i in range(12):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        time.sleep(0.033)  # ~30fps

    cap.release()
    print(f"  Captured {len(frames)} frames, shape: {frames[0].shape}")

    # Test with pooled mode
    with OpticalFlow.from_template(64, pooled=True) as flow:
        print("\n  POOLED MODE:")
        for i in range(1, min(len(frames), 11)):
            prev = cv2.resize(frames[i-1], (64, 64))
            curr = cv2.resize(frames[i], (64, 64))

            # Check frame difference
            diff = np.abs(prev.astype(int) - curr.astype(int))

            vx, vy, pdx, pdy, grid = corr_peak(flow, prev, curr)

            # Find top-3 peaks in grid
            sr = flow._search_range
            side = 2 * sr + 1
            flat = grid.ravel()
            top3_idx = np.argsort(flat)[-3:][::-1]
            top3 = [(idx % side - sr, idx // side - sr, flat[idx]) for idx in top3_idx]

            print(f"  Frame {i-1}→{i}: vx={vx:+.2f} vy={vy:+.2f} peak=({pdx:+d},{pdy:+d})  "
                  f"frame_diff: mean={diff.mean():.1f} max={diff.max()}  "
                  f"top3: {[(dx,dy,f'{v:.0f}') for dx,dy,v in top3]}")

    # Test with standard mode too
    with OpticalFlow.from_template(64, pooled=False) as flow:
        print("\n  STANDARD MODE:")
        for i in range(1, min(len(frames), 6)):
            prev = cv2.resize(frames[i-1], (64, 64))
            curr = cv2.resize(frames[i], (64, 64))
            vx, vy, pdx, pdy, grid = corr_peak(flow, prev, curr)
            print(f"  Frame {i-1}→{i}: vx={vx:+.2f} vy={vy:+.2f} peak=({pdx:+d},{pdy:+d})")

except Exception as e:
    print(f"  Webcam not available: {e}")
    print("  Skipping webcam test")

print("\n" + "=" * 70)
print("DONE")
