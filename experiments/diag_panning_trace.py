#!/usr/bin/env python3
"""Trace exactly what happens frame-by-frame in the panning GUI loop."""

import cv2
import numpy as np
import time
from libredgetpu.gui.camera import SyntheticCamera
from libredgetpu.optical_flow_module import OpticalFlow

print("=" * 70)
print("TRACE: Panning pattern frame-by-frame through OpticalFlow")
print("=" * 70)

# Create camera and hardware (mimicking OpticalFlowMode.__init__)
cam = SyntheticCamera(640, 480, pattern="panning")
print(f"Camera created at t={time.time():.3f}")

flow = OpticalFlow.from_template(64, pooled=True)
flow.open()
print(f"Hardware opened at t={time.time():.3f}")

prev_gray = None
for i in range(12):
    t_before = time.time()
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t_read = time.time()

    if prev_gray is None:
        prev_gray = gray.copy()
        print(f"\nFrame {i}: INIT (stored as prev_gray, {gray.shape})")
        continue

    # Exactly what OpticalFlowMode.process() does:
    prev_resized = cv2.resize(prev_gray, (64, 64))
    curr_resized = cv2.resize(gray, (64, 64))

    # Check frame similarity
    diff_full = np.abs(gray.astype(int) - prev_gray.astype(int))
    diff_64 = np.abs(curr_resized.astype(int) - prev_resized.astype(int))

    # Run compute
    vx, vy = flow.compute(prev_resized, curr_resized)
    t_done = time.time()

    # Also check: what if we directly run correlation without GUI resize?
    # (to compare)
    feat_prev = flow._extract_features_uint8(prev_resized)
    feat_curr = flow._extract_features_uint8(curr_resized)
    zp = np.int16(flow._output_info.zero_point)
    ft = (feat_prev.astype(np.int16) - zp).astype(np.float32)
    f1 = (feat_curr.astype(np.int16) - zp).astype(np.float32)
    corr_raw = flow._global_correlation(ft, f1).astype(np.float64)
    corr_norm = corr_raw / flow._overlap_counts
    sr = flow._search_range
    side = 2 * sr + 1
    grid = corr_norm.reshape(side, side)
    center_row = grid[sr, :]
    peak_dx = np.argmax(center_row) - sr
    peak_val = center_row.max()
    second_val = np.sort(center_row)[-2]
    gap = peak_val - second_val

    print(f"\nFrame {i}: vx={vx:+.3f} vy={vy:+.3f}  "
          f"peak_dx={peak_dx:+d} gap={gap:.1f}  "
          f"diff_64 mean={diff_64.mean():.1f} max={diff_64.max()}")

    # Print center row
    cr_str = " ".join(f"{v:.0f}" for v in center_row)
    print(f"  dy=0 corr: {cr_str}")

    prev_gray = gray.copy()

flow.close()

# ─── Now test: is the issue with the SyntheticCamera timing? ──────
print("\n\n" + "=" * 70)
print("TEST: Read all frames FIRST, then process")
print("=" * 70)

cam2 = SyntheticCamera(640, 480, pattern="panning")
frames = []
for i in range(8):
    _, frame = cam2.read()
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

flow2 = OpticalFlow.from_template(64, pooled=True)
flow2.open()

for i in range(1, len(frames)):
    prev_r = cv2.resize(frames[i-1], (64, 64))
    curr_r = cv2.resize(frames[i], (64, 64))
    vx, vy = flow2.compute(prev_r, curr_r)
    diff = np.abs(curr_r.astype(int) - prev_r.astype(int))
    print(f"  Frame {i-1}→{i}: vx={vx:+.3f} vy={vy:+.3f}  diff_64 mean={diff.mean():.1f}")

flow2.close()

# ─── Test: panning with larger gap between reads ──────────────────
print("\n\n" + "=" * 70)
print("TEST: Panning with time.sleep(0.5) between reads")
print("=" * 70)

cam3 = SyntheticCamera(640, 480, pattern="panning")
flow3 = OpticalFlow.from_template(64, pooled=True)
flow3.open()

prev_gray3 = None
for i in range(6):
    time.sleep(0.5)  # Simulate slow processing
    _, frame = cam3.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray3 is None:
        prev_gray3 = gray.copy()
        print(f"  Frame {i}: INIT")
        continue

    prev_r = cv2.resize(prev_gray3, (64, 64))
    curr_r = cv2.resize(gray, (64, 64))
    diff = np.abs(curr_r.astype(int) - prev_r.astype(int))
    vx, vy = flow3.compute(prev_r, curr_r)
    print(f"  Frame {i}: vx={vx:+.3f} vy={vy:+.3f}  diff_64 mean={diff.mean():.1f}")

    prev_gray3 = gray.copy()

flow3.close()

print("\n" + "=" * 70)
print("DONE")
