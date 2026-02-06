#!/usr/bin/env python3
"""Diagnostic: simulate the exact GUI processing loop.

Tests:
T1: Simulate OpticalFlowMode.process() loop with real webcam frames
T2: Simulate OpticalFlowMode.process() loop with synthetic patterns
T3: Simulate VisualCompassMode.process() loop with webcam frames
T4: Check if JPEG encode/decode in MJPEG pipeline affects results
"""

import cv2
import numpy as np
import time
from libredgetpu.gui.algorithm_modes import OpticalFlowMode, VisualCompassMode, HARDWARE_AVAILABLE
from libredgetpu.gui.camera import SyntheticCamera

print(f"HARDWARE_AVAILABLE: {HARDWARE_AVAILABLE}")

def simulate_gui_loop(mode, frame_source, n_frames, label):
    """Simulate the exact GUI get_frame() → process() loop."""
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"{'='*60}")

    mouse_state = {}
    results = []

    for i in range(n_frames):
        ret, frame = frame_source()
        if not ret:
            print(f"  Frame {i}: read failed")
            continue

        result = mode.process(frame, mouse_state)

        # Extract vx, vy from mode state
        if hasattr(mode, 'last_flow'):
            vx, vy = mode.last_flow
            results.append((vx, vy))
            if i > 0:  # Skip init frame
                print(f"  Frame {i:3d}: vx={vx:+.3f} vy={vy:+.3f} latency={mode.last_latency_ms:.1f}ms")
        elif hasattr(mode, 'cumulative_yaw'):
            yaw = mode.cumulative_yaw
            results.append(yaw)
            if i > 0:
                print(f"  Frame {i:3d}: yaw={yaw:+.3f}° latency={mode.last_latency_ms:.1f}ms")

    if results and hasattr(mode, 'last_flow'):
        vxs = [r[0] for r in results]
        vys = [r[1] for r in results]
        print(f"\n  Summary: vx range=[{min(vxs):+.2f}, {max(vxs):+.2f}] "
              f"mean={np.mean(vxs):+.3f} std={np.std(vxs):.3f}")
        print(f"           vy range=[{min(vys):+.2f}, {max(vys):+.2f}] "
              f"mean={np.mean(vys):+.3f} std={np.std(vys):.3f}")

    return results


# ─── T1: Real webcam + hardware OpticalFlow ───────────────────────
print("\n" + "#" * 60)
print("T1: REAL WEBCAM + HARDWARE OPTICAL FLOW")
print("#" * 60)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam")

    # Warmup (same as GUI)
    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    mode = OpticalFlowMode(synthetic=False)
    simulate_gui_loop(
        mode,
        lambda: cap.read(),
        20,
        "Real webcam + hardware optical flow"
    )
    mode.cleanup()
    cap.release()
except Exception as e:
    print(f"  Skipped: {e}")


# ─── T2: Synthetic patterns + hardware OpticalFlow ────────────────
print("\n" + "#" * 60)
print("T2: SYNTHETIC VIDEO + HARDWARE OPTICAL FLOW")
print("#" * 60)

for pattern in ["wandering_dot", "panning", "checkerboard", "noise"]:
    cam = SyntheticCamera(640, 480, pattern=pattern)
    mode = OpticalFlowMode(synthetic=False)
    simulate_gui_loop(
        mode,
        lambda: cam.read(),
        15,
        f"Synthetic '{pattern}' + hardware"
    )
    mode.cleanup()


# ─── T3: Real webcam + hardware VisualCompass ─────────────────────
print("\n" + "#" * 60)
print("T3: REAL WEBCAM + HARDWARE VISUAL COMPASS")
print("#" * 60)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam")

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    mode = VisualCompassMode(synthetic=False)
    simulate_gui_loop(
        mode,
        lambda: cap.read(),
        20,
        "Real webcam + hardware visual compass"
    )
    mode.cleanup()
    cap.release()
except Exception as e:
    print(f"  Skipped: {e}")


# ─── T4: JPEG round-trip test ─────────────────────────────────────
print("\n" + "#" * 60)
print("T4: JPEG ROUND-TRIP — does MJPEG encoding affect flow?")
print("#" * 60)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam")

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    from libredgetpu.optical_flow_module import OpticalFlow

    with OpticalFlow.from_template(64, pooled=True) as flow:
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Direct (no JPEG)
        prev_r = cv2.resize(gray1, (64, 64))
        curr_r = cv2.resize(gray2, (64, 64))
        vx_direct, vy_direct = flow.compute(prev_r, curr_r)

        # Via JPEG encode/decode (like MJPEG)
        _, buf1 = cv2.imencode('.jpg', frame1, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf2 = cv2.imencode('.jpg', frame2, [cv2.IMWRITE_JPEG_QUALITY, 85])
        dec1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)
        dec2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)
        gray1_j = cv2.cvtColor(dec1, cv2.COLOR_BGR2GRAY)
        gray2_j = cv2.cvtColor(dec2, cv2.COLOR_BGR2GRAY)
        prev_j = cv2.resize(gray1_j, (64, 64))
        curr_j = cv2.resize(gray2_j, (64, 64))
        vx_jpeg, vy_jpeg = flow.compute(prev_j, curr_j)

        print(f"  Direct:     vx={vx_direct:+.3f} vy={vy_direct:+.3f}")
        print(f"  Via JPEG85: vx={vx_jpeg:+.3f} vy={vy_jpeg:+.3f}")
        diff_max = max(abs(prev_r.astype(int) - prev_j.astype(int)).max(),
                      abs(curr_r.astype(int) - curr_j.astype(int)).max())
        print(f"  Max pixel diff from JPEG: {diff_max}")

    cap.release()
except Exception as e:
    print(f"  Skipped: {e}")

print("\n" + "=" * 60)
print("DONE")
