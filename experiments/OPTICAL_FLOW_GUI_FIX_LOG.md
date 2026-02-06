# Optical Flow GUI ±3 Oscillation — Investigation Log

## Date: Feb 8, 2026

## Symptom
GUI OpticalFlow mode shows oscillating ±3 on vx, VisualCompass noisy/non-functional.
Appeared after commit e72c354 (removed silent CPU fallback).

## Investigation (Scientific Method)

### Hypothesis 1: Hardware/relayout broken for pooled mode
**Test**: Identical frames on hardware with pooled=True
**Result**: Peak at (0,0), vx=0.000 — REJECTED

### Hypothesis 2: Noise sensitivity
**Test**: Added noise_std=0..50 to identical frames
**Result**: Peak stays at (0,0) even at noise_std=50 — REJECTED

### Hypothesis 3: Webcam frames cause issues
**Test**: Captured 12 real webcam frames, ran through full pipeline
**Result**: All vx=0.000 for static camera — REJECTED

### Hypothesis 4: GUI processing loop has race condition
**Test**: Simulated exact GUI loop (OpticalFlowMode.process()) with webcam
**Result**: All vx=0.000 — REJECTED

### Hypothesis 5: Synthetic panning pattern causes aliasing
**Test**: Simulated GUI loop with synthetic patterns
**Result**: wandering_dot=0, panning=±3, checkerboard=0, noise=0 — **CONFIRMED**

### Root Cause Analysis

**TWO independent issues:**

1. **`cv2.resize` default interpolation (`INTER_LINEAR`) aliases during large downscale.**
   - GUI resizes 640×480 → 64×64 (10× downscale) using bilinear interpolation
   - Bilinear interpolation samples only nearby pixels, skipping most of the input
   - High-frequency content (narrow stripes) creates moiré artifacts
   - Small physical shifts (4px in 640-space) appear as large jumps (±3 pooled px)
   - Fix: use `cv2.INTER_AREA` which properly averages all source pixels

2. **Panning pattern has stripes below Nyquist frequency of downsampled output.**
   - Stripes 8-32px wide at 640 resolution = 0.8-3.2px at 64×64 = sub-Nyquist
   - Even with INTER_AREA, stripes near the Nyquist limit can alias
   - Fix: use wider stripes (40-120px) that resolve to 4-12px at 64×64

### Verification

After both fixes:
- Panning + OpticalFlow: 15 frames, all vx=0.000 vy=0.000
- Panning + VisualCompass: 10 frames, all yaw=0.000°
- All other patterns: all zeros
- Real webcam: all zeros
- Full test suite: 403/403 pass

## Files Changed

| File | Change |
|------|--------|
| `libredgetpu/gui/algorithm_modes.py` | `interpolation=cv2.INTER_AREA` on all 8 resize calls |
| `libredgetpu/gui/camera.py` | Panning pattern: wider stripes (40-120px), varied intensity, thicker bars |
