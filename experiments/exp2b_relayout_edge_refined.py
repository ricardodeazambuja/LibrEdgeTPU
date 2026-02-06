#!/usr/bin/env python3
"""Experiment 2b: Refined edge localization with relayout.

Instead of argmax (misleading for constant backgrounds), use gradient-based
edge detection: find row with max |gradient| of Ch0 row means.
Also compare spatial structure overall.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from libredgetpu.optical_flow_module import OpticalFlow
from libredgetpu.delegate import relayout_output


def create_horizontal_edge(h, w, edge_row):
    """Create image with horizontal edge: top=255, bottom=0 at edge_row."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:edge_row, :] = 255
    return img


def find_edge_transition(row_means):
    """Find row where gradient is largest (edge transition)."""
    grad = np.abs(np.diff(row_means))
    return np.argmax(grad)


def main():
    print("Experiment 2b: Refined Relayout Edge Localization")
    print("=" * 75)

    flow = OpticalFlow.from_template(64, pooled=False)
    flow.open()

    try:
        if flow._cached_mode:
            output_layer = flow._eo_exe.output_layers[0]
        else:
            output_layer = flow._sa_exe.output_layers[0]

        h, w, nf = 64, 64, 8
        n = h * w * nf
        shape = (h, w, nf)

        edge_rows = [8, 16, 24, 32, 40, 48, 56]

        # Test 1: Edge transition detection (gradient-based)
        print("\nTest 1: Edge transition detection (max |gradient|)")
        print(f"{'Edge Row':>10} | {'Naive Trans':>12} | {'Naive Err':>10} | {'Relayout Trans':>15} | {'Relayout Err':>12}")
        print("-" * 75)

        naive_errors = []
        relayout_errors = []

        for edge_row in edge_rows:
            img = create_horizontal_edge(h, w, edge_row)
            img_normalized = img.astype(np.float32) / 255.0
            quantized = flow._quantize_input(img_normalized)
            raw_output = flow._execute_raw(quantized.tobytes())

            naive = np.frombuffer(raw_output, dtype=np.uint8)[:n].reshape(shape)
            relayouted = relayout_output(raw_output, output_layer)

            # Use gradient-based edge detection across ALL channels
            naive_all = naive.astype(float).mean(axis=(1, 2))  # mean across x and channels
            relay_all = relayouted.astype(float).mean(axis=(1, 2))

            naive_trans = find_edge_transition(naive_all)
            relay_trans = find_edge_transition(relay_all)

            naive_err = abs(naive_trans - edge_row)
            relay_err = abs(relay_trans - edge_row)

            naive_errors.append(naive_err)
            relayout_errors.append(relay_err)

            print(f"{edge_row:>10} | {naive_trans:>12} | {naive_err:>10} | {relay_trans:>15} | {relay_err:>12}")

        print("-" * 75)
        print(f"{'Mean':>10} | {'':>12} | {np.mean(naive_errors):>10.1f} | {'':>15} | {np.mean(relayout_errors):>12.1f}")

        # Test 2: Self-correlation (shift equivariance)
        print("\n\nTest 2: Shift Equivariance (spatial correlation)")
        print("Input: random texture, test if shifted input → shifted features")

        np.random.seed(42)
        texture = np.random.randint(0, 256, (h, w), dtype=np.uint8)

        def get_features(img):
            img_normalized = img.astype(np.float32) / 255.0
            quantized = flow._quantize_input(img_normalized)
            raw = flow._execute_raw(quantized.tobytes())
            naive = np.frombuffer(raw, dtype=np.uint8)[:n].reshape(shape)
            relayouted = relayout_output(raw, output_layer)
            return naive, relayouted

        base_naive, base_relay = get_features(texture)

        shifts = [(4, 0), (0, 4), (8, 0), (0, 8)]
        print(f"\n{'Shift (dx,dy)':>15} | {'Naive Corr':>12} | {'Relayout Corr':>14}")
        print("-" * 50)

        for dx, dy in shifts:
            shifted = np.roll(texture, shift=(dy, dx), axis=(0, 1))
            shift_naive, shift_relay = get_features(shifted)

            # Compute correlation in overlapping interior (exclude border)
            border = 7  # kernel size
            s = max(abs(dx), abs(dy)) + border

            # For naive
            if dy >= 0 and dx >= 0:
                base_crop_n = base_naive[s:h-s, s:w-s]
                shift_crop_n = shift_naive[s+dy:h-s+dy, s+dx:w-s+dx]
            elif dy >= 0 and dx < 0:
                base_crop_n = base_naive[s:h-s, s:w-s]
                shift_crop_n = shift_naive[s+dy:h-s+dy, s+dx:w-s+dx]
            else:
                base_crop_n = base_naive[s:h-s, s:w-s]
                shift_crop_n = shift_naive[s+dy:h-s+dy, s+dx:w-s+dx]

            # For relayout
            base_crop_r = base_relay[s:h-s, s:w-s]
            shift_crop_r = shift_relay[s+dy:h-s+dy, s+dx:w-s+dx]

            def pearson(a, b):
                a_flat = a.ravel().astype(float)
                b_flat = b.ravel().astype(float)
                if len(a_flat) == 0 or a_flat.std() < 1e-10 or b_flat.std() < 1e-10:
                    return 0.0
                return np.corrcoef(a_flat, b_flat)[0, 1]

            corr_naive = pearson(base_crop_n, shift_crop_n)
            corr_relay = pearson(base_crop_r, shift_crop_r)

            print(f"({dx:+d},{dy:+d}){'':>8} | {corr_naive:>12.4f} | {corr_relay:>14.4f}")

        # Test 3: Compare a single feature map visually (Ch0)
        print("\n\nTest 3: Feature map spatial structure (Ch0, random texture)")
        test_naive, test_relay = get_features(texture)

        # Check if relayout produces spatially coherent features
        # By looking at spatial autocorrelation at lag 1
        def spatial_autocorr_lag1(feat_2d):
            """Spatial autocorrelation at lag-1 in both directions."""
            h_corr = np.corrcoef(feat_2d[:, :-1].ravel(), feat_2d[:, 1:].ravel())[0, 1]
            v_corr = np.corrcoef(feat_2d[:-1, :].ravel(), feat_2d[1:, :].ravel())[0, 1]
            return h_corr, v_corr

        print(f"{'Channel':>8} | {'Naive H-AC':>12} | {'Naive V-AC':>12} | {'Relay H-AC':>12} | {'Relay V-AC':>12}")
        print("-" * 65)

        for ch in range(nf):
            nh, nv = spatial_autocorr_lag1(test_naive[:, :, ch].astype(float))
            rh, rv = spatial_autocorr_lag1(test_relay[:, :, ch].astype(float))
            print(f"{ch:>8} | {nh:>12.4f} | {nv:>12.4f} | {rh:>12.4f} | {rv:>12.4f}")

        # Summary
        print("\n" + "=" * 75)
        relay_mean_err = np.mean(relayout_errors)
        if relay_mean_err < 5:
            print(f"✓ PASS: Relayout edge error = {relay_mean_err:.1f} (< 5)")
        else:
            print(f"INFO: Relayout edge error = {relay_mean_err:.1f}")
            print("Note: Gabor features are bandpass — they DON'T have a peak at the edge")
            print("      position for simple step edges. The transition detection may be")
            print("      better than argmax but still imperfect.")

    finally:
        flow.close()


if __name__ == "__main__":
    main()
