#!/usr/bin/env python3
"""Test if the Edge TPU correctly localizes edges spatially.

If the Edge TPU puts the Gabor response at the correct spatial position
for BOTH horizontal and vertical edges, the spatial mapping is correct.
If edges are correctly located horizontally but not vertically,
we've found the bug.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.optical_flow_module import OpticalFlow


def main():
    h, w, c = 64, 64, 8

    print("Loading Edge TPU model (standard mode)...")
    flow = OpticalFlow.from_template(64, pooled=False)
    flow.open()

    # ── Test 1: Horizontal edge (should be localized in Y) ────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Horizontal Edge Localization")
    print("=" * 70)

    # Place horizontal edge at row 32
    img_h = np.zeros((h, w), dtype=np.uint8)
    img_h[:32, :] = 255

    feat_h = flow._extract_features_uint8(img_h)

    print("  Image: top half white (255), bottom half black (0)")
    print("  Edge at row 32. Ch0 (theta=0°) should peak at row ~32.\n")

    # Show row means for Ch0 (horizontal edge detector)
    for ch in [0, 2, 4, 6]:
        row_means = feat_h[:, :, ch].mean(axis=1)
        peak_row = np.argmax(row_means)
        print(f"  Ch{ch} row means (peak at row {peak_row}):")
        # Show every 4th row for brevity
        for r in range(0, h, 4):
            bar = "#" * int(row_means[r] / 10)
            print(f"    row {r:2d}: {row_means[r]:6.1f} {bar}")

    # ── Test 2: Vertical edge (should be localized in X) ──────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Vertical Edge Localization")
    print("=" * 70)

    # Place vertical edge at column 32
    img_v = np.zeros((h, w), dtype=np.uint8)
    img_v[:, :32] = 255

    feat_v = flow._extract_features_uint8(img_v)

    print("  Image: left half white (255), right half black (0)")
    print("  Edge at col 32. Ch2 (theta=90°) should peak at col ~32.\n")

    # Show column means for Ch2 (vertical edge detector)
    for ch in [0, 2, 4, 6]:
        col_means = feat_v[:, :, ch].mean(axis=0)
        peak_col = np.argmax(col_means)
        print(f"  Ch{ch} col means (peak at col {peak_col}):")
        for c in range(0, w, 4):
            bar = "#" * int(col_means[c] / 10)
            print(f"    col {c:2d}: {col_means[c]:6.1f} {bar}")

    # ── Test 3: Move the edge and check if feature peak moves ─────────────
    print("\n" + "=" * 70)
    print("TEST 3: Edge Movement Tracking")
    print("=" * 70)

    print("  Moving horizontal edge (Ch0 peak should track edge position):")
    for edge_row in [16, 24, 32, 40, 48]:
        img = np.zeros((h, w), dtype=np.uint8)
        img[:edge_row, :] = 255
        feat = flow._extract_features_uint8(img)
        row_means = feat[:, :, 0].mean(axis=1)
        peak = np.argmax(row_means)
        print(f"    Edge at row {edge_row}: Ch0 peak at row {peak} "
              f"(error={abs(peak - edge_row)} rows)")

    print()
    print("  Moving vertical edge (Ch2 peak should track edge position):")
    for edge_col in [16, 24, 32, 40, 48]:
        img = np.zeros((h, w), dtype=np.uint8)
        img[:, :edge_col] = 255
        feat = flow._extract_features_uint8(img)
        col_means = feat[:, :, 2].mean(axis=0)
        peak = np.argmax(col_means)
        print(f"    Edge at col {edge_col}: Ch2 peak at col {peak} "
              f"(error={abs(peak - edge_col)} cols)")

    # ── Test 4: Self-correlation excluding boundary ───────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: Self-Correlation Excluding 7-pixel Boundary")
    print("=" * 70)

    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    feat_base = flow._extract_features_uint8(img)

    margin = 7  # Gabor kernel size

    for shift_y, shift_x, desc in [(0, 4, "Right 4px"), (4, 0, "Down 4px"),
                                    (0, 8, "Right 8px"), (8, 0, "Down 8px")]:
        img_shifted = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
        feat_shifted = flow._extract_features_uint8(img_shifted)

        # Compare interior only (exclude boundary)
        if shift_x > 0:
            base_int = feat_base[margin:-margin, margin+shift_x:-margin, :]
            shift_int = feat_shifted[margin:-margin, margin:-margin-shift_x, :]
        elif shift_y > 0:
            base_int = feat_base[margin+shift_y:-margin, margin:-margin, :]
            shift_int = feat_shifted[margin:-margin-shift_y, margin:-margin, :]
        else:
            base_int = feat_base[margin:-margin, margin:-margin, :]
            shift_int = feat_shifted[margin:-margin, margin:-margin, :]

        per_ch = []
        for ch in range(c):
            b = base_int[:, :, ch].astype(float).ravel()
            s = shift_int[:, :, ch].astype(float).ravel()
            if b.std() < 0.01 or s.std() < 0.01:
                per_ch.append(float('nan'))
            else:
                per_ch.append(np.corrcoef(b, s)[0, 1])

        mean_c = np.nanmean(per_ch)
        corr_str = " ".join(f"{x:.3f}" for x in per_ch)
        print(f"  {desc}: mean_corr={mean_c:.3f}")
        print(f"    {corr_str}")

    # ── Test 5: Tiny shift equivariance (1 pixel) ─────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5: 1-Pixel Shift Equivariance (interior only)")
    print("=" * 70)

    for shift_y, shift_x, desc in [(0, 1, "Right 1px"), (1, 0, "Down 1px")]:
        img_shifted = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
        feat_shifted = flow._extract_features_uint8(img_shifted)

        # Interior comparison
        m = 7
        if shift_x > 0:
            base_int = feat_base[m:-m, m+shift_x:-m, :]
            shift_int = feat_shifted[m:-m, m:-m-shift_x, :]
        else:
            base_int = feat_base[m+shift_y:-m, m:-m, :]
            shift_int = feat_shifted[m:-m-shift_y, m:-m, :]

        per_ch = []
        for ch in range(c):
            b = base_int[:, :, ch].astype(float).ravel()
            s = shift_int[:, :, ch].astype(float).ravel()
            if b.std() < 0.01 or s.std() < 0.01:
                per_ch.append(float('nan'))
            else:
                per_ch.append(np.corrcoef(b, s)[0, 1])

        mean_c = np.nanmean(per_ch)
        corr_str = " ".join(f"{x:.3f}" for x in per_ch)
        print(f"  {desc}: mean_corr={mean_c:.3f}")
        print(f"    {corr_str}")

    flow.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
