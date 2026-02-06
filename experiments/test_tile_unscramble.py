#!/usr/bin/env python3
"""Test tile unscrambling hypotheses.

The Edge TPU DMA appears to write output in tiles rather than NHWC raster order.
Observed patterns: period-4 in rows, period-16 in columns.

Hypothesis: tiles are 4 rows × 16 cols × 8 channels = 512 bytes per tile.
The tiles are arranged as:
  T(0,0) T(0,1) T(0,2) T(0,3)     <- tile row 0 (rows 0-3)
  T(1,0) T(1,1) T(1,2) T(1,3)     <- tile row 1 (rows 4-7)
  ...

Written in row-major tile order. Within each tile, data is NHWC.

To get the correct pixel at global position (r, c, ch):
  tr = r // 4, tc = c // 16
  lr = r % 4, lc = c % 16
  ti = tr * 4 + tc  (4 tile columns)
  byte_offset = ti * 512 + (lr * 16 + lc) * 8 + ch
"""

import numpy as np
import sys
import os
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu.optical_flow_module import OpticalFlow


def unscramble_tiles(raw_bytes, h, w, c, tile_h=4, tile_w=16):
    """Unscramble tiled DMA output to NHWC raster order.

    Assumes tiles of (tile_h, tile_w, c) are written in row-major tile order.
    Within each tile, data is in NHWC order (row-major, c innermost).
    """
    n_tile_rows = h // tile_h
    n_tile_cols = w // tile_w
    tile_size = tile_h * tile_w * c

    result = np.zeros((h, w, c), dtype=np.uint8)

    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            ti = tr * n_tile_cols + tc
            tile_start = ti * tile_size
            tile_data = raw_bytes[tile_start:tile_start + tile_size]
            tile_3d = tile_data.reshape(tile_h, tile_w, c)

            r_start = tr * tile_h
            c_start = tc * tile_w
            result[r_start:r_start + tile_h, c_start:c_start + tile_w, :] = tile_3d

    return result


def main():
    h, w, c = 64, 64, 8

    print("Loading Edge TPU model (standard mode)...")
    flow = OpticalFlow.from_template(64, pooled=False)
    flow.open()

    from libredgetpu._quantize import quantize_uint8

    def get_raw_output(img):
        """Get raw bytes from Edge TPU."""
        img_norm = img.astype(np.float32) / 255.0
        quantized = quantize_uint8(img_norm, flow._input_info.scale,
                                    flow._input_info.zero_point)
        return flow._execute_raw(quantized.tobytes())

    # Try multiple tile sizes
    tile_sizes = [
        (4, 16, "4×16"),
        (4, 8, "4×8"),
        (8, 8, "8×8"),
        (4, 4, "4×4"),
        (8, 16, "8×16"),
        (16, 4, "16×4"),
        (16, 16, "16×16"),
        (1, 64, "1×64 (row)"),
        (64, 1, "64×1 (col)"),
    ]

    np.random.seed(42)
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    raw = get_raw_output(img)
    raw_bytes = np.frombuffer(raw, dtype=np.uint8)[:h * w * c]

    # Raster (current interpretation)
    feat_raster = raw_bytes.reshape(h, w, c)

    print("\n" + "=" * 70)
    print("TESTING TILE SIZES: Self-Correlation for Right 4px and Down 4px shifts")
    print("=" * 70)

    img_right = np.roll(img, 4, axis=1)
    img_down = np.roll(img, 4, axis=0)
    raw_right = get_raw_output(img_right)
    raw_down = get_raw_output(img_down)
    raw_right_bytes = np.frombuffer(raw_right, dtype=np.uint8)[:h * w * c]
    raw_down_bytes = np.frombuffer(raw_down, dtype=np.uint8)[:h * w * c]

    m = 7  # Exclude boundary

    for tile_h_val, tile_w_val, name in tile_sizes:
        if h % tile_h_val != 0 or w % tile_w_val != 0:
            continue

        feat_base = unscramble_tiles(raw_bytes, h, w, c, tile_h_val, tile_w_val)
        feat_right = unscramble_tiles(raw_right_bytes, h, w, c, tile_h_val, tile_w_val)
        feat_down = unscramble_tiles(raw_down_bytes, h, w, c, tile_h_val, tile_w_val)

        # Right 4px self-correlation
        base_r = feat_base[m:-m, m+4:-m, :]
        shift_r = feat_right[m:-m, m:-m-4, :]
        corrs_r = []
        for ch_idx in range(c):
            b = base_r[:, :, ch_idx].astype(float).ravel()
            s = shift_r[:, :, ch_idx].astype(float).ravel()
            if b.std() < 0.01 or s.std() < 0.01:
                corrs_r.append(float('nan'))
            else:
                corrs_r.append(np.corrcoef(b, s)[0, 1])
        mean_r = np.nanmean(corrs_r)

        # Down 4px self-correlation
        base_d = feat_base[m+4:-m, m:-m, :]
        shift_d = feat_down[m:-m-4, m:-m, :]
        corrs_d = []
        for ch_idx in range(c):
            b = base_d[:, :, ch_idx].astype(float).ravel()
            s = shift_d[:, :, ch_idx].astype(float).ravel()
            if b.std() < 0.01 or s.std() < 0.01:
                corrs_d.append(float('nan'))
            else:
                corrs_d.append(np.corrcoef(b, s)[0, 1])
        mean_d = np.nanmean(corrs_d)

        marker = " ← WINNER!" if mean_r > 0.5 and mean_d > 0.5 else ""
        print(f"  Tiles {name:8s}: Right_corr={mean_r:.3f}, Down_corr={mean_d:.3f}{marker}")

    # Also test NO tile transformation (raster)
    base_r = feat_raster[m:-m, m+4:-m, :]
    shift_r = raw_right_bytes.reshape(h, w, c)[m:-m, m:-m-4, :]
    corrs_r = []
    for ch_idx in range(c):
        b = base_r[:, :, ch_idx].astype(float).ravel()
        s = shift_r[:, :, ch_idx].astype(float).ravel()
        if b.std() < 0.01 or s.std() < 0.01:
            corrs_r.append(float('nan'))
        else:
            corrs_r.append(np.corrcoef(b, s)[0, 1])
    mean_r_raster = np.nanmean(corrs_r)

    base_d = feat_raster[m+4:-m, m:-m, :]
    shift_d = raw_down_bytes.reshape(h, w, c)[m:-m-4, m:-m, :]
    corrs_d = []
    for ch_idx in range(c):
        b = base_d[:, :, ch_idx].astype(float).ravel()
        s = shift_d[:, :, ch_idx].astype(float).ravel()
        if b.std() < 0.01 or s.std() < 0.01:
            corrs_d.append(float('nan'))
        else:
            corrs_d.append(np.corrcoef(b, s)[0, 1])
    mean_d_raster = np.nanmean(corrs_d)

    print(f"  Raster  (no unscramble): Right_corr={mean_r_raster:.3f}, "
          f"Down_corr={mean_d_raster:.3f}")

    # ── Test winning tile size with edge localization ─────────────────────
    print("\n" + "=" * 70)
    print("EDGE LOCALIZATION with different tile sizes")
    print("=" * 70)

    for tile_h_val, tile_w_val, name in tile_sizes:
        if h % tile_h_val != 0 or w % tile_w_val != 0:
            continue

        # Test horizontal edge at row 32
        img_edge = np.zeros((h, w), dtype=np.uint8)
        img_edge[:32, :] = 255
        raw_edge = get_raw_output(img_edge)
        raw_edge_bytes = np.frombuffer(raw_edge, dtype=np.uint8)[:h * w * c]
        feat_edge = unscramble_tiles(raw_edge_bytes, h, w, c, tile_h_val, tile_w_val)

        row_means = feat_edge[:, :, 0].mean(axis=1)
        peak = np.argmax(row_means)
        # Also find the transition point (where value drops below half max)
        half_max = row_means.max() / 2
        transition = np.argmin(row_means > half_max)

        print(f"  Tiles {name:8s}: H-edge at 32 → Ch0 peak={peak}, transition={transition}")

    flow.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
