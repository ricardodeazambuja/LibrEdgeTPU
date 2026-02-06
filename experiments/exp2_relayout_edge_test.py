#!/usr/bin/env python3
"""Experiment 2: Relayout Edge Localization Test.

Hypothesis: After relayout_output(), horizontal edge peak correctly tracks
edge position (not stuck at row 4).

Requires Edge TPU hardware.
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


def find_ch0_peak_row(features_3d):
    """Find the row with maximum Ch0 response (row-mean)."""
    ch0 = features_3d[:, :, 0].astype(np.float64)
    row_means = ch0.mean(axis=1)
    return np.argmax(row_means), row_means


def main():
    print("Experiment 2: Relayout Edge Localization")
    print("=" * 70)

    flow = OpticalFlow.from_template(64, pooled=False)
    flow.open()

    try:
        # Get the output layer for relayout
        if flow._cached_mode:
            output_layer = flow._eo_exe.output_layers[0]
        elif flow._sa_exe is not None:
            output_layer = flow._sa_exe.output_layers[0]
        else:
            print("ERROR: No executable found")
            return

        print(f"Output layer: y={output_layer.y_dim}, x={output_layer.x_dim}, z={output_layer.z_dim}")
        print(f"Tile layout: {'PRESENT' if output_layer.tile_layout is not None else 'NONE'}")
        print()

        h, w = 64, 64
        nf = 8
        n = h * w * nf
        shape = (h, w, nf)

        edge_rows = [8, 16, 24, 32, 40, 48, 56]

        print(f"{'Edge Row':>10} | {'Naive Peak':>12} | {'Naive Err':>10} | {'Relayout Peak':>14} | {'Relayout Err':>12}")
        print("-" * 70)

        naive_errors = []
        relayout_errors = []

        for edge_row in edge_rows:
            img = create_horizontal_edge(h, w, edge_row)

            # Normalize to float [0,1] then quantize (matching _extract_features_uint8)
            img_normalized = img.astype(np.float32) / 255.0
            quantized = flow._quantize_input(img_normalized)
            raw_output = flow._execute_raw(quantized.tobytes())

            # Method A: Naive reshape
            naive = np.frombuffer(raw_output, dtype=np.uint8)[:n].reshape(shape)
            naive_peak, _ = find_ch0_peak_row(naive)
            naive_err = abs(naive_peak - edge_row)

            # Method B: relayout_output
            relayouted = relayout_output(raw_output, output_layer)
            relayout_peak, _ = find_ch0_peak_row(relayouted)
            relayout_err = abs(relayout_peak - edge_row)

            naive_errors.append(naive_err)
            relayout_errors.append(relayout_err)

            print(f"{edge_row:>10} | {naive_peak:>12} | {naive_err:>10} | {relayout_peak:>14} | {relayout_err:>12}")

        print("-" * 70)
        print(f"{'Mean':>10} | {'':>12} | {np.mean(naive_errors):>10.1f} | {'':>14} | {np.mean(relayout_errors):>12.1f}")
        print()

        # Print detailed row profiles for one edge position
        test_edge = 32
        img = create_horizontal_edge(h, w, test_edge)
        img_normalized = img.astype(np.float32) / 255.0
        quantized = flow._quantize_input(img_normalized)
        raw_output = flow._execute_raw(quantized.tobytes())

        naive = np.frombuffer(raw_output, dtype=np.uint8)[:n].reshape(shape)
        relayouted = relayout_output(raw_output, output_layer)

        print(f"Detailed row profile for edge at row {test_edge} (Ch0 row means):")
        print(f"{'Row':>4} | {'Naive':>8} | {'Relayout':>10}")
        print("-" * 30)
        naive_ch0 = naive[:, :, 0].astype(float).mean(axis=1)
        relay_ch0 = relayouted[:, :, 0].astype(float).mean(axis=1)
        for r in range(0, h, 2):
            print(f"{r:>4} | {naive_ch0[r]:>8.1f} | {relay_ch0[r]:>10.1f}")

        # Also compare all 8 channels for the edge=32 case
        print(f"\nPer-channel peak rows for edge at row {test_edge}:")
        print(f"{'Ch':>4} | {'Naive Peak':>12} | {'Relayout Peak':>14}")
        print("-" * 35)
        for ch in range(nf):
            naive_ch = naive[:, :, ch].astype(float).mean(axis=1)
            relay_ch = relayouted[:, :, ch].astype(float).mean(axis=1)
            print(f"{ch:>4} | {np.argmax(naive_ch):>12} | {np.argmax(relay_ch):>14}")

        # Verdict
        print()
        if np.mean(relayout_errors) < 5:
            print("✓ PASS: Relayout fixes edge localization (mean error < 5)")
        else:
            print("✗ FAIL: Relayout does NOT fix edge localization")

    finally:
        flow.close()


if __name__ == "__main__":
    main()
