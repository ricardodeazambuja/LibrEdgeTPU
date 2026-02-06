#!/usr/bin/env python3
"""Experiment 1: Verify TileLayout exists in Gabor model DarwiNN executables.

Hypothesis: The Gabor model's DarwiNN executable contains a non-trivial
TileLayout (not None, with table lengths matching output dimensions).

No hardware needed — pure parsing of compiled .tflite files.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libredgetpu.tflite_parser import parse as parse_tflite
from libredgetpu.delegate import parse_darwinn, TYPE_STAND_ALONE, TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY

# Templates to test
TEMPLATES = {
    "standard_64": "libredgetpu/optical_flow/templates/gabor_64x64_7k_4o_2s_edgetpu.tflite",
    "pooled_64": "libredgetpu/optical_flow/templates/gabor_64x64_p4_edgetpu.tflite",
    "standard_128": "libredgetpu/optical_flow/templates/gabor_128x128_7k_4o_2s_edgetpu.tflite",
    "pooled_128": "libredgetpu/optical_flow/templates/gabor_128x128_p4_edgetpu.tflite",
}

TYPE_NAMES = {
    TYPE_STAND_ALONE: "STAND_ALONE",
    TYPE_PARAMETER_CACHING: "PARAMETER_CACHING",
    TYPE_EXECUTION_ONLY: "EXECUTION_ONLY",
}


def analyze_template(name, path):
    """Parse a template and print TileLayout information."""
    full_path = os.path.join(os.path.dirname(__file__), "..", path)
    if not os.path.exists(full_path):
        print(f"\n{'='*60}")
        print(f"SKIP: {name} — file not found: {path}")
        return

    print(f"\n{'='*60}")
    print(f"Template: {name} ({path})")
    print(f"{'='*60}")

    with open(full_path, "rb") as f:
        tflite_bytes = f.read()

    model = parse_tflite(tflite_bytes)
    print(f"  Input:  shape={model.input_tensor.shape}, scale={model.input_tensor.scale}, zp={model.input_tensor.zero_point}")
    print(f"  Output: shape={model.output_tensor.shape}, scale={model.output_tensor.scale}, zp={model.output_tensor.zero_point}")

    executables = parse_darwinn(model.custom_op_data)
    print(f"  DarwiNN executables: {len(executables)}")

    for i, exe in enumerate(executables):
        etype = TYPE_NAMES.get(exe.exec_type, f"UNKNOWN({exe.exec_type})")
        print(f"\n  Executable {i}: {etype}")
        print(f"    Input layers:  {len(exe.input_layers)}")
        print(f"    Output layers: {len(exe.output_layers)}")
        print(f"    DMA steps:     {len(exe.dma_steps)}")
        print(f"    Scratch bytes: {exe.scratch_size_bytes}")

        for j, layer in enumerate(exe.output_layers):
            print(f"\n    Output layer {j}: '{layer.name}'")
            print(f"      Dimensions: y={layer.y_dim}, x={layer.x_dim}, z={layer.z_dim}")
            expected_bytes = layer.y_dim * layer.x_dim * layer.z_dim
            print(f"      size_bytes:  {layer.size_bytes} (expected YXZ: {expected_bytes}, ratio: {layer.size_bytes / expected_bytes:.2f}x)")
            print(f"      zero_point:  {layer.zero_point}")
            print(f"      dequant:     {layer.dequant_factor}")

            tl = layer.tile_layout
            if tl is None:
                print(f"      tile_layout: *** NONE ***")
            else:
                print(f"      tile_layout: PRESENT ✓")
                tables = [
                    ("y_tile_id_map", tl.y_tile_id_map),
                    ("x_tile_id_map", tl.x_tile_id_map),
                    ("tile_byte_offsets", tl.tile_byte_offsets),
                    ("x_local_byte_offset", tl.x_local_byte_offset),
                    ("y_local_y_offset", tl.y_local_y_offset),
                    ("x_local_y_row_size", tl.x_local_y_row_size),
                ]
                for tname, tdata in tables:
                    first10 = tdata[:10]
                    print(f"        {tname}: len={len(tdata)}, first10={first10}")

                # Extra analysis: detect periodicity in y_tile_id_map
                ytids = tl.y_tile_id_map
                if len(ytids) > 1:
                    diffs = [ytids[i+1] - ytids[i] for i in range(min(len(ytids)-1, 20))]
                    print(f"        y_tile_id_map diffs (first 20): {diffs}")

                # x_tile_id_map periodicity
                xtids = tl.x_tile_id_map
                if len(xtids) > 1:
                    xdiffs = [xtids[i+1] - xtids[i] for i in range(min(len(xtids)-1, 20))]
                    print(f"        x_tile_id_map diffs (first 20): {xdiffs}")

                # Number of unique tiles
                all_tile_ids = set()
                for y in range(layer.y_dim):
                    for x in range(layer.x_dim):
                        tid = ytids[y] + xtids[x]
                        all_tile_ids.add(tid)
                print(f"        Unique tile IDs: {len(all_tile_ids)} (max tile offset index: {max(all_tile_ids)})")
                print(f"        tile_byte_offsets range: [{min(tl.tile_byte_offsets)}, {max(tl.tile_byte_offsets)}]")
                print(f"        Max byte addressed: {max(tl.tile_byte_offsets) + max(tl.y_local_y_offset) * max(tl.x_local_y_row_size) + max(tl.x_local_byte_offset) + layer.z_dim}")


if __name__ == "__main__":
    print("Experiment 1: Verify TileLayout in Gabor Model DarwiNN Executables")
    print("=" * 60)

    for name, path in TEMPLATES.items():
        analyze_template(name, path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If tile_layout is PRESENT for standard model but NONE for pooled,")
    print("this explains why pooled mode has different spatial bugs.")
    print("If PRESENT for both, relayout_output() should fix both.")
