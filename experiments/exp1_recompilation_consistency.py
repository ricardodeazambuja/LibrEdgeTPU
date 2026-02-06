"""Experiment 1: Recompilation consistency.

Question: When we compile the same Dense(N) architecture with different weight
values, do we get the same DarwiNN instructions?

Method:
  1. Create 4 Dense(256) models with different weight initializations
  2. Quantize each to int8 TFLite
  3. Compile each with edgetpu_compiler
  4. Extract DarwiNN executables
  5. Compare instruction bitstreams, param sizes, quant scales

Expected if H1 is true: Instructions are identical, only params differ.
"""

import os
import sys
import tempfile
import subprocess
import hashlib

import numpy as np

# Add project root to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from libredgetpu.tflite_parser import parse as parse_tflite, parse_full
from libredgetpu.delegate import parse_darwinn, TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY

N = 256
VARIANTS = {
    "zeros": lambda n: np.zeros((n, n), dtype=np.float32),
    "identity": lambda n: np.eye(n, dtype=np.float32),
    "uniform_wide": lambda n: np.random.default_rng(42).uniform(-1.0, 1.0, (n, n)).astype(np.float32),
    "uniform_narrow": lambda n: np.random.default_rng(99).uniform(-0.01, 0.01, (n, n)).astype(np.float32),
}


def create_quantized_tflite(n, weights, tmpdir, name):
    """Create a Dense(n) TFLite model with specific weights."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=n, input_shape=[n], use_bias=False),
    ])
    # Set specific weights
    model.layers[0].set_weights([weights])

    def representative_dataset():
        rng_rd = np.random.default_rng(0)
        for _ in range(256):
            data = rng_rd.uniform(-1.0, 1.0, [1, n])
            yield [data.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_bytes = converter.convert()

    path = os.path.join(tmpdir, f"{name}.tflite")
    with open(path, "wb") as f:
        f.write(tflite_bytes)

    # Extract quant params
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # Find weight tensor
    w_detail = None
    for td in interp.get_tensor_details():
        if td["shape"].tolist() == [n, n] and td["index"] not in (inp["index"], out["index"]):
            w_detail = td
            break

    quant = {
        "input_scale": float(inp["quantization_parameters"]["scales"][0]),
        "input_zp": int(inp["quantization_parameters"]["zero_points"][0]),
        "output_scale": float(out["quantization_parameters"]["scales"][0]),
        "output_zp": int(out["quantization_parameters"]["zero_points"][0]),
    }
    if w_detail:
        quant["weight_scale"] = float(w_detail["quantization_parameters"]["scales"][0])
        quant["weight_zp"] = int(w_detail["quantization_parameters"]["zero_points"][0])

    return path, tflite_bytes, quant


def compile_model(tflite_path, output_dir):
    """Compile with edgetpu_compiler."""
    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", output_dir, tflite_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"COMPILATION FAILED: {result.stderr[:500]}")
        return None
    base = os.path.splitext(os.path.basename(tflite_path))[0]
    compiled = os.path.join(output_dir, f"{base}_edgetpu.tflite")
    return compiled if os.path.isfile(compiled) else None


def extract_darwinn(compiled_path):
    """Extract DarwiNN executables from compiled model."""
    with open(compiled_path, "rb") as f:
        data = f.read()
    model = parse_tflite(data)
    return parse_darwinn(model.custom_op_data)


def sha256(data):
    return hashlib.sha256(data).hexdigest()[:16]


def main():
    print(f"=== Experiment 1: Recompilation Consistency (Dense {N}×{N}) ===\n")

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, weight_fn in VARIANTS.items():
            print(f"--- Variant: {name} ---")
            weights = weight_fn(N)
            print(f"  Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, "
                  f"mean={weights.mean():.4f}, std={weights.std():.4f}")

            # Create and quantize
            tflite_path, tflite_bytes, quant = create_quantized_tflite(
                N, weights, tmpdir, name
            )
            print(f"  Quant: input_scale={quant['input_scale']:.6f}, "
                  f"weight_scale={quant.get('weight_scale', '?')}, "
                  f"output_scale={quant['output_scale']:.6f}")

            # Compile
            out_dir = os.path.join(tmpdir, f"compiled_{name}")
            os.makedirs(out_dir)
            compiled = compile_model(tflite_path, out_dir)
            if compiled is None:
                print(f"  SKIPPED (compilation failed)")
                continue

            # Extract DarwiNN
            exes = extract_darwinn(compiled)
            info = {"quant": quant, "executables": []}

            for exe in exes:
                type_name = {0: "PARAMETER_CACHING", 1: "EXECUTION_ONLY", 2: "STAND_ALONE"}.get(
                    exe.exec_type, f"UNKNOWN({exe.exec_type})")
                exe_info = {
                    "type": type_name,
                    "n_bitstreams": len(exe.bitstreams),
                    "instr_hashes": [sha256(bs.data) for bs in exe.bitstreams],
                    "instr_sizes": [len(bs.data) for bs in exe.bitstreams],
                    "param_size": len(exe.parameters) if exe.parameters else 0,
                    "param_hash": sha256(exe.parameters) if exe.parameters else None,
                }
                info["executables"].append(exe_info)
                print(f"  {type_name}:")
                print(f"    Instructions: {exe_info['n_bitstreams']} bitstream(s), "
                      f"sizes={exe_info['instr_sizes']}, hashes={exe_info['instr_hashes']}")
                print(f"    Parameters: {exe_info['param_size']} bytes, hash={exe_info['param_hash']}")

            results[name] = info
            print()

    # === Analysis ===
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    names = list(results.keys())
    if len(names) < 2:
        print("Not enough variants to compare")
        return

    # Compare quant scales
    print("\n1. QUANTIZATION SCALES:")
    print(f"  {'Variant':<20} {'input_scale':<14} {'weight_scale':<14} {'output_scale':<14}")
    for name, info in results.items():
        q = info["quant"]
        print(f"  {name:<20} {q['input_scale']:<14.6f} {q.get('weight_scale', '?'):<14} {q['output_scale']:<14.6f}")

    # Compare instruction hashes
    print("\n2. INSTRUCTION BITSTREAM COMPARISON:")
    ref_name = names[0]
    ref = results[ref_name]
    for name in names[1:]:
        other = results[name]
        for i, (ref_exe, other_exe) in enumerate(zip(ref["executables"], other["executables"])):
            match = ref_exe["instr_hashes"] == other_exe["instr_hashes"]
            print(f"  {ref_name} vs {name} [{ref_exe['type']}]: "
                  f"{'IDENTICAL' if match else 'DIFFERENT'}")

    # Compare param sizes
    print("\n3. PARAMETER BLOB SIZES:")
    for name, info in results.items():
        for exe in info["executables"]:
            if exe["param_size"] > 0:
                print(f"  {name} [{exe['type']}]: {exe['param_size']} bytes")

    # Key conclusion
    print("\n4. CONCLUSIONS:")
    all_same_instr = True
    weight_scales_differ = False

    scales = set()
    for name, info in results.items():
        ws = info["quant"].get("weight_scale")
        if ws is not None:
            scales.add(round(ws, 10))

    if len(scales) > 1:
        weight_scales_differ = True
        print(f"  !! WEIGHT SCALES DIFFER: {scales}")
        print(f"  → This means requantization multipliers differ across compilations")
        print(f"  → H1 is likely FALSE: instructions will embed different multipliers")
    else:
        print(f"  Weight scales are consistent: {scales}")

    for name in names[1:]:
        for i, (ref_exe, other_exe) in enumerate(zip(ref["executables"], results[name]["executables"])):
            if ref_exe["instr_hashes"] != other_exe["instr_hashes"]:
                all_same_instr = False

    if all_same_instr:
        print(f"  ✓ H1 CONFIRMED: All instruction bitstreams identical across weight variants")
        print(f"  → set_weights() can swap params without touching instructions")
    else:
        print(f"  ✗ H1 REJECTED: Instruction bitstreams differ across weight variants")
        print(f"  → set_weights() must also update instruction bitstreams")
        if weight_scales_differ:
            print(f"  → Root cause: different weight scales → different requant multipliers in instructions")


if __name__ == "__main__":
    main()
