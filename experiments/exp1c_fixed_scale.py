"""Experiment 1c: Can patching weights with fixed quantization scale
produce identical EO instructions?

Critical hypothesis: If we keep the same weight_scale (and thus the same
requant multiplier), the edgetpu_compiler should produce identical EO
instructions, and we'd only need to swap PARAMETER_CACHING params.

Method:
1. Create template Dense(256) TFLite (random weights from Keras)
2. Patch weight buffer with different int8 values (same scale)
3. Compile each variant
4. Compare EO instruction hashes
"""

import os
import sys
import tempfile
import subprocess
import hashlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from libredgetpu.tflite_parser import parse as parse_tflite, parse_full
from libredgetpu.delegate import parse_darwinn, TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY

N = 256

def sha256(data):
    return hashlib.sha256(data).hexdigest()[:16]


def compile_and_analyze(tflite_bytes, name, tmpdir):
    """Compile TFLite bytes, return dict of exe info."""
    model_path = os.path.join(tmpdir, f"{name}.tflite")
    with open(model_path, "wb") as f:
        f.write(tflite_bytes)

    out_dir = os.path.join(tmpdir, f"out_{name}")
    os.makedirs(out_dir, exist_ok=True)

    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", out_dir, model_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"error": result.stderr[:200]}

    compiled = os.path.join(out_dir, f"{name}_edgetpu.tflite")
    if not os.path.isfile(compiled):
        return {"error": "compiled file not found"}

    with open(compiled, "rb") as f:
        cdata = f.read()

    model = parse_tflite(cdata)
    exes = parse_darwinn(model.custom_op_data)

    info = {
        "output_scale": model.output_tensor.scale,
        "output_zp": model.output_tensor.zero_point,
        "exes": {},
    }
    type_names = {0: "SA", 1: "PC", 2: "EO"}
    for exe in exes:
        tn = type_names.get(exe.exec_type, f"T{exe.exec_type}")
        info["exes"][tn] = {
            "instr_hash": sha256(exe.bitstreams[0].data) if exe.bitstreams else None,
            "instr_size": len(exe.bitstreams[0].data) if exe.bitstreams else 0,
            "param_size": len(exe.parameters) if exe.parameters else 0,
            "param_hash": sha256(exe.parameters) if exe.parameters else None,
        }
    return info


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"=== Experiment 1c: Fixed-Scale Weight Patching (Dense {N}) ===\n")

    # Step 1: Create the template TFLite
    print("Step 1: Creating template with TF random weights...")
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=N, input_shape=[N], use_bias=False),
    ])

    def representative_dataset():
        for i in range(256):
            data = np.zeros([1, N]) + i
            yield [data.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    template_bytes = converter.convert()

    # Get template weight scale
    interp = tf.lite.Interpreter(model_content=template_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    w_detail = None
    for td in interp.get_tensor_details():
        if td["shape"].tolist() == [N, N] and td["index"] not in (inp["index"], out["index"]):
            w_detail = td
            break

    template_w_scale = float(w_detail["quantization_parameters"]["scales"][0])
    template_w_zp = int(w_detail["quantization_parameters"]["zero_points"][0])
    template_weights_int8 = interp.get_tensor(w_detail["index"]).copy()

    print(f"  weight_scale = {template_w_scale:.8f}")
    print(f"  weight_zp = {template_w_zp}")
    print(f"  input_scale = {float(inp['quantization_parameters']['scales'][0])}")
    print(f"  output_scale = {float(out['quantization_parameters']['scales'][0])}")
    print(f"  template int8 weights: min={template_weights_int8.min()}, max={template_weights_int8.max()}")

    # Step 2: Patch weights — keep same TFLite structure, same quant params, just new int8 values
    print("\nStep 2: Creating patched variants (same quant scale, different int8 values)...")

    rng_A = np.random.default_rng(42)
    rng_B = np.random.default_rng(99)

    # Quantize new float weights with the TEMPLATE's scale
    def quantize_with_template_scale(weights_f32):
        return np.clip(
            np.round(weights_f32 / template_w_scale + template_w_zp),
            -128, 127
        ).astype(np.int8)

    variants = {
        "template": template_weights_int8.astype(np.int8),
        "random_A": quantize_with_template_scale(rng_A.uniform(-0.3, 0.3, (N, N)).astype(np.float32)),
        "random_B": quantize_with_template_scale(rng_B.uniform(-0.3, 0.3, (N, N)).astype(np.float32)),
        "identity": quantize_with_template_scale(np.eye(N, dtype=np.float32) * template_w_scale * 50),
        "small": quantize_with_template_scale(rng_A.uniform(-0.01, 0.01, (N, N)).astype(np.float32)),
    }

    for vname, w_int8 in variants.items():
        print(f"  {vname}: int8 range [{w_int8.min()}, {w_int8.max()}]")

    # Step 3: Patch each into the TFLite and compile
    print("\nStep 3: Patching and compiling...")

    full = parse_full(template_bytes)
    # Find the weight buffer
    weight_buffer_idx = None
    weight_buffer_offset = None
    buf_array = bytearray(template_bytes)
    expected_size = N * N
    for i, buffer_data in enumerate(full.buffers):
        if buffer_data is not None and len(buffer_data) == expected_size:
            offset = buf_array.find(buffer_data)
            if offset >= 0:
                weight_buffer_idx = i
                weight_buffer_offset = offset
                break

    if weight_buffer_offset is None:
        print("ERROR: Could not find weight buffer!")
        return

    print(f"  Weight buffer at offset {weight_buffer_offset} (buffer index {weight_buffer_idx})")

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for vname, w_int8 in variants.items():
            # Patch
            patched = bytearray(template_bytes)
            patched[weight_buffer_offset:weight_buffer_offset + expected_size] = w_int8.flatten().tobytes()
            patched = bytes(patched)

            info = compile_and_analyze(patched, vname, tmpdir)
            results[vname] = info

            if "error" in info:
                print(f"  {vname}: ERROR - {info['error']}")
            else:
                eo = info["exes"].get("EO", {})
                pc = info["exes"].get("PC", {})
                print(f"  {vname}: EO_instr={eo.get('instr_hash', '?')}, "
                      f"PC_params_hash={pc.get('param_hash', '?')}, "
                      f"out_scale={info['output_scale']:.4f}")

    # Step 4: Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    ref = results.get("template", {})
    if "error" in ref:
        print("Template failed to compile!")
        return

    ref_eo_hash = ref["exes"]["EO"]["instr_hash"]
    ref_pc_hash = ref["exes"]["PC"]["instr_hash"]

    print(f"\nReference (template):")
    print(f"  EO instr: {ref_eo_hash}")
    print(f"  PC instr: {ref_pc_hash}")
    print(f"  output_scale: {ref['output_scale']}")

    all_eo_match = True
    all_pc_match = True
    all_output_scale_match = True

    for vname, info in results.items():
        if vname == "template" or "error" in info:
            continue
        eo_match = info["exes"]["EO"]["instr_hash"] == ref_eo_hash
        pc_match = info["exes"]["PC"]["instr_hash"] == ref_pc_hash
        scale_match = abs(info["output_scale"] - ref["output_scale"]) < 1e-6

        if not eo_match:
            all_eo_match = False
        if not pc_match:
            all_pc_match = False
        if not scale_match:
            all_output_scale_match = False

        print(f"\n  {vname}:")
        print(f"    EO instr: {info['exes']['EO']['instr_hash']} "
              f"{'== MATCH' if eo_match else '!= DIFFERENT'}")
        print(f"    PC instr: {info['exes']['PC']['instr_hash']} "
              f"{'== MATCH' if pc_match else '!= DIFFERENT'}")
        print(f"    output_scale: {info['output_scale']:.6f} "
              f"{'== MATCH' if scale_match else '!= DIFFERENT'}")

    print("\n" + "="*60)
    print("CONCLUSIONS:")
    if all_eo_match:
        print("  [CONFIRMED] EO instructions IDENTICAL across all patched variants!")
        print("  → Patching weights with fixed scale preserves instruction stream")
        print("  → set_weights() only needs to swap PC params, NOT EO instructions")
    else:
        print("  [REJECTED] EO instructions differ across variants")
        print("  → Even with fixed weight scale, compiler changes EO instructions")

    if all_pc_match:
        print("  [CONFIRMED] PC instructions also identical (expected)")
    else:
        print("  [NOTE] PC instructions differ (unexpected)")

    if all_output_scale_match:
        print("  [CONFIRMED] Output quantization scale preserved")
    else:
        print("  [WARNING] Output scale changed — compiler recomputed quant params!")
        print("  → This means the compiler doesn't just use the TFLite quant params as-is")


if __name__ == "__main__":
    main()
