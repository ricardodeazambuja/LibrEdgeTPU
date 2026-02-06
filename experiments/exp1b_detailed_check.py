"""Experiment 1b: Detailed check of executable types and the cached vs streamed question.

The first experiment showed STAND_ALONE + EXECUTION_ONLY, not PARAMETER_CACHING + EXECUTION_ONLY.
This is surprising. Let's understand:
1. What executable types does our existing template (from template_gen) produce?
2. What does the experiment's model produce?
3. Is there a difference in how the model is constructed?

Also, check if the STAND_ALONE instructions encode the requant multiplier,
while the EXECUTION_ONLY instructions do not (since EO was identical).
"""

import os
import sys
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from libredgetpu.tflite_parser import parse as parse_tflite
from libredgetpu.delegate import parse_darwinn, TYPE_PARAMETER_CACHING, TYPE_EXECUTION_ONLY, TYPE_STAND_ALONE


def sha256(data):
    return hashlib.sha256(data).hexdigest()[:16]


def analyze_model(path, label):
    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"Path: {path}")
    print(f"{'='*60}")

    with open(path, "rb") as f:
        data = f.read()

    model = parse_tflite(data)
    print(f"Input: shape={model.input_tensor.shape}, scale={model.input_tensor.scale}, zp={model.input_tensor.zero_point}")
    print(f"Output: shape={model.output_tensor.shape}, scale={model.output_tensor.scale}, zp={model.output_tensor.zero_point}")

    exes = parse_darwinn(model.custom_op_data)
    print(f"Number of executables: {len(exes)}")

    for exe in exes:
        type_name = {0: "PARAMETER_CACHING", 1: "EXECUTION_ONLY", 2: "STAND_ALONE"}.get(
            exe.exec_type, f"UNKNOWN({exe.exec_type})")
        print(f"\n  [{type_name}]")
        print(f"    Bitstreams: {len(exe.bitstreams)}")
        for i, bs in enumerate(exe.bitstreams):
            print(f"      [{i}] {len(bs.data)} bytes, hash={sha256(bs.data)}")
        print(f"    Parameters: {len(exe.parameters) if exe.parameters else 0} bytes")
        if exe.parameters:
            print(f"      hash={sha256(exe.parameters)}")
        print(f"    Output layers: {len(exe.output_layers) if exe.output_layers else 0}")
        if exe.output_layers:
            for ol in exe.output_layers:
                print(f"      size={ol.size_bytes} bytes")
        print(f"    DMA steps: {len(exe.dma_steps) if exe.dma_steps else 0}")
        if hasattr(exe, 'caching_token'):
            print(f"    Caching token: {exe.caching_token}")


# Check existing templates
templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
for fname in sorted(os.listdir(templates_dir)):
    if fname.endswith("_edgetpu.tflite"):
        analyze_model(os.path.join(templates_dir, fname), f"Template: {fname}")

# Check existing template_gen.py output (the representative_dataset matters)
print("\n" + "="*60)
print("KEY QUESTION: Why did exp1 produce STAND_ALONE instead of PARAMETER_CACHING?")
print("="*60)
print("""
Hypothesis: The representative_dataset in exp1 was different from template_gen.
In template_gen, the dataset uses: np.zeros([1, n]) + i for i in range(256)
In exp1, the dataset is identical: np.zeros([1, n]) + i for i in range(256)

So the representative datasets are the same. The difference must be in the
weights themselves, or in model construction (set_weights vs random init).

The template_gen does NOT set_weights — it uses Keras random initialization.
exp1 sets specific weights. Let's check if the template is really PC+EO.
""")
