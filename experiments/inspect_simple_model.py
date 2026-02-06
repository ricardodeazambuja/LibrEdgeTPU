#!/usr/bin/env python3
"""Inspect the simple depthwise model to understand its structure."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libredgetpu.tflite_builder import build_simple_depthwise
from libredgetpu.tflite_parser import parse_full
import numpy as np

# Build minimal model
tflite_bytes, metadata = build_simple_depthwise(
    height=4, width=4, ksize=3, num_filters=1,
    kernel_weights=np.ones((1, 3, 3, 1), dtype=np.float32),
    input_scale=1.0/255.0,
    output_scale=0.02
)

print(f"Model size: {len(tflite_bytes)} bytes")
print(f"\nMetadata:")
for k, v in metadata.items():
    print(f"  {k}: {v}")

# Parse model
model = parse_full(tflite_bytes)

print(f"\n{'='*60}")
print("Tensor Details")
print(f"{'='*60}")

for i, tensor in enumerate(model.tensors):
    print(f"\nTensor {i}: {tensor.name}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type: {tensor.dtype}")
    print(f"  Scale: {tensor.scale}")
    print(f"  Zero point: {tensor.zero_point}")
    print(f"  Buffer index: {tensor.buffer_index}")

print(f"\n{'='*60}")
print("Operator Details")
print(f"{'='*60}")

for i, op in enumerate(model.operators):
    print(f"\nOperator {i}: {op.opcode_name}")
    print(f"  Inputs: {op.inputs}")
    print(f"  Outputs: {op.outputs}")
    print(f"  Builtin options type: {op.builtin_options_type}")

# Extract weights from buffer
weights_tensor = model.tensors[2]  # weights should be tensor 2
if weights_tensor.buffer_index is not None and weights_tensor.buffer_index < len(model.buffers):
    weights_data = model.buffers[weights_tensor.buffer_index]
    weights_int8 = np.frombuffer(weights_data, dtype=np.int8).reshape(weights_tensor.shape)
    print(f"\n{'='*60}")
    print("Weights Data")
    print(f"{'='*60}")
    print(f"Shape: {weights_int8.shape}")
    print(f"Int8 values:\n{weights_int8[0, :, :, 0]}")
    print(f"Range: [{weights_int8.min()}, {weights_int8.max()}]")
    print(f"Mean: {weights_int8.mean():.2f}")

# Extract bias
bias_tensor = model.tensors[3]
if bias_tensor.buffer_index is not None and bias_tensor.buffer_index < len(model.buffers):
    bias_data = model.buffers[bias_tensor.buffer_index]
    bias_int32 = np.frombuffer(bias_data, dtype=np.int32)
    print(f"\n{'='*60}")
    print("Bias Data")
    print(f"{'='*60}")
    print(f"Shape: {bias_int32.shape}")
    print(f"Values: {bias_int32}")
