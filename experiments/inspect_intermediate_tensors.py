#!/usr/bin/env python3
"""Inspect intermediate tensor values from TFLite interpreter."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
from libredgetpu.tflite_builder import build_simple_depthwise

# Build model
tflite_bytes, metadata = build_simple_depthwise(
    height=4, width=4, ksize=3, num_filters=1,
    kernel_weights=np.ones((1, 3, 3, 1), dtype=np.float32),
    input_scale=1.0/255.0,
    output_scale=0.02
)

# Load interpreter
interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
interpreter.allocate_tensors()

# Get tensor details
print("="*60)
print("All Tensors")
print("="*60)
tensor_details = interpreter.get_tensor_details()
for i, t in enumerate(tensor_details):
    print(f"\n[{i}] {t['name']}")
    print(f"    Index: {t['index']}")
    print(f"    Shape: {t['shape']}")
    print(f"    Dtype: {t['dtype']}")
    if 'quantization' in t and t['quantization'] != (0.0, 0):
        print(f"    Quantization: scale={t['quantization'][0]}, zp={t['quantization'][1]}")

# Prepare input: constant 128
input_img = np.full((1, 4, 4, 1), 128, dtype=np.uint8)

# Set input
input_details = interpreter.get_input_details()[0]
interpreter.set_tensor(input_details['index'], input_img)

# Run inference
interpreter.invoke()

# Get output
output_details = interpreter.get_output_details()[0]
output_data = interpreter.get_tensor(output_details['index'])

print(f"\n{'='*60}")
print("Input/Output Values")
print(f"{'='*60}")
print(f"\nInput (uint8):")
print(input_img[0, :, :, 0])

print(f"\nOutput (uint8):")
print(output_data[0, :, :, 0])

# Inspect intermediate tensors
print(f"\n{'='*60}")
print("Intermediate Tensor Values")
print(f"{'='*60}")

# Tensor 1: quantize_out (int8)
quantize_out = interpreter.get_tensor(1)
print(f"\nTensor 1: quantize_out (int8)")
print(quantize_out[0, :, :, 0])
print(f"Range: [{quantize_out.min()}, {quantize_out.max()}]")
print(f"Unique values: {np.unique(quantize_out)}")

# Tensor 2: weights (int8)
weights = interpreter.get_tensor(2)
print(f"\nTensor 2: weights (int8)")
print(f"Shape: {weights.shape}")
print(weights[0, :, :, 0])
print(f"Range: [{weights.min()}, {weights.max()}]")

# Tensor 3: bias (int32)
bias = interpreter.get_tensor(3)
print(f"\nTensor 3: bias (int32)")
print(bias)

# Tensor 4: conv_out (int8, before final quantize)
conv_out = interpreter.get_tensor(4)
print(f"\nTensor 4: conv_out (int8)")
print(conv_out[0, :, :, 0])
print(f"Range: [{conv_out.min()}, {conv_out.max()}]")

print(f"\n{'='*60}")
print("Analysis")
print(f"{'='*60}")

# Key question: Why is output non-constant when quantize_out should be all 0?
print(f"\nKEY INSIGHT:")
print(f"If quantize_out is all 0 (as expected for input=128),")
print(f"then conv accumulator should be all 0 (0 × weight = 0),")
print(f"leading to constant output. But output is NOT constant!")
print(f"\nThis means either:")
print(f"1. quantize_out is NOT all 0")
print(f"2. bias is NOT zero")
print(f"3. Something else is happening")
