#!/usr/bin/env python3
"""
Test to understand TFLite QUANTIZE operator behavior.

The QUANTIZE operator formula is:
  output_value = round((input_value - input_zp) * input_scale / output_scale + output_zp)

For our case:
  Input: uint8, scale=1/255, zp=0
  Output: int8, scale=1/255, zp=-128

  For input=128:
    output = round((128 - 0) * (1/255) / (1/255) + (-128))
           = round(128 - 128)
           = 0

This SHOULD work! Let's verify what's actually in the model.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

model_path = Path(__file__).parent.parent / 'libredgetpu' / 'optical_flow' / 'templates' / 'gabor_64x64_7k_4o_2s.tflite'

interp = tf.lite.Interpreter(model_path=str(model_path))
interp.allocate_tensors()

# Get tensor details
tensor_details = interp.get_tensor_details()

print("="*60)
print("Tensor Details:")
print("="*60)
for i, detail in enumerate(tensor_details):
    print(f"\nTensor {i}: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Dtype: {detail['dtype']}")
    if 'quantization_parameters' in detail:
        qp = detail['quantization_parameters']
        if 'scales' in qp and len(qp['scales']) > 0:
            print(f"  Scales: {qp['scales']}")
        if 'zero_points' in qp and len(qp['zero_points']) > 0:
            print(f"  Zero points: {qp['zero_points']}")

print("\n" + "="*60)
print("QUANTIZE Operator Test:")
print("="*60)

# Manually compute what the QUANTIZE operator should produce
input_uint8 = 128
input_scale = 1.0 / 255.0
input_zp = 0
output_scale = 1.0 / 255.0
output_zp = -128

# Formula: output = round((input - input_zp) * input_scale / output_scale + output_zp)
float_value = (input_uint8 - input_zp) * input_scale
print(f"Step 1: (input - input_zp) * input_scale = ({input_uint8} - {input_zp}) * {input_scale} = {float_value}")

quantized_value = float_value / output_scale + output_zp
print(f"Step 2: float_value / output_scale + output_zp = {float_value} / {output_scale} + {output_zp} = {quantized_value}")

final_value = np.clip(np.round(quantized_value), -128, 127).astype(np.int8)
print(f"Step 3: clip(round({quantized_value}), -128, 127) = {final_value}")

print(f"\nExpected output from QUANTIZE for uint8={input_uint8}: int8={final_value}")

# Now run the actual model
test_input = np.ones((1, 64, 64, 1), dtype=np.uint8) * 128
interp.set_tensor(0, test_input)
interp.invoke()

# Get the quantize_out tensor
quantize_out = interp.get_tensor(1)  # Tensor 1 is quantize_out
print(f"Actual output from TFLite QUANTIZE: {np.unique(quantize_out)}")
print(f"Mean: {quantize_out.mean():.4f}, Std: {quantize_out.std():.4f}")

if np.allclose(quantize_out, final_value):
    print("\n✅ QUANTIZE operator works as expected!")
else:
    print(f"\n❌ QUANTIZE operator produces unexpected values!")
    print(f"   Expected: {final_value}")
    print(f"   Got: unique values = {np.unique(quantize_out)}")
