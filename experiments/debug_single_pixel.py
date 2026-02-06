#!/usr/bin/env python3
"""Debug a single pixel computation step-by-step."""

import numpy as np

# Model parameters (from inspection)
input_scale = 0.003921568859368563
input_zp = 0
q_int8_scale = 0.003921568859368563
q_int8_zp = -128
weight_scale = 0.007874015718698502
weight_zp = 0
bias_scale = 3.0878494726493955e-05
conv_output_scale = 0.019999999552965164
conv_output_zp = -128
final_output_scale = 0.019999999552965164
final_output_zp = 0

# Input: constant 128 everywhere
input_uint8 = 128

print("="*60)
print("Manual Step-by-Step Computation")
print("="*60)

# Step 1: QUANTIZE uint8 → int8
# Formula: int8 = round((uint8 - input_zp) * input_scale / q_int8_scale + q_int8_zp)
# With input_scale = q_int8_scale, this simplifies to:
# int8 = round(uint8 - input_zp + q_int8_zp) = round(uint8 + (-128)) = uint8 - 128

input_float = (input_uint8 - input_zp) * input_scale
quantized_int8 = round(input_float / q_int8_scale + q_int8_zp)
print(f"\nStep 1: QUANTIZE uint8 → int8")
print(f"  Input uint8: {input_uint8}")
print(f"  → float: ({input_uint8} - {input_zp}) * {input_scale:.6f} = {input_float:.6f}")
print(f"  → int8: round({input_float:.6f} / {q_int8_scale:.6f} + {q_int8_zp}) = {quantized_int8}")

# For center pixel: 9 neighbors all = quantized_int8
# For corner pixel: 4 neighbors = quantized_int8, 5 neighbors = 0 (padding)
# For edge pixel: 6 neighbors = quantized_int8, 3 neighbors = 0

# Center pixel (2,2): 9 neighbors
print(f"\n--- Center Pixel (2,2) ---")
weight_int8 = 127  # all weights are 127
num_neighbors_center = 9
acc_center = num_neighbors_center * quantized_int8 * weight_int8
print(f"  Neighbors: {num_neighbors_center} × {quantized_int8} (int8)")
print(f"  Weight: {weight_int8} (int8)")
print(f"  INT32 accumulator: {num_neighbors_center} × {quantized_int8} × {weight_int8} = {acc_center}")

# Requantize: int32 → int8
# Formula: int8 = round(int32 * input_scale * weight_scale / output_scale + output_zp)
requant_float_center = acc_center * q_int8_scale * weight_scale / conv_output_scale
requant_int8_center = np.clip(round(requant_float_center + conv_output_zp), -128, 127)
print(f"  Requantize: {acc_center} * {q_int8_scale:.6f} * {weight_scale:.6f} / {conv_output_scale:.6f}")
print(f"            = {requant_float_center:.2f}")
print(f"            + {conv_output_zp} (zp) = {requant_float_center + conv_output_zp:.2f}")
print(f"            → int8 = {requant_int8_center}")

# ReLU
relu_int8_center = max(requant_int8_center, 0)
print(f"  After ReLU: max({requant_int8_center}, 0) = {relu_int8_center}")

# Final QUANTIZE int8 → uint8
# Formula: uint8 = round((int8 - output_zp) * conv_output_scale / final_output_scale + final_output_zp)
# With scales equal: uint8 = round(int8 + final_output_zp) = int8 + 128
final_uint8_center = np.clip(round(relu_int8_center + 128), 0, 255)
print(f"  Final uint8: {relu_int8_center} + 128 = {final_uint8_center}")

# Corner pixel (0,0): 4 neighbors
print(f"\n--- Corner Pixel (0,0) ---")
num_neighbors_corner = 4
acc_corner = num_neighbors_corner * quantized_int8 * weight_int8
print(f"  Neighbors: {num_neighbors_corner} × {quantized_int8} (int8), 5 × 0 (padding)")
print(f"  Weight: {weight_int8} (int8)")
print(f"  INT32 accumulator: {num_neighbors_corner} × {quantized_int8} × {weight_int8} = {acc_corner}")

requant_float_corner = acc_corner * q_int8_scale * weight_scale / conv_output_scale
requant_int8_corner = np.clip(round(requant_float_corner + conv_output_zp), -128, 127)
print(f"  Requantize: {acc_corner} * {q_int8_scale:.6f} * {weight_scale:.6f} / {conv_output_scale:.6f}")
print(f"            = {requant_float_corner:.2f}")
print(f"            + {conv_output_zp} (zp) = {requant_float_corner + conv_output_zp:.2f}")
print(f"            → int8 = {requant_int8_corner}")

relu_int8_corner = max(requant_int8_corner, 0)
print(f"  After ReLU: max({requant_int8_corner}, 0) = {relu_int8_corner}")

final_uint8_corner = np.clip(round(relu_int8_corner + 128), 0, 255)
print(f"  Final uint8: {relu_int8_corner} + 128 = {final_uint8_corner}")

# Edge pixel (0,1): 6 neighbors
print(f"\n--- Edge Pixel (0,1) ---")
num_neighbors_edge = 6
acc_edge = num_neighbors_edge * quantized_int8 * weight_int8
print(f"  Neighbors: {num_neighbors_edge} × {quantized_int8} (int8), 3 × 0 (padding)")
print(f"  Weight: {weight_int8} (int8)")
print(f"  INT32 accumulator: {num_neighbors_edge} × {quantized_int8} × {weight_int8} = {acc_edge}")

requant_float_edge = acc_edge * q_int8_scale * weight_scale / conv_output_scale
requant_int8_edge = np.clip(round(requant_float_edge + conv_output_zp), -128, 127)
print(f"  Requantize: {acc_edge} * {q_int8_scale:.6f} * {weight_scale:.6f} / {conv_output_scale:.6f}")
print(f"            = {requant_float_edge:.2f}")
print(f"            + {conv_output_zp} (zp) = {requant_float_edge + conv_output_zp:.2f}")
print(f"            → int8 = {requant_int8_edge}")

relu_int8_edge = max(requant_int8_edge, 0)
print(f"  After ReLU: max({requant_int8_edge}, 0) = {relu_int8_edge}")

final_uint8_edge = np.clip(round(relu_int8_edge + 128), 0, 255)
print(f"  Final uint8: {relu_int8_edge} + 128 = {final_uint8_edge}")

print(f"\n{'='*60}")
print("Predicted Output Pattern")
print(f"{'='*60}")
print(f"Corner (0,0): {final_uint8_corner}")
print(f"Edge   (0,1): {final_uint8_edge}")
print(f"Center (2,2): {final_uint8_center}")

print(f"\n{'='*60}")
print("TFLite Actual Output (from previous run)")
print(f"{'='*60}")
print("[[100 151 151 100]")
print(" [151 226 226 151]")
print(" [151 226 226 151]")
print(" [100 151 151 100]]")

print(f"\nComparison:")
print(f"  Corner predicted: {final_uint8_corner}, actual: 100")
print(f"  Edge predicted: {final_uint8_edge}, actual: 151")
print(f"  Center predicted: {final_uint8_center}, actual: 226")
