#!/usr/bin/env python3
"""
Pure numpy reference implementation of quantized depthwise convolution.

This simulates the exact TFLite behavior step-by-step for verification.
"""

import numpy as np


def manual_depthwise_conv(image_uint8, kernel_float,
                         input_scale, output_scale,
                         weight_scales, padding='SAME',
                         use_relu=True, verbose=False):
    """
    Reference implementation of quantized depthwise convolution.

    Simulates exact TFLite behavior:
    1. QUANTIZE: uint8 → int8
    2. Depthwise Conv: int8 weights, int32 accumulator
    3. Requantize: int32 → int8 with per-channel scales
    4. ReLU: clip negatives to 0
    5. Output QUANTIZE: int8 → uint8

    Args:
        image_uint8: Input image [H, W] as uint8
        kernel_float: Kernels [1, ksize, ksize, n_filters] as float32
        input_scale: Input quantization scale
        output_scale: Output quantization scale
        weight_scales: List of per-channel weight scales [n_filters]
        padding: 'SAME' or 'VALID'
        use_relu: Apply ReLU after conv
        verbose: Print intermediate values

    Returns:
        output_uint8: [H, W, n_filters] as uint8
    """
    h, w = image_uint8.shape
    ksize = kernel_float.shape[1]
    n_filters = kernel_float.shape[3]

    if verbose:
        print(f"\n=== Manual Depthwise Conv Reference ===")
        print(f"Input: {image_uint8.shape} uint8, range [{image_uint8.min()}, {image_uint8.max()}]")
        print(f"Kernel: {kernel_float.shape} float32")
        print(f"Input scale: {input_scale}, Output scale: {output_scale}")

    # ── Step 1: QUANTIZE uint8 → int8 ──────────────────────────────────
    # TFLite formula: int8 = round((uint8 - input_zp) / input_scale * q_int8_scale + q_int8_zp)
    # With input_scale = 1/255, input_zp = 0, q_int8_scale = 1/255, q_int8_zp = -128:
    # int8 = round((uint8 - 0) / (1/255) * (1/255) + (-128))
    #      = round(uint8 - 128)
    #      = uint8 - 128 (for uint8 in [0, 255])

    # Simpler: TF-style normalization
    # uint8 [0, 255] → float [0, 1] → quantized int8 [-128, 127]
    input_float = image_uint8.astype(np.float32) * input_scale
    q_int8_scale = 1.0 / 255.0
    q_int8_zp = -128

    input_int8 = np.clip(
        np.round(input_float / q_int8_scale + q_int8_zp),
        -128, 127
    ).astype(np.int8)

    if verbose:
        print(f"\nAfter QUANTIZE: int8 range [{input_int8.min()}, {input_int8.max()}], mean {input_int8.mean():.2f}")

    # ── Step 2: Quantize weights per-channel ───────────────────────────
    kernel_int8 = np.zeros_like(kernel_float, dtype=np.int8)

    for ch in range(n_filters):
        kernel_ch = kernel_float[0, :, :, ch]
        ch_scale = weight_scales[ch]
        kernel_int8[0, :, :, ch] = np.clip(
            np.round(kernel_ch / ch_scale),
            -127, 127
        ).astype(np.int8)

    if verbose:
        for ch in range(n_filters):
            k = kernel_float[0, :, :, ch]
            k_int8 = kernel_int8[0, :, :, ch]
            print(f"  Channel {ch}: float range [{k.min():.3f}, {k.max():.3f}], "
                  f"int8 range [{k_int8.min()}, {k_int8.max()}], scale {weight_scales[ch]:.6f}")

    # ── Step 3: Depthwise convolution with int32 accumulator ───────────
    # Each output channel is convolved independently with its kernel

    pad = ksize // 2 if padding == 'SAME' else 0
    output_int32 = np.zeros((h, w, n_filters), dtype=np.int32)

    # Manual convolution (slow but explicit)
    for ch in range(n_filters):
        kernel_ch = kernel_int8[0, :, :, ch]  # [ksize, ksize]

        for y in range(h):
            for x in range(w):
                # Accumulate
                acc = 0
                for ky in range(ksize):
                    for kx in range(ksize):
                        # Input coordinate with padding
                        iy = y + ky - pad
                        ix = x + kx - pad

                        # Zero padding outside boundaries
                        if 0 <= iy < h and 0 <= ix < w:
                            pixel = int(input_int8[iy, ix])
                            weight = int(kernel_ch[ky, kx])
                            acc += pixel * weight

                # Bias (always 0 in our models)
                output_int32[y, x, ch] = acc

    if verbose:
        print(f"\nAfter Conv (int32 accumulator):")
        for ch in range(n_filters):
            ch_vals = output_int32[:, :, ch]
            print(f"  Channel {ch}: range [{ch_vals.min()}, {ch_vals.max()}], mean {ch_vals.mean():.1f}")

    # ── Step 4: Requantize int32 → int8 with per-channel scales ────────
    # Formula: int8 = int32 * input_scale * weight_scale / output_scale
    # With input_scale = 1/255, this becomes:
    # int8 = int32 * (1/255) * weight_scale / output_scale

    output_int8 = np.zeros((h, w, n_filters), dtype=np.int8)

    for ch in range(n_filters):
        # Per-channel requantization
        multiplier = q_int8_scale * weight_scales[ch] / output_scale

        ch_vals = output_int32[:, :, ch].astype(np.float64) * multiplier
        ch_vals_int8 = np.clip(np.round(ch_vals), -128, 127).astype(np.int8)

        output_int8[:, :, ch] = ch_vals_int8

    if verbose:
        print(f"\nAfter Requantize (int8):")
        for ch in range(n_filters):
            ch_vals = output_int8[:, :, ch]
            print(f"  Channel {ch}: range [{ch_vals.min()}, {ch_vals.max()}], mean {ch_vals.mean():.2f}")

    # ── Step 5: ReLU (optional) ─────────────────────────────────────────
    if use_relu:
        output_int8 = np.maximum(output_int8, 0)

        if verbose:
            print(f"\nAfter ReLU:")
            for ch in range(n_filters):
                ch_vals = output_int8[:, :, ch]
                print(f"  Channel {ch}: range [{ch_vals.min()}, {ch_vals.max()}], mean {ch_vals.mean():.2f}")

    # ── Step 6: QUANTIZE int8 → uint8 ──────────────────────────────────
    # With output_scale, output_zp = 0 for ReLU (non-negative):
    # uint8 = int8 * output_scale / output_scale + 0 = int8 (but cast to uint8)
    # Simpler: just view int8 as uint8 for non-negative values

    output_uint8 = np.clip(output_int8.astype(np.int16) + 128, 0, 255).astype(np.uint8)

    if verbose:
        print(f"\nFinal Output (uint8):")
        for ch in range(n_filters):
            ch_vals = output_uint8[:, :, ch]
            print(f"  Channel {ch}: range [{ch_vals.min()}, {ch_vals.max()}], mean {ch_vals.mean():.2f}")

    return output_uint8


if __name__ == '__main__':
    print("Testing manual depthwise conv reference implementation")

    # Test case: 4×4 constant input, 3×3 kernel of all 1s
    image = np.full((4, 4), 128, dtype=np.uint8)
    kernel = np.ones((1, 3, 3, 1), dtype=np.float32)  # Single channel

    input_scale = 1.0 / 255.0
    weight_scale = 1.0 / 127.0  # Normalized kernel
    output_scale = 0.02  # Typical for small kernels

    result = manual_depthwise_conv(
        image, kernel,
        input_scale=input_scale,
        output_scale=output_scale,
        weight_scales=[weight_scale],
        verbose=True
    )

    print(f"\n=== Result ===")
    print(result[:, :, 0])
    print(f"\nExpected: all pixels should be approximately 128 (constant input → constant output)")
