#!/usr/bin/env python3
"""Systematic diagnosis of SpotTracker Edge TPU issue.

This script isolates the problem by testing minimal model variants.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def create_test_model(name: str, architecture: str, height: int = 16, width: int = 16):
    """Create a minimal test model with specific architecture.

    Architectures:
        "dense_x_only": Reshape → Softmax → Dense(1) with X-coords
        "dense_y_only": Reshape → Softmax → Dense(1) with Y-coords
        "dense_both": Reshape → Softmax → Dense(2) with [X, Y] coords
        "conv_x_only": Reshape → Softmax → Reshape[H,W,1] → Conv2D(1x1) with X-coords
        "conv_y_only": Reshape → Softmax → Reshape[H,W,1] → Conv2D(1x1) with Y-coords
        "mul_sum_x": Reshape → Softmax → Reshape[H,W,1] → Mul(X_grid) → GlobalAvgPool
        "mul_sum_y": Reshape → Softmax → Reshape[H,W,1] → Mul(Y_grid) → GlobalAvgPool
        "mul_sum_both": Reshape → Softmax → Reshape[H,W,1] → [Mul(X), Mul(Y)] → Concat
    """
    import tensorflow as tf

    # Create coordinate grids (no temperature scaling for clarity)
    x_coords = np.zeros((height, width), dtype=np.float32)
    for j in range(width):
        x_coords[:, j] = (j - (width - 1) / 2) / ((width - 1) / 2)  # -1 to +1

    y_coords = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        y_coords[i, :] = (i - (height - 1) / 2) / ((height - 1) / 2)  # -1 to +1

    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    inputs = tf.keras.Input(shape=(height, width, 1), batch_size=1, dtype=tf.float32)

    # Common: Reshape → Softmax
    reshaped = tf.keras.layers.Reshape((height * width,), name="flatten")(inputs)
    probs = tf.keras.layers.Softmax(name="softmax")(reshaped)

    if architecture == "dense_x_only":
        x_layer = tf.keras.layers.Dense(
            1, use_bias=False, trainable=False,
            kernel_initializer=tf.keras.initializers.Constant(x_flat.reshape(-1, 1)),
            name="x_sum"
        )
        outputs = x_layer(probs)

    elif architecture == "dense_y_only":
        y_layer = tf.keras.layers.Dense(
            1, use_bias=False, trainable=False,
            kernel_initializer=tf.keras.initializers.Constant(y_flat.reshape(-1, 1)),
            name="y_sum"
        )
        outputs = y_layer(probs)

    elif architecture == "dense_both":
        # Combined [H*W, 2] weight matrix
        combined_weights = np.stack([x_flat, y_flat], axis=1)  # [H*W, 2]
        xy_layer = tf.keras.layers.Dense(
            2, use_bias=False, trainable=False,
            kernel_initializer=tf.keras.initializers.Constant(combined_weights),
            name="xy_sum"
        )
        outputs = xy_layer(probs)

    elif architecture == "conv_x_only":
        # Reshape back to spatial, then 1x1 conv
        spatial = tf.keras.layers.Reshape((height, width, 1), name="to_spatial")(probs)
        # 1x1 conv with kernel [1,1,1,1], weight is mean of x_coords (just to test conv path)
        conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, use_bias=False, trainable=False,
            name="conv_x"
        )
        out = conv(spatial)
        conv.build(spatial.shape)
        conv.set_weights([np.array([[[[1.0]]]])])  # identity
        # Multiply by X coordinate grid
        x_grid = tf.constant(x_coords.reshape(1, height, width, 1), dtype=tf.float32)
        multiplied = tf.keras.layers.Multiply(name="mul_x")([out, x_grid])
        # Sum via global average pool then scale
        pooled = tf.keras.layers.GlobalAveragePooling2D(name="pool")(multiplied)
        # Scale by H*W to get sum instead of mean
        outputs = tf.keras.layers.Lambda(lambda x: x * height * width, name="scale")(pooled)

    elif architecture == "conv_y_only":
        spatial = tf.keras.layers.Reshape((height, width, 1), name="to_spatial")(probs)
        conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, use_bias=False, trainable=False,
            name="conv_y"
        )
        out = conv(spatial)
        conv.build(spatial.shape)
        conv.set_weights([np.array([[[[1.0]]]])])
        y_grid = tf.constant(y_coords.reshape(1, height, width, 1), dtype=tf.float32)
        multiplied = tf.keras.layers.Multiply(name="mul_y")([out, y_grid])
        pooled = tf.keras.layers.GlobalAveragePooling2D(name="pool")(multiplied)
        outputs = tf.keras.layers.Lambda(lambda x: x * height * width, name="scale")(pooled)

    elif architecture == "mul_sum_x":
        # Direct multiply + sum (no conv)
        spatial = tf.keras.layers.Reshape((height, width, 1), name="to_spatial")(probs)
        x_grid = tf.constant(x_coords.reshape(1, height, width, 1), dtype=tf.float32)
        multiplied = tf.keras.layers.Multiply(name="mul_x")([spatial, x_grid])
        pooled = tf.keras.layers.GlobalAveragePooling2D(name="pool")(multiplied)
        outputs = tf.keras.layers.Lambda(lambda x: x * height * width, name="scale")(pooled)

    elif architecture == "mul_sum_y":
        spatial = tf.keras.layers.Reshape((height, width, 1), name="to_spatial")(probs)
        y_grid = tf.constant(y_coords.reshape(1, height, width, 1), dtype=tf.float32)
        multiplied = tf.keras.layers.Multiply(name="mul_y")([spatial, y_grid])
        pooled = tf.keras.layers.GlobalAveragePooling2D(name="pool")(multiplied)
        outputs = tf.keras.layers.Lambda(lambda x: x * height * width, name="scale")(pooled)

    elif architecture == "mul_sum_both":
        # Both X and Y via multiply + sum
        spatial = tf.keras.layers.Reshape((height, width, 1), name="to_spatial")(probs)

        x_grid = tf.constant(x_coords.reshape(1, height, width, 1), dtype=tf.float32)
        mul_x = tf.keras.layers.Multiply(name="mul_x")([spatial, x_grid])
        pool_x = tf.keras.layers.GlobalAveragePooling2D(name="pool_x")(mul_x)
        x_out = tf.keras.layers.Lambda(lambda x: x * height * width, name="scale_x")(pool_x)

        y_grid = tf.constant(y_coords.reshape(1, height, width, 1), dtype=tf.float32)
        mul_y = tf.keras.layers.Multiply(name="mul_y")([spatial, y_grid])
        pool_y = tf.keras.layers.GlobalAveragePooling2D(name="pool_y")(mul_y)
        y_out = tf.keras.layers.Lambda(lambda x: x * height * width, name="scale_y")(pool_y)

        outputs = tf.keras.layers.Concatenate(name="offsets")([x_out, y_out])

    elif architecture == "conv1d_both":
        # Encode coordinate grids as Conv2D weights instead of constants
        # Reshape softmax output to [1, 1, H*W] - treat as 1x1 spatial, H*W channels
        spatial_1d = tf.keras.layers.Reshape((1, 1, height * width), name="to_1d")(probs)

        # X coordinate sum via Conv2D: kernel [1, 1, H*W, 1] with X coord weights
        x_conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, use_bias=False, trainable=False,
            name="x_conv"
        )
        x_out = x_conv(spatial_1d)
        x_conv.build(spatial_1d.shape)
        x_conv.set_weights([x_flat.reshape(1, 1, -1, 1)])

        # Y coordinate sum via Conv2D: kernel [1, 1, H*W, 1] with Y coord weights
        y_conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, use_bias=False, trainable=False,
            name="y_conv"
        )
        y_out = y_conv(spatial_1d)
        y_conv.build(spatial_1d.shape)
        y_conv.set_weights([y_flat.reshape(1, 1, -1, 1)])

        # Flatten outputs and concatenate
        x_flat_out = tf.keras.layers.Flatten(name="x_flat")(x_out)
        y_flat_out = tf.keras.layers.Flatten(name="y_flat")(y_out)
        outputs = tf.keras.layers.Concatenate(name="offsets")([x_flat_out, y_flat_out])

    elif architecture == "depthwise_both":
        # Use DepthwiseConv2D + Conv2D for weighted sum
        # Reshape to [H, W, 2] with duplicated channels
        spatial = tf.keras.layers.Reshape((height, width, 1), name="to_spatial")(probs)

        # Create two-channel tensor by concatenating
        doubled = tf.keras.layers.Concatenate(axis=-1, name="double")([spatial, spatial])

        # DepthwiseConv2D with full H×W kernel to do the weighted sum
        # kernel shape: [H, W, 2, 1] - one multiplier per position per channel
        dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(height, width),
            padding='valid',
            use_bias=False,
            trainable=False,
            name="depthwise"
        )
        out = dw_conv(doubled)
        dw_conv.build(doubled.shape)
        # Stack X and Y coordinate grids as depth multipliers
        kernel = np.stack([x_coords, y_coords], axis=-1).reshape(height, width, 2, 1)
        dw_conv.set_weights([kernel])

        # Output is [1, 1, 2], flatten to [2]
        outputs = tf.keras.layers.Flatten(name="offsets")(out)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def representative_dataset(height: int, width: int, n_samples: int = 100):
    """Generate representative dataset for quantization."""
    def gen():
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            data = rng.integers(0, 256, size=[1, height, width, 1]).astype(np.float32)
            yield [data]
    return gen


def quantize_model(model, height: int, width: int):
    """Convert to quantized TFLite."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(height, width)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8

    return converter.convert()


def compile_for_edgetpu(tflite_path: str, output_dir: str):
    """Run edgetpu_compiler."""
    if shutil.which("edgetpu_compiler") is None:
        return None, "edgetpu_compiler not found"

    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", output_dir, tflite_path],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        return None, f"Compilation failed: {result.stderr}"

    base = os.path.splitext(os.path.basename(tflite_path))[0]
    compiled_path = os.path.join(output_dir, f"{base}_edgetpu.tflite")
    log_path = os.path.join(output_dir, f"{base}_edgetpu.log")

    log_content = ""
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log_content = f.read()

    if not os.path.isfile(compiled_path):
        return None, f"No output file: {result.stdout}\n{log_content}"

    return compiled_path, log_content


def test_on_cpu(tflite_bytes: bytes, test_images: list):
    """Test quantized model on CPU TFLite interpreter."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()

    input_detail = interp.get_input_details()[0]
    output_detail = interp.get_output_details()[0]

    results = []
    for img in test_images:
        interp.set_tensor(input_detail["index"], img)
        interp.invoke()
        out = interp.get_tensor(output_detail["index"])
        results.append(out.copy())

    return results, input_detail, output_detail


def test_on_edgetpu(compiled_path: str, test_images: list, input_detail: dict, output_detail: dict):
    """Test compiled model on Edge TPU hardware."""
    from libredgetpu import SimpleInvoker

    with SimpleInvoker(compiled_path) as invoker:
        results = []
        for img in test_images:
            # Flatten to remove batch dimension, convert to bytes
            flat_img = img.flatten()
            raw_output = invoker.invoke_raw(flat_img.tobytes())
            out = np.frombuffer(raw_output, dtype=np.int8)
            results.append(out.copy())

    return results


def create_test_images(height: int, width: int):
    """Create test images with spots at different positions."""
    images = []
    positions = []

    # Uniform gray
    img = np.full((1, height, width, 1), 128, dtype=np.uint8)
    images.append(img)
    positions.append("uniform")

    # Bright spot upper-left
    img = np.zeros((1, height, width, 1), dtype=np.uint8)
    img[0, 1:4, 1:4, 0] = 255
    images.append(img)
    positions.append("upper-left")

    # Bright spot upper-right
    img = np.zeros((1, height, width, 1), dtype=np.uint8)
    img[0, 1:4, width-4:width-1, 0] = 255
    images.append(img)
    positions.append("upper-right")

    # Bright spot lower-left
    img = np.zeros((1, height, width, 1), dtype=np.uint8)
    img[0, height-4:height-1, 1:4, 0] = 255
    images.append(img)
    positions.append("lower-left")

    # Bright spot lower-right
    img = np.zeros((1, height, width, 1), dtype=np.uint8)
    img[0, height-4:height-1, width-4:width-1, 0] = 255
    images.append(img)
    positions.append("lower-right")

    # Bright spot center
    img = np.zeros((1, height, width, 1), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    img[0, cy-1:cy+2, cx-1:cx+2, 0] = 255
    images.append(img)
    positions.append("center")

    return images, positions


def main():
    import tensorflow as tf

    print("=" * 70)
    print("SpotTracker Systematic Diagnosis")
    print("=" * 70)

    height, width = 16, 16

    architectures = [
        "conv1d_both",       # Coordinate weights as Conv2D layer weights
        "depthwise_both",    # Full H×W DepthwiseConv2D weighted sum
        "dense_both",        # Original approach (compiler crashes)
        "mul_sum_both",      # Constant tensor multiply (CPU works, Edge TPU wrong)
    ]

    test_images, positions = create_test_images(height, width)

    for arch in architectures:
        print(f"\n{'='*70}")
        print(f"Architecture: {arch}")
        print("=" * 70)

        # Create model
        try:
            model = create_test_model(f"test_{arch}", arch, height, width)
            print(f"  Model created: {model.output_shape}")
        except Exception as e:
            print(f"  Model creation failed: {e}")
            continue

        # Test float model
        print("\n  Float model test:")
        for img, pos in zip(test_images, positions):
            out = model.predict(img.astype(np.float32), verbose=0)
            print(f"    {pos:12s}: {out.flatten()}")

        # Quantize
        try:
            tflite_bytes = quantize_model(model, height, width)
            print(f"\n  Quantized TFLite: {len(tflite_bytes)} bytes")
        except Exception as e:
            print(f"  Quantization failed: {e}")
            continue

        # Test on CPU
        print("\n  CPU TFLite test:")
        try:
            cpu_results, input_detail, output_detail = test_on_cpu(tflite_bytes, test_images)
            o_scale = output_detail["quantization_parameters"]["scales"][0]
            o_zp = output_detail["quantization_parameters"]["zero_points"][0]
            print(f"    output_scale={o_scale:.6f}, output_zp={o_zp}")
            for result, pos in zip(cpu_results, positions):
                raw = result.flatten()
                dequant = (raw.astype(np.float32) - o_zp) * o_scale
                print(f"    {pos:12s}: raw={raw}, dequant={dequant}")
        except Exception as e:
            print(f"  CPU test failed: {e}")
            continue

        # Compile for Edge TPU
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, f"{arch}.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_bytes)

            compiled_path, log = compile_for_edgetpu(tflite_path, tmpdir)

            if compiled_path is None:
                print(f"\n  Edge TPU compilation: FAILED")
                print(f"    {log[:500]}")
                continue

            # Check what ops are mapped
            print(f"\n  Edge TPU compilation: OK")
            for line in log.split("\n"):
                if "Mapped to Edge TPU" in line or "Not supported" in line:
                    print(f"    {line.strip()}")

            # Test on Edge TPU hardware
            print("\n  Edge TPU hardware test:")
            try:
                hw_results = test_on_edgetpu(compiled_path, test_images, input_detail, output_detail)
                for result, pos in zip(hw_results, positions):
                    raw = result[:output_detail["shape"][-1]]
                    dequant = (raw.astype(np.float32) - o_zp) * o_scale
                    print(f"    {pos:12s}: raw={raw}, dequant={dequant}")
            except Exception as e:
                print(f"    Hardware test failed: {e}")

    print("\n" + "=" * 70)
    print("Diagnosis complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
