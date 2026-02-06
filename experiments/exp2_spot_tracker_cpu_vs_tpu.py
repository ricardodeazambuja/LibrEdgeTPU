"""Experiment 2: SpotTracker CPU vs Edge TPU output comparison.

Question: How exactly do Edge TPU outputs differ from CPU TFLite for the
soft argmax tracker model? Is it a simple negation? An affine transform?
Something else entirely?

Method:
  Phase A — CPU baseline (no hardware needed):
    1. Load the uncompiled bright_16x16.tflite
    2. Run on CPU via tflite_runtime for 6 test images
    3. Record raw int8 output bytes AND dequantized float values
    4. Verify CPU values match expected soft argmax behavior

  Phase B — Edge TPU comparison (requires hardware):
    1. Load the compiled bright_16x16_edgetpu.tflite
    2. Run on Edge TPU via libredgetpu for the SAME 6 test images
    3. Record raw int8 output bytes AND dequantized float values
    4. Compare raw bytes: hw[i] vs cpu[i] per element

  Phase C — Isolated op models (requires TF + edgetpu_compiler + hardware):
    1. Softmax-only model: does TPU softmax match CPU?
    2. Conv2D-only model: does TPU weighted sum match CPU?
    3. X-only model: single output, no concat
    4. Y-only model: single output, no concat (with and without offset)

  Phase D — Linear fit:
    1. Across all test images, fit hw_raw = a * cpu_raw + b per output element
    2. Report the transformation the Edge TPU applies

Expected if simple negation: a=-1, b=0 (or b=small constant due to quantization).

Findings (2026-02-05):
  The Edge TPU hardware ALWAYS outputs bytes in uint8 format. For models with
  int8 output type (TFLite TensorType=9), the raw bytes have their sign bit
  flipped compared to CPU TFLite:

    hw_byte = cpu_byte XOR 0x80

  This was verified across 8 test images on the 16x16 SpotTracker model — every
  single byte matched perfectly after applying the XOR correction.

  The perceived "negation" that led to the negate_outputs hack was entirely caused
  by interpreting uint8 bytes as int8 without the sign-bit correction. The Edge TPU
  computes the EXACT SAME values as CPU TFLite.

  Fix applied: simple_invoker.py checks output dtype; if INT8, XOR raw bytes with
  0x80 before dequantization. spot_tracker.py negate_outputs hack removed.

  Phase C (isolated models) was NOT needed — Phase B fully explained the discrepancy.

Usage:
  python -m experiments.exp2_spot_tracker_cpu_vs_tpu            # CPU only
  python -m experiments.exp2_spot_tracker_cpu_vs_tpu --hardware  # CPU + TPU
  python -m experiments.exp2_spot_tracker_cpu_vs_tpu --isolate   # all phases
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "..", "tracker", "templates")
UNCOMPILED = os.path.join(TEMPLATE_DIR, "bright_16x16.tflite")
COMPILED = os.path.join(TEMPLATE_DIR, "bright_16x16_edgetpu.tflite")
SIDECAR = os.path.join(TEMPLATE_DIR, "bright_16x16_edgetpu.json")

HEIGHT, WIDTH = 16, 16

# ---------------------------------------------------------------------------
# Test image creation
# ---------------------------------------------------------------------------

def create_test_images():
    """Create 6 test images as uint8 [1, H, W, 1] arrays.

    Returns list of (name, image_array) tuples.
    """
    images = []

    # 1. Uniform gray (128)
    img = np.full((1, HEIGHT, WIDTH, 1), 128, dtype=np.uint8)
    images.append(("uniform_128", img))

    # 2. Single bright pixel at upper-left corner (1, 1)
    img = np.zeros((1, HEIGHT, WIDTH, 1), dtype=np.uint8)
    img[0, 1, 1, 0] = 255
    images.append(("pixel_UL(1,1)", img))

    # 3. Single bright pixel at upper-right corner (1, 14)
    img = np.zeros((1, HEIGHT, WIDTH, 1), dtype=np.uint8)
    img[0, 1, WIDTH - 2, 0] = 255
    images.append(("pixel_UR(1,14)", img))

    # 4. Single bright pixel at lower-left corner (14, 1)
    img = np.zeros((1, HEIGHT, WIDTH, 1), dtype=np.uint8)
    img[0, HEIGHT - 2, 1, 0] = 255
    images.append(("pixel_LL(14,1)", img))

    # 5. Single bright pixel at lower-right corner (14, 14)
    img = np.zeros((1, HEIGHT, WIDTH, 1), dtype=np.uint8)
    img[0, HEIGHT - 2, WIDTH - 2, 0] = 255
    images.append(("pixel_LR(14,14)", img))

    # 6. Single bright pixel at center (8, 8)
    img = np.zeros((1, HEIGHT, WIDTH, 1), dtype=np.uint8)
    img[0, HEIGHT // 2, WIDTH // 2, 0] = 255
    images.append(("pixel_C(8,8)", img))

    # 7. 3x3 bright patch upper-left
    img = np.zeros((1, HEIGHT, WIDTH, 1), dtype=np.uint8)
    img[0, 1:4, 1:4, 0] = 255
    images.append(("patch3_UL", img))

    # 8. 3x3 bright patch lower-right
    img = np.zeros((1, HEIGHT, WIDTH, 1), dtype=np.uint8)
    img[0, HEIGHT - 4:HEIGHT - 1, WIDTH - 4:WIDTH - 1, 0] = 255
    images.append(("patch3_LR", img))

    return images


def compute_expected_soft_argmax(image, temperature=0.1):
    """Compute expected soft argmax output in float32 (no quantization).

    Returns (x_coord, y_coord_with_offset) in the model's internal scale.
    """
    # Create coordinate grids matching spot_tracker_gen.py
    x_coords = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for j in range(WIDTH):
        x_coords[:, j] = (j - (WIDTH - 1) / 2) / ((WIDTH - 1) / 2) / temperature

    y_coords = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for i in range(HEIGHT):
        y_coords[i, :] = (i - (HEIGHT - 1) / 2) / ((HEIGHT - 1) / 2) / temperature

    y_offset = 1.0 / temperature  # = 10.0 for temperature=0.1
    y_coords_offset = y_coords + y_offset

    # Softmax over flattened image
    pixels = image[0, :, :, 0].astype(np.float32).flatten()
    # Subtract max for numerical stability
    pixels_shifted = pixels - np.max(pixels)
    exp_pixels = np.exp(pixels_shifted)
    probs = exp_pixels / np.sum(exp_pixels)

    # Weighted sum (dot product of probs with coordinate grids)
    x_out = np.sum(probs * x_coords.flatten())
    y_out = np.sum(probs * y_coords_offset.flatten())

    return float(x_out), float(y_out)


# ---------------------------------------------------------------------------
# Phase A: CPU TFLite baseline
# ---------------------------------------------------------------------------

def run_cpu_tflite(images):
    """Run uncompiled TFLite model on CPU via tflite_runtime.

    Returns list of (name, raw_int8, dequantized_float) tuples.
    """
    from tflite_runtime.interpreter import Interpreter

    if not os.path.exists(UNCOMPILED):
        print(f"  ERROR: Uncompiled model not found: {UNCOMPILED}")
        return []

    interpreter = Interpreter(model_path=UNCOMPILED)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale = input_details["quantization"][0]
    in_zp = input_details["quantization"][1]
    out_scale = output_details["quantization"][0]
    out_zp = output_details["quantization"][1]
    out_dtype = output_details["dtype"]

    print(f"  Input:  scale={in_scale}, zp={in_zp}, dtype={input_details['dtype']}")
    print(f"  Output: scale={out_scale}, zp={out_zp}, dtype={out_dtype}")
    print(f"  Input shape:  {input_details['shape']}")
    print(f"  Output shape: {output_details['shape']}")

    results = []
    for name, img in images:
        # Quantize input (same as spot_tracker.py)
        if in_scale > 0:
            quantized = np.clip(
                np.round(img.astype(np.float32) / max(in_scale, 1e-9) + in_zp),
                0, 255
            ).astype(np.uint8)
        else:
            quantized = img

        interpreter.set_tensor(input_details["index"], quantized)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])[0]  # remove batch

        # Dequantize
        dequant = (raw_output.astype(np.float32) - out_zp) * out_scale

        results.append((name, raw_output.copy(), dequant.copy()))

    return results


# ---------------------------------------------------------------------------
# Phase B: Edge TPU hardware
# ---------------------------------------------------------------------------

def run_edge_tpu(images):
    """Run compiled TFLite model on Edge TPU via libredgetpu.

    Returns list of (name, raw_int8, dequantized_float) tuples.
    """
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))
    from libredgetpu.simple_invoker import SimpleInvoker

    if not os.path.exists(COMPILED):
        print(f"  ERROR: Compiled model not found: {COMPILED}")
        return []

    with open(SIDECAR) as f:
        meta = json.load(f)

    out_scale = meta["output_scale"]
    out_zp = meta["output_zero_point"]
    in_scale = meta["input_scale"]
    in_zp = meta["input_zero_point"]

    print(f"  Sidecar: in_scale={in_scale}, in_zp={in_zp}")
    print(f"  Sidecar: out_scale={out_scale}, out_zp={out_zp}")

    results = []
    with SimpleInvoker(COMPILED) as invoker:
        for name, img in images:
            # Quantize input (same as spot_tracker.py)
            quantized = np.clip(
                np.round(img.astype(np.float32) / max(in_scale, 1e-9) + in_zp),
                0, 255
            ).astype(np.uint8)

            raw_bytes = invoker.invoke_raw(quantized.flatten().tobytes())
            raw_int8 = np.frombuffer(raw_bytes, dtype=np.int8)[:2]

            # Dequantize
            dequant = (raw_int8.astype(np.float32) - out_zp) * out_scale

            results.append((name, raw_int8.copy(), dequant.copy()))

    return results


# ---------------------------------------------------------------------------
# Phase C: Isolated models
# ---------------------------------------------------------------------------

def create_and_test_isolated_models(images, run_hw=False):
    """Create minimal models isolating individual ops, compare CPU vs TPU."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf
    from tflite_runtime.interpreter import Interpreter

    results = {}

    # ---- Model C1: Softmax only ----
    # Input [1,16,16,1] -> Reshape [256] -> Softmax -> Output [256]
    print("\n  --- C1: Softmax only ---")
    inp = tf.keras.Input(shape=(HEIGHT, WIDTH, 1), batch_size=1)
    x = tf.keras.layers.Reshape((HEIGHT * WIDTH,))(inp)
    x = tf.keras.layers.Softmax()(x)
    m_softmax = tf.keras.Model(inputs=inp, outputs=x)
    results["C1_softmax"] = _compile_and_compare(
        m_softmax, "softmax_only", images, run_hw,
        desc="Softmax only: does TPU softmax match CPU?"
    )

    # ---- Model C2: Conv2D only (all-ones kernel = sum) ----
    # Input [1,16,16,1] -> Conv2D(16x16, ones) -> Output [1,1,1,1]
    print("\n  --- C2: Conv2D sum (ones kernel) ---")
    inp = tf.keras.Input(shape=(HEIGHT, WIDTH, 1), batch_size=1)
    kernel = np.ones((HEIGHT, WIDTH, 1, 1), dtype=np.float32)
    x = tf.keras.layers.Conv2D(
        1, kernel_size=(HEIGHT, WIDTH), padding="valid",
        use_bias=False, trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(kernel),
    )(inp)
    x = tf.keras.layers.Flatten()(x)
    m_conv_sum = tf.keras.Model(inputs=inp, outputs=x)
    results["C2_conv_sum"] = _compile_and_compare(
        m_conv_sum, "conv_sum", images, run_hw,
        desc="Conv2D with all-ones kernel: is sum preserved?"
    )

    # ---- Model C3: X-only (softmax -> conv2d with x_coords -> single output) ----
    print("\n  --- C3: X-only (softmax + x_coord conv) ---")
    temperature = 0.1
    x_coords = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for j in range(WIDTH):
        x_coords[:, j] = (j - (WIDTH - 1) / 2) / ((WIDTH - 1) / 2) / temperature
    inp = tf.keras.Input(shape=(HEIGHT, WIDTH, 1), batch_size=1)
    flat = tf.keras.layers.Reshape((HEIGHT * WIDTH,))(inp)
    probs = tf.keras.layers.Softmax()(flat)
    spatial = tf.keras.layers.Reshape((HEIGHT, WIDTH, 1))(probs)
    x_kernel = x_coords.reshape(HEIGHT, WIDTH, 1, 1)
    xc = tf.keras.layers.Conv2D(
        1, kernel_size=(HEIGHT, WIDTH), padding="valid",
        use_bias=False, trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(x_kernel),
    )(spatial)
    out = tf.keras.layers.Flatten()(xc)
    m_xonly = tf.keras.Model(inputs=inp, outputs=out)
    results["C3_x_only"] = _compile_and_compare(
        m_xonly, "x_only", images, run_hw,
        desc="X-only: softmax + x_coord Conv2D, single output"
    )

    # ---- Model C4: Y-only WITHOUT offset ----
    print("\n  --- C4: Y-only, no offset ---")
    y_coords = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for i in range(HEIGHT):
        y_coords[i, :] = (i - (HEIGHT - 1) / 2) / ((HEIGHT - 1) / 2) / temperature
    inp = tf.keras.Input(shape=(HEIGHT, WIDTH, 1), batch_size=1)
    flat = tf.keras.layers.Reshape((HEIGHT * WIDTH,))(inp)
    probs = tf.keras.layers.Softmax()(flat)
    spatial = tf.keras.layers.Reshape((HEIGHT, WIDTH, 1))(probs)
    y_kernel = y_coords.reshape(HEIGHT, WIDTH, 1, 1)
    yc = tf.keras.layers.Conv2D(
        1, kernel_size=(HEIGHT, WIDTH), padding="valid",
        use_bias=False, trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(y_kernel),
    )(spatial)
    out = tf.keras.layers.Flatten()(yc)
    m_yonly = tf.keras.Model(inputs=inp, outputs=out)
    results["C4_y_no_offset"] = _compile_and_compare(
        m_yonly, "y_no_offset", images, run_hw,
        desc="Y-only: softmax + y_coord Conv2D, NO offset, single output"
    )

    # ---- Model C5: Y-only WITH offset ----
    print("\n  --- C5: Y-only, with +10 offset ---")
    y_coords_offset = y_coords + (1.0 / temperature)
    inp = tf.keras.Input(shape=(HEIGHT, WIDTH, 1), batch_size=1)
    flat = tf.keras.layers.Reshape((HEIGHT * WIDTH,))(inp)
    probs = tf.keras.layers.Softmax()(flat)
    spatial = tf.keras.layers.Reshape((HEIGHT, WIDTH, 1))(probs)
    y_kernel_off = y_coords_offset.reshape(HEIGHT, WIDTH, 1, 1)
    yc = tf.keras.layers.Conv2D(
        1, kernel_size=(HEIGHT, WIDTH), padding="valid",
        use_bias=False, trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(y_kernel_off),
    )(spatial)
    out = tf.keras.layers.Flatten()(yc)
    m_yoff = tf.keras.Model(inputs=inp, outputs=out)
    results["C5_y_with_offset"] = _compile_and_compare(
        m_yoff, "y_with_offset", images, run_hw,
        desc="Y-only: softmax + y_coord Conv2D, WITH +10 offset, single output"
    )

    # ---- Model C6: Full model but X and Y use SAME coords (no offset) ----
    print("\n  --- C6: Both X+Y, same coords (no offset) ---")
    inp = tf.keras.Input(shape=(HEIGHT, WIDTH, 1), batch_size=1)
    flat = tf.keras.layers.Reshape((HEIGHT * WIDTH,))(inp)
    probs = tf.keras.layers.Softmax()(flat)
    spatial = tf.keras.layers.Reshape((HEIGHT, WIDTH, 1))(probs)
    x_kernel = x_coords.reshape(HEIGHT, WIDTH, 1, 1)
    y_kernel_no_off = y_coords.reshape(HEIGHT, WIDTH, 1, 1)
    xc = tf.keras.layers.Conv2D(
        1, kernel_size=(HEIGHT, WIDTH), padding="valid",
        use_bias=False, trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(x_kernel),
        name="x_sum",
    )(spatial)
    yc = tf.keras.layers.Conv2D(
        1, kernel_size=(HEIGHT, WIDTH), padding="valid",
        use_bias=False, trainable=False,
        kernel_initializer=tf.keras.initializers.Constant(y_kernel_no_off),
        name="y_sum",
    )(spatial)
    xf = tf.keras.layers.Flatten(name="x_flat")(xc)
    yf = tf.keras.layers.Flatten(name="y_flat")(yc)
    out = tf.keras.layers.Concatenate()([xf, yf])
    m_both_no_off = tf.keras.Model(inputs=inp, outputs=out)
    results["C6_both_no_offset"] = _compile_and_compare(
        m_both_no_off, "both_no_offset", images, run_hw,
        desc="Full model but NO Y offset: does compiler merge X and Y?"
    )

    return results


def _compile_and_compare(model, name, images, run_hw, desc=""):
    """Quantize, compile, run CPU and optionally Edge TPU, compare."""
    from tflite_runtime.interpreter import Interpreter

    print(f"  {desc}")
    print(f"  Output shape: {model.output_shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Quantize ---
        saved_path = os.path.join(tmpdir, f"{name}.tflite")
        _quantize_model(model, saved_path)

        # --- CPU baseline ---
        print(f"  CPU TFLite:")
        interp = Interpreter(model_path=saved_path)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        out_scale = out_det["quantization"][0]
        out_zp = out_det["quantization"][1]
        print(f"    out_scale={out_scale:.6f}, out_zp={out_zp}")

        cpu_results = []
        for img_name, img in images:
            interp.set_tensor(in_det["index"], img)
            interp.invoke()
            raw = interp.get_tensor(out_det["index"])[0]
            dequant = (raw.astype(np.float32) - out_zp) * out_scale
            cpu_results.append((img_name, raw.copy(), dequant.copy()))
            print(f"    {img_name:20s}  raw={_fmt_array(raw):30s}  dequant={_fmt_floats(dequant)}")

        # --- Compile for Edge TPU ---
        hw_results = []
        if run_hw:
            compiled_path = _compile_for_edgetpu(saved_path, tmpdir)
            if compiled_path:
                print(f"  Edge TPU:")
                sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))
                from libredgetpu.simple_invoker import SimpleInvoker

                with SimpleInvoker(compiled_path) as invoker:
                    for img_name, img in images:
                        raw_bytes = invoker.invoke_raw(img.flatten().tobytes())
                        n_out = cpu_results[0][1].size  # match CPU output size
                        raw = np.frombuffer(raw_bytes, dtype=np.int8)[:n_out]
                        dequant = (raw.astype(np.float32) - out_zp) * out_scale
                        hw_results.append((img_name, raw.copy(), dequant.copy()))
                        print(f"    {img_name:20s}  raw={_fmt_array(raw):30s}  dequant={_fmt_floats(dequant)}")

    return {"cpu": cpu_results, "hw": hw_results, "out_scale": out_scale, "out_zp": out_zp}


def _quantize_model(model, output_path):
    """Quantize Keras model to int8 TFLite."""
    import tensorflow as tf

    def representative_dataset():
        for _ in range(100):
            yield [np.random.randint(0, 256, size=(1, HEIGHT, WIDTH, 1)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)


def _compile_for_edgetpu(tflite_path, tmpdir):
    """Compile TFLite for Edge TPU, return path or None."""
    import shutil
    if not shutil.which("edgetpu_compiler"):
        print("  WARNING: edgetpu_compiler not found, skipping Edge TPU compilation")
        return None

    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", tmpdir, tflite_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: Compilation failed:\n{result.stderr}")
        return None

    # Print compiler log
    base = os.path.splitext(os.path.basename(tflite_path))[0]
    log_path = os.path.join(tmpdir, f"{base}_edgetpu.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line and "Edge TPU Compiler" not in line and "Input:" not in line and "Output:" not in line:
                    print(f"    {line}")

    compiled = os.path.join(tmpdir, f"{base}_edgetpu.tflite")
    return compiled if os.path.exists(compiled) else None


# ---------------------------------------------------------------------------
# Phase D: Linear fit
# ---------------------------------------------------------------------------

def fit_linear_transform(cpu_results, hw_results):
    """Fit hw_raw = a * cpu_raw + b per output element."""
    if not cpu_results or not hw_results:
        print("  Cannot fit: missing CPU or hardware results")
        return

    n_outputs = cpu_results[0][1].size

    for elem in range(n_outputs):
        cpu_vals = np.array([r[1].flatten()[elem] for r in cpu_results], dtype=np.float64)
        hw_vals = np.array([r[1].flatten()[elem] for r in hw_results], dtype=np.float64)

        # Least squares: hw = a * cpu + b
        n = len(cpu_vals)
        if n < 2:
            print(f"  Element {elem}: not enough data points")
            continue

        A = np.column_stack([cpu_vals, np.ones(n)])
        result, residuals, rank, sv = np.linalg.lstsq(A, hw_vals, rcond=None)
        a, b = result

        # R-squared
        ss_res = np.sum((hw_vals - (a * cpu_vals + b)) ** 2)
        ss_tot = np.sum((hw_vals - np.mean(hw_vals)) ** 2)
        r_sq = 1 - ss_res / max(ss_tot, 1e-12)

        elem_name = "X" if elem == 0 else "Y"
        print(f"  Element {elem} ({elem_name}): hw_raw = {a:+.4f} * cpu_raw + {b:+.4f}  (R²={r_sq:.6f})")

        # Print per-image details
        for i, (name, _, _) in enumerate(cpu_results):
            predicted = a * cpu_vals[i] + b
            actual = hw_vals[i]
            err = actual - predicted
            print(f"    {name:20s}  cpu={cpu_vals[i]:+7.1f}  hw={actual:+7.1f}  "
                  f"predicted={predicted:+7.1f}  err={err:+5.1f}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_array(arr):
    """Format numpy array as compact string."""
    vals = arr.flatten()
    return "[" + ", ".join(f"{v:+4d}" for v in vals) + "]"


def _fmt_floats(arr):
    """Format float array as compact string."""
    vals = arr.flatten()
    return "[" + ", ".join(f"{v:+8.3f}" for v in vals) + "]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SpotTracker CPU vs Edge TPU experiment")
    parser.add_argument("--hardware", action="store_true", help="Run Edge TPU hardware tests")
    parser.add_argument("--isolate", action="store_true", help="Run isolated op model tests (needs TF)")
    args = parser.parse_args()

    images = create_test_images()

    # =========================================================
    # Phase A: CPU baseline with existing 16x16 model
    # =========================================================
    print("=" * 70)
    print("PHASE A: CPU TFLite baseline (existing bright_16x16 model)")
    print("=" * 70)

    # Compute float32 expected values first
    print("\n  Float32 expected (no quantization):")
    for name, img in images:
        x_exp, y_exp = compute_expected_soft_argmax(img)
        print(f"    {name:20s}  x={x_exp:+8.3f}  y_offset={y_exp:+8.3f}  "
              f"(y_raw={y_exp - 10:+8.3f})")

    print(f"\n  CPU TFLite ({UNCOMPILED}):")
    cpu_results = run_cpu_tflite(images)
    if cpu_results:
        print("\n  Summary:")
        for name, raw, dequant in cpu_results:
            print(f"    {name:20s}  raw={_fmt_array(raw):30s}  dequant={_fmt_floats(dequant)}")

    # =========================================================
    # Phase B: Edge TPU comparison
    # =========================================================
    hw_results = []
    if args.hardware:
        print("\n" + "=" * 70)
        print("PHASE B: Edge TPU hardware (existing bright_16x16 model)")
        print("=" * 70)

        hw_results = run_edge_tpu(images)
        if hw_results:
            print("\n  Summary:")
            for name, raw, dequant in hw_results:
                print(f"    {name:20s}  raw={_fmt_array(raw):30s}  dequant={_fmt_floats(dequant)}")

        # Side-by-side comparison
        if cpu_results and hw_results:
            print("\n  Side-by-side (CPU vs HW):")
            print(f"    {'Image':20s}  {'CPU_raw':>15s}  {'HW_raw':>15s}  {'Match?':>7s}")
            for (cn, cr, _), (hn, hr, _) in zip(cpu_results, hw_results):
                match = "YES" if np.array_equal(cr, hr) else "NO"
                print(f"    {cn:20s}  {_fmt_array(cr):>15s}  {_fmt_array(hr):>15s}  {match:>7s}")

    # =========================================================
    # Phase C: Isolated models
    # =========================================================
    if args.isolate:
        print("\n" + "=" * 70)
        print("PHASE C: Isolated model experiments")
        print("=" * 70)

        iso_results = create_and_test_isolated_models(images, run_hw=args.hardware)

        # Per-model linear fits if we have hardware data
        if args.hardware:
            for model_name, data in iso_results.items():
                if data["hw"]:
                    print(f"\n  Linear fit for {model_name}:")
                    fit_linear_transform(data["cpu"], data["hw"])

    # =========================================================
    # Phase D: Linear fit on existing model
    # =========================================================
    if cpu_results and hw_results:
        print("\n" + "=" * 70)
        print("PHASE D: Linear fit (existing model: hw_raw = a * cpu_raw + b)")
        print("=" * 70)
        fit_linear_transform(cpu_results, hw_results)

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    if not args.hardware:
        print("  (Run with --hardware to compare against Edge TPU)")
    if not args.isolate:
        print("  (Run with --isolate to test isolated op models)")


if __name__ == "__main__":
    main()
