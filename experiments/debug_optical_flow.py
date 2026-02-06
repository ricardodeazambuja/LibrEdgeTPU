"""Debug optical flow: compare Edge TPU vs CPU feature extraction and correlation.

This script:
1. Generates synthetic test images with known displacement
2. Extracts Gabor features using both Edge TPU and CPU reference
3. Compares feature maps (correlation, mean, std, extrema)
4. Computes optical flow using both backends
5. Reports discrepancies to identify the bug

Usage:
    python experiments/debug_optical_flow.py
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libredgetpu import OpticalFlow
from libredgetpu.tflite_builder import _generate_gabor_kernels
from libredgetpu.tflite_parser import parse_full
from libredgetpu._quantize import quantize_uint8, dequantize
import scipy.ndimage


def generate_gabor_features_cpu(image: np.ndarray, ksize: int = 7) -> np.ndarray:
    """CPU reference: apply 8 Gabor filters (4 orientations × 2 scales).

    Matches the Edge TPU model exactly:
    - 8 filters: orientations [0°, 45°, 90°, 135°] × wavelengths [4.0, 8.0]
    - 7×7 kernel size
    - ReLU activation (clip negative to zero)
    - SAME padding (zero-pad edges)

    Args:
        image: Grayscale uint8 image [H, W]
        ksize: Gabor kernel size (default 7)

    Returns:
        Features [H, W, 8] as float32
    """
    h, w = image.shape
    kernels = _generate_gabor_kernels(ksize=ksize)  # [7, 7, 1, 8]

    features = np.zeros((h, w, 8), dtype=np.float32)
    image_f = image.astype(np.float32)

    for i in range(8):
        kernel = kernels[:, :, 0, i]  # [7, 7]
        # scipy.ndimage.convolve uses SAME padding by default (mode='constant', cval=0)
        conv = scipy.ndimage.convolve(image_f, kernel, mode='constant', cval=0.0)
        # ReLU
        conv = np.maximum(conv, 0.0)
        features[:, :, i] = conv

    return features


def create_shifted_pattern(h: int, w: int, dx: int = 0, dy: int = 0) -> tuple:
    """Create two frames with a known displacement.

    Frame 1: Random noise pattern
    Frame 2: Frame 1 shifted by (dx, dy) pixels

    Args:
        h: Height
        w: Width
        dx: Horizontal shift (pixels, positive = rightward)
        dy: Vertical shift (pixels, positive = downward)

    Returns:
        (frame1, frame2) both uint8 [H, W]
    """
    # Create a random pattern with clear features
    np.random.seed(42)
    frame1 = (np.random.rand(h, w) * 255).astype(np.uint8)

    # Shift frame1 by (dx, dy) to create frame2
    frame2 = np.zeros_like(frame1)

    # Compute valid source and destination regions (handle boundary)
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)

    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)

    frame2[dst_y1:dst_y2, dst_x1:dst_x2] = frame1[src_y1:src_y2, src_x1:src_x2]

    return frame1, frame2


def compare_features(feat_tpu: np.ndarray, feat_cpu: np.ndarray, name: str):
    """Compare two feature maps and report statistics.

    Args:
        feat_tpu: Edge TPU features [H, W, C]
        feat_cpu: CPU reference features [H, W, C]
        name: Descriptive name for reporting
    """
    print(f"\n{'='*60}")
    print(f"Feature Comparison: {name}")
    print(f"{'='*60}")
    print(f"TPU shape: {feat_tpu.shape}, CPU shape: {feat_cpu.shape}")
    print(f"TPU dtype: {feat_tpu.dtype}, CPU dtype: {feat_cpu.dtype}")
    print(f"TPU range: [{feat_tpu.min():.3f}, {feat_tpu.max():.3f}]")
    print(f"CPU range: [{feat_cpu.min():.3f}, {feat_cpu.max():.3f}]")
    print(f"TPU mean/std: {feat_tpu.mean():.3f} / {feat_tpu.std():.3f}")
    print(f"CPU mean/std: {feat_cpu.mean():.3f} / {feat_cpu.std():.3f}")

    # Correlation (flatten)
    tpu_flat = feat_tpu.ravel()
    cpu_flat = feat_cpu.ravel()
    corr = np.corrcoef(tpu_flat, cpu_flat)[0, 1]
    print(f"Pearson correlation: {corr:.6f}")

    # MSE and RMSE
    mse = np.mean((feat_tpu - feat_cpu) ** 2)
    rmse = np.sqrt(mse)
    print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")

    # Per-channel stats
    print("\nPer-channel correlation:")
    for c in range(feat_tpu.shape[2]):
        tpu_c = feat_tpu[:, :, c].ravel()
        cpu_c = feat_cpu[:, :, c].ravel()
        corr_c = np.corrcoef(tpu_c, cpu_c)[0, 1]
        print(f"  Channel {c}: {corr_c:.6f}")


def debug_quantization(flow: OpticalFlow):
    """Inspect quantization parameters from the model.

    Args:
        flow: OpticalFlow instance (opened)
    """
    print(f"\n{'='*60}")
    print("Quantization Parameters")
    print(f"{'='*60}")

    # Input quantization
    input_info = flow._input_info
    print(f"Input tensor:")
    print(f"  Shape: {input_info.shape}")
    print(f"  Scale: {input_info.scale}")
    print(f"  Zero point: {input_info.zero_point}")
    print(f"  Range: [{input_info.zero_point * input_info.scale:.3f}, "
          f"{(255 + input_info.zero_point) * input_info.scale:.3f}]")

    # Output quantization
    output_info = flow._output_info
    print(f"\nOutput tensor:")
    print(f"  Shape: {output_info.shape}")
    print(f"  Scale: {output_info.scale}")
    print(f"  Zero point: {output_info.zero_point}")
    print(f"  Range: [{output_info.zero_point * output_info.scale:.3f}, "
          f"{(255 + output_info.zero_point) * output_info.scale:.3f}]")


def test_known_displacement(flow: OpticalFlow, dx_true: int, dy_true: int):
    """Test optical flow with a known ground-truth displacement.

    Args:
        flow: OpticalFlow instance (opened)
        dx_true: True horizontal displacement (pixels)
        dy_true: True vertical displacement (pixels)
    """
    h, w = flow.height, flow.width
    print(f"\n{'='*60}")
    print(f"Test: Known Displacement dx={dx_true}, dy={dy_true}")
    print(f"{'='*60}")

    # Create shifted pattern
    frame1, frame2 = create_shifted_pattern(h, w, dx_true, dy_true)

    # Extract features via Edge TPU
    feat1_tpu = flow.extract_features(frame1)
    feat2_tpu = flow.extract_features(frame2)

    # Extract features via CPU reference
    feat1_cpu_full = generate_gabor_features_cpu(frame1, ksize=7)
    feat2_cpu_full = generate_gabor_features_cpu(frame2, ksize=7)

    # If using fused pooling, pool the CPU features for comparison
    if flow.fused_pool:
        p = flow.fused_pool
        ph, pw = h // p, w // p
        feat1_cpu = feat1_cpu_full.reshape(ph, p, pw, p, 8).mean(axis=(1, 3))
        feat2_cpu = feat2_cpu_full.reshape(ph, p, pw, p, 8).mean(axis=(1, 3))
    else:
        feat1_cpu = feat1_cpu_full
        feat2_cpu = feat2_cpu_full

    # Compare features for frame1
    compare_features(feat1_tpu, feat1_cpu, f"Frame1 (dx={dx_true}, dy={dy_true})")

    # Compare features for frame2
    compare_features(feat2_tpu, feat2_cpu, f"Frame2 (dx={dx_true}, dy={dy_true})")

    # Compute optical flow using Edge TPU
    vx_tpu, vy_tpu = flow.compute(frame1, frame2)
    print(f"\nEdge TPU optical flow: vx={vx_tpu:.3f}, vy={vy_tpu:.3f}")

    # Compute optical flow using CPU reference (full pipeline)
    # Use the full-resolution CPU features and pool if needed
    if flow.fused_pool:
        # Pool the full-resolution features
        p = flow.fused_pool
        ph, pw = h // p, w // p
        feat1_pooled = feat1_cpu_full.reshape(ph, p, pw, p, 8).mean(axis=(1, 3))
        feat2_pooled = feat2_cpu_full.reshape(ph, p, pw, p, 8).mean(axis=(1, 3))
    else:
        # Apply CPU pooling with pool_factor
        p = flow.pool_factor
        ph, pw = h // p, w // p
        feat1_pooled = feat1_cpu_full.reshape(ph, p, pw, p, 8).mean(axis=(1, 3))
        feat2_pooled = feat2_cpu_full.reshape(ph, p, pw, p, 8).mean(axis=(1, 3))

    # Compute correlation (reuse flow's method)
    corr = flow._global_correlation(feat1_pooled, feat2_pooled)
    vx_cpu, vy_cpu = flow._soft_argmax(corr)
    print(f"CPU reference optical flow: vx={vx_cpu:.3f}, vy={vy_cpu:.3f}")

    # Ground truth (in pooled pixels)
    pool_factor = flow.fused_pool if flow.fused_pool else flow.pool_factor
    vx_true_pooled = dx_true / pool_factor
    vy_true_pooled = dy_true / pool_factor
    print(f"Ground truth (pooled pixels): vx={vx_true_pooled:.3f}, vy={vy_true_pooled:.3f}")

    # Errors
    err_tpu_x = abs(vx_tpu - vx_true_pooled)
    err_tpu_y = abs(vy_tpu - vy_true_pooled)
    err_cpu_x = abs(vx_cpu - vx_true_pooled)
    err_cpu_y = abs(vy_cpu - vy_true_pooled)

    print(f"TPU error: |Δvx|={err_tpu_x:.3f}, |Δvy|={err_tpu_y:.3f}")
    print(f"CPU error: |Δvx|={err_cpu_x:.3f}, |Δvy|={err_cpu_y:.3f}")

    if err_tpu_x > 0.5 or err_tpu_y > 0.5:
        print("\n⚠️  WARNING: Edge TPU optical flow has large error!")
    else:
        print("\n✓ Edge TPU optical flow matches ground truth")


def inspect_model_structure(tflite_path: str):
    """Parse TFLite model and report structure.

    Args:
        tflite_path: Path to _edgetpu.tflite file
    """
    print(f"\n{'='*60}")
    print("Model Structure")
    print(f"{'='*60}")

    with open(tflite_path, 'rb') as f:
        tflite_bytes = f.read()
    model = parse_full(tflite_bytes)

    print(f"Model: {tflite_path}")
    print(f"Tensors: {len(model.tensors)}")
    print(f"Operators: {len(model.operators)}")
    print(f"Graph inputs: {model.graph_inputs}")
    print(f"Graph outputs: {model.graph_outputs}")

    # Print operator sequence
    for i, op in enumerate(model.operators):
        # Get opcode name from op.opcode_name (stored directly in OperatorInfo)
        opcode_name = op.opcode_name

        # Get input/output tensor shapes
        input_shapes = [list(model.tensors[idx].shape) for idx in op.inputs if idx < len(model.tensors)]
        output_shapes = [list(model.tensors[idx].shape) for idx in op.outputs if idx < len(model.tensors)]

        print(f"  Op {i}: {opcode_name}")
        print(f"    Inputs: {input_shapes}")
        print(f"    Outputs: {output_shapes}")


def main():
    """Run full diagnostic suite."""
    print("="*60)
    print("OpticalFlow Debug Suite")
    print("="*60)

    # Parameters
    size = 64
    pooled = True  # Use pooled mode (matches GUI)

    # Check hardware availability
    try:
        import usb.core
        dev = usb.core.find(idVendor=0x18d1, idProduct=0x9302)
        if dev is None:
            dev = usb.core.find(idVendor=0x1a6e, idProduct=0x089a)
        if dev is None:
            print("\n⚠️  No Edge TPU detected. This script requires hardware.\n")
            return
    except Exception as e:
        print(f"\n⚠️  USB check failed: {e}\n")
        return

    # Create OpticalFlow instance
    print(f"\nInitializing OpticalFlow (size={size}, pooled={pooled})...")
    try:
        flow = OpticalFlow.from_template(size, pooled=pooled)
        flow.open()
    except Exception as e:
        print(f"\n❌ Failed to initialize OpticalFlow: {e}\n")
        import traceback
        traceback.print_exc()
        return

    try:
        # Inspect model structure
        from libredgetpu.optical_flow.templates import get_pooled_template, get_template
        if pooled:
            tflite_path, _ = get_pooled_template(size, pool_factor=4)
        else:
            tflite_path, _ = get_template(size)
        inspect_model_structure(tflite_path)

        # Debug quantization
        debug_quantization(flow)

        # Test suite: various known displacements
        test_cases = [
            (0, 0),    # No motion
            (4, 0),    # Right 4 pixels
            (-4, 0),   # Left 4 pixels
            (0, 4),    # Down 4 pixels
            (0, -4),   # Up 4 pixels
            (2, 2),    # Diagonal
            (-2, -2),  # Diagonal (opposite)
        ]

        for dx, dy in test_cases:
            test_known_displacement(flow, dx, dy)

        print(f"\n{'='*60}")
        print("Debug Suite Complete")
        print(f"{'='*60}\n")

    finally:
        flow.close()


if __name__ == "__main__":
    main()
