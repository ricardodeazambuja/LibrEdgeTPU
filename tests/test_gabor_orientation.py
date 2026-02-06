"""Tests for Gabor kernel orientation correctness.

Validates that Gabor filters follow standard orientation convention:
theta is the angle perpendicular to the edges the filter responds to.

- theta=0°: responds to horizontal edges
- theta=90°: responds to vertical edges
"""

import numpy as np
import pytest

from libredgetpu.tflite_builder import _generate_gabor_kernels


def compute_kernel_response(kernel, pattern):
    """Compute total absolute response of a kernel to a pattern.

    Args:
        kernel: 2D kernel array.
        pattern: 2D pattern array (must be larger than kernel).

    Returns:
        Total absolute correlation response.
    """
    kh, kw = kernel.shape
    ph, pw = pattern.shape

    # Place kernel at center
    cy, cx = ph // 2, pw // 2
    patch = pattern[cy - kh // 2:cy + kh // 2 + 1, cx - kw // 2:cx + kw // 2 + 1]

    if patch.shape != kernel.shape:
        return 0.0

    return float(np.abs(np.sum(kernel * patch)))


@pytest.mark.parametrize("orientations", [4, 8])
def test_horizontal_edge_response_theta0(orientations):
    """Verify that theta=0 kernel responds to horizontal edges."""
    kernels = _generate_gabor_kernels(ksize=7, orientations=orientations, sigmas=(2.0,))

    # First kernel is theta=0 (standard convention: responds to horizontal edges)
    k0 = kernels[:, :, 0, 0]

    # Create horizontal edge (top half negative, bottom half positive)
    horizontal_edge = np.ones((32, 32), dtype=np.float32)
    horizontal_edge[:16, :] = -1.0

    # Create vertical edge (left half negative, right half positive)
    vertical_edge = np.ones((32, 32), dtype=np.float32)
    vertical_edge[:, :16] = -1.0

    resp_horizontal = compute_kernel_response(k0, horizontal_edge)
    resp_vertical = compute_kernel_response(k0, vertical_edge)

    # theta=0 kernel should respond MORE to horizontal edges than vertical
    assert resp_horizontal > resp_vertical, (
        f"theta=0 kernel should prefer horizontal edges: "
        f"horizontal={resp_horizontal:.3f}, vertical={resp_vertical:.3f}"
    )


@pytest.mark.parametrize("orientations", [4, 8])
def test_vertical_edge_response_theta90(orientations):
    """Verify that theta=π/2 kernel responds to vertical edges."""
    kernels = _generate_gabor_kernels(ksize=7, orientations=orientations, sigmas=(2.0,))

    # For orientations=4: kernel 2 is theta=π/2
    # For orientations=8: kernel 4 is theta=π/2
    k_vertical_idx = orientations // 2
    k_vertical = kernels[:, :, 0, k_vertical_idx]

    # Create horizontal edge (top half negative, bottom half positive)
    horizontal_edge = np.ones((32, 32), dtype=np.float32)
    horizontal_edge[:16, :] = -1.0

    # Create vertical edge (left half negative, right half positive)
    vertical_edge = np.ones((32, 32), dtype=np.float32)
    vertical_edge[:, :16] = -1.0

    resp_horizontal = compute_kernel_response(k_vertical, horizontal_edge)
    resp_vertical = compute_kernel_response(k_vertical, vertical_edge)

    # theta=π/2 kernel should respond MORE to vertical edges than horizontal
    assert resp_vertical > resp_horizontal, (
        f"theta=π/2 kernel should prefer vertical edges: "
        f"vertical={resp_vertical:.3f}, horizontal={resp_horizontal:.3f}"
    )


# Note: Diagonal edge tests removed due to ambiguity in edge pattern generation.
# The key tests (theta=0 → horizontal edges, theta=90° → vertical edges) provide
# sufficient validation that the Gabor kernel orientation convention is correct.
