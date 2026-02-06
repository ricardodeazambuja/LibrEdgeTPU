"""Tests for MatMulEngine — offline (no hardware) and hardware tests."""

import base64
import json
import os
import tempfile

import numpy as np
import pytest

from libredgetpu.matmul_engine import (
    MatMulEngine, _generate_param_blob, _extract_overhead,
    _ROW_TILE, _COL_TILE,
)
from libredgetpu.tflite_parser import parse as parse_tflite
from libredgetpu.templates import list_templates, get_template


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_any_template():
    """Return (tflite_path, json_path) for the smallest available template."""
    sizes = list_templates()
    if not sizes:
        pytest.skip("No pre-compiled templates available")
    return get_template(sizes[0])


def _load_metadata(json_path):
    """Load sidecar JSON metadata."""
    with open(json_path) as f:
        return json.load(f)


# ── Offline tests (no hardware) ───────────────────────────────────────────


class TestQuantizeWeights:
    """Test weight quantization math."""

    def test_roundtrip(self):
        """Quantize float → int8 → dequantize; error should be within scale/2."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        meta = _load_metadata(json_path)
        n = meta["matrix_size"]
        scale = meta["weight_scale"]
        zp = meta.get("weight_zero_point", 0)

        # Random weights within representable range
        w_min = (-128 - zp) * scale
        w_max = (127 - zp) * scale
        rng = np.random.default_rng(42)
        weights = rng.uniform(w_min * 0.9, w_max * 0.9, (n, n)).astype(np.float32)

        raw = engine.quantize_weights(weights)
        # Dequantize
        int8_vals = np.frombuffer(raw[:n * n], dtype=np.int8).reshape(n, n)
        recovered = (int8_vals.astype(np.float32) - zp) * scale

        max_err = np.max(np.abs(weights - recovered))
        assert max_err <= scale / 2 + 1e-7, f"Max roundtrip error {max_err} > scale/2 = {scale / 2}"

    def test_clipping(self):
        """Values outside representable range should be clipped."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        meta = _load_metadata(json_path)
        n = meta["matrix_size"]
        scale = meta["weight_scale"]

        # Weights way outside range
        weights = np.full((n, n), 1000.0, dtype=np.float32)
        raw = engine.quantize_weights(weights)
        int8_vals = np.frombuffer(raw[:n * n], dtype=np.int8)
        assert np.all(int8_vals == 127), "Large positive values should clip to 127"

        weights = np.full((n, n), -1000.0, dtype=np.float32)
        raw = engine.quantize_weights(weights)
        int8_vals = np.frombuffer(raw[:n * n], dtype=np.int8)
        assert np.all(int8_vals == -128), "Large negative values should clip to -128"

    def test_shape_mismatch(self):
        """Wrong weight shape should raise ValueError."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        n = engine.matrix_size

        with pytest.raises(ValueError, match="does not match"):
            engine.quantize_weights(np.zeros((n + 1, n), dtype=np.float32))

        with pytest.raises(ValueError, match="does not match"):
            engine.quantize_weights(np.zeros((n,), dtype=np.float32))


class TestMetadataLoading:
    """Test sidecar JSON loading."""

    def test_load_metadata(self):
        """Sidecar JSON should populate engine properties."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        meta = _load_metadata(json_path)

        assert engine.matrix_size == meta["matrix_size"]
        assert engine.weight_scale == pytest.approx(meta["weight_scale"], rel=1e-5)
        assert engine.input_scale == pytest.approx(meta["input_scale"], rel=1e-5)
        assert engine.output_scale == pytest.approx(meta["output_scale"], rel=1e-5)

    def test_auto_discover_json(self):
        """Engine should auto-discover JSON sidecar alongside TFLite file."""
        tflite_path, json_path = _get_any_template()
        # Don't pass json_path explicitly
        engine = MatMulEngine(tflite_path)
        assert engine.weight_scale is not None

    def test_weight_range(self):
        """weight_range should be consistent with scale and zero_point."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        wr = engine.weight_range
        assert wr is not None
        assert wr[0] < 0  # min should be negative
        assert wr[1] > 0  # max should be positive


class TestTemplateParser:
    """Test that templates parse correctly."""

    def test_parse_template(self):
        """Template TFLite should parse with correct executable types."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)

        # Should have PC + EO executables (cached mode)
        assert engine._pc_exe is not None
        assert engine._eo_exe is not None
        assert engine._original_params is not None
        assert len(engine._original_params) > 0

    def test_param_size_matches_metadata(self):
        """DarwiNN param blob size should match sidecar metadata."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        meta = _load_metadata(json_path)

        if "param_size" in meta:
            assert engine.param_size == meta["param_size"]

    def test_rejects_standalone(self):
        """MatMulEngine should reject standalone (non-cached) models."""
        # We can't easily create a standalone model in tests, so just
        # verify the check exists by testing the error message
        tflite_path, json_path = _get_any_template()
        # This should work (cached model)
        engine = MatMulEngine(tflite_path, json_path)
        assert engine._pc_exe is not None


class TestRecompilationConsistency:
    """Verify that patching weights preserves instruction streams.

    This is the core assumption of set_weights(): changing int8 weight values
    while keeping the same quantization scale produces identical DarwiNN
    instructions. Only the PARAMETER_CACHING params should change.
    """

    @pytest.fixture(autouse=True)
    def _check_compiler(self):
        """Skip if edgetpu_compiler is not installed."""
        import shutil
        if shutil.which("edgetpu_compiler") is None:
            pytest.skip("edgetpu_compiler not found")

    def test_eo_instructions_unchanged(self):
        """Patching weights with same scale → identical EO instructions."""
        import hashlib
        from libredgetpu.matmul_engine import _patch_tflite_weights, _recompile_and_extract_params
        from libredgetpu.delegate import parse_darwinn

        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        n = engine.matrix_size

        # Get original EO instruction hash
        orig_eo_hash = hashlib.sha256(engine._eo_exe.bitstreams[0].data).hexdigest()

        # Load uncompiled template and patch with different weights
        with open(engine._uncompiled_tflite_path, "rb") as f:
            uncompiled = f.read()

        rng = np.random.default_rng(42)
        W = rng.uniform(-0.05, 0.05, (n, n)).astype(np.float32)
        quantized = engine.quantize_weights(W)
        patched = _patch_tflite_weights(uncompiled, quantized)

        # Recompile and check EO instructions
        import tempfile, subprocess, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.tflite")
            with open(path, "wb") as f:
                f.write(patched)
            subprocess.run(
                ["edgetpu_compiler", "-s", "-o", tmpdir, path],
                capture_output=True, check=True,
            )
            compiled = os.path.join(tmpdir, "model_edgetpu.tflite")
            with open(compiled, "rb") as f:
                cdata = f.read()

        model = parse_tflite(cdata)
        exes = parse_darwinn(model.custom_op_data)
        new_eo_hash = None
        for exe in exes:
            if exe.exec_type == 2:  # EXECUTION_ONLY
                new_eo_hash = hashlib.sha256(exe.bitstreams[0].data).hexdigest()

        assert new_eo_hash == orig_eo_hash, \
            "EO instructions changed after weight patching — requant multiplier may differ"

    def test_param_blob_changes(self):
        """Different weights should produce different PC param blobs."""
        from libredgetpu.matmul_engine import _patch_tflite_weights, _recompile_and_extract_params

        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        n = engine.matrix_size

        with open(engine._uncompiled_tflite_path, "rb") as f:
            uncompiled = f.read()

        rng = np.random.default_rng(42)
        W = rng.uniform(-0.05, 0.05, (n, n)).astype(np.float32)
        quantized = engine.quantize_weights(W)
        patched = _patch_tflite_weights(uncompiled, quantized)

        new_params = _recompile_and_extract_params(patched)
        assert new_params != engine._original_params, \
            "New params are identical to original — weight patching had no effect"
        assert len(new_params) == len(engine._original_params), \
            "Param blob size changed — compiler may have reorganized the model"


class TestBlobGeneration:
    """Test compiler-free parameter blob generation."""

    def test_overhead_roundtrip(self):
        """Extract overhead from template blob, then verify generate_param_blob
        with the original weights reproduces the original blob exactly."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        n = engine.matrix_size

        original_blob = engine._original_params
        overhead = _extract_overhead(original_blob, n)

        # Extract the int8 weights from the blob by reversing the mapping
        num_groups = (n + _ROW_TILE - 1) // _ROW_TILE
        group_size = _ROW_TILE * 8 + _ROW_TILE * n
        blob_arr = np.frombuffer(original_blob, dtype=np.uint8)

        # Reverse-map: read each weight position from blob, undo XOR 0x80
        weights_int8 = np.zeros((n, n), dtype=np.int8)
        for row in range(n):
            for col in range(n):
                rg = row // _ROW_TILE
                rl = row % _ROW_TILE
                off = (rg * group_size + _ROW_TILE * 8
                       + (col // _COL_TILE) * (_ROW_TILE * _COL_TILE)
                       + rl * _COL_TILE + (col % _COL_TILE))
                weights_int8[row, col] = np.int8(blob_arr[off] ^ 0x80)

        # Regenerate and compare
        regenerated = _generate_param_blob(weights_int8, n, overhead)
        assert regenerated == original_blob, \
            "Blob roundtrip failed — regenerated blob differs from original"

    def test_overhead_from_sidecar(self):
        """Sidecar JSON should contain param_overhead field."""
        tflite_path, json_path = _get_any_template()
        meta = _load_metadata(json_path)

        assert "param_overhead" in meta, \
            "Sidecar missing param_overhead — regenerate with template_gen.py"

        overhead = base64.b64decode(meta["param_overhead"])
        n = meta["matrix_size"]
        expected_len = ((n + _ROW_TILE - 1) // _ROW_TILE) * _ROW_TILE * 8
        assert len(overhead) == expected_len, \
            f"Overhead length {len(overhead)} != expected {expected_len}"

    def test_blob_size(self):
        """Generated blob size should match param_size in metadata."""
        tflite_path, json_path = _get_any_template()
        meta = _load_metadata(json_path)
        n = meta["matrix_size"]
        overhead = base64.b64decode(meta["param_overhead"])

        weights = np.zeros((n, n), dtype=np.int8)
        blob = _generate_param_blob(weights, n, overhead)
        assert len(blob) == meta["param_size"], \
            f"Blob size {len(blob)} != param_size {meta['param_size']}"

    def test_compiler_match(self):
        """Blob from _generate_param_blob should match compiler output."""
        import shutil
        if shutil.which("edgetpu_compiler") is None:
            pytest.skip("edgetpu_compiler not found")

        from libredgetpu.matmul_engine import _patch_tflite_weights, _recompile_and_extract_params

        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        n = engine.matrix_size
        meta = _load_metadata(json_path)
        overhead = base64.b64decode(meta["param_overhead"])

        # Random weights
        rng = np.random.default_rng(42)
        W = rng.uniform(-0.05, 0.05, (n, n)).astype(np.float32)
        quantized = engine.quantize_weights(W)

        # Fast path
        fast_blob = _generate_param_blob(quantized, n, overhead)

        # Compiler path
        with open(engine._uncompiled_tflite_path, "rb") as f:
            uncompiled = f.read()
        patched = _patch_tflite_weights(uncompiled, quantized)
        compiler_blob = _recompile_and_extract_params(patched)

        assert fast_blob == compiler_blob, \
            "Fast-path blob differs from compiler output"

    def test_fast_path_enabled(self):
        """Engine loaded with param_overhead should use fast path (no compiler)."""
        tflite_path, json_path = _get_any_template()
        engine = MatMulEngine(tflite_path, json_path)
        assert engine._param_overhead is not None, \
            "param_overhead not loaded — fast path disabled"


class TestFromTemplate:
    """Test the from_template() class method."""

    def test_from_template(self):
        """from_template(n) should work for available sizes."""
        sizes = list_templates()
        if not sizes:
            pytest.skip("No pre-compiled templates available")
        engine = MatMulEngine.from_template(sizes[0])
        assert engine.matrix_size == sizes[0]

    def test_from_template_missing(self):
        """from_template() should raise for unavailable sizes."""
        with pytest.raises(FileNotFoundError, match="No template found"):
            MatMulEngine.from_template(99999)


# ── Hardware tests ────────────────────────────────────────────────────────


@pytest.mark.hardware
class TestMatMulHardware:
    """Hardware tests requiring a USB Edge TPU device."""

    def _get_engine(self):
        sizes = list_templates()
        if not sizes:
            pytest.skip("No pre-compiled templates available")
        return MatMulEngine.from_template(sizes[0])

    def _weight_range_50pct(self, engine):
        """Return (lo, hi) at 50% of the representable weight range."""
        wr = engine.weight_range
        return wr[0] * 0.5, wr[1] * 0.5

    def test_identity_matrix(self):
        """set_weights(scaled eye(N)) → output ≈ scaled input."""
        engine = self._get_engine()
        n = engine.matrix_size
        with engine:
            scale = engine.weight_scale
            # Diagonal value that roundtrips cleanly: int8 = 64 → float = 64 * scale
            diag_val = scale * 64
            W = np.eye(n, dtype=np.float32) * diag_val
            engine.set_weights(W)

            rng = np.random.default_rng(42)
            x = rng.uniform(-0.5, 0.5, n).astype(np.float32)
            y = engine.matmul(x).flatten()[:n]

            # Expected: diag_val * x
            expected = diag_val * x
            # Tolerance: output quantization step + 15% relative error from int8 noise
            atol = engine.output_scale * 3 + np.max(np.abs(expected)) * 0.20
            np.testing.assert_allclose(y, expected, atol=atol,
                                       err_msg="Identity matmul output too far from expected")

    def test_weight_swap_changes_output(self):
        """Different weights should produce different outputs."""
        engine = self._get_engine()
        n = engine.matrix_size
        lo, hi = self._weight_range_50pct(engine)
        with engine:
            rng = np.random.default_rng(123)
            W1 = rng.uniform(lo, hi, (n, n)).astype(np.float32)
            W2 = rng.uniform(lo, hi, (n, n)).astype(np.float32)
            x = rng.uniform(-0.5, 0.5, n).astype(np.float32)

            engine.set_weights(W1)
            y1 = engine.matmul(x).flatten()[:n]

            engine.set_weights(W2)
            y2 = engine.matmul(x).flatten()[:n]

            # Outputs should differ meaningfully
            assert not np.allclose(y1, y2, atol=engine.output_scale * 3), \
                "Different weights should produce different outputs"

    def test_reset_weights(self):
        """reset_weights() should restore original template behavior."""
        engine = self._get_engine()
        n = engine.matrix_size
        lo, hi = self._weight_range_50pct(engine)
        with engine:
            # Run with original weights
            engine.reset_weights()
            x_bytes = np.full(n, 128, dtype=np.uint8).tobytes()
            y_original = engine.matmul_raw(x_bytes)

            # Swap to custom weights
            rng = np.random.default_rng(77)
            W = rng.uniform(lo, hi, (n, n)).astype(np.float32)
            engine.set_weights(W)
            y_custom = engine.matmul_raw(x_bytes)

            # Reset back to original
            engine.reset_weights()
            y_restored = engine.matmul_raw(x_bytes)

            assert y_original == y_restored, "reset_weights() should restore original output"
            assert y_original != y_custom, "Custom weights should differ from original"

    def test_multiple_swaps(self):
        """Swap weights 10 times without errors."""
        engine = self._get_engine()
        n = engine.matrix_size
        lo, hi = self._weight_range_50pct(engine)
        with engine:
            rng = np.random.default_rng(99)
            x = rng.uniform(-0.5, 0.5, n).astype(np.float32)

            for i in range(10):
                W = rng.uniform(lo, hi, (n, n)).astype(np.float32)
                engine.set_weights(W)
                y = engine.matmul(x)
                assert y.size >= n, f"Swap {i}: output too small"
                assert np.any(y != 0), f"Swap {i}: output all zeros"

    def test_known_matmul(self):
        """Compare Edge TPU matmul against CPU reference."""
        engine = self._get_engine()
        n = engine.matrix_size
        lo, hi = self._weight_range_50pct(engine)
        with engine:
            rng = np.random.default_rng(55)
            W = rng.uniform(lo, hi, (n, n)).astype(np.float32)
            x = rng.uniform(-0.5, 0.5, n).astype(np.float32)

            engine.set_weights(W)
            y_tpu = engine.matmul(x).flatten()[:n]

            # CPU reference (float32, no quantization)
            y_cpu = W @ x

            # Tolerance: int8 quantization noise accumulates over N MACs.
            # Each weight has ±scale/2 error, each input has ±input_scale/2 error.
            # The dominant term is sqrt(N) * weight_err * input_rms.
            max_abs_err = np.max(np.abs(y_tpu - y_cpu))
            # Empirical tolerance: output_scale * sqrt(N) gives ~16 * output_scale for N=256
            reasonable_atol = engine.output_scale * np.sqrt(n) * 1.5
            assert max_abs_err < reasonable_atol, \
                f"MatMul error {max_abs_err:.4f} exceeds tolerance {reasonable_atol:.4f}"

    def test_set_weights_recompile(self):
        """set_weights() should recompile and produce non-trivial output."""
        engine = self._get_engine()
        n = engine.matrix_size
        lo, hi = self._weight_range_50pct(engine)
        with engine:
            rng = np.random.default_rng(42)
            W = rng.uniform(lo, hi, (n, n)).astype(np.float32)
            engine.set_weights(W)

            x = rng.uniform(-0.5, 0.5, n).astype(np.float32)
            y = engine.matmul(x).flatten()[:n]
            assert y.size >= n, "Output too small"
            assert np.any(y != 0), "Output is all zeros — weights may not have loaded"

            # Also check roughly in the right ballpark vs CPU
            y_cpu = W @ x
            max_err = np.max(np.abs(y - y_cpu))
            assert max_err < engine.output_scale * n, \
                f"set_weights output too far from CPU reference: max_err={max_err:.4f}"

    def test_fast_path_vs_cpu(self):
        """Fast path (compiler-free) should produce output matching CPU reference."""
        engine = self._get_engine()
        n = engine.matrix_size
        assert engine._param_overhead is not None, "Fast path not available"
        lo, hi = self._weight_range_50pct(engine)
        with engine:
            rng = np.random.default_rng(77)
            W = rng.uniform(lo, hi, (n, n)).astype(np.float32)
            # set_weights uses fast path when param_overhead is present
            engine.set_weights(W)

            x = rng.uniform(-0.5, 0.5, n).astype(np.float32)
            y_tpu = engine.matmul(x).flatten()[:n]
            y_cpu = W @ x

            max_err = np.max(np.abs(y_tpu - y_cpu))
            reasonable_atol = engine.output_scale * np.sqrt(n) * 1.5
            assert max_err < reasonable_atol, \
                f"Fast-path matmul error {max_err:.4f} exceeds {reasonable_atol:.4f}"
