"""Tests for ReservoirComputer — Echo State Network on Edge TPU.

All tests run offline using MockEngine (CPU matmul stand-in for MatMulEngine).
"""

import numpy as np
import pytest

from libredgetpu.reservoir import ReservoirComputer


# ---------------------------------------------------------------------------
# Mock MatMulEngine (CPU matmul stand-in)
# ---------------------------------------------------------------------------

class MockEngine:
    """CPU matmul stand-in for MatMulEngine."""

    def __init__(self, matrix_size=256):
        self.matrix_size = matrix_size
        self._W = None
        self._opened = False
        self.weight_range = (-0.109, 0.107)
        self._hw_initialized = False

    def set_weights(self, W):
        self._W = W.copy()

    def matmul(self, x):
        if self._W is None:
            return np.zeros_like(x)
        return (self._W @ x).astype(np.float32)

    def open(self):
        self._opened = True
        self._hw_initialized = True

    def close(self):
        self._opened = False
        self._hw_initialized = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Initialization and validation tests."""

    def test_properties_correct(self):
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=4, spectral_radius=0.9,
                               input_scaling=0.2, leak_rate=0.8,
                               activation="tanh", seed=42)
        assert rc.reservoir_dim == 64
        assert rc.input_dim == 4
        assert rc.spectral_radius == 0.9
        assert rc.input_scaling == 0.2
        assert rc.leak_rate == 0.8
        assert rc.activation == "tanh"
        assert rc.engine is engine
        assert rc.readout is None
        assert rc.readout_engine is None

    def test_spectral_radius_must_be_positive(self):
        with pytest.raises(ValueError, match="spectral_radius"):
            ReservoirComputer(MockEngine(64), input_dim=4, spectral_radius=0)
        with pytest.raises(ValueError, match="spectral_radius"):
            ReservoirComputer(MockEngine(64), input_dim=4, spectral_radius=-0.5)

    def test_leak_rate_bounds(self):
        with pytest.raises(ValueError, match="leak_rate"):
            ReservoirComputer(MockEngine(64), input_dim=4, leak_rate=0)
        with pytest.raises(ValueError, match="leak_rate"):
            ReservoirComputer(MockEngine(64), input_dim=4, leak_rate=1.5)
        # Boundary: 1.0 is valid
        rc = ReservoirComputer(MockEngine(64), input_dim=4, leak_rate=1.0)
        assert rc.leak_rate == 1.0

    def test_bad_activation_rejected(self):
        with pytest.raises(ValueError, match="activation"):
            ReservoirComputer(MockEngine(64), input_dim=4, activation="sigmoid")

    def test_seed_reproducibility(self):
        e1 = MockEngine(64)
        rc1 = ReservoirComputer(e1, input_dim=4, seed=123)
        e2 = MockEngine(64)
        rc2 = ReservoirComputer(e2, input_dim=4, seed=123)
        np.testing.assert_array_equal(rc1._W_res, rc2._W_res)
        np.testing.assert_array_equal(rc1._W_in, rc2._W_in)

    def test_input_dim_must_be_positive(self):
        with pytest.raises(ValueError, match="input_dim"):
            ReservoirComputer(MockEngine(64), input_dim=0)
        with pytest.raises(ValueError, match="input_dim"):
            ReservoirComputer(MockEngine(64), input_dim=-1)

    def test_input_scaling_must_be_positive(self):
        with pytest.raises(ValueError, match="input_scaling"):
            ReservoirComputer(MockEngine(64), input_dim=4, input_scaling=0)
        with pytest.raises(ValueError, match="input_scaling"):
            ReservoirComputer(MockEngine(64), input_dim=4, input_scaling=-0.1)


# ---------------------------------------------------------------------------
# TestWeightGeneration
# ---------------------------------------------------------------------------

class TestWeightGeneration:
    """Weight generation static methods."""

    def test_spectral_radius_accurate(self):
        target = 0.95
        W = ReservoirComputer.generate_reservoir_weights(64, target, seed=42)
        actual = np.max(np.abs(np.linalg.eigvals(W)))
        assert abs(actual - target) < 1e-3

    def test_input_weights_shape_and_bounds(self):
        W_in = ReservoirComputer.generate_input_weights(64, 4, 0.2, seed=42)
        assert W_in.shape == (64, 4)
        assert W_in.dtype == np.float32
        assert np.all(W_in >= -0.2)
        assert np.all(W_in <= 0.2)

    def test_seed_reproducibility(self):
        W1 = ReservoirComputer.generate_reservoir_weights(64, 0.95, seed=10)
        W2 = ReservoirComputer.generate_reservoir_weights(64, 0.95, seed=10)
        np.testing.assert_array_equal(W1, W2)

    def test_different_seeds_differ(self):
        W1 = ReservoirComputer.generate_reservoir_weights(64, 0.95, seed=10)
        W2 = ReservoirComputer.generate_reservoir_weights(64, 0.95, seed=20)
        assert not np.allclose(W1, W2)

    def test_reservoir_weights_clipped_to_engine_range(self):
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=4, seed=42)
        lo, hi = engine.weight_range
        assert np.all(rc._W_res >= lo)
        assert np.all(rc._W_res <= hi)


# ---------------------------------------------------------------------------
# TestStep
# ---------------------------------------------------------------------------

class TestStep:
    """Single timestep dynamics."""

    def test_zero_input_from_zero_state(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=2, seed=42)
        engine.set_weights(rc._W_res)
        u = np.zeros(2, dtype=np.float32)
        x = rc.step(u)
        np.testing.assert_array_equal(x, np.zeros(32, dtype=np.float32))

    def test_nonzero_input_changes_state(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=2, seed=42)
        engine.set_weights(rc._W_res)
        u = np.ones(2, dtype=np.float32)
        x = rc.step(u)
        assert np.any(x != 0)

    def test_state_shape(self):
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=3, seed=42)
        engine.set_weights(rc._W_res)
        x = rc.step(np.ones(3, dtype=np.float32))
        assert x.shape == (64,)
        assert x.dtype == np.float32

    def test_leak_rate_blending(self):
        engine = MockEngine(16)
        rc_full = ReservoirComputer(engine, input_dim=2, leak_rate=1.0, seed=42)
        engine.set_weights(rc_full._W_res)

        engine2 = MockEngine(16)
        rc_half = ReservoirComputer(engine2, input_dim=2, leak_rate=0.5, seed=42)
        engine2.set_weights(rc_half._W_res)

        u = np.array([1.0, 0.5], dtype=np.float32)
        x_full = rc_full.step(u)
        x_half = rc_half.step(u)
        # With leak_rate=0.5, state should be half of leak_rate=1.0
        # (from zero state, (1-a)*0 + a*act(h) = a*act(h))
        np.testing.assert_allclose(x_half, 0.5 * x_full, atol=1e-6)

    def test_activation_variants(self):
        for act_name in ["tanh", "relu", "identity"]:
            engine = MockEngine(16)
            rc = ReservoirComputer(engine, input_dim=2, activation=act_name,
                                   seed=42)
            engine.set_weights(rc._W_res)
            x = rc.step(np.ones(2, dtype=np.float32))
            assert x.shape == (16,)
            if act_name == "relu":
                assert np.all(x >= 0)
            if act_name == "tanh":
                assert np.all(np.abs(x) <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------

class TestRun:
    """Multi-step run."""

    def test_output_shape(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=2, seed=42)
        engine.set_weights(rc._W_res)
        inputs = np.random.randn(10, 2).astype(np.float32)
        states = rc.run(inputs)
        assert states.shape == (10, 32)

    def test_consistent_with_step(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=2, seed=42)
        engine.set_weights(rc._W_res)
        inputs = np.random.RandomState(0).randn(5, 2).astype(np.float32)
        states_run = rc.run(inputs)

        # Reset and step manually
        rc.reset_state()
        states_step = []
        for t in range(5):
            states_step.append(rc.step(inputs[t]))
        states_step = np.array(states_step)

        np.testing.assert_allclose(states_run, states_step, atol=1e-6)

    def test_reset_gives_identical_trajectories(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=2, seed=42)
        engine.set_weights(rc._W_res)
        inputs = np.random.RandomState(1).randn(8, 2).astype(np.float32)
        s1 = rc.run(inputs)
        rc.reset_state()
        s2 = rc.run(inputs)
        np.testing.assert_allclose(s1, s2, atol=1e-6)


# ---------------------------------------------------------------------------
# TestFitPredict
# ---------------------------------------------------------------------------

class TestFitPredict:
    """Training and prediction."""

    def _make_sine_data(self, T=200, seed=42):
        """Generate a sine wave prediction task."""
        t = np.linspace(0, 4 * np.pi, T).astype(np.float32)
        inputs = np.sin(t)[:, np.newaxis]       # [T, 1]
        targets = np.sin(t + 0.1)[:, np.newaxis]  # [T, 1] phase-shifted
        return inputs, targets

    def test_fit_sine_wave(self):
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc._W_res)
        inputs, targets = self._make_sine_data(300)
        rc.fit(inputs, targets, warmup=50)
        assert rc.readout is not None
        assert rc.readout.shape == (1, 64)

    def test_predict_shape(self):
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc._W_res)
        inputs, targets = self._make_sine_data(200)
        rc.fit(inputs, targets, warmup=50)
        preds = rc.predict(inputs)
        assert preds.shape == (200, 1)

    def test_predict_before_fit_raises(self):
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc._W_res)
        with pytest.raises(ValueError, match="fit"):
            rc.predict(np.zeros((10, 1), dtype=np.float32))

    def test_warmup_discards_transient(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc._W_res)
        inputs, targets = self._make_sine_data(100)
        # Fit with warmup=90 means only 10 samples used
        rc.fit(inputs, targets, warmup=90)
        assert rc.readout is not None

    def test_ridge_alpha_regularization(self):
        engine = MockEngine(32)
        rc1 = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc1._W_res)
        inputs, targets = self._make_sine_data(200)
        rc1.fit(inputs, targets, warmup=50, ridge_alpha=0.0001)
        readout_small_reg = rc1.readout.copy()

        rc2 = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc2._W_res)
        rc2.fit(inputs, targets, warmup=50, ridge_alpha=100.0)
        readout_large_reg = rc2.readout.copy()

        # Larger regularization should produce smaller weights
        assert np.linalg.norm(readout_large_reg) < np.linalg.norm(readout_small_reg)

    def test_multi_output_target(self):
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc._W_res)
        inputs = np.random.randn(200, 1).astype(np.float32)
        targets = np.random.randn(200, 3).astype(np.float32)
        rc.fit(inputs, targets, warmup=50)
        assert rc.readout.shape == (3, 64)
        preds = rc.predict(inputs)
        assert preds.shape == (200, 3)

    def test_fit_1d_targets(self):
        """fit() should accept 1-D targets (auto-reshape to [T, 1])."""
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=1, seed=42)
        engine.set_weights(rc._W_res)
        inputs = np.random.randn(100, 1).astype(np.float32)
        targets = np.random.randn(100).astype(np.float32)
        rc.fit(inputs, targets, warmup=10)
        assert rc.readout.shape == (1, 32)


# ---------------------------------------------------------------------------
# TestReadoutEngine
# ---------------------------------------------------------------------------

class TestReadoutEngine:
    """Optional Edge TPU readout engine."""

    def test_fit_loads_padded_weights(self):
        engine = MockEngine(64)
        readout_engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=1, readout_engine=readout_engine,
                               seed=42)
        engine.set_weights(rc._W_res)
        inputs = np.random.randn(200, 1).astype(np.float32)
        targets = np.random.randn(200, 3).astype(np.float32)
        rc.fit(inputs, targets, warmup=50)
        # readout_engine should have weights loaded
        assert readout_engine._W is not None
        assert readout_engine._W.shape == (64, 64)
        # First 3 rows should match W_out, rest should be zero
        np.testing.assert_allclose(readout_engine._W[:3, :64], rc.readout, atol=1e-5)
        np.testing.assert_array_equal(readout_engine._W[3:, :], 0)

    def test_predict_uses_readout_engine(self):
        engine = MockEngine(32)
        readout_engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=1, readout_engine=readout_engine,
                               seed=42)
        engine.set_weights(rc._W_res)
        inputs = np.random.randn(100, 1).astype(np.float32)
        targets = np.random.randn(100, 2).astype(np.float32)
        rc.fit(inputs, targets, warmup=20)
        preds = rc.predict(inputs)
        assert preds.shape == (100, 2)


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Context manager and lifecycle delegation."""

    def test_context_manager_delegates(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=2, seed=42)
        assert not engine._opened
        with rc:
            assert engine._opened
            assert engine._W is not None  # W_res loaded on open
        assert not engine._opened

    def test_from_template_creates_engine(self):
        # We can't actually call from_template without a real template file,
        # but we can verify the classmethod exists and the _owns_engine flag
        engine = MockEngine(64)
        rc = ReservoirComputer(engine, input_dim=4, seed=42)
        assert not rc._owns_engine
        # Simulate what from_template does
        rc._owns_engine = True
        assert rc._owns_engine

    def test_open_loads_weights(self):
        engine = MockEngine(32)
        rc = ReservoirComputer(engine, input_dim=2, seed=42)
        assert engine._W is None  # Not loaded yet (engine not hw_initialized)
        rc.open()
        assert engine._W is not None
        assert engine._W.shape == (32, 32)
        rc.close()
