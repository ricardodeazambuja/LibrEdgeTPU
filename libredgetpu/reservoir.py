"""ReservoirComputer — Echo State Network on Edge TPU.

Reservoir computing uses a fixed random recurrent network where only the
readout layer is trained.  The key operation --- ``W_res @ x(t-1)`` --- is a
large matrix-vector multiply, which is exactly what :class:`MatMulEngine`
does.  The Edge TPU's 8-bit quantization noise actually benefits reservoir
dynamics by acting as natural regularization.

Each timestep (~0.6 ms):

.. code-block:: text

    Edge TPU:  h1 = W_res @ x(t-1)               # 0.28 ms via MatMulEngine
    CPU:       h2 = W_in  @ u(t)                  # microseconds (small M)
    CPU:       x(t) = (1-a)*x(t-1) + a*act(h1+h2) # leaky integration
    CPU:       y(t) = W_out @ x(t)                # readout (CPU or optional Edge TPU)

Usage::

    from libredgetpu import ReservoirComputer

    with ReservoirComputer.from_template(256, input_dim=4) as rc:
        rc.fit(training_inputs, training_targets, warmup=100)
        predictions = rc.predict(test_inputs)

The module uses **composition** (wrapping ``MatMulEngine``), consistent with
the ``VisualCompass`` -> ``OpticalFlow`` pattern.  No new Edge TPU model ---
reuses Dense(N) templates.
"""

from typing import Optional

import numpy as np

from .matmul_engine import MatMulEngine

__all__ = ["ReservoirComputer"]

# Supported activation functions (all applied on CPU)
_ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "identity": lambda x: x,
}


class ReservoirComputer:
    """Echo State Network backed by Edge TPU matrix multiplication.

    The reservoir weight matrix ``W_res`` (N x N) is loaded onto the Edge TPU
    via :class:`MatMulEngine`.  Input projection ``W_in`` (N x M) and readout
    ``W_out`` (K x N) live on the CPU.

    Args:
        engine: A :class:`MatMulEngine` instance (opened or unopened).
        input_dim: Dimensionality of the input signal ``u(t)``.
        spectral_radius: Target spectral radius of ``W_res``.  Must be > 0.
        input_scaling: Scale for uniform input weight matrix.  Must be > 0.
        leak_rate: Leaky integration coefficient in (0, 1].
        activation: Activation function name: ``"tanh"``, ``"relu"``, or
            ``"identity"``.
        seed: Random seed for reproducible weight generation.
        readout_engine: Optional second :class:`MatMulEngine` for large
            output dimensions.  If provided, ``fit()`` zero-pads ``W_out``
            to [N, N] and loads it onto this engine.
    """

    def __init__(
        self,
        engine: MatMulEngine,
        input_dim: int,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.1,
        leak_rate: float = 1.0,
        activation: str = "tanh",
        seed: Optional[int] = None,
        readout_engine: Optional[MatMulEngine] = None,
    ) -> None:
        # --- Validation ---
        if spectral_radius <= 0:
            raise ValueError(
                f"spectral_radius must be > 0, got {spectral_radius}"
            )
        if input_scaling <= 0:
            raise ValueError(
                f"input_scaling must be > 0, got {input_scaling}"
            )
        if leak_rate <= 0 or leak_rate > 1.0:
            raise ValueError(
                f"leak_rate must be in (0, 1.0], got {leak_rate}"
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {set(_ACTIVATIONS)}, "
                f"got {activation!r}"
            )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")

        self._engine = engine
        self._input_dim = int(input_dim)
        self._spectral_radius = float(spectral_radius)
        self._input_scaling = float(input_scaling)
        self._leak_rate = float(leak_rate)
        self._activation_name = activation
        self._activation_fn = _ACTIVATIONS[activation]
        self._seed = seed
        self._readout_engine = readout_engine
        self._owns_engine = False
        self._owns_readout_engine = False

        # Reservoir dimension from engine
        self._n = engine.matrix_size

        # Generate fixed random weights
        self._W_res = self.generate_reservoir_weights(
            self._n, self._spectral_radius, seed=seed
        )
        self._W_in = self.generate_input_weights(
            self._n, self._input_dim, self._input_scaling,
            seed=(seed + 1) if seed is not None else None,
        )

        # Clip reservoir weights to engine's representable range
        w_range = engine.weight_range
        if w_range is not None:
            self._W_res = np.clip(self._W_res, w_range[0], w_range[1])

        # Recurrent state vector
        self._state = np.zeros(self._n, dtype=np.float32)

        # Readout (set by fit())
        self._readout = None  # shape [K, N] or None
        self._output_dim = None

        # Eager weight loading: if engine looks open, load W_res now
        if hasattr(engine, '_hw_initialized') and engine._hw_initialized:
            engine.set_weights(self._W_res.astype(np.float32))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_template(
        cls,
        reservoir_dim: int,
        input_dim: int,
        spectral_radius: float = 0.95,
        input_scaling: float = 0.1,
        leak_rate: float = 1.0,
        activation: str = "tanh",
        seed: Optional[int] = None,
        readout_engine: Optional[MatMulEngine] = None,
        firmware_path: Optional[str] = None,
    ) -> "ReservoirComputer":
        """Create a ReservoirComputer from a pre-compiled MatMulEngine template.

        Args:
            reservoir_dim: Reservoir size N (must match a Dense(N) template).
            input_dim: Input signal dimensionality M.
            spectral_radius: Target spectral radius of W_res (default 0.95).
            input_scaling: Input weight scale (default 0.1).
            leak_rate: Leaky integration coefficient (default 1.0).
            activation: Activation name (default ``"tanh"``).
            seed: Random seed for reproducibility.
            readout_engine: Optional second MatMulEngine for readout.
            firmware_path: Edge TPU firmware path; auto-downloaded if None.

        Returns:
            ReservoirComputer instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the specified size.
        """
        engine = MatMulEngine.from_template(
            reservoir_dim, firmware_path=firmware_path
        )
        obj = cls(
            engine,
            input_dim,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            activation=activation,
            seed=seed,
            readout_engine=readout_engine,
        )
        obj._owns_engine = True
        return obj

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance the reservoir by one timestep.

        Args:
            u: Input signal, shape ``[M]``, float32.

        Returns:
            Updated reservoir state ``x(t)``, shape ``[N]``, float32.
        """
        u = np.asarray(u, dtype=np.float32)
        # h1 = W_res @ x  (via Edge TPU or mock)
        h1 = self._engine.matmul(self._state)
        # h2 = W_in @ u  (CPU, small)
        h2 = self._W_in @ u
        # Leaky integration
        pre_activation = h1 + h2
        activated = self._activation_fn(pre_activation).astype(np.float32)
        alpha = self._leak_rate
        self._state = ((1.0 - alpha) * self._state + alpha * activated).astype(
            np.float32
        )
        return self._state.copy()

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Drive the reservoir with a sequence of inputs.

        Args:
            inputs: Input signals, shape ``[T, M]``, float32.

        Returns:
            Reservoir states, shape ``[T, N]``, float32.
        """
        inputs = np.asarray(inputs, dtype=np.float32)
        T = inputs.shape[0]
        states = np.empty((T, self._n), dtype=np.float32)
        for t in range(T):
            states[t] = self.step(inputs[t])
        return states

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        warmup: int = 0,
        ridge_alpha: float = 1e-6,
    ) -> None:
        """Train the readout layer via ridge regression.

        Drives the reservoir with *inputs*, discards the first *warmup*
        steps, and solves for ``W_out`` that maps states to *targets*.

        .. note:: Resets the reservoir state to zeros before driving.

        Args:
            inputs: Training inputs, shape ``[T, M]``.
            targets: Training targets, shape ``[T, K]`` or ``[T]``.
            warmup: Number of initial transient steps to discard.
            ridge_alpha: Ridge regularization coefficient.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]

        T = inputs.shape[0]
        if targets.shape[0] != T:
            raise ValueError(
                f"inputs length {T} != targets length {targets.shape[0]}"
            )
        if warmup < 0 or warmup >= T:
            raise ValueError(
                f"warmup must be in [0, T), got {warmup} with T={T}"
            )

        # Run reservoir and collect states
        self.reset_state()
        states = self.run(inputs)

        # Discard warmup transient
        states_train = states[warmup:]
        targets_train = targets[warmup:]

        # Ridge regression via augmented least squares
        # Solve: (S^T S + alpha I) W_out^T = S^T Y
        # Using lstsq on the augmented system for numerical stability
        K = targets_train.shape[1]
        N = self._n
        reg = np.sqrt(ridge_alpha) * np.eye(N, dtype=np.float32)
        A = np.vstack([states_train, reg])
        b = np.vstack([targets_train, np.zeros((N, K), dtype=np.float32)])
        W_out_T, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        self._readout = W_out_T.T.astype(np.float32)  # [K, N]
        self._output_dim = K

        # Optionally load readout onto Edge TPU engine
        if self._readout_engine is not None:
            n_ro = self._readout_engine.matrix_size
            padded = np.zeros((n_ro, n_ro), dtype=np.float32)
            padded[:K, :N] = self._readout
            self._readout_engine.set_weights(padded)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run the reservoir and apply the trained readout.

        .. note:: Resets the reservoir state to zeros before driving.

        Args:
            inputs: Input signals, shape ``[T, M]``.

        Returns:
            Predictions, shape ``[T, K]``.

        Raises:
            ValueError: If ``fit()`` has not been called.
        """
        if self._readout is None:
            raise ValueError("No readout trained. Call fit() first.")

        self.reset_state()
        states = self.run(inputs)

        if self._readout_engine is not None:
            # Use Edge TPU for readout
            T = states.shape[0]
            K = self._output_dim
            predictions = np.empty((T, K), dtype=np.float32)
            for t in range(T):
                y_full = self._readout_engine.matmul(states[t])
                predictions[t] = y_full[:K]
            return predictions
        else:
            # CPU readout
            return (states @ self._readout.T).astype(np.float32)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset the reservoir state to zeros."""
        self._state = np.zeros(self._n, dtype=np.float32)

    @property
    def state(self) -> np.ndarray:
        """Copy of the current reservoir state x(t), shape [N]."""
        return self._state.copy()

    # ------------------------------------------------------------------
    # Weight generation (static)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_reservoir_weights(
        n: int, spectral_radius: float, seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate a random reservoir weight matrix with target spectral radius.

        Args:
            n: Matrix dimension.
            spectral_radius: Desired spectral radius (largest eigenvalue magnitude).
            seed: Random seed for reproducibility.

        Returns:
            Weight matrix of shape ``[n, n]``, float32.
        """
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n, n)).astype(np.float32)
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W *= spectral_radius / current_radius
        return W

    @staticmethod
    def generate_input_weights(
        n: int, m: int, input_scaling: float, seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate a random input projection matrix.

        Args:
            n: Reservoir dimension.
            m: Input dimension.
            input_scaling: Uniform distribution bound.
            seed: Random seed for reproducibility.

        Returns:
            Weight matrix of shape ``[n, m]``, float32.
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(
            -input_scaling, input_scaling, (n, m)
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reservoir_dim(self) -> int:
        """Reservoir size N."""
        return self._n

    @property
    def input_dim(self) -> int:
        """Input signal dimensionality M."""
        return self._input_dim

    @property
    def spectral_radius(self) -> float:
        """Target spectral radius of W_res."""
        return self._spectral_radius

    @property
    def input_scaling(self) -> float:
        """Input weight scaling factor."""
        return self._input_scaling

    @property
    def leak_rate(self) -> float:
        """Leaky integration coefficient."""
        return self._leak_rate

    @property
    def activation(self) -> str:
        """Activation function name."""
        return self._activation_name

    @property
    def readout(self) -> Optional[np.ndarray]:
        """Trained readout matrix W_out [K, N], or None if not yet trained."""
        return self._readout.copy() if self._readout is not None else None

    @property
    def engine(self) -> MatMulEngine:
        """Underlying MatMulEngine for reservoir computation."""
        return self._engine

    @property
    def readout_engine(self) -> Optional[MatMulEngine]:
        """Optional MatMulEngine for readout computation."""
        return self._readout_engine

    # ------------------------------------------------------------------
    # Lifecycle (delegates to engine)
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the underlying MatMulEngine and load reservoir weights."""
        self._engine.open()
        self._engine.set_weights(self._W_res.astype(np.float32))
        if self._readout_engine is not None and self._owns_readout_engine:
            self._readout_engine.open()

    def close(self) -> None:
        """Close the underlying engine(s)."""
        self._engine.close()
        if self._readout_engine is not None and self._owns_readout_engine:
            self._readout_engine.close()

    def __enter__(self) -> "ReservoirComputer":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
