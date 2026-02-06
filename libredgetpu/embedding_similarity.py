"""EmbeddingSimilarity — Cosine Similarity Search on Edge TPU.

Visual place recognition needs two steps: extract an embedding from an image
(e.g., headless MobileNet via :class:`SimpleInvoker`, ~4-8 ms) and then find
the closest match in a database.  The second step is a matrix-vector multiply
— exactly what :class:`MatMulEngine` does (0.28 ms).

This module wraps **only** the similarity search (MatMulEngine).  The user
manages the backbone model separately.  This follows the VisualCompass pattern
(doesn't own the camera) and avoids USB sharing complexity.

Scaling strategy:
    L2-normalized embeddings have values in [-1, 1], but MatMulEngine's weight
    range is much narrower (e.g., [-0.109, +0.107] for Dense(256)).  Embeddings
    are scaled to fit the weight range, then scores are unscaled to recover
    cosine similarity.  With int8 quantization, similarity resolution is ~0.18
    (about 5-6 levels across [0, 1]).  Rankings are reliable; absolute scores
    are coarse.

Usage::

    from libredgetpu import EmbeddingSimilarity

    with EmbeddingSimilarity.from_template(256) as sim:
        sim.add("place_A", embedding_a)
        sim.add("place_B", embedding_b)

        results = sim.query(query_embedding, top_k=3)
        for label, score in results:
            print(f"{label}: {score:.3f}")

The module uses **composition** (wrapping ``MatMulEngine``), consistent with
the ``VisualCompass`` -> ``OpticalFlow`` pattern.  No new Edge TPU model ---
reuses Dense(N) templates.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from .matmul_engine import MatMulEngine

__all__ = ["EmbeddingSimilarity"]


class EmbeddingSimilarity:
    """Cosine similarity search backed by Edge TPU matrix multiplication.

    Stores a database of L2-normalized embeddings as rows of the MatMulEngine
    weight matrix.  Queries compute cosine similarity via a single matmul.

    Args:
        engine: A :class:`MatMulEngine` instance (opened or unopened).
            Must have ``weight_range`` available (sidecar JSON with
            quantization metadata).
    """

    def __init__(self, engine: MatMulEngine) -> None:
        if engine.weight_range is None:
            raise ValueError(
                "MatMulEngine must have weight_range available "
                "(sidecar JSON with quantization metadata required)"
            )

        self._engine = engine
        self._owns_engine = False

        w_min, w_max = engine.weight_range
        self._scale_factor = min(abs(w_min), abs(w_max))

        n = engine.matrix_size
        self._capacity = n
        self._embedding_dim = n

        # Database: rows are scaled L2-normalized embeddings, zero-padded
        self._embeddings = np.zeros((n, n), dtype=np.float32)
        # Normalized (pre-scale) embeddings for save/load portability
        self._normalized = np.zeros((n, n), dtype=np.float32)
        self._labels: list = []
        self._weights_loaded = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_template(
        cls,
        embedding_dim: int,
        *,
        firmware_path: Optional[str] = None,
    ) -> "EmbeddingSimilarity":
        """Create an EmbeddingSimilarity from a pre-compiled MatMulEngine template.

        Args:
            embedding_dim: Embedding dimension (must match a Dense(N) template).
            firmware_path: Edge TPU firmware path; auto-downloaded if None.

        Returns:
            EmbeddingSimilarity instance (not yet opened).

        Raises:
            FileNotFoundError: If no template exists for the specified size.
        """
        engine = MatMulEngine.from_template(
            embedding_dim, firmware_path=firmware_path
        )
        obj = cls(engine)
        obj._owns_engine = True
        return obj

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------

    def add(self, label: str, embedding: np.ndarray) -> None:
        """Add a single embedding to the database.

        Args:
            label: Unique identifier for this embedding.
            embedding: Float32 vector of shape ``[D]``.

        Raises:
            ValueError: If label is duplicate, capacity is full, dimension
                mismatch, or embedding is a zero vector.
        """
        if label in self._labels:
            raise ValueError(f"Duplicate label: {label!r}")
        if len(self._labels) >= self._capacity:
            raise ValueError(
                f"Database full: {self._capacity} entries "
                f"(capacity = matrix_size)"
            )
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} != "
                f"expected {self._embedding_dim}"
            )

        normed = self._l2_normalize(embedding)
        idx = len(self._labels)
        self._normalized[idx] = normed
        self._embeddings[idx] = normed * self._scale_factor
        self._labels.append(label)
        self._weights_loaded = False

    def add_batch(
        self, labels: list, embeddings: np.ndarray
    ) -> None:
        """Add multiple embeddings to the database.

        Args:
            labels: List of unique identifiers.
            embeddings: Float32 array of shape ``[K, D]``.

        Raises:
            ValueError: If any label is duplicate, capacity exceeded,
                dimension mismatch, or any embedding is a zero vector.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if len(labels) != embeddings.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} != embeddings count "
                f"{embeddings.shape[0]}"
            )
        for i, label in enumerate(labels):
            self.add(label, embeddings[i])

    def remove(self, label: str) -> None:
        """Remove an embedding from the database.

        Shifts subsequent rows up and zero-pads the last row.

        Args:
            label: The label to remove.

        Raises:
            KeyError: If label not found.
        """
        if label not in self._labels:
            raise KeyError(f"Label not found: {label!r}")

        idx = self._labels.index(label)
        n = len(self._labels)

        # Shift rows up
        if idx < n - 1:
            self._embeddings[idx:n - 1] = self._embeddings[idx + 1:n]
            self._normalized[idx:n - 1] = self._normalized[idx + 1:n]

        # Zero-pad last occupied row
        self._embeddings[n - 1] = 0.0
        self._normalized[n - 1] = 0.0
        self._labels.pop(idx)
        self._weights_loaded = False

    def set_database(
        self, labels: list, embeddings: np.ndarray
    ) -> None:
        """Replace the entire database.

        Args:
            labels: List of unique identifiers.
            embeddings: Float32 array of shape ``[K, D]``.

        Raises:
            ValueError: If duplicates, K > capacity, dimension mismatch,
                or any zero vector.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if len(labels) != embeddings.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} != embeddings count "
                f"{embeddings.shape[0]}"
            )
        if len(labels) > self._capacity:
            raise ValueError(
                f"Database size {len(labels)} exceeds capacity "
                f"{self._capacity}"
            )
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate labels in set_database()")
        if embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} != "
                f"expected {self._embedding_dim}"
            )

        # Reset
        self._embeddings[:] = 0.0
        self._normalized[:] = 0.0
        self._labels.clear()

        for i, label in enumerate(labels):
            normed = self._l2_normalize(embeddings[i])
            self._normalized[i] = normed
            self._embeddings[i] = normed * self._scale_factor
            self._labels.append(label)

        self._weights_loaded = False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self, embedding: np.ndarray, top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """Find the most similar embeddings in the database.

        Args:
            embedding: Query vector, shape ``[D]``, float32.
            top_k: Number of results to return.

        Returns:
            List of ``(label, cosine_similarity)`` tuples sorted by
            descending similarity.

        Raises:
            ValueError: If database is empty or dimension mismatch.
        """
        if len(self._labels) == 0:
            raise ValueError("Database is empty")

        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Query dimension {embedding.shape[0]} != "
                f"expected {self._embedding_dim}"
            )

        normed = self._l2_normalize(embedding)

        # Lazy weight upload
        if not self._weights_loaded:
            self._upload_weights()

        raw = self._engine.matmul(normed)
        scores = raw / self._scale_factor

        # Only consider populated entries
        n = len(self._labels)
        scores = scores[:n]

        top_k = min(top_k, n)
        indices = np.argsort(scores)[::-1][:top_k]

        return [(self._labels[i], float(scores[i])) for i in indices]

    def query_batch(
        self, embeddings: np.ndarray, top_k: int = 1
    ) -> List[List[Tuple[str, float]]]:
        """Query multiple embeddings.

        Args:
            embeddings: Query vectors, shape ``[B, D]``, float32.
            top_k: Number of results per query.

        Returns:
            List of result lists, one per query.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        return [self.query(embeddings[i], top_k=top_k)
                for i in range(embeddings.shape[0])]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the database to a ``.npz`` file.

        Stores L2-normalized (pre-scale) embeddings so the database can
        be loaded into an engine with a different weight range.

        Args:
            path: Output file path (recommended: ``.npz`` extension).
        """
        n = len(self._labels)
        np.savez(
            path,
            normalized=self._normalized[:n],
            labels=np.array(self._labels, dtype=object),
        )

    def load(self, path: str) -> None:
        """Load a database from a ``.npz`` file.

        Re-scales embeddings to the current engine's weight range.

        Args:
            path: Path to ``.npz`` file saved by :meth:`save`.

        Raises:
            ValueError: If embedding dimension doesn't match.
        """
        data = np.load(path, allow_pickle=True)
        normalized = data["normalized"].astype(np.float32)
        labels = list(data["labels"])

        if normalized.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Saved embedding dimension {normalized.shape[1]} != "
                f"engine dimension {self._embedding_dim}"
            )

        # Reset and populate
        self._embeddings[:] = 0.0
        self._normalized[:] = 0.0
        self._labels.clear()

        n = normalized.shape[0]
        self._normalized[:n] = normalized
        self._embeddings[:n] = normalized * self._scale_factor
        self._labels.extend(labels)
        self._weights_loaded = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector.

        Args:
            embedding: Float32 vector.

        Returns:
            Unit-length vector, float32.

        Raises:
            ValueError: If the vector is zero.
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Cannot L2-normalize a zero vector")
        return (embedding / norm).astype(np.float32)

    def _upload_weights(self) -> None:
        """Upload the current embedding matrix to the engine."""
        self._engine.set_weights(self._embeddings)
        self._weights_loaded = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def database_size(self) -> int:
        """Number of embeddings currently stored."""
        return len(self._labels)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality D."""
        return self._embedding_dim

    @property
    def capacity(self) -> int:
        """Maximum number of embeddings (= matrix_size)."""
        return self._capacity

    @property
    def labels(self) -> list:
        """Copy of the current label list."""
        return list(self._labels)

    @property
    def engine(self) -> MatMulEngine:
        """Underlying MatMulEngine."""
        return self._engine

    @property
    def scale_factor(self) -> float:
        """Scale factor used to fit embeddings into weight range."""
        return self._scale_factor

    # ------------------------------------------------------------------
    # Lifecycle (delegates to engine)
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the underlying MatMulEngine and upload weights if non-empty."""
        self._engine.open()
        if len(self._labels) > 0:
            self._upload_weights()

    def close(self) -> None:
        """Close the underlying engine."""
        self._engine.close()
        self._weights_loaded = False

    def __enter__(self) -> "EmbeddingSimilarity":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
