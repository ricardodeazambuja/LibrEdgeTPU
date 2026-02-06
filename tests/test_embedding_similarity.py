"""Tests for EmbeddingSimilarity — Cosine Similarity Search on Edge TPU.

All tests run offline using MockEngine (CPU matmul stand-in for MatMulEngine).
"""

import os
import tempfile

import numpy as np
import pytest

from libredgetpu.embedding_similarity import EmbeddingSimilarity


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


def _random_embedding(dim, seed=None):
    """Generate a random unit-length embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Initialization and validation tests."""

    def test_properties_correct(self):
        engine = MockEngine(64)
        sim = EmbeddingSimilarity(engine)
        assert sim.embedding_dim == 64
        assert sim.capacity == 64
        assert sim.database_size == 0
        assert sim.labels == []
        assert sim.engine is engine

    def test_empty_database(self):
        engine = MockEngine(128)
        sim = EmbeddingSimilarity(engine)
        assert sim.database_size == 0

    def test_capacity_equals_matrix_size(self):
        for n in [32, 64, 256]:
            engine = MockEngine(n)
            sim = EmbeddingSimilarity(engine)
            assert sim.capacity == n

    def test_no_weight_range_raises(self):
        engine = MockEngine(64)
        engine.weight_range = None
        with pytest.raises(ValueError, match="weight_range"):
            EmbeddingSimilarity(engine)

    def test_scale_factor_math(self):
        engine = MockEngine(64)
        engine.weight_range = (-0.109, 0.107)
        sim = EmbeddingSimilarity(engine)
        assert sim.scale_factor == pytest.approx(0.107)

        engine2 = MockEngine(64)
        engine2.weight_range = (-0.05, 0.10)
        sim2 = EmbeddingSimilarity(engine2)
        assert sim2.scale_factor == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# TestDatabase
# ---------------------------------------------------------------------------

class TestDatabase:
    """Database management tests."""

    def test_add_single(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        emb = _random_embedding(32, seed=1)
        sim.add("a", emb)
        assert sim.database_size == 1
        assert sim.labels == ["a"]

    def test_add_batch(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        embs = np.stack([_random_embedding(32, seed=i) for i in range(5)])
        labels = [f"item_{i}" for i in range(5)]
        sim.add_batch(labels, embs)
        assert sim.database_size == 5
        assert sim.labels == labels

    def test_remove(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        for i in range(3):
            sim.add(f"item_{i}", _random_embedding(32, seed=i))
        sim.remove("item_1")
        assert sim.database_size == 2
        assert "item_1" not in sim.labels
        assert sim.labels == ["item_0", "item_2"]

    def test_set_database(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        sim.add("old", _random_embedding(32, seed=0))
        embs = np.stack([_random_embedding(32, seed=i) for i in range(4)])
        labels = [f"new_{i}" for i in range(4)]
        sim.set_database(labels, embs)
        assert sim.database_size == 4
        assert sim.labels == labels

    def test_duplicate_label_raises(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        sim.add("a", _random_embedding(32, seed=0))
        with pytest.raises(ValueError, match="Duplicate"):
            sim.add("a", _random_embedding(32, seed=1))

    def test_capacity_overflow_raises(self):
        engine = MockEngine(4)
        sim = EmbeddingSimilarity(engine)
        for i in range(4):
            sim.add(f"item_{i}", _random_embedding(4, seed=i))
        with pytest.raises(ValueError, match="full"):
            sim.add("overflow", _random_embedding(4, seed=99))

    def test_remove_nonexistent_raises(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        with pytest.raises(KeyError, match="not found"):
            sim.remove("nonexistent")


# ---------------------------------------------------------------------------
# TestQuery
# ---------------------------------------------------------------------------

class TestQuery:
    """Query tests."""

    def test_self_similarity_near_one(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        emb = _random_embedding(32, seed=42)
        sim.add("target", emb)
        # Force weight upload
        sim._upload_weights()
        results = sim.query(emb, top_k=1)
        assert len(results) == 1
        label, score = results[0]
        assert label == "target"
        # MockEngine uses float matmul, so similarity should be very close to 1
        assert score == pytest.approx(1.0, abs=0.01)

    def test_top_k_ordering(self):
        dim = 32
        engine = MockEngine(dim)
        sim = EmbeddingSimilarity(engine)

        # Create embeddings with known similarity to a query
        query = _random_embedding(dim, seed=0)
        # Similar embedding (slightly perturbed)
        similar = query + 0.1 * _random_embedding(dim, seed=1)
        similar = similar / np.linalg.norm(similar)
        # Different embedding
        different = _random_embedding(dim, seed=99)

        sim.add("similar", similar)
        sim.add("different", different)
        sim._upload_weights()

        results = sim.query(query, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "similar"
        assert results[0][1] > results[1][1]

    def test_empty_database_raises(self):
        engine = MockEngine(32)
        sim = EmbeddingSimilarity(engine)
        with pytest.raises(ValueError, match="empty"):
            sim.query(_random_embedding(32, seed=0))

    def test_score_ordering(self):
        dim = 32
        engine = MockEngine(dim)
        sim = EmbeddingSimilarity(engine)

        query = _random_embedding(dim, seed=0)
        # Embeddings at varying similarity
        rng = np.random.default_rng(42)
        for i in range(10):
            noise = rng.standard_normal(dim).astype(np.float32) * (i * 0.3)
            emb = query + noise
            emb = emb / np.linalg.norm(emb)
            sim.add(f"item_{i}", emb)

        sim._upload_weights()
        results = sim.query(query, top_k=10)
        scores = [s for _, s in results]
        # Scores should be in descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_orthogonal_near_zero(self):
        dim = 32
        engine = MockEngine(dim)
        sim = EmbeddingSimilarity(engine)

        # Create two orthogonal vectors
        a = np.zeros(dim, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(dim, dtype=np.float32)
        b[1] = 1.0

        sim.add("orthogonal", b)
        sim._upload_weights()
        results = sim.query(a, top_k=1)
        _, score = results[0]
        assert abs(score) < 0.05

    def test_opposite_near_negative_one(self):
        dim = 32
        engine = MockEngine(dim)
        sim = EmbeddingSimilarity(engine)

        emb = _random_embedding(dim, seed=42)
        opposite = -emb

        sim.add("opposite", opposite)
        sim._upload_weights()
        results = sim.query(emb, top_k=1)
        _, score = results[0]
        assert score == pytest.approx(-1.0, abs=0.01)


# ---------------------------------------------------------------------------
# TestNormalization
# ---------------------------------------------------------------------------

class TestNormalization:
    """L2 normalization and scaling tests."""

    def test_l2_normalize_correctness(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        normed = EmbeddingSimilarity._l2_normalize(v)
        assert np.linalg.norm(normed) == pytest.approx(1.0, abs=1e-5)
        assert normed[0] == pytest.approx(0.6, abs=1e-5)
        assert normed[1] == pytest.approx(0.8, abs=1e-5)

    def test_scale_factor_applied_to_stored_rows(self):
        engine = MockEngine(8)
        engine.weight_range = (-0.109, 0.107)
        sim = EmbeddingSimilarity(engine)
        emb = _random_embedding(8, seed=42)
        sim.add("test", emb)

        # The stored row should be normed * scale_factor
        normed = emb / np.linalg.norm(emb)
        expected = normed * sim.scale_factor
        np.testing.assert_allclose(sim._embeddings[0], expected, atol=1e-6)

    def test_zero_vector_raises(self):
        engine = MockEngine(8)
        sim = EmbeddingSimilarity(engine)
        with pytest.raises(ValueError, match="zero vector"):
            sim.add("zero", np.zeros(8, dtype=np.float32))


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Persistence tests."""

    def test_round_trip(self):
        engine = MockEngine(16)
        sim = EmbeddingSimilarity(engine)
        embs = np.stack([_random_embedding(16, seed=i) for i in range(5)])
        labels = [f"item_{i}" for i in range(5)]
        sim.add_batch(labels, embs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db.npz")
            sim.save(path)

            sim2 = EmbeddingSimilarity(MockEngine(16))
            sim2.load(path)

            assert sim2.database_size == 5
            assert sim2.labels == labels
            np.testing.assert_allclose(
                sim2._normalized[:5], sim._normalized[:5], atol=1e-6
            )

    def test_load_into_fresh_instance(self):
        engine = MockEngine(16)
        sim = EmbeddingSimilarity(engine)
        sim.add("a", _random_embedding(16, seed=0))
        sim.add("b", _random_embedding(16, seed=1))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db.npz")
            sim.save(path)

            fresh = EmbeddingSimilarity(MockEngine(16))
            assert fresh.database_size == 0
            fresh.load(path)
            assert fresh.database_size == 2
            assert fresh.labels == ["a", "b"]

    def test_dimension_mismatch_raises(self):
        engine = MockEngine(16)
        sim = EmbeddingSimilarity(engine)
        sim.add("a", _random_embedding(16, seed=0))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "db.npz")
            sim.save(path)

            wrong_dim = EmbeddingSimilarity(MockEngine(32))
            with pytest.raises(ValueError, match="dimension"):
                wrong_dim.load(path)


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Context manager and lifecycle delegation."""

    def test_context_manager_delegates(self):
        engine = MockEngine(16)
        sim = EmbeddingSimilarity(engine)
        assert not engine._opened
        with sim:
            assert engine._opened
        assert not engine._opened

    def test_from_template_ownership(self):
        # Can't call from_template without real template, but verify flag
        engine = MockEngine(16)
        sim = EmbeddingSimilarity(engine)
        assert not sim._owns_engine
        sim._owns_engine = True
        assert sim._owns_engine

    def test_open_loads_weights(self):
        engine = MockEngine(16)
        sim = EmbeddingSimilarity(engine)
        sim.add("a", _random_embedding(16, seed=0))
        assert engine._W is None
        sim.open()
        assert engine._W is not None
        assert engine._W.shape == (16, 16)
        sim.close()
        assert not sim._weights_loaded


# ---------------------------------------------------------------------------
# TestQueryBatch
# ---------------------------------------------------------------------------

class TestQueryBatch:
    """Batch query tests."""

    def test_batch_shape(self):
        dim = 16
        engine = MockEngine(dim)
        sim = EmbeddingSimilarity(engine)
        for i in range(5):
            sim.add(f"item_{i}", _random_embedding(dim, seed=i))
        sim._upload_weights()

        queries = np.stack([_random_embedding(dim, seed=i + 100)
                           for i in range(3)])
        results = sim.query_batch(queries, top_k=2)
        assert len(results) == 3
        for r in results:
            assert len(r) == 2

    def test_batch_consistent_with_single(self):
        dim = 16
        engine = MockEngine(dim)
        sim = EmbeddingSimilarity(engine)
        for i in range(5):
            sim.add(f"item_{i}", _random_embedding(dim, seed=i))
        sim._upload_weights()

        queries = np.stack([_random_embedding(dim, seed=i + 100)
                           for i in range(3)])
        batch_results = sim.query_batch(queries, top_k=2)

        for i in range(3):
            single_results = sim.query(queries[i], top_k=2)
            assert len(batch_results[i]) == len(single_results)
            for (bl, bs), (sl, ss) in zip(batch_results[i], single_results):
                assert bl == sl
                assert bs == pytest.approx(ss, abs=1e-6)
