"""Tests for EmbeddingCache."""

import numpy as np

from zotero_arxiv_daily.embedding_cache import EmbeddingCache, text_key


def test_load_missing_file_returns_empty(tmp_path):
    cache = EmbeddingCache.load(tmp_path / "emb.npz", model_key="m1")
    assert cache.get("anything") is None


def test_put_get_save_reload_roundtrip(tmp_path):
    path = tmp_path / "emb.npz"
    cache = EmbeddingCache(path, model_key="m1")
    cache.put("hello world", np.array([1.0, 2.0, 3.0]))
    cache.save()

    reloaded = EmbeddingCache.load(path, model_key="m1")
    assert reloaded.get("hello world") is not None
    assert np.allclose(reloaded.get("hello world"), [1.0, 2.0, 3.0])
    assert reloaded.get("different text") is None


def test_model_key_mismatch_invalidates(tmp_path):
    path = tmp_path / "emb.npz"
    cache = EmbeddingCache(path, model_key="model-a")
    cache.put("text", np.array([0.5, 0.5]))
    cache.save()

    other = EmbeddingCache.load(path, model_key="model-b")
    assert other.get("text") is None
    assert other.vectors == {}


def test_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "emb.npz"
    path.write_bytes(b"not an npz file")
    cache = EmbeddingCache.load(path, model_key="m1")
    assert cache.vectors == {}


def test_text_key_is_stable_and_content_addressed():
    assert text_key("abc") == text_key("abc")
    assert text_key("abc") != text_key("abd")
