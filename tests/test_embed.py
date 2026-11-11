"""Tests for ranking: cosine similarity, nearest-paper scoring, cache, topic clustering."""

import numpy as np
import pytest

from zotero_arxiv_daily.embed import EmbeddingCache, Ranker, cosine_similarity
from zotero_arxiv_daily.paper import CorpusPaper
from tests.canned_responses import make_sample_paper


class FakeEmbedder:
    """Deterministic embeddings: a paper maps to a one-hot-ish vector keyed by
    its first abstract word, so similarity is fully predictable."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def model_cache_key(self) -> str:
        return "fake|dim"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            key = text.split()[0] if text else "empty"
            vec[hash(key) % self.dim] = 1.0
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


def test_cosine_similarity_normalizes():
    sim = cosine_similarity(np.asarray([[3.0, 0.0]]), np.asarray([[10.0, 0.0], [0.0, 5.0]]))
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(0.0)


class TableEmbedder:
    """Embeds a text as the vector listed for its first word."""

    def __init__(self, vectors: dict):
        self.vectors = vectors

    def model_cache_key(self) -> str:
        return "table"

    def embed(self, texts):
        return np.asarray([self.vectors[text.split()[0]] for text in texts], dtype=np.float32)


def _corpus_paper(abstract: str) -> CorpusPaper:
    return CorpusPaper("C", abstract, added_date=__import__("datetime").datetime(2026, 1, 1), paths=[])


def test_rank_orders_by_corpus_similarity(config):
    embedder = TableEmbedder({"alpha": [1, 0, 0], "beta": [0, 1, 0], "gamma": [0, 0, 1]})
    corpus = [_corpus_paper("alpha text"), _corpus_paper("alpha again"), _corpus_paper("beta text")]
    candidates = [make_sample_paper(title="Gamma paper", abstract="gamma text"),
                  make_sample_paper(title="Beta paper", abstract="beta text"),
                  make_sample_paper(title="Alpha paper", abstract="alpha text")]
    ranked = Ranker(config, embedder).rank(candidates, corpus)
    # a corpus smaller than NEAREST_PAPERS is averaged whole
    assert [p.title for p in ranked] == ["Alpha paper", "Beta paper", "Gamma paper"]
    assert [p.score for p in ranked] == pytest.approx([10 * 2 / 3, 10 * 1 / 3, 0.0])


def test_rank_prefers_a_paper_inside_one_interest_to_one_between_two(config):
    """The library holds a small interest and a large one. A candidate squarely
    in the small one must beat a candidate halfway between the two; an average
    over the whole library puts the halfway paper first (0.71 against 0.29)."""
    embedder = TableEmbedder({"small": [1, 0], "large": [0, 1], "between": [1, 1]})
    corpus = [_corpus_paper(f"small {i}") for i in range(12)] + [_corpus_paper(f"large {i}") for i in range(30)]
    candidates = [make_sample_paper(title="Between", abstract="between the two"),
                  make_sample_paper(title="Inside", abstract="small interest")]
    ranked = Ranker(config, embedder).rank(candidates, corpus)
    assert [p.title for p in ranked] == ["Inside", "Between"]
    assert ranked[0].score == pytest.approx(10.0)
    assert ranked[1].score == pytest.approx(10 * 0.5 ** 0.5)


def test_embedding_cache_roundtrip_and_model_key_invalidation(tmp_path):
    path = tmp_path / "cache.npz"
    cache = EmbeddingCache(path, "modelA")
    cache.put("hello", np.asarray([1.0, 2.0]))
    cache.save()

    reloaded = EmbeddingCache.load(path, "modelA")
    assert reloaded.get("hello") is not None
    # a different model key rebuilds the cache instead of mixing vectors
    switched = EmbeddingCache.load(path, "modelB")
    assert switched.get("hello") is None


def test_embed_corpus_reuses_cache_and_prunes_stale(config, tmp_path):
    config.executor.embedding_cache_file = str(tmp_path / "cache.npz")
    embedder = FakeEmbedder()
    ranker = Ranker(config, embedder)

    old_texts = ["alpha text", "beta text"]
    ranker._embed_corpus(old_texts)
    cache = ranker.embedding_cache
    assert len(cache.vectors) == 2

    # one text replaced: its vector stays cached, the stale one is pruned
    ranker._embed_corpus(["alpha text", "gamma text"])
    assert len(cache.vectors) == 2


def test_assign_topics_groups_similar_papers(config):
    class SameVectorEmbedder(FakeEmbedder):
        def embed(self, texts):
            vec = np.zeros((len(texts), 4), dtype=np.float32)
            vec[:, 0] = 1.0
            return vec

    config.executor.topic_threshold = 0.5
    ranker = Ranker(config, SameVectorEmbedder())
    papers = [
        make_sample_paper(title="Paper one", abstract="a"),
        make_sample_paper(title="Paper two", abstract="b"),
        make_sample_paper(title="Paper three", abstract="c"),
    ]
    ranker.assign_topics(papers)
    assert all(p.topic == "Paper one" for p in papers)


def test_assign_topics_handles_embedder_failure(config):
    class BrokenEmbedder(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("boom")

    ranker = Ranker(config, BrokenEmbedder())
    papers = [make_sample_paper(title="t", abstract="a")]
    ranker.assign_topics(papers)  # must not raise
    assert papers[0].topic is None


def test_model_is_loaded_once(monkeypatch):
    from zotero_arxiv_daily.embed import LocalEmbedder

    loads = []

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 4), dtype=np.float32)

    def fake_load(self):
        loads.append(1)
        return FakeEncoder()

    monkeypatch.setattr(LocalEmbedder, "_load_model", fake_load)
    embedder = LocalEmbedder("fake-model")
    embedder.embed(["a"])
    embedder.embed(["b", "c"])
    assert len(loads) == 1


def test_embed_waits_for_a_preload_instead_of_loading_twice(monkeypatch):
    import threading

    from zotero_arxiv_daily.embed import LocalEmbedder

    loads = []
    loading = threading.Event()
    finish = threading.Event()

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 4), dtype=np.float32)

    def slow_load(self):
        loads.append(1)
        loading.set()
        assert finish.wait(timeout=10)
        return FakeEncoder()

    monkeypatch.setattr(LocalEmbedder, "_load_model", slow_load)
    embedder = LocalEmbedder("fake-model")
    embedder.preload()
    assert loading.wait(timeout=10)  # the background load is under way
    threading.Timer(0.05, finish.set).start()
    assert embedder.embed(["a", "b"]).shape == (2, 4)  # blocks until that load is done
    assert len(loads) == 1
