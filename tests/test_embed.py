"""Tests for ranking: cosine similarity, time decay, cache, topic clustering."""

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


def test_rank_orders_by_weighted_corpus_similarity(config):
    corpus = [
        CorpusPaper("C1", "alpha text", added_date=__import__("datetime").datetime(2026, 3, 1), paths=[]),
        CorpusPaper("C2", "beta text", added_date=__import__("datetime").datetime(2026, 1, 1), paths=[]),
    ]
    candidates = [make_sample_paper(title="Alpha paper", abstract="alpha text"),
                  make_sample_paper(title="Beta paper", abstract="beta text")]
    ranker = Ranker(config, FakeEmbedder())
    ranked = ranker.rank(candidates, corpus)
    # alpha matches the newer corpus paper, so it must come first
    assert ranked[0].title == "Alpha paper"
    assert ranked[0].score is not None and ranked[0].score > 0


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
