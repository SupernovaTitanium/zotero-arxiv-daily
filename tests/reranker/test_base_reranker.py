"""Tests for BaseReranker: scoring, sorting, time decay, embedding cache, unknown reranker."""

import numpy as np
import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.reranker.base import BaseReranker, get_reranker_cls
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker whose embed() returns preassigned vectors per text.

    Bypasses BaseReranker.__init__ (no embedding cache), like the old
    fixed-similarity-matrix stub did.
    """

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.config = None
        self._vectors = vectors
        self.embed_calls: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return np.array([self._vectors[t] for t in texts], dtype=float)


class CacheStubReranker(BaseReranker):
    """Stub that goes through BaseReranker.__init__, so the embedding cache loads."""

    def __init__(self, config, vectors: dict[str, np.ndarray]):
        super().__init__(config)
        self._vectors = vectors
        self.embed_calls: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        return np.array([self._vectors[t] for t in texts], dtype=float)


def _basis_vectors(corpus, dim: int) -> dict[str, np.ndarray]:
    """Map each corpus abstract to a unit basis vector by recency (newest first)."""
    order = sorted(corpus, key=lambda x: x.added_date, reverse=True)
    vectors = {}
    for i, c in enumerate(order):
        v = np.zeros(dim)
        v[i] = 1.0
        vectors[c.abstract] = v
    return vectors


def test_rerank_scores_and_sorts():
    dim = 4
    corpus = make_sample_corpus(3)
    vectors = _basis_vectors(corpus, dim)
    # candidate 0 is orthogonal to the whole corpus; candidate 1 points at it
    vectors["abs zero"] = np.array([0.0, 0.0, 0.0, 1.0])
    vectors["abs one"] = np.array([1.0, 1.0, 1.0, 0.0])

    papers = [
        make_sample_paper(title="Paper 0", abstract="abs zero"),
        make_sample_paper(title="Paper 1", abstract="abs one"),
    ]
    ranked = StubReranker(vectors).rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score


def test_rerank_time_decay_weighting():
    dim = 3
    corpus = make_sample_corpus(3)
    vectors = _basis_vectors(corpus, dim)
    vectors["aligned old"] = vectors[sorted(corpus, key=lambda x: x.added_date)[0].abstract]
    vectors["aligned new"] = vectors[sorted(corpus, key=lambda x: x.added_date, reverse=True)[0].abstract]

    ranked_old = StubReranker(vectors).rerank(
        [make_sample_paper(title="P", abstract="aligned old")], corpus
    )
    ranked_new = StubReranker(vectors).rerank(
        [make_sample_paper(title="P", abstract="aligned new")], corpus
    )
    # Same raw similarity (1.0), but the newer corpus paper has higher decay weight
    assert ranked_new[0].score > ranked_old[0].score


def test_rerank_single_candidate_single_corpus():
    dim = 2
    corpus = make_sample_corpus(1)
    vectors = _basis_vectors(corpus, dim)
    vectors["same"] = vectors[corpus[0].abstract]
    ranked = StubReranker(vectors).rerank([make_sample_paper(abstract="same")], corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_rerank_reuses_cached_corpus_embeddings(config, tmp_path):
    dim = 3
    corpus = make_sample_corpus(3)
    vectors = _basis_vectors(corpus, dim)
    vectors["candidate text"] = vectors[corpus[0].abstract]

    cache_file = tmp_path / "corpus_embeddings.npz"
    with open_dict(config.reranker):
        config.reranker.corpus_cache_file = str(cache_file)

    reranker = CacheStubReranker(config, vectors)
    ranked_first = reranker.rerank([make_sample_paper(abstract="candidate text")], corpus)
    assert cache_file.exists()
    # first run embeds the corpus and the candidate
    assert len(reranker.embed_calls) == 2

    # second run: corpus abstracts all come from the cache, only the candidate is embedded
    ranked_second = reranker.rerank([make_sample_paper(abstract="candidate text")], corpus)
    assert reranker.embed_calls[-1] == ["candidate text"]
    assert ranked_second[0].score == pytest.approx(ranked_first[0].score)


def test_rerank_embeds_only_new_corpus_papers(config, tmp_path):
    dim = 3
    corpus = make_sample_corpus(3)
    vectors = _basis_vectors(corpus, dim)
    vectors["candidate text"] = np.array([1.0, 0.0, 0.0])
    vectors["brand new abstract"] = np.array([0.0, 1.0, 0.0])

    cache_file = tmp_path / "corpus_embeddings.npz"
    with open_dict(config.reranker):
        config.reranker.corpus_cache_file = str(cache_file)

    reranker = CacheStubReranker(config, vectors)
    reranker.rerank([make_sample_paper(abstract="candidate text")], corpus)

    grown_corpus = corpus + [make_sample_corpus(3)[0]]
    grown_corpus[-1].abstract = "brand new abstract"
    reranker.rerank([make_sample_paper(abstract="candidate text")], grown_corpus)
    assert reranker.embed_calls[-1] == ["candidate text"]  # candidate embed
    assert reranker.embed_calls[-2] == ["brand new abstract"]  # only the new corpus abstract


def test_rerank_rebuilds_cache_when_model_changes(config, tmp_path):
    dim = 3
    corpus = make_sample_corpus(3)
    vectors = _basis_vectors(corpus, dim)
    vectors["candidate text"] = vectors[corpus[0].abstract]

    cache_file = tmp_path / "corpus_embeddings.npz"
    with open_dict(config.reranker):
        config.reranker.corpus_cache_file = str(cache_file)

    CacheStubReranker(config, vectors).rerank([make_sample_paper(abstract="candidate text")], corpus)
    assert cache_file.exists()

    # simulate a model switch by changing the cache's model key
    from zotero_arxiv_daily.embedding_cache import EmbeddingCache
    stale = EmbeddingCache.load(cache_file, model_key="some-other-model")
    assert stale.vectors == {}  # mismatched key -> cache ignored

    reranker = CacheStubReranker(config, vectors)  # same key as before: reuse
    reranker.rerank([make_sample_paper(abstract="candidate text")], corpus)
    assert reranker.embed_calls[-1] == ["candidate text"]


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")
