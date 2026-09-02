from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
from ..embedding_cache import EmbeddingCache, text_key
import numpy as np
from typing import Type
from loguru import logger


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config
        cache_file = config.reranker.get("corpus_cache_file", None)
        self.embedding_cache = (
            EmbeddingCache.load(cache_file, self.model_cache_key()) if cache_file else None
        )

    def model_cache_key(self) -> str:
        """Identity of the embedding space. A cache built with a different key
        cannot be reused, so switching model or provider rebuilds it."""
        return type(self).__name__

    @abstractmethod
    def embed(self, texts:list[str]) -> np.ndarray:
        """Embed texts into vectors of shape [n, d] (comparable by cosine)."""
        raise NotImplementedError

    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        return cosine_similarity(self.embed(s1), self.embed(s2))

    def _embed_corpus(self, texts:list[str]) -> np.ndarray:
        """Embed corpus texts, reusing cached vectors for unchanged abstracts.
        Entries for papers no longer in the corpus are pruned on save."""
        cache = getattr(self, "embedding_cache", None)
        if cache is None or not texts:
            return np.asarray(self.embed(texts), dtype=np.float32)
        stale = set(cache.vectors) - {text_key(t) for t in texts}
        for key in stale:
            del cache.vectors[key]
        vectors: list = [cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing or stale:
            if missing:
                new_vectors = np.asarray(self.embed([texts[i] for i in missing]), dtype=np.float32)
                for i, vec in zip(missing, new_vectors):
                    cache.put(texts[i], vec)
                    vectors[i] = vec
            cache.save()
        logger.info(
            f"Corpus embeddings: {len(texts) - len(missing)} reused from cache, "
            f"{len(missing)} newly embedded, {len(stale)} pruned"
        )
        return np.stack(vectors)

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        corpus_matrix = self._embed_corpus([c.abstract for c in corpus])
        candidate_matrix = np.asarray(self.embed([c.abstract for c in candidates]), dtype=np.float32)
        sim = cosine_similarity(candidate_matrix, corpus_matrix)
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
