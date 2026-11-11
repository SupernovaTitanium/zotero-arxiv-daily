"""Embedding-based ranking: local sentence-transformers embeddings with a
disk cache, similarity to the nearest corpus papers, weekly-review preference boosts,
and greedy topic clustering for the email."""

from __future__ import annotations

import hashlib
import logging
import threading
import warnings
from pathlib import Path

import numpy as np
from loguru import logger

from .config import Config
from .paper import CorpusPaper, Paper
from .preferences import Preferences, apply_preferences


# ---------------------------------------------------------------------------
# Embedding disk cache
# ---------------------------------------------------------------------------

def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Corpus abstract vectors in an npz file, namespaced by a model key so a
    model/provider switch rebuilds the cache instead of mixing vectors."""

    def __init__(self, path: str | Path, model_key: str):
        self.path = Path(path)
        self.model_key = model_key
        self.vectors: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, path: str | Path, model_key: str) -> "EmbeddingCache":
        cache = cls(path, model_key)
        if not cache.path.exists():
            return cache
        try:
            with np.load(cache.path) as data:
                stored_key = str(data["model_key"])
                if stored_key != model_key:
                    logger.warning(
                        f"Embedding model changed ({stored_key} -> {model_key}); rebuilding embedding cache"
                    )
                    return cache
                hashes = [str(h) for h in data["hashes"].tolist()]
                vectors = np.asarray(data["vectors"], dtype=np.float32)
            if len(hashes) != len(vectors):
                raise ValueError("hash/vector length mismatch")
            cache.vectors = dict(zip(hashes, vectors))
            logger.info(f"Loaded {len(cache.vectors)} cached corpus embeddings from {cache.path}")
        except Exception as e:
            logger.warning(f"Ignoring unreadable embedding cache {cache.path}: {e}")
        return cache

    def get(self, text: str) -> np.ndarray | None:
        return self.vectors.get(text_key(text))

    def put(self, text: str, vector: np.ndarray) -> None:
        self.vectors[text_key(text)] = np.asarray(vector, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.vectors:
            hashes = np.array(list(self.vectors))
            vectors = np.stack(list(self.vectors.values()))
        else:
            hashes = np.array([], dtype="<U64")
            vectors = np.zeros((0,), dtype=np.float32)
        np.savez_compressed(
            self.path,
            model_key=np.array(self.model_key),
            hashes=hashes,
            vectors=vectors,
        )


# ---------------------------------------------------------------------------
# Local embedder
# ---------------------------------------------------------------------------

class LocalEmbedder:
    def __init__(self, model: str, encode_kwargs: dict | None = None):
        self.model = model
        self.encode_kwargs = encode_kwargs or {}
        self._model = None
        self._load_lock = threading.Lock()

    def preload(self) -> None:
        """Start loading the model on a background thread. Importing torch and
        reading the weights takes ~18 s and needs nothing from the pipeline,
        so the caller overlaps it with network waits; ``embed`` blocks until
        the load is done. A load that fails here is attempted again by
        ``embed``, which then raises in the caller's thread."""
        threading.Thread(target=self._encoder, name="embedder-preload", daemon=True).start()

    def model_cache_key(self) -> str:
        # Keep this string stable: it namespaces the on-disk corpus cache.
        params = ",".join(f"{k}={v}" for k, v in sorted(self.encode_kwargs.items()))
        return f"local|{self.model}|{params}"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging
        from huggingface_hub.utils import logging as hf_logging

        transformers_logging.set_verbosity_error()
        hf_logging.set_verbosity_error()
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning)
        return SentenceTransformer(self.model, trust_remote_code=True)

    def _encoder(self):
        with self._load_lock:
            if self._model is None:
                self._model = self._load_model()
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        encoder = self._encoder()
        return np.asarray(encoder.encode(texts, **self.encode_kwargs, show_progress_bar=True))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

# A candidate is scored against the library papers closest to it, not against
# the whole library. A library covers several interests, and an average over
# all of it puts papers that are vaguely near every interest above papers
# squarely inside one. Measured on a 1269-paper library: each of the 300 most
# recently added papers was ranked among 645 new arXiv papers, using only the
# library as it stood a week before the paper was added. 85% reached the top
# 100 with the mean of the 10 nearest papers, 66% with the recency-weighted
# mean over the whole library that this replaced; 5 to 20 nearest papers
# score within one point of 10, and the gain holds for a 90-day gap.
NEAREST_PAPERS = 10


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


class Ranker:
    def __init__(self, config: Config, embedder: LocalEmbedder):
        self.config = config
        self.embedder = embedder
        cache_file = config.executor.embedding_cache_file
        self.embedding_cache = (
            EmbeddingCache.load(cache_file, embedder.model_cache_key()) if cache_file else None
        )

    def _embed_corpus(self, texts: list[str]) -> np.ndarray:
        """Embed corpus texts, reusing cached vectors for unchanged abstracts.
        Entries for papers no longer in the corpus are pruned on save."""
        cache = self.embedding_cache
        if cache is None or not texts:
            return np.asarray(self.embedder.embed(texts), dtype=np.float32)
        stale = set(cache.vectors) - {text_key(t) for t in texts}
        for key in stale:
            del cache.vectors[key]
        vectors: list = [cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing or stale:
            if missing:
                new_vectors = np.asarray(
                    self.embedder.embed([texts[i] for i in missing]), dtype=np.float32
                )
                for i, vec in zip(missing, new_vectors):
                    cache.put(texts[i], vec)
                    vectors[i] = vec
            cache.save()
        logger.info(
            f"Corpus embeddings: {len(texts) - len(missing)} reused from cache, "
            f"{len(missing)} newly embedded, {len(stale)} pruned"
        )
        return np.stack(vectors)

    def rank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        """Score each candidate by its mean cosine similarity to the
        ``NEAREST_PAPERS`` corpus papers closest to it (times 10), and sort
        descending."""
        corpus_matrix = self._embed_corpus([c.abstract for c in corpus])
        candidate_matrix = np.asarray(self.embedder.embed([c.abstract for c in candidates]), dtype=np.float32)
        sim = cosine_similarity(candidate_matrix, corpus_matrix)
        nearest = min(NEAREST_PAPERS, len(corpus))
        scores = np.sort(sim, axis=1)[:, -nearest:].mean(axis=1) * 10
        for score, candidate in zip(scores, candidates):
            candidate.score = float(score)
        return sorted(candidates, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)

    def apply_preferences(self, papers: list[Paper], prefs: Preferences) -> list[Paper]:
        executor = self.config.executor
        before = [p.title for p in papers[:5]]
        apply_preferences(papers, prefs, executor.preference_boost_weight, executor.preference_mute_weight)
        if [p.title for p in papers[:5]] != before:
            logger.info(f"Preferences reordered top papers (boost={prefs.boost}, mute={prefs.mute})")
        return papers

    def assign_topics(self, papers: list[Paper]) -> None:
        """Greedy threshold clustering over candidate embeddings; clusters with
        2+ papers get a topic label (the top paper's title) shown in the email."""
        if not papers:
            return
        try:
            embeddings = np.asarray(
                self.embedder.embed([f"{p.title}. {p.abstract}" for p in papers]), dtype=np.float32
            )
        except Exception as e:
            logger.warning(f"Topic clustering skipped: {e}")
            return
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        threshold = self.config.executor.topic_threshold
        clusters: list[dict] = []  # {"centroid": vec, "members": [idx]}
        for i in range(len(papers)):
            best, best_sim = None, threshold
            for cluster in clusters:
                sim = float(embeddings[i] @ cluster["centroid"])
                if sim >= best_sim:
                    best, best_sim = cluster, sim
            if best is None:
                clusters.append({"centroid": embeddings[i].copy(), "members": [i]})
            else:
                best["members"].append(i)
                best["centroid"] = embeddings[best["members"]].mean(axis=0)
        for cluster in clusters:
            if len(cluster["members"]) < 2:
                continue
            label = papers[cluster["members"][0]].title or ""
            label = label[:50] + ("…" if len(label) > 50 else "")
            for idx in cluster["members"]:
                papers[idx].topic = label
        n_topics = sum(1 for c in clusters if len(c["members"]) >= 2)
        logger.info(f"Grouped {len(papers)} papers into {n_topics} multi-paper topics")
