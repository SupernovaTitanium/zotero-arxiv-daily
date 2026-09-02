"""Disk cache of corpus abstract embeddings.

Corpus abstracts barely change between runs, so their vectors are stored in a
npz file keyed by sha256(text). The file is namespaced by a model key (model
name plus relevant parameters), so switching the embedding model or provider
invalidates the cache automatically instead of mixing incompatible vectors.
"""

import hashlib
from pathlib import Path

import numpy as np
from loguru import logger


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
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
