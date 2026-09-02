from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
@register_reranker("local")
class LocalReranker(BaseReranker):
    def model_cache_key(self) -> str:
        encode_kwargs = self.config.reranker.local.encode_kwargs or {}
        params = ",".join(f"{k}={v}" for k, v in sorted(encode_kwargs.items()))
        return f"local|{self.config.reranker.local.model}|{params}"

    def _build_encoder(self):
        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        return SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        encoder = self._build_encoder()
        encode_kwargs = (
            self.config.reranker.local.encode_kwargs
            if self.config.reranker.local.encode_kwargs
            else {}
        )
        return np.asarray(encoder.encode(texts, **encode_kwargs, show_progress_bar=True))
