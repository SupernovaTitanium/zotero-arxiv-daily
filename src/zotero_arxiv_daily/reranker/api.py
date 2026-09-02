from .base import BaseReranker, register_reranker
from ..rate_limit import rate_limit_openai_client
from openai import OpenAI
import numpy as np
@register_reranker("api")
class ApiReranker(BaseReranker):
    def model_cache_key(self) -> str:
        return f"api|{self.config.reranker.api.base_url}|{self.config.reranker.api.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        client = rate_limit_openai_client(
            OpenAI(api_key=self.config.reranker.api.key, base_url=self.config.reranker.api.base_url),
            self.config.reranker.api.get("requests_per_minute"),
            max_retries=self.config.reranker.api.get("rate_limit_max_retries", 5),
            backoff_seconds=self.config.reranker.api.get("rate_limit_backoff_seconds", 30),
            max_interval_seconds=self.config.reranker.api.get("rate_limit_max_interval_seconds", 300),
        )
        batch_size = self.config.reranker.api.get("batch_size") or 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.config.reranker.api.model
            )
            all_embeddings.extend([r.embedding for r in response.data])
        return np.array(all_embeddings, dtype=np.float32)
