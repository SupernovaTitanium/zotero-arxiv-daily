from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, RawPaperItem
from ..utils import normalize_title
from tqdm import tqdm
from typing import Type
from time import sleep
from loguru import logger


class BaseRetriever(ABC):
    name: str
    def __init__(self, config:DictConfig):
        self.config = config
        self.retriever_config = getattr(config.source,self.name)

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[RawPaperItem]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper:RawPaperItem) -> Paper | None:
        pass

    def _raw_keys(self, raw_paper:RawPaperItem) -> list[str]:
        """Dedup keys of a raw paper. Filtering happens before convert_to_paper,
        so keys must be computable without full-text extraction."""
        title = getattr(raw_paper, "title", None)
        if title is None and isinstance(raw_paper, dict):
            title = raw_paper.get("title")
        return ["title:" + normalize_title(title)] if title else []

    def retrieve_papers(self, seen_keys:set[str]|None=None) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        logger.info("Processing papers...")
        papers = []
        skipped = 0
        for raw_paper in tqdm(raw_papers, total=len(raw_papers), desc="Converting papers"):
            if seen_keys and set(self._raw_keys(raw_paper)) & seen_keys:
                skipped += 1
                continue
            try:
                paper = self.convert_to_paper(raw_paper)
            except Exception as exc:
                logger.warning(f"Skipping paper {getattr(raw_paper, 'title', raw_paper)}: {exc}")
                continue
            if paper is not None:
                papers.append(paper)
            sleep(1)
        if skipped:
            logger.info(f"Skipped {skipped} papers already seen (recommended before or already in Zotero)")
        return papers

registered_retrievers = {}

def register_retriever(name:str):
    def decorator(cls):
        registered_retrievers[name] = cls
        cls.name = name
        return cls
    return decorator

def get_retriever_cls(name:str) -> Type[BaseRetriever]:
    if name not in registered_retrievers:
        raise ValueError(f"Retriever {name} not found")
    return registered_retrievers[name]
