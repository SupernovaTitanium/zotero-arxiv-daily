from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
from openai import OpenAI

from .utils import normalize_doi, normalize_title
from .summarize import generate_affiliations_for_paper, generate_tldr_for_paper

RawPaperItem = TypeVar('RawPaperItem')


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    teaser: Optional[str] = None
    tldr_markdown: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    doi: Optional[str] = None
    source_id: Optional[str] = None

    def dedup_keys(self) -> list[str]:
        keys = []
        if self.doi:
            keys.append("doi:" + normalize_doi(self.doi))
        if self.title:
            keys.append("title:" + normalize_title(self.title))
        if self.source_id:
            keys.append(f"sid:{self.source}:{self.source_id}")
        return keys

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        return generate_tldr_for_paper(self, openai_client, llm_params)

    def generate_affiliations(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        return generate_affiliations_for_paper(self, openai_client, llm_params)

@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
    doi: Optional[str] = None

    def dedup_keys(self) -> list[str]:
        keys = []
        if self.doi:
            keys.append("doi:" + normalize_doi(self.doi))
        if self.title:
            keys.append("title:" + normalize_title(self.title))
        return keys
