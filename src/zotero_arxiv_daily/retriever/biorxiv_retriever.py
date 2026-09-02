from datetime import datetime, timedelta

import requests
from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from ..utils import normalize_doi, normalize_title
from loguru import logger
from typing import Any
from time import sleep

@register_retriever("biorxiv")
class BiorxivRetriever(BaseRetriever):
    server = "biorxiv"

    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config.category is None:
            raise ValueError(f"category must be specified for {self.name}")

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        lookback_days = int(self.config.executor.get("lookback_days", 1) or 1)
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=lookback_days - 1)
        retry_num = 10
        delay_time = 10
        collection: list[dict[str, Any]] = []
        cursor = 0
        # The details API returns at most 100 records per page; follow the cursor
        # until the date range is exhausted (with a sane page cap).
        for _ in range(100):
            api_url = f"https://api.biorxiv.org/details/{self.server}/{from_date}/{to_date}/{cursor}"
            for i in range(retry_num):
                try:
                    response = requests.get(api_url)
                    response.raise_for_status()
                    break
                except Exception as e:
                    if i == retry_num - 1:
                        raise e
                    else:
                        logger.warning(f"Failed to retrieve papers: {str(e)}. Retry in {delay_time} seconds.")
                        sleep(delay_time)
            result = response.json()
            page = result.get('collection', [])
            collection.extend(page)
            messages = result.get('messages', [{}])[0]
            total = int(messages.get('total', 0) or 0)
            next_cursor = str(messages.get('cursor', '') or '')
            if len(page) == 0 or not next_cursor or len(collection) >= total > 0:
                break
            cursor = next_cursor
            sleep(1)
        if len(collection) == 0:
            logger.warning(f"No paper found in {from_date}~{to_date}. API Message: {result.get('messages')}")
            return []
        categories = [c.lower() for c in self.retriever_config.category]
        collection = [c for c in collection if c['category'].lower() in categories]
        if self.config.executor.debug:
            collection = collection[:10]
        return collection

    def _raw_keys(self, raw_paper: dict[str, Any]) -> list[str]:
        keys = []
        if raw_paper.get('doi'):
            keys.append("doi:" + normalize_doi(raw_paper['doi']))
        if raw_paper.get('title'):
            keys.append("title:" + normalize_title(raw_paper['title']))
        return keys


    def convert_to_paper(self, raw_paper:dict[str, Any]) -> Paper | None:
        title = raw_paper['title']
        authors = [a.strip() for a in raw_paper['authors'].split(';')]
        abstract = raw_paper['abstract']
        pdf_url = f"https://www.{self.server}.org/content/{raw_paper['doi']}v{raw_paper['version']}.full.pdf"
        full_text = None # biorxiv forbids scraping its pdf
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=pdf_url,
            pdf_url=pdf_url,
            full_text=full_text,
            doi=raw_paper['doi'],
            source_id=raw_paper['doi'],
        )
