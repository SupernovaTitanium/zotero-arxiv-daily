import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .construct_email import render_email
from .mailer import send_email
from .personal_summary import generate_teasers_batch, get_summary_mode
from .preferences import apply_preferences, load_preferences
from .protocol import CorpusPaper
from .rate_limit import rate_limit_openai_client
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .state import RecommendedHistory
from .utils import glob_match


def rate_limit_chat_client(client, requests_per_minute: float | int | None, sleep=time.sleep, monotonic=time.monotonic):
    return rate_limit_openai_client(client, requests_per_minute, sleep=sleep, monotonic=monotonic)

def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


def _retry(fn, attempts: int = 3, delay: int = 30):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1:
                raise
            logger.warning(f"Request failed ({e}); retry {attempt + 1}/{attempts} in {delay}s")
            time.sleep(delay)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = rate_limit_openai_client(
            OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url),
            config.llm.get("requests_per_minute", 10),
            max_retries=config.llm.get("rate_limit_max_retries", 5),
            backoff_seconds=config.llm.get("rate_limit_backoff_seconds", 30),
            max_interval_seconds=config.llm.get("rate_limit_max_interval_seconds", 300),
        )
        self.state_file = config.executor.get("state_file", None)
        self.history = (
            RecommendedHistory.load(self.state_file) if self.state_file else None
        )
        self.history_days = int(config.executor.get("history_days", 30) or 30)
        self.output_dir = config.executor.get("output_dir", None)
        prefs_file = config.executor.get("preferences_file", None)
        self.preferences = load_preferences(prefs_file)
        self.pref_boost_weight = float(config.executor.get("preference_boost_weight", 1.0) or 1.0)
        self.pref_mute_weight = float(config.executor.get("preference_mute_weight", 1.5) or 1.5)
        self.topic_threshold = float(config.executor.get("topic_threshold", 0.5) or 0.5)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = _retry(lambda: zot.everything(zot.collections()))
        collections = {c['key']:c for c in collections}
        corpus = _retry(lambda: zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint')))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            # Deleted collections referenced by a paper are skipped, not fatal.
            parts = []
            while col_key and col_key in collections:
                parts.append(collections[col_key]['data']['name'])
                col_key = collections[col_key]['data']['parentCollection'] or None
            return '/'.join(reversed(parts))
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = [p for p in paths if p]
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths'],
            doi=c['data'].get('DOI') or None,
        ) for c in corpus]

    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def build_seen_keys(self, corpus: list[CorpusPaper]) -> set[str]:
        """Keys of papers that must not be recommended: already in the Zotero
        corpus (matched by DOI or title, across sources) or already processed
        in a previous run. The set is shared across retrievers, which add the
        keys of papers they convert, deduplicating within the same run too."""
        seen: set[str] = set()
        for c in corpus:
            seen.update(c.dedup_keys())
        if self.history is not None:
            seen.update(self.history.seen_keys())
        logger.info(f"Deduplicating against {len(seen)} known keys")
        return seen

    def record_history(self, presented: list, all_papers: list) -> None:
        if self.history is None:
            return
        presented_ids = {id(p) for p in presented}
        for p in all_papers:
            self.history.record_paper(
                p.dedup_keys(),
                title=p.title or "",
                abstract=p.abstract or "",
                presented=id(p) in presented_ids,
            )
        self.history.prune(self.history_days)
        self.history.save()
        logger.info(f"Recorded {len(presented)}/{len(all_papers)} presented papers into {self.state_file}")

    def _apply_preferences(self, papers: list) -> list:
        """Apply the weekly-review preference keywords on top of the embedding
        scores, then re-sort. A no-op without a preferences file."""
        if self.preferences is None or self.preferences.is_empty():
            return papers
        before = [p.title for p in papers[:5]]
        apply_preferences(papers, self.preferences, self.pref_boost_weight, self.pref_mute_weight)
        after = [p.title for p in papers[:5]]
        if before != after:
            logger.info(f"Preferences reordered top papers (boost={self.preferences.boost}, mute={self.preferences.mute})")
        return papers

    def _assign_topics(self, papers: list) -> None:
        """Greedy threshold clustering over candidate embeddings; clusters with
        2+ papers get a topic label (the top paper's title) shown in the email."""
        if not papers:
            return
        try:
            embeddings = np.asarray(self.reranker.embed([f"{p.title}. {p.abstract}" for p in papers]), dtype=np.float32)
        except Exception as e:
            logger.warning(f"Topic clustering skipped: {e}")
            return
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        clusters: list[dict] = []  # {"centroid": vec, "members": [idx]}
        for i in range(len(papers)):
            best, best_sim = None, self.topic_threshold
            for c in clusters:
                sim = float(embeddings[i] @ c["centroid"])
                if sim >= best_sim:
                    best, best_sim = c, sim
            if best is None:
                clusters.append({"centroid": embeddings[i].copy(), "members": [i]})
            else:
                members = best["members"]
                members.append(i)
                best["centroid"] = embeddings[members].mean(axis=0)
        for c in clusters:
            if len(c["members"]) < 2:
                continue
            label = papers[c["members"][0]].title or ""
            label = label[:50] + ("…" if len(label) > 50 else "")
            for idx in c["members"]:
                papers[idx].topic = label
        n_topics = sum(1 for c in clusters if len(c["members"]) >= 2)
        logger.info(f"Grouped {len(papers)} papers into {n_topics} multi-paper topics")

    def _fetch_full_texts(self, papers: list) -> None:
        """Two-stage pipeline: fetch full text only for the top ranked papers,
        in parallel. The rest keep metadata-only summaries from the abstract."""
        top_n = int(self.config.executor.get("fulltext_paper_num", 30) or 0)
        workers = max(1, int(self.config.executor.get("fulltext_workers", 4) or 4))
        targets = papers[:top_n]
        if not targets:
            return

        def _fetch(paper):
            retriever = self.retrievers.get(paper.source)
            if retriever is None:
                return paper, None
            return paper, retriever.fetch_full_text(paper)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch, p) for p in targets]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching full texts"):
                paper, full_text = future.result()
                if full_text:
                    paper.full_text = full_text
        got = sum(1 for p in targets if p.full_text)
        logger.info(f"Full text fetched for {got}/{len(targets)} top papers")

    def _write_outputs(self, email_html: str, run_summary: dict) -> None:
        if not self.output_dir:
            return
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"email_{run_summary['date']}.html").write_text(email_html, encoding="utf-8")
        (out / f"run_summary_{run_summary['date']}.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Wrote run outputs to {out}")

    def run(self):
        timings: dict[str, float] = {}
        t_start = time.monotonic()

        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        seen_keys = self.build_seen_keys(corpus)

        t_retrieve = time.monotonic()
        all_papers = []
        retrieved_counts: dict[str, int] = {}
        skipped_counts: dict[str, int] = {}
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers(seen_keys=seen_keys)
            retrieved_counts[source] = len(papers)
            skipped_counts[source] = getattr(retriever, "last_skipped", 0)
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        timings["retrieve"] = round(time.monotonic() - t_retrieve, 1)

        reranked_papers = []
        presented = []
        llm_requests = None
        if len(all_papers) > 0:
            t_rerank = time.monotonic()
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = self._apply_preferences(reranked_papers)
            timings["rerank"] = round(time.monotonic() - t_rerank, 1)
            presented = reranked_papers[:self.config.executor.max_paper_num]
            self._assign_topics(presented)

            t_fulltext = time.monotonic()
            self._fetch_full_texts(presented)
            timings["fulltext"] = round(time.monotonic() - t_fulltext, 1)

            t_tldr = time.monotonic()
            mode = get_summary_mode(self.config.llm)
            if mode == "teaser":
                logger.info("Generating teasers (batched)...")
                llm_requests = generate_teasers_batch(self.openai_client, self.config.llm, presented)
            else:
                logger.info("Generating TLDR and affiliations...")
                for p in tqdm(presented):
                    p.generate_tldr(self.openai_client, self.config.llm)
                if mode in {"full", "legacy"}:
                    for p in presented:
                        p.generate_affiliations(self.openai_client, self.config.llm)
            timings["tldr"] = round(time.monotonic() - t_tldr, 1)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return

        timings["total"] = round(time.monotonic() - t_start, 1)
        run_summary = {
            "date": date.today().isoformat(),
            "timings_seconds": timings,
            "counts": {
                "corpus": len(corpus),
                "retrieved": retrieved_counts,
                "dedup_skipped": skipped_counts,
                "ranked": len(reranked_papers),
                "presented": min(len(reranked_papers), self.config.executor.max_paper_num),
            },
            "llm_requests": llm_requests,
            "papers": [
                {
                    "rank": i + 1,
                    "title": p.title,
                    "source": p.source,
                    "score": round(p.score, 4) if p.score is not None else None,
                    "url": p.url,
                    "presented": i < self.config.executor.max_paper_num,
                    "has_full_text": bool(p.full_text),
                    "topic": getattr(p, "topic", None),
                    "summary": (p.teaser or p.tldr or "")[:300],
                }
                for i, p in enumerate(reranked_papers)
            ],
        }

        logger.info("Sending email...")
        email_content = render_email(reranked_papers[:self.config.executor.max_paper_num], self.config.llm.summary)
        # Outputs are written before sending, so a failed send still leaves the
        # rendered email and full ranking behind for inspection.
        self._write_outputs(email_content, run_summary)
        send_email(self.config, email_content)
        logger.info(f"Email sent successfully ({run_summary['counts']}, {timings})")
        # Only persist history after the email is delivered, so a failed run
        # does not silently swallow that day's papers.
        self.record_history(presented, all_papers)
