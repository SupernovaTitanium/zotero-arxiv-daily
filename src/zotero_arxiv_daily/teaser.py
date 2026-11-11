"""Batched teaser generation: one LLM request per batch of papers, falling
back to per-paper requests for papers missing from a malformed response."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from openai import OpenAI, OpenAIError

from .config import LlmConfig
from .rate_limit import rate_limit_openai_client

# A batch of 10 papers takes 16-30 s on the local proxy. Left to its default
# the client waits 600 s for a response, and repeats a request twice, so one
# stuck request could hold a run for half an hour. The two quick repeats are
# kept: they absorb a passing 502 from the proxy ("server_is_overloaded",
# seen 2026-11-11), which would otherwise cost the pipeline's 60 s wait.
REQUEST_TIMEOUT_SECONDS = 180
REQUEST_RETRIES = 2


class TokenUsage:
    """Requests made through one client, failed ones included, and the
    tokens the API reported for those that succeeded. A normal run of 100
    papers is 10 requests, about 33,000 prompt tokens (nine tenths of them
    the abstracts themselves) and 8,000 completion tokens; a number far
    above that means requests are being repeated."""

    def __init__(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    def count_request(self) -> None:
        with self._lock:
            self.requests += 1

    def add(self, usage) -> None:
        if usage is None:  # not every OpenAI-compatible server reports usage
            return
        with self._lock:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0


def make_llm_client(llm: LlmConfig) -> OpenAI:
    """One rate-limited client per run: the limiter's adaptive minimum-interval
    state must be shared across all calls, so never build a client per request.
    The client carries the run's token count as ``token_usage``."""
    client = rate_limit_openai_client(
        OpenAI(api_key=llm.api_key, base_url=llm.base_url, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=REQUEST_RETRIES),
        llm.requests_per_minute,
        max_retries=llm.rate_limit_max_retries,
        backoff_seconds=llm.rate_limit_backoff_seconds,
        max_interval_seconds=llm.rate_limit_max_interval_seconds,
    )
    client.token_usage = TokenUsage()
    return client


def _chat(client: OpenAI, llm: LlmConfig, system: str, prompt: str) -> str:
    # Streaming is deliberately not supported here: non-streaming keeps batch
    # JSON parsing simple and the rate limiter's per-request accounting exact.
    extra = {k: v for k, v in llm.generation_kwargs.items() if k != "stream"}
    token_usage = getattr(client, "token_usage", None)
    if token_usage is not None:
        token_usage.count_request()
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        model=llm.model,
        max_tokens=llm.max_tokens,
        **extra,
    )
    if token_usage is not None:
        token_usage.add(getattr(response, "usage", None))
    return response.choices[0].message.content or ""


def _plain(teaser: str) -> str:
    """Undo TeX escapes the model copied from the abstract along with a
    number: about 5% of arXiv abstracts write percentages as ``53.1\\%``."""
    return re.sub(r"\\([%&_#$])", r"\1", teaser)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _paper_context(title: str, abstract: str) -> str:
    return f"題目：{title or '[來源缺失]'}\n\n摘要：{abstract or '[來源缺失]'}"


# Why the prompt names the reader and asks for the result: with "研究問題、
# 核心方法或新意、為什麼值得看" alone, 62-85% of a day's teasers opened with a
# question, 9 of 100 carried a number although about 26 abstracts reported
# one, and results were replaced by "展現潛力" or "值得看其…" endings.
_READER = (
    "讀者是一位研究者，每天早上掃過約一百條速覽，靠每一條決定要不要打開那篇論文，"
    "所以每一條都要短，並讓他看出論文做了什麼、得到什麼結果；空泛的評語對他沒有用。"
)


def _requirements(llm: LlmConfig, each: str) -> str:
    """The lines shared by the batch prompt (``each`` = "每段") and the
    single-paper prompt (``each`` empty)."""
    # Models do not count characters: asked for the whole limit, 3 teasers in
    # 100 ran past it and were cut in mid-sentence. Asking for two thirds of
    # it leaves the cut as a guard that should not fire.
    target = llm.teaser_char_limit * 2 // 3
    return (
        f"- 使用 {llm.language}。\n"
        f"- {each}{target} 個字以內，寫成一到兩句直述句，不要用問句。\n"
        f"- {each}依序交代：研究問題、核心方法或新意、最主要的一項結果。\n"
        "- 摘要給出關鍵數字時，寫出那一項結果的數字和比較對象，數字照摘要原樣抄寫；其餘數字省略。\n"
        "- 保留摘要對結果的限定：和誰比、在什麼條件下、是否只在部分情況成立。\n"
        "- 摘要沒有給出結果時，只寫問題與方法；不要用「展現潛力」「值得關注」這類評語補字數。\n"
        "- 不要誇大，不要加入輸入中沒有的結論。\n"
    )


def generate_teaser(client: OpenAI, llm: LlmConfig, title: str, abstract: str) -> str:
    prompt = (
        "你是嚴謹的學術每日摘要編輯。請寫一段極短速覽。\n"
        f"{_READER}\n\n"
        "要求：\n"
        f"{_requirements(llm, '')}"
        "- 不要使用 Markdown。\n"
        "- 只輸出速覽文字。\n\n"
        f"可用資料：\n"
        f"{_paper_context(_clip(title, 500), _clip(abstract, 3000))}"
    )
    teaser = _plain(_chat(client, llm, "你是一個精簡的學術摘要專家。", prompt)).strip()
    return teaser[: llm.teaser_char_limit].rstrip()


def _teasers_for_batch(client: OpenAI, llm: LlmConfig, batch: list) -> dict[int, str]:
    lines = [
        f"[{i}] " + _paper_context(_clip(p.title, 500), _clip(p.abstract or "", 2000))
        for i, p in enumerate(batch)
    ]
    prompt = (
        "你是嚴謹的學術每日摘要編輯。請為下列每一篇論文各寫一段極短速覽。\n"
        f"{_READER}\n\n"
        "要求：\n"
        f"{_requirements(llm, '每段')}"
        "- 輸出 JSON 陣列，依論文編號排序，格式："
        '[{"index": 0, "teaser": "..."}, {"index": 1, "teaser": "..."}]\n'
        "- 只輸出 JSON，不要其他文字。\n\n"
        "論文清單：\n" + "\n\n".join(lines)
    )
    raw = _chat(client, llm, "你是一個精簡的學術摘要專家。", prompt)
    start, end = raw.find("["), raw.rfind("]")
    parsed = json.loads(raw[start : end + 1])
    if isinstance(parsed, dict):
        parsed = parsed.get("results", parsed.get("teasers", []))
    teasers = {}
    for item in parsed:
        text = _plain(str(item.get("teaser", ""))).strip()
        if text:
            teasers[int(item["index"])] = text[: llm.teaser_char_limit].rstrip()
    return teasers


def generate_teasers_batch(client: OpenAI, llm: LlmConfig, papers: list) -> int:
    """Generate teasers for many papers with one LLM request per batch.

    Sets ``paper.teaser`` for every paper; papers missing from a malformed
    batch response fall back to one-by-one generation. Returns the number of
    LLM requests made. When the endpoint itself fails (``OpenAIError``) the
    error is raised once the batches in flight have finished; the papers of
    the batches that succeeded keep their teasers, so the caller can ask
    again for the rest.

    Batches run ``llm.max_concurrency`` at a time: a batch request spends
    ~17 s waiting on the model, so issuing them one after another was the
    longest stage of a normal run. The client's rate limiter still spaces the
    request starts by ``llm.requests_per_minute``."""
    batch_size = max(1, llm.teaser_batch_size)
    # Set when a batch fails for good (its per-paper fallback raised): the run
    # is lost, so batches that have not started yet send nothing.
    failed = threading.Event()

    def run_batch(batch: list) -> int:
        if failed.is_set():
            return 0
        requests = 0
        teasers = {}
        if batch_size > 1:
            requests += 1  # the API is called even if the response turns out unusable
            try:
                teasers = _teasers_for_batch(client, llm, batch)
            except OpenAIError:
                # The endpoint failed, not the response: asking again paper by
                # paper would send ten more requests into the same failure.
                failed.set()
                raise
            except Exception as e:
                logger.warning(f"Batch teaser generation failed ({e}); falling back to per-paper requests")
        for i, paper in enumerate(batch):
            text = teasers.get(i)
            if not text:
                try:
                    text = generate_teaser(client, llm, paper.title, paper.abstract)
                except BaseException:
                    failed.set()
                    raise
                requests += 1
            paper.teaser = text
        return requests

    batches = [papers[start : start + batch_size] for start in range(0, len(papers), batch_size)]
    if not batches:
        return 0
    with ThreadPoolExecutor(max_workers=max(1, min(llm.max_concurrency, len(batches)))) as pool:
        return sum(pool.map(run_batch, batches))
