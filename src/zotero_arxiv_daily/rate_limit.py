import time
from email.utils import parsedate_to_datetime
from typing import Any

from loguru import logger


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None) if response is not None else None
    if status_code == 429 or response_status == 429:
        return True
    return exc.__class__.__name__ == "RateLimitError"


def _get_header(headers: Any, name: str) -> Any:
    if headers is None or not hasattr(headers, "get"):
        return None
    return headers.get(name) or headers.get(name.lower())


def _retry_after_seconds(exc: Exception, now: float) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else getattr(exc, "headers", None)
    retry_after = _get_header(headers, "Retry-After")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(retry_after))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return max(0.0, retry_at.timestamp() - now)


class _RateLimitedCreateResource:
    def __init__(
        self,
        resource,
        *,
        name: str,
        min_interval: float,
        max_retries: int,
        backoff_seconds: float,
        max_interval_seconds: float,
        sleep=time.sleep,
        monotonic=time.monotonic,
        wall_time=time.time,
    ):
        self._resource = resource
        self._name = name
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._max_interval_seconds = max_interval_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._last_started_at = None

    def _wait_for_slot(self) -> float:
        now = self._monotonic()
        if self._last_started_at is not None and self._min_interval > 0:
            wait = self._min_interval - (now - self._last_started_at)
            if wait > 0:
                self._sleep(wait)
                now = self._monotonic()
        self._last_started_at = now
        return now

    def _slow_down_after_rate_limit(self) -> None:
        if self._min_interval <= 0:
            self._min_interval = min(self._max_interval_seconds, max(1.0, self._backoff_seconds))
            return
        self._min_interval = min(self._max_interval_seconds, max(self._min_interval * 2, self._min_interval))

    def _rate_limit_wait(self, exc: Exception, attempt: int) -> float:
        retry_after = _retry_after_seconds(exc, self._wall_time())
        if retry_after is not None:
            return min(self._max_interval_seconds, retry_after)
        return min(self._max_interval_seconds, self._backoff_seconds * (2 ** attempt))

    def create(self, *args, **kwargs):
        for attempt in range(self._max_retries + 1):
            self._wait_for_slot()
            try:
                return self._resource.create(*args, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= self._max_retries:
                    raise
                wait = self._rate_limit_wait(exc, attempt)
                self._slow_down_after_rate_limit()
                logger.warning(
                    f"{self._name} hit rate limit; retry {attempt + 1}/{self._max_retries} "
                    f"in {wait:.1f}s; future minimum interval is {self._min_interval:.1f}s"
                )
                self._sleep(wait)
                self._last_started_at = None

        raise RuntimeError("unreachable")

    def __getattr__(self, name):
        return getattr(self._resource, name)


class _RateLimitedChat:
    def __init__(self, chat, **kwargs):
        self._chat = chat
        self.completions = _RateLimitedCreateResource(
            chat.completions,
            name="chat.completions.create",
            **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._chat, name)


class _RateLimitedOpenAI:
    def __init__(
        self,
        client,
        *,
        requests_per_minute: float | int | None = None,
        max_retries: int = 5,
        backoff_seconds: float = 30.0,
        max_interval_seconds: float = 300.0,
        sleep=time.sleep,
        monotonic=time.monotonic,
        wall_time=time.time,
    ):
        self._client = client
        min_interval = 0.0
        if requests_per_minute:
            requests_per_minute = float(requests_per_minute)
            if requests_per_minute > 0:
                min_interval = 60.0 / requests_per_minute
        kwargs = {
            "min_interval": min_interval,
            "max_retries": max(0, int(max_retries)),
            "backoff_seconds": max(0.0, float(backoff_seconds)),
            "max_interval_seconds": max(0.0, float(max_interval_seconds)),
            "sleep": sleep,
            "monotonic": monotonic,
            "wall_time": wall_time,
        }
        if hasattr(client, "chat"):
            self.chat = _RateLimitedChat(client.chat, **kwargs)
        if hasattr(client, "embeddings"):
            self.embeddings = _RateLimitedCreateResource(
                client.embeddings,
                name="embeddings.create",
                **kwargs,
            )

    def __getattr__(self, name):
        return getattr(self._client, name)


def rate_limit_openai_client(
    client,
    requests_per_minute: float | int | None = None,
    *,
    max_retries: int = 5,
    backoff_seconds: float = 30.0,
    max_interval_seconds: float = 300.0,
    sleep=time.sleep,
    monotonic=time.monotonic,
    wall_time=time.time,
):
    if not requests_per_minute and max_retries <= 0:
        return client
    return _RateLimitedOpenAI(
        client,
        requests_per_minute=requests_per_minute,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        max_interval_seconds=max_interval_seconds,
        sleep=sleep,
        monotonic=monotonic,
        wall_time=wall_time,
    )
