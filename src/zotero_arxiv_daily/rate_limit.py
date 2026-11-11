import threading
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
        # Requests are issued from several threads at once; the lock guards
        # the three fields below. Sleeping always happens outside it.
        self._lock = threading.Lock()
        self._next_start = float("-inf")  # earliest time the next request may start
        self._widened_at = float("-inf")  # when a 429 last widened the interval

    def _wait_for_slot(self) -> float:
        """Block until this caller may start a request; return the time its
        slot was granted. Slots are never granted closer together than the
        minimum interval. A waiter re-checks after sleeping, because another
        caller may have taken the slot or a 429 may have pushed every start
        back meanwhile."""
        while True:
            with self._lock:
                now = self._monotonic()
                if now >= self._next_start:
                    self._next_start = now + self._min_interval
                    return now
                wait = self._next_start - now
            self._sleep(wait)

    def _back_off(self, wait: float, started: float) -> None:
        """After a 429: hold every caller back for ``wait`` seconds and widen
        the interval. A 429 on a request that was already in flight when the
        interval last widened belongs to the same burst and does not widen it
        again."""
        with self._lock:
            now = self._monotonic()
            if started >= self._widened_at:
                if self._min_interval <= 0:
                    self._min_interval = min(self._max_interval_seconds, max(1.0, self._backoff_seconds))
                else:
                    self._min_interval = min(self._max_interval_seconds, self._min_interval * 2)
                self._widened_at = now
            self._next_start = max(self._next_start, now + wait)

    def _rate_limit_wait(self, exc: Exception, attempt: int) -> float:
        retry_after = _retry_after_seconds(exc, self._wall_time())
        if retry_after is not None:
            return min(self._max_interval_seconds, retry_after)
        return min(self._max_interval_seconds, self._backoff_seconds * (2 ** attempt))

    def create(self, *args, **kwargs):
        for attempt in range(self._max_retries + 1):
            started = self._wait_for_slot()
            try:
                return self._resource.create(*args, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                wait = self._rate_limit_wait(exc, attempt)
                # Before deciding whether this caller retries: the other
                # callers must hold back even when this one gives up.
                self._back_off(wait, started)
                if attempt >= self._max_retries:
                    raise
                logger.warning(
                    f"{self._name} hit rate limit; retry {attempt + 1}/{self._max_retries} "
                    f"in {wait:.1f}s; future minimum interval is {self._min_interval:.1f}s"
                )

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
