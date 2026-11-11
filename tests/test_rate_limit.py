"""Tests for the rate-limited client wrapper: request spacing, 429 backoff,
and use from several threads at once."""

import threading
from types import SimpleNamespace

import pytest

from zotero_arxiv_daily.rate_limit import rate_limit_openai_client


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class RateLimitError(Exception):
    """Recognised by status code, like openai.RateLimitError."""

    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
        self.response = SimpleNamespace(status_code=429, headers=headers)


def _limited(create, clock, requests_per_minute=60, **kwargs):
    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return rate_limit_openai_client(
        raw,
        requests_per_minute,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        wall_time=lambda: 0.0,
        **kwargs,
    )


def _scripted(clock, outcomes, starts):
    """A create() that records its start time and plays back outcomes."""
    remaining = list(outcomes)

    def create(**kwargs):
        starts.append(clock.now)
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return create


def test_request_starts_are_spaced_by_the_minimum_interval():
    clock, starts = FakeClock(), []
    client = _limited(_scripted(clock, ["a", "b", "c"], starts), clock)
    assert [client.chat.completions.create() for _ in range(3)] == ["a", "b", "c"]
    assert starts == [0.0, 1.0, 2.0]
    assert clock.sleeps == [1.0, 1.0]


def test_a_slow_request_uses_up_its_own_interval():
    clock, starts = FakeClock(), []

    def slow_create(**kwargs):
        starts.append(clock.now)
        clock.now += 5.0

    client = _limited(slow_create, clock)
    client.chat.completions.create()
    client.chat.completions.create()
    assert starts == [0.0, 5.0]
    assert clock.sleeps == []


def test_429_waits_for_retry_after_and_doubles_the_interval():
    clock, starts = FakeClock(), []
    client = _limited(_scripted(clock, [RateLimitError(retry_after=7), "ok", "next"], starts), clock)
    assert client.chat.completions.create() == "ok"
    assert client.chat.completions.create() == "next"
    # retried after the 7 s pause; the following request is 2 s (doubled) later
    assert starts == [0.0, 7.0, 9.0]


def test_429_without_retry_after_backs_off_exponentially():
    clock, starts = FakeClock(), []
    client = _limited(
        _scripted(clock, [RateLimitError(), RateLimitError(), "ok"], starts), clock, backoff_seconds=30
    )
    assert client.chat.completions.create() == "ok"
    assert starts == [0.0, 30.0, 90.0]


def test_429_gives_up_after_max_retries():
    clock, starts = FakeClock(), []
    client = _limited(_scripted(clock, [RateLimitError(1)] * 3, starts), clock, max_retries=2)
    with pytest.raises(RateLimitError):
        client.chat.completions.create()
    assert len(starts) == 3


def test_429_that_exhausts_the_retries_still_holds_the_other_callers_back():
    clock, starts = FakeClock(), []
    client = _limited(_scripted(clock, [RateLimitError(retry_after=120), "other caller"], starts), clock, max_retries=0)
    with pytest.raises(RateLimitError):
        client.chat.completions.create()
    assert client.chat.completions.create() == "other caller"
    assert starts == [0.0, 120.0]


def test_other_errors_are_not_retried():
    clock, starts = FakeClock(), []
    client = _limited(_scripted(clock, [ValueError("bad request")], starts), clock)
    with pytest.raises(ValueError):
        client.chat.completions.create()
    assert len(starts) == 1
    assert clock.sleeps == []


def test_unthrottled_client_still_backs_off_on_429():
    clock, starts = FakeClock(), []
    client = _limited(_scripted(clock, ["a", "b", RateLimitError(5), "c", "d"], starts), clock, requests_per_minute=0)
    assert [client.chat.completions.create() for _ in range(4)] == ["a", "b", "c", "d"]
    # no spacing until the 429; a 30 s interval (the backoff default) from then on
    assert starts == [0.0, 0.0, 0.0, 5.0, 35.0]


def test_429s_of_one_burst_widen_the_interval_once():
    clock = FakeClock()
    limiter = _limited(lambda **kwargs: None, clock).chat.completions
    first = limiter._wait_for_slot()  # starts at 0.0
    second = limiter._wait_for_slot()  # starts at 1.0, while the first is still in flight
    clock.now = 1.5
    limiter._back_off(10.0, first)
    limiter._back_off(10.0, second)  # in flight when the interval widened: same burst
    assert limiter._min_interval == 2.0
    # every caller is held back until the pause is over
    assert limiter._wait_for_slot() == 11.5
    third = limiter._wait_for_slot()
    assert third == 13.5
    limiter._back_off(10.0, third)  # started after the widening: a new burst
    assert limiter._min_interval == 4.0


def test_concurrent_callers_get_slots_no_closer_than_the_interval():
    interval = 0.02
    limiter = rate_limit_openai_client(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))),
        60 / interval,
    ).chat.completions
    starts = []
    threads = [threading.Thread(target=lambda: starts.append(limiter._wait_for_slot())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    starts.sort()
    assert len(starts) == 8
    assert all(later - earlier >= interval for earlier, later in zip(starts, starts[1:]))
