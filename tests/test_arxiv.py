"""Tests for the arXiv module: OAI-PMH harvest (primary), search API fallback, dedup."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests
from arxiv import HTTPError as ArxivHTTPError

from zotero_arxiv_daily import arxiv
from tests.canned_responses import make_sample_paper


NOW = datetime(2026, 9, 21, 5, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=3)

SAMPLE_OAI_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:2609.01234</identifier>
        <datestamp>2026-09-20</datestamp>
        <setSpec>cs:cs:LG</setSpec>
        <setSpec>cs:cs:CL</setSpec>
      </header>
      <metadata>
        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
          <id>2609.01234</id>
          <submitter>Hans Müller</submitter>
          <version version="v1"><date>Fri, 18 Sep 2026 17:59:01 GMT</date><size>293kb</size></version>
          <version version="v2"><date>Sat, 19 Sep 2026 09:00:00 GMT</date><size>300kb</size></version>
          <title>A  Müller
  paper</title>
          <authors>Hans M\\"uller (TU Wien), Anna Lee</authors>
          <categories>cs.LG cs.CL</categories>
          <doi>10.1234/ABC</doi>
          <abstract>  An abstract.
</abstract>
        </arXivRaw>
      </metadata>
    </record>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:2609.09999</identifier>
        <datestamp>2026-09-20</datestamp>
      </header>
    </record>
    <record>
      <header>
        <identifier>oai:arXiv.org:2609.05678</identifier>
        <datestamp>2026-09-20</datestamp>
        <setSpec>physics:astro-ph:CO</setSpec>
      </header>
      <metadata>
        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
          <id>2609.05678</id>
          <version version="v1"><date>Sat, 19 Sep 2026 01:00:00 GMT</date></version>
          <title>An astro paper</title>
          <authors>Ann A.</authors>
          <categories>astro-ph.CO</categories>
          <abstract>Stars.</abstract>
        </arXivRaw>
      </metadata>
    </record>
    <resumptionToken>token-1</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""


def _record(arxiv_id, submitted, categories, title=None, authors="A. Author", doi=None):
    return {
        "id": arxiv_id,
        "submitted": submitted,
        "categories": categories,
        "title": title or f"Paper {arxiv_id}",
        "abstract": f"Abstract of {arxiv_id}.",
        "authors": authors,
        "doi": doi,
    }


def _stub_pages(monkeypatch, pages):
    """Serve canned (records, token) pages from ``_fetch_oai_page`` in order;
    returns the request params and the sleeps the harvest made."""
    requested, sleeps = [], []
    remaining = list(pages)

    def fetch(params):
        requested.append(dict(params))
        return remaining.pop(0)

    monkeypatch.setattr(arxiv, "_fetch_oai_page", fetch)
    monkeypatch.setattr(arxiv, "sleep", sleeps.append)
    return requested, sleeps


# ---------------------------------------------------------------------------
# OAI-PMH parsing
# ---------------------------------------------------------------------------

def test_parse_oai_page_records_and_token():
    records, token = arxiv._parse_oai_page(SAMPLE_OAI_PAGE)
    assert token == "token-1"
    assert [r["id"] for r in records] == ["2609.01234", "2609.05678"]  # the deleted record has no metadata
    first = records[0]
    assert first["submitted"] == datetime(2026, 9, 18, 17, 59, 1, tzinfo=timezone.utc)  # v1, not the latest version
    assert first["categories"] == ["cs.LG", "cs.CL"]
    assert first["doi"] == "10.1234/ABC"
    assert first["authors"] == 'Hans M\\"uller (TU Wien), Anna Lee'
    assert records[1]["doi"] is None


def test_parse_oai_page_no_records_match():
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="noRecordsMatch"/>'
        "</OAI-PMH>"
    )
    records, token = arxiv._parse_oai_page(xml)
    assert records == []
    assert token is None


def test_parse_oai_page_rejects_dtd():
    xml = '<?xml version="1.0"?><!DOCTYPE OAI-PMH [<!ENTITY x "y">]><OAI-PMH/>'
    with pytest.raises(RuntimeError, match="DOCTYPE"):
        arxiv._parse_oai_page(xml)


@pytest.mark.parametrize(
    "xml, message",
    [
        (SAMPLE_OAI_PAGE.replace('version="v1"', 'version="v9"'), "no v1 submission date"),
        (SAMPLE_OAI_PAGE.replace("17:59:01 GMT", "17:59:01"), "without a time zone"),
        (SAMPLE_OAI_PAGE.replace("http://arxiv.org/OAI/arXivRaw/", "http://arxiv.org/OAI/arXivRaw2/"), "without arXivRaw metadata"),
        (SAMPLE_OAI_PAGE.replace("ListRecords", "ListIdentifiers"), "neither an error nor a ListRecords"),
        ("<html><body>Scheduled maintenance</body></html>", "not an OAI-PMH response"),
    ],
)
def test_parse_oai_page_unexpected_shape_fails_loudly(xml, message):
    # none of these may read as "nothing new today"
    with pytest.raises(ValueError, match=message):
        arxiv._parse_oai_page(xml)


def test_oai_sets_one_per_archive():
    assert arxiv._oai_sets(["cs.AI", "cs.LG", "stat.ML", "astro-ph.CO", "quant-ph"]) == [
        "cs",
        "stat",
        "physics:astro-ph",
        "physics:quant-ph",
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Wang Jia and Hang Xu", ["Wang Jia", "Hang Xu"]),
        ("Kanishka Misra, Najoung Kim", ["Kanishka Misra", "Najoung Kim"]),
        ("Dae-Young Yun, Sang-Hyo Kim, and Jong-Seon No", ["Dae-Young Yun", "Sang-Hyo Kim", "Jong-Seon No"]),
        (
            "Alexandre Audibert and Aur\\'elien Gauffre and Massih-Reza Amini",
            ["Alexandre Audibert", "Aurélien Gauffre", "Massih-Reza Amini"],
        ),
        (
            "A. Einstein (1), B. Podolsky (1 and 2) ((1) IAS, Princeton, (2) Caltech)",
            ["A. Einstein", "B. Podolsky"],
        ),
        ("{\\L}ukasz Kaiser, S{\\o}ren Hauberg, Fran\\c{c}ois Chollet", ["Łukasz Kaiser", "Søren Hauberg", "François Chollet"]),
        ('Hans M\\"uller\n  and Martin Luther King, Jr.', ["Hans Müller", "Martin Luther King Jr."]),
        ("Andrew Anderson", ["Andrew Anderson"]),  # "and" inside a name is not a separator
        ("(ATLAS Collaboration)", ["ATLAS Collaboration"]),  # the only names there are: not an affiliation
        ("P. Erd\\H{o}s, J. Ho\\v{r}ej\\v{s}\\'{\\i}, \\r{A}. \\k{A}", ["P. Erdős", "J. Hořejší", "Å. Ą"]),
        ("M. A. Yatk{\\i}n", ["M. A. Yatkın"]),
        ("", []),
    ],
)
def test_split_authors(text, expected):
    assert arxiv._split_authors(text) == expected


# ---------------------------------------------------------------------------
# OAI-PMH harvest
# ---------------------------------------------------------------------------

def test_harvest_follows_resumption_tokens_with_a_delay_between_requests(monkeypatch):
    requested, sleeps = _stub_pages(
        monkeypatch,
        [
            ([_record("2609.00001", START, ["cs.AI"])], "token-1"),
            ([_record("2609.00002", START, ["cs.AI"])], None),
            ([_record("2609.00003", START, ["stat.ML"])], None),
        ],
    )
    papers = arxiv._papers_from_oai(["cs.AI", "stat.ML"], False, START, NOW)
    assert [p.source_id for p in papers] == ["2609.00001", "2609.00002", "2609.00003"]
    assert requested == [
        {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": "cs", "from": "2026-09-18", "until": "2026-09-21"},
        {"verb": "ListRecords", "resumptionToken": "token-1"},
        {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": "stat", "from": "2026-09-18", "until": "2026-09-21"},
    ]
    # one polite delay before every request except the first
    assert sleeps == [arxiv.OAI_PAGE_DELAY_SECONDS] * 2


def test_harvest_keeps_recent_announcements_by_identifier_order(monkeypatch):
    cutoff = START - timedelta(days=arxiv.ANNOUNCEMENT_MARGIN_DAYS)
    _stub_pages(
        monkeypatch,
        [
            (
                [
                    # replacements of older papers: their identifiers predate the recent range
                    _record("1107.3279", datetime(2011, 7, 17, tzinfo=timezone.utc), ["cs.AI"]),
                    _record("cs/0601001", datetime(2006, 1, 2, tzinfo=timezone.utc), ["cs.AI"]),
                    _record("2609.00100", cutoff - timedelta(seconds=1), ["cs.AI"]),
                    # the lowest identifier submitted since the cutoff: the recent range starts here,
                    # even though this record itself is outside the configured categories
                    _record("2609.00200", cutoff, ["math.CO"]),
                    # submitted before the window start but announced inside it: the
                    # case the search API's submission-time window used to lose
                    _record("2609.00300", START - timedelta(hours=10), ["cs.AI"]),
                    # held by moderation since June and announced now: an old
                    # submission date with a recent identifier
                    _record("2609.00400", datetime(2026, 6, 26, tzinfo=timezone.utc), ["cs.AI"]),
                    _record("2609.00500", NOW - timedelta(hours=12), ["cs.AI"]),
                ],
                None,
            )
        ],
    )
    papers = arxiv._papers_from_oai(["cs.AI"], False, START, NOW)
    assert [p.source_id for p in papers] == ["2609.00300", "2609.00400", "2609.00500"]


def test_harvest_of_replacements_only_has_nothing_new(monkeypatch):
    _stub_pages(monkeypatch, [([_record("2601.00001", datetime(2026, 1, 2, tzinfo=timezone.utc), ["cs.AI"])], None)])
    assert arxiv._papers_from_oai(["cs.AI"], False, START, NOW) == []


def test_harvest_returns_a_paper_found_in_two_sets_once(monkeypatch):
    record = _record("2609.00001", START, ["cs.AI", "stat.ML"])
    _stub_pages(monkeypatch, [([record], None), ([record], None)])
    assert len(arxiv._papers_from_oai(["cs.AI", "stat.ML"], False, START, NOW)) == 1


def test_harvest_category_scope(monkeypatch):
    records = [
        _record("2609.00001", START, ["cs.AI", "stat.ML"]),
        _record("2609.00002", START, ["stat.ML", "cs.AI"]),  # cross-listed into cs.AI
        _record("2609.00003", START, ["cs.DM"]),
    ]
    _stub_pages(monkeypatch, [(records, None)])
    assert [p.source_id for p in arxiv._papers_from_oai(["cs.AI"], False, START, NOW)] == ["2609.00001"]
    _stub_pages(monkeypatch, [(records, None)])
    assert [p.source_id for p in arxiv._papers_from_oai(["cs.AI"], True, START, NOW)] == [
        "2609.00001",
        "2609.00002",
    ]


def test_harvest_builds_papers_from_records(monkeypatch):
    records, _ = arxiv._parse_oai_page(SAMPLE_OAI_PAGE)
    _stub_pages(monkeypatch, [(records, None)])
    (paper,) = arxiv._papers_from_oai(["cs.LG"], False, START, NOW)
    assert paper.title == "A Müller paper"  # whitespace collapsed
    assert paper.abstract == "An abstract."
    assert paper.authors == ["Hans Müller", "Anna Lee"]
    assert paper.url == "https://arxiv.org/abs/2609.01234"
    assert paper.pdf_url == "https://arxiv.org/pdf/2609.01234"
    assert set(paper.dedup_keys()) == {"doi:10.1234/abc", "title:amllerpaper", "sid:arxiv:2609.01234"}


def test_fetch_oai_page_retries_then_raises(monkeypatch):
    calls, sleeps = [], []

    def failing_get(url, params, timeout):
        calls.append(params)
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(arxiv.requests, "get", failing_get)
    monkeypatch.setattr(arxiv, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        arxiv._fetch_oai_page({"verb": "ListRecords"})
    assert len(calls) == 3
    assert sleeps == [30, 60]


def test_fetch_oai_page_decodes_utf8_and_retries_a_garbled_page(monkeypatch):
    bodies = [b"<OAI-PMH><truncated", SAMPLE_OAI_PAGE.encode("utf-8")]
    monkeypatch.setattr(
        arxiv.requests,
        "get",
        lambda url, params, timeout: SimpleNamespace(content=bodies.pop(0), raise_for_status=lambda: None),
    )
    monkeypatch.setattr(arxiv, "sleep", lambda seconds: None)
    records, token = arxiv._fetch_oai_page({"verb": "ListRecords"})
    assert records[0]["title"].split() == ["A", "Müller", "paper"]  # no mojibake
    assert token == "token-1"


# ---------------------------------------------------------------------------
# Search API fallback
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error
        self._session = SimpleNamespace(request=lambda *a, **kw: None)

    def results(self, search):
        if self._error:
            raise self._error
        return iter(self._results)


def _search_result(arxiv_id, primary):
    return SimpleNamespace(
        title=f"Paper {arxiv_id}",
        authors=[SimpleNamespace(name="Author A")],
        summary="Abstract.",
        entry_id=f"http://arxiv.org/abs/{arxiv_id}v1",
        pdf_url=f"http://arxiv.org/pdf/{arxiv_id}v1",
        doi=None,
        primary_category=primary,
    )


@pytest.mark.parametrize("status", [406, 429])
def test_search_api_error_raises_without_a_backoff_loop(monkeypatch, status):
    monkeypatch.setattr(arxiv, "Client", lambda **kw: _FakeClient(error=ArxivHTTPError("u", 0, status)))
    sleep_calls = []
    monkeypatch.setattr(arxiv, "sleep", sleep_calls.append)
    with pytest.raises(RuntimeError, match="HTTPError"):
        arxiv._papers_from_search_api(["cs.AI"], False, START, NOW)
    assert sleep_calls == []


def test_search_api_empty_feed_is_a_failure(monkeypatch):
    monkeypatch.setattr(arxiv, "Client", lambda **kw: _FakeClient())
    with pytest.raises(RuntimeError, match="empty feed"):
        arxiv._papers_from_search_api(["cs.AI"], False, START, NOW)


def test_search_api_converts_results_and_filters_by_primary_category(monkeypatch):
    results = [_search_result("2609.00001", "cs.AI"), _search_result("2609.00002", "stat.ML")]
    monkeypatch.setattr(arxiv, "Client", lambda **kw: _FakeClient(results))
    papers = arxiv._papers_from_search_api(["cs.AI"], False, START, NOW)
    assert [p.source_id for p in papers] == ["2609.00001"]  # version stripped
    monkeypatch.setattr(arxiv, "Client", lambda **kw: _FakeClient(results))
    assert len(arxiv._papers_from_search_api(["cs.AI"], True, START, NOW)) == 2


def test_short_id_strips_version():
    assert arxiv._short_id("https://arxiv.org/abs/2609.01234v2") == "2609.01234"
    assert arxiv._short_id("https://arxiv.org/abs/2609.01234") == "2609.01234"


# ---------------------------------------------------------------------------
# retrieve_papers
# ---------------------------------------------------------------------------

def _fail(message):
    def raiser(*args, **kwargs):
        raise RuntimeError(message)

    return raiser


def test_retrieve_papers_dedups_against_seen_keys(config, monkeypatch):
    paper_a = make_sample_paper(title="Paper A", source_id="2609.00001")
    paper_b = make_sample_paper(title="Paper B", source_id="2609.00002")
    monkeypatch.setattr(arxiv, "_papers_from_oai", lambda *a, **kw: [paper_a, paper_b])
    monkeypatch.setattr(arxiv, "_papers_from_search_api", _fail("the fallback must not run"))
    seen = {"title:papera"}  # paper A is already known
    papers = arxiv.retrieve_papers(config, seen)
    assert [p.source_id for p in papers] == ["2609.00002"]
    # returned papers' keys were added to the shared set
    assert "sid:arxiv:2609.00002" in seen


def test_retrieve_papers_window_is_the_lookback(config, monkeypatch):
    config.executor.lookback_days = 5
    windows = []
    monkeypatch.setattr(arxiv, "_papers_from_oai", lambda cats, cross, start, now: windows.append(now - start) or [])
    arxiv.retrieve_papers(config, set())
    assert windows == [timedelta(days=5)]


def test_retrieve_papers_empty_harvest_is_not_a_failure(config, monkeypatch):
    monkeypatch.setattr(arxiv, "_papers_from_oai", lambda *a, **kw: [])
    monkeypatch.setattr(arxiv, "_papers_from_search_api", _fail("the fallback must not run"))
    assert arxiv.retrieve_papers(config, set()) == []


def test_retrieve_papers_falls_back_to_search_api(config, monkeypatch):
    paper = make_sample_paper(title="Paper A", source_id="2609.00001")
    monkeypatch.setattr(arxiv, "_papers_from_oai", _fail("harvest down"))
    monkeypatch.setattr(arxiv, "_papers_from_search_api", lambda *a, **kw: [paper])
    assert arxiv.retrieve_papers(config, set()) == [paper]


def test_retrieve_papers_fails_loudly_when_both_routes_fail(config, monkeypatch):
    monkeypatch.setattr(arxiv, "_papers_from_oai", _fail("harvest down"))
    monkeypatch.setattr(arxiv, "_papers_from_search_api", _fail("throttled"))
    with pytest.raises(RuntimeError, match="Both arXiv routes failed.*harvest down.*throttled"):
        arxiv.retrieve_papers(config, set())


def test_retrieve_papers_debug_keeps_ten(config, monkeypatch):
    config.executor.debug = True
    papers = [make_sample_paper(title=f"Paper {i}", source_id=f"2609.{i:05d}") for i in range(25)]
    monkeypatch.setattr(arxiv, "_papers_from_oai", lambda *a, **kw: papers)
    assert len(arxiv.retrieve_papers(config, set())) == 10


# ---------------------------------------------------------------------------
# Dedup keys
# ---------------------------------------------------------------------------

def test_paper_dedup_keys():
    paper = make_sample_paper(doi="https://doi.org/10.1234/ABC", source_id="2026.00001")
    keys = paper.dedup_keys()
    assert "doi:10.1234/abc" in keys
    assert "sid:arxiv:2026.00001" in keys


def test_non_ascii_title_produces_no_title_key():
    assert not [k for k in make_sample_paper(title="深度學習", doi=None).dedup_keys() if k.startswith("title:")]
    assert "sid:arxiv:2026.00001" in make_sample_paper(title="深度學習", doi=None).dedup_keys()
    assert "title:deeplearning" in make_sample_paper(title="Deep Learning").dedup_keys()
