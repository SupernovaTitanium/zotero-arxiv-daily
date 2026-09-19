"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

from omegaconf import open_dict

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def _fake_result(title: str, entry_id: str, primary_category: str, doi: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        title=title,
        authors=[SimpleNamespace(name="Test Author")],
        summary="Test abstract",
        pdf_url=f"https://arxiv.org/pdf/{entry_id}",
        entry_id=f"https://arxiv.org/abs/{entry_id}",
        source_url=lambda: f"https://arxiv.org/e-print/{entry_id}",
        primary_category=primary_category,
        doi=doi,
    )


def _install_fake_client(monkeypatch, results, captured=None):
    class FakeSession:
        # matches the attribute the production code patches for request timeouts
        def request(self, *args, **kwargs):
            raise AssertionError("unexpected HTTP request in test")

    class FakeClient:
        def __init__(self, **kw):
            self._session = FakeSession()
        def results(self, search):
            if captured is not None:
                captured["query"] = search.query
            return iter(results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)


def test_arxiv_retriever_uses_date_range_query_and_filters_cross_lists(config, monkeypatch):
    captured: dict = {}
    results = [
        _fake_result("New Paper", "2609.00001v1", "cs.AI", doi="10.1234/abc"),
        # primary category not subscribed -> excluded unless include_cross_list
        _fake_result("Cross Paper", "2609.00002v1", "physics.bio-ph"),
    ]
    _install_fake_client(monkeypatch, results, captured)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert "submittedDate:[" in captured["query"]
    assert "cat:cs.AI" in captured["query"] and "cat:cs.CV" in captured["query"]
    assert [p.title for p in papers] == ["New Paper"]
    assert papers[0].doi == "10.1234/abc"
    assert papers[0].source_id == "2609.00001"
    # two-stage pipeline: retrieval is metadata-only
    assert papers[0].full_text is None


def test_arxiv_retriever_includes_cross_list_when_configured(config, monkeypatch):
    results = [
        _fake_result("New Paper", "2609.00001v1", "cs.AI"),
        _fake_result("Cross Paper", "2609.00002v1", "physics.bio-ph"),
    ]
    _install_fake_client(monkeypatch, results)
    with open_dict(config.source.arxiv):
        config.source.arxiv.include_cross_list = True
    papers = ArxivRetriever(config).retrieve_papers()
    assert {p.title for p in papers} == {"New Paper", "Cross Paper"}


def test_arxiv_retriever_filters_seen_keys_before_conversion(config, monkeypatch):
    results = [
        _fake_result("Already Seen Paper", "2609.00003v1", "cs.AI"),
        _fake_result("Fresh Paper", "2609.00004v1", "cs.AI"),
    ]
    _install_fake_client(monkeypatch, results)
    converted = []
    original_convert = ArxivRetriever.convert_to_paper

    def _tracking_convert(self, raw_paper):
        converted.append(raw_paper.title)
        return original_convert(self, raw_paper)

    monkeypatch.setattr(ArxivRetriever, "convert_to_paper", _tracking_convert)
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers(seen_keys={"sid:arxiv:2609.00003"})

    assert converted == ["Fresh Paper"]
    assert [p.title for p in papers] == ["Fresh Paper"]


def test_short_id_strips_version_suffix():
    assert ArxivRetriever._short_id("https://arxiv.org/abs/2609.01234v2") == "2609.01234"
    assert ArxivRetriever._short_id("https://arxiv.org/abs/2609.01234") == "2609.01234"


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        # generous timeout: on Windows the spawn subprocess needs a few seconds to boot
        _sleep_and_return, ("done", 0.01), timeout=10, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=10, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]


def test_fetch_full_text_prefers_tar_then_html_then_pdf(config, monkeypatch):
    retriever = ArxivRetriever.__new__(ArxivRetriever)
    retriever.config = config
    paper = _fake_result("New Paper", "2609.00001v1", "cs.AI")
    paper = retriever.convert_to_paper(paper)

    calls = []
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda p: calls.append("tar") or "tar text")
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda p: calls.append("html") or "html text")
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda p: calls.append("pdf") or "pdf text")

    assert retriever.fetch_full_text(paper) == "tar text"
    assert calls == ["tar"]

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda p: None)
    assert retriever.fetch_full_text(paper) == "html text"
    assert calls == ["tar", "html"]

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda p: None)
    assert retriever.fetch_full_text(paper) == "pdf text"
    assert calls == ["tar", "html", "pdf"]


def test_oai_category_codes():
    assert arxiv_retriever._oai_category_codes(
        ["cs:cs:LG", "physics:astro-ph:CO", "physics:quant-ph", "stat:stat:ML"]
    ) == ["cs.LG", "astro-ph.CO", "quant-ph", "stat.ML"]


def _oai_record_xml(arxiv_id, datestamp, set_specs, title, abstract, authors):
    specs = "".join(f"<setSpec>{s}</setSpec>" for s in set_specs)
    creators = "".join(f"<dc:creator>{a}</dc:creator>" for a in authors)
    return (
        f"<record><header><identifier>oai:arXiv.org:{arxiv_id}</identifier>"
        f"<datestamp>{datestamp}</datestamp>{specs}</header>"
        f"<metadata><oai_dc:dc xmlns:oai_dc=\"http://www.openarchives.org/OAI/2.0/oai_dc/\" "
        f"xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>{title}</dc:title>"
        f"{creators}<dc:description>{abstract}</dc:description></oai_dc:dc></metadata></record>"
    )


def _oai_page_xml(records_xml, token=None):
    token_el = f"<resumptionToken>{token}</resumptionToken>" if token else "<resumptionToken/>"
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<OAI-PMH xmlns=\"http://www.openarchives.org/OAI/2.0/\">"
        "<responseDate>2026-09-14T00:00:00Z</responseDate>"
        "<request verb=\"ListRecords\">https://oaipmh.arxiv.org/oai</request>"
        f"<ListRecords>{records_xml}{token_el}</ListRecords>"
        "</OAI-PMH>"
    )


def _install_fake_oai(monkeypatch, pages):
    calls = []

    def _fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        xml, token = pages[len(calls) - 1]
        body = _oai_page_xml(xml, token)
        # Mirror the real client: arXiv serves text/xml with no charset, so
        # .text misdecodes the UTF-8 body as ISO-8859-1 while .content keeps
        # the raw bytes.
        return SimpleNamespace(
            status_code=200,
            content=body.encode("utf-8"),
            text=body.encode("utf-8").decode("iso-8859-1"),
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(arxiv_retriever.requests, "get", _fake_get)
    return calls


def test_retriever_falls_back_to_oai_when_search_api_fails(config, monkeypatch):
    _install_fake_client(monkeypatch, [])  # search API answers, but with 0 papers
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda seconds: None)
    records = (
        _oai_record_xml("2609.00001", "2026-09-13", ["cs:cs:AI", "cs:cs:LG"],
                        "OAI  Paper", "OAI abstract", ["Alice", "Müller"])
        # primary (first setSpec) not subscribed and include_cross_list=False -> excluded
        + _oai_record_xml("2609.00002", "2026-09-13", ["cs:cs:LG", "cs:cs:AI"],
                          "Cross-listed paper", "x", ["Carol"])
        # no subscribed category at all -> excluded
        + _oai_record_xml("2609.00003", "2026-09-13", ["physics:quant-ph"],
                          "Unrelated", "y", ["Dave"])
    )
    _install_fake_oai(monkeypatch, [(records, None)])

    papers = ArxivRetriever(config).retrieve_papers()

    assert [p.title for p in papers] == ["OAI Paper"]
    assert papers[0].source_id == "2609.00001"
    assert papers[0].abstract == "OAI abstract"
    assert papers[0].url == "https://arxiv.org/abs/2609.00001"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2609.00001"
    assert papers[0].authors == ["Alice", "Müller"]


def test_retriever_raises_when_both_routes_fail(config, monkeypatch):
    _install_fake_client(monkeypatch, [])
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda seconds: None)
    empty = '<error code="noRecordsMatch">empty</error>'
    _install_fake_oai(monkeypatch, [(empty, None)])

    import pytest

    with pytest.raises(RuntimeError, match="Both arXiv routes failed"):
        ArxivRetriever(config).retrieve_papers()
