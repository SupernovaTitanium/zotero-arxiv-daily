"""Tests for the arXiv module: OAI parsing, dedup keys, retrieval fallback."""

import pytest

from zotero_arxiv_daily import arxiv
from tests.canned_responses import make_sample_paper


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
        <oai_dc:dc xmlns:oai_dc="http://purl.org/dc/elements/1.1/" xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>A  MÃ¼ller   paper</dc:title>
          <dc:description>An abstract.</dc:description>
          <dc:creator>Hans Müller</dc:creator>
          <dc:creator>Anna Lee</dc:creator>
        </oai_dc:dc>
      </metadata>
    </record>
    <record>
      <header>
        <identifier>oai:arXiv.org:2609.05678</identifier>
        <datestamp>2026-09-20</datestamp>
        <setSpec>physics:astro-ph:CO</setSpec>
      </header>
      <metadata>
        <oai_dc:dc xmlns:oai_dc="http://purl.org/dc/elements/1.1/" xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>An astro paper</dc:title>
          <dc:description>Stars.</dc:description>
          <dc:creator>Ann A.</dc:creator>
        </oai_dc:dc>
      </metadata>
    </record>
    <resumptionToken>token-1</resumptionToken>
  </ListRecords>
</OAI-PMH>
"""


def test_parse_oai_page_records_and_token():
    records, token = arxiv._parse_oai_page(SAMPLE_OAI_PAGE)
    assert token == "token-1"
    assert len(records) == 2
    first = records[0]
    assert first["id"] == "2609.01234"
    assert first["title"] == "A  MÃ¼ller   paper"  # raw text kept; collapsing happens in _build_arxiv_result
    assert arxiv._build_arxiv_result(first).title == "A MÃ¼ller paper"
    assert first["authors"] == ["Hans Müller", "Anna Lee"]
    assert first["set_specs"] == ["cs:cs:LG", "cs:cs:CL"]


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


def test_oai_category_codes():
    assert arxiv._oai_category_codes(["cs:cs:LG", "physics:astro-ph:CO", "physics:quant-ph"]) == [
        "cs.LG",
        "astro-ph.CO",
        "quant-ph",
    ]


def test_short_id_strips_version():
    assert arxiv._short_id("https://arxiv.org/abs/2609.01234v2") == "2609.01234"
    assert arxiv._short_id("https://arxiv.org/abs/2609.01234") == "2609.01234"


def test_raw_keys_include_doi_title_and_source_id():
    record = {
        "id": "2609.01234",
        "datestamp": "2026-09-20",
        "set_specs": ["cs:cs:LG"],
        "title": "A title",
        "abstract": "Abstract.",
        "authors": ["A"],
    }
    raw = arxiv._build_arxiv_result(record)
    keys = arxiv._raw_keys(raw)
    assert "sid:arxiv:2609.01234" in keys
    assert "title:atitle" in keys


def test_build_arxiv_result_primary_category_from_first_setspec():
    record = {
        "id": "2609.01234",
        "datestamp": "2026-09-20",
        "set_specs": ["cs:cs:LG", "cs:cs:CL"],
        "title": "A title",
        "abstract": "Abstract.",
        "authors": ["A"],
    }
    raw = arxiv._build_arxiv_result(record)
    assert raw.primary_category == "cs.LG"
    assert raw.entry_id == "https://arxiv.org/abs/2609.01234"


def test_retrieve_papers_dedups_against_seen_keys(config, monkeypatch):
    raw_a = arxiv._build_arxiv_result(
        {"id": "2609.00001", "datestamp": "2026-09-20", "set_specs": ["cs:cs:LG"], "title": "Paper A", "abstract": "a", "authors": ["A"]}
    )
    raw_b = arxiv._build_arxiv_result(
        {"id": "2609.00002", "datestamp": "2026-09-20", "set_specs": ["cs:cs:LG"], "title": "Paper B", "abstract": "b", "authors": ["B"]}
    )
    monkeypatch.setattr(arxiv, "_papers_from_search_api", lambda *a, **kw: [raw_a, raw_b])
    seen = {"title:papera"}  # paper A is already known
    papers = arxiv.retrieve_papers(config, seen)
    assert [p.source_id for p in papers] == ["2609.00002"]
    # converted papers' keys were added to the shared set
    assert "sid:arxiv:2609.00002" in seen


def test_retrieve_papers_falls_back_to_oai_and_fails_loudly(config, monkeypatch):
    def raise_api(*a, **kw):
        raise RuntimeError("throttled")

    monkeypatch.setattr(arxiv, "_papers_from_search_api", raise_api)
    monkeypatch.setattr(arxiv, "_papers_from_oai", lambda *a, **kw: [])
    with pytest.raises(RuntimeError, match="Both arXiv routes failed"):
        arxiv.retrieve_papers(config, set())


def test_paper_dedup_keys():
    paper = make_sample_paper(doi="https://doi.org/10.1234/ABC", source_id="2026.00001")
    keys = paper.dedup_keys()
    assert "doi:10.1234/abc" in keys
    assert "sid:arxiv:2026.00001" in keys
