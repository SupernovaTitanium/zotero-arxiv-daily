"""Tests for email rendering: empty state, topic grouping, teaser clipping."""

from zotero_arxiv_daily.email import render_email
from tests.canned_responses import make_sample_paper


def test_render_empty_paper_list():
    html = render_email([], 150)
    assert "No Papers Today" in html
    assert "今日超級速覽" not in html


def test_render_contains_summary_section_and_papers():
    papers = [
        make_sample_paper(title="Paper A", teaser="Teaser A"),
        make_sample_paper(title="Paper B", teaser="Teaser B"),
    ]
    html = render_email(papers, 150)
    assert "今日超級速覽" in html
    assert "Paper A" in html
    assert "Teaser B" in html
    assert html.strip().endswith("</html>")


def test_render_groups_papers_under_topic_headers():
    papers = [
        make_sample_paper(title="Paper A", teaser="a", topic="Topic One"),
        make_sample_paper(title="Paper B", teaser="b", topic="Topic One"),
        make_sample_paper(title="Paper C", teaser="c", topic=None),
    ]
    html = render_email(papers, 150)
    assert html.count("📂 Topic One") == 1  # topic header printed once


def test_render_clips_long_teasers():
    paper = make_sample_paper(title="Paper A", teaser="x" * 300)
    html = render_email([paper], 150)
    assert ("x" * 150 + "...") in html
    assert ("x" * 300) not in html


def test_render_escapes_html_in_titles():
    paper = make_sample_paper(title="<script>alert(1)</script>", teaser="ok")
    html = render_email([paper], 150)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
