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


def _titles_and_marks(html: str) -> list[str]:
    """The rendered list reduced to its order: topic headers, rules, paper titles."""
    import re

    return re.findall(r"📂 [^<]+|🔗 [^<]+|<div style=\"border-bottom: 1px solid #f0d9c0;\"></div>", html)


RULE = '<div style="border-bottom: 1px solid #f0d9c0;"></div>'


def test_render_gathers_a_topic_whose_papers_are_apart_in_the_ranking():
    papers = [
        make_sample_paper(title="A1", teaser="t", topic="Topic A"),
        make_sample_paper(title="B1", teaser="t", topic="Topic B"),
        make_sample_paper(title="A2", teaser="t", topic="Topic A"),
        make_sample_paper(title="B2", teaser="t", topic="Topic B"),
    ]
    assert _titles_and_marks(render_email(papers, 150)) == [
        "📂 Topic A", "🔗 A1", "🔗 A2", "📂 Topic B", "🔗 B1", "🔗 B2",
    ]


def test_render_keeps_a_paper_without_topic_at_its_rank_and_outside_the_topic_above():
    papers = [
        make_sample_paper(title="Alone 1", teaser="t", topic=None),
        make_sample_paper(title="A1", teaser="t", topic="Topic A"),
        make_sample_paper(title="Alone 2", teaser="t", topic=None),
        make_sample_paper(title="Alone 3", teaser="t", topic=None),
        make_sample_paper(title="A2", teaser="t", topic="Topic A"),
    ]
    assert _titles_and_marks(render_email(papers, 150)) == [
        "🔗 Alone 1", "📂 Topic A", "🔗 A1", "🔗 A2", RULE, "🔗 Alone 2", "🔗 Alone 3",
    ]


def test_render_shows_a_notice_above_the_list_and_escapes_it():
    paper = make_sample_paper(title="Paper A", teaser="t")
    html = render_email([paper], 150, notice="LLM <down>")
    assert "⚠ LLM &lt;down&gt;" in html
    assert html.index("⚠") < html.index("Paper A")
    assert "⚠" not in render_email([paper], 150)


def test_render_names_the_host_in_the_footer():
    paper = make_sample_paper(title="Paper A", teaser="__FOOTER__")
    html = render_email([paper], 150, host="Light1<x>")
    assert "執行主機：Light1&lt;x&gt;" in html
    assert html.index("Paper A") < html.index("執行主機")
    assert "__FOOTER__" in html  # paper text is not treated as a placeholder
    assert "執行主機" in render_email([], 150, host="Light2")
    assert "執行主機" not in render_email([paper], 150)
    assert "Github Action" not in html


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
