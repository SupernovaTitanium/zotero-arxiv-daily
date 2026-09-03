"""Tests for zotero_arxiv_daily.construct_email: render_email, get_block_html."""

from zotero_arxiv_daily.construct_email import render_email, get_block_html, get_empty_html
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "MIT" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "No Papers Today" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "..." in html
    assert "Author 8" in html
    assert "Author 9" in html
    # Middle authors should be truncated
    assert "Author 5" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" in html


def test_render_email_teaser_mode_summary_only():
    paper = make_sample_paper(
        title="<b>Unsafe Title</b>",
        teaser="<script>alert(1)</script>",
        tldr="Detailed text",
        affiliations=["MIT"],
    )
    html = render_email([paper], {"mode": "teaser", "teaser_char_limit": 150})
    assert "今日超級速覽" in html
    assert "&lt;b&gt;Unsafe Title&lt;/b&gt;" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Detailed text" not in html
    assert "Unknown Affiliation" not in html
    assert "<strong>Relevance:</strong>" not in html


def test_render_email_full_mode_summary_and_deep_digest():
    paper = make_sample_paper(
        teaser="Short overview",
        tldr="Fallback detail",
        tldr_markdown="**A. 總結敘事**\n- useful",
        affiliations=["MIT"],
    )
    html = render_email([paper], {"mode": "full", "teaser_char_limit": 150})
    assert "今日超級速覽" in html
    assert "深度速覽" in html
    assert "回到今日超級速覽" in html
    assert "MIT" in html
    assert "PDF" in html


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "Auth", "3.5", "Summary", "http://pdf.url", "MIT")
    assert "Title" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "No Papers Today" in html


def test_render_email_topic_headers_teaser_mode():
    paper_a = make_sample_paper(title="Topic One Paper", abstract="abstract a")
    paper_a.teaser = "Teaser a"
    paper_a.topic = "Vision Research 等 2 篇"
    paper_b = make_sample_paper(title="Topic Two Paper", abstract="abstract b")
    paper_b.teaser = "Teaser b"
    html = render_email([paper_a, paper_b], {"mode": "teaser", "teaser_char_limit": 100})
    assert "📂 Vision Research 等 2 篇" in html
    # header appears once, before the first member
    assert html.index("📂") < html.index("Topic One Paper")
