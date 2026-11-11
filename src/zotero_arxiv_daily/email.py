"""Teaser-only email rendering: a "今日超級速覽" list grouped by topic."""

from __future__ import annotations

import html

from .paper import Paper

FRAMEWORK = """
<!DOCTYPE HTML>
<html>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div style="color: #888; font-size: 12px;">
__FOOTER__
</div>

</body>
</html>
"""


def _empty_html() -> str:
    return """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
      <td style="font-size: 20px; font-weight: bold; color: #333;">
          No Papers Today. Take a Rest!
      </td>
    </tr>
    </table>
    """


def _paper_url(paper: Paper) -> str:
    return paper.pdf_url or paper.url


def _authors_text(paper: Paper) -> str:
    if len(paper.authors) <= 5:
        return ", ".join(paper.authors)
    return ", ".join(paper.authors[:3] + ["..."] + paper.authors[-2:])


def _by_topic(papers: list[Paper]) -> list[list[Paper]]:
    """The papers of one topic together, in ranked order, with the topics in
    the order of their best-ranked paper. A paper without a topic is a group of
    its own at its own rank. (The ranked list interleaves the topics: printing
    a header at every change repeated one header up to 8 times in an email and
    left topic-less papers under the header before them.)"""
    groups: dict[str | int, list[Paper]] = {}
    for rank, paper in enumerate(papers):
        groups.setdefault(paper.topic or rank, []).append(paper)
    return list(groups.values())


def _frame(content: str, host: str) -> str:
    footer = f"執行主機：{html.escape(host)}" if host else ""
    return FRAMEWORK.replace("__FOOTER__", footer).replace("__CONTENT__", content)


def render_email(
    papers: list[Paper], teaser_char_limit: int, notice: str | None = None, host: str = ""
) -> str:
    """``notice`` is shown above the list: what went wrong in a run that
    still had papers to send. ``host`` names the machine that ran the
    pipeline, shown in the footer."""
    if not papers:
        return _frame(_empty_html(), host)

    items = []
    under_topic = False
    for group in _by_topic(papers):
        topic = group[0].topic
        if topic:
            items.append(
                '<li style="list-style: none; margin: 12px 0 4px -20px;">'
                '<div style="font-size: 13px; font-weight: bold; color: #b5502a;'
                " border-bottom: 1px solid #f0d9c0; padding-bottom: 2px;\">"
                f'📂 {html.escape(topic)}</div></li>'
            )
        elif under_topic:
            # A bare rule closes the topic above, so that this paper does not
            # read as one of its members.
            items.append(
                '<li style="list-style: none; margin: 12px 0 8px -20px;">'
                '<div style="border-bottom: 1px solid #f0d9c0;"></div></li>'
            )
        under_topic = bool(topic)
        for p in group:
            summary = p.teaser or p.abstract or ""
            if len(summary) > teaser_char_limit:
                summary = summary[:teaser_char_limit].rstrip() + "..."
            items.append(
                '<li style="margin-bottom: 8px;">'
                f'<a href="{html.escape(_paper_url(p))}" style="color: #d9534f; text-decoration: underline; font-weight: 700;">'
                f'🔗 {html.escape(p.title)}</a> '
                f'<span style="color: #666; font-size: 0.9em;">({html.escape(_authors_text(p))})</span>：'
                f'<span style="color: #333;">{html.escape(summary)}</span></li>'
            )
    notice_html = (
        '<div style="margin-bottom: 8px; padding: 6px 10px; border: 1px solid #d9534f;'
        f' border-radius: 4px; color: #a02622;">⚠ {html.escape(notice)}</div>'
        if notice
        else ""
    )
    content = f"""
<a id="super-summary" name="super-summary" style="display:block;height:1px;line-height:1px;"></a>
<div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #fff5e6; font-family: Arial, sans-serif; font-size: 14px; color: #333; line-height: 1.5;">
  <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">今日超級速覽</div>
  {notice_html}
  <ul style="margin: 0; padding-left: 20px;">
    {"".join(items)}
  </ul>
</div>
"""
    return _frame(content, host)
