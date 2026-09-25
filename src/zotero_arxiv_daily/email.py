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
<div>
To unsubscribe, remove your email in your Github Action setting.
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


def render_email(papers: list[Paper], teaser_char_limit: int) -> str:
    if not papers:
        return FRAMEWORK.replace("__CONTENT__", _empty_html())

    items = []
    current_topic = None
    for p in papers:
        if p.topic and p.topic != current_topic:
            current_topic = p.topic
            items.append(
                '<li style="list-style: none; margin: 12px 0 4px -20px;">'
                '<div style="font-size: 13px; font-weight: bold; color: #b5502a;'
                " border-bottom: 1px solid #f0d9c0; padding-bottom: 2px;\">"
                f'📂 {html.escape(p.topic)}</div></li>'
            )
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
    content = f"""
<a id="super-summary" name="super-summary" style="display:block;height:1px;line-height:1px;"></a>
<div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #fff5e6; font-family: Arial, sans-serif; font-size: 14px; color: #333; line-height: 1.5;">
  <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">今日超級速覽</div>
  <ul style="margin: 0; padding-left: 20px;">
    {"".join(items)}
  </ul>
</div>
"""
    return FRAMEWORK.replace("__CONTENT__", content)
