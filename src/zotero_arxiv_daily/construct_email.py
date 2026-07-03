from .protocol import Paper
from .personal_summary import DEEP_DIGEST_TITLE, SUMMARY_ANCHOR_ID
import html
import math
import re


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
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

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def _summary_mode(summary_config) -> str:
    if summary_config is None:
        return "legacy"
    return str(summary_config.get("mode", "teaser")).lower()


def _teaser_char_limit(summary_config) -> int:
    try:
        return max(1, int(summary_config.get("teaser_char_limit", 150)))
    except (TypeError, ValueError):
        return 150


def _anchor_from_url(url: str) -> str:
    return "paper-" + re.sub(r"[^a-zA-Z0-9_-]", "-", url or "unknown")


def _paper_url(paper: Paper) -> str:
    return paper.pdf_url or paper.url


def _authors_text(paper: Paper) -> str:
    author_list = [a for a in paper.authors]
    if len(author_list) <= 5:
        return ', '.join(author_list)
    return ', '.join(author_list[:3] + ['...'] + author_list[-2:])


def _affiliations_text(paper: Paper) -> str:
    if paper.affiliations is None:
        return 'Unknown Affiliation'
    affiliations = paper.affiliations[:5]
    text = ', '.join(affiliations)
    return text + ', ...' if len(paper.affiliations) > 5 else text


def _rate_text(paper: Paper) -> str:
    return round(paper.score, 1) if paper.score is not None else 'Unknown'


def _markdown_text_html(text: str) -> str:
    return (
        '<div style="white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word;">'
        f"{html.escape(text)}"
        "</div>"
    )


def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None, block_id:str=None, summary_anchor:str=None):
    anchor_wrapper = f'<a id="{block_id}" name="{block_id}" style="display:block;height:1px;line-height:1px;"></a>' if block_id else ''
    back_link_row = ""
    if summary_anchor:
        back_link_row = (
            "<tr>"
            f'<td style="font-size: 12px; padding: 4px 0 0 0;">'
            f'<a href="#{summary_anchor}" style="color: #d9534f; text-decoration: none;">回到今日超級速覽 ↑</a>'
            "</td>"
            "</tr>"
        )
    block_template = """
    {anchor_wrapper}
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    {back_link_row}
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(
        anchor_wrapper=anchor_wrapper,
        back_link_row=back_link_row,
        title=title,
        authors=authors,
        rate=rate,
        tldr=tldr,
        pdf_url=pdf_url,
        affiliations=affiliations,
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def _build_summary_section(papers: list[Paper], summary_config, include_details: bool) -> str:
    limit = _teaser_char_limit(summary_config)
    items = []
    for p in papers:
        summary = p.teaser or p.tldr or p.abstract or ""
        if len(summary) > limit:
            summary = summary[:limit].rstrip() + "..."
        href = f"#{_anchor_from_url(p.url)}" if include_details else _paper_url(p)
        items.append(
            '<li style="margin-bottom: 8px;">'
            f'<a href="{html.escape(href)}" style="color: #d9534f; text-decoration: underline; font-weight: 700;">'
            f'🔗 {html.escape(p.title)}</a> '
            f'<span style="color: #666; font-size: 0.9em;">({html.escape(_authors_text(p))})</span>：'
            f'<span style="color: #333;">{html.escape(summary)}</span></li>'
        )
    return f"""
<a id="{SUMMARY_ANCHOR_ID}" name="{SUMMARY_ANCHOR_ID}" style="display:block;height:1px;line-height:1px;"></a>
<div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #fff5e6; font-family: Arial, sans-serif; font-size: 14px; color: #333; line-height: 1.5;">
  <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">今日超級速覽</div>
  <ul style="margin: 0; padding-left: 20px;">
    {"".join(items)}
  </ul>
</div>
"""


def render_email(papers:list[Paper], summary_config=None) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())

    mode = _summary_mode(summary_config)
    include_summary = mode in {"full", "teaser", "abstract"}
    include_details = mode in {"legacy", "full"}

    if include_summary:
        parts.append(_build_summary_section(papers, summary_config, include_details=include_details))
        if not include_details:
            return framework.replace('__CONTENT__', ''.join(parts))

    for p in papers:
        tldr = p.tldr or ""
        if mode == "full":
            tldr = f"<strong>{DEEP_DIGEST_TITLE}</strong><br>" + _markdown_text_html(p.tldr_markdown or p.tldr or "")
        else:
            tldr = html.escape(tldr)
        parts.append(get_block_html(
            html.escape(p.title),
            html.escape(_authors_text(p)),
            html.escape(str(_rate_text(p))),
            tldr,
            html.escape(_paper_url(p)),
            html.escape(_affiliations_text(p)),
            block_id=_anchor_from_url(p.url) if mode == "full" else None,
            summary_anchor=SUMMARY_ANCHOR_ID if mode == "full" else None,
        ))

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
