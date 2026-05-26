#!/usr/bin/env python3
"""
Shared Markdown → HTML renderer for the editorial QA site.

Public surface:
  parse_volume(text, vol_num) -> Volume
  Volume.questions -> list[Question]
  render_markdown(md_text, prefix) -> str (HTML fragment)
  render_question(q) -> str (HTML fragment for one question)
  extract_lede(question, max_chars=80) -> str
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


FENCE_RE = re.compile(r"^```([\w-]*)\s*$")
H_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ULIST_RE = re.compile(r"^(\s*)([-*])\s+(.*)$")
OLIST_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
QHEAD_RE = re.compile(r"^## Q(\d+)\.\s*(.*?)\s*$")
SECTION_RE = re.compile(r"^### (\d+)\.\s*(.*?)\s*$")


@dataclass
class Section:
    n: int
    title: str
    body: str


@dataclass
class Question:
    q: int
    title: str
    sections: List[Section]
    raw: str = ""
    vol_num: int = 0  # 1..10; set by parse_volume


@dataclass
class Volume:
    title: str
    intro: str
    questions: List[Question] = field(default_factory=list)
    vol_num: int = 0


def parse_volume(text: str, vol_num: int) -> Volume:
    """Parse a volume Markdown. vol_num is the integer 1..10; used for ID generation."""
    lines = text.splitlines()
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:])
    else:
        body = text

    heads: List[Tuple[int, int, str]] = []
    for i, ln in enumerate(body.splitlines()):
        m = QHEAD_RE.match(ln)
        if m:
            heads.append((i, int(m.group(1)), m.group(2).strip()))

    if not heads:
        return Volume(title=title, intro=body.strip(), questions=[], vol_num=vol_num)

    body_lines = body.splitlines()
    intro = "\n".join(body_lines[: heads[0][0]]).strip()

    questions: List[Question] = []
    for j, (start, qid, qtitle) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(body_lines)
        block = "\n".join(body_lines[start:end])
        q = _parse_question(qid, qtitle, block)
        q.vol_num = vol_num
        questions.append(q)
    return Volume(title=title, intro=intro, questions=questions, vol_num=vol_num)


def _parse_question(qid: int, qtitle: str, block: str) -> Question:
    lines = block.splitlines()
    sections: List[Section] = []
    current: Optional[Tuple[int, str, List[str]]] = None
    for ln in lines[1:]:
        m = SECTION_RE.match(ln)
        if m:
            if current is not None:
                n, t, buf = current
                sections.append(Section(n=n, title=t, body="\n".join(buf).strip()))
            current = (int(m.group(1)), m.group(2).strip(), [])
        else:
            if current is None:
                continue
            current[2].append(ln)
    if current is not None:
        n, t, buf = current
        sections.append(Section(n=n, title=t, body="\n".join(buf).strip()))
    return Question(q=qid, title=qtitle, sections=sections, raw=block)


def extract_lede(question: Question, max_chars: int = 80) -> str:
    sec1 = next((s for s in question.sections if s.n == 1), None)
    if not sec1:
        return ""
    paragraphs = [p.strip() for p in sec1.body.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    text = paragraphs[0].replace("\n", " ")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = LINK_RE.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{html.escape(m.group(1))}</a>',
        out,
    )
    out = BOLD_RE.sub(r"<strong>\1</strong>", out)
    out = INLINE_CODE_RE.sub(
        lambda m: f"<code>{html.escape(m.group(1), quote=False)}</code>", out
    )
    return out


def render_markdown(md: str, prefix: str = "") -> str:
    lines = md.splitlines()
    out: List[str] = []
    i = 0
    in_code = False
    code_lang = ""
    para: List[str] = []
    list_stack: List[Tuple[str, int]] = []

    def flush_para() -> None:
        if para:
            joined = " ".join(p.strip() for p in para if p.strip())
            if joined:
                out.append(f"<p>{_inline(joined)}</p>")
            para.clear()

    def close_lists(to_indent: int = -1) -> None:
        while list_stack and list_stack[-1][1] > to_indent:
            kind, _ = list_stack.pop()
            out.append(f"</{kind}>")

    while i < len(lines):
        line = lines[i]
        fence = FENCE_RE.match(line)
        if fence:
            if in_code:
                out.append("</code></pre>")
                in_code = False
                code_lang = ""
            else:
                flush_para(); close_lists()
                code_lang = fence.group(1)
                cls = f' class="lang-{code_lang}"' if code_lang else ""
                out.append(f"<pre><code{cls}>")
                in_code = True
            i += 1
            continue
        if in_code:
            out.append(html.escape(line, quote=False))
            i += 1
            continue
        if not line.strip():
            flush_para(); close_lists()
            i += 1
            continue
        mh = H_RE.match(line)
        if mh:
            flush_para(); close_lists()
            level = len(mh.group(1))
            text = mh.group(2)
            out.append(f"<h{level}>{_inline(text)}</h{level}>")
            i += 1
            continue
        mu = ULIST_RE.match(line)
        if mu:
            flush_para()
            indent = len(mu.group(1))
            item = mu.group(3)
            while list_stack and list_stack[-1][1] > indent:
                kind, _ = list_stack.pop()
                out.append(f"</{kind}>")
            if not list_stack or list_stack[-1][1] < indent or list_stack[-1][0] != "ul":
                if list_stack and list_stack[-1][1] == indent and list_stack[-1][0] != "ul":
                    out.append(f"</{list_stack.pop()[0]}>")
                out.append("<ul>")
                list_stack.append(("ul", indent))
            out.append(f"<li>{_inline(item)}</li>")
            i += 1
            continue
        mo = OLIST_RE.match(line)
        if mo:
            flush_para()
            indent = len(mo.group(1))
            item = mo.group(3)
            while list_stack and list_stack[-1][1] > indent:
                kind, _ = list_stack.pop()
                out.append(f"</{kind}>")
            if not list_stack or list_stack[-1][1] < indent or list_stack[-1][0] != "ol":
                if list_stack and list_stack[-1][1] == indent and list_stack[-1][0] != "ol":
                    out.append(f"</{list_stack.pop()[0]}>")
                out.append("<ol>")
                list_stack.append(("ol", indent))
            out.append(f"<li>{_inline(item)}</li>")
            i += 1
            continue
        if line.startswith("> "):
            flush_para(); close_lists()
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
            i += 1
            continue
        if line.startswith("|") and "|" in line[1:]:
            flush_para(); close_lists()
            rows: List[List[str]] = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            if len(rows) >= 2 and all(set(c) <= set("-: ") and c for c in rows[1]):
                head = rows[0]; body = rows[2:]
            else:
                head = None; body = rows
            out.append("<table>")
            if head:
                out.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead>")
            out.append("<tbody>")
            for row in body:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue
        para.append(line.strip())
        i += 1
    flush_para(); close_lists()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def render_question(q: Question, related: Optional[List[dict]] = None) -> str:
    """Render one question to HTML using the per-volume ID schema.

    `related`, if provided and non-empty, attaches a `<aside class="see-also">`
    block after the question's last section. Each entry should provide
    `vol`, `vol_num`, `q`, and `title` (matching the search-index record shape).
    """
    vn = f"v{q.vol_num:02d}"
    parts: List[str] = []
    parts.append(
        f'<h2 id="{vn}-q-{q.q}"><span class="qchip">Q{q.q}</span>{_inline(q.title)}</h2>'
    )
    for s in q.sections:
        parts.append(f'<h3 id="{vn}-q-{q.q}-s-{s.n}">{s.n}. {_inline(s.title)}</h3>')
        if s.body:
            parts.append(render_markdown(s.body, prefix=f"{vn}q{q.q}s{s.n}"))
    if related:
        items: List[str] = []
        for r in related:
            href = f'{r["vol"]}.html#v{r["vol_num"]:02d}-q-{r["q"]}'
            items.append(
                f'    <li><a href="{html.escape(href, quote=True)}">'
                f'<span class="vchip">Vol {r["vol_num"]:02d}</span>'
                f'<span class="qchip">Q{r["q"]}</span>'
                f'<span class="ttl">{_inline(r["title"])}</span></a></li>'
            )
        parts.append(
            '<aside class="see-also">\n'
            '  <h4>相关题</h4>\n'
            '  <ul>\n'
            + "\n".join(items) + "\n"
            '  </ul>\n'
            '</aside>'
        )
    return "\n".join(parts)
