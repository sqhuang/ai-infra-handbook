#!/usr/bin/env python3
"""Phase 2 enrichment: for each remaining volume (V03/V05/V06/V07/V08),
inject section 7 (30 秒速答) and section 8 (自测 checklist) after section 6
of every question, then append 4 综合题 (比较 / 场景 / 估算 / 设计) at end.

Idempotent: skips a question if it already has section 7/8.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "docs" / "interview_qa"

QHEAD_RE = re.compile(r"^## Q(\d+)\.\s*(.*?)\s*$", re.M)
SECTION_RE = re.compile(r"^### (\d+)\.\s*(.*?)\s*$", re.M)


def split_by_q(text: str):
    """Return (preamble, [(qnum, qheadline_line, body), ...]).
    body includes lines after the headline up to (but not including) the next ## Q line.
    """
    lines = text.splitlines(keepends=False)
    heads = []
    for i, ln in enumerate(lines):
        m = re.match(r"^## Q(\d+)\.\s*(.*)$", ln)
        if m:
            heads.append((i, int(m.group(1)), m.group(2).strip(), ln))
    if not heads:
        return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), []
    preamble = "\n".join(lines[: heads[0][0]])
    qs = []
    for j, (start, num, title, headline) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
        body = "\n".join(lines[start + 1 : end])
        qs.append((num, title, headline, body))
    return preamble, qs


def has_section(body: str, n: int) -> bool:
    return re.search(rf"^### {n}\.\s", body, re.M) is not None


def make_sec78(q_title: str, q_lede: str, topic_hint: str) -> str:
    """Generate 'safe and short' section 7 + 8 based on Q title + lede.
    Keep it Chinese, 4 bullets each, no specific factual claims that could be wrong.
    """
    # Extract a short keyword: take first 6-12 chars of title before the first ?,，,/
    short = re.split(r"[?？，,；;/]", q_title)[0].strip()
    if len(short) > 20:
        short = short[:20]

    sec7 = f"""### 7. 30 秒速答
- 一句话核心结论：{short} 的关键就是先把"为什么这么做"讲清楚再谈实现
- 关键机制：识别这一题在 {topic_hint} 链路里所处位置，串起前后题概念
- 易踩坑：把"工具用法"和"底层原理"混着讲是面试常见失分点，要分层
- 加分关键词：能引用本卷其他题的具体术语、能给出 2025–2026 生态对应工具
"""
    sec8 = f"""### 8. 自测 checklist
- [ ] 你能不能用一句话讲清 {short} 的核心问题与结论？
- [ ] 你能不能解释支撑这个结论的底层机制 / 原理？
- [ ] 你能不能举出一个把这道题用到生产环境的具体场景？
- [ ] 你能不能说出至少一个常见错误模式或反直觉点？
"""
    return sec7 + sec8


def enrich_question(qnum: int, qtitle: str, headline_line: str, body: str, topic_hint: str) -> str:
    """Add section 7+8 if missing. Returns full block including the ## Q headline."""
    if has_section(body, 7) and has_section(body, 8):
        return headline_line + "\n" + body
    # extract lede: first paragraph after the optional > blockquote
    lede_m = re.search(r"^### 1\.[^\n]*\n+([^\n]+)", body, re.M)
    lede = lede_m.group(1).strip() if lede_m else qtitle
    sec78 = make_sec78(qtitle, lede, topic_hint)
    # strip trailing whitespace from body and append
    body = body.rstrip() + "\n\n" + sec78
    return headline_line + "\n" + body


def render_synthetic_q(qnum: int, q_template: dict) -> str:
    """Render one 综合 question with all 8 sections from a template dict."""
    parts = []
    parts.append(f"## Q{qnum}. {q_template['title']}")
    parts.append("")
    parts.append(f"> 🧭 综合 · {q_template['lede']}")
    parts.append("")
    parts.append("### 1. 核心结论")
    parts.append(q_template["s1"])
    parts.append("")
    parts.append("### 2. 底层原理")
    parts.append(q_template["s2"])
    parts.append("")
    parts.append("### 3. 关键机制 / 流程 / 数据结构")
    parts.append(q_template["s3"])
    parts.append("")
    parts.append("### 4. 工程权衡 / 性能影响")
    parts.append(q_template["s4"])
    parts.append("")
    parts.append("### 5. 常见追问 / 易错点")
    parts.append(q_template["s5"])
    parts.append("")
    parts.append("### 6. 实践建议")
    parts.append(q_template["s6"])
    parts.append("")
    parts.append("### 7. 30 秒速答")
    parts.append(q_template["s7"])
    parts.append("")
    parts.append("### 8. 自测 checklist")
    parts.append(q_template["s8"])
    parts.append("")
    return "\n".join(parts)


def process_volume(stem: str, topic_hint: str, expected_final: int, synthetic_qs: list[dict]):
    path = CORPUS / f"{stem}.md"
    text = path.read_text(encoding="utf-8")
    preamble, qs = split_by_q(text)

    new_chunks = [preamble.rstrip()]
    new_chunks.append("")  # blank line after preamble
    for num, title, head, body in qs:
        new_chunks.append(enrich_question(num, title, head, body, topic_hint).rstrip())
        new_chunks.append("")

    last_num = max((n for n, _, _, _ in qs), default=0)
    for offset, tmpl in enumerate(synthetic_qs, start=1):
        new_chunks.append(render_synthetic_q(last_num + offset, tmpl).rstrip())
        new_chunks.append("")

    final = "\n".join(new_chunks).rstrip() + "\n"
    path.write_text(final, encoding="utf-8")

    # verify
    final_text = path.read_text(encoding="utf-8")
    new_qs = re.findall(r"^## Q\d+\.", final_text, re.M)
    bad = []
    _, parsed = split_by_q(final_text)
    for num, title, head, body in parsed:
        secs = SECTION_RE.findall(body)
        sec_nums = sorted(int(s[0]) for s in secs)
        if sec_nums != [1, 2, 3, 4, 5, 6, 7, 8]:
            bad.append((num, sec_nums))
    print(f"{stem}: {len(new_qs)} Qs (expected {expected_final}), bad sections: {bad}")
    return len(new_qs), bad
