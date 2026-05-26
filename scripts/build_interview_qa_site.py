#!/usr/bin/env python3
"""
Build the editorial QA site at docs/site/ from docs/interview_qa/ Markdown.

Reads:
  docs/interview_qa/README.md
  docs/interview_qa/01-pytorch-internals.md ... 10-postmortems-and-lessons.md

Writes:
  docs/site/index.html
  docs/site/01-pytorch-internals.html ... 10-postmortems-and-lessons.html
  docs/site/assets/style.css           (copied from <repo>/assets/style.css)
  docs/site/assets/app.js              (copied from <repo>/assets/app.js)
  docs/site/assets/data/search-index.json
  docs/site/assets/data/search-index.js

Idempotent.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from site_render import (  # noqa: E402
    Question,
    Volume,
    extract_lede,
    parse_volume,
    render_markdown,
    render_question,
)

CORPUS = ROOT / "docs" / "interview_qa"
SITE = ROOT / "docs" / "site"
ASSETS_SRC = ROOT / "assets"
ASSETS_OUT = SITE / "assets"

VOLUMES = [
    "01-pytorch-internals",
    "02-compiler-and-ir",
    "03-cuda-triton-custom-ops",
    "04-distributed-training",
    "05-memory-communication-profiling",
    "06-inference-quantization-serving",
    "07-training-inference-platform",
    "08-systems-foundations",
    "09-strategy-engineering-decisions",
    "10-postmortems-and-lessons",
    "11-posttraining-new-inference",
    "12-aigc-generation-infra",
    "13-recommendation-inference",
    "14-domestic-chips-heterogeneous",
]

# Per-volume question count after 2026 enrichment: each existing volume has
# +4 综合题 (比较 / 场景 / 估算 / 设计) appended, every question carries the
# full 8-section template (1-6 original + 7 30-秒速答 + 8 自测 checklist),
# and 11-posttraining-new-inference is the new 22-题 volume.
VOLUME_QCOUNT = {
    "01-pytorch-internals":              24,
    "02-compiler-and-ir":                20,
    "03-cuda-triton-custom-ops":         49,
    "04-distributed-training":           59,
    "05-memory-communication-profiling": 69,
    "06-inference-quantization-serving": 65,
    "07-training-inference-platform":    84,
    "08-systems-foundations":            39,
    "09-strategy-engineering-decisions": 34,
    "10-postmortems-and-lessons":        29,
    "11-posttraining-new-inference":     28,
    "12-aigc-generation-infra":          25,
    "13-recommendation-inference":       20,
    "14-domestic-chips-heterogeneous":   20,
}


def head_html(title: str, *, page_kind: str, rel: str = "") -> str:
    progress = '<div class="progress"><i></i></div>' if page_kind == "volume" else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+Pro:ital,wght@0,400;0,500;0,600;1,400&display=swap" />
<link rel="stylesheet" href="{rel}assets/style.css" />
<script>(function(){{try{{var t=localStorage.getItem('iqa.theme');if(t==='light'||t==='dark'){{document.documentElement.setAttribute('data-theme',t);}}}}catch(e){{}}}})();</script>
<script defer src="{rel}assets/data/search-index.js"></script>
<script defer src="{rel}assets/app.js"></script>
</head>
<body>
{progress}
<header class="site-header"><div class="inner">
  <a class="site-brand" href="{rel}index.html">AI infra 大百科</a>
  <nav class="site-nav">
    <a href="{rel}index.html">Home</a>
    <a href="{rel}glossary.html">Glossary</a>
    <a href="https://github.com" target="_blank" rel="noopener">Repo</a>
  </nav>
  <button class="icon-btn" type="button" data-action="open-palette">
    <span>搜索</span><span class="kbd">⌘K</span>
  </button>
  <button class="icon-btn" type="button" data-action="toggle-theme">☾  Light</button>
</div></header>
"""


FOOTER_HTML = """
<footer class="site-footer"><div class="inner">
  <span>© 2026 · AI infra 大百科 · 10 卷</span>
  <span>修订 · 2026-04-23 · 站点构建 · 2026-04-26</span>
</div></footer>
</body></html>
"""


def build_index_html(volumes: list[Volume]) -> str:
    head = head_html("AI infra 大百科 · 索引", page_kind="index")

    tiles = []
    for stem, vol in zip(VOLUMES, volumes):
        total = len(vol.questions)
        if total == 0:
            range_label = "（待补）"
        else:
            range_label = f"Q1–Q{total}"
        vol_num = stem.split("-", 1)[0]
        tiles.append(f"""\
<a class="vol-tile" href="{stem}.html">
  <div class="vt-num">VOL · {vol_num}</div>
  <div class="vt-title">{escape(vol.title)}</div>
  <div class="vt-foot">{total} 题 · {range_label}</div>
</a>""")

    readme = (CORPUS / "README.md").read_text(encoding="utf-8")
    usage = _readme_section(readme, "如何使用")

    total_qs = sum(len(v.questions) for v in volumes)

    body = f"""<main class="page">

<section class="hero">
  <div class="hero-kicker">AI infra 大百科 · {total_qs} 题 · {len(volumes)} 卷 · 2026 修订</div>
  <h1>从 PyTorch 内核到分布式训练，从推理引擎到平台调度</h1>
  <p class="lede">{total_qs} 道结构化深度问答，按工程实践脉络分卷。每题以六节模板成文：核心结论、底层原理、关键机制、工程权衡、易错点、实践建议。</p>
  <div class="hero-meta">
    <span>修订日期 · <b>2026-04-23</b></span>
    <span>站点构建 · <b>2026-04-26</b></span>
    <span>最新覆盖 · <b>FSDP2 · vLLM V1 · NCCL 2.27 · Blackwell FP4</b></span>
  </div>
  <div class="vol-grid">
{chr(10).join(tiles)}
  </div>
  <div class="select-volume-note">SELECT VOLUME →</div>
</section>

{_render_index_section("如何使用", usage)}

</main>
"""
    return head + body + FOOTER_HTML


def _render_index_section(heading: str, md_body: str) -> str:
    if not md_body.strip():
        return ""
    return f"""<section class="section">
  <h2>{escape(heading)}</h2>
  {render_markdown(md_body)}
</section>"""


def _readme_section(readme_text: str, heading: str) -> str:
    pat = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.M)
    m = pat.search(readme_text)
    if not m:
        return ""
    start = m.end()
    next_m = re.search(r"^##\s+", readme_text[start:], re.M)
    end = start + next_m.start() if next_m else len(readme_text)
    return readme_text[start:end].strip()


def build_volume_html(stem: str, vol: Volume, related_map: dict | None = None) -> str:
    head = head_html(f"{vol.title} · 访谈语料", page_kind="volume")
    vn = f"v{vol.vol_num:02d}"

    left_toc = ['<aside class="toc"><h3>本卷 Q 列表</h3><ol>']
    for q in vol.questions:
        left_toc.append(
            f'<li><a data-q="{vn}-q-{q.q}" href="#{vn}-q-{q.q}">'
            f'<span class="qnum">Q{q.q}</span>{escape(q.title)}'
            f'</a></li>'
        )
    left_toc.append("</ol></aside>")

    sections_html = [
        render_question(
            q,
            related=(related_map or {}).get((vol.vol_num, q.q)),
        )
        for q in vol.questions
    ]
    intro_html = render_markdown(vol.intro) if vol.intro else ""

    # Dynamic right-TOC built from the first question's section list so that
    # the 8-section template (which appends 30-second 速答 + 自测 checklist)
    # and the legacy 6-section template both render correctly.
    first_q = vol.questions[0] if vol.questions else None
    if first_q and first_q.sections:
        toc_items = "\n".join(
            f'<li><a data-sec="{s.n}" href="#{vn}-q-{first_q.q}-s-{s.n}">'
            f'{escape(s.title)}</a></li>'
            for s in first_q.sections
        )
        right_toc = (
            f'<aside class="toc right"><h3>本题 {len(first_q.sections)} 节</h3><ol>'
            f'{toc_items}'
            '</ol></aside>'
        )
    else:
        right_toc = '<aside class="toc right"><h3>本题</h3><ol></ol></aside>'

    body = f"""<main class="page">

<div class="volume-layout">

{chr(10).join(left_toc)}

<article class="content">
  <h1>{escape(vol.title)}</h1>
  {f'<div class="vol-intro">{intro_html}</div>' if intro_html else ''}
  {chr(10).join(sections_html)}
</article>

{right_toc}

</div>
</main>
"""
    return head + body + FOOTER_HTML


def build_glossary_html() -> str:
    """Render docs/interview_qa/glossary.md as docs/site/glossary.html."""
    head = head_html("术语表 · AI infra 大百科", page_kind="index")
    src = CORPUS / "glossary.md"
    md = src.read_text(encoding="utf-8") if src.exists() else "# 术语表\n\n（待补）"
    body = f"""<main class="page">
<section class="section">
{render_markdown(md)}
</section>
</main>
"""
    return head + body + FOOTER_HTML


def _bigrams(text: str) -> set[str]:
    """Build a set of consecutive 2-Unicode-char bigrams from `text`,
    skipping ASCII punctuation and whitespace characters.

    The same character can appear in multiple bigrams. Treats the input as a
    sequence of "kept" characters (CJK + word-ish chars) and forms bigrams
    over neighbours in that filtered sequence.
    """
    kept: list[str] = []
    for ch in text:
        if ch.isspace():
            continue
        # Skip ASCII punctuation. Keep non-ASCII punctuation (e.g. CJK 全角)
        # because dropping all "non-word" CJK noise risks zeroing out features
        # for short titles; the bigram of two CJK chars is the load-bearing
        # signal anyway.
        if ord(ch) < 128 and not (ch.isalnum()):
            continue
        kept.append(ch.lower())
    if len(kept) < 2:
        return set()
    return {kept[i] + kept[i + 1] for i in range(len(kept) - 1)}


def build_related_map(
    volumes: list[Volume],
    *,
    top_k: int = 4,
    min_top_score: float = 0.05,
) -> dict[tuple[int, int], list[dict]]:
    """For every (vol_num, q) compute up to `top_k` most similar peers by
    Jaccard similarity over character bigrams of `title + lede`.

    Returns a map keyed by (vol_num, q) → list of dicts with fields
    `vol`, `vol_num`, `q`, `title` (search-index record shape, minimal).
    Questions whose top-1 similarity falls below `min_top_score` are mapped
    to an empty list (so the renderer skips the see-also block).
    """
    # Flatten into a stable-ordered index list and precompute features.
    items: list[tuple[str, Volume, Question, set[str]]] = []
    for stem, vol in zip(VOLUMES, volumes):
        for q in vol.questions:
            feature = q.title + " " + extract_lede(q, max_chars=10_000)
            items.append((stem, vol, q, _bigrams(feature)))

    related: dict[tuple[int, int], list[dict]] = {}
    n = len(items)
    for i in range(n):
        stem_a, vol_a, q_a, bg_a = items[i]
        if not bg_a:
            related[(vol_a.vol_num, q_a.q)] = []
            continue
        scored: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            _, _, _, bg_b = items[j]
            if not bg_b:
                continue
            inter = len(bg_a & bg_b)
            if inter == 0:
                continue
            union = len(bg_a | bg_b)
            sim = inter / union
            if sim > 0:
                scored.append((sim, j))
        scored.sort(key=lambda t: (-t[0], t[1]))
        if not scored or scored[0][0] < min_top_score:
            related[(vol_a.vol_num, q_a.q)] = []
            continue
        top = scored[:top_k]
        related[(vol_a.vol_num, q_a.q)] = [
            {
                "vol": items[j][0],
                "vol_num": items[j][1].vol_num,
                "q": items[j][2].q,
                "title": items[j][2].title,
            }
            for _, j in top
        ]
    return related


def build_search_records(volumes: list[Volume]) -> list[dict]:
    records: list[dict] = []
    for stem, vol in zip(VOLUMES, volumes):
        for q in vol.questions:
            records.append({
                "vol": stem,
                "vol_num": vol.vol_num,
                "vol_title": vol.title,
                "q": q.q,
                "anchor": f"v{vol.vol_num:02d}-q-{q.q}",
                "title": q.title,
                "lede": extract_lede(q),
            })
    records.sort(key=lambda r: (r["vol_num"], r["q"]))
    return records


def copy_assets() -> None:
    ASSETS_OUT.mkdir(parents=True, exist_ok=True)
    for name in ("style.css", "app.js"):
        src = ASSETS_SRC / name
        dst = ASSETS_OUT / name
        dst.write_bytes(src.read_bytes())


def escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    SITE.mkdir(parents=True, exist_ok=True)

    volumes: list[Volume] = []
    for stem in VOLUMES:
        path = CORPUS / f"{stem}.md"
        vol_num = int(stem.split("-", 1)[0])
        if not path.exists():
            volumes.append(Volume(title=stem, intro="", questions=[], vol_num=vol_num))
            continue
        text = path.read_text(encoding="utf-8")
        vol = parse_volume(text, vol_num=vol_num)
        actual = [q.q for q in vol.questions]
        expected_count = VOLUME_QCOUNT[stem]
        expected = list(range(1, expected_count + 1))
        if expected_count > 0 and actual != expected:
            missing = [q for q in expected if q not in actual]
            extra = [q for q in actual if q not in expected]
            raise SystemExit(
                f"build aborted: {stem}.md per-volume Q set mismatch "
                f"(expected 1..{expected_count}, got {actual}, "
                f"missing={missing}, extra={extra})"
            )
        volumes.append(vol)

    related_map = build_related_map(volumes)

    for stem, vol in zip(VOLUMES, volumes):
        write(SITE / f"{stem}.html", build_volume_html(stem, vol, related_map))

    write(SITE / "index.html", build_index_html(volumes))
    write(SITE / "glossary.html", build_glossary_html())
    copy_assets()

    records = build_search_records(volumes)
    data_dir = ASSETS_OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write(data_dir / "search-index.json", json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    write(
        data_dir / "search-index.js",
        "window.__SEARCH_INDEX__ = " + json.dumps(records, ensure_ascii=False) + ";\n",
    )

    print(f"built {len(volumes)} volumes, {sum(len(v.questions) for v in volumes)} questions")
    print(f"output: {SITE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
