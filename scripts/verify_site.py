#!/usr/bin/env python3
"""
Verify the editorial QA site at docs/site/ under per-volume ID schema.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"

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

FINAL_QCOUNT = {
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


def main() -> int:
    errors: list[str] = []
    sys.path.insert(0, str(ROOT / "scripts"))

    bjs = (ROOT / "scripts" / "build_interview_qa_site.py").read_text(encoding="utf-8")
    qcount: dict[str, int] = {}
    for m in re.finditer(r'"([0-9]{2}-[a-z0-9-]+)":\s+(\d+)', bjs):
        qcount[m.group(1)] = int(m.group(2))
    if not qcount:
        print("could not parse VOLUME_QCOUNT from build_interview_qa_site.py", file=sys.stderr)
        return 1

    for name in ["index.html", *(f"{s}.html" for s in VOLUMES)]:
        p = SITE / name
        if not p.exists():
            errors.append(f"missing page: {p}")

    total_qs = 0
    for stem in VOLUMES:
        p = SITE / f"{stem}.html"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        vol_num = int(stem.split("-", 1)[0])
        prefix = f"v{vol_num:02d}-q-"
        ids = sorted(int(m.group(1)) for m in re.finditer(rf'id="{prefix}(\d+)"', text))
        expected = list(range(1, qcount.get(stem, 0) + 1))
        if ids != expected:
            missing = [q for q in expected if q not in ids]
            extra = [q for q in ids if q not in expected]
            errors.append(f"{stem}.html id mismatch: expected 1..{qcount.get(stem, 0)} got {ids} "
                          f"(missing={missing}, extra={extra})")
        total_qs += len(ids)
        if qcount.get(stem, 0) > FINAL_QCOUNT[stem]:
            errors.append(f"{stem}: VOLUME_QCOUNT exceeds FINAL_QCOUNT for this volume")

    json_path = SITE / "assets" / "data" / "search-index.json"
    if not json_path.exists():
        errors.append(f"missing: {json_path}")
    else:
        try:
            recs = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"search-index.json invalid: {e}")
            recs = []
        if len(recs) != total_qs:
            errors.append(f"search-index.json has {len(recs)} records, expected {total_qs}")
        for r in recs[:3]:
            for f in ("vol", "vol_num", "vol_title", "q", "anchor", "title", "lede"):
                if f not in r:
                    errors.append(f"search-index.json record missing field {f}: {r}")
                    break

    js_path = SITE / "assets" / "data" / "search-index.js"
    if not js_path.exists():
        errors.append(f"missing: {js_path}")
    else:
        js = js_path.read_text(encoding="utf-8")
        if not js.startswith("window.__SEARCH_INDEX__"):
            errors.append("search-index.js missing window.__SEARCH_INDEX__ assignment")

    for name in ("style.css", "app.js"):
        p = SITE / "assets" / name
        if not p.exists():
            errors.append(f"missing: {p}")

    if (SITE / "index.html").exists():
        idx_text = (SITE / "index.html").read_text(encoding="utf-8")
        for href in ("assets/style.css", "assets/app.js", "assets/data/search-index.js"):
            if href not in idx_text:
                errors.append(f"index.html missing reference to {href}")

    for stem in VOLUMES:
        p = SITE / f"{stem}.html"
        if not p.exists():
            continue
        if qcount.get(stem, 0) == 0:
            continue
        text = p.read_text(encoding="utf-8")
        if 'class="toc"' not in text:
            errors.append(f"{stem}.html missing left aside.toc")
        if 'class="toc right"' not in text:
            errors.append(f"{stem}.html missing right aside.toc.right")
        if 'class="content"' not in text:
            errors.append(f"{stem}.html missing article.content")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print(f"OK ({total_qs} questions across {len(VOLUMES)} volumes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
