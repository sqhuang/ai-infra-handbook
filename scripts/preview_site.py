#!/usr/bin/env python3
"""
Local preview server for docs/site/.

Serves docs/site/ at http://localhost:<port> (default 8765).
"""
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
        # Default logger is noisy. Keep only error logs.
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            super().log_message(fmt, *args)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="127.0.0.1")
    args = p.parse_args()

    if not SITE.exists():
        print(f"docs/site/ not found at {SITE}", file=sys.stderr)
        print("Run: python3 scripts/build_interview_qa_site.py", file=sys.stderr)
        return 1

    os.chdir(SITE)
    with socketserver.TCPServer((args.bind, args.port), QuietHandler) as httpd:
        url = f"http://{args.bind}:{args.port}/index.html"
        print(f"serving {SITE} at {url}")
        print("Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
