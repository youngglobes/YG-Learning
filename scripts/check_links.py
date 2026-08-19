#!/usr/bin/env python3
"""Check that every relative markdown link in the repo resolves.

Exits non-zero and lists the offenders if any are dead.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LINK = re.compile(r"\]\((\.\.?/[^)\s#]+)")
SKIP = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    dead: list[str] = []
    checked = 0

    for md in sorted(root.rglob("*.md")):
        if any(part in SKIP for part in md.parts):
            continue
        for target in LINK.findall(md.read_text(encoding="utf-8")):
            checked += 1
            if not (md.parent / target).exists():
                dead.append(f"{md.relative_to(root)} -> {target}")

    if dead:
        print(f"{len(dead)} dead link(s) of {checked} checked:\n")
        for d in dead:
            print(f"  DEAD  {d}")
        return 1

    print(f"All {checked} relative links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
