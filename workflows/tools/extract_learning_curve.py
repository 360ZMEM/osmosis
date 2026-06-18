"""Extract learning curve from RSL-RL train.log into JSON.

Usage:
    python tools/extract_learning_curve.py <train.log> <out.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PAT_ITER = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
PAT_KV = re.compile(r"^\s+([\w/ ()-]+?):\s+(-?[\d\.eE+-]+)\s*$")


def parse(log: str) -> list[dict]:
    blocks: list[dict] = []
    cur: dict | None = None
    for line in log.splitlines():
        m = PAT_ITER.search(line)
        if m:
            if cur is not None:
                blocks.append(cur)
            cur = {"iter": int(m.group(1)), "total": int(m.group(2))}
            continue
        if cur is None:
            continue
        m = PAT_KV.match(line)
        if m:
            key = m.group(1).strip()
            try:
                cur[key] = float(m.group(2))
            except ValueError:
                pass
    if cur is not None:
        blocks.append(cur)
    return blocks


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: extract_learning_curve.py <train.log> <out.json>", file=sys.stderr)
        return 2
    log = Path(sys.argv[1]).read_text(errors="ignore")
    blocks = parse(log)
    Path(sys.argv[2]).write_text(json.dumps(blocks, indent=2))
    print(f"[OK] wrote {len(blocks)} iterations -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
