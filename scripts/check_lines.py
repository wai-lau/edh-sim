#!/usr/bin/env python
"""Fail if any tracked .py file exceeds the line cap (ruff has no such rule)."""
import pathlib
import sys

MAX = 500
SKIP = {".venv", ".git", "__pycache__", "build", "dist"}

bad = []
for p in sorted(pathlib.Path(".").rglob("*.py")):
    if SKIP & set(p.parts):
        continue
    n = sum(1 for _ in p.open(encoding="utf-8"))
    if n > MAX:
        bad.append((p, n))

for p, n in bad:
    print(f"{p}: {n} lines > {MAX}")
sys.exit(1 if bad else 0)
