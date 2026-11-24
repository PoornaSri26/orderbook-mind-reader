#!/usr/bin/env python
"""Freeze the pre-registration: fill the timestamp, write the hash, remind to commit."""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "PREREGISTRATION.md"
HASH = ROOT / "PREREGISTRATION.sha256"


def main() -> int:
    text = PRE.read_text(encoding="utf-8")
    if "__PENDING__" in text:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        PRE.write_text(text.replace("__PENDING__", now), encoding="utf-8")
        print(f"stamped pre-registration at {now}")
    elif not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        print("PREREGISTRATION.md has neither __PENDING__ nor an ISO stamp", file=sys.stderr)
        return 1
    else:
        print("already stamped; refreshing hash only")
    digest = hashlib.sha256(PRE.read_bytes()).hexdigest()
    HASH.write_text(f"{digest}  PREREGISTRATION.md\n", encoding="utf-8")
    print(f"sha256 {digest}\nnow: git add -A && git commit -m 'Freeze pre-registration'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
