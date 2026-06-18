#!/usr/bin/env python3
"""Print the latest-attempt phase statuses from the frozen step1/2 progress site."""

import json
import sys

PATH = "runs/frozen_step12_site/state.json"

try:
    state = json.load(open(PATH))
except Exception as exc:  # noqa: BLE001
    print(f"state_read_error:{exc}")
    sys.exit(0)

rows = state.get("rows", [])
latest = max((int(r.get("attempt", 0) or 0) for r in rows), default=0)
parts = []
for row in rows:
    if int(row.get("attempt", 0) or 0) != latest:
        continue
    phase = str(row.get("phase", ""))
    if phase.endswith("-start") or not phase:
        continue
    parts.append(f"{phase}={row.get('status')}")
print(f"attempt {latest} | " + " ".join(parts))
