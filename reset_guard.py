#!/usr/bin/env python3
"""Flag a reset log that shows the wrong-way / phantom-distance signature.

A real backward reset moves distance gradually. The cycle-2 fault teleported
77->223mm in a single pulse while the robot physically drove forward. We flag
any jump between consecutive distance reads larger than THRESHOLD_MM so the
driver can stop before a second wrong-way reset.
"""

import re
import sys

THRESHOLD_MM = 120.0

path = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"
try:
    text = open(path).read()
except Exception as exc:  # noqa: BLE001
    print(f"GUARD_ERROR:{exc}")
    sys.exit(0)

dists = [float(m) for m in re.findall(r"dist=([0-9]+(?:\.[0-9]+)?)mm", text)]
max_jump = 0.0
for a, b in zip(dists, dists[1:]):
    max_jump = max(max_jump, abs(b - a))

not_honest = "DONE_NOT_HONEST" in text
if max_jump > THRESHOLD_MM:
    print(f"SUSPECT teleport max_jump={max_jump:.1f}mm (>{THRESHOLD_MM:.0f}); reset may have driven wrong-way")
elif not_honest:
    print("NOT_HONEST reset did not reach honest pose")
else:
    print(f"OK max_jump={max_jump:.1f}mm")
