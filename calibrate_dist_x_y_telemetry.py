#!/usr/bin/env python3
"""Standalone wrapper for telemetry calibration.

The implementation lives in helper_calibrate_telemetry so the same flow can be
launched from the main calibration menu.
"""

from __future__ import annotations

import helper_calibrate_telemetry


def main() -> int:
    return int(helper_calibrate_telemetry.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
