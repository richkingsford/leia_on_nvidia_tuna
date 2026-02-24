#!/usr/bin/env python3
"""Debug entrypoint for the offline E2E preflight simulation."""

from helper_e2e_simulation import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_RESET,
    COLOR_WHITE,
    COLOR_YELLOW,
    DEFAULT_STEP_ORDER,
    LogEntry,
    collect_simulation_logs,
    run_preflight,
)
from helper_e2e_simulation import main as _helper_main


def main() -> int:
    return int(_helper_main())


if __name__ == "__main__":
    raise SystemExit(main())
