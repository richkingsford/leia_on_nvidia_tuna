#!/usr/bin/env python3
"""Interactive calibration launcher.

Shows available calibration workflows and runs the selected implementation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

from calibration import helper_calibrate_dist
from calibration import helper_calibrate_motion
from calibration import helper_calibrate_speed
from calibration import helper_calibrate_x
from calibration import helper_calibrate_x_axis
from calibration import helper_calibrate_y


@dataclass(frozen=True)
class CalibrateOption:
    key: str
    label: str
    runner: Callable[[], int | None]


OPTIONS: tuple[CalibrateOption, ...] = (
    CalibrateOption("x", "X-axis curve calibration", helper_calibrate_x.main),
    CalibrateOption("y", "Y-axis curve calibration", helper_calibrate_y.main),
    CalibrateOption("dist", "Distance curve calibration", helper_calibrate_dist.main),
    CalibrateOption("speed", "Speed endpoint calibration", helper_calibrate_speed.main),
    CalibrateOption("motion", "Motion tick conversion calibration", helper_calibrate_motion.main),
    CalibrateOption("x-axis-legacy", "Legacy X-axis learning experiment", helper_calibrate_x_axis.main),
)


def _print_menu() -> None:
    print("\nCalibration Options")
    print("-------------------")
    for index, option in enumerate(OPTIONS, start=1):
        print(f"  {index}. {option.label} [{option.key}]")
    print("  q. Quit")


def _resolve_choice(text: str) -> CalibrateOption | None:
    token = str(text or "").strip().lower()
    if not token:
        return None
    for index, option in enumerate(OPTIONS, start=1):
        if token in (str(index), str(option.key).lower()):
            return option
    return None


def _pick_interactive() -> CalibrateOption | None:
    while True:
        _print_menu()
        choice = input("Select calibration to run: ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            return None
        selected = _resolve_choice(choice)
        if selected is not None:
            return selected
        print(f"Unknown selection: {choice!r}. Please choose a number, key, or q.")


def _run_selected(option: CalibrateOption, passthrough_args: list[str]) -> int:
    original_argv = list(sys.argv)
    try:
        sys.argv = [f"calibrate:{option.key}"] + list(passthrough_args)
        result = option.runner()
        if result is None:
            return 0
        return int(result)
    finally:
        sys.argv = original_argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive calibration launcher")
    parser.add_argument(
        "--choice",
        type=str,
        default=None,
        help="Optional non-interactive selection key or menu number.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available options and exit.",
    )
    args, passthrough_args = parser.parse_known_args()

    if bool(args.list):
        _print_menu()
        return 0

    if args.choice is not None:
        selected = _resolve_choice(str(args.choice))
        if selected is None:
            print(f"Unknown --choice value: {args.choice!r}")
            _print_menu()
            return 2
    else:
        selected = _pick_interactive()
        if selected is None:
            print("No calibration selected.")
            return 0

    print(f"Running: {selected.label}")
    return _run_selected(selected, passthrough_args)


if __name__ == "__main__":
    raise SystemExit(main())