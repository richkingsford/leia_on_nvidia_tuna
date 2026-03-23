#!/usr/bin/env python3
"""
Fast cyan vision consistency probe.

Reads frames from the cyan brick detector and prints the three live brick
properties every frame:
  - x_axis
  - dist
  - y_axis

This script never imports or sends robot commands.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

# Allow running directly from debug/ while importing project helpers.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper_brick_detector_yolo import BrickDetector


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "nan"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(num) or math.isinf(num):
        return "nan"
    return f"{num:+.2f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print cyan model x_axis/dist/y_axis every frame as fast as possible."
        )
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional ONNX model path (defaults to brick_yolo_v4.onnx).",
    )
    parser.add_argument(
        "--throttle-ms",
        type=float,
        default=0.0,
        help="Optional sleep after each frame (default: 0 = max speed).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    stop = False

    def _handle_stop(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    detector = BrickDetector(debug=False, model_path=args.model_path)
    throttle_s = max(0.0, float(args.throttle_ms) / 1000.0)

    start_t = time.time()
    frame_idx = 0

    print("# cyan vision consistency stream (Ctrl+C to stop)")
    print("# columns: frame t_s found x_axis_mm dist_mm y_axis_mm")

    try:
        while not stop:
            frame_idx += 1
            now = time.time()
            rel_t = now - start_t

            found, _angle, dist, offset_x, _conf, cam_h, _above, _below = detector.read()

            # Runtime convention maps detector offset_x->x_axis and cam_h->y_axis.
            x_axis = float(offset_x) if found else None
            y_axis = float(cam_h) if found else None
            dist_mm = float(dist) if found else None

            print(
                f"{frame_idx} {rel_t:.3f} {1 if found else 0} "
                f"{_fmt_num(x_axis)} {_fmt_num(dist_mm)} {_fmt_num(y_axis)}"
            )

            if throttle_s > 0.0:
                time.sleep(throttle_s)
    finally:
        detector.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
