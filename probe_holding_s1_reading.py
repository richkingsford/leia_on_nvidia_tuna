#!/usr/bin/env python3
"""No-motion holding-profile vision probe.

Reuses a_follow_the_brick._read_brick_measurement so the numbers match the live
game exactly. Constructs the vision detector, forces the holding profile, and
prints per-frame dist/x/source for N reads. Never creates a robot
object, so no motion is possible.
"""
from __future__ import annotations

import argparse
import statistics
import time

import a_follow_the_brick as F
from helper_brick_detector_native_oak import BrickDetector


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--interval-s", type=float, default=0.15)
    ap.add_argument("--profile", choices=("empty", "holding"), default="empty")
    args = ap.parse_args(argv)

    F._set_game_profile(args.profile)
    print(f"[VIS] profile={args.profile}", flush=True)

    vision = BrickDetector(debug=True)
    vision.set_runtime_tuning(**dict(F.CROWN_PROFILE_TUNING))
    F._warmup(vision)

    keys = (
        "holding", "target_masked_for_holding", "holding_xz_source",
        "dist_mm", "x_mm", "confident",
        "y_mm",
        "confidence_pct", "reason",
    )
    print("[PROBE] frame | " + " ".join(keys), flush=True)
    readings = []
    for i in range(int(args.frames)):
        r = F._read_brick_measurement(vision)
        readings.append(r)
        vals = []
        for k in keys:
            v = r.get(k)
            if isinstance(v, float):
                v = f"{v:.1f}"
            vals.append(str(v))
        print(f"[PROBE] {i:02d} | " + " ".join(vals), flush=True)
        v = vision
        print(
            "[SRC] "
            f"geom={getattr(v,'last_geometry_source',None)} "
            f"bbox_src={getattr(v,'last_bbox_distance_source',None)} "
            f"bbox_dist={getattr(v,'last_bbox_dist',None)} "
            f"W_dist={getattr(v,'last_bbox_width_dist',None)} "
            f"H_dist={getattr(v,'last_bbox_height_dist',None)} "
            f"depth={getattr(v,'last_depth_dist',None)} "
            f"raw={getattr(v,'last_raw_dist',None)} "
            f"W_px={getattr(v,'last_bbox_w_px',None)} "
            f"disp='{getattr(v,'last_distance_display_text',None)}'",
            flush=True,
        )
        time.sleep(float(args.interval_s))
    numeric = {}
    for key in ("dist_mm", "x_mm", "y_mm"):
        values = []
        for reading in readings:
            try:
                values.append(float(reading.get(key)))
            except (TypeError, ValueError):
                pass
        if values:
            numeric[key] = statistics.median(values)
    if numeric:
        print(
            "[MAJORITY] median of "
            f"{len(readings)} frames: "
            f"dist={numeric.get('dist_mm', float('nan')):.1f}mm "
            f"x={numeric.get('x_mm', float('nan')):+.1f}mm "
            f"y={numeric.get('y_mm', float('nan')):+.1f}mm",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
