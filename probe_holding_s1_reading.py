#!/usr/bin/env python3
"""No-motion holding-profile vision probe.

Reuses a_follow_the_brick._read_brick_measurement so the numbers match the live
game exactly. Constructs the vision detector, forces the holding profile, and
prints per-frame dist/x/source/calibration for N reads. Never creates a robot
object, so no motion is possible.
"""
from __future__ import annotations

import argparse
import time

import a_follow_the_brick as F
from helper_brick_detector_yolo import BrickDetector


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--interval-s", type=float, default=0.15)
    args = ap.parse_args(argv)

    F._set_game_profile("holding")
    cal = F._holding_target_distance_calibration_config()
    print(f"[CAL] enabled={cal.get('enabled')} points={cal.get('points')}", flush=True)
    print(f"[VIS] holding_target_vision={F._holding_target_vision_config()}", flush=True)

    vision = BrickDetector(debug=True)
    vision.set_runtime_tuning(**dict(F.CROWN_PROFILE_TUNING))
    F._warmup(vision)

    keys = (
        "holding", "target_masked_for_holding", "holding_xz_source",
        "uncalibrated_dist_mm", "dist_mm", "x_mm", "confident",
        "confidence_pct", "holding_distance_calibrated", "reason",
    )
    print("[PROBE] frame | " + " ".join(keys), flush=True)
    for i in range(int(args.frames)):
        r = F._read_brick_measurement(vision)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
