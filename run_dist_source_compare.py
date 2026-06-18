#!/usr/bin/env python3
"""Compare close-range distance sources at a known true distance.

For each frame, capture:
  - unmasked stack (current game source): width-based pinhole + W_px stability
  - masked contour (held brick removed): width-based pinhole on the contour box
  - height-based
Prints per-frame values and a stability summary (min/max/range) per source.
NO robot is constructed -> no motion.
"""
from __future__ import annotations

import argparse

import a_follow_the_brick as F
from helper_brick_detector_yolo import BrickDetector
from helper_holding_brick import (
    HoldingMaskLock,
    detect_holding_brick,
    detect_masked_target_brick_contour,
    mask_held_brick_for_target_frame,
)


def _rng(label, vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return f"{label}: (none)"
    return f"{label}: n={len(vals)} min={min(vals):.1f} max={max(vals):.1f} range={max(vals)-min(vals):.1f} mean={sum(vals)/len(vals):.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--true-mm", type=float, default=None)
    args = ap.parse_args(argv)

    F._set_game_profile("holding")
    vision = BrickDetector(debug=True)
    vision.set_runtime_tuning(**dict(F.CROWN_PROFILE_TUNING))
    F._warmup(vision)
    lock = HoldingMaskLock()

    u_w, c_w, u_wdist, u_hdist, c_dist, wpx, cwpx = [], [], [], [], [], [], []
    if args.true_mm is not None:
        print(f"[COMPARE] true distance = {args.true_mm:.0f}mm", flush=True)
    for i in range(int(args.frames)):
        result = vision.read()
        u_found = isinstance(result, tuple) and len(result) >= 8 and bool(result[0])
        u_used = float(result[2]) if u_found else None
        wd = getattr(vision, "last_bbox_width_dist", None)
        hd = getattr(vision, "last_bbox_height_dist", None)
        w_px = getattr(vision, "last_bbox_w_px", None)
        src = getattr(vision, "last_bbox_distance_source", None)

        raw_frame = getattr(vision, "raw_frame", None)
        hr = lock.update(detect_holding_brick(raw_frame))
        cdist = cw = None
        if bool(hr.get("holding")) and raw_frame is not None:
            masked = mask_held_brick_for_target_frame(raw_frame, hr)
            if masked is not None:
                cr = detect_masked_target_brick_contour(masked, detector=vision)
                if bool(cr.get("found")):
                    cdist = float(cr.get("dist_mm"))
                    bbox = cr.get("bbox") or [0, 0, 0, 0]
                    cw = float(bbox[2]) if len(bbox) >= 3 else None

        if u_used is not None: u_w.append(u_used)
        if wd is not None: u_wdist.append(float(wd))
        if hd is not None: u_hdist.append(float(hd))
        if w_px is not None: wpx.append(float(w_px))
        if cdist is not None: c_dist.append(cdist)
        if cw is not None: cwpx.append(cw)
        print(
            f"[F{i:02d}] unmasked_used={u_used if u_used is None else round(u_used,1)} src={src} "
            f"W_px={w_px} Wdist={None if wd is None else round(wd,1)} Hdist={None if hd is None else round(hd,1)} "
            f"| contour_dist={None if cdist is None else round(cdist,1)} contour_w_px={cw} holding={bool(hr.get('holding'))}",
            flush=True,
        )

    print("\n[SUMMARY]", flush=True)
    print("  " + _rng("unmasked stack (used)", u_w), flush=True)
    print("  " + _rng("unmasked W_px       ", wpx), flush=True)
    print("  " + _rng("unmasked width-dist ", u_wdist), flush=True)
    print("  " + _rng("unmasked height-dist", u_hdist), flush=True)
    print("  " + _rng("masked contour dist ", c_dist), flush=True)
    print("  " + _rng("masked contour W_px ", cwpx), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
