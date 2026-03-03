#!/usr/bin/env python3
"""
zdebug_floor_lift_estimator.py

Live diagnostics for lift estimation sources:
- ArUco cam_h path
- tiny ROI floor-lift path
- fused world lift_height + source/quality

Controls:
- q: quit
"""

import time

import cv2

from helper_vision_aruco import ArucoBrickVision
from telemetry_robot import WorldModel


def _fmt_num(value, decimals=2):
    try:
        return f"{float(value):.{int(decimals)}f}"
    except Exception:
        return "n/a"


def main():
    vision = ArucoBrickVision(debug=True)
    world = WorldModel()

    print("[ZDEBUG] Floor-lift estimator diagnostics started.")
    print("[ZDEBUG] Press 'q' in the video window to quit.")

    last_log_ts = 0.0
    while True:
        found, angle, dist, offset_x, conf, cam_h, brick_above, brick_below = vision.read()

        floor_lift_mm = getattr(vision, "floor_lift_mm", None)
        floor_lift_quality = getattr(vision, "floor_lift_quality", 0.0)
        floor_lift_ok = bool(getattr(vision, "floor_lift_ok", False))
        floor_lift_age = getattr(vision, "floor_lift_age_frames", None)

        world.update_vision(
            found,
            dist,
            angle,
            conf,
            offset_x,
            cam_h,
            brick_above,
            brick_below,
            floor_lift_mm=floor_lift_mm,
            floor_lift_quality=floor_lift_quality,
        )

        frame = getattr(vision, "current_frame", None)
        if frame is not None:
            hud_lines = [
                f"found={bool(found)} conf={_fmt_num(conf, 1)} cam_h={_fmt_num(cam_h, 1)}",
                f"floor_lift_mm={_fmt_num(floor_lift_mm, 1)} q={_fmt_num(floor_lift_quality, 2)} ok={floor_lift_ok} age={floor_lift_age}",
                (
                    "lift_height="
                    f"{_fmt_num(getattr(world, 'lift_height', None), 1)} "
                    f"src={str(getattr(world, 'lift_height_source', 'n/a'))} "
                    f"q={_fmt_num(getattr(world, 'lift_height_quality', None), 2)}"
                ),
                f"lift_anchor={_fmt_num(getattr(world, 'lift_height_anchor', None), 1)}",
            ]

            y = 26
            for line in hud_lines:
                cv2.putText(
                    frame,
                    str(line),
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                y += 22

            cv2.imshow("zdebug_floor_lift_estimator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        now = time.time()
        if (now - last_log_ts) >= 0.5:
            last_log_ts = now
            print(
                "[LIFT] "
                f"found={bool(found)} "
                f"conf={_fmt_num(conf, 1)} "
                f"cam_h={_fmt_num(cam_h, 1)} "
                f"floor={_fmt_num(floor_lift_mm, 1)} "
                f"floor_q={_fmt_num(floor_lift_quality, 2)} "
                f"floor_ok={floor_lift_ok} "
                f"floor_age={floor_lift_age} "
                f"lift={_fmt_num(getattr(world, 'lift_height', None), 1)} "
                f"src={str(getattr(world, 'lift_height_source', 'n/a'))} "
                f"lift_q={_fmt_num(getattr(world, 'lift_height_quality', None), 2)}"
            )

    cv2.destroyAllWindows()
    vision.close()


if __name__ == "__main__":
    main()
