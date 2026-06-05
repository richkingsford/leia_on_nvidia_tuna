#!/usr/bin/env python3
"""Run a short physical forward/back speed ramp with brick-distance samples."""

from __future__ import annotations

import time

from helper_robot_control import Robot
from helper_speed_test import (
    DEFAULT_CEILING_PWM_SCALE,
    DEFAULT_FLOOR_PWM_SCALE,
    build_speed_test_sequence,
    format_speed_test_table,
    write_speed_test_artifacts,
)


def main() -> int:
    sequence = build_speed_test_sequence(
        phase_duration_s=1.20,
        interval_ms=150,
        reverse_mode="backward",
        floor_pwm_scale=DEFAULT_FLOOR_PWM_SCALE,
        ceiling_pwm_scale=DEFAULT_CEILING_PWM_SCALE,
    )
    robot = Robot()
    records = []
    start_s = time.monotonic()
    last_phase = None
    try:
        print("[SPEED] Physical run: camera disabled; forward ramp, stop, backward ramp, then final stop.", flush=True)
        robot.stop()
        for idx, pulse in enumerate(sequence):
            if last_phase is not None and pulse.phase_name != last_phase:
                print("[SPEED] Interphase stop before backward ramp.", flush=True)
                robot.stop()
                time.sleep(0.20)
                start_s = time.monotonic() - (idx * pulse.interval_ms / 1000.0)
            last_phase = pulse.phase_name

            send_result = robot.send_command_pwm(
                pulse.cmd,
                int(pulse.model_pwm),
                duration_ms=int(pulse.interval_ms),
            )
            records.append(
                {
                    "phase_name": pulse.phase_name,
                    "cmd": pulse.cmd,
                    "t_ms": int(idx) * int(pulse.interval_ms),
                    "phase_t_ms": pulse.t_ms,
                    "phase_step": pulse.phase_step,
                    "phase_steps": pulse.phase_steps,
                    "score": pulse.score,
                    "model_power": pulse.model_power,
                    "model_pwm": pulse.model_pwm,
                    "interval_ms": pulse.interval_ms,
                    "send_result": send_result,
                    "vision": {
                        "found": False,
                        "dist_mm": None,
                        "x_mm": None,
                        "conf": None,
                        "status": "camera disabled",
                        "source": None,
                    },
                }
            )
            next_start_s = float(start_s) + (float(idx + 1) * float(pulse.interval_ms) / 1000.0)
            remaining_s = float(next_start_s) - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)
        robot.stop()
        result = {"ok": True, "execute": True, "pulses": records}
        print("[SPEED] Final stop sent. Actual sent table:", flush=True)
        print(format_speed_test_table(sequence, records=records), flush=True)
        artifacts = write_speed_test_artifacts(
            result,
            log_dir="logs/speed_tests",
            run_label="physical_2p4_forward_stop_backward_no_camera",
            metadata={
                "requested_motion_s": 2.5,
                "actual_motion_s": 2.4,
                "interval_ms": 150,
                "phase_s": 1.2,
                "reverse_mode": "backward",
                "floor_pwm_scale": DEFAULT_FLOOR_PWM_SCALE,
                "ceiling_pwm_scale": DEFAULT_CEILING_PWM_SCALE,
                "interphase_stop_s": 0.2,
                "camera_disabled": True,
                "sample_dist": False,
            },
        )
        print(f"[SPEED] JSON log saved: {artifacts['json_path']}", flush=True)
        print(f"[SPEED] PWM chart saved: {artifacts['html_path']}", flush=True)
        return 0
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
