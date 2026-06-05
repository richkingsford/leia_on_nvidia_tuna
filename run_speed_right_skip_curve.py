#!/usr/bin/env python3
"""Run a no-camera gentle-curve trial by duty-cycling the right tread."""

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


SKIP_TARGET = "r"
SKIP_EVERY_N = 2
SKIP_PHASE_OFFSET = 1
SKIP_PWM = 0


def _wheel_actions_for_pulse(cmd: str) -> dict[str, str]:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "b":
        return {"l": "f", "r": "b"}
    return {"l": "b", "r": "f"}


def _actions_for_right_skip_curve(
    cmd: str,
    *,
    pwm: int,
    phase_step_zero_based: int,
) -> tuple[list[dict], bool, int, int]:
    drive_actions = _wheel_actions_for_pulse(cmd)
    should_skip = (int(phase_step_zero_based) - int(SKIP_PHASE_OFFSET)) % int(SKIP_EVERY_N) == 0
    left_pwm = int(pwm)
    right_pwm = int(SKIP_PWM if should_skip else pwm)
    actions = [
        {"target": "l", "action": drive_actions["l"], "pwm": int(left_pwm)},
        {"target": "r", "action": drive_actions["r"], "pwm": int(right_pwm)},
    ]
    return actions, bool(should_skip), int(left_pwm), int(right_pwm)


def main() -> int:
    sequence = build_speed_test_sequence(
        phase_duration_s=1.20,
        interval_ms=150,
        reverse_mode="backward",
        floor_pwm_scale=DEFAULT_FLOOR_PWM_SCALE,
        ceiling_pwm_scale=DEFAULT_CEILING_PWM_SCALE,
    )
    if not sequence:
        raise SystemExit("No pulses built.")

    robot = Robot()
    records = []
    start_s = time.monotonic()
    last_phase = None
    try:
        print(
            "[SPEED] Physical skip-curve run: camera disabled; both treads use the same PWM ramp, "
            "but right tread receives 0 PWM on every other 150ms pulse.",
            flush=True,
        )
        robot.stop()
        for idx, pulse in enumerate(sequence):
            if last_phase is not None and pulse.phase_name != last_phase:
                print("[SPEED] Interphase stop before backward skip curve ramp.", flush=True)
                robot.stop()
                time.sleep(0.20)
                start_s = time.monotonic() - (idx * pulse.interval_ms / 1000.0)
            last_phase = pulse.phase_name

            actions, skipped_right, left_pwm, right_pwm = _actions_for_right_skip_curve(
                pulse.cmd,
                pwm=int(pulse.model_pwm),
                phase_step_zero_based=int(pulse.phase_step) - 1,
            )
            send_result = robot.send_custom_actions_pwm(
                pulse.cmd,
                actions,
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
                    "curve_experiment": {
                        "skip_target": str(SKIP_TARGET),
                        "skip_every_n": int(SKIP_EVERY_N),
                        "skip_pwm": int(SKIP_PWM),
                        "skipped_right": bool(skipped_right),
                        "left_pwm_requested": int(left_pwm),
                        "right_pwm_requested": int(right_pwm),
                        "operator_note": "Both wheels follow the same ramp except the right tread is zeroed on alternating pulses.",
                    },
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
            run_label="physical_2p4_right_every_other_zero_curve_no_camera",
            metadata={
                "requested_motion_s": 2.5,
                "actual_motion_s": 2.4,
                "interval_ms": 150,
                "phase_s": 1.2,
                "reverse_mode": "backward",
                "floor_pwm_scale": DEFAULT_FLOOR_PWM_SCALE,
                "ceiling_pwm_scale": DEFAULT_CEILING_PWM_SCALE,
                "skip_target": str(SKIP_TARGET),
                "skip_every_n": int(SKIP_EVERY_N),
                "skip_pwm": int(SKIP_PWM),
                "ramp_pwm_min": int(sequence[0].model_pwm),
                "ramp_pwm_max": int(sequence[-1].model_pwm),
                "compound_custom_actions": True,
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
