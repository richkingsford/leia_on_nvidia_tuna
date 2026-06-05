#!/usr/bin/env python3
"""Run a no-camera curve ramp with the right wheel held at floor PWM."""

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


LOCKED_TARGET = "l"
LOCKED_PWM_OVERRIDE = 0


def _wheel_actions_for_pulse(cmd: str) -> dict[str, str]:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "b":
        return {"l": "f", "r": "b"}
    return {"l": "b", "r": "f"}


def _actions_for_locked_target_curve(
    cmd: str,
    *,
    locked_target: str,
    locked_pwm: int,
    ramp_pwm: int,
) -> list[dict]:
    target_key = str(locked_target or "").strip().lower()
    if target_key not in {"l", "r"}:
        raise ValueError(f"locked_target must be 'l' or 'r', got {locked_target!r}")
    ramp_target = "r" if target_key == "l" else "l"
    drive_actions = _wheel_actions_for_pulse(cmd)
    by_target = {
        target_key: {
            "target": target_key,
            "action": drive_actions[target_key],
            "pwm": int(locked_pwm),
        },
        ramp_target: {
            "target": ramp_target,
            "action": drive_actions[ramp_target],
            "pwm": int(ramp_pwm),
        },
    }
    return [dict(by_target[target]) for target in ("l", "r")]


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
    floor_pwm = int(sequence[0].model_pwm)
    locked_pwm_value = int(LOCKED_PWM_OVERRIDE)
    locked_target = str(LOCKED_TARGET).strip().lower()
    ramp_target = "r" if locked_target == "l" else "l"

    robot = Robot()
    records = []
    start_s = time.monotonic()
    last_phase = None
    try:
        print(
            f"[SPEED] Physical curve run: camera disabled; target {locked_target.upper()} fixed at {locked_pwm_value} PWM, "
            f"target {ramp_target.upper()} ramps.",
            flush=True,
        )
        robot.stop()
        for idx, pulse in enumerate(sequence):
            if last_phase is not None and pulse.phase_name != last_phase:
                print("[SPEED] Interphase stop before backward curve ramp.", flush=True)
                robot.stop()
                time.sleep(0.20)
                start_s = time.monotonic() - (idx * pulse.interval_ms / 1000.0)
            last_phase = pulse.phase_name

            locked_pwm = int(locked_pwm_value)
            ramp_pwm = int(pulse.model_pwm)
            actions = _actions_for_locked_target_curve(
                pulse.cmd,
                locked_target=locked_target,
                locked_pwm=locked_pwm,
                ramp_pwm=ramp_pwm,
            )
            send_result = robot.send_custom_actions_pwm(
                pulse.cmd,
                actions,
                duration_ms=int(pulse.interval_ms),
            )
            left_pwm = int(locked_pwm if locked_target == "l" else ramp_pwm)
            right_pwm = int(locked_pwm if locked_target == "r" else ramp_pwm)
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
                        "locked_target": str(locked_target),
                        "ramp_target": str(ramp_target),
                        "locked_pwm_requested": int(locked_pwm),
                        "ramp_pwm_requested": int(ramp_pwm),
                        "left_pwm_requested": int(left_pwm),
                        "right_pwm_requested": int(right_pwm),
                        "operator_note": "Physical right appears to map to the locked target for this test.",
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
            run_label=f"physical_2p4_locked_{locked_target}_zero_{ramp_target}_ramp_no_camera",
            metadata={
                "requested_motion_s": 2.5,
                "actual_motion_s": 2.4,
                "interval_ms": 150,
                "phase_s": 1.2,
                "reverse_mode": "backward",
                "floor_pwm_scale": DEFAULT_FLOOR_PWM_SCALE,
                "ceiling_pwm_scale": DEFAULT_CEILING_PWM_SCALE,
                "locked_target": str(locked_target),
                "ramp_target": str(ramp_target),
                "locked_pwm": int(locked_pwm_value),
                "floor_pwm": int(floor_pwm),
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
