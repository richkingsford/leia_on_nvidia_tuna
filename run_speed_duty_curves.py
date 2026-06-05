#!/usr/bin/env python3
"""Run labeled duty-cycle curve diagnostics with the camera disabled."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from helper_robot_control import Robot
from helper_speed_test import (
    DEFAULT_CEILING_PWM_SCALE,
    DEFAULT_FLOOR_PWM_SCALE,
    build_speed_test_sequence,
    format_speed_test_table,
    write_speed_test_artifacts,
)


DIRECTIONS = {
    "fr": {"cmd": "f", "turn": "r", "skip_target": "r", "summary": "forward while arcing right"},
    "fl": {"cmd": "f", "turn": "l", "skip_target": "l", "summary": "forward while arcing left"},
    "br": {"cmd": "b", "turn": "r", "skip_target": "r", "summary": "backward while arcing right"},
    "bl": {"cmd": "b", "turn": "l", "skip_target": "l", "summary": "backward while arcing left"},
}
STRENGTHS = {
    "gentle": {"mode": "skip_on", "every_n": 4, "phase_offset": 3, "skip_pwm": 0, "pwm_scale": 1.0},
    "medium": {"mode": "skip_on", "every_n": 2, "phase_offset": 1, "skip_pwm": 0, "pwm_scale": 1.0},
    "strong": {"mode": "run_on", "every_n": 3, "phase_offset": 2, "skip_pwm": 0, "pwm_scale": 1.0},
    "superstrong": {"mode": "hold_zero", "every_n": 1, "phase_offset": 0, "skip_pwm": 0, "pwm_scale": 1.0},
}
DEFAULT_DIRECTION_ORDER = ("fr", "bl", "fl", "br")
PHASE_DURATION_S = 1.20
INTERVAL_MS = 150
LOG_DIR = Path("logs/speed_tests")


def _label_for(direction_key: str, strength: str) -> str:
    return f"{direction_key}_{strength}"


def _wheel_actions_for_cmd(cmd: str) -> dict[str, str]:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "b":
        return {"l": "f", "r": "b"}
    return {"l": "b", "r": "f"}


def _actions_for_curve(
    cmd: str,
    *,
    pwm: int,
    step_zero_based: int,
    skip_target: str,
    every_n: int,
    phase_offset: int,
    skip_pwm: int,
    mode: str,
) -> tuple[list[dict], bool, int, int]:
    target_key = str(skip_target or "").strip().lower()
    if target_key not in {"l", "r"}:
        raise ValueError(f"skip_target must be 'l' or 'r', got {skip_target!r}")
    every_n = max(1, int(every_n))
    on_selected_tick = (int(step_zero_based) - int(phase_offset)) % every_n == 0
    mode_key = str(mode or "skip_on").strip().lower()
    if mode_key == "hold_zero":
        should_skip = True
    elif mode_key == "run_on":
        should_skip = not bool(on_selected_tick)
    else:
        should_skip = bool(on_selected_tick)
    drive_actions = _wheel_actions_for_cmd(cmd)
    left_pwm = int(skip_pwm if should_skip and target_key == "l" else pwm)
    right_pwm = int(skip_pwm if should_skip and target_key == "r" else pwm)
    left_action = drive_actions["l"]
    right_action = drive_actions["r"]
    return (
        [
            {"target": "l", "action": left_action, "pwm": int(left_pwm)},
            {"target": "r", "action": right_action, "pwm": int(right_pwm)},
        ],
        bool(should_skip),
        int(left_pwm),
        int(right_pwm),
    )


def _single_phase_sequence(cmd: str):
    sequence = build_speed_test_sequence(
        phase_duration_s=PHASE_DURATION_S,
        interval_ms=INTERVAL_MS,
        reverse_mode="backward",
        floor_pwm_scale=DEFAULT_FLOOR_PWM_SCALE,
        ceiling_pwm_scale=DEFAULT_CEILING_PWM_SCALE,
    )
    want_cmd = str(cmd or "").strip().lower()
    for pulse in sequence:
        if str(pulse.cmd).strip().lower() == want_cmd:
            yield pulse


def _write_stable_aliases(artifacts: dict, label: str) -> dict:
    aliases = {}
    for key, suffix in (("html_path", ".html"), ("json_path", ".json")):
        src = Path(str(artifacts[key]))
        dst = Path(src.parent) / f"{label}{suffix}"
        shutil.copyfile(src, dst)
        aliases[key.replace("_path", "_alias")] = str(dst)
    return aliases


def run_curve(direction_key: str, strength: str, *, settle_s: float) -> dict:
    direction = dict(DIRECTIONS[direction_key])
    strength_cfg = dict(STRENGTHS[strength])
    label = _label_for(direction_key, strength)
    sequence = list(_single_phase_sequence(direction["cmd"]))
    if not sequence:
        raise RuntimeError(f"No pulses built for {label}.")

    robot = Robot()
    records = []
    start_s = time.monotonic()
    try:
        print(
            f"[CURVE] {label}: camera disabled; {direction['summary']}; "
            f"inside target {direction['skip_target'].upper()} pattern {strength_cfg['mode']} "
            f"{'for the whole run' if str(strength_cfg['mode']) == 'hold_zero' else 'every ' + str(strength_cfg['every_n']) + ' x ' + str(INTERVAL_MS) + 'ms pulse(s)'}.",
            flush=True,
        )
        robot.stop()
        time.sleep(float(settle_s))
        start_s = time.monotonic()

        for idx, pulse in enumerate(sequence):
            drive_pwm = max(0, min(255, int(round(float(pulse.model_pwm) * float(strength_cfg.get("pwm_scale", 1.0))))))
            actions, skipped, left_pwm, right_pwm = _actions_for_curve(
                pulse.cmd,
                pwm=int(drive_pwm),
                step_zero_based=int(pulse.phase_step) - 1,
                skip_target=str(direction["skip_target"]),
                every_n=int(strength_cfg["every_n"]),
                phase_offset=int(strength_cfg["phase_offset"]),
                skip_pwm=int(strength_cfg["skip_pwm"]),
                mode=str(strength_cfg["mode"]),
            )
            send_result = robot.send_custom_actions_pwm(
                pulse.cmd,
                actions,
                duration_ms=int(pulse.interval_ms),
            )
            records.append(
                {
                    "phase_name": str(label),
                    "cmd": pulse.cmd,
                    "t_ms": int(idx) * int(pulse.interval_ms),
                    "phase_t_ms": int(idx) * int(pulse.interval_ms),
                    "phase_step": int(idx + 1),
                    "phase_steps": len(sequence),
                    "score": pulse.score,
                    "model_power": pulse.model_power,
                    "model_pwm": int(drive_pwm),
                    "base_model_pwm": pulse.model_pwm,
                    "interval_ms": pulse.interval_ms,
                    "send_result": send_result,
                    "curve_experiment": {
                        "curve_label": str(label),
                        "direction": "forward" if direction["cmd"] == "f" else "backward",
                        "turn": "right" if direction["turn"] == "r" else "left",
                        "strength": str(strength),
                        "skip_target": str(direction["skip_target"]),
                        "duty_mode": str(strength_cfg["mode"]),
                        "every_n": int(strength_cfg["every_n"]),
                        "phase_offset": int(strength_cfg["phase_offset"]),
                        "skip_pwm": int(strength_cfg["skip_pwm"]),
                        "pwm_scale": float(strength_cfg.get("pwm_scale", 1.0)),
                        "skipped": bool(skipped),
                        "left_pwm_requested": int(left_pwm),
                        "right_pwm_requested": int(right_pwm),
                        "left_action_requested": str(actions[0].get("action")),
                        "right_action_requested": str(actions[1].get("action")),
                        "operator_note": "Duty curve uses the same smooth ramp on both treads, with the inside tread zeroed on selected pulses.",
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
        print(f"[CURVE] {label}: final stop sent. Actual sent table:", flush=True)
        print(format_speed_test_table(sequence, records=records), flush=True)
        artifacts = write_speed_test_artifacts(
            result,
            log_dir=LOG_DIR,
            run_label=str(label),
            metadata={
                "curve_label": str(label),
                "direction": "forward" if direction["cmd"] == "f" else "backward",
                "turn": "right" if direction["turn"] == "r" else "left",
                "strength": str(strength),
                "summary": str(direction["summary"]),
                "requested_motion_s": PHASE_DURATION_S,
                "actual_motion_s": PHASE_DURATION_S,
                "interval_ms": INTERVAL_MS,
                "phase_s": PHASE_DURATION_S,
                "floor_pwm_scale": DEFAULT_FLOOR_PWM_SCALE,
                "ceiling_pwm_scale": DEFAULT_CEILING_PWM_SCALE,
                "skip_target": str(direction["skip_target"]),
                "duty_mode": str(strength_cfg["mode"]),
                "every_n": int(strength_cfg["every_n"]),
                "phase_offset": int(strength_cfg["phase_offset"]),
                "skip_every_n": int(strength_cfg["every_n"]),
                "skip_phase_offset": int(strength_cfg["phase_offset"]),
                "skip_pwm": int(strength_cfg["skip_pwm"]),
                "pwm_scale": float(strength_cfg.get("pwm_scale", 1.0)),
                "ramp_pwm_min": int(sequence[0].model_pwm),
                "ramp_pwm_max": int(sequence[-1].model_pwm),
                "scaled_ramp_pwm_min": max(0, min(255, int(round(float(sequence[0].model_pwm) * float(strength_cfg.get("pwm_scale", 1.0)))))),
                "scaled_ramp_pwm_max": max(0, min(255, int(round(float(sequence[-1].model_pwm) * float(strength_cfg.get("pwm_scale", 1.0)))))),
                "compound_custom_actions": True,
                "camera_disabled": True,
                "sample_dist": False,
            },
        )
        aliases = _write_stable_aliases(artifacts, label)
        print(f"[CURVE] {label}: JSON log saved: {artifacts['json_path']}", flush=True)
        print(f"[CURVE] {label}: PWM chart saved: {artifacts['html_path']}", flush=True)
        print(f"[CURVE] {label}: stable chart alias: {aliases['html_alias']}", flush=True)
        return {"label": label, "artifacts": artifacts, "aliases": aliases}
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="*", help="Curve labels such as fr_gentle or bl_medium.")
    parser.add_argument("--strength", choices=tuple(STRENGTHS), help="Run all direction permutations for one strength.")
    parser.add_argument("--settle-s", type=float, default=0.35, help="Stop/settle delay before each curve.")
    args = parser.parse_args()

    labels = list(args.labels)
    if args.strength:
        labels.extend(_label_for(direction_key, args.strength) for direction_key in DEFAULT_DIRECTION_ORDER)
    if not labels:
        raise SystemExit("Choose curve labels or pass --strength gentle|medium|strong|superstrong.")

    planned: list[tuple[str, str]] = []
    valid = {_label_for(direction_key, strength) for direction_key in DIRECTIONS for strength in STRENGTHS}
    unknown = [label for label in labels if label not in valid]
    if unknown:
        raise SystemExit(
            "Unknown curve label(s): "
            + ", ".join(unknown)
            + ". Valid labels: "
            + ", ".join(sorted(valid))
        )
    for label in labels:
        direction_key, strength = label.split("_", 1)
        planned.append((direction_key, strength))

    completed = []
    for idx, (direction_key, strength) in enumerate(planned):
        if idx > 0:
            time.sleep(0.70)
        completed.append(run_curve(direction_key, strength, settle_s=float(args.settle_s)))
    print("[CURVE] Completed labels: " + ", ".join(item["label"] for item in completed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
