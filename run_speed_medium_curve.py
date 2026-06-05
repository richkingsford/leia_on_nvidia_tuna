#!/usr/bin/env python3
"""Run labeled medium duty-cycle curve diagnostics with the camera disabled."""

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


CURVES = {
    "fr_medium": {"cmd": "f", "turn": "r", "skip_target": "r", "summary": "forward while arcing right"},
    "fl_medium": {"cmd": "f", "turn": "l", "skip_target": "l", "summary": "forward while arcing left"},
    "br_medium": {"cmd": "b", "turn": "r", "skip_target": "r", "summary": "backward while arcing right"},
    "bl_medium": {"cmd": "b", "turn": "l", "skip_target": "l", "summary": "backward while arcing left"},
}
DEFAULT_ORDER = ("fr_medium", "bl_medium", "fl_medium", "br_medium")
SKIP_EVERY_N = 2
SKIP_PHASE_OFFSET = 1
SKIP_PWM = 0
PHASE_DURATION_S = 1.20
INTERVAL_MS = 150
LOG_DIR = Path("logs/speed_tests")


def _wheel_actions_for_cmd(cmd: str) -> dict[str, str]:
    cmd_key = str(cmd or "").strip().lower()
    if cmd_key == "b":
        return {"l": "f", "r": "b"}
    return {"l": "b", "r": "f"}


def _actions_for_curve(cmd: str, *, pwm: int, step_zero_based: int, skip_target: str) -> tuple[list[dict], bool, int, int]:
    target_key = str(skip_target or "").strip().lower()
    if target_key not in {"l", "r"}:
        raise ValueError(f"skip_target must be 'l' or 'r', got {skip_target!r}")
    should_skip = (int(step_zero_based) - int(SKIP_PHASE_OFFSET)) % int(SKIP_EVERY_N) == 0
    drive_actions = _wheel_actions_for_cmd(cmd)
    left_pwm = int(SKIP_PWM if should_skip and target_key == "l" else pwm)
    right_pwm = int(SKIP_PWM if should_skip and target_key == "r" else pwm)
    return (
        [
            {"target": "l", "action": drive_actions["l"], "pwm": int(left_pwm)},
            {"target": "r", "action": drive_actions["r"], "pwm": int(right_pwm)},
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


def run_curve(label: str, *, settle_s: float) -> dict:
    curve = dict(CURVES[label])
    sequence = list(_single_phase_sequence(curve["cmd"]))
    if not sequence:
        raise RuntimeError(f"No pulses built for {label}.")

    robot = Robot()
    records = []
    start_s = time.monotonic()
    try:
        print(
            f"[CURVE] {label}: camera disabled; {curve['summary']}; "
            f"skip target {curve['skip_target'].upper()} gets 0 PWM every other {INTERVAL_MS}ms pulse.",
            flush=True,
        )
        robot.stop()
        time.sleep(float(settle_s))
        start_s = time.monotonic()

        for idx, pulse in enumerate(sequence):
            actions, skipped, left_pwm, right_pwm = _actions_for_curve(
                pulse.cmd,
                pwm=int(pulse.model_pwm),
                step_zero_based=int(pulse.phase_step) - 1,
                skip_target=str(curve["skip_target"]),
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
                    "model_pwm": pulse.model_pwm,
                    "interval_ms": pulse.interval_ms,
                    "send_result": send_result,
                    "curve_experiment": {
                        "curve_label": str(label),
                        "direction": "forward" if curve["cmd"] == "f" else "backward",
                        "turn": "right" if curve["turn"] == "r" else "left",
                        "strength": "medium",
                        "skip_target": str(curve["skip_target"]),
                        "skip_every_n": int(SKIP_EVERY_N),
                        "skip_pwm": int(SKIP_PWM),
                        "skipped": bool(skipped),
                        "left_pwm_requested": int(left_pwm),
                        "right_pwm_requested": int(right_pwm),
                        "operator_note": "Medium curve uses the same smooth ramp on both treads, with the inside tread zeroed on alternating pulses.",
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
                "direction": "forward" if curve["cmd"] == "f" else "backward",
                "turn": "right" if curve["turn"] == "r" else "left",
                "strength": "medium",
                "summary": str(curve["summary"]),
                "requested_motion_s": PHASE_DURATION_S,
                "actual_motion_s": PHASE_DURATION_S,
                "interval_ms": INTERVAL_MS,
                "phase_s": PHASE_DURATION_S,
                "floor_pwm_scale": DEFAULT_FLOOR_PWM_SCALE,
                "ceiling_pwm_scale": DEFAULT_CEILING_PWM_SCALE,
                "skip_target": str(curve["skip_target"]),
                "skip_every_n": int(SKIP_EVERY_N),
                "skip_pwm": int(SKIP_PWM),
                "ramp_pwm_min": int(sequence[0].model_pwm),
                "ramp_pwm_max": int(sequence[-1].model_pwm),
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
    parser.add_argument("labels", nargs="*", help="Curve labels to execute.")
    parser.add_argument("--all-medium", action="store_true", help="Run fr/bl/fl/br medium curves in a return-ish order.")
    parser.add_argument("--settle-s", type=float, default=0.35, help="Stop/settle delay before each curve.")
    args = parser.parse_args()

    labels = list(args.labels)
    if args.all_medium:
        labels = list(DEFAULT_ORDER)
    if not labels:
        raise SystemExit("Choose at least one curve label or pass --all-medium.")
    unknown = [label for label in labels if label not in CURVES]
    if unknown:
        raise SystemExit(
            "Unknown curve label(s): "
            + ", ".join(unknown)
            + ". Valid labels: "
            + ", ".join(sorted(CURVES))
        )

    completed = []
    for idx, label in enumerate(labels):
        if idx > 0:
            time.sleep(0.70)
        completed.append(run_curve(label, settle_s=float(args.settle_s)))
    print("[CURVE] Completed labels: " + ", ".join(item["label"] for item in completed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
