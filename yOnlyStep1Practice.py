#!/usr/bin/env python3
"""Practice Step 1 y alignment with mast-only full-power pulses."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from alignmentTest import DEFAULT_STREAM_URL, CameraTelemetryReader, read_stream_sample
from a_follow_the_brick import (
    _adaptive_y_mast_duration_ms,
    _dist_target_mm,
    _follow_y_axis_config,
    _min_y_curve_closes_mm,
    _win_effective_tolerance,
    _x_target_mm,
)
from helper_robot_control import Robot


PWM = 255
DEFAULT_DURATION_S = 30 * 60
DEFAULT_OUT = Path("runs/y_only_step1_practice.jsonl")
MIN_CONFIDENCE_PCT = 75.0
MAX_PULSE_MS = 900
MAX_RESET_PULSE_MS = 4000
MIN_PULSE_MS = 500
OBSERVE_SETTLE_S = 0.55
MISSING_RECOVERY_DOWN_MS = 350
DEFAULT_PERTURB_MS = 200
DEFAULT_HAPPY_STREAK_BEFORE_PERTURB = 3
MAX_ABS_X_FOR_Y_ONLY_MM = 25.0
MAX_X_JUMP_FOR_Y_ONLY_MM = 25.0


def _sample_to_dict(sample) -> dict:
    return {
        "found": bool(sample.found),
        "dist_mm": float(sample.dist_mm),
        "x_mm": float(sample.x_mm),
        "y_mm": float(sample.y_mm),
        "confidence_pct": float(sample.confidence_pct),
        "source": str(sample.source),
    }


def _read_sample(reader: CameraTelemetryReader) -> dict:
    return _sample_to_dict(reader.read())


class StreamTelemetryReader:
    def __init__(self, stream_url: str = DEFAULT_STREAM_URL) -> None:
        self._stream_url = str(stream_url or DEFAULT_STREAM_URL)

    def close(self) -> None:
        return None

    def read(self):
        return read_stream_sample(self._stream_url)


def _read_stable_sample(reader: CameraTelemetryReader, *, frames: int = 5, sample_s: float = 0.05) -> dict:
    samples: list[dict] = []
    for idx in range(max(1, int(frames))):
        samples.append(_read_sample(reader))
        if idx < max(1, int(frames)) - 1:
            time.sleep(max(0.0, float(sample_s)))
    visible = [sample for sample in samples if _visible(sample)]
    if not visible:
        return samples[-1]
    return {
        "found": True,
        "dist_mm": float(statistics.median(float(sample["dist_mm"]) for sample in visible)),
        "x_mm": float(statistics.median(float(sample["x_mm"]) for sample in visible)),
        "y_mm": float(statistics.median(float(sample["y_mm"]) for sample in visible)),
        "confidence_pct": float(statistics.median(float(sample["confidence_pct"]) for sample in visible)),
        "source": str(visible[-1].get("source", "camera")),
        "stable_visible_frames": len(visible),
        "stable_total_frames": len(samples),
    }


def _stop_robot() -> None:
    robot = Robot(exit_on_failure=False)
    try:
        robot.stop()
    finally:
        robot.close()


def _send_mast(cmd: str, duration_ms: int, *, max_duration_ms: int = MAX_PULSE_MS) -> dict:
    duration_ms = max(1, min(int(duration_ms), int(max_duration_ms)))
    robot = Robot(exit_on_failure=False)
    try:
        robot.stop()
        result = robot.send_command_pwm(str(cmd), PWM, duration_ms=int(duration_ms))
        time.sleep(float(duration_ms) / 1000.0)
        robot.stop()
        return dict(result or {})
    finally:
        robot.close()


def _opposite_mast_cmd(cmd: str | None) -> str:
    return "u" if str(cmd or "").strip().lower() == "d" else "d"


def _visible(sample: dict) -> bool:
    return bool(sample.get("found")) and float(sample.get("confidence_pct") or 0.0) >= MIN_CONFIDENCE_PCT


def _plan_duration_ms(cmd: str, gap_outside_tol_mm: float, y_cfg: dict, *, last_regressed: bool = False) -> int:
    gap = max(0.0, float(gap_outside_tol_mm))
    ms = _adaptive_y_mast_duration_ms(
        gap,
        y_cfg,
        near_end=True,
        cmd=str(cmd).strip().lower(),
        duration_err=gap,
    )
    if last_regressed and gap <= 5.0:
        ms = int(round(float(ms) * 0.7))
    return max(MIN_PULSE_MS, min(MAX_PULSE_MS, int(ms)))


def _fmt_sample(sample: dict | None) -> str:
    if not isinstance(sample, dict) or not _visible(sample):
        conf = "N/A" if not isinstance(sample, dict) else f"{float(sample.get('confidence_pct') or 0.0):.0f}%"
        return f"missing/conf={conf}"
    return (
        f"dist={float(sample['dist_mm']):.1f} x={float(sample['x_mm']):+.1f} "
        f"y={float(sample['y_mm']):+.1f} conf={float(sample['confidence_pct']):.0f}%"
    )


def _bad_y_only_lock(before: dict | None, after: dict | None = None) -> str | None:
    if isinstance(before, dict) and _visible(before):
        before_x = float(before["x_mm"])
        if abs(before_x) > MAX_ABS_X_FOR_Y_ONLY_MM:
            return f"before |x|={abs(before_x):.1f}>{MAX_ABS_X_FOR_Y_ONLY_MM:.1f}"
    if isinstance(before, dict) and isinstance(after, dict) and _visible(before) and _visible(after):
        before_x = float(before["x_mm"])
        after_x = float(after["x_mm"])
        if abs(after_x) > MAX_ABS_X_FOR_Y_ONLY_MM:
            return f"after |x|={abs(after_x):.1f}>{MAX_ABS_X_FOR_Y_ONLY_MM:.1f}"
        x_jump = abs(after_x - before_x)
        if x_jump > MAX_X_JUMP_FOR_Y_ONLY_MM:
            return f"x jump={x_jump:.1f}>{MAX_X_JUMP_FOR_Y_ONLY_MM:.1f}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--target-y-mm", type=float, default=None)
    parser.add_argument("--tol-y-mm", type=float, default=None)
    parser.add_argument("--hard-floor-y-mm", type=float, default=None)
    parser.add_argument("--max-pulses", type=int, default=10000)
    parser.add_argument("--target-happy-reads", type=int, default=0)
    parser.add_argument("--missing-recovery-down-ms", type=int, default=MISSING_RECOVERY_DOWN_MS)
    parser.add_argument("--initial-missing-recovery-cmd", choices=("u", "d"), default="d")
    parser.add_argument("--max-missing-recoveries", type=int, default=6)
    parser.add_argument("--read-frames", type=int, default=5)
    parser.add_argument("--read-sample-s", type=float, default=0.05)
    parser.add_argument("--settle-s", type=float, default=OBSERVE_SETTLE_S)
    parser.add_argument("--enforce-x-lock", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--perturb-ms", type=int, default=DEFAULT_PERTURB_MS)
    parser.add_argument("--reset-after-happy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-cmd", choices=("u", "d"), default="u")
    parser.add_argument("--reset-min-ms", type=int, default=1350)
    parser.add_argument("--reset-max-ms", type=int, default=3150)
    parser.add_argument("--require-reset-before-count", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--happy-streak-before-perturb", type=int, default=DEFAULT_HAPPY_STREAK_BEFORE_PERTURB)
    parser.add_argument("--stream-url", default=None)
    args = parser.parse_args()

    y_cfg = _follow_y_axis_config()
    target_y = float(args.target_y_mm if args.target_y_mm is not None else y_cfg.get("win_target_mm", 9.0))
    hard_floor_y = args.hard_floor_y_mm
    if hard_floor_y is None and y_cfg.get("hard_floor_y_mm") is not None:
        hard_floor_y = float(y_cfg.get("hard_floor_y_mm"))
    raw_tol_y = float(args.tol_y_mm if args.tol_y_mm is not None else y_cfg.get("win_tol_mm", 4.0))
    tol_y = _win_effective_tolerance(raw_tol_y)
    if tol_y <= 0.0:
        tol_y = raw_tol_y
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + max(0.0, float(args.duration_s))
    pulse_count = 0
    y_happy_count = 0
    happy_streak = 0
    trial_count = 0
    next_perturb_cmd = "d"
    reset_required_before_count = bool(args.require_reset_before_count)
    missing_recovery_cmd = str(args.initial_missing_recovery_cmd)
    missing_recovery_count = 0
    last_regressed_by_cmd: dict[str, bool] = {"u": False, "d": False}

    print(
        f"[Y-ONLY] Step 1 y-only practice for {float(args.duration_s):.0f}s. "
        f"target y={target_y:+.1f}±{tol_y:.1f} effective ({raw_tol_y:.1f} raw); "
        f"dist target={_dist_target_mm():.1f}; "
        f"x target={_x_target_mm():+.1f}; mast PWM fixed at {PWM}.",
        flush=True,
    )

    reader = StreamTelemetryReader(args.stream_url) if args.stream_url else CameraTelemetryReader()
    with out_path.open("a", encoding="utf-8") as out:
        try:
            while time.monotonic() < deadline and pulse_count < int(args.max_pulses):
                if int(args.target_happy_reads) > 0 and y_happy_count >= int(args.target_happy_reads):
                    print(f"[Y-ONLY] target met: {y_happy_count} y-happy reads.", flush=True)
                    break
                before = _read_stable_sample(reader, frames=int(args.read_frames), sample_s=float(args.read_sample_s))
                if not _visible(before):
                    if missing_recovery_count >= int(args.max_missing_recoveries):
                        print(
                            f"[Y-ONLY] HARD STOP: still missing after {missing_recovery_count} recovery pulse(s).",
                            flush=True,
                        )
                        out.write(
                            json.dumps(
                                {
                                    "kind": "missing_recovery_stop",
                                    "recoveries": missing_recovery_count,
                                    "sample": before,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        out.flush()
                        break
                    recovery_ms = max(1, min(MAX_PULSE_MS, int(args.missing_recovery_down_ms)))
                    recovery_cmd = str(missing_recovery_cmd or "d").strip().lower()
                    print(
                        f"[Y-ONLY] vision missing before act ({_fmt_sample(before)}); "
                        f"{recovery_cmd.upper()} recovery {recovery_ms}ms",
                        flush=True,
                    )
                    _send_mast(recovery_cmd, recovery_ms)
                    missing_recovery_count += 1
                    pulse_count += 1
                    time.sleep(max(0.0, float(args.settle_s)))
                    after = _read_stable_sample(reader, frames=int(args.read_frames), sample_s=float(args.read_sample_s))
                    out.write(json.dumps({"kind": "visibility_recovery", "cmd": recovery_cmd, "duration_ms": recovery_ms, "before": before, "after": after}, sort_keys=True) + "\n")
                    out.flush()
                    if _visible(after):
                        missing_recovery_cmd = recovery_cmd
                    else:
                        print("[Y-ONLY] still missing after recovery pulse; holding before next read.", flush=True)
                        time.sleep(1.0)
                    continue
                missing_recovery_count = 0
                bad_lock = _bad_y_only_lock(before) if bool(args.enforce_x_lock) else None
                if bad_lock:
                    print(f"[Y-ONLY] HARD STOP: bad y-only lock before motion ({bad_lock}); no mast act.", flush=True)
                    out.write(json.dumps({"kind": "bad_lock_stop", "reason": bad_lock, "sample": before}, sort_keys=True) + "\n")
                    out.flush()
                    break

                y = float(before["y_mm"])
                y_err = y - target_y
                abs_err = abs(float(y_err))
                if abs_err <= tol_y:
                    if bool(args.reset_after_happy) and bool(reset_required_before_count):
                        reset_min = max(1, min(MAX_RESET_PULSE_MS, int(args.reset_min_ms)))
                        reset_max = max(reset_min, min(MAX_RESET_PULSE_MS, int(args.reset_max_ms)))
                        reset_ms = int(random.randint(reset_min, reset_max))
                        reset_cmd = str(args.reset_cmd).strip().lower()
                        trial_count += 1
                        print(
                            f"[Y-ONLY] pre-win reset {trial_count}: {reset_cmd.upper()} {reset_ms}ms from happy; win not counted yet.",
                            flush=True,
                        )
                        send_result = _send_mast(reset_cmd, reset_ms, max_duration_ms=MAX_RESET_PULSE_MS)
                        pulse_count += 1
                        time.sleep(max(0.0, float(args.settle_s)))
                        after = _read_stable_sample(reader, frames=int(args.read_frames), sample_s=float(args.read_sample_s))
                        out.write(
                            json.dumps(
                                {
                                    "kind": "pre_win_reset",
                                    "trial": trial_count,
                                    "cmd": reset_cmd,
                                    "duration_ms": reset_ms,
                                    "pwm": PWM,
                                    "before": before,
                                    "after": after,
                                    "send_result": send_result,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        out.flush()
                        missing_recovery_cmd = reset_cmd if _visible(after) else _opposite_mast_cmd(reset_cmd)
                        reset_required_before_count = False
                        happy_streak = 0
                        continue
                    y_happy_count += 1
                    happy_streak += 1
                    print(f"[Y-ONLY] Y HAPPY #{y_happy_count}: {_fmt_sample(before)} y_err={y_err:+.1f}mm", flush=True)
                    out.write(
                        json.dumps(
                            {
                                "kind": "y_happy",
                                "sample": before,
                                "y_err_mm": y_err,
                                "counted_after_reset": bool(args.reset_after_happy),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    out.flush()
                    if bool(args.reset_after_happy):
                        reset_required_before_count = True
                    if happy_streak >= max(1, int(args.happy_streak_before_perturb)) and pulse_count < int(args.max_pulses):
                        perturb_cmd = next_perturb_cmd
                        next_perturb_cmd = "u" if next_perturb_cmd == "d" else "d"
                        perturb_ms = max(MIN_PULSE_MS, min(MAX_PULSE_MS, int(args.perturb_ms)))
                        trial_count += 1
                        print(
                            f"[Y-ONLY] trial {trial_count}: perturb {perturb_cmd.upper()} "
                            f"{perturb_ms}ms from happy, then recover.",
                            flush=True,
                        )
                        send_result = _send_mast(perturb_cmd, perturb_ms)
                        pulse_count += 1
                        time.sleep(max(0.0, float(args.settle_s)))
                        after = _read_stable_sample(reader, frames=int(args.read_frames), sample_s=float(args.read_sample_s))
                        bad_lock = _bad_y_only_lock(before, after) if bool(args.enforce_x_lock) else None
                        out.write(
                            json.dumps(
                                {
                                    "kind": "happy_perturb",
                                    "trial": trial_count,
                                    "cmd": perturb_cmd,
                                    "duration_ms": perturb_ms,
                                    "pwm": PWM,
                                    "before": before,
                                    "after": after,
                                    "send_result": send_result,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        out.flush()
                        missing_recovery_cmd = perturb_cmd if _visible(after) else _opposite_mast_cmd(perturb_cmd)
                        if bad_lock:
                            print(f"[Y-ONLY] HARD STOP: bad y-only lock after perturb ({bad_lock}).", flush=True)
                            break
                        happy_streak = 0
                        continue
                    time.sleep(0.8)
                    continue

                # U raises the observed y reading; D lowers it.
                cmd = "d" if y_err > 0.0 else "u"
                happy_streak = 0
                gap_outside = max(0.0, abs_err - tol_y)
                duration_ms = _plan_duration_ms(cmd, gap_outside, y_cfg, last_regressed=last_regressed_by_cmd.get(cmd, False))
                min_down_mm = _min_y_curve_closes_mm(y_cfg, cmd="d") if cmd == "d" else None
                projected_floor_hit = (
                    hard_floor_y is not None
                    and cmd == "d"
                    and min_down_mm is not None
                    and (y - float(min_down_mm)) <= float(hard_floor_y)
                )
                if hard_floor_y is not None and cmd == "d" and (y <= float(hard_floor_y) or projected_floor_hit):
                    print(
                        f"[Y-ONLY] HARD FLOOR HOLD: y={y:+.1f}mm floor={float(hard_floor_y):+.1f}mm; no D act.",
                        flush=True,
                    )
                    out.write(
                        json.dumps(
                            {
                                "kind": "hard_floor_hold",
                                "cmd": cmd,
                                "sample": before,
                                "target_y_mm": target_y,
                                "floor_y_mm": float(hard_floor_y),
                                "y_err_mm": y_err,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    out.flush()
                    time.sleep(0.8)
                    continue
                print(
                    f"[Y-ONLY] {cmd.upper()} {duration_ms}ms: {_fmt_sample(before)} "
                    f"y_err={y_err:+.1f}mm gap_outside={gap_outside:.1f}mm",
                    flush=True,
                )
                send_result = _send_mast(cmd, duration_ms)
                pulse_count += 1
                time.sleep(max(0.0, float(args.settle_s)))
                after = _read_stable_sample(reader, frames=int(args.read_frames), sample_s=float(args.read_sample_s))
                bad_lock = _bad_y_only_lock(before, after) if bool(args.enforce_x_lock) else None

                improved = None
                overshot = False
                after_y_err = None
                if _visible(after):
                    missing_recovery_cmd = cmd
                    after_y_err = float(after["y_mm"]) - target_y
                    improved = abs(after_y_err) < abs(y_err)
                    overshot = (after_y_err > 0.0) != (y_err > 0.0)
                    last_regressed_by_cmd[cmd] = bool(improved is False)
                    verdict = "improved" if improved else "regressed"
                    if overshot:
                        verdict += "+overshot"
                    print(
                        f"[Y-ONLY] after: {_fmt_sample(after)} y_err={after_y_err:+.1f}mm "
                        f"({verdict})",
                        flush=True,
                    )
                else:
                    missing_recovery_cmd = _opposite_mast_cmd(cmd)
                    last_regressed_by_cmd[cmd] = True
                    print(f"[Y-ONLY] after: {_fmt_sample(after)}; holding before next read.", flush=True)

                out.write(
                    json.dumps(
                        {
                            "kind": "mast_y_trial",
                            "cmd": cmd,
                            "duration_ms": duration_ms,
                            "pwm": PWM,
                            "before": before,
                            "after": after,
                            "before_y_err_mm": y_err,
                            "after_y_err_mm": after_y_err,
                            "improved": improved,
                            "overshot": overshot,
                            "send_result": send_result,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                out.flush()
                if bad_lock:
                    print(f"[Y-ONLY] HARD STOP: bad y-only lock after mast act ({bad_lock}).", flush=True)
                    break
        finally:
            reader.close()
            _stop_robot()

    print(f"[Y-ONLY] done. pulses={pulse_count}, y_happy_reads={y_happy_count}, data={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
