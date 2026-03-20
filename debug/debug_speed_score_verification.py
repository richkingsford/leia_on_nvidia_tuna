#!/usr/bin/env python3
"""
Interactive speed score verification tool.

Runs random robot actions at random speed scores and asks the operator to
confirm whether the observed speed matched the requested score.
"""

import argparse
import random
import time
from dataclasses import dataclass

from helper_robot_control import Robot
import telemetry_process
import telemetry_robot as telemetry_robot_module


ACTION_LABELS = {
    "f": "move forward",
    "b": "move backward",
    "l": "turn left",
    "r": "turn right",
    "u": "lift up",
    "d": "lift down",
}

ALLOWED_COMMANDS = tuple(ACTION_LABELS.keys())


@dataclass
class TrialResult:
    trial_idx: int
    cmd: str
    requested_score: int
    score_model: int | None
    score_effective: int | None
    pwm: int | None
    power: float | None
    duration_ms: int | None
    accurate: bool
    actual_score: int | None
    feedback_raw: str


def log_line(message: str) -> None:
    print(str(message), flush=True)


def parse_commands(raw: str) -> list[str]:
    values = []
    for token in str(raw or "").split(","):
        cmd = token.strip().lower()
        if not cmd:
            continue
        if cmd not in ALLOWED_COMMANDS:
            raise ValueError(f"Unsupported command '{cmd}'. Allowed: {','.join(ALLOWED_COMMANDS)}")
        values.append(cmd)
    if not values:
        raise ValueError("No valid commands provided.")
    return values


def _coerce_int(value, fallback=None):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _coerce_float(value, fallback=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _score_display(value):
    score_val = _coerce_int(value)
    if score_val is None:
        return "n/a"
    return f"{score_val}%"


def _power_display(value):
    power_val = _coerce_float(value)
    if power_val is None:
        return "n/a"
    return f"{power_val:.3f}"


def _ms_display(value):
    ms_val = _coerce_int(value)
    if ms_val is None:
        return "n/a"
    return str(ms_val)


def _summarize_inaccurate(results: list[TrialResult]) -> None:
    inaccurate = [item for item in results if not item.accurate]
    if not inaccurate:
        log_line("[DEBUG-SPEED] Summary: no inaccurate trials were reported.")
        return

    log_line(
        f"[DEBUG-SPEED] Summary: {len(inaccurate)} inaccurate trial(s) out of {len(results)} completed."
    )
    for item in inaccurate:
        log_line(
            "[DEBUG-SPEED]   "
            f"trial={item.trial_idx} cmd={item.cmd.upper()} "
            f"requested={item.requested_score}% "
            f"effective={_score_display(item.score_effective)} "
            f"reported={_score_display(item.actual_score)} "
            f"(pwm={item.pwm if item.pwm is not None else 'n/a'}, "
            f"power={_power_display(item.power)}, "
            f"{_ms_display(item.duration_ms)}ms)"
        )


def _ask_for_feedback(min_score: int, max_score: int) -> tuple[bool, int | None, str, bool]:
    while True:
        raw = input(
            "[DEBUG-SPEED] Accurate? Type 'y' for yes, or type the observed speed score "
            f"({min_score}-{max_score}). Type 'q' to stop: "
        ).strip().lower()
        if raw in ("q", "quit"):
            return False, None, raw, True
        if raw in ("y", "yes"):
            return True, None, raw, False
        maybe_score = _coerce_int(raw)
        if maybe_score is None:
            log_line("[DEBUG-SPEED] Invalid input. Enter 'y', 'q', or a numeric speed score.")
            continue
        if maybe_score < min_score or maybe_score > max_score:
            log_line(
                f"[DEBUG-SPEED] Score out of bounds ({min_score}-{max_score}). Try again."
            )
            continue
        return False, int(maybe_score), raw, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Random speed score verification helper.")
    parser.add_argument("--acts", type=int, default=15, help="Number of random trials to run (default: 15).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--commands",
        type=str,
        default="f,b,l,r",
        help="Comma-separated commands from f,b,l,r,u,d (default: f,b,l,r).",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=telemetry_robot_module.SPEED_SCORE_MIN,
        help=f"Minimum requested speed score (default: {telemetry_robot_module.SPEED_SCORE_MIN}).",
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=telemetry_robot_module.SPEED_SCORE_MAX,
        help=f"Maximum requested speed score (default: {telemetry_robot_module.SPEED_SCORE_MAX}).",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.25,
        help="Extra seconds to wait after each action before asking for feedback (default: 0.25).",
    )
    parser.add_argument(
        "--auto-mode",
        action="store_true",
        help="Route actions through auto-mode command handling (uses auto caps/tuning).",
    )
    args = parser.parse_args()

    try:
        commands = parse_commands(args.commands)
    except ValueError as exc:
        log_line(f"[DEBUG-SPEED] {exc}")
        return 2

    acts = max(1, int(args.acts))
    global_min = int(telemetry_robot_module.SPEED_SCORE_MIN)
    global_max = int(telemetry_robot_module.SPEED_SCORE_MAX)
    min_score = max(global_min, int(args.min_score))
    max_score = min(global_max, int(args.max_score))
    if min_score > max_score:
        min_score, max_score = max_score, min_score

    rng = random.Random(args.seed)
    trials = [(rng.choice(commands), rng.randint(min_score, max_score)) for _ in range(acts)]
    results: list[TrialResult] = []

    auto_cap = getattr(telemetry_process, "AUTO_SPEED_SCORE_HARD_MAX", None)

    log_line("[DEBUG-SPEED] Starting speed score verification.")
    log_line(f"[DEBUG-SPEED] acts={acts} seed={args.seed if args.seed is not None else 'random'}")
    log_line(
        f"[DEBUG-SPEED] commands={','.join(commands)} score_range={min_score}-{max_score} auto_mode={bool(args.auto_mode)}"
    )
    if args.auto_mode and auto_cap is not None:
        log_line(f"[DEBUG-SPEED] NOTE: auto mode hard cap currently {int(auto_cap)}%.")

    robot = None
    aborted = False
    try:
        robot = Robot()
        for idx, (cmd, requested_score) in enumerate(trials, start=1):
            est_power, est_pwm, est_score, est_ms = telemetry_robot_module.speed_power_pwm_for_cmd(
                cmd,
                requested_score,
            )
            label = ACTION_LABELS.get(cmd, cmd)
            log_line(
                "[DEBUG-SPEED] "
                f"Trial {idx}/{acts}: about to {label} ({cmd.upper()}) at requested {requested_score}% "
                f"(est: pwm={est_pwm}, power={est_power:.3f}, {est_ms}ms)."
            )
            gate = input("[DEBUG-SPEED] Press Enter to execute (or type 'q' to stop): ").strip().lower()
            if gate in ("q", "quit"):
                aborted = True
                break

            action_meta = telemetry_process.send_robot_command(
                robot,
                world=None,
                step="DEBUG_SPEED_VERIFY",
                cmd=cmd,
                speed=0.0,
                speed_score=requested_score,
                auto_mode=bool(args.auto_mode),
            )

            if not isinstance(action_meta, dict):
                log_line("[DEBUG-SPEED] WARN: command send returned no metadata; skipping feedback for this trial.")
                continue

            score_model = _coerce_int(action_meta.get("score_model"))
            score_effective = _coerce_int(action_meta.get("score_effective"), score_model)
            pwm = _coerce_int(action_meta.get("pwm"))
            power = _coerce_float(action_meta.get("power"))
            duration_ms = _coerce_int(action_meta.get("duration_ms"))
            cmd_sent = str(action_meta.get("cmd_sent") or cmd).upper()

            log_line(
                "[DEBUG-SPEED] Sent "
                f"{cmd_sent} {_score_display(score_effective)} "
                f"(requested={requested_score}%, model={_score_display(score_model)}, "
                f"pwm={pwm if pwm is not None else 'n/a'}, power={_power_display(power)}, "
                f"{_ms_display(duration_ms)}ms)."
            )

            sleep_s = max(0.0, float(duration_ms or 0) / 1000.0) + max(0.0, float(args.settle_s))
            if sleep_s > 0:
                time.sleep(sleep_s)

            accurate, actual_score, feedback_raw, stop_now = _ask_for_feedback(min_score, max_score)
            if stop_now:
                aborted = True
                break

            if actual_score is not None and score_effective is not None and int(actual_score) == int(score_effective):
                accurate = True

            results.append(
                TrialResult(
                    trial_idx=idx,
                    cmd=cmd,
                    requested_score=int(requested_score),
                    score_model=score_model,
                    score_effective=score_effective,
                    pwm=pwm,
                    power=power,
                    duration_ms=duration_ms,
                    accurate=bool(accurate),
                    actual_score=actual_score,
                    feedback_raw=feedback_raw,
                )
            )

            if robot is not None:
                robot.stop()
    except KeyboardInterrupt:
        aborted = True
        log_line("[DEBUG-SPEED] Interrupted by user.")
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
            try:
                robot.close()
            except Exception:
                pass

    if aborted:
        log_line(f"[DEBUG-SPEED] Run stopped early. Completed trials: {len(results)}.")
    else:
        log_line(f"[DEBUG-SPEED] Run complete. Completed trials: {len(results)}.")

    _summarize_inaccurate(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

