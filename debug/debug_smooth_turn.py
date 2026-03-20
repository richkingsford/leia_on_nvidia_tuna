#!/usr/bin/env python3
"""
zdebug_smooth_turn.py
--------------------
Compartmentalized experiment: prove the robot can turn smoothly left and right
without pausing, using the speed-score model from world_model_robot.json.

- No brick vision needed (watch the brick manually if desired).
- No livestream needed.
- Prints easy-to-read terminal logs of the commanded PWM curve.

Default profile:
  - Hold 1% for 0.25s
  - Smoothly ramp up 1% -> 15%
  - Hold 15% briefly
  - Ramp down faster 15% -> 1%
  - Stop
  - Repeat for the opposite direction
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

import telemetry_robot
from helper_robot_control import Robot


def _clamp01(value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _smoothstep01(value: float) -> float:
    x = _clamp01(value)
    return x * x * (3.0 - (2.0 * x))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def _ceil_div(numer: float, denom: float) -> int:
    if denom <= 0:
        return 0
    return int(math.ceil(float(numer) / float(denom)))


def _pwm_for_turn_score(score: int) -> Tuple[int, int]:
    _, pwm, _, duration_ms = telemetry_robot.speed_power_pwm_for_cmd("l", score)
    try:
        pwm_val = int(round(float(pwm)))
    except (TypeError, ValueError):
        pwm_val = 0
    try:
        dur_val = int(round(float(duration_ms)))
    except (TypeError, ValueError):
        dur_val = int(getattr(telemetry_robot, "ACT_DURATION_MS", 0) or 0)
    return int(pwm_val), max(1, int(dur_val))


def _turn_anchor_pwms(scores: Iterable[int]) -> Dict[int, int]:
    anchors: Dict[int, int] = {}
    for score in scores:
        pwm, _ = _pwm_for_turn_score(int(score))
        anchors[int(score)] = int(pwm)
    return anchors


def _send_turn_pwm(robot: Optional[Robot], cmd: str, pwm: int, duration_ms: int, *, dry_run: bool) -> None:
    if dry_run:
        return
    if robot is None:
        raise RuntimeError("robot is None (dry_run must be True to avoid sending commands)")
    robot.send_command_pwm(cmd, int(pwm), duration_ms=int(duration_ms))


def _schedule_hold(seq: List[Tuple[str, int]], pwm: int, hold_ms: int, dt_ms: int) -> None:
    steps = max(1, _ceil_div(hold_ms, dt_ms))
    for _ in range(steps):
        seq.append(("hold", int(pwm)))


def _schedule_ramp(
    seq: List[Tuple[str, int]],
    *,
    start_score: int,
    end_score: int,
    start_pwm: int,
    end_pwm: int,
    label: str,
    stage_ms: int,
    dt_ms: int,
) -> None:
    steps = max(1, _ceil_div(stage_ms, dt_ms))
    for k in range(1, steps + 1):
        t = float(k) / float(steps)
        eased = _smoothstep01(t)
        pwm = int(round(_lerp(float(start_pwm), float(end_pwm), eased)))
        seq.append((label, pwm))


def _run_turn_curve(
    robot: Optional[Robot],
    *,
    cmd: str,
    direction_name: str,
    min_score: int,
    peak_score: int,
    hold_ms: int,
    peak_hold_ms: int,
    accel_stage_ms: int,
    decel_stage_ms: int,
    dt_ms: int,
    overlap_ms: int,
    dry_run: bool,
) -> None:
    scores = list(range(int(min_score), int(peak_score) + 1))
    anchors = _turn_anchor_pwms(scores)

    desired_duration_ms = int(dt_ms + overlap_ms)
    model_duration_max_ms = max(_pwm_for_turn_score(s)[1] for s in scores)
    duration_ms = max(1, int(dt_ms), int(max(desired_duration_ms, model_duration_max_ms)))
    remap = getattr(telemetry_robot, "COMMAND_REMAP", {}) or {}
    hw_cmd = remap.get(cmd, cmd)

    print()
    print(f"=== Smooth Turn: {direction_name} ===")
    print(f"[CFG] cmd={cmd.upper()} (hw={hw_cmd.upper()}) scores={scores}")
    print(
        f"[CFG] hold={hold_ms}ms, peak_hold={peak_hold_ms}ms, accel_stage={accel_stage_ms}ms, "
        f"decel_stage={decel_stage_ms}ms, dt={dt_ms}ms, cmd_duration={duration_ms}ms "
        f"(dt+overlap={desired_duration_ms}ms, model_max={model_duration_max_ms}ms), dry_run={dry_run}"
    )
    print("[MODEL] Turn PWM anchors:")
    for s in scores:
        pwm, model_ms = _pwm_for_turn_score(s)
        print(f"  - {s:>3}% -> pwm{pwm:>3} (model_dur={model_ms}ms)")

    seq: List[Tuple[str, int]] = []
    _schedule_hold(seq, anchors[scores[0]], hold_ms, dt_ms)

    for a, b in zip(scores, scores[1:]):
        _schedule_ramp(
            seq,
            start_score=a,
            end_score=b,
            start_pwm=anchors[a],
            end_pwm=anchors[b],
            label=f"{a}%->{b}%",
            stage_ms=accel_stage_ms,
            dt_ms=dt_ms,
        )

    if peak_hold_ms and int(peak_hold_ms) > 0:
        _schedule_hold(seq, anchors[scores[-1]], int(peak_hold_ms), dt_ms)

    for a, b in zip(reversed(scores), reversed(scores[:-1])):
        _schedule_ramp(
            seq,
            start_score=a,
            end_score=b,
            start_pwm=anchors[a],
            end_pwm=anchors[b],
            label=f"{a}%->{b}%",
            stage_ms=decel_stage_ms,
            dt_ms=dt_ms,
        )

    # Stop at the end (stand-still).
    seq.append(("stop", 0))

    t0 = time.perf_counter()
    next_send_t = time.perf_counter()
    last_sent_pwm: Optional[int] = None
    last_sent_t: Optional[float] = None
    send_times: List[float] = []
    send_write_ms: List[float] = []
    for idx, (phase, pwm) in enumerate(seq, 1):
        now = time.perf_counter()
        if now < next_send_t:
            time.sleep(next_send_t - now)
        loop_t = time.perf_counter()
        elapsed_s = loop_t - t0

        pwm_val = int(pwm)
        pwm_changed = (last_sent_pwm is None) or (pwm_val != int(last_sent_pwm))
        should_send = False
        if phase == "stop":
            should_send = True
        elif pwm_changed:
            should_send = True
        elif last_sent_t is None:
            should_send = True
        else:
            # Keepalive guard: we only evaluate once per dt, so refresh early enough
            # that the next check still lands well before the act duration expires.
            keepalive_after_ms = max(1.0, float(duration_ms) - (2.0 * float(dt_ms)))
            if (loop_t - float(last_sent_t)) * 1000.0 >= keepalive_after_ms:
                should_send = True

        if should_send:
            mark = "*" if pwm_changed else "~"
            send_start = time.perf_counter()
            if phase == "stop":
                print(f"[{cmd.upper()}] {mark} t={elapsed_s:6.3f}s  STOP")
                _send_turn_pwm(robot, cmd, 0, duration_ms, dry_run=dry_run)
                pwm_val = 0
            else:
                print(f"[{cmd.upper()}] {mark} t={elapsed_s:6.3f}s  pwm{pwm_val:3d}  ({phase})  dur={duration_ms}ms")
                _send_turn_pwm(robot, cmd, pwm_val, duration_ms, dry_run=dry_run)
            send_end = time.perf_counter()
            send_times.append(send_end)
            send_write_ms.append((send_end - send_start) * 1000.0)
            last_sent_pwm = int(pwm_val)
            last_sent_t = float(send_end)

        next_send_t += float(dt_ms) / 1000.0

    if not dry_run and robot is not None:
        # Extra safety: send a stop for the turning channel, then global stop.
        robot.send_command_pwm(cmd, 0, duration_ms=duration_ms)
        robot.stop()

    if send_times:
        gaps_ms = [(b - a) * 1000.0 for a, b in zip(send_times, send_times[1:])]
        max_gap_ms = max(gaps_ms) if gaps_ms else 0.0
        avg_gap_ms = (sum(gaps_ms) / len(gaps_ms)) if gaps_ms else 0.0
        max_write_ms = max(send_write_ms) if send_write_ms else 0.0
        avg_write_ms = (sum(send_write_ms) / len(send_write_ms)) if send_write_ms else 0.0
        slack_ms = float(duration_ms) - float(max_gap_ms)
        print(
            f"[TIMING] sends={len(send_times)} gap_ms(avg={avg_gap_ms:.1f}, max={max_gap_ms:.1f}) "
            f"write_ms(avg={avg_write_ms:.1f}, max={max_write_ms:.1f}) slack_vs_duration={slack_ms:.1f}ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument("--peak-score", type=int, default=15)
    parser.add_argument("--hold-ms", type=int, default=250, help="Hold 1% for this long (ms).")
    parser.add_argument("--peak-hold-ms", type=int, default=150, help="Hold peak score for this long (ms).")
    parser.add_argument("--accel-stage-ms", type=int, default=250, help="Duration (ms) between each score on ramp-up.")
    parser.add_argument("--decel-stage-ms", type=int, default=150, help="Duration (ms) between each score on ramp-down.")
    parser.add_argument("--dt-ms", type=int, default=50, help="Command update interval (ms).")
    parser.add_argument("--overlap-ms", type=int, default=30, help="Extra duration beyond dt (ms) to avoid gaps.")
    parser.add_argument("--pause-between-s", type=float, default=1.0, help="Pause between left and right (seconds).")
    parser.add_argument("--dry-run", action="store_true", help="Print the curve without connecting/sending to robot.")
    args = parser.parse_args()

    min_score = telemetry_robot.normalize_speed_score(args.min_score, default=args.min_score)
    peak_score = telemetry_robot.normalize_speed_score(args.peak_score, default=args.peak_score)
    if peak_score < min_score:
        min_score, peak_score = peak_score, min_score

    robot: Optional[Robot] = None
    if not args.dry_run:
        robot = Robot()
        robot.stop()
        time.sleep(0.2)

    try:
        _run_turn_curve(
            robot,
            cmd="l",
            direction_name="LEFT",
            min_score=min_score,
            peak_score=peak_score,
            hold_ms=int(args.hold_ms),
            peak_hold_ms=int(args.peak_hold_ms),
            accel_stage_ms=int(args.accel_stage_ms),
            decel_stage_ms=int(args.decel_stage_ms),
            dt_ms=int(args.dt_ms),
            overlap_ms=int(args.overlap_ms),
            dry_run=bool(args.dry_run),
        )
        if args.pause_between_s and float(args.pause_between_s) > 0:
            time.sleep(float(args.pause_between_s))
        _run_turn_curve(
            robot,
            cmd="r",
            direction_name="RIGHT",
            min_score=min_score,
            peak_score=peak_score,
            hold_ms=int(args.hold_ms),
            peak_hold_ms=int(args.peak_hold_ms),
            accel_stage_ms=int(args.accel_stage_ms),
            decel_stage_ms=int(args.decel_stage_ms),
            dt_ms=int(args.dt_ms),
            overlap_ms=int(args.overlap_ms),
            dry_run=bool(args.dry_run),
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping...")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
