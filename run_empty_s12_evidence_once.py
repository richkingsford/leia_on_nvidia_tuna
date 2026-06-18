"""Run one command-line empty S1 -> S2 proof attempt and update the progress site."""

from __future__ import annotations

import argparse
from pathlib import Path

import a_follow_the_brick as follow
import frozen_step12_trials as frozen
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot


def _attempt_number(site: frozen.ProgressSite) -> int:
    attempts = []
    for row in list(site.state.get("rows") or []):
        try:
            attempts.append(int(row.get("attempt")))
        except (TypeError, ValueError):
            pass
    return (max(attempts) + 1) if attempts else 1


def _capture(
    site: frozen.ProgressSite,
    vision: BrickDetector,
    *,
    attempt: int,
    phase: str,
    step: str | None,
    status: str,
    reason: str,
    reading: dict | None,
    decision_log: list[dict] | None = None,
    result: dict | None = None,
) -> None:
    frozen._capture_phase(
        site,
        vision,
        1,
        attempt,
        phase,
        status,
        reason,
        reading,
        0,
        step=step,
        honest_trial=None,
    )
    rows = site.state.setdefault("rows", [])
    if rows and isinstance(rows[-1], dict):
        if decision_log:
            rows[-1]["decision_log"] = list(decision_log)
        if isinstance(result, dict):
            rows[-1]["result"] = _json_safe(result)
        site._write()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _step2_decision_log(step2_result: dict | None) -> list[dict]:
    if not isinstance(step2_result, dict):
        return []
    precision = step2_result.get("precision_counts")
    samples = precision.get("gap_closure_samples") if isinstance(precision, dict) else None
    out: list[dict] = []
    for index, sample in enumerate(samples or [], start=1):
        if not isinstance(sample, dict):
            continue
        out.append(
            {
                "t_ms": index,
                "action": sample.get("action"),
                "axis": sample.get("axis"),
                "before_err": sample.get("before_err"),
                "after_err": sample.get("after_err"),
                "reduction": sample.get("reduction"),
                "duration_ms": sample.get("duration_ms"),
                "reason": "step2_precision_gap_closure",
            }
        )
    if not out:
        out.append(
            {
                "t_ms": 0,
                "action": "STEP2_NO_PRECISION_LOG",
                "reason": str(step2_result.get("reason") or "step2_no_reason"),
                "target_met": bool(step2_result.get("target_met")),
            }
        )
    return out


def run(args: argparse.Namespace) -> int:
    follow._set_game_profile("empty")
    site = frozen.ProgressSite(Path(args.site_dir), "Leia Empty S1/S2 Command-Line Proof")
    site.set_experiment(
        "empty-s12-command-line",
        "Command-line proof: run S1 then S2 with no AI handholding; record honest reads/photos.",
    )
    site.set_targets()
    attempt = _attempt_number(site)
    site.update_summary(f"Attempt {attempt}: checking current pose, then running S1 -> S2 by command line")

    vision = None
    robot = None
    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        robot = Robot()

        current = follow._wait_for_confident_brick(vision, timeout_s=5.0, sample_s=0.12)
        s1_now = bool(follow._step1_dist_x_target_ready(current))
        s2_now, s2_now_reason, _ = follow._step2_targets_ready(current)
        current_reason = f"current_pose s1={s1_now} s2={bool(s2_now)} {s2_now_reason}"
        print(f"[S12] {current_reason}: {follow._custom_sequence_reading_text(current)}", flush=True)
        _capture(
            site,
            vision,
            attempt=attempt,
            phase="start",
            step="step2" if bool(s2_now) else "step1",
            status="info",
            reason=current_reason,
            reading=current,
        )

        stats = follow._follow_loop(
            vision,
            robot,
            duration_s=float(args.step1_timeout_s),
            reset_after_win=False,
            stop_after_win=True,
            stop_after_step2=False,
            complete_after_step3=False,
            max_cycles=1,
            step2_probe_before_forward=False,
            debug_mode=False,
            require_step1_motion_before_win=False,
        )
        s1_reading = stats.get("last_step1_win") if isinstance(stats.get("last_step1_win"), dict) else stats.get("latest_step1_gap")
        s1_win = bool(int(stats.get("win_count", 0) or 0) > 0 and isinstance(stats.get("last_step1_win"), dict))
        s1_reason = "step1_win" if s1_win else str(stats.get("last_action") or "step1_failed")
        _capture(
            site,
            vision,
            attempt=attempt,
            phase="step1",
            step="step1",
            status="win" if s1_win else "fail",
            reason=s1_reason,
            reading=s1_reading,
            decision_log=stats.get("decision_log") if isinstance(stats, dict) else None,
            result=stats,
        )
        if not s1_win:
            site.update_summary(f"Attempt {attempt}: S1 failed by command line; S2 not attempted")
            return 1

        step2_result = follow._run_step2_seat_sequence(vision, robot)
        follow._record_step2_stats(stats, step2_result)
        s2_reading = step2_result.get("reading") if isinstance(step2_result, dict) else None
        s2_win = bool(follow._confirmed_step_result(step2_result))
        s2_reason = str(step2_result.get("reason") if isinstance(step2_result, dict) else "step2_failed")
        _capture(
            site,
            vision,
            attempt=attempt,
            phase="step2",
            step="step2",
            status="win" if s2_win else "fail",
            reason=s2_reason,
            reading=s2_reading,
            decision_log=_step2_decision_log(step2_result),
            result=step2_result,
        )
        site.update_summary(
            f"Attempt {attempt}: command-line S1 {'win' if s1_win else 'fail'}, "
            f"S2 {'win' if s2_win else 'fail'} ({s2_reason})"
        )
        return 0 if s2_win else 2
    finally:
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", default=str(frozen.DEFAULT_SITE_DIR))
    parser.add_argument("--step1-timeout-s", type=float, default=35.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
