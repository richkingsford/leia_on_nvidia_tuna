#!/usr/bin/env python3
"""Run five honest empty Step 1 trials and write evidence to the frozen trials site."""

from __future__ import annotations

import argparse
import time
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
    trial: int,
    attempt: int,
    phase: str,
    status: str,
    reason: str,
    reading: dict,
    step: str | None = None,
    honest_trial: bool | None = None,
    mast_attempts: int = 0,
    decision_log: list[dict] | None = None,
    failure_explanation: str | None = None,
    failure_diagnosis: str | None = None,
    failure_plan: str | None = None,
) -> None:
    frozen._capture_phase(
        site,
        vision,
        trial,
        attempt,
        phase,
        status,
        reason,
        reading,
        mast_attempts,
        step=step,
        honest_trial=honest_trial,
    )
    if decision_log or failure_explanation or failure_diagnosis or failure_plan:
        rows = site.state.setdefault("rows", [])
        if rows and isinstance(rows[-1], dict):
            if decision_log:
                rows[-1]["decision_log"] = list(decision_log)
            if failure_explanation:
                rows[-1]["failure_explanation"] = str(failure_explanation)
            if failure_diagnosis:
                rows[-1]["failure_diagnosis"] = str(failure_diagnosis)
            if failure_plan:
                rows[-1]["failure_plan"] = str(failure_plan)
            site._write()


def _top_miss_reason(stats: dict | None) -> str:
    if not isinstance(stats, dict):
        return "no_stats"
    reasons = stats.get("miss_reasons")
    if not isinstance(reasons, dict) or not reasons:
        return str(stats.get("last_action") or "unknown")
    try:
        reason, count = max(reasons.items(), key=lambda item: int(item[1] or 0))
        return f"{reason}={int(count or 0)}"
    except Exception:
        return str(stats.get("last_action") or "unknown")


def _trial_outcome_line(trial: int, ok: bool, reading: dict | None, reason: str) -> str:
    try:
        dist = float((reading or {}).get("dist_mm"))
        x = float((reading or {}).get("x_mm"))
        pose = f"dist={dist:.1f}mm x={x:+.1f}mm"
    except (TypeError, ValueError):
        pose = "dist/x=N/A"
    return f"trial {int(trial)}: {'win' if ok else 'fail'} {pose} reason={reason}"


def _step1_met(reading: dict | None) -> bool:
    return bool(frozen._evaluate_reading(reading, "step1").get("target_met"))


def _step1_win_met(reading: dict | None) -> bool:
    """Score earned S1 wins with the same noise-grace gate used by the controller."""
    return bool(_step1_met(reading) or follow._step1_dist_x_target_ready(reading))


def _confirmed_step1_win_reading(stats: dict | None, fallback: dict | None = None) -> dict:
    """Use the follow loop's confirmed win read instead of a later model-switch read."""
    snapshot = (stats or {}).get("last_step1_win") if isinstance(stats, dict) else None
    if not isinstance(snapshot, dict):
        return dict(fallback) if isinstance(fallback, dict) else {}
    out = dict(fallback) if isinstance(fallback, dict) else {}
    for key in ("dist_mm", "x_mm", "y_mm"):
        if snapshot.get(key) is not None:
            out[key] = snapshot.get(key)
    out["confident"] = True
    out["visible"] = True
    out["reason"] = "confirmed_follow_loop_step1_win"
    out["follow_loop_confirmed_win_read"] = True
    return out


def _step1_dist_lower_bound_mm() -> float:
    return float(follow._dist_target_mm()) - float(follow._dist_tol_mm())


def _reset_dist_lower_bound_mm() -> float:
    reset_cfg = follow._reset_motion_config().get("reverse_turn")
    reset_cfg = reset_cfg if isinstance(reset_cfg, dict) else {}
    return float(follow._reset_dist_floor_mm(reset_cfg))


def _reset_pose_met(reading: dict | None) -> bool:
    """True only when the reset script landed inside the displayed reset gate."""
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        return False
    reset_cfg = follow._reset_motion_config().get("reverse_turn")
    reset_cfg = reset_cfg if isinstance(reset_cfg, dict) else {}
    try:
        return bool(follow._reset_xy_target_ready(reading, reset_cfg))
    except Exception:
        return False


def _read_confident(vision: BrickDetector, timeout_s: float = 4.0) -> dict:
    return follow._wait_for_confident_brick(vision, timeout_s=float(timeout_s), sample_s=0.12)


def _wall_recovery_pose_ok(reading: dict | None, *, min_conf_pct: float = 50.0) -> bool:
    if not isinstance(reading, dict):
        return False
    try:
        dist_mm = float(reading.get("dist_mm"))
        x_mm = float(reading.get("x_mm"))
        conf = float(reading.get("conf"))
    except (TypeError, ValueError):
        return False
    reason = str(reading.get("reason") or "")
    return bool(
        bool(reading.get("visible"))
        and dist_mm < 490.0
        and abs(x_mm) < 160.0
        and conf >= float(min_conf_pct)
        and "ghost" not in reason
        and "placeholder" not in reason
    )


def _read_wall_recovery_pose(
    vision: BrickDetector,
    timeout_s: float = 8.0,
    *,
    min_conf_pct: float = 50.0,
    required_samples: int = 3,
) -> dict:
    deadline = time.monotonic() + float(timeout_s)
    candidates: list[dict] = []
    best: dict = {}
    samples = 0
    while time.monotonic() <= deadline or samples <= 0:
        reading = follow._read_brick_measurement(vision)
        samples += 1
        if isinstance(reading, dict):
            best = reading
        if _wall_recovery_pose_ok(reading, min_conf_pct=float(min_conf_pct)):
            candidate = dict(reading)
            candidates.append(candidate)
            candidates = candidates[-5:]
            if len(candidates) >= int(required_samples):
                recent = candidates[-int(required_samples):]
                try:
                    dist_values = [float(row.get("dist_mm")) for row in recent]
                    x_values = [float(row.get("x_mm")) for row in recent]
                except (TypeError, ValueError):
                    dist_values = []
                    x_values = []
                if (
                    dist_values
                    and x_values
                    and max(dist_values) - min(dist_values) <= 35.0
                    and max(x_values) - min(x_values) <= 20.0
                ):
                    out = dict(recent[-1])
                    if not bool(out.get("confident")):
                        out["wall_recovery_relaxed_confidence"] = True
                        out["reason"] = f"wall_recovery_{out.get('reason') or 'visible_low_confidence'}"
                    out["wall_recovery_stable_samples"] = int(len(recent))
                    out["wall_recovery_read_samples"] = int(samples)
                    return out
        time.sleep(0.12)
    if isinstance(best, dict):
        best["_wall_recovery_read_samples"] = int(samples)
        return best
    return {}


def _settle_vision_startup(
    vision: BrickDetector,
    seconds: float = 5.0,
    *,
    min_seconds: float = 1.0,
    required_confident: int = 3,
    stable_dist_span_mm: float = 35.0,
    stable_x_span_mm: float = 20.0,
) -> dict:
    """Read through camera startup until the robot-camera stream is usable.

    The old proof runner always burned a fixed 15s here.  That was safe, but it
    made every no-reset Step 1 attempt feel glacial even when the OAK stream had
    already settled.  Keep a bounded warmup, but exit as soon as we have a small
    consecutive cluster of confident, stable, in-range robot-camera readings.
    """
    started = time.monotonic()
    deadline = time.monotonic() + float(seconds)
    stable: list[dict] = []
    best: dict = {}
    while time.monotonic() < deadline:
        try:
            reading = follow._read_brick_measurement(vision)
        except Exception:
            reading = {}
        if isinstance(reading, dict):
            best = reading
            try:
                dist_mm = float(reading.get("dist_mm"))
                x_mm = float(reading.get("x_mm"))
            except (TypeError, ValueError):
                dist_mm = None
                x_mm = None
            if (
                bool(reading.get("confident"))
                and dist_mm is not None
                and x_mm is not None
                and dist_mm <= follow._virtual_safety_max_dist_mm()
            ):
                stable.append({"dist_mm": dist_mm, "x_mm": x_mm, "reading": reading})
                stable = stable[-max(1, int(required_confident)) :]
                dist_values = [float(row["dist_mm"]) for row in stable]
                x_values = [float(row["x_mm"]) for row in stable]
                if (
                    time.monotonic() - started >= float(min_seconds)
                    and len(stable) >= max(1, int(required_confident))
                    and max(dist_values) - min(dist_values) <= float(stable_dist_span_mm)
                    and max(x_values) - min(x_values) <= float(stable_x_span_mm)
                ):
                    return {
                        "ready": True,
                        "elapsed_s": time.monotonic() - started,
                        "samples": len(stable),
                        "reading": reading,
                    }
            else:
                stable = []
        time.sleep(0.10)
    return {
        "ready": False,
        "elapsed_s": time.monotonic() - started,
        "samples": len(stable),
        "reading": best,
    }


def _safe_confident_read(vision: BrickDetector, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + float(timeout_s)
    best = {}
    while time.monotonic() < deadline:
        reading = _read_confident(vision, timeout_s=1.2)
        if isinstance(reading, dict):
            best = reading
        try:
            dist_mm = float(reading.get("dist_mm"))
        except (AttributeError, TypeError, ValueError):
            dist_mm = None
        if bool(reading.get("confident")) and dist_mm is not None and dist_mm <= follow._virtual_safety_max_dist_mm():
            return reading
        time.sleep(0.2)
    return best


def _suspicious_reset_too_close_read(reading: dict | None) -> bool:
    dist_mm = _dist_value(reading)
    if dist_mm is None or dist_mm >= _reset_dist_lower_bound_mm():
        return False
    source = str((reading or {}).get("vision_geometry_source") or "").strip().lower()
    return source.startswith("native_rect_width")


def _stabilize_suspicious_reset_read(
    vision: BrickDetector,
    reading: dict,
    *,
    timeout_s: float = 8.0,
) -> dict:
    """Give native-rect warmup a chance before calling reset too close."""
    if not _suspicious_reset_too_close_read(reading):
        return reading
    try:
        initial_dist = float(reading.get("dist_mm"))
    except (TypeError, ValueError):
        initial_dist = None
    initial_source = str(reading.get("vision_geometry_source") or "unknown")
    print(
        "[RESET] Suspicious too-close native-rect read after reset; "
        f"holding still to stabilize vision (dist={_mm(initial_dist)} source={initial_source}).",
        flush=True,
    )
    deadline = time.monotonic() + float(timeout_s)
    recent: list[dict] = []
    best = dict(reading)
    while time.monotonic() < deadline:
        candidate = follow._read_brick_measurement(vision)
        if isinstance(candidate, dict):
            best = dict(candidate)
        if not isinstance(candidate, dict) or not bool(candidate.get("confident")):
            time.sleep(0.18)
            continue
        dist_mm = _dist_value(candidate)
        if dist_mm is None or dist_mm > follow._virtual_safety_max_dist_mm():
            time.sleep(0.18)
            continue
        recent.append(dict(candidate))
        recent = recent[-4:]
        good_recent = [
            row
            for row in recent
            if _dist_value(row) is not None
            and float(_dist_value(row)) >= _reset_dist_lower_bound_mm()
        ]
        if len(good_recent) >= 2:
            dist_values = [float(_dist_value(row)) for row in good_recent[-2:]]
            if max(dist_values) - min(dist_values) <= 25.0:
                out = dict(good_recent[-1])
                out["reason"] = (
                    f"{out.get('reason') or 'confident_visible'};"
                    f"reset_too_close_reread_from_{_mm(initial_dist)}_{initial_source}"
                )
                print(
                    "[RESET] Stabilized reset read accepted: "
                    f"dist={float(out.get('dist_mm')):.1f}mm "
                    f"x={float(out.get('x_mm')):+.1f}mm "
                    f"source={out.get('vision_geometry_source')}",
                    flush=True,
                )
                return out
        time.sleep(0.18)
    print(
        "[RESET] Stabilization did not escape the too-close read; keeping latest evidence.",
        flush=True,
    )
    return best


def _read_recovery_pose(vision: BrickDetector, timeout_s: float = 3.5) -> dict:
    """Return the best non-ghost confident pose seen during a recovery readback."""
    deadline = time.monotonic() + float(timeout_s)
    best_confident: dict | None = None
    best_any: dict = {}
    samples = 0
    while time.monotonic() <= deadline or samples <= 0:
        reading = follow._read_brick_measurement(vision)
        samples += 1
        if isinstance(reading, dict):
            best_any = reading
        try:
            dist_mm = float(reading.get("dist_mm"))
            conf = float(reading.get("conf"))
        except (AttributeError, TypeError, ValueError):
            time.sleep(0.12)
            continue
        reason = str(reading.get("reason") or "")
        if bool(reading.get("confident")) and dist_mm < 490.0 and "ghost" not in reason and "placeholder" not in reason:
            if best_confident is None:
                best_confident = dict(reading)
            else:
                try:
                    best_dist = float(best_confident.get("dist_mm"))
                    best_conf = float(best_confident.get("conf") or 0.0)
                except (TypeError, ValueError):
                    best_dist = 9999.0
                    best_conf = -1.0
                if conf > best_conf or (abs(dist_mm - best_dist) < 20.0 and dist_mm < best_dist):
                    best_confident = dict(reading)
        time.sleep(0.12)
    if isinstance(best_confident, dict):
        best_confident["_recovery_read_samples"] = int(samples)
        best_confident["_recovery_single_frame_ok"] = True
        return best_confident
    if isinstance(best_any, dict):
        best_any["_recovery_read_samples"] = int(samples)
        return best_any
    return {}


def _dist_value(reading: dict | None) -> float | None:
    try:
        return float((reading or {}).get("dist_mm"))
    except (TypeError, ValueError):
        return None


def _x_value(reading: dict | None) -> float:
    try:
        return float((reading or {}).get("x_mm")) - float(follow._x_target_mm())
    except (TypeError, ValueError):
        return 0.0


def _mm(value: float | None, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:+.1f}" if signed else f"{number:.1f}"


def _axis_band_diagnosis(reading: dict | None, step: str = "step1") -> list[str]:
    evaluation = frozen._evaluate_reading(reading, step)
    axes = evaluation.get("axes") if isinstance(evaluation.get("axes"), dict) else {}
    out: list[str] = []
    for axis in ("dist", "x"):
        data = axes.get(axis) if isinstance(axes.get(axis), dict) else {}
        if bool(data.get("ok")):
            continue
        try:
            value = float(data.get("value_mm"))
            target = float(data.get("target_mm"))
            tol = float(data.get("tol_mm"))
        except (TypeError, ValueError):
            out.append(f"{axis}: no scored read")
            continue
        low = target - tol
        high = target + tol
        signed = axis == "x"
        if value < low:
            out.append(
                f"{axis} undershot: {_mm(value, signed=signed)} is {_mm(low - value)}mm below "
                f"happy band [{_mm(low, signed=signed)}, {_mm(high, signed=signed)}]"
            )
        elif value > high:
            out.append(
                f"{axis} overshot: {_mm(value, signed=signed)} is {_mm(value - high)}mm above "
                f"happy band [{_mm(low, signed=signed)}, {_mm(high, signed=signed)}]"
            )
    return out


def _gap_sample_opinions(stats: dict | None, *, limit: int = 4) -> list[str]:
    samples = (stats or {}).get("gap_closure_samples") if isinstance(stats, dict) else None
    if not isinstance(samples, list):
        return []
    opinions: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        axis = str(sample.get("axis") or "gap")
        action = str(sample.get("action") or "UNKNOWN")
        try:
            before_err = float(sample.get("before_err"))
            after_err = float(sample.get("after_err"))
            before_abs = abs(before_err)
            after_abs = abs(after_err)
            tol = float(sample.get("tol", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if bool(sample.get("overshot")) and not bool(sample.get("overshot_within_tolerance")):
            opinions.append(
                f"{axis} overshot after {action}: err {_mm(before_err, signed=True)} -> {_mm(after_err, signed=True)}"
            )
        elif (
            bool(sample.get("regressed"))
            and float(sample.get("regression_mm", max(0.0, after_abs - before_abs)) or 0.0)
            >= float(follow.NOISE_MARGIN_MM)
        ):
            opinions.append(
                f"{axis} wrong-way after {action}: gap widened to {_mm(after_abs)}mm"
            )
        elif after_abs > max(0.0, tol):
            opinions.append(
                f"{axis} undershot after {action}: still {_mm(after_abs - max(0.0, tol))}mm outside happy"
            )
        if len(opinions) >= int(limit):
            break
    return opinions


def _step1_failure_diagnosis(reading: dict | None, stats: dict | None = None) -> str:
    parts = []
    reason = str((reading or {}).get("reason") or "")
    if not bool((reading or {}).get("confident")):
        if "placeholder_500" in reason or "500" in reason:
            parts.append(
                "vision failed before motion: detector only saw the rejected 500mm placeholder/far-suspect stack; "
                "best fix is to restore a confident stack read before moving, ideally pose dist 230-260mm inside the 275mm wall"
            )
        else:
            parts.append(
                "vision failed before motion: no confident stack read, so the safe choice is no wheel act; "
                "best fix is livestream/camera/pose recovery before retrying"
            )
    wrong_way_detail = (stats or {}).get("step1_last_worse_act_detail") if isinstance(stats, dict) else None
    if isinstance(wrong_way_detail, dict):
        action = str(wrong_way_detail.get("action") or "UNKNOWN")
        count = int(wrong_way_detail.get("count", 0) or 0)
        parts.append(f"wrong-way suspected {count}x after {action}")
    parts.extend(_gap_sample_opinions(stats))
    parts.extend(_axis_band_diagnosis(reading, "step1"))
    try:
        problem = follow._most_egregious_problem(stats if isinstance(stats, dict) else {})
    except Exception:
        problem = None
    if isinstance(problem, dict) and problem.get("problem"):
        parts.append(f"opinion: {problem.get('problem')}; fix: {problem.get('fix')}")
    return " | ".join(str(part) for part in parts if part) or "no actionable diagnosis captured"


def _reset_failure_lines(reading: dict | None, reason: str) -> tuple[str, str, str]:
    dist = _dist_value(reading)
    read_reason = str((reading or {}).get("reason") or reason or "unknown")
    source = str((reading or {}).get("vision_geometry_source") or "-")
    if "virtual_wall_recovery_wrong_way_dist" in str(reason):
        return (
            "Leia tried a tiny recovery move, but the distance gap widened instead of shrinking.",
            f"The recovery primitive is wrong for this pose or command sign; final read was dist={_mm(dist)}mm.",
            "Disable that primitive for wall recovery and run a tiny supervised direction/curve probe to identify the gap-closing command.",
        )
    if "virtual_wall_recovery_wrong_way_x" in str(reason):
        return (
            "Leia tried a tiny recovery move, but the X gap widened while trying to recover.",
            f"The recovery curve is turning the wrong way for this pose; final read was x={_mm(_x_value(reading), signed=True)}mm from target.",
            "Flip or strengthen the wall-recovery curve choice, then retest with one tiny verified act.",
        )
    if "virtual_wall_recovery_lost_stable_wall_pose" in str(reason):
        return (
            "Leia made a recovery move, then stopped because the wall pose was no longer stable enough to trust.",
            "The proof runner accepts relaxed confidence only while recovering from beyond the wall, and that stability check failed.",
            "Restore a stable stack view or tune the wall recovery pulse so the next read remains consistent.",
        )
    if "virtual_wall_recovery_lost_confident_read" in str(reason):
        return (
            "Leia made recovery progress, then stopped because the detector fell back to the ghost 500mm read.",
            "The movement primitive was closing distance; the blocker is far-distance stack vision becoming unstable during recovery.",
            "Keep Leia stopped when vision ghosts; debug the far-distance/livestream detector before asking the proof runner to continue.",
        )
    if dist is not None and dist > follow._virtual_safety_max_dist_mm():
        return (
            f"Leia is still beyond the {follow._virtual_safety_max_dist_mm():.0f}mm wall after the recovery check.",
            f"The final start-pose read is dist={dist:.1f}mm, so the script must close distance before claiming a proof reset.",
            "Use only verified gap-closing recovery acts; stop and tune the primitive if the next read does not get closer.",
        )
    if "reset_too_close_for_step1" in str(reason):
        return (
            "Leia's reset left it too close to the stack for a Step 1 proof attempt.",
            f"The reset read was dist={_mm(dist)}mm, below the reset-start lower bound {_reset_dist_lower_bound_mm():.1f}mm.",
            "Fix reset so it verifies real backward progress before handing control to Step 1.",
        )
    if not bool((reading or {}).get("confident")):
        return (
            "Leia stopped before moving because it could not get a confident stack read.",
            f"The detector reported {read_reason} from {source}, so dist/x were not trustworthy enough to score.",
            "Restore a confident camera view inside the 275mm wall before tuning movement.",
        )
    return (
        "Leia stopped because the reset pose was not an honest Step 1 starting pose.",
        f"The reset check failed with reason: {reason}.",
        "Use the current read to adjust reset distance/X limits before retrying.",
    )


def _step1_failure_lines(reading: dict | None, stats: dict | None) -> tuple[str, str, str]:
    axis_bits = _axis_band_diagnosis(reading, "step1")
    gap_bits = _gap_sample_opinions(stats, limit=1)
    if axis_bits:
        explanation = f"Leia did not reach Step 1 happy; {axis_bits[0]}."
    else:
        explanation = "Leia did not reach Step 1 happy before the follow loop stopped."
    diagnosis = gap_bits[0] if gap_bits else _step1_failure_diagnosis(reading, stats)
    plan = "Tune the act that produced that miss, then rerun one proof trial without changing unrelated behavior."
    if "wrong-way" in diagnosis or "widened" in diagnosis:
        plan = "Block or flip that movement choice and retest with one cautious single-trial run."
    elif "overshot" in diagnosis:
        plan = "Shorten the responsible pulse/curve and retest with one cautious single-trial run."
    elif "undershot" in diagnosis:
        plan = "Slightly increase the responsible pulse/curve and retest with one cautious single-trial run."
    return explanation, diagnosis, plan


def _recover_inside_virtual_wall(
    vision: BrickDetector,
    robot: frozen.MastFrozenRobot,
    *,
    max_attempts: int = 80,
    initial_reading: dict | None = None,
) -> tuple[bool, dict, str]:
    """If Leia starts too far back, close the gap with tiny verified acts only."""
    reason = "already_inside_virtual_wall"
    last_reading: dict = {}
    next_reading: dict | None = dict(initial_reading) if isinstance(initial_reading, dict) else None
    wrong_way_streak = 0
    no_progress_streak = 0
    saturated_far_streak = 0
    for attempt in range(1, int(max_attempts) + 1):
        if isinstance(next_reading, dict) and _wall_recovery_pose_ok(next_reading):
            reading = dict(next_reading)
            next_reading = None
        else:
            reading = _read_wall_recovery_pose(vision, timeout_s=3.0)
        if isinstance(reading, dict):
            last_reading = reading
        dist_mm = _dist_value(reading)
        if _wall_recovery_pose_ok(reading) and dist_mm is not None and dist_mm <= follow._virtual_safety_max_dist_mm():
            return True, reading, reason
        if not _wall_recovery_pose_ok(reading) or dist_mm is None:
            reason = "virtual_wall_recovery_no_stable_wall_pose"
            break

        if no_progress_streak > 0:
            plan = {
                "kind": "drive",
                "cmd": "f",
                "action": "VIRTUAL_WALL_RECOVERY_FWD",
                "dist_err": dist_mm - float(follow._dist_target_mm()),
                "x_err": _x_value(reading),
                "duration_ms": 250,
                "distance_creep": True,
                "reason": "virtual_wall_recovery_dist_first_after_no_progress",
            }
        else:
            plan = follow._virtual_safety_forward_recovery_plan(
                reading,
                dist_err=dist_mm - float(follow._dist_target_mm()),
                x_err=_x_value(reading),
                y_err=None,
            )
            plan = dict(plan)
        before_x = _x_value(reading)
        duration_ms = int(plan.get("duration_ms", 180) or 180)
        if dist_mm > (float(follow._virtual_safety_max_dist_mm()) + 50.0):
            duration_ms = max(duration_ms, 250)
        plan["duration_ms"] = min(duration_ms, 250)
        plan["action"] = f"RESET_{str(plan.get('action') or 'VIRTUAL_WALL_RECOVERY')}"
        plan["allow_long_duration"] = False
        print(
            f"[PROOF] Virtual wall recovery {attempt}/{int(max_attempts)}: "
            f"dist={dist_mm:.1f}mm > {follow._virtual_safety_max_dist_mm():.1f}mm; "
            f"forward-only {plan.get('action')} {int(plan.get('duration_ms', 0) or 0)}ms.",
            flush=True,
        )
        follow._execute_follow_action(robot, plan, reading)
        time.sleep((float(plan.get("duration_ms", 300) or 300) / 1000.0) + 0.45)
        follow._stop_robot(robot)
        after = _read_wall_recovery_pose(vision, timeout_s=3.5)
        if isinstance(after, dict):
            last_reading = after
        after_dist = _dist_value(after)
        if not _wall_recovery_pose_ok(after) or after_dist is None:
            reason = (
                f"virtual_wall_recovery_lost_stable_wall_pose_after_{attempt}_"
                f"last_good_dist_{float(dist_mm):.1f}_x_{float(before_x):+.1f}_"
                f"via_{plan.get('action')}"
            )
            break
        after_x = _x_value(after)
        dist_delta = float(dist_mm) - float(after_dist)
        x_abs_delta = abs(float(before_x)) - abs(float(after_x))
        print(
            f"[PROOF] Virtual wall recovery readback {attempt}: "
            f"dist {dist_mm:.1f}->{after_dist:.1f}mm delta={dist_delta:+.1f}; "
            f"x_err {before_x:+.1f}->{after_x:+.1f}mm abs_delta={x_abs_delta:+.1f}.",
            flush=True,
        )
        if after_dist <= follow._virtual_safety_max_dist_mm():
            return True, after, f"virtual_wall_recovered_inside_wall_{attempt}"
        wrong_bits = []
        if after_dist > (float(dist_mm) + 2.0):
            wrong_bits.append(f"dist_{float(dist_mm):.1f}_to_{float(after_dist):.1f}")
        if x_abs_delta < -8.0:
            wrong_bits.append(f"x_{float(before_x):+.1f}_to_{float(after_x):+.1f}")
        if wrong_bits:
            wrong_way_streak += 1
            reason = (
                f"virtual_wall_recovery_wrong_way_streak_{wrong_way_streak}_"
                f"{'_'.join(wrong_bits)}_"
                f"via_{plan.get('action')}"
            )
            if wrong_way_streak >= 3:
                break
        else:
            wrong_way_streak = 0

        saturated_far_read = (
            after_dist >= (float(follow._virtual_safety_max_dist_mm()) + 20.0)
            and abs(float(dist_mm) - float(after_dist)) <= 0.5
            and str(plan.get("action") or "").endswith("_FWD")
        )
        if saturated_far_read:
            saturated_far_streak += 1
            no_progress_streak = max(1, no_progress_streak)
            reason = (
                f"virtual_wall_recovery_far_dist_saturated_streak_{saturated_far_streak}_"
                f"dist_{float(dist_mm):.1f}_to_{float(after_dist):.1f}_via_{plan.get('action')}"
            )
            if saturated_far_streak >= 12:
                break
            next_reading = dict(after)
            continue
        saturated_far_streak = 0

        made_progress = dist_delta > 0.5 or x_abs_delta > 1.5
        if not made_progress:
            no_progress_streak += 1
            reason = (
                f"virtual_wall_recovery_no_progress_streak_{no_progress_streak}_"
                f"dist_{float(dist_mm):.1f}_to_{float(after_dist):.1f}_"
                f"x_{float(before_x):+.1f}_to_{float(after_x):+.1f}_via_{plan.get('action')}"
            )
            if no_progress_streak >= 3:
                break
        else:
            no_progress_streak = 0
            reason = f"virtual_wall_verified_recovery_{attempt}"
        next_reading = dict(after)

    return False, last_reading, reason


def _open_x_offset_once(
    vision: BrickDetector,
    robot: frozen.MastFrozenRobot,
    reading: dict,
    *,
    duration_ms: int,
) -> dict:
    reset_cfg = follow._reset_motion_config().get("reverse_turn")
    reset_cfg = reset_cfg if isinstance(reset_cfg, dict) else {}
    try:
        x_mm = float(reading.get("x_mm"))
    except (TypeError, ValueError):
        return reading
    turn_cmd = follow._turn_cmd_to_open_x_gap(x_mm, "l")
    send_result = follow._reset_sharp_turn_adjust(
        robot,
        turn_cmd=turn_cmd,
        reading=reading,
        duration_ms=int(duration_ms),
        reset_cfg=reset_cfg,
        reason="empty_s1_honest_offset",
    )
    if isinstance(send_result, dict) and bool(send_result.get("blocked")):
        out = dict(reading)
        out["reason"] = f"honest_offset_blocked:{send_result.get('reason')}"
        return out
    return follow._reset_read_after_adjustment(
        vision,
        robot,
        duration_ms=int(duration_ms),
        settle_s=0.35,
        context="empty_s1_honest_offset",
        fallback=reading,
        back_budget=None,
    )


def _ensure_honest_reset(
    site: frozen.ProgressSite,
    vision: BrickDetector,
    robot: frozen.MastFrozenRobot,
    *,
    trial: int,
    attempt: int,
    phase: str,
) -> tuple[bool, dict, str]:
    capture_step = "step1" if str(phase) == "step1-start" else None
    wall_reading = _read_wall_recovery_pose(vision, timeout_s=8.0)
    wall_dist = _dist_value(wall_reading)
    if not _wall_recovery_pose_ok(wall_reading) or wall_dist is None:
        reason = "brick_not_visible_stably_before_reset_motion"
        _capture(
            site,
            vision,
            trial=trial,
            attempt=attempt,
            phase=phase,
            status="fail",
            reason=f"reset_not_confident:{reason}",
            reading=wall_reading if isinstance(wall_reading, dict) else {},
            step=capture_step,
            mast_attempts=len(robot.mast_attempts),
        )
        return False, wall_reading if isinstance(wall_reading, dict) else {}, reason
    if bool((wall_reading or {}).get("confident")) and wall_dist is not None and wall_dist > follow._virtual_safety_max_dist_mm():
        _capture(
            site,
            vision,
            trial=trial,
            attempt=attempt,
            phase=phase,
            status="start",
            reason="reset_beyond_virtual_wall:verified_recovery_start",
            reading=wall_reading,
            step=capture_step,
            mast_attempts=len(robot.mast_attempts),
        )
        recovered, recovered_reading, recovered_reason = _recover_inside_virtual_wall(
            vision,
            robot,
            initial_reading=wall_reading,
        )
        if not recovered:
            fail_phase = "step1" if capture_step == "step1" else phase
            _capture(
                site,
                vision,
                trial=trial,
                attempt=attempt,
                phase=fail_phase,
                status="fail",
                reason=f"reset_beyond_virtual_wall:{recovered_reason}",
                reading=recovered_reading,
                step=capture_step,
                mast_attempts=len(robot.mast_attempts),
            )
            return False, recovered_reading, recovered_reason

    reset_result = follow._run_reset_sequence(vision, robot)
    reading = reset_result.get("reading") if isinstance(reset_result, dict) else None
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        reading = _safe_confident_read(vision)
    reason = str(reset_result.get("reason") if isinstance(reset_result, dict) else "reset_failed")
    if not isinstance(reading, dict) or not bool(reading.get("confident")):
        _capture(
            site,
            vision,
            trial=trial,
            attempt=attempt,
            phase=phase,
            status="fail",
            reason=f"reset_not_confident:{reason}",
            reading=reading if isinstance(reading, dict) else {},
            step=capture_step,
            mast_attempts=len(robot.mast_attempts),
        )
        return False, reading if isinstance(reading, dict) else {}, reason
    stabilized = _stabilize_suspicious_reset_read(vision, reading)
    if isinstance(stabilized, dict) and stabilized is not reading:
        if _suspicious_reset_too_close_read(reading) and not _suspicious_reset_too_close_read(stabilized):
            reason = f"{reason};reset_read_stabilized"
        reading = stabilized
    try:
        if float(reading.get("dist_mm")) > follow._virtual_safety_max_dist_mm():
            recovered, recovered_reading, recovered_reason = _recover_inside_virtual_wall(
                vision,
                robot,
                initial_reading=reading,
            )
            if recovered:
                reading = recovered_reading
                reason = f"{reason};post_reset_virtual_wall_recovered:{recovered_reason}"
            else:
                fail_phase = "step1" if capture_step == "step1" else phase
                _capture(
                    site,
                    vision,
                    trial=trial,
                    attempt=attempt,
                    phase=phase,
                    status="start",
                    reason=f"post_reset_beyond_virtual_wall:{reason}:verified_recovery_start",
                    reading=reading,
                    step=capture_step,
                    mast_attempts=len(robot.mast_attempts),
                )
                _capture(
                    site,
                    vision,
                    trial=trial,
                    attempt=attempt,
                    phase=fail_phase,
                    status="fail",
                    reason=f"post_reset_beyond_virtual_wall:{recovered_reason}",
                    reading=recovered_reading,
                    step=capture_step,
                    mast_attempts=len(robot.mast_attempts),
                )
                return False, recovered_reading, recovered_reason
    except (TypeError, ValueError):
        pass

    reset_cfg = follow._reset_motion_config().get("reverse_turn")
    reset_cfg = reset_cfg if isinstance(reset_cfg, dict) else {}

    dist_after_reset = _dist_value(reading)
    reset_lower_bound = _reset_dist_lower_bound_mm()
    if dist_after_reset is not None and dist_after_reset < reset_lower_bound:
        print(
            "[RESET] Proof reset distance polish: "
            f"dist={dist_after_reset:.1f}mm is below true reset floor {reset_lower_bound:.1f}mm; "
            "running bounded reset adjustment before scoring.",
            flush=True,
        )
        adjusted, _target_met, adjustment_attempts = follow._adjust_reset_until_xy_target(
            vision,
            robot,
            reading=reading,
            reset_cfg=reset_cfg,
            initial_turn_cmd="l",
            back_budget=None,
        )
        if isinstance(adjusted, dict):
            reading = adjusted
            reason = f"{reason};proof_dist_back_adjust_{int(adjustment_attempts)}"
            dist_after_reset = _dist_value(reading)

    if dist_after_reset is not None and dist_after_reset < reset_lower_bound:
        reason = f"reset_too_close_for_step1:{reason}"
        _capture(
            site,
            vision,
            trial=trial,
            attempt=attempt,
            phase=phase,
            status="fail",
            reason=reason,
            reading=reading if isinstance(reading, dict) else {},
            step=capture_step,
            mast_attempts=len(robot.mast_attempts),
        )
        return False, reading if isinstance(reading, dict) else {}, reason

    try:
        x_min = float(reset_cfg.get("x_offset_min_mm", 0.0))
        x_max = float(reset_cfg.get("x_offset_max_mm", x_min))
    except (TypeError, ValueError):
        x_min = 0.0
        x_max = 0.0
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    already_step1_polished = False
    for polish_idx in range(2):
        reset_clean = _reset_pose_met(reading)
        already_step1 = _step1_met(reading)
        if reset_clean and not already_step1:
            break
        try:
            abs_x = abs(float((reading or {}).get("x_mm")))
        except (TypeError, ValueError):
            break
        if reset_clean and already_step1:
            if already_step1_polished:
                reason = f"{reason};proof_x_offset_polish_limit_already_step1"
                break
            polish_reason = "already_step1_happy"
        elif abs_x < x_min:
            polish_reason = "below_reset_band"
        else:
            break
        polish_ms = (
            int(follow.RESET_FINAL_X_POLISH_MIN_MS)
            if polish_reason == "already_step1_happy"
            else int(follow.RESET_FINAL_X_POLISH_MAX_MS)
        )
        print(
            "[RESET] Proof reset x-offset polish: "
            f"|x|={abs_x:.1f}mm reason={polish_reason} reset band {x_min:.1f}-{x_max:.1f}mm; "
            f"sending bounded {int(polish_ms)}ms reset turn.",
            flush=True,
        )
        reading = _open_x_offset_once(
            vision,
            robot,
            reading,
            duration_ms=int(polish_ms),
        )
        reason = f"{reason};proof_x_offset_polish_{polish_idx + 1}"
        if polish_reason == "already_step1_happy":
            already_step1_polished = True

    clean_reset = _reset_pose_met(reading)
    honest = bool(clean_reset and not _step1_met(reading))
    if not clean_reset:
        reason = f"reset_target_miss:{reason}"
    elif not honest:
        reason = f"reset_already_step1_happy:{reason}"
    _capture(
        site,
        vision,
        trial=trial,
        attempt=attempt,
        phase=phase,
        status="win" if honest else "fail",
        reason=reason if honest else f"reset_not_honest:{reason}",
        reading=reading if isinstance(reading, dict) else {},
        step=capture_step,
        mast_attempts=len(robot.mast_attempts),
    )
    return honest, reading if isinstance(reading, dict) else {}, reason


def run(args: argparse.Namespace) -> int:
    follow._set_game_profile("empty")
    # This proof exercise is X+dist only. Keep the mast/y axis logged for
    # visibility, but do not let it make a reset look honest or a win look bad.
    frozen.Y_LOCK_TARGET_MM = float(follow._y_win_target_mm())
    frozen.Y_LOCK_TOL_MM = max(float(getattr(frozen, "Y_LOCK_TOL_MM", 5.0)), 20.0)
    site = frozen.ProgressSite(Path(args.site_dir), "Leia Empty Step 1 Honest Reset Trials")
    if bool(args.skip_pre_reset):
        summary = (
            "Empty Step 1 only: trial starts from Leia's current pose by request, "
            "requires real Step 1 motion, then stops after the Step 1 win."
        )
    elif bool(args.skip_post_win_reset):
        summary = (
            "Empty Step 1 only: every trial starts from a non-happy reset pose, "
            "requires real Step 1 motion, then stops after the Step 1 win."
        )
    else:
        summary = (
            "Empty Step 1 only: every trial starts from a non-happy reset pose, "
            "requires real Step 1 motion, then finishes with another honest reset."
        )
    site.set_experiment("honest-empty-s1-reset-proof", summary)
    site.set_targets()
    attempt = _attempt_number(site)
    site.update_summary(f"Starting iteration {attempt}: 0/{int(args.trials)} honest Step 1 wins")
    print(f"[SITE] {Path(args.site_dir).resolve()}", flush=True)
    print(f"[SITE] Suggested URL: {frozen._local_url(int(args.port))}", flush=True)

    vision = None
    base_robot = None
    robot = None
    wins = 0
    outcomes: list[str] = []
    try:
        vision = BrickDetector(debug=True)
        set_tuning = getattr(vision, "set_runtime_tuning", None)
        if callable(set_tuning):
            set_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        print("[VISION] Settling camera startup until stable (max 6.0s)...", flush=True)
        startup_settle = _settle_vision_startup(vision, seconds=6.0)
        settle_elapsed_s = float(startup_settle.get("elapsed_s", 0.0) or 0.0)
        if bool(startup_settle.get("ready")):
            print(f"[VISION] Camera startup ready after {settle_elapsed_s:.1f}s.", flush=True)
        else:
            print(f"[VISION] Camera startup settle used full {settle_elapsed_s:.1f}s; continuing to pregame gate.", flush=True)
        base_robot = Robot()
        robot = frozen.MastFrozenRobot(base_robot)

        for trial in range(1, int(args.trials) + 1):
            if bool(args.skip_pre_reset):
                site.update_summary(f"Iteration {attempt}: trial {trial}/{int(args.trials)} starting from current pose")
                reset_reading = _read_confident(vision)
                reset_reason = "current_pose_no_pre_reset_by_request"
                reset_ok = bool(reset_reading.get("confident"))
                _capture(
                    site,
                    vision,
                    trial=trial,
                    attempt=attempt,
                    phase="step1-start",
                    status="info" if reset_ok else "fail",
                    reason=reset_reason,
                    reading=reset_reading,
                    step="step1",
                    mast_attempts=len(robot.mast_attempts),
                )
            else:
                site.update_summary(f"Iteration {attempt}: trial {trial}/{int(args.trials)} resetting honestly")
                reset_ok, reset_reading, reset_reason = _ensure_honest_reset(
                    site,
                    vision,
                    robot,
                    trial=trial,
                    attempt=attempt,
                    phase="step1-start",
                )
            if not reset_ok:
                diagnosis = _step1_failure_diagnosis(reset_reading)
                explanation_line, diagnosis_line, plan_line = _reset_failure_lines(reset_reading, reset_reason)
                _capture(
                    site,
                    vision,
                    trial=trial,
                    attempt=attempt,
                    phase="trial",
                    status="fail",
                    reason=f"trial incomplete: reset not honest ({reset_reason}) | {diagnosis}",
                    reading=reset_reading,
                    honest_trial=False,
                    mast_attempts=len(robot.mast_attempts),
                    failure_explanation=explanation_line,
                    failure_diagnosis=diagnosis_line,
                    failure_plan=plan_line,
                )
                outcomes.append(_trial_outcome_line(trial, False, reset_reading, f"reset_not_honest:{reset_reason}"))
                if not bool(args.practice_continue):
                    site.update_summary(f"Stopped after trial {trial}: reset was not honest; {explanation_line}")
                    return 2
                site.update_summary(
                    f"Practice iteration {attempt}: trial {trial}/{int(args.trials)} reset failed; continuing for data"
                )
                time.sleep(0.2)
                continue

            site.update_summary(f"Iteration {attempt}: trial {trial}/{int(args.trials)} pursuing Step 1")
            stats = follow._follow_loop(
                vision,
                robot,
                duration_s=float(args.duration_s),
                reset_after_win=False,
                stop_after_win=True,
                stop_after_step2=False,
                debug_mode=False,
                require_step1_motion_before_win=True,
            )
            fallback_reading = _read_confident(vision)
            if int(stats.get("win_count", 0) or 0) >= 1:
                step1_reading = _confirmed_step1_win_reading(stats, fallback=fallback_reading)
            else:
                step1_reading = fallback_reading
            step1_ok = int(stats.get("win_count", 0) or 0) >= 1 and _step1_win_met(step1_reading)
            step1_reason = str(stats.get("last_action") or "step1_follow_loop_done")
            if not step1_ok:
                step1_reason = f"{step1_reason} | {_step1_failure_diagnosis(step1_reading, stats)}"
            _capture(
                site,
                vision,
                trial=trial,
                attempt=attempt,
                phase="step1",
                status="win" if step1_ok else "fail",
                reason=step1_reason,
                reading=step1_reading,
                step="step1",
                mast_attempts=len(robot.mast_attempts),
                decision_log=stats.get("decision_log") if isinstance(stats, dict) else None,
            )
            if not step1_ok:
                diagnosis = _step1_failure_diagnosis(step1_reading, stats)
                explanation_line, diagnosis_line, plan_line = _step1_failure_lines(step1_reading, stats)
                _capture(
                    site,
                    vision,
                    trial=trial,
                    attempt=attempt,
                    phase="trial",
                    status="fail",
                    reason=f"trial incomplete: Step 1 target not honestly met | {diagnosis}",
                    reading=step1_reading,
                    honest_trial=False,
                    mast_attempts=len(robot.mast_attempts),
                    decision_log=stats.get("decision_log") if isinstance(stats, dict) else None,
                    failure_explanation=explanation_line,
                    failure_diagnosis=diagnosis_line,
                    failure_plan=plan_line,
                )
                outcomes.append(
                    _trial_outcome_line(
                        trial,
                        False,
                        step1_reading,
                        f"{_top_miss_reason(stats)}; {diagnosis}",
                    )
                )
                if not bool(args.practice_continue):
                    site.update_summary(f"Stopped after trial {trial}: Step 1 failed; {explanation_line}")
                    return 3
                site.update_summary(
                    f"Practice iteration {attempt}: {wins}/{trial} Step 1 wins; last failed, continuing for data"
                )
                time.sleep(0.2)
                continue

            if bool(args.skip_post_win_reset):
                wins += 1
                outcomes.append(_trial_outcome_line(trial, True, step1_reading, "step1_win"))
                trial_reason = (
                    "current non-happy pose -> earned Step 1; stopped before post-win reset by request"
                    if bool(args.skip_pre_reset)
                    else "clean reset -> earned Step 1; stopped before post-win reset by request"
                )
                _capture(
                    site,
                    vision,
                    trial=trial,
                    attempt=attempt,
                    phase="trial",
                    status="win",
                    reason=trial_reason,
                    reading=step1_reading,
                    honest_trial=True,
                    mast_attempts=len(robot.mast_attempts),
                    decision_log=stats.get("decision_log") if isinstance(stats, dict) else None,
                )
                site.update_summary(f"Victory: iteration {attempt} won {wins}/{int(args.trials)} clean reset -> Step 1 trials")
                time.sleep(0.2)
                continue

            site.update_summary(f"Iteration {attempt}: trial {trial}/{int(args.trials)} post-win honest reset")
            post_ok, post_reading, post_reason = _ensure_honest_reset(
                site,
                vision,
                robot,
                trial=trial,
                attempt=attempt,
                phase="post-reset",
            )
            if not post_ok:
                diagnosis = _step1_failure_diagnosis(post_reading)
                explanation_line, diagnosis_line, plan_line = _reset_failure_lines(post_reading, post_reason)
                _capture(
                    site,
                    vision,
                    trial=trial,
                    attempt=attempt,
                    phase="trial",
                    status="fail",
                    reason=f"trial incomplete: post-win reset not honest ({post_reason}) | {diagnosis}",
                    reading=post_reading,
                    honest_trial=False,
                    mast_attempts=len(robot.mast_attempts),
                    failure_explanation=explanation_line,
                    failure_diagnosis=diagnosis_line,
                    failure_plan=plan_line,
                )
                outcomes.append(
                    _trial_outcome_line(
                        trial,
                        False,
                        post_reading,
                        f"post_win_reset_not_honest:{post_reason}",
                    )
                )
                if not bool(args.practice_continue):
                    site.update_summary(f"Stopped after trial {trial}: post-win reset was not honest; {explanation_line}")
                    return 4
                site.update_summary(
                    f"Practice iteration {attempt}: {wins}/{trial} Step 1 wins; post-win reset failed, continuing for data"
                )
                time.sleep(0.2)
                continue

            wins += 1
            outcomes.append(_trial_outcome_line(trial, True, step1_reading, "honest_step1_win"))
            _capture(
                site,
                vision,
                trial=trial,
                attempt=attempt,
                phase="trial",
                status="win",
                reason="honest reset -> earned Step 1 -> honest reset",
                reading=post_reading,
                honest_trial=True,
                mast_attempts=len(robot.mast_attempts),
                decision_log=stats.get("decision_log") if isinstance(stats, dict) else None,
            )
            site.update_summary(f"Iteration {attempt}: {wins}/{int(args.trials)} honest Step 1 wins")
            time.sleep(0.2)

        if bool(args.practice_continue):
            summary = "; ".join(outcomes[-int(args.trials) :])
            site.update_summary(
                f"Practice complete: iteration {attempt} won {wins}/{int(args.trials)} empty Step 1 trials. {summary}"
            )
            return 0 if int(wins) == int(args.trials) else 5
        site.update_summary(f"Victory: iteration {attempt} won {wins}/{int(args.trials)} honest Step 1 trials")
        return 0
    finally:
        try:
            if robot is not None:
                follow._stop_robot(robot)
        finally:
            close_robot = getattr(base_robot, "close", None)
            if callable(close_robot):
                close_robot()
        close_vision = getattr(vision, "close", None)
        if callable(close_vision):
            close_vision()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=35.0)
    parser.add_argument("--site-dir", default=str(frozen.DEFAULT_SITE_DIR))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--skip-post-win-reset",
        action="store_true",
        help="Stop each trial after clean reset -> Step 1 win, without the extra post-win reset.",
    )
    parser.add_argument(
        "--skip-pre-reset",
        action="store_true",
        help="Start the trial from Leia's current pose and record that start pose instead of issuing a reset.",
    )
    parser.add_argument(
        "--practice-continue",
        action="store_true",
        help="Record all requested trials for practice instead of stopping after the first failed trial.",
    )
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
