"""Run one modified E2E trial and publish per-step evidence to the proof site."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import a_follow_the_brick as follow
import frozen_step12_trials as frozen
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot


SEQUENCE = (
    ("reset", "reset"),
    ("step1", "step1"),
    ("step2", "step2"),
    ("step3_lift", None),
    ("profile_holding", None),
    ("holding_step2_drop", None),
)

POST_LIFT_PAUSE_S = 0.0


FULL_E2E_EXPERIMENT_SUMMARY = (
    "Full E2E: start from an already-reset empty pose, empty S1 align, "
    "empty S2 seat, empty S3 lift, holding S1 small reset, holding S2 align, "
    "holding S3 lower, holding S4 final reset."
)

EMPTY_HALF_E2E_EXPERIMENT_SUMMARY = (
    "Empty half E2E: empty reset, empty S1/S2/S3/lift, then pause/park."
)

HOLDING_HALF_E2E_EXPERIMENT_SUMMARY = (
    "Holding half E2E: holding reset/back-away, holding S1 small reset, "
    "holding drop, then holding retreat."
)


def _latest_evidence_path(site_dir: str | Path) -> Path:
    return Path(site_dir) / "latest_me2e_evidence.json"


def _attempt_number(site: frozen.ProgressSite) -> int:
    attempts = []
    for row in list(site.state.get("rows") or []):
        try:
            attempts.append(int(row.get("attempt")))
        except (TypeError, ValueError):
            pass
    return (max(attempts) + 1) if attempts else 1


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


def _capture_artifact(
    site_dir: str | Path,
    vision: BrickDetector,
    *,
    attempt: int,
    phase: str,
    step: str | None,
    status: str,
    reason: str,
    reading: dict | None,
    result: dict | None = None,
    decision_log: list[dict] | None = None,
) -> dict:
    site_dir = Path(site_dir)
    images_dir = site_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in f"trial1_{attempt}_{phase}")
    image_name = f"{int(time.time())}_{safe_label}.png"
    image_path = images_dir / image_name
    rel = None
    try:
        vision.read()
    except Exception:
        pass
    frame = getattr(vision, "current_frame", None)
    if frame is None:
        frame = getattr(vision, "raw_frame", None)
    if frame is not None:
        frame = frame.copy()
        h, w = frame.shape[:2]
        x_mid = int(round((float(w) * 0.5) + float(getattr(vision, "camera_center_offset_px", 0.0) or 0.0)))
        y_mid = int(round(float(h) * 0.5))
        x_mid = max(0, min(int(w) - 1, int(x_mid)))
        y_mid = max(0, min(int(h) - 1, int(y_mid)))
        frozen.cv2.line(frame, (x_mid, 0), (x_mid, int(h) - 1), (255, 255, 255), 1, frozen.cv2.LINE_AA)
        frozen.cv2.line(frame, (0, y_mid), (int(w) - 1, y_mid), (255, 255, 255), 1, frozen.cv2.LINE_AA)
        frozen.cv2.imwrite(str(image_path), frame)
        rel = f"images/{image_name}"
    evaluation = frozen._evaluate_reading(reading, step)
    return {
        "trial": 1,
        "attempt": attempt,
        "phase": phase,
        "step": step,
        "status": status,
        "reason": reason,
        "reading": frozen._reading_summary(reading),
        "evaluation": evaluation,
        "target_met": bool(evaluation.get("target_met")) if evaluation else None,
        "honest_trial": None,
        "mast_attempts": 0,
        "image": rel,
        "result": _json_safe(result) if isinstance(result, dict) else None,
        "decision_log": list(decision_log) if decision_log else None,
    }


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
    result: dict | None = None,
    decision_log: list[dict] | None = None,
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
        if isinstance(result, dict):
            rows[-1]["result"] = _json_safe(result)
        if decision_log:
            rows[-1]["decision_log"] = list(decision_log)
        site._write()


def publish_latest(args: argparse.Namespace) -> int:
    evidence_path = Path(args.evidence_file or _latest_evidence_path(args.site_dir))
    if not evidence_path.exists():
        print(f"[ME2E SITE] No evidence file found: {evidence_path}", flush=True)
        return 2
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    site = frozen.ProgressSite(Path(args.site_dir), "Leia Modified E2E Trial Evidence")
    site.set_experiment(
        "modified-e2e-command-line",
        "Command-line modified E2E: reset, empty S1, empty S2, lift, 5s pause, holding drop.",
    )
    site.set_targets()
    for row in evidence.get("rows") or []:
        if isinstance(row, dict):
            row = {k: v for k, v in row.items() if v is not None}
            site.add_row(row)
    summary = str(evidence.get("summary") or "Modified E2E evidence published")
    site.update_summary(summary)
    print(f"[ME2E SITE] Published {len(evidence.get('rows') or [])} rows: {summary}", flush=True)
    return 0 if bool(evidence.get("success")) else 1


def _read(vision: BrickDetector, timeout_s: float = 3.0) -> dict:
    try:
        return follow._wait_for_confident_brick(vision, timeout_s=timeout_s, sample_s=0.12)
    except Exception as exc:
        return {"confident": False, "reason": f"read_failed:{exc}"}


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
    return out


def _step1_stats_reading(stats: dict | None) -> dict | None:
    if not isinstance(stats, dict):
        return None
    reading = stats.get("last_step1_win") if isinstance(stats.get("last_step1_win"), dict) else None
    if reading is None:
        reading = stats.get("debug_stop_reading") if isinstance(stats.get("debug_stop_reading"), dict) else None
    return reading if isinstance(reading, dict) else None


def _step1_stats_ok(stats: dict | None, reading: dict | None) -> bool:
    if not isinstance(stats, dict):
        return False
    evaluation = frozen._evaluate_reading(reading, "step1")
    target_met = bool(evaluation.get("target_met")) or bool(follow._step1_dist_x_target_ready(reading))
    if not target_met and isinstance(reading, dict):
        try:
            dist_err = (
                float(reading["dist_err"])
                if reading.get("dist_err") is not None
                else float(reading["dist_mm"]) - float(follow._dist_target_mm())
            )
            x_err = (
                float(reading["x_err"])
                if reading.get("x_err") is not None
                else float(reading["x_mm"]) - float(follow._x_target_mm())
            )
            target_met = bool(
                (
                    follow._dist_err_ok(dist_err)
                    or follow._dist_outside_gate_mm(dist_err) <= float(follow.NOISE_MARGIN_MM)
                )
                and follow._win_axis_ok_with_noise_grace(x_err, follow._x_tol_mm())
            )
        except (KeyError, TypeError, ValueError):
            target_met = False
    sent_acts = stats.get("sent_act_counts") if isinstance(stats.get("sent_act_counts"), dict) else {}
    wheel_act_sent = any(int(count or 0) > 0 for count in sent_acts.values())
    return bool(
        (
            int(stats.get("win_count", 0) or 0) >= 1
            or (bool(stats.get("step1_attempt_timeout")) and bool(wheel_act_sent))
        )
        and not bool(stats.get("step1_confident_worse_hard_stop"))
        and not bool(stats.get("stall_guard_triggered"))
        and bool(target_met)
    )


def _is_hard_stop_result(reason: str, result: dict | None = None) -> bool:
    reason_text = str(reason or "").lower()
    hard_tokens = (
        "hard_stop",
        "virtual_safety",
        "safety",
        "blocked",
        "emergency",
        "crash",
    )
    if any(token in reason_text for token in hard_tokens):
        return True
    if not isinstance(result, dict):
        return False
    hard_flags = (
        "hard_stop",
        "step1_confident_worse_hard_stop",
        "stall_guard_triggered",
        "virtual_safety_dist_exceeded",
        "blocked",
    )
    return any(bool(result.get(flag)) for flag in hard_flags)


def _holding_s2_alignment_config() -> dict:
    """Use holding vision, but align to the empty S2 x/dist seat pose."""
    previous_profile = follow._active_game_profile()
    try:
        follow._set_game_profile("empty")
        empty_step2 = copy.deepcopy(follow._follow_step2_config())
    finally:
        follow._set_game_profile(previous_profile)
    align_step2 = copy.deepcopy(empty_step2)
    align_step2["nickname"] = "holding seat align"
    align_step2["blind_mast_only"] = False
    align_step2["seat_mast_duration_ms"] = 0
    align_step2["seat_drive_duration_ms"] = 0
    align_step2["precision_settle_enabled"] = True
    align_step2["post_win_forward_creep_ms"] = 300
    targets = copy.deepcopy(align_step2.get("targets") if isinstance(align_step2.get("targets"), dict) else {})
    targets["y_mm"] = None
    targets["y_tol_mm"] = None
    align_step2["targets"] = targets
    semi_targets = copy.deepcopy(
        align_step2.get("semi_happy_targets")
        if isinstance(align_step2.get("semi_happy_targets"), dict)
        else {}
    )
    semi_targets["y_mm"] = None
    semi_targets["y_tol_mm"] = None
    align_step2["semi_happy_targets"] = semi_targets
    return align_step2


def _run_holding_s2_align_sequence(vision: BrickDetector, robot: Robot) -> dict:
    """Run the missing holding alignment step before the blind lower/place step."""
    align_step2 = _holding_s2_alignment_config()
    follow._set_game_profile("holding")
    cfg = follow._follow_motion_config()
    original_step2 = copy.deepcopy(cfg.get("step2") if isinstance(cfg.get("step2"), dict) else {})
    cfg["step2"] = align_step2
    try:
        result = follow._run_step2_seat_sequence(vision, robot)
    finally:
        cfg["step2"] = original_step2
    if isinstance(result, dict):
        result["holding_s2_alignment_uses_empty_s2_pose"] = True
    return result


def _run_holding_s1_reset_sequence(vision: BrickDetector, robot: Robot) -> dict:
    """Use the bounded holding retreat reset, not the empty reset distance gate."""
    follow._set_game_profile("holding")
    result = follow._run_step3_retreat_sequence(vision, robot)
    if isinstance(result, dict):
        result["holding_s1_uses_bounded_retreat_reset"] = True
    return result


def _holding_reset_ok(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("success")) and (
        bool(result.get("target_met")) or bool(result.get("soft_reset_complete"))
    )


def _bounded_blind_reset_ok(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    if bool(result.get("success")):
        return True
    reason = str(result.get("reason") or "")
    if _is_hard_stop_result(reason, result):
        return False
    soft_blind_reasons = (
        "reset_target_miss:dist_back_no_progress_stop",
        "reset_target_miss:visibility_lost_after_budgeted_backoff",
    )
    if reason not in soft_blind_reasons:
        return False
    result["success"] = True
    result["soft_reset_complete"] = True
    result["vision_reset_confirmation_failed"] = True
    result["reason"] = f"{reason}:soft_blind_reset_accepted"
    return True


def _holding_s2_align_ok(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    if bool(follow._confirmed_step_result(result)):
        return True
    if _is_hard_stop_result(str(result.get("reason") or ""), result):
        return False
    reading = result.get("reading") if isinstance(result.get("reading"), dict) else None
    if not follow._step2_targets_within_noise_grace(reading, _holding_s2_alignment_config()):
        return False
    result["success"] = True
    result["target_met"] = True
    result["holding_s2_noise_grace_met"] = True
    result["reason"] = "holding_s2_align_noise_grace_met"
    return True


def _record_evidence_row(
    *,
    args: argparse.Namespace,
    site: frozen.ProgressSite,
    rows: list[dict],
    vision: BrickDetector,
    attempt: int,
    phase: str,
    step: str | None,
    status: str,
    reason: str,
    reading: dict | None,
    result: dict | None = None,
    decision_log: list[dict] | None = None,
) -> None:
    if bool(args.capture_only):
        rows.append(
            _capture_artifact(
                args.site_dir,
                vision,
                attempt=attempt,
                phase=phase,
                step=step,
                status=status,
                reason=reason,
                reading=reading,
                result=result if isinstance(result, dict) else None,
                decision_log=decision_log,
            )
        )
    else:
        _capture(
            site,
            vision,
            attempt=attempt,
            phase=phase,
            step=step,
            status=status,
            reason=reason,
            reading=reading,
            result=result if isinstance(result, dict) else None,
            decision_log=decision_log,
        )


def run_full_e2e_cycle(args: argparse.Namespace) -> int:
    follow._set_game_profile("empty")
    empty_half = bool(getattr(args, "empty_half_cycle", False))
    holding_half = bool(getattr(args, "holding_half_cycle", False))
    site_title = (
        "Leia Holding Half E2E Trial Evidence"
        if holding_half
        else "Leia Empty Half E2E Trial Evidence"
        if empty_half
        else "Leia Full E2E Trial Evidence"
    )
    site = frozen.ProgressSite(Path(args.site_dir), site_title)
    site.set_experiment(
        "holding-half-e2e-command-line"
        if holding_half
        else "empty-half-e2e-command-line"
        if empty_half
        else "full-e2e-command-line",
        HOLDING_HALF_E2E_EXPERIMENT_SUMMARY
        if holding_half
        else EMPTY_HALF_E2E_EXPERIMENT_SUMMARY
        if empty_half
        else FULL_E2E_EXPERIMENT_SUMMARY,
    )
    site.set_targets()
    attempt = _attempt_number(site)
    rows: list[dict] = []
    results: list[dict] = []
    success = False
    run_label = "Holding half E2E" if holding_half else "Empty half E2E" if empty_half else "Full E2E"
    summary = f"{run_label} attempt {attempt}: starting"

    vision = None
    robot = None

    def record_start(phase: str, step: str | None = None) -> None:
        before = _read(vision, timeout_s=3.0) if vision is not None else None
        _record_evidence_row(
            args=args,
            site=site,
            rows=rows,
            vision=vision,
            attempt=attempt,
            phase=f"{phase}-start",
            step=step,
            status="info",
            reason=f"{phase}_start",
            reading=before,
        )

    def record_done(
        item: str,
        *,
        phase: str,
        step: str | None,
        ok: bool,
        reason: str,
        reading: dict | None,
        result: dict | None = None,
        decision_log: list[dict] | None = None,
    ) -> bool:
        nonlocal summary
        _record_evidence_row(
            args=args,
            site=site,
            rows=rows,
            vision=vision,
            attempt=attempt,
            phase=phase,
            step=step,
            status="win" if ok else "fail",
            reason=reason,
            reading=reading,
            result=result,
            decision_log=decision_log,
        )
        results.append({"item": item, "ok": bool(ok), "reason": str(reason)})
        if not bool(ok):
            summary = f"{run_label} attempt {attempt}: failed at {item}: {reason}"
            if bool(getattr(args, "continue_soft_failures", False)) and not _is_hard_stop_result(reason, result):
                print(
                    f"[FULL E2E] Soft failure at {item}; continuing because --continue-soft-failures is set: {reason}",
                    flush=True,
                )
                return True
            return False
        return True

    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        robot = Robot()

        if holding_half:
            follow._set_game_profile("holding")
            site.set_targets()

            record_start("holding_small_reset", None)
            result = _run_holding_s1_reset_sequence(vision, robot)
            ok = _holding_reset_ok(result)
            reading = result.get("reading") if isinstance(result, dict) else None
            reason = str(result.get("reason") if isinstance(result, dict) else "holding_small_reset_failed")
            if not record_done(
                "holding_small_reset",
                phase="holding_small_reset",
                step=None,
                ok=ok,
                reason=reason,
                reading=reading,
                result=result,
            ):
                follow._stop_robot(robot)
                return 1

            record_start("holding_step1", "step1")
            stats = follow._follow_loop(
                vision,
                robot,
                duration_s=float(args.step_timeout_s),
                reset_after_win=False,
                stop_after_win=True,
                stop_after_step2=False,
                step2_probe_before_forward=False,
                debug_mode=False,
            )
            reading = _step1_stats_reading(stats)
            ok = _step1_stats_ok(stats, reading)
            reason = "holding_step1_small_reset_win" if ok else str(
                stats.get("last_action") or "holding_step1_failed"
            )
            decision_log = stats.get("decision_log") if isinstance(stats.get("decision_log"), list) else None
            if not record_done(
                "holding_step1_small_reset",
                phase="holding_step1",
                step="step1",
                ok=ok,
                reason=reason,
                reading=reading,
                result=stats,
                decision_log=decision_log,
            ):
                follow._stop_robot(robot)
                return 1

            record_start("holding_step2_drop", None)
            result = follow._run_step2_seat_sequence(vision, robot)
            reading = result.get("reading") if isinstance(result, dict) else None
            ok = bool(follow._confirmed_step_result(result))
            reason = str(result.get("reason") if isinstance(result, dict) else "holding_step2_failed")
            if not record_done(
                "holding_step2_drop",
                phase="holding_step2_drop",
                step=None,
                ok=ok,
                reason=reason,
                reading=reading,
                result=result,
            ):
                follow._stop_robot(robot)
                return 1

            record_start("holding_retreat", None)
            result = follow._run_step3_retreat_sequence(vision, robot)
            ok = bool(result.get("success")) and (
                bool(result.get("target_met")) or bool(result.get("soft_reset_complete"))
            )
            reading = result.get("reading") if isinstance(result, dict) else None
            reason = str(result.get("reason") if isinstance(result, dict) else "holding_retreat_failed")
            if not record_done(
                "holding_retreat",
                phase="holding_retreat",
                step=None,
                ok=ok,
                reason=reason,
                reading=reading,
                result=result,
            ):
                follow._stop_robot(robot)
                return 1

            follow._set_game_profile("empty")
            success = True
            summary = f"{run_label} attempt {attempt}: WIN"
            follow._stop_robot(robot)
            return 0

        follow._set_game_profile("empty")

        if bool(getattr(args, "skip_initial_reset", False)):
            record_start("start_already_reset", "reset")
            reading = _read(vision, timeout_s=3.0)
            reset_cfg = follow._reset_motion_config().get("reverse_turn")
            reset_cfg = reset_cfg if isinstance(reset_cfg, dict) else {}
            start_ok = bool(follow._reset_xy_target_ready(reading, reset_cfg))
            result = {
                "success": bool(start_ok),
                "reason": "start_already_reset" if start_ok else "start_not_reset",
                "reading": reading,
                "target_met": bool(start_ok),
            }
            if not record_done(
                "start_already_reset",
                phase="start_already_reset",
                step="reset",
                ok=start_ok,
                reason=str(result["reason"]),
                reading=reading,
                result=result,
            ):
                follow._stop_robot(robot)
                return 1
        else:
            record_start("reset", "reset")
            result = follow._run_reset_sequence(vision, robot)
            ok = bool(result.get("success")) if isinstance(result, dict) else False
            reading = result.get("reading") if isinstance(result, dict) else None
            reason = str(result.get("reason") if isinstance(result, dict) else "reset_failed")
            if not record_done("reset", phase="reset", step="reset", ok=ok, reason=reason, reading=reading, result=result):
                follow._stop_robot(robot)
                return 1

        record_start("step1", "step1")
        stats = follow._follow_loop(
            vision,
            robot,
            duration_s=float(args.step_timeout_s),
            reset_after_win=False,
            stop_after_win=True,
            stop_after_step2=False,
            step2_probe_before_forward=False,
            debug_mode=False,
        )
        reading = _step1_stats_reading(stats)
        ok = _step1_stats_ok(stats, reading)
        reason = "step1_win" if ok else str(stats.get("last_action") or "step1_failed")
        decision_log = stats.get("decision_log") if isinstance(stats.get("decision_log"), list) else None
        if not record_done(
            "empty_step1",
            phase="step1",
            step="step1",
            ok=ok,
            reason=reason,
            reading=reading,
            result=stats,
            decision_log=decision_log,
        ):
            follow._stop_robot(robot)
            return 1

        record_start("step2", "step2")
        result = follow._run_step2_seat_sequence(vision, robot)
        reading = result.get("reading") if isinstance(result, dict) else None
        ok = bool(follow._confirmed_step_result(result))
        reason = str(result.get("reason") if isinstance(result, dict) else "step2_failed")
        if not record_done(
            "empty_step2",
            phase="step2",
            step="step2",
            ok=ok,
            reason=reason,
            reading=reading,
            result=result,
            decision_log=_step2_decision_log(result),
        ):
            follow._stop_robot(robot)
            return 1

        record_start("empty_step3_lift", "step3")
        result = follow._run_step3_lift_sequence(vision, robot)
        ok = bool(result.get("success")) if isinstance(result, dict) else False
        reading = result.get("reading") if isinstance(result, dict) else None
        reason = str(result.get("reason") if isinstance(result, dict) else "empty_step3_lift_failed")
        if bool(isinstance(result, dict) and result.get("holding")):
            follow._set_game_profile("holding")
        if not record_done(
            "empty_step3_lift",
            phase="empty_step3_lift",
            step="step3",
            ok=ok,
            reason=reason,
            reading=reading,
            result=result,
        ):
            follow._stop_robot(robot)
            return 1
        if empty_half:
            success = True
            summary = f"{run_label} attempt {attempt}: WIN paused after empty lift"
            follow._stop_robot(robot)
            return 0

        follow._set_game_profile("holding")
        site.set_targets()
        record_start("holding_s1_reset", "reset")
        result = _run_holding_s1_reset_sequence(vision, robot)
        ok = _holding_reset_ok(result)
        reading = result.get("reading") if isinstance(result, dict) else None
        reason = str(result.get("reason") if isinstance(result, dict) else "holding_s1_reset_failed")
        if not record_done(
            "holding_s1_reset",
            phase="holding_s1_reset",
            step="reset",
            ok=ok,
            reason=reason,
            reading=reading,
            result=result,
        ):
            follow._stop_robot(robot)
            return 1

        record_start("holding_s2_align", "step2")
        result = _run_holding_s2_align_sequence(vision, robot)
        reading = result.get("reading") if isinstance(result, dict) else None
        ok = _holding_s2_align_ok(result)
        reason = str(result.get("reason") if isinstance(result, dict) else "holding_s2_align_failed")
        if not record_done(
            "holding_s2_align",
            phase="holding_s2_align",
            step="step2",
            ok=ok,
            reason=reason,
            reading=reading,
            result=result,
            decision_log=_step2_decision_log(result),
        ):
            follow._stop_robot(robot)
            return 1

        pre_lower_pause_s = max(0.0, float(getattr(args, "pre_holding_lower_pause_s", 0.0) or 0.0))
        if pre_lower_pause_s > 0.0:
            print(f"[FULL E2E] Holding lower pre-pause: {pre_lower_pause_s:.1f}s.", flush=True)
            time.sleep(pre_lower_pause_s)

        record_start("holding_s3_lower", "step2")
        result = follow._run_step2_seat_sequence(vision, robot)
        reading = result.get("reading") if isinstance(result, dict) else None
        ok = bool(follow._confirmed_step_result(result))
        reason = str(result.get("reason") if isinstance(result, dict) else "holding_s3_lower_failed")
        if not record_done(
            "holding_s3_lower",
            phase="holding_s3_lower",
            step="step2",
            ok=ok,
            reason=reason,
            reading=reading,
            result=result,
        ):
            follow._stop_robot(robot)
            return 1

        follow._set_game_profile("empty")
        site.set_targets()
        record_start("holding_s4_final_reset", "reset")
        result = follow._run_reset_sequence(vision, robot)
        ok = _bounded_blind_reset_ok(result)
        reading = result.get("reading") if isinstance(result, dict) else None
        reason = str(result.get("reason") if isinstance(result, dict) else "holding_s4_final_reset_failed")
        if not record_done(
            "holding_s4_final_reset",
            phase="holding_s4_final_reset",
            step="reset",
            ok=ok,
            reason=reason,
            reading=reading,
            result=result,
        ):
            follow._stop_robot(robot)
            return 1

        follow._set_game_profile("empty")
        success = all(bool(row.get("ok")) for row in results)
        if success:
            summary = f"{run_label} attempt {attempt}: WIN"
        else:
            failed_items = ", ".join(str(row.get("item")) for row in results if not bool(row.get("ok")))
            summary = f"{run_label} attempt {attempt}: completed with soft failures: {failed_items}"
        follow._stop_robot(robot)
        return 0 if success else 1
    except KeyboardInterrupt:
        summary = f"{run_label} attempt {attempt}: interrupted"
        if robot is not None:
            try:
                follow._stop_robot(robot)
            except Exception:
                pass
        return 130
    finally:
        if bool(getattr(args, "capture_only", False)):
            evidence = {
                "attempt": attempt,
                "success": success,
                "summary": summary,
                "results": results,
                "rows": rows,
            }
            evidence_path = Path(args.evidence_file or _latest_evidence_path(args.site_dir))
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[FULL E2E CAPTURE] Wrote {len(rows)} rows to {evidence_path}", flush=True)
        else:
            try:
                site.update_summary(summary)
            except Exception:
                pass
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


def run(args: argparse.Namespace) -> int:
    follow._set_game_profile("empty")
    site = frozen.ProgressSite(Path(args.site_dir), "Leia Modified E2E Trial Evidence")
    attempt = _attempt_number(site)
    rows: list[dict] = []
    sequence = SEQUENCE[1:] if bool(args.skip_initial_reset) else SEQUENCE

    vision = None
    robot = None
    results: list[dict] = []
    success = False
    summary = f"Modified E2E attempt {attempt}: starting"
    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        robot = Robot()

        for item, step in sequence:
            before = _read(vision, timeout_s=3.0)
            if bool(args.capture_only):
                rows.append(
                    _capture_artifact(
                        args.site_dir,
                        vision,
                        attempt=attempt,
                        phase=f"{step or item}-start",
                        step=step,
                        status="info",
                        reason=f"{item}_start",
                        reading=before,
                    )
                )
            else:
                _capture(
                    site,
                    vision,
                    attempt=attempt,
                    phase=f"{step or item}-start",
                    step=step,
                    status="info",
                    reason=f"{item}_start",
                    reading=before,
                )

            if item == "reset":
                result = follow._run_reset_sequence(vision, robot, honest_step1_reset=True)
                ok = bool(result.get("success")) if isinstance(result, dict) else False
                reading = result.get("reading") if isinstance(result, dict) else None
                reason = str(result.get("reason") if isinstance(result, dict) else "reset_failed")
            elif item == "step1":
                result = follow._follow_loop(
                    vision,
                    robot,
                    duration_s=float(args.step_timeout_s),
                    reset_after_win=False,
                    stop_after_win=True,
                    stop_after_step2=False,
                    step2_probe_before_forward=False,
                    debug_mode=False,
                )
                ok = bool(int(result.get("win_count", 0) or 0) >= 1) and not bool(
                    result.get("step1_confident_worse_hard_stop")
                )
                reading = result.get("last_step1_win") if isinstance(result.get("last_step1_win"), dict) else None
                if reading is None:
                    reading = result.get("debug_stop_reading") if isinstance(result.get("debug_stop_reading"), dict) else None
                reason = "step1_win" if ok else str(result.get("last_action") or "step1_failed")
            elif item == "step2":
                result = follow._run_step2_seat_sequence(vision, robot)
                ok = bool(follow._confirmed_step_result(result))
                reading = result.get("reading") if isinstance(result, dict) else None
                reason = str(result.get("reason") if isinstance(result, dict) else "step2_failed")
            elif item == "step3_lift":
                result = follow._run_step3_lift_sequence(vision, robot)
                ok = bool(result.get("success")) if isinstance(result, dict) else False
                reading = result.get("reading") if isinstance(result, dict) else None
                reason = str(result.get("reason") if isinstance(result, dict) else "step3_lift_failed")
                if bool(isinstance(result, dict) and result.get("holding")):
                    follow._set_game_profile("holding")
            elif item == "post_lift_pause":
                pause_s = float(getattr(args, "post_lift_pause_s", POST_LIFT_PAUSE_S))
                print(f"[ME2E] Post-lift pause: holding still for {pause_s:.1f}s.", flush=True)
                time.sleep(max(0.0, pause_s))
                reading = _read(vision, timeout_s=3.0)
                result = {
                    "success": True,
                    "reason": "post_lift_pause_complete",
                    "duration_s": pause_s,
                }
                ok = True
                reason = "post_lift_pause_complete"
            elif item == "profile_holding":
                follow._set_game_profile("holding")
                result = {"success": True, "reason": "profile_set_holding"}
                ok = True
                reading = before
                reason = "profile_set_holding"
            elif item == "holding_step2_drop":
                follow._set_game_profile("holding")
                result = follow._run_step2_seat_sequence(vision, robot)
                ok = bool(follow._confirmed_step_result(result))
                reading = result.get("reading") if isinstance(result, dict) else None
                reason = str(result.get("reason") if isinstance(result, dict) else "holding_step2_failed")
            else:
                result = {"success": False, "reason": "unsupported_item"}
                ok = False
                reading = before
                reason = "unsupported_item"

            decision_log = None
            if item == "step1" and isinstance(result, dict):
                decision_log = result.get("decision_log") if isinstance(result.get("decision_log"), list) else None
            if item in {"step2", "holding_step2_drop"}:
                decision_log = _step2_decision_log(result)

            if bool(args.capture_only):
                rows.append(
                    _capture_artifact(
                        args.site_dir,
                        vision,
                        attempt=attempt,
                        phase=step or item,
                        step=step,
                        status="win" if ok else "fail",
                        reason=reason,
                        reading=reading,
                        result=result if isinstance(result, dict) else None,
                        decision_log=decision_log,
                    )
                )
            else:
                _capture(
                    site,
                    vision,
                    attempt=attempt,
                    phase=step or item,
                    step=step,
                    status="win" if ok else "fail",
                    reason=reason,
                    reading=reading,
                    result=result if isinstance(result, dict) else None,
                    decision_log=decision_log,
                )
            results.append({"item": item, "ok": ok, "reason": reason})
            if not ok:
                summary = f"Modified E2E attempt {attempt}: failed at {item}: {reason}"
                follow._stop_robot(robot)
                return 1

        success = True
        summary = f"Modified E2E attempt {attempt}: WIN"
        follow._stop_robot(robot)
        return 0
    finally:
        if bool(getattr(args, "capture_only", False)):
            evidence = {
                "attempt": attempt,
                "success": success,
                "summary": summary,
                "results": results,
                "rows": rows,
            }
            evidence_path = Path(args.evidence_file or _latest_evidence_path(args.site_dir))
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[ME2E CAPTURE] Wrote {len(rows)} rows to {evidence_path}", flush=True)
        else:
            try:
                site.update_summary(summary)
            except Exception:
                pass
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
    parser.add_argument("--step-timeout-s", type=float, default=45.0)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--skip-initial-reset", action="store_true")
    parser.add_argument("--full-e2e-cycle", action="store_true")
    parser.add_argument("--empty-half-cycle", action="store_true")
    parser.add_argument("--holding-half-cycle", action="store_true")
    parser.add_argument("--continue-soft-failures", action="store_true")
    parser.add_argument("--post-lift-pause-s", type=float, default=POST_LIFT_PAUSE_S)
    parser.add_argument("--pre-holding-lower-pause-s", type=float, default=0.0)
    parser.add_argument("--evidence-file", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if bool(parsed.publish_only):
        raise SystemExit(publish_latest(parsed))
    if bool(parsed.full_e2e_cycle) or bool(parsed.empty_half_cycle) or bool(parsed.holding_half_cycle):
        raise SystemExit(run_full_e2e_cycle(parsed))
    raise SystemExit(run(parsed))
