"""Run one modified E2E trial and publish per-step evidence to the proof site."""

from __future__ import annotations

import argparse
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
    ("step3_lift", "step3"),
    ("profile_holding", None),
    ("holding_step2_drop", "step2"),
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
        "Command-line modified E2E: reset, empty S1, empty S2, lift, holding drop.",
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
    parser.add_argument("--evidence-file", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if bool(parsed.publish_only):
        raise SystemExit(publish_latest(parsed))
    raise SystemExit(run(parsed))
