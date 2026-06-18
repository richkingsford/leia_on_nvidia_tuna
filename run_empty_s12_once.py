#!/usr/bin/env python3
"""Run one proof: clean reset, empty Step 1 win, empty Step 2 win, then park."""

from __future__ import annotations

from pathlib import Path

import a_follow_the_brick as follow
import frozen_step12_trials as site_mod


SITE_DIR = Path("runs/frozen_step12_site")
TRIAL = 1


def attempt_number(site: site_mod.ProgressSite) -> int:
    attempts = []
    for row in list(site.state.get("rows") or []):
        try:
            attempts.append(int(row.get("attempt")))
        except (TypeError, ValueError):
            pass
    return max(attempts) + 1 if attempts else 1


def capture(
    site: site_mod.ProgressSite,
    vision,
    *,
    trial: int,
    attempt: int,
    phase: str,
    status: str,
    reason: str,
    reading: dict,
    step: str | None = None,
    mast_attempts: int = 0,
    decision_log: list[dict] | None = None,
    honest_trial: bool | None = None,
) -> None:
    site_mod._capture_phase(
        site,
        vision,
        trial,
        attempt,
        phase,
        status,
        reason,
        reading if isinstance(reading, dict) else {},
        mast_attempts,
        step=step,
        honest_trial=honest_trial,
    )
    if decision_log:
        rows = site.state.setdefault("rows", [])
        if rows and isinstance(rows[-1], dict):
            rows[-1]["decision_log"] = list(decision_log)
            site._write()


def reset_clean(reading: dict) -> bool:
    cfg = follow._reset_motion_config().get("reverse_turn")
    cfg = cfg if isinstance(cfg, dict) else {}
    try:
        return bool(follow._reset_xy_target_ready(reading, cfg))
    except Exception:
        return False


def step_met(reading: dict, step: str) -> bool:
    try:
        return bool(site_mod._evaluate_reading(reading, step).get("target_met"))
    except Exception:
        return False


def short_read(reading: dict | None) -> str:
    if not isinstance(reading, dict):
        return "read=N/A"

    def fmt(key: str, *, signed: bool = False) -> str:
        try:
            val = float(reading.get(key))
            return f"{val:+.1f}" if signed else f"{val:.1f}"
        except Exception:
            return "N/A"

    return (
        f"dist={fmt('dist_mm')} "
        f"x={fmt('x_mm', signed=True)} "
        f"y={fmt('y_mm', signed=True)} "
        f"conf={fmt('conf')}"
    )


def open_reset_x_if_already_step1(vision, robot, reading: dict, reason: str) -> tuple[dict, str]:
    """Keep reset honest: reset may overlap S1 dist, but not the full S1 gate."""
    if not (reset_clean(reading) and step_met(reading, "step1")):
        return reading, reason
    reset_cfg = follow._reset_motion_config().get("reverse_turn")
    reset_cfg = reset_cfg if isinstance(reset_cfg, dict) else {}
    current = reading
    for duration_ms in (80, 100, 120, 150):
        try:
            x_mm = float(current.get("x_mm"))
        except (TypeError, ValueError):
            return current, f"{reason};reset_x_offset_invalid_x"
        turn_cmd = follow._turn_cmd_to_open_x_gap(x_mm, "l")
        send_result = follow._reset_small_turn_adjust(
            robot,
            turn_cmd=turn_cmd,
            reading=current,
            duration_ms=int(duration_ms),
            reset_cfg=reset_cfg,
            reason="proof_reset_not_step1_yet",
        )
        if isinstance(send_result, dict) and bool(send_result.get("blocked")):
            return current, f"{reason};reset_x_offset_blocked:{send_result.get('reason')}"
        current = follow._reset_read_after_adjustment(
            vision,
            robot,
            duration_ms=int(duration_ms),
            settle_s=0.25,
            context="proof_reset_not_step1_yet",
            fallback=current,
            back_budget=None,
        )
        reason = f"{reason};opened_reset_x_{duration_ms}ms"
        if reset_clean(current) and not step_met(current, "step1"):
            return current, reason
        if not reset_clean(current):
            return current, f"{reason};left_reset_gate"
    return current, f"{reason};still_step1_after_x_offset"


def main() -> int:
    follow._set_game_profile("empty")
    proof_site = site_mod.ProgressSite(SITE_DIR, "Leia Empty Step 1/2 Proof")
    proof_site.set_experiment(
        "honest-empty-s12-proof",
        "Clean reset -> Step 1 -> Step 2, then park.",
    )
    proof_site.set_targets()
    attempt = attempt_number(proof_site)
    print(f"[PROOF] Attempt {attempt}: reset -> S1 -> S2, site={SITE_DIR}", flush=True)

    vision = None
    robot = None
    try:
        vision = follow.BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        robot = follow.Robot()

        pre = follow._wait_for_confident_brick(vision, timeout_s=6.0, sample_s=0.12)
        print("[PROOF] pregame " + short_read(pre), flush=True)

        reset_result = follow._run_reset_sequence(vision, robot)
        reset_reading = reset_result.get("reading") if isinstance(reset_result, dict) else {}
        reset_reason = str(reset_result.get("reason") if isinstance(reset_result, dict) else "reset_failed")
        reset_reading, reset_reason = open_reset_x_if_already_step1(
            vision,
            robot,
            reset_reading,
            reset_reason,
        )
        reset_ok = bool(reset_result.get("success")) and reset_clean(reset_reading) and not step_met(reset_reading, "step1")
        print(f"[PROOF] reset clean={reset_ok} reason={reset_reason} {short_read(reset_reading)}", flush=True)
        capture(
            proof_site,
            vision,
            trial=TRIAL,
            attempt=attempt,
            phase="step1-start",
            status="win" if reset_ok else "fail",
            reason=reset_reason if reset_ok else f"reset_not_clean:{reset_reason}",
            reading=reset_reading,
            step="step1",
        )
        if not reset_ok:
            proof_site.update_summary(f"Attempt {attempt}: reset failed before S1/S2 proof")
            return 2

        stats = follow._follow_loop(
            vision,
            robot,
            duration_s=45.0,
            reset_after_win=False,
            stop_after_win=True,
            stop_after_step2=False,
            debug_mode=False,
            require_step1_motion_before_win=True,
        )
        s1_reading = follow._wait_for_confident_brick(vision, timeout_s=4.0, sample_s=0.12)
        s1_ok = bool(stats.get("win_count", 0) >= 1 and step_met(s1_reading, "step1"))
        print(f"[PROOF] S1 win={s1_ok} {short_read(s1_reading)} last_action={stats.get('last_action')}", flush=True)
        capture(
            proof_site,
            vision,
            trial=TRIAL,
            attempt=attempt,
            phase="step1",
            status="win" if s1_ok else "fail",
            reason="HAPPY" if s1_ok else f"step1_not_met:{stats.get('last_action', 'unknown')}",
            reading=s1_reading,
            step="step1",
            decision_log=stats.get("decision_log") if isinstance(stats.get("decision_log"), list) else None,
        )
        if not s1_ok:
            capture(
                proof_site,
                vision,
                trial=TRIAL,
                attempt=attempt,
                phase="trial",
                status="fail",
                reason="trial incomplete: Step 1 not honestly met",
                reading=s1_reading,
                honest_trial=False,
            )
            proof_site.update_summary(f"Attempt {attempt}: stopped after Step 1 failure")
            return 3

        capture(
            proof_site,
            vision,
            trial=TRIAL,
            attempt=attempt,
            phase="step2-start",
            status="start",
            reason="step1_win_start_step2",
            reading=s1_reading,
            step="step2",
        )
        step2_result = follow._run_step2_seat_sequence(vision, robot)
        s2_reading = step2_result.get("reading") if isinstance(step2_result, dict) else {}
        s2_reason = str(step2_result.get("reason") if isinstance(step2_result, dict) else "step2_failed")
        s2_ok = bool(step2_result.get("success") and step2_result.get("target_met") and step_met(s2_reading, "step2"))
        print(f"[PROOF] S2 win={s2_ok} reason={s2_reason} {short_read(s2_reading)}", flush=True)
        capture(
            proof_site,
            vision,
            trial=TRIAL,
            attempt=attempt,
            phase="step2",
            status="win" if s2_ok else "fail",
            reason=s2_reason,
            reading=s2_reading,
            step="step2",
        )
        capture(
            proof_site,
            vision,
            trial=TRIAL,
            attempt=attempt,
            phase="trial",
            status="win" if s2_ok else "fail",
            reason="clean reset -> Step 1 win -> Step 2 win; parked" if s2_ok else f"trial incomplete: {s2_reason}",
            reading=s2_reading,
            honest_trial=bool(s2_ok),
        )
        proof_site.update_summary(
            f"Victory: attempt {attempt} clean reset plus 2 wins"
            if s2_ok
            else f"Attempt {attempt}: stopped after Step 2 failure"
        )
        return 0 if s2_ok else 4
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
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        print("[PROOF] URL http://192.168.1.39:8765/", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
