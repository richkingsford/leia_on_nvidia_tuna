#!/usr/bin/env python3
"""Run one or more auto-steps from the command line, optionally looping N trials.

═══════════════════════════════════════════════════════════════════════════════
PRACTICE LOOPS vs PRODUCTION RUNS — READ THIS BEFORE TOUCHING ANYTHING
═══════════════════════════════════════════════════════════════════════════════

This script is a PRACTICE LOOP runner. It is NOT a production run runner.
These are fundamentally different things and must NEVER be confused.

PRACTICE LOOP (this file):
  - Uses simplified, hand-written step implementations (_run_exit_wall_with_arc_turns,
    _run_find_brick_with_arc_scan, _run_align_brick_practice, etc.)
  - Designed for rehearsing and tuning robot motion in isolation
  - No production telemetry machinery, no attempt logging to the main log system
  - Speed is hard-capped at CLI_SAFETY_MAX_SPEED_SCORE = 1
  - Turn duration is hard-capped at _PRACTICE_MAX_TURN_MS = 2500 ms
  - Steps succeed/fail based on simple vision heuristics, not production success criteria
  - Safe to run repeatedly without side effects on the production state machine

PRODUCTION RUN (a_MAIN.py / run_auto_step):
  - Uses the full telemetry_process pipeline: speed curves, gap planners,
    forward-while-turning assist, per-step policy configs, attempt logging
  - Governs actual brick pick-and-place operations on the real task
  - Speed, turn timing, and alignment thresholds are all calibrated
  - Step outcomes update world state and drive the next step selection
  - Mistakes here have real consequences (dropped bricks, failed placements)

The `run_auto_step` import exists in this file ONLY as a fallback for any step
token that doesn't have a custom practice implementation (the `else` branch in
`run_step_sequence`). It should never be invoked for the core practice steps
EXIT_WALL, FIND_BRICK, ALIGN_BRICK, or FIND_WALL.

═══════════════════════════════════════════════════════════════════════════════
DEFINITIONS
═══════════════════════════════════════════════════════════════════════════════

TRIAL:
  A single attempt at either picking up OR placing a single brick.
  One trial = one full pass through the practice step sequence (EXIT_WALL →
  FIND_BRICK → ALIGN_BRICK). At the end of the trial the robot either has a
  brick in the gripper (pick trial) or has released one (place trial).

RUN:
  One or more trials executed back-to-back in a single session.
  A run is what you launch from the command line with --trials N.
  The robot stays connected across all trials in a run; the vision session
  and robot connection are shared and torn down only when the run ends.

═══════════════════════════════════════════════════════════════════════════════
RULES — NON-NEGOTIABLE, DO NOT CHANGE WITHOUT EXPLICIT DISCUSSION
═══════════════════════════════════════════════════════════════════════════════

RULE 1 — EXIT THE STACK TO THE LEFT:
  When the robot exits the brick supply stack (EXIT_WALL step), it MUST arc
  to the LEFT. Exiting right drives the robot deeper into the supply area and
  risks collision. The current _run_exit_wall_with_arc_turns implementation
  turns RIGHT — this must be corrected before production use.

RULE 2 — ODD TRIALS PICK, EVEN TRIALS PLACE:
  Trial 1 = pick, trial 2 = place, trial 3 = pick, trial 4 = place, ...
  The step sequence for a pick trial ends with the gripper closing on a brick.
  The step sequence for a place trial ends with the gripper releasing the brick.
  The trial number (1-indexed) determines which action to perform.

RULE 3 — 100 TRIALS PER FULL RUN:
  A complete practice run consists of exactly 100 trials: 50 pick trials
  (odd) and 50 place trials (even). Use --trials 100 for a full run.
  This gives enough data to evaluate consistency and diagnose drift.

RULE 4 — PICK/PLACE SPLIT IS 50/50:
  Because of the odd/even rule, a 100-trial run is always exactly half picks
  and half places. Never run an odd number of trials if you want a balanced
  dataset — you will end up with one more pick than place or vice versa.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import time

import telemetry_process as _telemetry_process

# Safety constants applied to every CLI-launched run.
# Speed: match existing telemetry_process default (25). Lower means slower.
CLI_SAFETY_MAX_SPEED_SCORE = 1
# Turn: hard cap on a single turn pulse in one direction (ms).
CLI_SAFETY_MAX_TURN_MS = 2000

# Backup constants: if the brick is too close at trial start, back up first.
# Triggered when brick is visible and closer than _BACKUP_TRIGGER_DIST_MM.
# Backs up in _BACKUP_PULSE_MS pulses until dist >= _BACKUP_TARGET_DIST_MM.
# Hard limits: total motion time <= _BACKUP_MAX_S; dist increase <= _BACKUP_MAX_DIST_MM.
_BACKUP_TRIGGER_DIST_MM = 120.0  # back up if brick is this close at trial start
_BACKUP_TARGET_DIST_MM = 200.0  # back up to this distance so EXIT_WALL can lose it
_BACKUP_MAX_DIST_MM = 350.0     # never move more than 350mm further from the brick
_BACKUP_MAX_S = 4.0             # never back up for more than 4 seconds total
_BACKUP_PULSE_MS = 400
_BACKUP_SPEED_SCORE = 1

from a_MAIN import (
    VISION_MODE_CYAN,
    _DEFAULT_CYAN_PROFILE,
    _DEFAULT_CYAN_VISIBILITY,
    _apply_cyan_profile,
    _apply_cyan_visibility,
    _stream_state_cyan_profile,
    _stream_state_cyan_visibility,
    AppState,
    build_vision,
    close_log,
    refresh_brick_telemetry,
    resolve_step_token,
    run_auto_step,
)
import telemetry_robot as _telemetry_robot
from helper_manual_turn_arc_assist import execute_manual_turn_arc_plan
from helper_turn_drive_motion import build_turn_drive_motion_plan
from helper_robot_control import Robot
from helper_vision_config import normalize_vision_mode

# ─── HARD SAFETY RULE ────────────────────────────────────────────────────────
# The robot must NEVER be commanded to turn for more than 2.5 seconds in a
# single command, and the total motor-on time for any practice step must not
# exceed 2.5 seconds.  Every arc command duration is capped to this value.
# Centering pulses and step max-times all derive from it.
_PRACTICE_MAX_TURN_MS = 2500   # 2.5 s — absolute hard cap, never raise this
# EXIT_WALL gets a longer budget because the brick supply can be wide; the robot
# needs more time to clear it entirely before FIND_BRICK sweeps back.
_EXIT_WALL_MAX_TURN_MS = 4000  # 4.0 s — EXIT_WALL only
# ─────────────────────────────────────────────────────────────────────────────

_ARC_SCAN_POLL_S = 0.10        # vision poll interval while motor is turning (s)

# EXIT_WALL (practice): continuous right arc turn, stop when brick is gone.
_EXIT_WALL_CONSEC_NOT_VISIBLE = 3   # consecutive not-visible reads → stop + confirm
_EXIT_WALL_CONFIRM_READS = 3        # stationary reads after stop

# FIND_BRICK / FIND_WALL (practice): arc turns, stop when brick confirmed found.
_FIND_BRICK_CONSEC_VISIBLE = 1      # consecutive visible reads → stop + confirm
_FIND_BRICK_CONFIRM_READS = 3       # stationary reads after stop
# Centering: nudge left/right until x_axis is near zero.  Each pulse is 300 ms;
# max 8 acts = 2.4 s total, which fits within the 2.5 s hard cap.
_FIND_BRICK_CENTER_X_TOL_MM = 25
_FIND_BRICK_CENTER_TURN_MS = 300
_FIND_BRICK_CENTER_MAX_ACTS = 8
# Detections beyond this distance are treated as false positives and ignored.
# Real false positives appeared at 4044mm, 10255mm, 10819mm — 1500mm blocks all
# of those while still allowing real detections at ~500-1100mm range.
_PRACTICE_MAX_DETECT_DIST_MM = 1500.0
# False-positive ratio: if a new reading is >50% further than the last accepted
# reading, dismiss it as a false positive.  Without a prior reference, cap the
# first detection at half the hard limit (750mm) — anything beyond that is noise.
_PRACTICE_FP_RATIO = 1.5
_PRACTICE_FP_INIT_MAX_MM = _PRACTICE_MAX_DETECT_DIST_MM / 2.0  # 750 mm


def _is_fp_dist(dist_mm: float, ref_mm: float | None) -> bool:
    """Return True if dist_mm looks like a false positive.

    Two-tier check:
      1. Hard cap: dist > _PRACTICE_MAX_DETECT_DIST_MM → always FP.
      2. Relative: if ref_mm known, dist > ref_mm * 1.5 → FP.
         If ref_mm is None (first detection in sequence), dist > 750mm → FP.
    """
    if dist_mm > _PRACTICE_MAX_DETECT_DIST_MM:
        return True
    if ref_mm is None:
        return dist_mm > _PRACTICE_FP_INIT_MAX_MM
    return dist_mm > ref_mm * _PRACTICE_FP_RATIO

# ALIGN_BRICK (practice): simple custom aligner — no production step machinery.
_ALIGN_BRICK_MAX_ACTS = 15
_ALIGN_BRICK_TARGET_DIST_MM = 155.0   # approach target (mm)
_ALIGN_BRICK_DIST_TOL_MM = 20.0       # acceptable distance window
_ALIGN_BRICK_X_TOL_MM = 25.0          # acceptable x-axis window
_ALIGN_BRICK_APPROACH_MS = 800        # forward/back drive pulse (ms)
_ALIGN_BRICK_TURN_MS = 300            # centering turn pulse (ms)
_ALIGN_BRICK_RECOVERY_MS = 400        # left-turn pulse when brick not visible (ms)


def resolve_step_argument(token: str):
    return resolve_step_token(str(token or "").strip())


def _prime_vision(
    app_state,
    *,
    reads: int = 16,
    sleep_s: float = 0.08,
    require_visible_streak: int = 2,
) -> dict:
    reads_used = max(1, int(reads))
    required_streak = max(1, int(require_visible_streak))
    pose = {"visible": None, "dist_mm": None, "x_axis_mm": None}
    visible_streak = 0
    visible_samples = 0
    for _ in range(int(reads_used)):
        refresh_brick_telemetry(app_state, read_vision=True)
        brick = getattr(app_state.world, "brick", {}) or {}
        visible_now = bool(brick.get("visible"))
        if visible_now:
            visible_samples += 1
            visible_streak += 1
        else:
            visible_streak = 0
        pose = {
            "visible": bool(brick.get("visible")),
            "dist_mm": brick.get("dist"),
            "x_axis_mm": brick.get("x_axis", brick.get("x")),
        }
        if visible_streak >= required_streak:
            break
        time.sleep(max(0.0, float(sleep_s)))
    pose["visible_samples"] = int(visible_samples)
    pose["reads_used"] = int(reads_used)
    return pose


def _apply_cli_safety_overrides(app_state) -> None:
    """Apply speed and turn-duration safety limits to every CLI-launched run."""
    # Cap autonomous speed score. CLI_SAFETY_MAX_SPEED_SCORE sets the ceiling
    # fed to _cap_auto_speed_score; the existing default is 25.
    _telemetry_process.AUTO_SPEED_SCORE_HARD_MAX = CLI_SAFETY_MAX_SPEED_SCORE
    print(
        f"[SAFETY] Speed capped to score {CLI_SAFETY_MAX_SPEED_SCORE}; "
        f"max single turn duration = {CLI_SAFETY_MAX_TURN_MS} ms."
    )


def _build_app_state(vision_mode: str | None):
    mode_raw = None if vision_mode is None else str(vision_mode).strip().lower()
    if mode_raw in (None, "", "auto"):
        app_state = AppState()
        mode_norm = str(getattr(app_state, "active_vision_mode", "cyan"))
    else:
        mode_norm = normalize_vision_mode(mode_raw)
        app_state = AppState(vision_mode=mode_norm)
    app_state.robot = Robot()
    app_state.vision = build_vision(mode_norm)
    if normalize_vision_mode(mode_norm) == VISION_MODE_CYAN and app_state.vision is not None:
        active_cyan_profile = _stream_state_cyan_profile(app_state, _DEFAULT_CYAN_PROFILE)
        active_cyan_visibility = _stream_state_cyan_visibility(app_state, _DEFAULT_CYAN_VISIBILITY)
        _apply_cyan_profile(app_state, app_state.vision, active_cyan_profile)
        _apply_cyan_visibility(app_state, app_state.vision, active_cyan_visibility)
    _apply_cli_safety_overrides(app_state)
    return app_state, mode_norm


def _teardown_app_state(app_state):
    try:
        close_log(app_state, marker=None)
    except Exception:
        pass
    try:
        import helper_xyz_coords as _xyz
        log_path = getattr(app_state, "log_path", None)
        if log_path:
            _xyz.write_run_view_from_log(log_path)
        else:
            _xyz.write_run_view_from_log()
    except Exception:
        pass
    try:
        app_state.vision.close()
    except Exception:
        pass
    try:
        app_state.robot.close()
    except Exception:
        pass


def _backup_if_too_close(app_state) -> dict:
    """Back up if the brick is visible and closer than _BACKUP_TRIGGER_DIST_MM.

    Hard limits: total motion <= _BACKUP_MAX_S seconds AND dist increase <= _BACKUP_MAX_DIST_MM.
    Both limits are checked before every pulse so neither can be exceeded.
    Called at the start of every practice-loop trial.
    """
    from telemetry_robot import speed_power_pwm_for_cmd as _sppc

    refresh_brick_telemetry(app_state, read_vision=True)
    brick = getattr(app_state.world, "brick", {}) or {}
    visible = bool(brick.get("visible"))
    dist_mm = brick.get("dist")

    if not visible or dist_mm is None or float(dist_mm) >= _BACKUP_TRIGGER_DIST_MM:
        return {"backed_up": False, "dist_mm": float(dist_mm) if dist_mm is not None else None}

    dist_mm = float(dist_mm)
    initial_dist_mm = dist_mm
    _, pwm, _, _ = _sppc("b", _BACKUP_SPEED_SCORE)
    total_motion_s = 0.0
    pulse_num = 0

    print(
        f"[BACKUP] Brick at {dist_mm:.0f}mm — backing up to ~{_BACKUP_TARGET_DIST_MM:.0f}mm "
        f"(limits: {_BACKUP_MAX_S:.0f}s / {_BACKUP_MAX_DIST_MM:.0f}mm)."
    )

    while True:
        remaining_s = _BACKUP_MAX_S - total_motion_s
        if remaining_s < 0.1:
            print(f"[BACKUP] Reached {_BACKUP_MAX_S:.0f}s time limit — stopping.")
            break
        if dist_mm >= initial_dist_mm + _BACKUP_MAX_DIST_MM:
            print(f"[BACKUP] Reached {_BACKUP_MAX_DIST_MM:.0f}mm distance limit — stopping.")
            break

        this_pulse_ms = min(_BACKUP_PULSE_MS, int(remaining_s * 1000))
        app_state.robot.send_command_pwm("b", pwm, duration_ms=this_pulse_ms)
        time.sleep(this_pulse_ms / 1000.0 + 0.15)
        total_motion_s += this_pulse_ms / 1000.0
        pulse_num += 1

        refresh_brick_telemetry(app_state, read_vision=True)
        brick = getattr(app_state.world, "brick", {}) or {}
        new_dist = brick.get("dist")
        new_visible = bool(brick.get("visible"))

        if not new_visible or new_dist is None:
            print(f"[BACKUP] Pulse {pulse_num}: brick no longer visible — done.")
            break

        dist_mm = float(new_dist)
        print(
            f"[BACKUP] Pulse {pulse_num}: dist={dist_mm:.0f}mm "
            f"(+{dist_mm - initial_dist_mm:.0f}mm from start, {total_motion_s:.1f}s used)"
        )

        if dist_mm >= _BACKUP_TARGET_DIST_MM:
            print(f"[BACKUP] Reached target {_BACKUP_TARGET_DIST_MM:.0f}mm — done.")
            break

    app_state.robot.stop()
    return {
        "backed_up": True,
        "dist_mm": dist_mm,
        "pulses": pulse_num,
        "total_motion_s": round(total_motion_s, 2),
        "dist_gained_mm": round(dist_mm - initial_dist_mm, 1),
    }


_BACK_BEFORE_SWEEP_MS = 400   # backward pulse before each sweep arc

def _execute_arc_turn(robot, cmd: str, score: int, hold_duration_ms: int, *, drive_mode: str = "forward"):
    """Arc turn: outer tread at full PWM, inner tread stopped.
    drive_mode='forward' → robot arcs forward while turning.
    """
    plan = build_turn_drive_motion_plan(
        cmd=cmd,
        score=score,
        hold_duration_ms=hold_duration_ms,
        profile_override={
            "drive_mode": drive_mode,
            "inner_ratio": 0.0,
            "outer_ratio": 1.0,
        },
    )
    if not isinstance(plan, dict):
        return None
    return robot.send_custom_actions_pwm(
        str(plan.get("cmd") or cmd),
        plan.get("actions") or [],
        duration_ms=int(plan.get("duration_ms") or hold_duration_ms),
    )


def _execute_back_then_arc_turn(robot, cmd: str, score: int, hold_duration_ms: int):
    """Back up briefly (both treads), then forward arc turn. For sweep scans."""
    _, back_pwm, _, _ = _telemetry_robot.speed_power_pwm_for_cmd("b", score)
    robot.send_command_pwm("b", int(back_pwm), duration_ms=_BACK_BEFORE_SWEEP_MS)
    time.sleep(_BACK_BEFORE_SWEEP_MS / 1000.0 + 0.05)
    return _execute_arc_turn(robot, cmd, score, hold_duration_ms, drive_mode="forward")


def _run_exit_wall_with_arc_turns(app_state) -> bool:
    """EXIT_WALL (practice): arc turn until brick is gone, then stop.

    Hard rule: total motor-on time never exceeds _PRACTICE_MAX_TURN_MS (2.5 s).
    Polls vision every 100 ms during the turn. Stops the instant 3 consecutive
    not-visible reads are seen, then does a stationary confirmation check.
    Declares success at the 2.5 s limit regardless (turned far enough).

    *** RULE 1 — EXIT THE STACK TO THE LEFT ***
    The robot MUST exit the stack by arcing LEFT. Currently this function arcs
    RIGHT (cmd="r") which is WRONG for production. The right turn was used
    during early practice tuning to validate arc-assist motion; it must be
    flipped to LEFT before this step is used in a real pick/place run.
    cmd="r" below → must become cmd="l" and wire l.s,r.f before production use.
    """
    pose = _prime_vision(app_state)
    print(f"[EXIT_WALL ARC] Start: visible={pose.get('visible')}, dist={pose.get('dist_mm')}")
    if not pose.get("visible"):
        print("[EXIT_WALL ARC] Brick already not visible — success.")
        return True

    poll_ms = int(_ARC_SCAN_POLL_S * 1000)
    budget_ms = _EXIT_WALL_MAX_TURN_MS
    print(f"[EXIT_WALL ARC] Right turn — polling every {poll_ms}ms, budget {budget_ms}ms.")

    result = _execute_arc_turn(app_state.robot, cmd="r", score=1, hold_duration_ms=budget_ms, drive_mode="forward")
    wire = result.get("wire_text", "?") if isinstance(result, dict) else "?"
    print(f"[EXIT_WALL ARC]   wire: {wire}")

    consec_not_visible = 0
    last_x_axis = None   # track which side brick was on when last seen
    deadline = time.monotonic() + budget_ms / 1000.0

    while time.monotonic() < deadline:
        refresh_brick_telemetry(app_state, read_vision=True)
        brick = getattr(app_state.world, "brick", {}) or {}
        visible = bool(brick.get("visible"))
        dist_raw = brick.get("dist")
        if visible and dist_raw is not None and float(dist_raw) > _PRACTICE_MAX_DETECT_DIST_MM:
            visible = False  # ignore faraway false positives during exit turn

        if not visible:
            consec_not_visible += 1
            if consec_not_visible >= _EXIT_WALL_CONSEC_NOT_VISIBLE:
                try:
                    app_state.robot.stop()
                except Exception:
                    pass
                elapsed_ms = budget_ms - (deadline - time.monotonic()) * 1000
                print(
                    f"[EXIT_WALL ARC] {consec_not_visible} consecutive not-visible "
                    f"at ~{elapsed_ms:.0f}ms — confirming while stationary..."
                )
                confirmed_gone = 0
                for _ in range(_EXIT_WALL_CONFIRM_READS):
                    refresh_brick_telemetry(app_state, read_vision=True)
                    brick = getattr(app_state.world, "brick", {}) or {}
                    conf_vis = bool(brick.get("visible"))
                    conf_dist = brick.get("dist")
                    if conf_vis and conf_dist is not None and float(conf_dist) > _PRACTICE_MAX_DETECT_DIST_MM:
                        conf_vis = False
                    if not conf_vis:
                        confirmed_gone += 1
                    time.sleep(_ARC_SCAN_POLL_S)
                # Store last known side so FIND_BRICK can choose sweep direction.
                app_state._practice_last_exit_x_axis = last_x_axis
                if confirmed_gone >= _EXIT_WALL_CONFIRM_READS - 1:
                    print(f"[EXIT_WALL ARC] Confirmed gone ({confirmed_gone}/{_EXIT_WALL_CONFIRM_READS}) — success. Last x={last_x_axis}")
                    return True
                print(f"[EXIT_WALL ARC] Brick reappeared ({confirmed_gone}/{_EXIT_WALL_CONFIRM_READS} gone) — budget spent.")
                return True  # budget is spent; don't turn more
        else:
            x_raw = brick.get("x_axis", brick.get("x"))
            if x_raw is not None:
                last_x_axis = float(x_raw)
            if consec_not_visible > 0:
                print(f"[EXIT_WALL ARC] Visible again at {float(dist_raw or 0):.0f}mm — resetting streak.")
            consec_not_visible = 0

        time.sleep(_ARC_SCAN_POLL_S)

    # Budget exhausted — store last known x_axis for FIND_BRICK.
    app_state._practice_last_exit_x_axis = last_x_axis

    try:
        app_state.robot.stop()
    except Exception:
        pass
    print(f"[EXIT_WALL ARC] {budget_ms}ms budget exhausted — declaring success.")
    return True


def _center_brick_x_axis(app_state, ref_dist: float | None = None) -> None:
    """Nudge left/right until brick x_axis is near zero, with one recovery attempt.

    ref_dist: last accepted distance — seeds the 50% false-positive filter.
    If the brick disappears during a centering pulse, reverses that pulse once
    to try to recover it before giving up.
    """
    last_hotkey: str | None = None
    last_cmd: str | None = None
    last_turn_ms: int = _FIND_BRICK_CENTER_TURN_MS
    _ref = ref_dist  # updated on each accepted reading

    for act_num in range(1, _FIND_BRICK_CENTER_MAX_ACTS + 1):
        refresh_brick_telemetry(app_state, read_vision=True)
        brick = getattr(app_state.world, "brick", {}) or {}
        visible = bool(brick.get("visible"))
        dist = brick.get("dist")

        # FP filter: reject if >50% further than last accepted (or >750mm on first read).
        if visible and dist is not None:
            if _is_fp_dist(float(dist), _ref):
                ref_label = f"ref={_ref:.0f}mm" if _ref is not None else f">{_PRACTICE_FP_INIT_MAX_MM:.0f}mm"
                print(f"[CENTER] Act {act_num}: dist={float(dist):.0f}mm (false positive, {ref_label}) — stopping.")
                break
            _ref = float(dist)

        if not visible:
            print(f"[CENTER] Brick lost at act {act_num}.")
            if last_hotkey is not None:
                rec_hotkey = "e" if last_hotkey == "q" else "q"
                rec_cmd = "r" if last_cmd == "l" else "l"
                rec_ms = max(100, last_turn_ms // 2)  # half the turn that lost the brick
                print(f"[CENTER] Recovery: reversing last turn ({rec_hotkey}/{rec_cmd}, {rec_ms}ms)...")
                execute_manual_turn_arc_plan(robot=app_state.robot, hotkey=rec_hotkey, cmd=rec_cmd, score=1, hold_duration_ms=rec_ms)
                time.sleep(rec_ms / 1000.0 + 0.1)
                try:
                    app_state.robot.stop()
                except Exception:
                    pass
                refresh_brick_telemetry(app_state, read_vision=True)
                rec_brick = getattr(app_state.world, "brick", {}) or {}
                if bool(rec_brick.get("visible")):
                    print("[CENTER] Brick recovered — continuing centering.")
                    continue  # keep centering from the recovered position
                print("[CENTER] Recovery failed — giving up.")
            break

        x_axis = brick.get("x_axis", brick.get("x")) or 0.0
        dist_str = f"{float(dist):.0f}mm" if dist is not None else "?"
        print(f"[CENTER] Act {act_num}: x_axis={float(x_axis):.0f}mm, dist={dist_str}")
        if abs(float(x_axis)) <= _FIND_BRICK_CENTER_X_TOL_MM:
            print(f"[CENTER] Centered (|x|={abs(float(x_axis)):.0f}mm).")
            break

        # Proportional pulse: larger x offset → longer turn to converge in fewer acts.
        # Gain=2.0 and cap=500ms keeps speed on large offsets without overshooting.
        turn_ms = min(500, max(150, int(abs(float(x_axis)) * 2.0)))
        # Positive x_axis = brick to the right → turn LEFT to center.
        # Negative x_axis = brick to the left → turn RIGHT to center.
        last_hotkey, last_cmd = ("q", "l") if float(x_axis) > 0 else ("e", "r")
        last_turn_ms = turn_ms  # saved so recovery can use half this duration
        execute_manual_turn_arc_plan(robot=app_state.robot, hotkey=last_hotkey, cmd=last_cmd, score=1, hold_duration_ms=turn_ms)
        time.sleep(turn_ms / 1000.0 + 0.1)
        try:
            app_state.robot.stop()
        except Exception:
            pass


def _arc_scan_direction(app_state, *, hotkey: str, cmd: str, label: str) -> bool:
    """Sweep in one direction for up to _PRACTICE_MAX_TURN_MS (2.5 s) total.

    Hard rule: motor never runs more than 2.5 s total for this sweep.
    Polls every 100 ms. Stops on 3 consecutive visible reads, confirms while
    stationary, then centers. Returns False if budget exhausted without finding.
    """
    budget_ms = _PRACTICE_MAX_TURN_MS
    poll_ms = int(_ARC_SCAN_POLL_S * 1000)
    print(f"[FIND_BRICK {label}] Sweeping — budget {budget_ms}ms, poll {poll_ms}ms.")

    _execute_back_then_arc_turn(app_state.robot, cmd=cmd, score=1, hold_duration_ms=budget_ms)

    consec_visible = 0
    scan_ref: float | None = None  # last accepted distance for 50% FP filter
    deadline = time.monotonic() + budget_ms / 1000.0

    while time.monotonic() < deadline:
        refresh_brick_telemetry(app_state, read_vision=True)
        brick = getattr(app_state.world, "brick", {}) or {}
        visible = bool(brick.get("visible"))

        dist = (getattr(app_state.world, "brick", {}) or {}).get("dist")
        if visible and dist is not None and _is_fp_dist(float(dist), scan_ref):
            ref_label = f"ref={scan_ref:.0f}mm" if scan_ref is not None else f">{_PRACTICE_FP_INIT_MAX_MM:.0f}mm"
            print(f"[FIND_BRICK {label}]   filtered dist={float(dist):.0f}mm (fp: {ref_label})")
            visible = False
        elif visible:
            if dist is not None:
                scan_ref = float(dist)
            dist_str = f"{float(dist):.0f}mm" if dist is not None else "?"
            x_raw = brick.get("x_axis", brick.get("x"))
            x_str = f"{float(x_raw):.0f}mm" if x_raw is not None else "?"
            print(f"[FIND_BRICK {label}]   visible dist={dist_str} x={x_str} consec={consec_visible+1}")

        if visible:
            consec_visible += 1
            if consec_visible >= _FIND_BRICK_CONSEC_VISIBLE:
                try:
                    app_state.robot.stop()
                except Exception:
                    pass
                elapsed_ms = budget_ms - (deadline - time.monotonic()) * 1000
                print(
                    f"[FIND_BRICK {label}] {consec_visible} consecutive visible "
                    f"at ~{elapsed_ms:.0f}ms — confirming while stationary..."
                )
                time.sleep(0.15)  # let camera stabilize after motors stop
                confirmed_visible = 0
                last_dist = None
                for _ in range(_FIND_BRICK_CONFIRM_READS):
                    refresh_brick_telemetry(app_state, read_vision=True)
                    brick = getattr(app_state.world, "brick", {}) or {}
                    conf_vis = bool(brick.get("visible"))
                    conf_dist = brick.get("dist")
                    if conf_vis and conf_dist is not None and _is_fp_dist(float(conf_dist), scan_ref):
                        conf_vis = False
                    if conf_vis:
                        if conf_dist is not None:
                            scan_ref = float(conf_dist)
                        confirmed_visible += 1
                        last_dist = conf_dist
                    time.sleep(_ARC_SCAN_POLL_S)
                if confirmed_visible >= 1:
                    dist_str = f"{float(last_dist):.0f}mm" if last_dist is not None else "?"
                    print(f"[FIND_BRICK {label}] Confirmed visible at {dist_str} — centering...")
                    _center_brick_x_axis(app_state, ref_dist=scan_ref)
                    return True
                # Confirmation failed — brick was at frame edge during sweep but gone when stopped.
                # Reverse the sweep by a short pulse to backtrack to the detection point.
                print(f"[FIND_BRICK {label}] Lost in confirmation ({confirmed_visible}/{_FIND_BRICK_CONFIRM_READS}) — backtracking...")
                rev_hotkey = "e" if hotkey == "q" else "q"
                rev_cmd = "r" if cmd == "l" else "l"
                _execute_arc_turn(app_state.robot, cmd=rev_cmd, score=1, hold_duration_ms=200, drive_mode="forward")
                time.sleep(0.2 + 0.1)
                try:
                    app_state.robot.stop()
                except Exception:
                    pass
                retry_visible = 0
                retry_dist = None
                for _ in range(3):
                    refresh_brick_telemetry(app_state, read_vision=True)
                    rb = getattr(app_state.world, "brick", {}) or {}
                    rv = bool(rb.get("visible"))
                    rd = rb.get("dist")
                    if rv and rd is not None and _is_fp_dist(float(rd), scan_ref):
                        rv = False
                    if rv:
                        if rd is not None:
                            scan_ref = float(rd)
                        retry_visible += 1
                        retry_dist = rd
                    time.sleep(_ARC_SCAN_POLL_S)
                if retry_visible >= 1:
                    dist_str = f"{float(retry_dist):.0f}mm" if retry_dist is not None else "?"
                    print(f"[FIND_BRICK {label}] Backtrack recovered brick at {dist_str} — centering...")
                    _center_brick_x_axis(app_state, ref_dist=scan_ref)
                    return True
                print(f"[FIND_BRICK {label}] Backtrack failed — budget spent.")
                return False  # don't keep scanning; budget is conceptually spent
        else:
            consec_visible = 0

        time.sleep(_ARC_SCAN_POLL_S)

    try:
        app_state.robot.stop()
    except Exception:
        pass
    print(f"[FIND_BRICK {label}] {budget_ms}ms budget exhausted — not found.")
    return False


def _run_find_wall_practice(app_state) -> bool:
    """FIND_WALL (practice): scan left until the brick supply is confidently visible.

    If the brick is already visible on startup, succeed immediately. Otherwise
    sweep left — same logic as FIND_BRICK. This reorients the robot at the start
    of every trial without using any production step machinery.
    """
    pose = _prime_vision(app_state)
    print(f"[FIND_WALL PRACTICE] Start: visible={pose.get('visible')}, dist={pose.get('dist_mm')}")
    if pose.get("visible"):
        print("[FIND_WALL PRACTICE] Brick already visible — success.")
        return True

    print(f"[FIND_WALL PRACTICE] Brick not visible — scanning (budget {_PRACTICE_MAX_TURN_MS}ms each direction).")
    # Left sweep first, then right as fallback. Each sweep is capped at 2.5 s.
    if _arc_scan_direction(app_state, hotkey="q", cmd="l", label="WALL-LEFT"):
        return True
    if _arc_scan_direction(app_state, hotkey="e", cmd="r", label="WALL-RIGHT"):
        return True
    print("[FIND_WALL PRACTICE] Could not find brick — failed.")
    return False


def _run_find_brick_with_arc_scan(app_state) -> bool:
    """FIND_BRICK: sweep left then right, each capped at _PRACTICE_MAX_TURN_MS (2.5 s).

    EXIT_WALL turned right, so Phase 1 sweeps left. Phase 2 sweeps right as fallback.
    """
    print(f"[FIND_BRICK ARC] Budget {_PRACTICE_MAX_TURN_MS}ms per sweep direction.")

    # If the brick is already visible and genuinely close, skip the sweep.
    # Use a tight threshold (400mm) so distant noise doesn't short-circuit the sweep.
    pose = _prime_vision(app_state)
    if pose.get("visible"):
        dist_now = pose.get("dist_mm")
        dist_val = float(dist_now) if dist_now is not None else None
        if dist_val is not None and dist_val <= _PRACTICE_FP_INIT_MAX_MM:
            print(f"[FIND_BRICK ARC] Brick already visible at {dist_val:.0f}mm — centering (skipping sweep).")
            _center_brick_x_axis(app_state, ref_dist=dist_val)
            return True
        elif dist_val is not None:
            print(f"[FIND_BRICK ARC] Visible at {dist_val:.0f}mm but looks like noise — sweeping.")

    print("[FIND_BRICK ARC] Phase 1: sweeping LEFT...")
    if _arc_scan_direction(app_state, hotkey="q", cmd="l", label="LEFT"):
        return True

    print("[FIND_BRICK ARC] Left exhausted — Phase 2: sweeping RIGHT...")
    if _arc_scan_direction(app_state, hotkey="e", cmd="r", label="RIGHT"):
        return True

    print("[FIND_BRICK ARC] Both sweeps exhausted — failed.")
    return False


def _run_align_brick_practice(app_state) -> bool:
    """ALIGN_BRICK (practice): custom aligner outside production step machinery.

    When the brick is not visible, turns left to re-acquire (since EXIT_WALL
    turned right). When visible, centers x_axis then closes distance.
    Hard limit: _ALIGN_BRICK_MAX_ACTS acts total.
    """
    from telemetry_robot import speed_power_pwm_for_cmd as _sppc

    _, fwd_pwm, _, _ = _sppc("f", 1)
    _, bwd_pwm, _, _ = _sppc("b", 1)

    print(f"[ALIGN_BRICK PRACTICE] Starting — target {_ALIGN_BRICK_TARGET_DIST_MM:.0f}mm, {_ALIGN_BRICK_MAX_ACTS} acts max.")

    _prev_dist = None  # track last accepted distance to detect detector switching targets

    for act_num in range(1, _ALIGN_BRICK_MAX_ACTS + 1):
        refresh_brick_telemetry(app_state, read_vision=True)
        brick = getattr(app_state.world, "brick", {}) or {}
        visible = bool(brick.get("visible"))
        dist = brick.get("dist")
        x_raw = brick.get("x_axis", brick.get("x"))

        # Distance filter — reject extreme false positives.
        if visible and dist is not None and float(dist) > _PRACTICE_MAX_DETECT_DIST_MM:
            visible = False

        # Jump filter — if distance is >50% further than the previous accepted reading,
        # the detector likely switched to a different (farther) target.
        if visible and dist is not None and _prev_dist is not None:
            if float(dist) > float(_prev_dist) * _PRACTICE_FP_RATIO:
                print(f"[ALIGN_BRICK PRACTICE] Act {act_num}: dist jumped {_prev_dist:.0f}→{float(dist):.0f}mm (>{_PRACTICE_FP_RATIO:.0%} of prev) — false positive, ignoring.")
                visible = False

        if not visible:
            # Pause briefly and let vision re-acquire — do NOT turn, as turning
            # rotates the robot sideways and inflates subsequent distance readings.
            print(f"[ALIGN_BRICK PRACTICE] Act {act_num}: not visible — waiting for re-acquire.")
            time.sleep(0.3)
            continue

        dist_val = float(dist)
        x_val = float(x_raw or 0.0)
        dist_err = dist_val - _ALIGN_BRICK_TARGET_DIST_MM
        _prev_dist = dist_val  # update accepted distance for jump detection
        print(f"[ALIGN_BRICK PRACTICE] Act {act_num}: dist={dist_val:.0f}mm err={dist_err:+.0f}mm, x={x_val:.0f}mm")

        # Success check.
        if abs(dist_err) <= _ALIGN_BRICK_DIST_TOL_MM and abs(x_val) <= _ALIGN_BRICK_X_TOL_MM:
            print(f"[ALIGN_BRICK PRACTICE] Aligned!")
            return True

        # Center x_axis first before closing distance.
        if abs(x_val) > _ALIGN_BRICK_X_TOL_MM:
            hotkey, cmd = ("q", "l") if x_val > 0 else ("e", "r")
            # Proportional pulse: gain=2.0, cap=500ms — fast on large offsets, no overshoot.
            align_turn_ms = min(500, max(150, int(abs(x_val) * 2.0)))
            print(f"[ALIGN_BRICK PRACTICE]   Centering x={x_val:.0f}mm → {hotkey} ({align_turn_ms}ms).")
            execute_manual_turn_arc_plan(
                robot=app_state.robot, hotkey=hotkey, cmd=cmd, score=1,
                hold_duration_ms=align_turn_ms,
            )
            time.sleep(align_turn_ms / 1000.0 + 0.1)
        elif dist_err > 0:
            print(f"[ALIGN_BRICK PRACTICE]   Approaching — {dist_err:.0f}mm to close.")
            app_state.robot.send_command_pwm("f", int(fwd_pwm), duration_ms=_ALIGN_BRICK_APPROACH_MS)
            time.sleep(_ALIGN_BRICK_APPROACH_MS / 1000.0 + 0.1)
        else:
            print(f"[ALIGN_BRICK PRACTICE]   Too close — backing up {-dist_err:.0f}mm.")
            app_state.robot.send_command_pwm("b", int(bwd_pwm), duration_ms=_ALIGN_BRICK_APPROACH_MS)
            time.sleep(_ALIGN_BRICK_APPROACH_MS / 1000.0 + 0.1)

        try:
            app_state.robot.stop()
        except Exception:
            pass

    print(f"[ALIGN_BRICK PRACTICE] Hit {_ALIGN_BRICK_MAX_ACTS}-act limit — failed.")
    return False


def run_single_auto_step(*, step_token: str, vision_mode: str | None):
    step_obj = resolve_step_argument(step_token)
    if step_obj is None:
        return {"ok": False, "error": f"unknown_step:{step_token}"}

    app_state, mode_norm = _build_app_state(vision_mode)
    try:
        pose_before = _prime_vision(app_state)
        print(
            "[AUTO STEP CLI] Starting "
            f"{step_obj.value} with {mode_norm} vision "
            f"(visible={pose_before.get('visible')}, "
            f"dist={pose_before.get('dist_mm')}, x_axis={pose_before.get('x_axis_mm')})."
        )
        ok = bool(run_auto_step(app_state, step_obj))
        step_key = str(getattr(step_obj, "value", step_obj))
        action_history = list(
            ((getattr(app_state, "session_step_actions_by_step", {}) or {}).get(step_key) or [])
        )
        actions_required = int(action_history[-1]) if action_history else 0
        return {
            "ok": bool(ok),
            "step": step_obj.value,
            "vision_mode": str(mode_norm),
            "pose_before": pose_before,
            "actions_required": int(actions_required),
            "run_log_path": None if getattr(app_state, "log_path", None) is None else str(app_state.log_path),
        }
    finally:
        _teardown_app_state(app_state)


def run_step_sequence(
    *,
    steps: list[str],
    trials: int,
    vision_mode: str | None,
    pause_between_trials_s: float = 1.0,
):
    """Run a sequence of steps N times, reusing a single robot+vision session.

    PRACTICE LOOP TRIAL/RUN SEMANTICS (see module docstring for full definitions):

    TRIAL NUMBERING AND PICK/PLACE ALTERNATION (RULES 2 & 4):
      Trials are 1-indexed. Odd trials (1, 3, 5, ...) are PICK trials — the
      robot approaches a brick and picks it up. Even trials (2, 4, 6, ...) are
      PLACE trials — the robot carries and places the brick it picked up on the
      previous trial. This alternation is strict: you cannot place without first
      picking, and you cannot pick twice in a row without placing in between.
      TODO: enforce pick/place mode selection per trial number here.

    FULL RUN = 100 TRIALS (RULE 3):
      Pass --trials 100 on the command line for a complete practice run.
      That gives 50 pick trials (odd) and 50 place trials (even). Running
      fewer trials is fine for debugging, but is not a "full run."

    THIS IS NOT PRODUCTION (see module docstring):
      This function drives the simplified practice implementations, not the
      production telemetry pipeline. Do not add production logic here.
    """
    step_objs = []
    for token in steps:
        obj = resolve_step_argument(token)
        if obj is None:
            return {"ok": False, "error": f"unknown_step:{token}"}
        step_objs.append(obj)

    app_state, mode_norm = _build_app_state(vision_mode)
    step_names = ", ".join(o.value for o in step_objs)
    print(f"[AUTO STEP CLI] Starting {trials}-trial loop: {step_names} ({mode_norm} vision)")

    trial_results = []
    try:
        for trial_num in range(1, int(trials) + 1):
            print(f"\n[AUTO STEP CLI] --- Trial {trial_num}/{trials} ---")
            # Ensure we are facing the brick before starting. If not visible
            # (robot drifted from previous failed trial), do an orientation sweep
            # so EXIT_WALL has something to exit from.
            refresh_brick_telemetry(app_state, read_vision=True)
            _pre_brick = getattr(app_state.world, "brick", {}) or {}
            _pre_vis = bool(_pre_brick.get("visible"))
            _pre_dist = _pre_brick.get("dist")
            if not _pre_vis or (_pre_dist is not None and float(_pre_dist) > _PRACTICE_MAX_DETECT_DIST_MM):
                print("[AUTO STEP CLI] Brick not visible at trial start — orienting...")
                _run_find_brick_with_arc_scan(app_state)
            _backup_if_too_close(app_state)
            step_results = []
            trial_ok = True
            for step_obj in step_objs:
                pose_before = _prime_vision(app_state)
                print(
                    f"[AUTO STEP CLI] {step_obj.value} "
                    f"(visible={pose_before.get('visible')}, dist={pose_before.get('dist_mm')})"
                )
                step_name = str(getattr(step_obj, "value", "")).upper()
                if step_name == "FIND_WALL":
                    ok = _run_find_wall_practice(app_state)
                elif step_name == "EXIT_WALL":
                    ok = _run_exit_wall_with_arc_turns(app_state)
                elif step_name == "FIND_BRICK":
                    ok = _run_find_brick_with_arc_scan(app_state)
                elif step_name == "ALIGN_BRICK":
                    ok = _run_align_brick_practice(app_state)
                else:
                    ok = bool(run_auto_step(app_state, step_obj))
                step_key = str(getattr(step_obj, "value", step_obj))
                action_history = list(
                    ((getattr(app_state, "session_step_actions_by_step", {}) or {}).get(step_key) or [])
                )
                actions_required = int(action_history[-1]) if action_history else 0
                step_results.append({
                    "step": step_obj.value,
                    "ok": ok,
                    "actions_required": actions_required,
                })
                if not ok:
                    trial_ok = False
                    print(f"[AUTO STEP CLI] {step_obj.value} FAILED — continuing to next trial.")
                    break

            trial_results.append({"trial": trial_num, "ok": trial_ok, "steps": step_results})
            successes_so_far = sum(1 for r in trial_results if r["ok"])
            print(
                f"[AUTO STEP CLI] Trial {trial_num} {'OK' if trial_ok else 'FAILED'} "
                f"— {successes_so_far}/{trial_num} succeeded so far."
            )

            if trial_num < trials and pause_between_trials_s > 0:
                time.sleep(pause_between_trials_s)

    finally:
        _teardown_app_state(app_state)

    ok_count = sum(1 for r in trial_results if r["ok"])
    return {
        "ok": ok_count == len(trial_results),
        "trials": len(trial_results),
        "successes": ok_count,
        "failures": len(trial_results) - ok_count,
        "steps": step_names,
        "vision_mode": mode_norm,
        "results": trial_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one or more auto-steps from the CLI.")
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Single step number or name (e.g. 7, align_brick). Default: 7",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step numbers/names to run in sequence (e.g. 1,2).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of times to repeat the step sequence. Default: 1",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=1.0,
        help="Pause between trials in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--vision",
        type=str,
        choices=("auto", "cyan", "yolo", "leia", "aruco"),
        default="auto",
        help="Vision mode. Default: auto",
    )
    args = parser.parse_args()

    vision_mode = None if str(args.vision).strip().lower() == "auto" else str(args.vision)

    if args.steps or (args.trials and args.trials > 1):
        steps_raw = args.steps or args.step or "7"
        steps = [s.strip() for s in steps_raw.split(",") if s.strip()]
        result = run_step_sequence(
            steps=steps,
            trials=max(1, int(args.trials or 1)),
            vision_mode=vision_mode,
            pause_between_trials_s=max(0.0, float(args.pause_s)),
        )
    else:
        result = run_single_auto_step(
            step_token=str(args.step or "7"),
            vision_mode=vision_mode,
        )

    print(json.dumps(result, indent=2))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
