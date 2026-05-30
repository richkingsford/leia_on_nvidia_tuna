#!/usr/bin/env python3
"""
siori_step2_trial.py — Calibration-driven, phase-separated alignment controller.

KEY CALIBRATION INSIGHT (measured 2026-05-29 at 266mm operating distance):
  - drive forward closes dist at ~0.024 mm/ms  BUT shoves y UP at ~0.018 mm/ms
  - mast corrects y at only ~0.003 mm/ms  (6x slower than wheels push it)
  - mast is ~orthogonal to dist & x (mast moves them ~0 mm/ms)
  => You CANNOT hold y while driving. So separate the axes IN TIME:
       Phase 1 APPROACH : close dist + x with bias turns (ignore y)
       Phase 2 Y-SETTLE : stop wheels, drive y home with calibrated mast pulses
       Phase 3 CONFIRM  : re-check; micro-correct whichever axis drifted; win

Turn/bias moves reuse the PROVEN bang-bang primitives (calibrated curves, 4/5),
following the user's rule: close dist+x with gentle/strong turns; x-only acts
are rare (only when dist is already in the gate).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import a_follow_the_brick as follow
from helper_brick_detector_native_oak import BrickDetector
from helper_robot_control import Robot

# ── Win gate ──────────────────────────────────────────────────────────────────
STOP_OFFSET_MM = 216.8
DIST_TOL_MM    = 50.0
X_TARGET_MM    = 5.9
X_TOL_MM       = 5.0     # widened from 3 — achievable given min-arc granularity, fine for picking
Y_TARGET_MM    = -42.7
Y_TOL_MM       = 5.0

# ── Calibration (mm per ms), measured at operating distance ───────────────────
DIST_RATE   = 0.024      # forward/back drive: dist mm per ms
TURN_X_RATE = 0.16       # arc/bias turn: x mm per ms (from validated left/right curve)
MAST_RATE   = 0.003      # mast: y mm per ms
DAMP        = 0.65       # undershoot factor → converge without overshoot

# ── Move duration caps ────────────────────────────────────────────────────────
DRIVE_MIN_MS, DRIVE_MAX_MS = 120, 300
TURN_MIN_MS,  TURN_MAX_MS  = 130, 260
XONLY_MS                   = 140        # rare fine x pivot when dist already in gate
MAST_MIN_MS,  MAST_MAX_MS  = 300, 1200
MAST_SETTLE_S              = 0.45       # coast settle after mast (orthogonal, so safe)
DRIVE_SETTLE_S             = 0.25

# ── Loop ──────────────────────────────────────────────────────────────────────
MAX_TRIAL_S      = 28.0
CONFIRM_FRAMES   = 2
Y_SETTLE_MAX_ACTS = 8
PREGAME_MAST_UP_MS = 1500
PREGAME_VIS_TIMEOUT_S = 6.0

OUT_PATH = Path("runs/siori_step2.jsonl")


def _read(vision) -> dict | None:
    r = follow._read_brick_measurement(vision, jump_guard=True)
    return r if r.get("confident") else None


def _errs(r: dict):
    return (float(r["dist_mm"]) - STOP_OFFSET_MM,
            float(r["x_mm"]) - X_TARGET_MM,
            float(r["y_mm"]) - Y_TARGET_MM)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _drive(robot, cmd, ms):
    robot.send_command_pwm(cmd, 103, duration_ms=int(ms))
    time.sleep(ms / 1000.0 + DRIVE_SETTLE_S)
    follow._stop_robot(robot)


def _mast(robot, cmd, ms):
    robot.send_command_pwm(cmd, 255, duration_ms=int(ms))
    time.sleep(ms / 1000.0 + MAST_SETTLE_S)
    follow._stop_robot(robot)


def _bias(robot, turn_cmd, drive_mode, strength, ms, reading):
    follow._send_drive_bias(
        robot, turn_cmd=turn_cmd, drive_mode=drive_mode, strength=strength,
        duration_ms=int(ms), reading=reading, context="siori2_bias",
    )
    time.sleep(ms / 1000.0 + DRIVE_SETTLE_S)
    follow._stop_robot(robot)


def _x_turn_cmd(x_err: float) -> str:
    # positive x_err → turn right (per empty profile positive_error_turn_cmd='r')
    return "r" if x_err > 0 else "l"


# ─────────────────────────────────────────────────────────────────────────────
def _pregame(vision, robot, t) -> bool:
    r = _read(vision)
    if r is None:
        print(f"[S2] T{t} pregame: not visible — mast up {PREGAME_MAST_UP_MS}ms", flush=True)
        _mast(robot, "u", PREGAME_MAST_UP_MS)
        dl = time.monotonic() + PREGAME_VIS_TIMEOUT_S
        while time.monotonic() < dl:
            r = _read(vision)
            if r:
                break
            time.sleep(0.15)
    if r is None:
        print(f"[S2] T{t} pregame: brick not found", flush=True)
        return False
    d, x, y = _errs(r)
    print(f"[S2] T{t} pregame ok: dist={d:+.0f} x={x:+.0f} y={y:+.0f}", flush=True)
    return True


def _run_trial(vision, robot, t) -> dict:
    deadline = time.monotonic() + MAX_TRIAL_S
    phase = "APPROACH"
    confirm = 0
    moves = 0
    first = None
    last = {}
    y_acts = 0

    while time.monotonic() < deadline:
        r = _read(vision)
        if r is None:
            follow._stop_robot(robot)
            time.sleep(0.1)
            continue
        last = dict(r)
        d, x, y = _errs(r)
        if first is None:
            first = {"dist_err": round(d, 1), "x_err": round(x, 1), "y_err": round(y, 1)}

        dist_ok = abs(d) <= DIST_TOL_MM
        x_ok    = abs(x) <= X_TOL_MM
        y_ok    = abs(y) <= Y_TOL_MM

        # ── WIN ──
        if dist_ok and x_ok and y_ok:
            confirm += 1
            follow._stop_robot(robot)
            print(f"[S2] T{t} HAPPY #{confirm} dist={d:+.1f} x={x:+.1f} y={y:+.1f}", flush=True)
            if confirm >= CONFIRM_FRAMES:
                return {"trial": t, "won": True, "moves": moves, "first": first,
                        "final": {"dist_err": round(d, 1), "x_err": round(x, 1), "y_err": round(y, 1)},
                        "failure": None}
            time.sleep(0.1)
            continue
        confirm = 0

        # ── PHASE 1: APPROACH (dist + x, ignore y) ──
        if not (dist_ok and x_ok):
            phase = "APPROACH"
            if not x_ok and not dist_ok:
                # bias turn closes BOTH dist and x
                turn = _x_turn_cmd(x)
                drive_mode = "forward" if d > 0 else "backward"
                strength = "strong" if abs(x) > 15 else "gentle"
                ms = _clamp(DAMP * abs(x) / TURN_X_RATE, TURN_MIN_MS, TURN_MAX_MS)
                print(f"[S2] T{t} APPROACH BIAS_{turn.upper()}/{strength}/{drive_mode} {ms:.0f}ms "
                      f"dist={d:+.1f} x={x:+.1f} y={y:+.1f}", flush=True)
                _bias(robot, turn, drive_mode, strength, ms, r)
            elif not dist_ok:
                # x is fine → straight drive to close dist
                cmd = "f" if d > 0 else "b"
                ms = _clamp(DAMP * abs(d) / DIST_RATE, DRIVE_MIN_MS, DRIVE_MAX_MS)
                print(f"[S2] T{t} APPROACH DRIVE_{cmd.upper()} {ms:.0f}ms "
                      f"dist={d:+.1f} x={x:+.1f} y={y:+.1f}", flush=True)
                _drive(robot, cmd, ms)
            else:
                # dist in gate, x out → RARE x-only gentle pivot
                turn = _x_turn_cmd(x)
                print(f"[S2] T{t} APPROACH XONLY_{turn.upper()} {XONLY_MS}ms (rare) "
                      f"dist={d:+.1f} x={x:+.1f} y={y:+.1f}", flush=True)
                _bias(robot, turn, "forward", "gentle", XONLY_MS, r)
            moves += 1
            continue

        # ── PHASE 2: Y-SETTLE (stationary mast) ──
        if not y_ok:
            phase = "Y_SETTLE"
            cmd = "d" if y > 0 else "u"
            ms = _clamp(DAMP * abs(y) / MAST_RATE, MAST_MIN_MS, MAST_MAX_MS)
            print(f"[S2] T{t} Y_SETTLE MAST_{cmd.upper()} {ms:.0f}ms y={y:+.1f} "
                  f"(dist={d:+.1f} x={x:+.1f})", flush=True)
            _mast(robot, cmd, ms)
            moves += 1
            y_acts += 1
            if y_acts > Y_SETTLE_MAX_ACTS:
                print(f"[S2] T{t} y-settle exceeded {Y_SETTLE_MAX_ACTS} acts", flush=True)
                # fall through; let loop re-evaluate (maybe dist/x drifted)
                y_acts = 0
            continue

    # timed out
    d, x, y = _errs(last) if last else (999, 999, 999)
    if abs(y) > Y_TOL_MM and abs(y) >= abs(d) and abs(y) >= abs(x):
        failure = "y_gap_not_closed"
    elif abs(d) > DIST_TOL_MM:
        failure = "dist_gap_not_closed"
    elif abs(x) > X_TOL_MM:
        failure = "x_gap_not_closed"
    else:
        failure = "timeout_near_gate"
    return {"trial": t, "won": False, "moves": moves, "first": first,
            "final": {"dist_err": round(d, 1), "x_err": round(x, 1), "y_err": round(y, 1)},
            "failure": failure}


def _reset(vision, robot):
    _mast(robot, "u", 400)
    follow._set_game_profile("empty")
    base = follow._reset_motion_config()
    cfg = copy.deepcopy(base)
    rev = cfg.get("reverse_turn") if isinstance(cfg.get("reverse_turn"), dict) else {}
    rev["post_pause_s"] = 0.2
    rev["settle_s"] = 0.0
    st = rev.get("straight_back_first") if isinstance(rev.get("straight_back_first"), dict) else {}
    try:
        orig = int(round(float(st.get("duration_ms", 3000))))
    except (TypeError, ValueError):
        orig = 3000
    st["duration_ms"] = max(600, int(round(orig * 0.30)))
    rev["straight_back_first"] = st
    rev["dist_target_mm"] = STOP_OFFSET_MM + 75.0
    rev["dist_tol_mm"] = 85.0
    rev["x_offset_min_mm"] = 5.0
    rev["x_offset_max_mm"] = 35.0
    rev["target_abs_x_mm"] = 15.0
    mu = cfg.get("mast_up") if isinstance(cfg.get("mast_up"), dict) else {}
    mu["enabled"] = False
    cfg["mast_up"] = mu
    cfg["reverse_turn"] = rev
    old = follow._reset_motion_config
    follow._reset_motion_config = lambda: copy.deepcopy(cfg)
    try:
        follow._run_reset_sequence(vision, robot, rng=random)
    finally:
        follow._reset_motion_config = old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    follow._set_game_profile("empty")
    vision = BrickDetector(debug=True)
    robot = Robot()
    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        wins = 0
        with out.open("w", encoding="utf-8") as fh:
            for t in range(1, args.trials + 1):
                print(f"\n[S2] ═══ Trial {t}/{args.trials} ═══", flush=True)
                if not _pregame(vision, robot, t):
                    rec = {"trial": t, "won": False, "moves": 0, "first": None,
                           "final": None, "failure": "pregame_no_visibility"}
                else:
                    rec = _run_trial(vision, robot, t)
                wins += 1 if rec["won"] else 0
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()
                print(f"[S2] T{t}: {'WIN ✓' if rec['won'] else 'MISS ✗'} "
                      f"failure={rec.get('failure')} moves={rec.get('moves')} "
                      f"final={rec.get('final')}", flush=True)
                if t < args.trials:
                    _reset(vision, robot)
        print(f"\n[S2] DONE wins={wins}/{args.trials}  data={out}", flush=True)
        return 0
    finally:
        follow._stop_robot(robot)
        try: robot.close()
        except Exception: pass
        try: vision.close()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
