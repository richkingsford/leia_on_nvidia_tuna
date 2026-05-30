#!/usr/bin/env python3
"""
siori_step1_trial.py  —  Siori's Astolfi-style PD controller for Step 1 alignment.

Implements Siori's exact formulation:
  head_err  = atan2(-x_off, dist)
  dist_err  = distance - STOP_OFFSET
  v         = (Kp_d*dist_err_m + Kd_d*d_dist_err_m) * cos(head_err)
  omega     = Kp_h*head_err + Kd_h*d_head_err
  left      = v - omega    right = v + omega
  + coupled proportional saturation (curvature preserved under clipping)
  + zeta=1.5 overdamped  →  zero-overshoot guaranteed

Compartmentalised: does NOT modify any existing file.
Usage:
  python siori_step1_trial.py [--trials 5] [--out runs/siori_XXX.jsonl]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import a_follow_the_brick as follow
from helper_astolfi_controller import AstolfiState, astolfi_wheel_command, overdamped_pd_gains
from helper_brick_detector_native_oak import BrickDetector
from helper_brick_visibility_safety import guarded_send_custom_actions_pwm
from helper_robot_control import Robot

# ── Win gate (matches bang-bang baseline) ─────────────────────────────────────
STOP_OFFSET_MM = 216.8   # dist target (gate centre)
DIST_TOL_MM    = 50.0    # ±50mm  (current proven working tolerance)
X_TARGET_MM    = 5.9
X_TOL_MM       = 3.0
Y_TARGET_MM    = -42.7
Y_TOL_MM       = 5.0     # tight ±5mm zone throughout

# ── PD gains (analytically tuned, zeta=1.5 = slightly overdamped) ────────────
ZETA            = 1.5
DIST_SETTLE_S   = 1.5    # target settling time for distance axis
HEAD_SETTLE_S   = 2.0    # target settling time for heading axis

# ── Wheel command mapping ─────────────────────────────────────────────────────
CONTROL_HZ      = 20.0   # 50ms per tick
MAX_WHEEL_MS    = 300    # max pulse duration per tick
MIN_WHEEL_MS    = 100    # min pulse duration (motor engage floor)
DEADBAND        = 0.03   # very low — must respond to tiny heading errors at close range
LINEAR_LIMIT    = 0.85   # v saturation
ANGULAR_LIMIT   = 0.50   # omega saturation  (keeps turns moderate)
WHEEL_LIMIT     = 1.0    # overall wheel scale
# Deriv smoothing and clamp
DERIV_ALPHA     = 0.35
MAX_DIST_DERIV  = 0.12   # m/s
MAX_HEAD_DERIV  = 0.25   # rad/s
# When further than this from target, suppress counter-rotation (no fishtailing)
SAME_DIR_GAP_MM  = 15.0   # pure arcs only when within 15mm of dist target
MIN_BIAS_SPEED   = 0.15   # minimum wheel speed on trailing wheel for bias turns

# ── Wide-x override: when x is badly off, use bang-bang bias first ─────────────
X_BIAS_THR_MM   = 35.0   # if |x_err| > this, issue a bias-override pulse
X_BIAS_MS       = 250    # duration of bias-override pulse

# ── Y axis (mast) — minimal movement philosophy ───────────────────────────────
Y_PRIORITY_DIST_MM = 999.0  # always available to correct y when x is settled
Y_PRIORITY_X_MM    = 10.0   # and within this mm of x target
Y_MAST_PULSE_MIN   = 400    # ms
Y_MAST_PULSE_MAX   = 700    # ms
Y_MAST_FULL_GAP    = 8.0    # mm gap that maps to max pulse
Y_MAST_SETTLE_S    = 0.50   # coast settle after mast move (longer = more stable reading)

# ── Pregame recovery ──────────────────────────────────────────────────────────
PREGAME_MAST_UP_MS     = 800    # SHORT — only if brick not visible; keeps y near target
PREGAME_VIS_TIMEOUT_S  = 5.0    # wait for brick visibility
PREGAME_Y_MAX_ACTS     = 10     # mast acts to settle y before trial
PREGAME_Y_BAND_MM      = 5.0    # target: within ±5mm of y_target
PREGAME_Y_PULSE_MS     = 350    # short pulse — less overshoot during y settle

# ── Trial timing ─────────────────────────────────────────────────────────────
MAX_TRIAL_S     = 20.0
CONFIRM_FRAMES  = 2      # consecutive happy reads needed for win

# ── Output ────────────────────────────────────────────────────────────────────
OUT_PATH = Path("runs/siori_step1_trial.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gains():
    return overdamped_pd_gains(
        damping_ratio=ZETA,
        dist_settle_time_s=DIST_SETTLE_S,
        heading_settle_time_s=HEAD_SETTLE_S,
    )


def _wheel_action(target: str, cmd: float) -> dict:
    """Map a normalised wheel command to a motor action dict."""
    if abs(cmd) <= 0.0:
        return {"target": target, "action": "s", "pwm": 0, "duration_ms": 0}
    # Left tread wire is inverted relative to right.
    if target == "l":
        action = "b" if cmd > 0.0 else "f"
    else:
        action = "f" if cmd > 0.0 else "b"
    return {"target": target, "action": action, "pwm": 103}


def _build_wheel_packet(left: float, right: float) -> tuple[list[dict], int, str]:
    """
    Convert normalised left/right to motor action packet.
    Enforces arc-assist (no tank turns): if signs differ, zero the weaker wheel.
    Returns (actions, max_duration_ms, human_label).
    """
    # Enforce no-tank-turn rule: convert to arc-assist if signs differ
    if left * right < 0.0:
        if abs(left) >= abs(right):
            right = 0.0
        else:
            left = 0.0

    actions, durations = [], []
    for target, val in (("l", left), ("r", right)):
        mag = abs(val) / max(WHEEL_LIMIT, 1e-9)
        if mag < DEADBAND:
            actions.append({"target": target, "action": "s", "pwm": 0, "duration_ms": 0})
        else:
            dur = int(round(MIN_WHEEL_MS + (MAX_WHEEL_MS - MIN_WHEEL_MS) * min(1.0, mag)))
            a = _wheel_action(target, val)
            a["duration_ms"] = max(MIN_WHEEL_MS, dur)
            a["pwm"] = 103
            actions.append(a)
            durations.append(a["duration_ms"])

    max_ms = max(durations) if durations else 0

    # Human label
    l_act = next((a["action"] for a in actions if a["target"] == "l"), "s")
    r_act = next((a["action"] for a in actions if a["target"] == "r"), "s")
    if l_act == "s" and r_act == "s":
        label = "STOP"
    elif l_act == "s":
        label = "ARC_R" if r_act == "f" else "ARC_R_BCK"
    elif r_act == "s":
        label = "ARC_L" if l_act == "b" else "ARC_L_BCK"
    elif l_act == "b" and r_act == "f":
        label = "FWD"
    elif l_act == "f" and r_act == "b":
        label = "BCK"
    else:
        label = f"DIFF(l={l_act},r={r_act})"

    return actions, max_ms, label


def _happy(reading: dict) -> bool:
    try:
        de = abs(float(reading["dist_mm"]) - STOP_OFFSET_MM)
        xe = abs(float(reading["x_mm"]) - X_TARGET_MM)
        ye = abs(float(reading["y_mm"]) - Y_TARGET_MM)
    except (TypeError, ValueError, KeyError):
        return False
    return de <= DIST_TOL_MM and xe <= X_TOL_MM and ye <= Y_TOL_MM


def _snap(reading: dict) -> dict:
    out = {}
    for k in ("dist_mm", "x_mm", "y_mm", "conf", "confident"):
        out[k] = reading.get(k)
    try:
        out["dist_err"] = round(float(reading["dist_mm"]) - STOP_OFFSET_MM, 2)
        out["x_err"]    = round(float(reading["x_mm"]) - X_TARGET_MM, 2)
        out["y_err"]    = round(float(reading["y_mm"]) - Y_TARGET_MM, 2)
    except (TypeError, ValueError, KeyError):
        pass
    return out


def _y_mast_pulse_ms(y_err: float) -> int:
    gap = max(0.0, abs(y_err) - Y_TOL_MM)
    frac = min(1.0, gap / Y_MAST_FULL_GAP)
    return int(round(Y_MAST_PULSE_MIN + frac * (Y_MAST_PULSE_MAX - Y_MAST_PULSE_MIN)))


# ─────────────────────────────────────────────────────────────────────────────
# Pregame: robust brick visibility + y settle
# ─────────────────────────────────────────────────────────────────────────────

def _pregame(vision: BrickDetector, robot: Robot, trial_n: int) -> bool:
    """
    1. Unconditionally raise mast (clears blockage, user-validated).
    2. Wait for brick visibility.
    3. Settle y within PREGAME_Y_BAND_MM.
    Returns True if brick is visible after pregame.
    """
    # Only raise mast if brick not already visible — keeps y near target
    r_check = follow._read_brick_measurement(vision, jump_guard=False)
    if not r_check.get("confident"):
        print(f"[SIORI] T{trial_n} pregame: brick not visible — mast up {PREGAME_MAST_UP_MS}ms", flush=True)
        robot.send_command_pwm("u", 255, duration_ms=PREGAME_MAST_UP_MS)
        time.sleep(PREGAME_MAST_UP_MS / 1000.0 + 0.3)
    else:
        print(f"[SIORI] T{trial_n} pregame: brick already visible, skipping mast-up", flush=True)

    # Wait for visibility
    deadline = time.monotonic() + PREGAME_VIS_TIMEOUT_S
    while time.monotonic() < deadline:
        r = follow._read_brick_measurement(vision, jump_guard=False)
        if r.get("confident"):
            break
        time.sleep(0.15)
    else:
        print(f"[SIORI] T{trial_n} pregame: brick not visible after {PREGAME_VIS_TIMEOUT_S}s", flush=True)
        return False

    # Y settle (minimal — ±PREGAME_Y_BAND_MM is fine to start)
    prev_abs_err = None
    for act_i in range(PREGAME_Y_MAX_ACTS):
        r = follow._read_brick_measurement(vision, jump_guard=False)
        if not r.get("confident"):
            break
        try:
            y_err = float(r["y_mm"]) - Y_TARGET_MM
        except (TypeError, ValueError, KeyError):
            break
        abs_err = abs(y_err)
        if abs_err <= PREGAME_Y_BAND_MM:
            print(f"[SIORI] T{trial_n} pregame: y settled y_err={y_err:+.1f}mm after {act_i} acts", flush=True)
            break
        # Stop if getting worse
        if prev_abs_err is not None and abs_err > prev_abs_err + 2.0:
            print(f"[SIORI] T{trial_n} pregame: y worsening ({prev_abs_err:.1f}→{abs_err:.1f}mm), stopping", flush=True)
            break
        cmd = "d" if y_err > 0.0 else "u"
        ms = PREGAME_Y_PULSE_MS
        print(f"[SIORI] T{trial_n} pregame: y act {act_i+1}/{PREGAME_Y_MAX_ACTS} cmd={cmd} {ms}ms y_err={y_err:+.1f}mm", flush=True)
        robot.send_command_pwm(cmd, 255, duration_ms=ms)
        time.sleep(ms / 1000.0 + Y_MAST_SETTLE_S)
        prev_abs_err = abs_err

    r = follow._read_brick_measurement(vision, jump_guard=False)
    if r.get("confident"):
        try:
            print(f"[SIORI] T{trial_n} pregame done: dist={r['dist_mm']:.1f} x={r['x_mm']:+.1f} y={r['y_mm']:+.1f}", flush=True)
        except (TypeError, ValueError, KeyError):
            pass
        return True
    print(f"[SIORI] T{trial_n} pregame: brick lost during y settle", flush=True)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Trial: 20 Hz Astolfi PD control loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(vision: BrickDetector, robot: Robot, trial_n: int) -> dict:
    gains  = _gains()
    state  = AstolfiState()
    dt_s   = 1.0 / CONTROL_HZ
    deadline = time.monotonic() + MAX_TRIAL_S
    ticks, moves, happy_frames = 0, 0, 0
    first = None
    last_reading: dict = {}
    last_label = "—"
    last_control: dict = {}
    ticks_since_mast = 999  # suppress y correction just after mast move

    print(f"[SIORI] T{trial_n} loop start  gains kp_d={gains.kp_d:.2f} kd_d={gains.kd_d:.2f} "
          f"kp_h={gains.kp_h:.2f} kd_h={gains.kd_h:.2f}", flush=True)

    while time.monotonic() < deadline:
        tick_start = time.monotonic()
        reading = follow._read_brick_measurement(vision, jump_guard=True)
        last_reading = dict(reading)

        if not reading.get("confident"):
            follow._stop_robot(robot)
            time.sleep(dt_s)
            ticks += 1
            ticks_since_mast += 1
            print(f"[SIORI] T{trial_n} tick={ticks:3d} NOT_VISIBLE", flush=True)
            continue

        if first is None:
            first = _snap(reading)

        dist_mm = float(reading["dist_mm"])
        x_mm    = float(reading["x_mm"])
        y_mm    = float(reading.get("y_mm", Y_TARGET_MM))
        dist_err = dist_mm - STOP_OFFSET_MM
        x_err    = x_mm - X_TARGET_MM
        y_err    = y_mm - Y_TARGET_MM

        # ── Win check ─────────────────────────────────────────────────────────
        if _happy(reading):
            happy_frames += 1
            follow._stop_robot(robot)
            print(f"[SIORI] T{trial_n} tick={ticks:3d} HAPPY #{happy_frames}  "
                  f"dist={dist_err:+.1f} x={x_err:+.1f} y={y_err:+.1f}", flush=True)
            if happy_frames >= CONFIRM_FRAMES:
                return {
                    "trial": trial_n, "won": True,
                    "ticks": ticks, "moves": moves,
                    "first": first, "final": _snap(reading),
                    "last_control": last_control,
                    "failure": None,
                }
            time.sleep(dt_s)
            ticks += 1
            ticks_since_mast += 1
            continue
        happy_frames = 0

        # ── Y priority correction (minimal-movement philosophy) ───────────────
        if (ticks_since_mast >= 3
                and abs(dist_err) <= Y_PRIORITY_DIST_MM
                and abs(x_err) <= Y_PRIORITY_X_MM
                and abs(y_err) > Y_TOL_MM):
            cmd = "d" if y_err > 0.0 else "u"
            ms  = _y_mast_pulse_ms(y_err)
            print(f"[SIORI] T{trial_n} tick={ticks:3d} MAST_{cmd.upper()} {ms}ms  "
                  f"y_err={y_err:+.1f}", flush=True)
            robot.send_command_pwm(cmd, 255, duration_ms=ms)
            moves += 1
            time.sleep(ms / 1000.0 + Y_MAST_SETTLE_S)
            follow._stop_robot(robot)
            ticks += 1
            ticks_since_mast = 0
            last_label = f"MAST_{cmd.upper()}"
            continue
        ticks_since_mast += 1

        # ── Wide-x bias override: if x is badly off, use bang-bang bias ───────
        if abs(x_err) > X_BIAS_THR_MM:
            turn_cmd = follow._turn_cmd_to_close_x_gap(x_err) or ("r" if x_err > 0.0 else "l")
            drive_mode = "forward" if dist_err >= 0.0 else "backward"
            send_result = follow._send_drive_bias(
                robot,
                turn_cmd=turn_cmd,
                drive_mode=drive_mode,
                strength="adaptive",
                duration_ms=X_BIAS_MS,
                reading=reading,
                context="siori_x_bias_override",
            )
            label = f"BIAS_{turn_cmd.upper()}"
            print(f"[SIORI] T{trial_n} tick={ticks:3d} {label:<12} "
                  f"dist={dist_err:+.1f} x={x_err:+.1f} y={y_err:+.1f}", flush=True)
            moves += 1
            time.sleep(max(dt_s, X_BIAS_MS / 1000.0))
            follow._stop_robot(robot)
            ticks += 1
            last_label = label
            continue

        # ── Core Astolfi PD command ───────────────────────────────────────────
        control = astolfi_wheel_command(
            x_off_mm=x_err * -1.0,   # sign: positive x_err → right turn needed
            distance_mm=dist_mm,
            stop_offset_mm=STOP_OFFSET_MM,
            bearing_y_offset_mm=min(dist_mm, 120.0),  # cap so x errors produce larger heading corrections at close range
            dt_s=dt_s,
            gains=gains,
            state=state,
            derivative_alpha=DERIV_ALPHA,
            linear_limit=LINEAR_LIMIT,
            angular_limit=ANGULAR_LIMIT,
            wheel_limit=WHEEL_LIMIT,
            max_dist_derivative_m_s=MAX_DIST_DERIV,
            max_head_derivative_rad_s=MAX_HEAD_DERIV,
        )
        last_control = dict(control)

        left  = float(control["left"])
        right = float(control["right"])

        # Suppress counter-rotation when far from target (no fishtailing)
        if abs(dist_err) > SAME_DIR_GAP_MM:
            # Far from target: bias turn — both wheels active so robot closes dist AND x
            # Only zero a wheel when very close (pure arc for fine x snap)
            v_sign = 1.0 if control["v"] >= 0.0 else -1.0
            if left  * v_sign < 0.0: left  = MIN_BIAS_SPEED * v_sign
            if right * v_sign < 0.0: right = MIN_BIAS_SPEED * v_sign

        actions, packet_ms, label = _build_wheel_packet(left, right)

        print(f"[SIORI] T{trial_n} tick={ticks:3d} {label:<12} "
              f"dist={dist_err:+.1f} x={x_err:+.1f} y={y_err:+.1f}  "
              f"v={control['v']:+.3f} ω={control['omega']:+.3f}  "
              f"head={math.degrees(control['head_err_rad']):+.1f}°  "
              f"{packet_ms}ms", flush=True)

        if packet_ms > 0:
            guarded_send_custom_actions_pwm(
                robot, "f" if control["v"] >= 0.0 else "b",
                actions, duration_ms=packet_ms,
                reading=reading, context="siori_pd_step1",
            )
            moves += 1
            time.sleep(max(dt_s, packet_ms / 1000.0))
            follow._stop_robot(robot)
        else:
            follow._stop_robot(robot)
            time.sleep(dt_s)

        elapsed = time.monotonic() - tick_start
        if elapsed < dt_s:
            time.sleep(dt_s - elapsed)
        ticks += 1
        last_label = label

    # Timed out — categorize failure
    try:
        final = _snap(last_reading)
        de = abs(final.get("dist_err", 999))
        xe = abs(final.get("x_err", 999))
        ye = abs(final.get("y_err", 999))
        if not last_reading.get("confident"):
            failure = "visibility_lost"
        elif ye > Y_TOL_MM and ye > de and ye > xe:
            failure = "y_gap_not_closed"
        elif de > DIST_TOL_MM:
            failure = "dist_gap_not_closed"
        elif xe > X_TOL_MM:
            failure = "x_gap_not_closed"
        else:
            failure = "timeout_near_gate"
    except Exception:
        failure = "unknown"

    return {
        "trial": trial_n, "won": False,
        "ticks": ticks, "moves": moves,
        "first": first, "final": _snap(last_reading),
        "last_control": last_control,
        "failure": failure,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reset between trials
# ─────────────────────────────────────────────────────────────────────────────

def _reset(vision: BrickDetector, robot: Robot) -> dict:
    """Small mast up then standard position reset."""
    print("[SIORI] reset: mast up 500ms", flush=True)
    robot.send_command_pwm("u", 255, duration_ms=500)
    time.sleep(0.6)

    follow._set_game_profile("empty")
    base = follow._reset_motion_config()
    reset_cfg = copy.deepcopy(base)
    reverse = reset_cfg.get("reverse_turn") if isinstance(reset_cfg.get("reverse_turn"), dict) else {}
    reverse["post_pause_s"] = 0.2
    reverse["settle_s"] = 0.0
    straight = reverse.get("straight_back_first") if isinstance(reverse.get("straight_back_first"), dict) else {}
    try:
        orig_ms = int(round(float(straight.get("duration_ms", 3000))))
    except (TypeError, ValueError):
        orig_ms = 3000
    straight["duration_ms"] = max(600, int(round(orig_ms * 0.25)))
    reverse["straight_back_first"] = straight
    reverse["dist_target_mm"] = STOP_OFFSET_MM + 75.0
    reverse["dist_tol_mm"] = 85.0
    reverse["x_offset_min_mm"] = 5.0
    reverse["x_offset_max_mm"] = 35.0
    reverse["target_abs_x_mm"] = 15.0
    mast_up = reset_cfg.get("mast_up") if isinstance(reset_cfg.get("mast_up"), dict) else {}
    mast_up["enabled"] = False
    reset_cfg["mast_up"] = mast_up
    reverse["x_goal_curve"] = dict(reverse.get("x_goal_curve") or {})
    reverse["x_goal_curve"]["max_duration_ms"] = 200
    reset_cfg["reverse_turn"] = reverse

    old_cfg = follow._reset_motion_config
    follow._reset_motion_config = lambda: copy.deepcopy(reset_cfg)
    try:
        result = follow._run_reset_sequence(vision, robot, rng=random)
    finally:
        follow._reset_motion_config = old_cfg
    return result if isinstance(result, dict) else {}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    follow._set_game_profile("empty")

    vision = BrickDetector(debug=True)
    robot  = Robot()
    try:
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)

        wins = 0
        records = []

        with out_path.open("w", encoding="utf-8") as fh:
            for t in range(1, args.trials + 1):
                print(f"\n[SIORI] ═══ Trial {t}/{args.trials} ═══", flush=True)

                ok = _pregame(vision, robot, t)
                if not ok:
                    rec = {
                        "trial": t, "won": False, "ticks": 0, "moves": 0,
                        "first": None, "final": None, "last_control": {},
                        "failure": "pregame_no_visibility",
                    }
                else:
                    rec = _run_trial(vision, robot, t)

                wins += 1 if rec.get("won") else 0
                records.append(rec)
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                fh.flush()

                print(f"[SIORI] T{t} result: {'WIN ✓' if rec['won'] else 'MISS ✗'}  "
                      f"failure={rec.get('failure')}  "
                      f"ticks={rec.get('ticks')}  moves={rec.get('moves')}", flush=True)

                if t < args.trials:
                    _reset(vision, robot)

        print(f"\n[SIORI] completed {args.trials} trials  wins={wins}/{args.trials}  data={out_path}", flush=True)
        return 0

    finally:
        follow._stop_robot(robot)
        try:
            robot.close()
        except Exception:
            pass
        try:
            vision.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
