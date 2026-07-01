#!/usr/bin/env python3
"""Practice empty Step 1 with one live-observed graceful dist+x movement."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import a_follow_the_brick as follow
import frozen_step12_trials as proof
from helper_brick_detector_yolo import BrickDetector
from helper_robot_control import Robot
from telemetry_robot import SPEED_SCORE_MIN, speed_power_pwm_for_cmd


DEFAULT_SITE_DIR = Path("runs/frozen_step12_site")
DEFAULT_ITERATIONS = 1
DEFAULT_TRIALS_PER_ITERATION = 5
RESET_DIST_OFFSET_MIN_MM = 15.0
RESET_DIST_OFFSET_MAX_MM = 45.0
RESET_X_GAP_MIN_MM = 8.0
RESET_X_GAP_MAX_MM = 20.0
RESET_X_TARGET_GAP_MM = 14.0
RESET_BACK_PWM = int(speed_power_pwm_for_cmd("b", SPEED_SCORE_MIN)[1])
RESET_BACK_CHUNK_MS = 300
RESET_BACK_MAX_CHUNKS = 1
RESET_START_MIN_OFFSET_MM = 15.0
RESET_START_MAX_OFFSET_MM = 45.0
MAX_ARC_MS = 1500
MIN_ARC_MS = 260
LIVE_SAMPLE_S = 0.055
RESET_MAX_AUTO_BACKUP_NEEDED_MM = 0.0
RESET_LIVE_MAX_MS = 0
RESET_LIVE_STOP_OFFSET_MM = 18.0
RESET_LIVE_ABORT_OFFSET_MM = 45.0
RESET_LIVE_MIN_MS = 300
RESET_STAGE_MAX_REVERSE_ATTEMPTS = 1
ALLOW_AUTO_REVERSE_STAGING = False
FORWARD_BASE_PWM = int(speed_power_pwm_for_cmd("f", SPEED_SCORE_MIN)[1])
FORWARD_MICRO_X_DEADBAND_MM = 1.0
FORWARD_WRONG_WAY_STOP_MM = 8.0

# Turn-by-timed-hold tuning. At the single crawl speed the only way to turn is
# to keep both treads at crawl, then hold the INNER tread at 0 for a spell while
# the outer keeps crawling forward — longer hold = sharper turn. We always roll
# forward; we never run the treads at different speeds.
#   BOTH phase  -> both treads crawl forward (no turn)
#   HOLD phase  -> inner tread at 0, outer keeps crawling (this is the turn)
# The outer tread runs continuously across BOTH+HOLD in one packet (smooth); the
# inner tread runs only the BOTH span then stops for the HOLD span.
GRACEFUL_BOTH_MS = 200          # both-tread span (>= breakaway floor)
GRACEFUL_HOLD_GENTLE_MS = 200   # inner-at-0 span for a gentle turn
GRACEFUL_HOLD_SHARP_MS = 400    # inner-at-0 span for a sharp turn
GRACEFUL_SHARP_X_MM = 14.0      # |x gap| above this uses the sharp hold
GRACEFUL_STRAIGHT_MS = 300      # straight crawl span when x is already centered
GRACEFUL_WIN_BUDGET_S = 16.0    # total time to crawl to the Step 1 win


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _url(port: int) -> str:
    try:
        host = socket.gethostbyname(socket.gethostname())
    except Exception:
        host = "plainOrin"
    return f"http://{host}:{int(port)}/"


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, *, signed: bool = False) -> str:
    number = _num(value)
    if number is None:
        return "N/A"
    return f"{number:+.1f}" if signed else f"{number:.1f}"


def _reading_summary(reading: dict | None) -> dict:
    if not isinstance(reading, dict):
        return {}
    out = {}
    for key in (
        "visible",
        "confident",
        "conf",
        "dist_mm",
        "x_mm",
        "y_mm",
        "reason",
        "green_stack_candidate_count",
        "green_contour_fallback",
        "green_contour_override",
        "vision_geometry_source",
    ):
        if key in reading:
            value = reading.get(key)
            out[key] = round(value, 3) if isinstance(value, float) else value
    return out


def _axis(reading: dict | None, axis: str) -> dict:
    if axis == "dist":
        target = float(proof._direct_step1_dist_target())
        tol = float(proof.DIRECT_STEP1_DIST_TOL_MM)
        value = _num((reading or {}).get("dist_mm") if isinstance(reading, dict) else None)
    elif axis == "x":
        target = float(follow._x_target_mm())
        tol = float(proof.DIRECT_STEP1_X_TOL_MM)
        value = _num((reading or {}).get("x_mm") if isinstance(reading, dict) else None)
    else:
        return {"target_mm": None, "tol_mm": None, "value_mm": None, "ok": False, "closeness_pct": 0.0}
    err = None if value is None else float(value) - target
    ok = bool(err is not None and abs(float(err)) <= tol)
    closeness = 0.0 if err is None else max(0.0, min(100.0, 100.0 * (1.0 - abs(float(err)) / tol)))
    return {
        "target_mm": target,
        "tol_mm": tol,
        "value_mm": value,
        "err_mm": err,
        "ok": ok,
        "closeness_pct": round(closeness, 1),
    }


def _target_met(reading: dict | None) -> bool:
    return bool(_axis(reading, "dist")["ok"] and _axis(reading, "x")["ok"])


def _virtual_wall_exceeded(reading: dict | None) -> bool:
    try:
        return bool(follow._virtual_safety_dist_exceeded(reading))
    except Exception:
        return False


def _usable_stack_reading(reading: dict | None) -> bool:
    if not isinstance(reading, dict):
        return False
    if bool(reading.get("confident")):
        return True
    dist = _num(reading.get("dist_mm"))
    x_val = _num(reading.get("x_mm"))
    conf = _num(reading.get("conf"))
    return bool(dist is not None and x_val is not None and conf is not None and conf >= 70.0)


def _gap_abs(reading: dict | None, axis: str) -> float | None:
    data = _axis(reading, axis)
    err = _num(data.get("err_mm"))
    tol = _num(data.get("tol_mm"))
    if err is None or tol is None:
        return None
    return max(0.0, abs(float(err)) - float(tol))


def _crawl_pwm(cmd: str) -> int:
    try:
        return int(speed_power_pwm_for_cmd(str(cmd or "f").strip().lower(), SPEED_SCORE_MIN)[1])
    except Exception:
        return int(FORWARD_BASE_PWM)


def _crawl_cap_actions(actions: list[dict]) -> list[dict]:
    capped = []
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        row = dict(action)
        cmd = str(row.get("action") or "").strip().lower()
        if cmd in {"f", "b", "l", "r"}:
            requested = int(round(float(row.get("pwm") or 0)))
            floor = _crawl_pwm(cmd)
            row["pwm"] = max(int(requested), int(floor)) if requested > 0 else 0
            row["crawl_floor_applied"] = requested > 0 and requested < int(floor)
        capped.append(row)
    return capped


class S1Site:
    def __init__(self, site_dir: Path):
        self.site_dir = Path(site_dir)
        self.images_dir = self.site_dir / "images"
        self.state_path = self.site_dir / "state_empty_s1.json"
        self.index_path = self.site_dir / "index.html"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        old_state = self.site_dir / "state.json"
        if old_state.exists():
            backup = self.site_dir / f"state_before_empty_s1_{int(time.time())}.json"
            try:
                shutil.copy2(old_state, backup)
            except Exception:
                pass
        self.state = {
            "title": "Leia Empty Step 1: 5-Win Streak",
            "started_at": _now(),
            "updated_at": _now(),
            "summary": "Goal: 5 consecutive empty S1 wins",
            "trials_per_iteration": DEFAULT_TRIALS_PER_ITERATION,
            "iterations": [],
        }
        self.write()

    def capture(self, vision: BrickDetector, label: str, reading: dict | None) -> tuple[str | None, dict]:
        try:
            vision.read()
        except Exception:
            pass
        frame = getattr(vision, "current_frame", None)
        if frame is None:
            frame = getattr(vision, "raw_frame", None)
        rel = None
        if frame is not None:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(label))
            name = f"{int(time.time())}_{safe}.png"
            path = self.images_dir / name
            cv2.imwrite(str(path), frame)
            rel = f"images/{name}"
        return rel, _reading_summary(reading)

    def add_iteration(self, row: dict) -> None:
        self.state.setdefault("iterations", []).append(dict(row))
        self.state["updated_at"] = _now()
        rows = list(self.state.get("iterations") or [])
        wins = sum(1 for item in rows if item.get("status") == "win")
        grouped = {}
        for item in rows:
            grouped.setdefault(int(item.get("iteration", 0) or 0), []).append(item)
        trials_per = int(self.state.get("trials_per_iteration") or DEFAULT_TRIALS_PER_ITERATION)
        perfect = sum(
            1
            for trials in grouped.values()
            if len(trials) >= trials_per and all(trial.get("status") == "win" for trial in trials[:trials_per])
        )
        self.state["summary"] = f"{wins}/{len(rows)} trial wins; {perfect}/{len(grouped)} perfect iterations"
        self.write()

    def write(self) -> None:
        self.site_dir.mkdir(parents=True, exist_ok=True)
        state_tmp = self.state_path.with_suffix(".json.tmp")
        index_tmp = self.index_path.with_suffix(".html.tmp")
        state_tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        index_tmp.write_text(self.render(), encoding="utf-8")
        state_tmp.replace(self.state_path)
        index_tmp.replace(self.index_path)

    def axis_html(self, reading: dict | None, axis: str) -> str:
        data = _axis(reading, axis)
        pct = max(0.0, min(100.0, float(data.get("closeness_pct") or 0.0)))
        ok_class = " axis-ok" if data.get("ok") else ""
        signed = axis == "x"
        target = _num(data.get("target_mm"))
        tol = _num(data.get("tol_mm"))
        value = _fmt(data.get("value_mm"), signed=signed)
        if target is None or tol is None:
            range_text = "not scored"
            phrase = value
        else:
            low = _fmt(target - tol, signed=signed)
            high = _fmt(target + tol, signed=signed)
            range_text = f"{low} to {high}"
            phrase = f"{value} {'is' if data.get('ok') else 'is not'} between"
        return (
            f'<div class="axis{ok_class}">'
            f'<div class="axis-top"><b>{html.escape(axis.upper())}</b><span>{html.escape(range_text)}</span></div>'
            f'<div class="axis-phrase">{html.escape(phrase)}</div>'
            f'<div class="bar"><i style="width:{pct:.1f}%"></i></div>'
            f'<div class="axis-sub">err {_fmt(data.get("err_mm"), signed=True)} mm, close {pct:.0f}%</div>'
            "</div>"
        )

    def state_icon_html(self, label: str, state: str, text: str) -> str:
        state_key = str(state or "info").strip().lower()
        glyph = {"win": "✓", "fail": "!", "start": "S", "move": "M", "end": "E"}.get(state_key, "i")
        return (
            f'<div class="state-chip {html.escape(state_key)}">'
            f'<span class="state-icon">{html.escape(glyph)}</span>'
            f'<div><b>{html.escape(label)}</b><small>{html.escape(text)}</small></div>'
            f'</div>'
        )

    def photo_html(self, image: str | None, caption: str) -> str:
        if not image:
            return f'<figure class="proof-photo"><figcaption>{html.escape(caption)}</figcaption><span class="muted">no photo</span></figure>'
        return (
            f'<figure class="proof-photo"><figcaption>{html.escape(caption)}</figcaption>'
            f'<a href="{html.escape(image)}"><img src="{html.escape(image)}" alt="{html.escape(caption)}"></a></figure>'
        )

    def render(self) -> str:
        rows = list(self.state.get("iterations") or [])
        wins = sum(1 for row in rows if row.get("status") == "win")
        trials_per = int(self.state.get("trials_per_iteration") or DEFAULT_TRIALS_PER_ITERATION)
        grouped = {}
        for row in rows:
            grouped.setdefault(int(row.get("iteration", 0) or 0), []).append(row)
        perfect_iterations = sum(
            1
            for trials in grouped.values()
            if len(trials) >= trials_per and all(trial.get("status") == "win" for trial in trials[:trials_per])
        )
        sections = []
        for iteration in sorted(grouped.keys(), reverse=True):
            trials = sorted(grouped[iteration], key=lambda item: int(item.get("trial", 0) or 0))
            iter_wins = sum(1 for row in trials if row.get("status") == "win")
            iter_status = "win" if len(trials) >= trials_per and iter_wins == trials_per else "fail"
            cards = []
            for row in trials:
                status = str(row.get("status") or "info")
                card_class = "trial-card case-study" if bool(row.get("case_study")) else "trial-card"
                start = row.get("start_reading") if isinstance(row.get("start_reading"), dict) else {}
                end = row.get("end_reading") if isinstance(row.get("end_reading"), dict) else {}
                move_note = str(row.get("move_note") or "")
                result_text = str(row.get("reason") or "")
                cards.append(
                    f'<article class="{card_class}">'
                    f'<div class="trial-head"><b>Trial {int(row.get("trial", 0) or 0)}</b>'
                    f'<span class="status {html.escape(status)}">{html.escape(status)}</span></div>'
                    f'<div class="experiment-name">{html.escape(str(row.get("experiment") or ""))}</div>'
                    f'<div class="experiment-line">{html.escape(str(row.get("experiment_line") or ""))}</div>'
                    f'<div class="state-strip">'
                    f'{self.state_icon_html("Start", "start", "honest staged S1 start pose")}'
                    f'{self.state_icon_html("Move", "move", move_note[:150] if move_note else "no move sent")}'
                    f'{self.state_icon_html("Result", status, result_text)}'
                    f'</div>'
                    f'<div class="proof-row">'
                    f'{self.photo_html(row.get("start_image"), "start")}'
                    f'<div class="bars"><h3>Start</h3>{self.axis_html(start, "dist")}{self.axis_html(start, "x")}'
                    f'<h3>End</h3>{self.axis_html(end, "dist")}{self.axis_html(end, "x")}'
                    f'<div class="move-note">{html.escape(str(row.get("move_note") or ""))}</div></div>'
                    f'{self.photo_html(row.get("end_image"), "end")}'
                    f'</div></article>'
                )
            sections.append(
                f'<section class="iteration-group">'
                f'<div class="iteration-head"><h2>Iteration {iteration}</h2>'
                f'<span class="status {iter_status}">{iter_wins}/{len(trials)} wins</span></div>'
                f'<div class="trial-list">{"".join(cards)}</div></section>'
            )
        cards_html = "".join(sections) if sections else '<p class="empty">No iterations yet.</p>'
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(str(self.state.get("title")))}</title>
  <style>
    body {{ margin:0; font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif; background:#f5f7f8; color:#17202a; }}
    header {{ padding:18px 22px; background:#193549; color:white; }}
    h1 {{ margin:0 0 6px; font-size:22px; }}
    main {{ padding:18px 22px 36px; }}
    .summary {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:16px; }}
    .metric {{ background:white; border:1px solid #d9e0e8; border-radius:6px; padding:10px 12px; min-width:180px; }}
    .metric b {{ display:block; font-size:20px; }}
    .trial-grid {{ display:grid; grid-template-columns:1fr; gap:12px; }}
    .iteration-group {{ padding:10px 0 16px; border-top:1px solid #ccd7df; }}
    .iteration-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin:0 0 10px; }}
    .iteration-head h2 {{ margin:0; font-size:18px; }}
    .trial-list {{ display:grid; grid-template-columns:1fr; gap:12px; }}
    .trial-card {{ background:white; border:1px solid #d9e0e8; border-radius:6px; padding:12px; }}
    .trial-card.case-study {{ border-color:#bd5a45; box-shadow:0 0 0 2px rgba(189,90,69,.14); }}
    .trial-head {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
    .status {{ display:inline-block; min-width:56px; padding:2px 8px; border-radius:999px; text-align:center; font-weight:700; }}
    .win {{ background:#dff6e8; color:#17633a; }}
    .fail {{ background:#ffe3df; color:#8c261e; }}
    .info {{ background:#e4eefc; color:#1d4f8f; }}
    .experiment-name {{ margin-top:6px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; color:#33485b; }}
    .experiment-line {{ margin-top:4px; color:#253341; }}
    .meta,.muted {{ color:#6b7785; font-size:12px; }}
    .state-strip {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-top:10px; }}
    .state-chip {{ display:flex; gap:8px; align-items:flex-start; padding:8px; border:1px solid #d9e0e8; border-radius:6px; background:#f8fafb; }}
    .state-chip b {{ display:block; font-size:12px; }}
    .state-chip small {{ display:block; color:#566574; font-size:11px; }}
    .state-icon {{ flex:0 0 22px; width:22px; height:22px; display:grid; place-items:center; border-radius:50%; font-weight:800; background:#e4eefc; color:#1d4f8f; }}
    .state-chip.win .state-icon {{ background:#dff6e8; color:#17633a; }}
    .state-chip.fail .state-icon {{ background:#ffe3df; color:#8c261e; }}
    .proof-row {{ display:grid; grid-template-columns:minmax(220px,32%) 1fr minmax(220px,32%); gap:14px; margin-top:10px; align-items:start; }}
    figure {{ margin:0; }}
    .proof-photo img {{ width:100%; height:300px; object-fit:cover; border:1px solid #d9e0e8; border-radius:4px; background:#111; }}
    figcaption {{ margin:0 0 6px; font-size:12px; font-weight:700; color:#566574; text-transform:uppercase; }}
    .bars {{ display:grid; gap:7px; }}
    h3 {{ margin:0; font-size:12px; text-transform:uppercase; color:#566574; }}
    .axis-top {{ display:flex; justify-content:space-between; gap:8px; font-size:12px; }}
    .axis-phrase {{ margin-top:2px; font-size:12px; }}
    .bar {{ height:10px; margin:4px 0; background:#edf1f4; border-radius:999px; overflow:hidden; }}
    .bar i {{ display:block; height:100%; background:#bd5a45; }}
    .axis-ok .bar i {{ background:#2b8a58; }}
    .axis-sub {{ color:#566574; font-size:12px; }}
    .move-note {{ margin-top:6px; padding-top:8px; border-top:1px solid #e6ebf0; color:#33485b; }}
    .empty {{ padding:14px; background:white; border:1px solid #d9e0e8; border-radius:6px; }}
    @media (max-width: 900px) {{ .proof-row,.state-strip {{ grid-template-columns:1fr; }} .proof-photo img {{ height:240px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(str(self.state.get("title")))}</h1>
    <div>Updated: {html.escape(str(self.state.get("updated_at")))}</div>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span>Graceful S1 wins</span><b>{wins}/{len(rows)}</b></div>
      <div class="metric"><span>Perfect iterations</span><b>{perfect_iterations}/{len(grouped)}</b></div>
      <div class="metric"><span>Target</span><b>{int(self.state.get("trials_per_iteration") or trials_per)} S1 attempts</b></div>
      <div class="metric"><span>Status</span><b>{html.escape(str(self.state.get("summary")))}</b></div>
    </section>
    <section class="trial-grid">{cards_html}</section>
  </main>
</body>
</html>
"""


def _configure_empty_s1() -> None:
    follow._set_game_profile("empty")
    cfg = follow._follow_motion_config()
    step2 = cfg.get("step2") if isinstance(cfg.get("step2"), dict) else {}
    proof._disable_y_gates(cfg, step2)
    _install_relaxed_top_contour_reader()
    proof.HONEST_RESET_DIST_OFFSET_MIN_MM = float(RESET_DIST_OFFSET_MIN_MM)
    proof.HONEST_RESET_DIST_OFFSET_MAX_MM = float(RESET_DIST_OFFSET_MAX_MM)
    proof.HONEST_RESET_X_GAP_MIN_MM = float(RESET_X_GAP_MIN_MM)
    proof.HONEST_RESET_X_GAP_MAX_MM = float(RESET_X_GAP_MAX_MM)
    proof.HONEST_RESET_X_TARGET_GAP_MM = float(RESET_X_TARGET_GAP_MM)
    proof.DIRECT_LIVE_OBSERVE_MOVES = True
    proof.DIRECT_LIVE_SAMPLE_S = float(LIVE_SAMPLE_S)
    proof.DIRECT_LIVE_FINAL_SETTLE_S = 0.0
    proof.DIRECT_STEP1_X_TOL_MM = 7.0
    proof.DIRECT_STEP1_DIST_TOL_MM = 15.0
    proof._set_direct_stable_reads(3)


def _relaxed_green_stack_candidates(frame) -> list[dict]:
    if frame is None:
        return []
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except Exception:
        return []
    mask = cv2.inRange(
        hsv,
        np.array([45, 90, 25], dtype=np.uint8),
        np.array([105, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((25, 45), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    frame_h, frame_w = frame.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 700.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = float(x) + float(w) / 2.0
        if (
            y < 15
            or y > int(frame_h * 0.62)
            or x < 20
            or x + w > int(frame_w - 20)
            or cx < float(proof.DIRECT_X_MIN_SAFE_CX_PX)
            or cx > float(proof.DIRECT_X_MAX_SAFE_CX_PX)
            or w < 28
            or h < 40
            or w > int(frame_w * 0.55)
        ):
            continue
        candidates.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "cx": round(cx, 1),
                "area": round(area, 1),
            }
        )
    return sorted(candidates, key=lambda box: float(box["cx"]))


def _install_relaxed_top_contour_reader() -> None:
    proof._green_stack_candidates = _relaxed_green_stack_candidates


def _curve_actions(reading: dict, strength: str, boost: float) -> tuple[str, list[dict]]:
    x_err = _num(reading.get("x_mm"))
    target = float(follow._x_target_mm())
    if x_err is None:
        x_err = target
    x_delta = float(x_err) - target
    if abs(x_delta) <= float(FORWARD_MICRO_X_DEADBAND_MM):
        return "straight", _crawl_cap_actions(follow._straight_drive_actions("f", _crawl_pwm("f")))
    cmd = follow._turn_cmd_to_close_x_gap(x_delta)
    if cmd not in {"l", "r"}:
        cmd = "l" if (float(x_err) - target) > 0 else "r"
    if abs(x_delta) <= float(proof.DIRECT_STEP1_X_TOL_MM):
        curve = {"inner_pwm": int(FORWARD_BASE_PWM) - 5, "outer_pwm": int(FORWARD_BASE_PWM) + 5}
    else:
        curve = follow._turn_bias_curve_for_drive_mode("forward", str(strength))
        curve = follow._boost_curve_pwms(curve, float(boost))
        curve = dict(curve)
        curve["inner_pwm"] = max(int(curve.get("inner_pwm", 0) or 0), int(FORWARD_BASE_PWM) - 20)
        curve["outer_pwm"] = max(int(curve.get("outer_pwm", 0) or 0), int(FORWARD_BASE_PWM) + 5)
    actions = follow._scaled_actions(follow._turn_bias_actions(drive_mode="forward", turn_cmd=cmd, curve=curve))
    return cmd, _crawl_cap_actions(actions)


def _straight_segment(pwm: int, ms: int) -> list[dict]:
    """Both treads crawl forward at the same speed for the same duration."""
    return [
        {"target": "l", "action": "b", "pwm": int(pwm), "duration_ms": int(ms)},
        {"target": "r", "action": "f", "pwm": int(pwm), "duration_ms": int(ms)},
    ]


def _duty_turn_segment(turn_cmd: str, pwm: int, both_ms: int, hold_ms: int) -> list[dict]:
    """One BOTH+HOLD turn cycle in a single packet.

    The outer tread crawls forward continuously for both_ms+hold_ms; the inner
    tread crawls only for both_ms then drops to 0 for the hold_ms span. Same
    speed on both treads — the turn comes purely from the inner tread's shorter
    on-time. turn_cmd 'l' drives the right tread (holds left); 'r' drives the
    left tread (holds right).
    """
    outer_ms = int(both_ms) + int(hold_ms)
    inner_ms = int(both_ms)
    if str(turn_cmd).strip().lower() == "l":
        outer = {"target": "r", "action": "f", "pwm": int(pwm), "duration_ms": outer_ms}
        inner = {"target": "l", "action": "b", "pwm": int(pwm), "duration_ms": inner_ms}
    else:
        outer = {"target": "l", "action": "b", "pwm": int(pwm), "duration_ms": outer_ms}
        inner = {"target": "r", "action": "f", "pwm": int(pwm), "duration_ms": inner_ms}
    return [outer, inner]


def _one_graceful_move(
    vision: BrickDetector,
    robot: proof.MastFrozenRobot,
    start: dict,
    *,
    duration_ms: int,
    strength: str,
    boost: float,
) -> tuple[dict, dict]:
    """Crawl forward to the Step 1 win, steering only by timed inner-wheel holds.

    Always rolling forward; a turn is a HOLD (inner tread at 0) whose length sets
    the sharpness. duration_ms/strength/boost from the harness are advisory; the
    crawl runs until the win or GRACEFUL_WIN_BUDGET_S.
    """
    pwm = _crawl_pwm("f")
    before_mast = len(robot.mast_attempts)
    start_dist = _num(start.get("dist_mm")) if isinstance(start, dict) else None
    x_target = float(follow._x_target_mm())
    deadline = time.time() + float(GRACEFUL_WIN_BUDGET_S)
    latest = start if isinstance(start, dict) else {}
    live_hit = wrong_way_stop = virtual_wall_stop = False
    samples = 0
    last_turn = "straight"
    last_send = None

    while time.time() < deadline:
        cur = latest if isinstance(latest, dict) else {}
        x_err = _num(cur.get("x_mm"))
        x_delta = (float(x_err) - x_target) if x_err is not None else 0.0
        if abs(x_delta) <= float(FORWARD_MICRO_X_DEADBAND_MM):
            actions = _straight_segment(pwm, int(GRACEFUL_STRAIGHT_MS))
            seg_ms = int(GRACEFUL_STRAIGHT_MS)
            last_turn = "straight"
        else:
            turn_cmd = follow._turn_cmd_to_close_x_gap(x_delta)
            if turn_cmd not in {"l", "r"}:
                turn_cmd = "r" if x_delta > 0 else "l"
            hold_ms = int(GRACEFUL_HOLD_SHARP_MS) if abs(x_delta) > float(GRACEFUL_SHARP_X_MM) else int(GRACEFUL_HOLD_GENTLE_MS)
            actions = _duty_turn_segment(turn_cmd, pwm, int(GRACEFUL_BOTH_MS), hold_ms)
            seg_ms = int(GRACEFUL_BOTH_MS) + hold_ms
            last_turn = f"{turn_cmd}/hold{hold_ms}"
        last_send = robot.send_custom_actions_pwm("f", actions, duration_ms=int(seg_ms))
        seg_end = time.time() + float(seg_ms) / 1000.0
        stop = False
        while time.time() < seg_end:
            time.sleep(float(LIVE_SAMPLE_S))
            live = _quick_read(vision)
            if isinstance(live, dict) and bool(live.get("confident")):
                latest = live
                samples += 1
                dist = _num(live.get("dist_mm"))
                if _virtual_wall_exceeded(live):
                    virtual_wall_stop = True
                    stop = True
                    break
                if (
                    start_dist is not None
                    and dist is not None
                    and float(dist) > float(start_dist) + float(FORWARD_WRONG_WAY_STOP_MM)
                ):
                    wrong_way_stop = True
                    stop = True
                    break
                if _target_met(live):
                    live_hit = True
                    stop = True
                    break
        if stop:
            break

    follow._stop_robot(robot)
    final = _quick_read(vision)
    if not isinstance(final, dict) or not bool(final.get("confident")):
        final = latest if isinstance(latest, dict) else {}
    info = {
        "turn_cmd": last_turn,
        "duration_ms": int(duration_ms),
        "strength": str(strength),
        "boost": float(boost),
        "live_hit": bool(live_hit),
        "live_samples": int(samples),
        "send_result": last_send,
        "mast_attempt_delta": int(len(robot.mast_attempts) - before_mast),
        "wrong_way_stop": bool(wrong_way_stop),
        "virtual_wall_stop": bool(virtual_wall_stop),
    }
    return final, info


def _quick_read(vision: BrickDetector) -> dict:
    reading = proof._direct_contour_read(vision)
    if isinstance(reading, dict) and bool(reading.get("confident")):
        return reading
    try:
        fallback = follow._read_brick_measurement(vision, jump_guard=False)
    except TypeError:
        fallback = follow._read_brick_measurement(vision)
    return fallback if isinstance(fallback, dict) else reading


def _dist_offset(reading: dict | None) -> float | None:
    dist = _num((reading or {}).get("dist_mm") if isinstance(reading, dict) else None)
    if dist is None:
        return None
    return float(dist) - float(proof._direct_step1_dist_target())


def _x_offset(reading: dict | None) -> float | None:
    x_val = _num((reading or {}).get("x_mm") if isinstance(reading, dict) else None)
    if x_val is None:
        return None
    return float(x_val) - float(follow._x_target_mm())


def _start_pose_ready(reading: dict | None) -> bool:
    dist_offset = _dist_offset(reading)
    if dist_offset is None or _x_offset(reading) is None:
        return False
    return float(RESET_START_MIN_OFFSET_MM) <= float(dist_offset) <= float(RESET_START_MAX_OFFSET_MM)


def _live_reverse_to_start_band(
    vision: BrickDetector,
    robot: proof.MastFrozenRobot,
    reading: dict,
    *,
    label: str,
) -> tuple[dict, list[dict]]:
    target = float(proof._direct_step1_dist_target())
    stop_dist = target + float(RESET_LIVE_STOP_OFFSET_MM)
    start_dist = _num(reading.get("dist_mm"))
    if start_dist is None:
        return reading, []
    needed_mm = max(0.0, float(stop_dist) - float(start_dist))
    duration_ms = max(int(RESET_LIVE_MIN_MS), min(int(RESET_LIVE_MAX_MS), int(round(needed_mm * 55.0))))
    actions = _crawl_cap_actions(follow._straight_drive_actions("b", _crawl_pwm("b")))
    before_mast = len(robot.mast_attempts)
    send_result = robot.send_custom_actions_pwm("b", actions, duration_ms=int(duration_ms))
    action_log = [
        {
            "cmd": "b",
            "mode": "live_reverse_to_start_band",
            "pwm": int(RESET_BACK_PWM),
            "duration_ms": int(duration_ms),
            "before_dist_mm": float(start_dist),
            "target_stop_dist_mm": float(stop_dist),
            "send_result": send_result,
        }
    ]
    deadline = time.time() + (float(duration_ms) / 1000.0)
    latest = reading
    samples = 0
    misses = 0
    while time.time() < deadline:
        time.sleep(float(LIVE_SAMPLE_S))
        live = _quick_read(vision)
        if isinstance(live, dict) and _usable_stack_reading(live) and bool(live.get("confident")):
            latest = live
            samples += 1
            misses = 0
            dist = _num(live.get("dist_mm"))
            if dist is not None and float(dist) >= stop_dist:
                break
            if dist is not None and float(dist) >= target + float(RESET_LIVE_ABORT_OFFSET_MM):
                break
        else:
            misses += 1
            if misses >= 4:
                break
    follow._stop_robot(robot)
    final = _quick_read(vision)
    if (
        (not isinstance(final, dict) or not bool(final.get("confident")))
        and isinstance(latest, dict)
        and bool(latest.get("confident"))
    ):
        final = latest
    if not isinstance(final, dict) or not _usable_stack_reading(final):
        final = latest if isinstance(latest, dict) else {}
    action_log[0]["live_samples"] = int(samples)
    action_log[0]["after_dist_mm"] = _num(final.get("dist_mm")) if isinstance(final, dict) else None
    action_log[0]["mast_attempt_delta"] = int(len(robot.mast_attempts) - before_mast)
    return final, action_log


def _stage_empty_s1_start_pose(
    vision: BrickDetector,
    robot: proof.MastFrozenRobot,
    *,
    label: str,
) -> tuple[bool, str, dict, list[dict]]:
    reading = _quick_read(vision)
    actions: list[dict] = []
    if not _usable_stack_reading(reading):
        return False, f"{label}_not_confident", reading, actions
    target = float(proof._direct_step1_dist_target())
    min_dist = target + float(RESET_START_MIN_OFFSET_MM)
    max_dist = target + float(RESET_START_MAX_OFFSET_MM)
    initial_dist = _num(reading.get("dist_mm"))
    if initial_dist is None:
        return False, f"{label}_dist_invalid", reading, actions
    if _virtual_wall_exceeded(reading):
        return False, f"{label}_virtual_wall_exceeded_{float(initial_dist):.1f}mm", reading, actions
    if _target_met(reading):
        return False, f"{label}_already_s1_happy_manual_reset_required", reading, actions
    backup_needed = float(min_dist) - float(initial_dist)
    if backup_needed > 0.0 and not bool(ALLOW_AUTO_REVERSE_STAGING):
        return False, f"{label}_too_close_no_reverse_allowed_need_{backup_needed:.1f}mm", reading, actions
    if bool(ALLOW_AUTO_REVERSE_STAGING):
        if backup_needed > float(RESET_MAX_AUTO_BACKUP_NEEDED_MM):
            return False, f"{label}_too_close_for_auto_reset_need_{backup_needed:.1f}mm", reading, actions
        for _attempt in range(int(RESET_STAGE_MAX_REVERSE_ATTEMPTS)):
            dist = _num(reading.get("dist_mm"))
            if dist is None or float(dist) >= min_dist:
                break
            reading, reverse_actions = _live_reverse_to_start_band(vision, robot, reading, label=label)
            actions.extend(reverse_actions)
            if not _usable_stack_reading(reading):
                return False, f"{label}_lost_confidence_after_live_reverse", reading, actions
    dist = _num(reading.get("dist_mm"))
    if dist is not None and float(dist) < min_dist:
        return False, f"{label}_still_too_close_after_live_reverse_{float(dist):.1f}", reading, actions
    dist = _num(reading.get("dist_mm"))
    if dist is None:
        return False, f"{label}_dist_invalid_after_backoff", reading, actions
    if float(dist) > max_dist:
        return False, f"{label}_too_far_after_backoff_{float(dist):.1f}", reading, actions
    if _start_pose_ready(reading):
        actions.append({"cmd": "x_gap", "reason": "skipped_natural_x_error", "ok": True})
        return True, f"{label}_start_pose_ready", reading, actions
    if _x_offset(reading) is not None and _dist_offset(reading) is not None:
        return False, f"{label}_start_pose_not_ready", reading, actions
    return False, f"{label}_start_pose_reading_invalid", reading, actions


def _variant(iteration: int, previous: dict | None, start: dict | None = None) -> dict:
    dist_gap = _gap_abs(start, "dist") if isinstance(start, dict) else None
    x_gap = _gap_abs(start, "x") if isinstance(start, dict) else None
    dist_gap_val = 18.0 if dist_gap is None else max(0.0, float(dist_gap))
    x_gap_val = 0.0 if x_gap is None else max(0.0, float(x_gap))
    duration = 420 + int(round(dist_gap_val * 35.0))
    boost = 1.0
    strength = "gentle" if x_gap_val <= 12.0 else "medium"
    if isinstance(previous, dict):
        end = previous.get("end_reading") if isinstance(previous.get("end_reading"), dict) else {}
        dist_err = _axis(end, "dist").get("err_mm")
        x_err = _axis(end, "x").get("err_mm")
        if _num(dist_err) is not None and float(dist_err) > 12:
            duration += 110
        if _num(dist_err) is not None and float(dist_err) < -10:
            duration -= 180
        if _num(x_err) is not None and abs(float(x_err)) > float(proof.DIRECT_STEP1_X_TOL_MM):
            boost += 0.05
            if abs(float(x_err)) > 18:
                strength = "medium"
    duration = max(MIN_ARC_MS, min(MAX_ARC_MS, int(duration)))
    return {
        "duration_ms": duration,
        "strength": strength,
        "boost": round(boost, 3),
        "name": f"empty-s1-graceful-{iteration:02d}",
        "line": (
            f"One distance-scaled live-observed forward {strength} bias arc for up to {duration}ms; "
            f"start dist gap {dist_gap_val:.1f}mm, start x gap {x_gap_val:.1f}mm; "
            f"curve boost {boost:.2f}; strict proof band dist +/-15mm, x +/-7mm; "
            "no y motion, no follow-loop nudges."
        ),
    }


def run(args: argparse.Namespace) -> int:
    _configure_empty_s1()
    site = S1Site(Path(args.site_dir))
    print(f"[SITE] {Path(args.site_dir).resolve()}", flush=True)
    print(f"[SITE] Suggested URL: {_url(int(args.port))}", flush=True)
    vision = None
    base_robot = None
    robot = None
    previous = None
    try:
        vision = BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(follow.CROWN_PROFILE_TUNING))
        follow._warmup(vision)
        base_robot = Robot()
        robot = proof.MastFrozenRobot(base_robot)
        total_iterations = int(args.iterations)
        trials_per_iteration = int(args.trials_per_iteration)
        site.state["trials_per_iteration"] = int(trials_per_iteration)
        site.write()
        for iteration in range(1, total_iterations + 1):
            for trial in range(1, trials_per_iteration + 1):
                params = _variant(iteration, previous)
                site.state["summary"] = (
                    f"Running iteration {iteration}/{total_iterations}, "
                    f"trial {trial}/{trials_per_iteration}"
                )
                site.write()
                ok, reset_reason, start, reset_actions = _stage_empty_s1_start_pose(
                    vision,
                    robot,
                    label=f"empty_s1_iter{iteration}_trial{trial}_reset",
                )
                start_image, start_summary = site.capture(
                    vision,
                    f"empty_s1_iter{iteration}_trial{trial}_start",
                    start,
                )
                if not ok or not _usable_stack_reading(start):
                    already_happy = "already_s1_happy" in str(reset_reason)
                    row = {
                        "iteration": iteration,
                        "trial": trial,
                        "status": "info" if already_happy else "fail",
                        "case_study": not bool(already_happy),
                        "experiment": params["name"],
                        "experiment_line": params["line"],
                        "reason": (
                            "already at S1 happy; not counted as a fresh attempt"
                            if already_happy
                            else f"reset failed: {reset_reason}"
                        ),
                        "start_image": start_image,
                        "end_image": start_image,
                        "start_reading": start_summary,
                        "end_reading": start_summary,
                        "reset_actions": reset_actions,
                        "move_note": (
                            "No S1 move sent because Leia is already at S1; move her to an honest start pose or allow a capped reset."
                            if already_happy
                            else "No S1 move sent because the reset/start pose was not honest."
                        ),
                    }
                    site.add_iteration(row)
                    previous = row
                    site.state["summary"] = (
                        "Stopped: Leia is already S1 happy; no fake repeated wins counted"
                        if already_happy
                        else f"Stopped: start pose was not honest ({reset_reason}); no reverse staging sent"
                    )
                    site.write()
                    print(f"[RESULT] Stopped before trial motion: {reset_reason}", flush=True)
                    return 4 if already_happy else 3
                params = _variant(iteration, previous, start)
                final, move = _one_graceful_move(
                    vision,
                    robot,
                    start,
                    duration_ms=int(params["duration_ms"]),
                    strength=str(params["strength"]),
                    boost=float(params["boost"]),
                )
                end_image, end_summary = site.capture(
                    vision,
                    f"empty_s1_iter{iteration}_trial{trial}_end",
                    final,
                )
                start_dist_gap = _gap_abs(start, "dist")
                start_x_gap = _gap_abs(start, "x")
                end_dist_gap = _gap_abs(final, "dist")
                end_x_gap = _gap_abs(final, "x")
                dist_reduction = None if start_dist_gap is None or end_dist_gap is None else start_dist_gap - end_dist_gap
                x_reduction = None if start_x_gap is None or end_x_gap is None else start_x_gap - end_x_gap
                graceful = bool(
                    _target_met(final)
                    and (dist_reduction is None or dist_reduction >= -0.5)
                    and (x_reduction is None or x_reduction >= -0.5)
                    and int(move.get("mast_attempt_delta", 0)) == 0
                    and not bool(move.get("wrong_way_stop"))
                    and not bool(move.get("virtual_wall_stop"))
                )
                reason = (
                    "single graceful live arc landed in empty S1"
                    if graceful
                    else (
                        "hard-stopped: forward arc widened distance"
                        if bool(move.get("wrong_way_stop"))
                        else (
                            "hard-stopped: virtual wall exceeded"
                            if bool(move.get("virtual_wall_stop"))
                            else "single arc did not land in both dist and x target bands"
                        )
                    )
                )
                move_note = (
                    f"{move['strength']} {move['turn_cmd'].upper()} bias, {move['duration_ms']}ms max, "
                    f"{move['live_samples']} live reads, live_hit={move['live_hit']}; "
                    f"dist gap {_fmt(start_dist_gap)} -> {_fmt(end_dist_gap)}, "
                    f"x gap {_fmt(start_x_gap)} -> {_fmt(end_x_gap)}, "
                    f"x err {_fmt(_axis(start, 'x').get('err_mm'), signed=True)} -> "
                    f"{_fmt(_axis(final, 'x').get('err_mm'), signed=True)}."
                )
                row = {
                    "iteration": iteration,
                    "trial": trial,
                    "status": "win" if graceful else "fail",
                    "case_study": not bool(graceful),
                    "experiment": params["name"],
                    "experiment_line": params["line"],
                    "reason": reason,
                    "start_image": start_image,
                    "end_image": end_image,
                    "start_reading": start_summary,
                    "end_reading": end_summary,
                    "reset_actions": reset_actions,
                    "move": move,
                    "move_note": move_note,
                }
                site.add_iteration(row)
                previous = row
                if not bool(graceful):
                    follow._stop_robot(robot)
                    site.state["summary"] = (
                        f"Stopped at trial {trial}/{trials_per_iteration}: "
                        f"{reason}; using this attempt as the case study"
                    )
                    site.write()
                    print(
                        f"[RESULT] Stopped after failed S1 attempt {trial}/{trials_per_iteration}: {reason}",
                        flush=True,
                    )
                    return 2
        rows = list(site.state.get("iterations") or [])
        wins = sum(1 for item in rows if item.get("status") == "win")
        grouped = {}
        for item in rows:
            grouped.setdefault(int(item.get("iteration", 0) or 0), []).append(item)
        perfect = sum(
            1
            for trials in grouped.values()
            if len(trials) >= trials_per_iteration and all(row.get("status") == "win" for row in trials[:trials_per_iteration])
        )
        site.state["summary"] = f"Done: {wins}/{len(rows)} consecutive S1 wins; {perfect}/{total_iterations} perfect streaks"
        site.write()
        print(f"[RESULT] {wins}/{len(rows)} graceful one-move S1 wins; {perfect}/{total_iterations} perfect streaks", flush=True)
        return 0 if perfect >= total_iterations else 2
    finally:
        if robot is not None:
            try:
                follow._stop_robot(robot)
            except Exception:
                pass
        if base_robot is not None:
            try:
                base_robot.close()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--trials-per-iteration", type=int, default=DEFAULT_TRIALS_PER_ITERATION)
    parser.add_argument("--site-dir", default=str(DEFAULT_SITE_DIR))
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
