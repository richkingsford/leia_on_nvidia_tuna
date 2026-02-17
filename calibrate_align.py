#!/usr/bin/env python3
import json
import random
import time
import statistics
import threading
import cv2
import numpy as np
import os
from pathlib import Path
from collections import deque

from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
import telemetry_robot as telemetry_robot_module
from telemetry_robot import WorldModel, draw_telemetry_overlay, StepState
from telemetry_process import (
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_ORANGE_BRIGHT,
    COLOR_RED,
    COLOR_RESET,
    COLOR_WHITE,
    compute_stream_gate_summary,
    format_control_action_line,
    send_robot_command,
    send_robot_command_pwm,
)
from helper_stream_server import StreamServer, format_stream_url
from helper_gate_utils import load_process_steps, gate_satisfied

# Experiment Config
NUM_TRIALS = 10
DATA_FILE = "world_model_align.json"
RESULTS_FILE = str(
    Path(DATA_FILE).with_name(f"{Path(DATA_FILE).stem}_results{Path(DATA_FILE).suffix}")
)
EFFICIENCY_CHART_FILE = str(
    Path(DATA_FILE).with_name(f"{Path(DATA_FILE).stem}_efficiency.png")
)
ACT_PROGRESS_CHART_FILE = str(
    Path(DATA_FILE).with_name(f"{Path(DATA_FILE).stem}_act_progress.png")
)
ALPHA = 0.25         # Learning rate

# Success Gates (Loaded dynamically)
STEPS = load_process_steps()
ALIGN_GATES = STEPS.get("ALIGN_BRICK", {}).get("success_gates", {})
# Fallbacks if file missing
X_GATE_METRIC = "xAxis_offset_abs"
_X_GATE = ALIGN_GATES.get(X_GATE_METRIC) or ALIGN_GATES.get("xAxis_offset") or {}
try:
    X_GATE_TOL_MM = float(_X_GATE.get("tol", 1.5))
except (TypeError, ValueError):
    X_GATE_TOL_MM = 1.5
try:
    X_GATE_TARGET_MM = float(_X_GATE.get("target", -2.24))
except (TypeError, ValueError):
    X_GATE_TARGET_MM = -2.24
DIST_GATE_METRIC = "dist"
_DIST_GATE = ALIGN_GATES.get(DIST_GATE_METRIC) or {}
try:
    DIST_GATE_TARGET_MM = float(_DIST_GATE.get("target", 100.0))
except (TypeError, ValueError):
    DIST_GATE_TARGET_MM = 100.0
try:
    DIST_GATE_TOL_MM = abs(float(_DIST_GATE.get("tol", 1.5)))
except (TypeError, ValueError):
    DIST_GATE_TOL_MM = 1.5

# Experiment-local X-axis control tuning.
# If left/right feel backwards, flip X_AXIS_SIGN to -1.0 (no other files need changes).
X_AXIS_SIGN = 1.0
X_TARGET_MM = float(X_GATE_TARGET_MM)
X_TOL_MM = float(X_GATE_TOL_MM)
LOG_FLUSH_EVERY_EVENTS = 50
# Distance-normalized X tolerance: if we're farther than (dist target + tol),
# widen the X threshold so alignment remains achievable farther back.
X_TOL_RELAX_MULTIPLIER = 1.4
X_TOL_GROWTH_PER_MM_BACK = 0.05
X_TOL_MAX_MM = 4.0

# Advanced Control Tunings
BACKLASH_S = 0.08         
COARSE_SCORE = 50         # Max coarse speed score (cap)
RESET_MIN_AWAY_MM = 15.0
RESET_MAX_AWAY_MM = 35.0
RESET_SCORE = 1
RESET_MAX_ACTS = 80
RESET_OBSERVE_TIMEOUT_S = 1.5
PRECISION_SCORE_BASE = 1   # Base slow score (we'll bump PWM ~15%)
# Slow/observe speed scaling:
# We previously bumped +15%; this bumps an additional +15% on top => 1.15 * 1.15 = 1.3225 (+32.25%).
PRECISION_BUMP_PCT = 32.25
PRECISION_BUMP_MAX_PCT = 75.0  # Cap for precision bump escalation
PRECISION_ESCALATE_AFTER = 20  # Acts with no progress before bumping
PRECISION_ESCALATE_STEP = 5.0  # Slow ramp while stalled
SETTLE_TIME_S = 0.0       # Pause removed for constant motion
MIN_K = 4.0               
NOISE_FLOOR_MM = 0.2      
POWER_INCREMENT = 10      
LEARN_SPEED_START = 10    # Starting coarse speed (gentler initial exploration)
LEARN_SPEED_STEP = 1      # Always increase every trial (slower ramp)
LEARN_SPEED_BONUS = 2     # Extra increase if trial beats recent average
OBSERVE_SLEEP_S = 0.02    # Small sleep while sampling to keep CPU down
# Require each control decision to use a post-act observation that is at least
# this mature relative to the previous command duration. This reduces acting on
# stale camera samples without introducing long hard sleeps.
POST_ACT_OBSERVE_RATIO = 0.55
POST_ACT_OBSERVE_MIN_S = 0.03
POST_ACT_OBSERVE_MAX_S = 0.25
INTER_TRIAL_SETTLE_S = 0.6
STALL_WINDOW = 3          # Detect stalling after ~2 acts (3 samples)
STALL_BUMP_SCORE = 2      # Add a small speed bump when stalling (coarse mode)
STALL_BUMP_PCT = 5.0      # Extra bump for precision mode when stalling
VISION_LOST_CONSEC_FRAMES = 3     # Trigger loss quickly when no brick frames arrive.
VISION_LOST_MAX_S = 0.35           # Hard cap on blind sampling window.
ALIGN_OBSERVE_TIMEOUT_S = 0.6      # Faster loop cadence for alignment control.
RECOVERY_REVERSE_MAX_ACTS = 6      # How many recent acts to invert for recovery.
RECOVERY_REVERSE_OBSERVE_S = 0.8
RECOVERY_SCAN_MAX_ATTEMPTS = 8
RECOVERY_REVERSE_ALLOWED_PHASES = {"align", "reset"}
RECOVERY_REVERSE_SKIP_REASON_PREFIXES = ("recovery_",)
RECOVERY_REVERSE_SKIP_REASONS = {"between_trials"}
MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES = 3
MAX_ADJUSTS_PER_TRIAL = 20
RECOVERY_REVERSE_SCORE = 6
RECOVERY_SCAN_SCORE = 8
RECOVERY_SCAN_BURST_ACTS = 2
RECOVERY_SCAN_OBSERVE_S = 0.6

class AlignCalibrator:
    def __init__(self):
        self.robot = Robot()
        self.vision = ArucoBrickVision(debug=False)
        self.world = WorldModel()
        self.world.step_state = StepState.ALIGN_BRICK
        self.run_id = int(time.time())
        self.current_trial = None
        self.current_phase = None
        self.act_idx = 0
        self._unsaved_events = 0
        self._last_obs_offset_x = None
        self._last_obs_delta_offset_x = None
        self._last_obs_dist = None
        self._last_obs_delta_dist = None
        self.data_log = []
        self.results_file_path = Path(RESULTS_FILE)
        self.chart_file_path = Path(EFFICIENCY_CHART_FILE)
        self.act_progress_chart_file_path = Path(ACT_PROGRESS_CHART_FILE)
        self.results_log = {}
        self.learned_speed_score = LEARN_SPEED_START
        self.keepalive_dir = "l"
        self.precision_bump_pct = PRECISION_BUMP_PCT
        self.precision_stall_acts = 0
        self.recent_acts = deque(maxlen=32)
        self.recovery_stats = {
            "attempts": 0,
            "recovered": 0,
            "reverse_attempts": 0,
            "reverse_recovered": 0,
            "scan_attempts": 0,
            "scan_recovered": 0,
        }
        self.consecutive_unrecovered_vision_losses = 0
        self.abort_run_early = False
        self.trial_vision_lost_events = 0
        self.trial_recovery_attempts = 0
        self.trial_recovery_successes = 0
        self.trial_adjust_count = 0
        self.trial_progress_acts = 0
        self.trial_worsening_acts = 0
        self._last_logged_abs_x_err = None
        self._last_act_sent_at = None
        self._last_act_duration_s = 0.0
        
        # Histories for smarter learning and metrics
        self.error_history = deque(maxlen=10)
        self.offset_history = deque(maxlen=STALL_WINDOW)
        self.success_frames = 0
        self.REQUIRED_SUCCESS_FRAMES = 3
        
        # Asymmetrical K (L vs R)
        self.K_L = 300.0
        self.K_R = 300.0
        self.last_dir = None
        
        if Path(DATA_FILE).exists():
            try:
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        self.data_log = data
                        last = self.data_log[-1]
                        self.K_L = max(MIN_K, last.get("K_L_after", 300.0))
                        self.K_R = max(MIN_K, last.get("K_R_after", 300.0))
            except: pass

        self.lock = threading.Lock()
        self.current_frame = None
        self.running = True

        # Start Stream Server
        self.server = StreamServer(self._get_frame, port=5000)
        self.server.start()
        print(f"[ALIGN] Livestream: {format_stream_url('127.0.0.1', 5000)}")
        print(
            f"[ALIGN] X gate ({X_GATE_METRIC}): target {X_GATE_TARGET_MM:+.2f}mm tol {X_GATE_TOL_MM:.2f}mm "
            f"(using target {X_TARGET_MM:+.2f} tol {X_TOL_MM:.2f}, sign {X_AXIS_SIGN:+.0f})"
        )
        print(
            f"[ALIGN] Distance gate ({DIST_GATE_METRIC}): target {DIST_GATE_TARGET_MM:.2f}mm tol {DIST_GATE_TOL_MM:.2f}mm "
            f"-> adaptive X tol up to {X_TOL_MAX_MM:.2f}mm when far back "
            f"(base x{X_TOL_RELAX_MULTIPLIER:.2f}, growth {X_TOL_GROWTH_PER_MM_BACK:.3f}/mm)."
        )

    def _get_frame(self):
        with self.lock: return self.current_frame

    def _x_err_mm(self, offset_x):
        try:
            return (float(offset_x) * float(X_AXIS_SIGN)) - float(X_TARGET_MM)
        except (TypeError, ValueError):
            return 0.0

    def _active_x_tol_mm(self, dist_mm):
        base_tol = float(X_TOL_MM) * float(X_TOL_RELAX_MULTIPLIER)
        try:
            dist_val = float(dist_mm)
        except (TypeError, ValueError):
            return float(base_tol)
        start_widen_mm = float(DIST_GATE_TARGET_MM) + float(DIST_GATE_TOL_MM)
        if dist_val <= start_widen_mm:
            return float(base_tol)
        widened = float(base_tol) + ((dist_val - start_widen_mm) * float(X_TOL_GROWTH_PER_MM_BACK))
        return float(max(base_tol, min(float(X_TOL_MAX_MM), widened)))

    def _mark_last_act_timing(self, event):
        if not isinstance(event, dict):
            return
        try:
            sent_at = float(event.get("timestamp"))
        except (TypeError, ValueError):
            sent_at = float(time.time())
        duration_raw = event.get("duration_ms")
        if duration_raw is None:
            duration_raw = event.get("duration_model_ms")
        try:
            duration_s = max(0.0, float(duration_raw) / 1000.0)
        except (TypeError, ValueError):
            duration_s = 0.0
        self._last_act_sent_at = float(sent_at)
        self._last_act_duration_s = float(duration_s)

    def _observe_not_before_ts(self):
        if self._last_act_sent_at is None:
            return None
        settle_s = float(self._last_act_duration_s) * float(POST_ACT_OBSERVE_RATIO)
        settle_s = max(float(POST_ACT_OBSERVE_MIN_S), settle_s)
        settle_s = min(float(POST_ACT_OBSERVE_MAX_S), settle_s)
        return float(self._last_act_sent_at) + float(settle_s)

    def _flush_logs(self, force=False):
        if not force and self._unsaved_events < LOG_FLUSH_EVERY_EVENTS:
            return
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(self.data_log, f, indent=(2 if force else None))
            self._unsaved_events = 0
        except Exception as e:
            print(f"[LOG] Failed to write {DATA_FILE}: {e}")

    def _append_event(self, event):
        if not isinstance(event, dict):
            return
        self.data_log.append(event)
        self._unsaved_events += 1
        self._flush_logs(force=False)

    def _init_results_log(self):
        self.results_log = {
            "schema_version": 1,
            "run_id": int(self.run_id),
            "timestamp": float(time.time()),
            "data_file": str(Path(DATA_FILE)),
            "results_file": str(self.results_file_path),
            "efficiency_chart_file": str(self.chart_file_path),
            "act_progress_chart_file": str(self.act_progress_chart_file_path),
            "config": {
                "num_trials": int(NUM_TRIALS),
                "x_gate_metric": str(X_GATE_METRIC),
                "x_gate_target_mm": float(X_GATE_TARGET_MM),
                "x_gate_tol_mm": float(X_GATE_TOL_MM),
                "dist_gate_metric": str(DIST_GATE_METRIC),
                "dist_gate_target_mm": float(DIST_GATE_TARGET_MM),
                "dist_gate_tol_mm": float(DIST_GATE_TOL_MM),
                "x_axis_sign": float(X_AXIS_SIGN),
                "x_target_mm": float(X_TARGET_MM),
                "x_tol_mm": float(X_TOL_MM),
                "x_tol_relax_multiplier": float(X_TOL_RELAX_MULTIPLIER),
                "x_tol_growth_per_mm_back": float(X_TOL_GROWTH_PER_MM_BACK),
                "x_tol_max_mm": float(X_TOL_MAX_MM),
                "learn_speed_start": int(LEARN_SPEED_START),
                "learn_speed_step": int(LEARN_SPEED_STEP),
                "learn_speed_bonus": int(LEARN_SPEED_BONUS),
                "vision_lost_consec_frames": int(VISION_LOST_CONSEC_FRAMES),
                "vision_lost_max_s": float(VISION_LOST_MAX_S),
                "align_observe_timeout_s": float(ALIGN_OBSERVE_TIMEOUT_S),
                "post_act_observe_ratio": float(POST_ACT_OBSERVE_RATIO),
                "post_act_observe_min_s": float(POST_ACT_OBSERVE_MIN_S),
                "post_act_observe_max_s": float(POST_ACT_OBSERVE_MAX_S),
                "inter_trial_settle_s": float(INTER_TRIAL_SETTLE_S),
                "recovery_reverse_max_acts": int(RECOVERY_REVERSE_MAX_ACTS),
                "recovery_reverse_score": int(RECOVERY_REVERSE_SCORE),
                "recovery_scan_max_attempts": int(RECOVERY_SCAN_MAX_ATTEMPTS),
                "recovery_scan_score": int(RECOVERY_SCAN_SCORE),
                "recovery_scan_burst_acts": int(RECOVERY_SCAN_BURST_ACTS),
                "recovery_scan_observe_s": float(RECOVERY_SCAN_OBSERVE_S),
                "max_consecutive_unrecovered_vision_losses": int(MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES),
                "max_adjusts_per_trial": int(MAX_ADJUSTS_PER_TRIAL),
            },
            "trials": [],
            "recovery_summary": None,
            "efficiency_chart": None,
            "act_progress_chart": None,
            "status": "running",
            "completed_trials": 0,
            "early_stop": False,
            "early_stop_reason": None,
            "finished_at": None,
        }

    def _write_results(self):
        try:
            self.results_file_path.write_text(json.dumps(self.results_log, indent=2) + "\n")
        except Exception as e:
            print(f"[LOG] Failed to write {self.results_file_path}: {e}")

    def log_act(self, cmd, mode, score=None, bump_pct=None, reason=None, extra=None):
        offset_x = self._last_obs_offset_x
        dist = self._last_obs_dist
        err_mm = self._x_err_mm(offset_x)
        x_tol_active = self._active_x_tol_mm(dist)
        hw_cmd = None
        try:
            hw_cmd = (getattr(telemetry_robot_module, "COMMAND_REMAP", {}) or {}).get(cmd, cmd)
        except Exception:
            hw_cmd = cmd
        event = {
            "type": "act",
            "run_id": self.run_id,
            "trial": self.current_trial,
            "phase": self.current_phase,
            "act": self.act_idx,
            "timestamp": time.time(),
            "offset_x": offset_x,
            "delta_offset_x": self._last_obs_delta_offset_x,
            "dist": dist,
            "delta_dist": self._last_obs_delta_dist,
            "x_err_mm": err_mm,
            "target_mm": X_TARGET_MM,
            "x_tol_base_mm": float(X_TOL_MM),
            "x_tol_active_mm": float(x_tol_active),
            "cmd": cmd,
            "hw_cmd": hw_cmd,
            "mode": mode,
            "score": score,
            "precision_bump_pct": bump_pct,
            "reason": reason,
        }
        if isinstance(extra, dict):
            event.update(extra)
        hist_score = None
        if isinstance(extra, dict):
            hist_score = extra.get("score_requested")
        if hist_score is None:
            hist_score = score
        try:
            hist_score = int(round(float(hist_score))) if hist_score is not None else None
        except (TypeError, ValueError):
            hist_score = None
        try:
            hist_bump = float(bump_pct) if bump_pct is not None else None
        except (TypeError, ValueError):
            hist_bump = None
        self.recent_acts.append(
            {
                "cmd": str(cmd),
                "mode": str(mode),
                "score": hist_score,
                "bump_pct": hist_bump,
                "reason": str(reason) if reason else None,
                "phase": self.current_phase,
                "timestamp": event["timestamp"],
            }
        )
        self._mark_last_act_timing(event)
        self._emit_act_console(event)
        self._append_event(event)
        self.act_idx += 1
        return event

    def _emit_act_console(self, event):
        if not isinstance(event, dict):
            return
        cmd = event.get("cmd")
        phase = str(event.get("phase") or "?")
        reason = str(event.get("reason") or "")
        phase_key = phase.strip().lower()
        reason_key = reason.strip().lower()
        # Suppress noisy setup logs between trials; keep data in JSON events.
        if phase_key in {"reset", "reset_recovery"} or reason_key in {"between_trials"}:
            return
        trial = event.get("trial")
        act_idx = event.get("act")

        score_for_display = event.get("score_requested")
        if score_for_display is None:
            score_for_display = event.get("score")
        if score_for_display is None:
            score_for_display = event.get("score_model")

        cmd_sent = event.get("cmd_sent")
        cmd_plan_text = str(cmd).upper() if cmd is not None else "?"
        cmd_sent_text = str(cmd_sent).upper() if cmd_sent is not None else cmd_plan_text
        cmd_token = cmd_sent_text or cmd_plan_text

        approx_score = None
        try:
            if score_for_display is not None:
                approx_score = int(round(float(score_for_display)))
        except (TypeError, ValueError):
            approx_score = None
        if approx_score is None:
            try:
                power_val = float(event.get("power"))
                cmd_for_quant = str(cmd_sent if cmd_sent is not None else cmd).strip().lower()
                if cmd_for_quant:
                    _, q_score = telemetry_robot_module.quantize_speed(cmd_for_quant, speed=power_val)
                    approx_score = int(round(float(q_score)))
            except Exception:
                approx_score = None
        score_text = f"{int(approx_score)}%" if approx_score is not None else "?%"

        detail_items = [f"~{score_text}"]
        if cmd is not None and cmd_sent is not None:
            cmd_plan_lower = str(cmd).strip().lower()
            cmd_sent_lower = str(cmd_sent).strip().lower()
            if cmd_plan_lower and cmd_sent_lower and cmd_plan_lower != cmd_sent_lower:
                detail_items.append(f"plan={cmd_plan_text}")
        if cmd_sent is not None:
            detail_items.append(f"sent={cmd_sent_text}")
        try:
            pwm_val = int(round(float(event.get("pwm"))))
            detail_items.append(f"pwm={pwm_val}")
        except (TypeError, ValueError):
            pass
        try:
            power_val = float(event.get("power"))
            detail_items.append(f"power={power_val:.3f}")
        except (TypeError, ValueError):
            pass
        duration_raw = event.get("duration_ms")
        if duration_raw is None:
            duration_raw = event.get("duration_model_ms")
        try:
            duration_ms = int(round(float(duration_raw)))
            detail_items.append(f"{duration_ms}ms")
        except (TypeError, ValueError):
            pass
        if len(detail_items) > 1:
            sent_details = f"({detail_items[0]} | " + ", ".join(detail_items[1:]) + ")"
        else:
            sent_details = f"({detail_items[0]})"

        phase_color = COLOR_RED if phase.lower().startswith("recovery") else COLOR_WHITE
        print(
            f"{phase_color}[ACT #{int(act_idx) if act_idx is not None else -1}] "
            f"T{int(trial) if trial is not None else -1} {phase.upper()}{COLOR_RESET} "
            f"{COLOR_ORANGE_BRIGHT}{cmd_token} {score_text}{COLOR_RESET} "
            f"{COLOR_WHITE}{sent_details}{COLOR_RESET}"
        )

        try:
            x_err_mm = float(event.get("x_err_mm"))
        except (TypeError, ValueError):
            x_err_mm = None
        try:
            offset_x = float(event.get("offset_x"))
        except (TypeError, ValueError):
            offset_x = None
        try:
            dist_mm = float(event.get("dist"))
        except (TypeError, ValueError):
            dist_mm = None
        curr_abs_err = abs(float(x_err_mm)) if x_err_mm is not None else None
        prev_abs_err = self._last_logged_abs_x_err
        delta_abs_err = None
        offset_color = COLOR_WHITE
        if curr_abs_err is not None and prev_abs_err is not None:
            delta_abs_err = float(prev_abs_err) - float(curr_abs_err)
            if delta_abs_err > 0.02:
                offset_color = COLOR_GREEN
            elif delta_abs_err < -0.02:
                offset_color = COLOR_RED
            else:
                offset_color = COLOR_WHITE

        reason_line = f"{COLOR_GRAY}reason={reason or 'n/a'}"
        if x_err_mm is not None:
            reason_line += f" | x_err={x_err_mm:+.2f}mm"
        try:
            x_tol_active = float(event.get("x_tol_active_mm"))
        except (TypeError, ValueError):
            x_tol_active = float(X_TOL_MM)
        reason_line += f" | x_tol={x_tol_active:.2f}mm"
        if offset_x is not None:
            reason_line += f" | offset={offset_color}{offset_x:+.2f}{COLOR_GRAY}mm"
            if delta_abs_err is not None:
                direction = "better" if delta_abs_err > 0.02 else ("worse" if delta_abs_err < -0.02 else "flat")
                reason_line += f" ({direction} {abs(delta_abs_err):.2f}mm)"
        if dist_mm is not None:
            reason_line += f" | dist={dist_mm:.1f}mm"
        reason_line += f"{COLOR_RESET}"
        print(f"      {reason_line}")

        try:
            power_val = float(event.get("power"))
        except (TypeError, ValueError):
            power_val = telemetry_robot_module.manual_speed_for_cmd(str(cmd), PRECISION_SCORE_BASE)
        metric_reason = (
            f"x_axis:{abs(float(x_err_mm)):.2f}mm"
            if x_err_mm is not None
            else "x_axis:unknown"
        )
        friendly = format_control_action_line(str(cmd), power_val, reason=metric_reason)
        for line in str(friendly).splitlines():
            print(f"      {line}")

        if curr_abs_err is not None:
            self._last_logged_abs_x_err = float(curr_abs_err)

    def send_coarse_command(self, cmd, score, reason=None, extra=None):
        sent = send_robot_command(
            self.robot,
            self.world,
            StepState.ALIGN_BRICK,
            cmd,
            0.0,
            speed_score=score,
        )
        if not isinstance(sent, dict):
            print(
                f"{COLOR_RED}[ACT] send_command failed: cmd={cmd} score={score}% "
                f"phase={self.current_phase} reason={reason}{COLOR_RESET}"
            )
            return
        merged = {"score_requested": score}
        merged.update(sent)
        if isinstance(extra, dict):
            merged.update(extra)
        return self.log_act(cmd, mode="coarse", score=sent.get("score_model"), reason=reason, extra=merged)

    def _precision_scaled_power(self, cmd, bump_pct=None):
        bump = self.precision_bump_pct if bump_pct is None else bump_pct
        base_power = telemetry_robot_module.manual_speed_for_cmd(cmd, PRECISION_SCORE_BASE)
        cap_power = telemetry_robot_module.manual_speed_for_cmd(cmd, COARSE_SCORE)
        return min(base_power * (1.0 + (bump / 100.0)), cap_power)

    def _precision_scaled_pwm(self, cmd, bump_pct=None):
        bump = self.precision_bump_pct if bump_pct is None else bump_pct
        _, base_pwm, _, duration_ms = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, PRECISION_SCORE_BASE)
        _, cap_pwm, _, _ = telemetry_robot_module.speed_power_pwm_for_cmd(cmd, COARSE_SCORE)
        scaled_pwm = int(round(base_pwm * (1.0 + (bump / 100.0))))
        return min(scaled_pwm, cap_pwm), duration_ms

    def send_precision_command(self, cmd, bump_pct=None, reason=None, extra=None):
        bump_used = self.precision_bump_pct if bump_pct is None else float(bump_pct)
        pwm, duration_ms = self._precision_scaled_pwm(cmd, bump_pct=bump_used)
        power = self._precision_scaled_power(cmd, bump_pct=bump_used)
        sent = send_robot_command_pwm(
            self.robot,
            self.world,
            StepState.ALIGN_BRICK,
            cmd,
            power,
            pwm,
            duration_ms,
        )
        if not isinstance(sent, dict):
            print(
                f"{COLOR_RED}[ACT] send_command failed: cmd={cmd} bump={bump_used:.1f}% "
                f"phase={self.current_phase} reason={reason}{COLOR_RESET}"
            )
            return
        merged = {"bump_pct": bump_used, "pwm_requested": pwm}
        merged.update(sent)
        if isinstance(extra, dict):
            merged.update(extra)
        return self.log_act(cmd, mode="precision", bump_pct=bump_used, reason=reason, extra=merged)

    def update_world(self, found, angle, dist, offset_x, conf, cam_h, above, below, frame):
        self.world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
        if found:
            try:
                offset_val = float(offset_x)
            except (TypeError, ValueError):
                offset_val = None
            try:
                dist_val = float(dist)
            except (TypeError, ValueError):
                dist_val = None

            if offset_val is None:
                self._last_obs_delta_offset_x = None
            elif self._last_obs_offset_x is None:
                self._last_obs_delta_offset_x = None
            else:
                try:
                    self._last_obs_delta_offset_x = float(offset_val) - float(self._last_obs_offset_x)
                except (TypeError, ValueError):
                    self._last_obs_delta_offset_x = None

            if dist_val is None:
                self._last_obs_delta_dist = None
            elif self._last_obs_dist is None:
                self._last_obs_delta_dist = None
            else:
                try:
                    self._last_obs_delta_dist = float(dist_val) - float(self._last_obs_dist)
                except (TypeError, ValueError):
                    self._last_obs_delta_dist = None

            self._last_obs_offset_x = offset_val
            self._last_obs_dist = dist_val
        else:
            self._last_obs_delta_offset_x = None
            self._last_obs_delta_dist = None
            self._last_obs_offset_x = None
            self._last_obs_dist = None
        
        # Draw Overlay
        gate_summary, analytics = compute_stream_gate_summary(self.world, "ALIGN_BRICK", active=True)
        
        # In-Line Enhancement for Livestream
        try:
            offset_disp = f"{float(offset_x):+.1f}mm"
        except (TypeError, ValueError):
            offset_disp = "--"
        x_err = self._x_err_mm(offset_x)
        active_x_tol = self._active_x_tol_mm(dist)
        x_err_disp = f"{x_err:+.1f}mm" if found else "--"
        if not found:
            status = "LOST"
        else:
            status = "OK" if abs(x_err) <= active_x_tol else "PENDING"
        extra = [
            f"X_OFF: {offset_disp}",
            f"X_ERR: {x_err_disp}",
            f"X_TOL: {active_x_tol:.2f}mm (base {X_TOL_MM:.2f})",
            f"DIST: {dist:.0f}mm" if found else "DIST: --",
            f"GATE: {status}"
        ]
        
        display = frame.copy()
        draw_telemetry_overlay(
            display, 
            self.world, 
            gate_summary=gate_summary,
            extra_messages=extra,
            highlight_metric=X_GATE_METRIC
        )
        
        with self.lock: self.current_frame = display

    def get_stable_pose(
        self,
        samples=3,
        timeout_s=2.0,
        min_sample_time=None,
    ):
        """
        Reads 'samples' unique frames and returns averaged (offset_x, angle).
        Ensures consistency between samples.
        """
        poses = []
        start_t = time.time()
        blind_frames = 0
        blind_start_t = None
        while len(poses) < samples and (time.time() - start_t < timeout_s) and self.running:
            found, angle, dist, offset_x, conf, cam_h, above, below = self.vision.read()
            # Update world model for the livestream
            self.update_world(found, angle, dist, offset_x, conf, cam_h, above, below, self.vision.current_frame)
            
            if found:
                blind_frames = 0
                blind_start_t = None
                # Ensure we have a unique frame (or significantly different)
                try:
                    off_val = float(offset_x)
                except (TypeError, ValueError):
                    off_val = None
                try:
                    dist_val = float(dist)
                except (TypeError, ValueError):
                    dist_val = None
                if off_val is None or dist_val is None:
                    time.sleep(OBSERVE_SLEEP_S)
                    continue
                sample_ts = float(time.time())
                if min_sample_time is not None:
                    try:
                        min_ts = float(min_sample_time)
                    except (TypeError, ValueError):
                        min_ts = None
                    if min_ts is not None and sample_ts < min_ts:
                        time.sleep(OBSERVE_SLEEP_S)
                        continue
                if not poses:
                    poses.append(
                        {
                            "offset_x": off_val,
                            "dist": dist_val,
                            "angle": angle,
                            "confidence": conf,
                            "obs_ts": sample_ts,
                        }
                    )
                else:
                    last = poses[-1]
                    if abs(off_val - last.get("offset_x", off_val)) > 0.1 or abs(dist_val - last.get("dist", dist_val)) > 0.5:
                        poses.append(
                            {
                                "offset_x": off_val,
                                "dist": dist_val,
                                "angle": angle,
                                "confidence": conf,
                                "obs_ts": sample_ts,
                            }
                        )
                if len(poses) < samples:
                    time.sleep(OBSERVE_SLEEP_S) 
            else: 
                blind_frames += 1
                if blind_start_t is None:
                    blind_start_t = time.time()
                blind_seconds = float(time.time() - float(blind_start_t))
                if blind_frames >= int(VISION_LOST_CONSEC_FRAMES) or blind_seconds >= float(VISION_LOST_MAX_S):
                    self.robot.stop()
                    self.trial_vision_lost_events = int(self.trial_vision_lost_events + 1)
                    self._append_event(
                        {
                            "type": "vision_lost_detected",
                            "run_id": self.run_id,
                            "trial": self.current_trial,
                            "phase": self.current_phase,
                            "timestamp": time.time(),
                            "blind_frames": int(blind_frames),
                            "blind_seconds": float(blind_seconds),
                        }
                    )
                    break
                if len(poses) < samples:
                    time.sleep(OBSERVE_SLEEP_S)
        
        if not poses or not self.running: return None

        # Average results
        avg_offset = statistics.mean([p["offset_x"] for p in poses])
        avg_dist = statistics.mean([p["dist"] for p in poses])
        obs_ts = max(float(p.get("obs_ts", start_t)) for p in poses)
        
        return {"offset_x": avg_offset, "dist": avg_dist, "obs_ts": float(obs_ts)}

    @staticmethod
    def _inverse_cmd(cmd):
        return {
            "l": "r",
            "r": "l",
            "f": "b",
            "b": "f",
            "u": "d",
            "d": "u",
        }.get(str(cmd))

    def recovery_reverse_recent(self, max_acts=RECOVERY_REVERSE_MAX_ACTS, *, emit_logs=True):
        history = list(self.recent_acts)
        if not history:
            return {"attempted": False, "recovered": False, "acts_attempted": 0}
        plan = []
        for row in reversed(history[-int(max_acts):]):
            phase = str(row.get("phase") or "").lower()
            if phase not in RECOVERY_REVERSE_ALLOWED_PHASES:
                continue
            reason = str(row.get("reason") or "")
            if reason in RECOVERY_REVERSE_SKIP_REASONS:
                continue
            if any(reason.startswith(prefix) for prefix in RECOVERY_REVERSE_SKIP_REASON_PREFIXES):
                continue
            inv = self._inverse_cmd(row.get("cmd"))
            if inv is None:
                continue
            plan.append(
                {
                    "cmd": str(inv),
                    "origin_cmd": str(row.get("cmd")),
                    "mode": str(row.get("mode") or "precision"),
                    "score": row.get("score"),
                    "bump_pct": row.get("bump_pct"),
                }
            )
        if not plan:
            return {"attempted": False, "recovered": False, "acts_attempted": 0}

        self._append_event(
            {
                "type": "recovery_reverse_start",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "phase": self.current_phase,
                "timestamp": time.time(),
                "plan": plan,
            }
        )
        if emit_logs:
            print("[RECOVERY] Reversing recent acts...")
        for idx, step in enumerate(plan, start=1):
            mode = str(step.get("mode") or "precision")
            cmd = str(step["cmd"])
            score = step.get("score")
            try:
                score = int(round(float(score)))
            except (TypeError, ValueError):
                score = int(RESET_SCORE)
            recovery_score = max(int(RECOVERY_REVERSE_SCORE), int(score))
            if emit_logs:
                print(
                    f"[RECOVERY] reverse step {idx}/{len(plan)}: cmd={cmd.upper()} "
                    f"score={int(recovery_score)}%"
                )
            event = self.send_coarse_command(
                cmd,
                score=max(1, min(COARSE_SCORE, int(recovery_score))),
                reason="recovery_reverse",
                extra={
                    "recovery_index": int(idx),
                    "recovery_total": int(len(plan)),
                    "origin_cmd": str(step.get("origin_cmd")),
                    "origin_mode": str(mode),
                    "origin_bump_pct": step.get("bump_pct"),
                },
            )
            if not isinstance(event, dict):
                self._append_event(
                    {
                        "type": "recovery_reverse_send_failed",
                        "run_id": self.run_id,
                        "trial": self.current_trial,
                        "phase": self.current_phase,
                        "timestamp": time.time(),
                        "index": int(idx),
                        "total": int(len(plan)),
                        "cmd": cmd,
                        "mode": mode,
                    }
                )

            pose = self.get_stable_pose(
                samples=1,
                timeout_s=float(RECOVERY_REVERSE_OBSERVE_S),
                min_sample_time=self._observe_not_before_ts(),
            )
            found = pose is not None
            self._append_event(
                {
                    "type": "recovery_reverse_act",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "phase": self.current_phase,
                    "timestamp": time.time(),
                    "index": int(idx),
                    "total": int(len(plan)),
                    "cmd": cmd,
                    "mode": mode,
                    "origin_cmd": str(step.get("origin_cmd")),
                    "found_after_act": bool(found),
                    "offset_x": (float(pose["offset_x"]) if found else None),
                }
            )
            if found:
                self._append_event(
                    {
                        "type": "recovery_reverse_end",
                        "run_id": self.run_id,
                        "trial": self.current_trial,
                        "phase": self.current_phase,
                        "timestamp": time.time(),
                        "recovered": True,
                        "acts_attempted": int(idx),
                    }
                )
                if emit_logs:
                    print(f"[RECOVERY] Recovered by reversing acts (step {idx}/{len(plan)}).")
                return {"attempted": True, "recovered": True, "acts_attempted": int(idx)}

        self._append_event(
            {
                "type": "recovery_reverse_end",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "phase": self.current_phase,
                "timestamp": time.time(),
                "recovered": False,
                "acts_attempted": int(len(plan)),
            }
        )
        return {"attempted": True, "recovered": False, "acts_attempted": int(len(plan))}

    def run_reset_phase(self):
        """
        Deliberate, vision-driven reset:
          - Pick a random desired abs(x_err) away from the ALIGN gate target.
          - Turn in a direction and validate via vision after each act.
          - Log each reset act with before/after observations (offset + dist).
        """
        pose_start = self.get_stable_pose(
            samples=1,
            timeout_s=2.0,
            min_sample_time=self._observe_not_before_ts(),
        )
        if not pose_start:
            return None

        start_off = float(pose_start["offset_x"])
        start_dist = float(pose_start.get("dist", 0.0))
        start_err = float(self._x_err_mm(start_off))
        desired_abs_err = float(random.uniform(RESET_MIN_AWAY_MM, RESET_MAX_AWAY_MM))

        reset_dir = random.choice(["l", "r"])
        self.keepalive_dir = reset_dir

        self._append_event(
            {
                "type": "reset_start",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "start_offset_x": start_off,
                "start_x_err_mm": start_err,
                "start_abs_err_mm": abs(start_err),
                "start_dist": start_dist,
                "goal_abs_err_mm": desired_abs_err,
                "reset_score": int(RESET_SCORE),
                "reset_dir_initial": reset_dir,
            }
        )

        reset_acts = 0
        last_pose = pose_start
        last_abs_err = abs(start_err)
        dist_samples = [start_dist]
        reset_failed_vision = False
        while self.running and reset_acts < int(RESET_MAX_ACTS):
            # Observe BEFORE deciding the next act (no keepalive so we don't double-command).
            pre = self.get_stable_pose(
                samples=1,
                timeout_s=RESET_OBSERVE_TIMEOUT_S,
                min_sample_time=self._observe_not_before_ts(),
            )
            if not pre:
                self.current_phase = "reset_recovery"
                if not self.recover_visibility(source="reset_pre"):
                    reset_failed_vision = True
                    self.current_phase = "reset"
                    break
                self.current_phase = "reset"
                continue

            off_before = float(pre["offset_x"])
            dist_before = float(pre.get("dist", 0.0))
            x_err_before = float(self._x_err_mm(off_before))
            abs_err_before = abs(x_err_before)
            dist_samples.append(dist_before)

            last_pose = pre
            last_abs_err = abs_err_before
            if abs_err_before >= desired_abs_err:
                break

            extra = {
                "goal_abs_err_mm": desired_abs_err,
                "offset_x_before": off_before,
                "x_err_mm_before": x_err_before,
                "abs_err_mm_before": abs_err_before,
                "dist_before": dist_before,
                "reset_act_idx": int(reset_acts),
            }
            event = self.send_coarse_command(reset_dir, score=int(RESET_SCORE), reason="reset", extra=extra)
            reset_acts += 1
            time.sleep(max(0.02, float(BACKLASH_S)))

            post = self.get_stable_pose(
                samples=1,
                timeout_s=RESET_OBSERVE_TIMEOUT_S,
                min_sample_time=self._observe_not_before_ts(),
            )
            if not post:
                if isinstance(event, dict):
                    event.update(
                        {
                            "obs_after_found": False,
                            "result": "vision_lost_after_act",
                        }
                    )
                self.current_phase = "reset_recovery"
                if not self.recover_visibility(source="reset_post"):
                    reset_failed_vision = True
                    self.current_phase = "reset"
                    break
                self.current_phase = "reset"
                continue

            off_after = float(post["offset_x"])
            dist_after = float(post.get("dist", 0.0))
            x_err_after = float(self._x_err_mm(off_after))
            abs_err_after = abs(x_err_after)
            dist_samples.append(dist_after)

            if isinstance(event, dict):
                event.update(
                    {
                        "obs_after_found": True,
                        "offset_x_after": off_after,
                        "x_err_mm_after": x_err_after,
                        "abs_err_mm_after": abs_err_after,
                        "dist_after": dist_after,
                        "delta_offset_x_act": off_after - off_before,
                        "delta_x_err_mm_act": x_err_after - x_err_before,
                        "delta_abs_err_mm_act": abs_err_after - abs_err_before,
                        "delta_dist_act": dist_after - dist_before,
                    }
                )

            # Direction sanity: if we accidentally moved closer to the gate, flip.
            if abs_err_after < abs_err_before:
                reset_dir = "r" if reset_dir == "l" else "l"
                self.keepalive_dir = reset_dir

            last_pose = post
            last_abs_err = abs_err_after
            if abs_err_after >= desired_abs_err:
                break

        self._append_event(
            {
                "type": "reset_end",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "acts": int(reset_acts),
                "goal_abs_err_mm": desired_abs_err,
                "final_offset_x": float(last_pose.get("offset_x")) if last_pose else None,
                "final_x_err_mm": float(self._x_err_mm(last_pose.get("offset_x"))) if last_pose else None,
                "final_abs_err_mm": float(last_abs_err),
                "dist_min": float(min(dist_samples)) if dist_samples else None,
                "dist_max": float(max(dist_samples)) if dist_samples else None,
                "dist_avg": float(statistics.mean(dist_samples)) if dist_samples else None,
                "reset_dir_final": reset_dir,
                "reset_score": int(RESET_SCORE),
                "failed_vision": bool(reset_failed_vision),
            }
        )

        if bool(reset_failed_vision):
            return None
        return last_pose

    def recovery_scan(self, max_attempts=RECOVERY_SCAN_MAX_ATTEMPTS, *, emit_logs=True):
        """Slowly scan nearby area to find the brick again."""
        if emit_logs:
            print("[RECOVERY] Sight lost. Scanning nearby area...")
        # Sweep pattern: two acts each direction so we avoid tiny reversal pulses.
        for i in range(1, max_attempts + 1):
            block = int((i - 1) // int(max(1, RECOVERY_SCAN_BURST_ACTS)))
            scan_dir = 'l' if (block % 2 == 0) else 'r'
            self.keepalive_dir = scan_dir
            burst_idx = int(((i - 1) % int(max(1, RECOVERY_SCAN_BURST_ACTS))) + 1)
            if emit_logs:
                print(
                    f"[RECOVERY] scan act {i}/{max_attempts}: cmd={scan_dir.upper()} "
                    f"score={int(RECOVERY_SCAN_SCORE)}% burst={burst_idx}/{int(max(1, RECOVERY_SCAN_BURST_ACTS))}"
                )
            event = self.send_coarse_command(
                scan_dir,
                score=int(RECOVERY_SCAN_SCORE),
                reason="recovery_scan",
                extra={
                    "scan_attempt": int(i),
                    "scan_total": int(max_attempts),
                    "scan_burst_index": int(burst_idx),
                    "scan_burst_size": int(max(1, RECOVERY_SCAN_BURST_ACTS)),
                },
            )
            if not isinstance(event, dict):
                self._append_event(
                    {
                        "type": "recovery_scan_send_failed",
                        "run_id": self.run_id,
                        "trial": self.current_trial,
                        "phase": self.current_phase,
                        "timestamp": time.time(),
                        "scan_attempt": int(i),
                        "scan_total": int(max_attempts),
                        "cmd": str(scan_dir),
                    }
                )
            if self.get_stable_pose(
                samples=1,
                timeout_s=float(RECOVERY_SCAN_OBSERVE_S),
                min_sample_time=self._observe_not_before_ts(),
            ):
                if emit_logs:
                    print(f"[RECOVERY] Found brick again on scan act {i}!")
                return True
        return False

    def recover_visibility(self, source="unknown"):
        source_key = str(source or "").strip().lower()
        emit_recovery_logs = not source_key.startswith("reset")
        self.recovery_stats["attempts"] = int(self.recovery_stats.get("attempts", 0) + 1)
        self.trial_recovery_attempts = int(self.trial_recovery_attempts + 1)
        self._append_event(
            {
                "type": "recovery_start",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "phase": self.current_phase,
                "timestamp": time.time(),
                "source": str(source),
            }
        )
        self.robot.stop()
        reverse_result = self.recovery_reverse_recent(emit_logs=emit_recovery_logs)
        reverse_attempted = bool(reverse_result.get("attempted")) if isinstance(reverse_result, dict) else False
        reverse_recovered = bool(reverse_result.get("recovered")) if isinstance(reverse_result, dict) else False
        if reverse_attempted:
            self.recovery_stats["reverse_attempts"] = int(self.recovery_stats.get("reverse_attempts", 0) + 1)
            if reverse_recovered:
                self.recovery_stats["reverse_recovered"] = int(self.recovery_stats.get("reverse_recovered", 0) + 1)

        recovered = bool(reverse_recovered)
        scan_attempted = False
        scan_recovered = False
        if not recovered:
            scan_attempted = True
            self.recovery_stats["scan_attempts"] = int(self.recovery_stats.get("scan_attempts", 0) + 1)
            scan_recovered = bool(self.recovery_scan(emit_logs=emit_recovery_logs))
            recovered = bool(scan_recovered)
            if scan_recovered:
                self.recovery_stats["scan_recovered"] = int(self.recovery_stats.get("scan_recovered", 0) + 1)
        if recovered:
            self.recovery_stats["recovered"] = int(self.recovery_stats.get("recovered", 0) + 1)
            self.trial_recovery_successes = int(self.trial_recovery_successes + 1)
            self.consecutive_unrecovered_vision_losses = 0
        else:
            self.consecutive_unrecovered_vision_losses = int(self.consecutive_unrecovered_vision_losses + 1)
            if (
                self.consecutive_unrecovered_vision_losses >= int(MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES)
                and not bool(self.abort_run_early)
            ):
                self.abort_run_early = True
                self._append_event(
                    {
                        "type": "run_early_stop",
                        "run_id": self.run_id,
                        "trial": self.current_trial,
                        "phase": self.current_phase,
                        "timestamp": time.time(),
                        "reason": "consecutive_unrecovered_vision_losses",
                        "count": int(self.consecutive_unrecovered_vision_losses),
                        "threshold": int(MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES),
                    }
                )
                print(
                    "[ALIGN] Early stop armed: "
                    f"{int(self.consecutive_unrecovered_vision_losses)} consecutive unrecovered vision losses "
                    f"(threshold {int(MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES)})."
                )
        self._append_event(
            {
                "type": "recovery_end",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "phase": self.current_phase,
                "timestamp": time.time(),
                "source": str(source),
                "recovered": bool(recovered),
                "reverse_attempted": bool(reverse_attempted),
                "reverse_recovered": bool(reverse_recovered),
                "scan_attempted": bool(scan_attempted),
                "scan_recovered": bool(scan_recovered),
            }
        )
        return bool(recovered)

    def print_trial_recovery_summary(self, trial_idx, *, success, reason):
        attempts = int(self.trial_recovery_attempts)
        recovered = int(self.trial_recovery_successes)
        vision_lost = int(self.trial_vision_lost_events)
        adjusts = int(self.trial_adjust_count)
        print(
            f"[TRIAL] {int(trial_idx)} summary: success={bool(success)} reason={str(reason)} | "
            f"vision_lost={vision_lost} | recovery_used={attempts} | recovered={recovered}/{attempts} | "
            f"adjusts={adjusts}/{int(MAX_ADJUSTS_PER_TRIAL)} | "
            f"progress={int(self.trial_progress_acts)} worsening={int(self.trial_worsening_acts)}"
        )
        self._append_event(
            {
                "type": "trial_summary",
                "run_id": self.run_id,
                "trial": int(trial_idx),
                "timestamp": time.time(),
                "success": bool(success),
                "reason": str(reason),
                "vision_lost_events": int(vision_lost),
                "recovery_attempts": int(attempts),
                "recovery_recovered": int(recovered),
                "adjust_count": int(adjusts),
                "adjust_limit": int(MAX_ADJUSTS_PER_TRIAL),
                "progress_acts": int(self.trial_progress_acts),
                "worsening_acts": int(self.trial_worsening_acts),
            }
        )
        if isinstance(self.results_log, dict):
            trial_row = {
                "trial": int(trial_idx),
                "timestamp": float(time.time()),
                "success": bool(success),
                "reason": str(reason),
                "vision_lost_events": int(vision_lost),
                "recovery_attempts": int(attempts),
                "recovery_recovered": int(recovered),
                "learned_speed_score": int(self.learned_speed_score),
                "adjust_count": int(adjusts),
                "adjust_limit": int(MAX_ADJUSTS_PER_TRIAL),
                "progress_acts": int(self.trial_progress_acts),
                "worsening_acts": int(self.trial_worsening_acts),
            }
            trials = self.results_log.get("trials")
            if isinstance(trials, list):
                trials.append(trial_row)
                self.results_log["completed_trials"] = int(len(trials))
            self._write_results()

    def print_recovery_summary(self):
        attempts = int(self.recovery_stats.get("attempts", 0))
        recovered = int(self.recovery_stats.get("recovered", 0))
        reverse_attempts = int(self.recovery_stats.get("reverse_attempts", 0))
        reverse_recovered = int(self.recovery_stats.get("reverse_recovered", 0))
        scan_attempts = int(self.recovery_stats.get("scan_attempts", 0))
        scan_recovered = int(self.recovery_stats.get("scan_recovered", 0))

        overall_rate = (float(recovered) / float(attempts)) if attempts > 0 else 0.0
        reverse_rate = (float(reverse_recovered) / float(reverse_attempts)) if reverse_attempts > 0 else 0.0
        scan_rate = (float(scan_recovered) / float(scan_attempts)) if scan_attempts > 0 else 0.0

        print(
            "[ALIGN] Recovery summary: "
            f"overall {recovered}/{attempts} ({overall_rate * 100.0:.1f}%), "
            f"reverse {reverse_recovered}/{reverse_attempts} ({reverse_rate * 100.0:.1f}%), "
            f"scan {scan_recovered}/{scan_attempts} ({scan_rate * 100.0:.1f}%)."
        )
        summary = {
            "attempts": int(attempts),
            "recovered": int(recovered),
            "overall_rate": float(overall_rate),
            "reverse_attempts": int(reverse_attempts),
            "reverse_recovered": int(reverse_recovered),
            "reverse_rate": float(reverse_rate),
            "scan_attempts": int(scan_attempts),
            "scan_recovered": int(scan_recovered),
            "scan_rate": float(scan_rate),
        }
        self._append_event(
            {
                "type": "recovery_summary",
                "run_id": self.run_id,
                "timestamp": time.time(),
                **summary,
            }
        )
        return summary

    def update_learning_speed(self, total_time, avg_recent=None):
        delta = LEARN_SPEED_STEP
        if avg_recent is not None and total_time < avg_recent:
            delta += LEARN_SPEED_BONUS
        self.learned_speed_score = min(COARSE_SCORE, self.learned_speed_score + delta)
        print(f"[LEARN] Next trial speed score: {self.learned_speed_score}%")

    def _success_seconds_by_trial(self):
        by_trial = {}
        for row in self.data_log:
            if not isinstance(row, dict) or row.get("type") != "trial":
                continue
            trial = row.get("trial")
            seconds = row.get("seconds_to_success")
            try:
                trial_idx = int(trial)
                sec_val = float(seconds)
            except (TypeError, ValueError):
                continue
            by_trial[trial_idx] = float(sec_val)
        return by_trial

    def _efficiency_rows(self):
        if not isinstance(self.results_log, dict):
            return []
        trials = self.results_log.get("trials")
        if not isinstance(trials, list):
            return []
        success_seconds = self._success_seconds_by_trial()
        rows = []
        for row in trials:
            if not isinstance(row, dict):
                continue
            trial = row.get("trial")
            try:
                trial_idx = int(trial)
            except (TypeError, ValueError):
                continue
            success = bool(row.get("success"))
            secs = success_seconds.get(trial_idx)
            if success and secs is not None and secs >= 0.0:
                efficiency = max(0.0, min(100.0, 100.0 / (1.0 + float(secs))))
            elif success:
                efficiency = 25.0
            else:
                efficiency = 0.0
            rows.append(
                {
                    "trial": int(trial_idx),
                    "success": bool(success),
                    "reason": str(row.get("reason") or ""),
                    "seconds_to_success": (None if secs is None else float(secs)),
                    "adjust_count": row.get("adjust_count"),
                    "adjust_limit": row.get("adjust_limit"),
                    "efficiency": float(efficiency),
                }
            )
        return rows

    def _act_progress_rows(self):
        """Per successful trial, capture ALIGN acts up to victory."""
        success_end_ts_by_trial = {}
        for row in self.data_log:
            if not isinstance(row, dict) or row.get("type") != "trial_summary":
                continue
            if not bool(row.get("success")):
                continue
            try:
                trial_idx = int(row.get("trial"))
                ts = float(row.get("timestamp"))
            except (TypeError, ValueError):
                continue
            prev = success_end_ts_by_trial.get(trial_idx)
            if prev is None or ts < float(prev):
                success_end_ts_by_trial[trial_idx] = float(ts)

        if not success_end_ts_by_trial:
            return []

        by_trial = {}
        for row in self.data_log:
            if not isinstance(row, dict) or row.get("type") != "act":
                continue
            if str(row.get("phase") or "").lower() != "align":
                continue
            try:
                trial_idx = int(row.get("trial"))
                ts = float(row.get("timestamp"))
            except (TypeError, ValueError):
                continue
            victory_ts = success_end_ts_by_trial.get(trial_idx)
            if victory_ts is None or ts > float(victory_ts):
                continue
            try:
                abs_x_err_mm = abs(float(row.get("x_err_mm")))
            except (TypeError, ValueError):
                continue
            try:
                x_tol_active_mm = float(row.get("x_tol_active_mm"))
            except (TypeError, ValueError):
                x_tol_active_mm = None
            by_trial.setdefault(trial_idx, []).append(
                {
                    "timestamp": float(ts),
                    "abs_x_err_mm": float(abs_x_err_mm),
                    "x_tol_active_mm": x_tol_active_mm,
                }
            )

        out = []
        for trial_idx in sorted(by_trial):
            points_raw = sorted(by_trial.get(trial_idx) or [], key=lambda item: float(item.get("timestamp", 0.0)))
            if not points_raw:
                continue
            points = []
            prev_err = None
            for act_in_trial, point in enumerate(points_raw, start=1):
                err = float(point["abs_x_err_mm"])
                if prev_err is None:
                    trend = "start"
                elif err < (prev_err - 0.02):
                    trend = "better"
                elif err > (prev_err + 0.02):
                    trend = "worse"
                else:
                    trend = "flat"
                points.append(
                    {
                        "act_in_trial": int(act_in_trial),
                        "abs_x_err_mm": float(err),
                        "x_tol_active_mm": point.get("x_tol_active_mm"),
                        "trend": trend,
                    }
                )
                prev_err = float(err)
            final_tol = points[-1].get("x_tol_active_mm")
            try:
                final_tol = float(final_tol) if final_tol is not None else None
            except (TypeError, ValueError):
                final_tol = None
            out.append(
                {
                    "trial": int(trial_idx),
                    "victory_ts": float(success_end_ts_by_trial.get(trial_idx)),
                    "victory_tol_mm": final_tol,
                    "points": points,
                }
            )
        return out

    def write_act_progress_chart(self):
        rows = self._act_progress_rows()
        if not rows:
            return {
                "ok": False,
                "path": str(self.act_progress_chart_file_path),
                "reason": "no_success_trial_act_rows",
            }

        width = 1400
        height = 900
        margin_left = 95
        margin_right = 40
        margin_top = 90
        margin_bottom = 110
        x0 = margin_left
        y0 = margin_top
        x1 = width - margin_right
        y1 = height - margin_bottom
        plot_w = x1 - x0
        plot_h = y1 - y0

        img = np.full((height, width, 3), 247, dtype=np.uint8)
        cv2.rectangle(img, (x0, y0), (x1, y1), (210, 210, 210), 1)

        max_acts = max(len(row.get("points") or []) for row in rows)
        max_acts = max(1, int(max_acts))

        all_errs = []
        for row in rows:
            for point in row.get("points") or []:
                try:
                    all_errs.append(float(point.get("abs_x_err_mm")))
                except (TypeError, ValueError):
                    continue
        max_err = max(all_errs) if all_errs else float(X_TOL_MM)
        y_max = max(float(max_err) * 1.10, float(X_TOL_MM) * 2.0, 1.0)

        # Y grid in mm.
        y_ticks = 6
        for i in range(y_ticks + 1):
            val = float(y_max) * float(i) / float(y_ticks)
            y = int(round(y1 - (val / float(y_max)) * plot_h))
            cv2.line(img, (x0, y), (x1, y), (225, 225, 225), 1)
            cv2.putText(
                img,
                f"{val:.1f}mm",
                (18, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (95, 95, 95),
                1,
                cv2.LINE_AA,
            )

        # X ticks by act index.
        x_tick_step = max(1, int(round(float(max_acts) / 10.0)))
        for act in range(1, max_acts + 1, x_tick_step):
            if max_acts <= 1:
                x = x0
            else:
                x = int(round(x0 + (float(act - 1) / float(max_acts - 1)) * plot_w))
            cv2.line(img, (x, y0), (x, y1), (236, 236, 236), 1)
            cv2.putText(
                img,
                str(int(act)),
                (x - 8, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )

        # Reference line: base normalized tolerance.
        base_tol = float(X_TOL_MM) * float(X_TOL_RELAX_MULTIPLIER)
        base_tol_clamped = min(float(y_max), max(0.0, float(base_tol)))
        y_tol = int(round(y1 - (base_tol_clamped / float(y_max)) * plot_h))
        cv2.line(img, (x0, y_tol), (x1, y_tol), (0, 140, 255), 1)
        cv2.putText(
            img,
            f"base normalized tol {base_tol:.2f}mm",
            (x1 - 260, max(y0 + 16, y_tol - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 120, 220),
            1,
            cv2.LINE_AA,
        )

        palette = [
            (60, 110, 220),
            (210, 130, 30),
            (120, 170, 70),
            (170, 90, 180),
            (40, 150, 170),
            (200, 80, 120),
        ]

        legend_x = x0 + 8
        legend_y = y0 + 18
        legend_step = 23

        for idx, row in enumerate(rows):
            trial = int(row.get("trial"))
            points = row.get("points") or []
            if not points:
                continue
            color = palette[idx % len(palette)]
            poly_pts = []
            for point in points:
                act_idx = int(point.get("act_in_trial") or 1)
                err = float(point.get("abs_x_err_mm") or 0.0)
                if max_acts <= 1:
                    x = x0
                else:
                    x = int(round(x0 + (float(act_idx - 1) / float(max_acts - 1)) * plot_w))
                y = int(round(y1 - (min(float(y_max), max(0.0, float(err))) / float(y_max)) * plot_h))
                poly_pts.append((x, y))

            if len(poly_pts) >= 2:
                cv2.polylines(img, [np.array(poly_pts, dtype=np.int32)], False, color, 2)

            # Per-act trend markers.
            for p_idx, point in enumerate(points):
                x, y = poly_pts[p_idx]
                trend = str(point.get("trend") or "")
                if trend == "better":
                    dot = (60, 170, 70)
                elif trend == "worse":
                    dot = (60, 60, 210)
                elif trend == "flat":
                    dot = (120, 120, 120)
                else:
                    dot = color
                cv2.circle(img, (x, y), 4, dot, -1)

            # Victory marker.
            vx, vy = poly_pts[-1]
            cv2.circle(img, (vx, vy), 8, (0, 215, 255), 2)
            cv2.circle(img, (vx, vy), 2, (0, 215, 255), -1)

            final_err = float(points[-1].get("abs_x_err_mm") or 0.0)
            final_tol = row.get("victory_tol_mm")
            tol_text = ""
            try:
                if final_tol is not None:
                    tol_text = f", tol {float(final_tol):.2f}mm"
            except (TypeError, ValueError):
                tol_text = ""
            legend_text = f"T{trial}: {len(points)} acts, final {final_err:.2f}mm{tol_text}"
            ly = int(legend_y + (idx * legend_step))
            cv2.line(img, (legend_x, ly - 4), (legend_x + 20, ly - 4), color, 2)
            cv2.putText(
                img,
                legend_text,
                (legend_x + 28, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (55, 55, 55),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            "ALIGN Per-Act Progress To Victory",
            (x0, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "|x_err| in mm per act (lower is better). Dot colors: green=better, red=worse, gray=flat.",
            (x0, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "x-axis: act index within trial (align phase only, up to success)",
            (x0 + 360, y1 + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

        ok = bool(cv2.imwrite(str(self.act_progress_chart_file_path), img))
        return {
            "ok": bool(ok),
            "path": str(self.act_progress_chart_file_path),
            "success_trial_count": int(len(rows)),
            "max_acts_in_success_trial": int(max_acts),
            "max_abs_x_err_mm": float(max_err),
        }

    def write_efficiency_chart(self):
        rows = self._efficiency_rows()
        if not rows:
            return {"ok": False, "path": str(self.chart_file_path), "reason": "no_trial_rows"}

        n = len(rows)
        width = 1280
        height = 720
        margin_left = 90
        margin_right = 30
        margin_top = 80
        margin_bottom = 90
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        x0 = margin_left
        y0 = margin_top
        x1 = margin_left + plot_w
        y1 = margin_top + plot_h

        img = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.rectangle(img, (x0, y0), (x1, y1), (210, 210, 210), 1)

        # Horizontal gridlines + y labels (0..100%).
        for pct in range(0, 101, 20):
            y = int(round(y1 - (pct / 100.0) * plot_h))
            color = (200, 200, 200) if pct > 0 else (150, 150, 150)
            cv2.line(img, (x0, y), (x1, y), color, 1)
            cv2.putText(
                img,
                f"{pct}%",
                (18, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )

        step = float(plot_w) / float(max(1, n))
        bar_w = max(10, int(step * 0.6))
        efficiencies = [float(row.get("efficiency", 0.0) or 0.0) for row in rows]
        avg_eff = float(sum(efficiencies) / float(max(1, len(efficiencies))))

        line_pts = []
        for idx, row in enumerate(rows):
            x_center = int(round(x0 + (idx + 0.5) * step))
            eff = float(row.get("efficiency", 0.0) or 0.0)
            y_top = int(round(y1 - (eff / 100.0) * plot_h))
            color = (60, 170, 70) if bool(row.get("success")) else (60, 60, 200)
            cv2.rectangle(
                img,
                (x_center - bar_w // 2, y_top),
                (x_center + bar_w // 2, y1 - 1),
                color,
                -1,
            )
            cv2.putText(
                img,
                str(int(row.get("trial", idx + 1))),
                (x_center - 12, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            line_pts.append((x_center, y_top))

        if len(line_pts) >= 2:
            cv2.polylines(img, [np.array(line_pts, dtype=np.int32)], False, (190, 120, 0), 2)

        y_avg = int(round(y1 - (avg_eff / 100.0) * plot_h))
        cv2.line(img, (x0, y_avg), (x1, y_avg), (0, 140, 255), 1)
        cv2.putText(
            img,
            f"avg {avg_eff:.1f}%",
            (x1 - 135, max(y0 + 16, y_avg - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 120, 220),
            1,
            cv2.LINE_AA,
        )

        success_count = sum(1 for row in rows if bool(row.get("success")))
        cv2.putText(
            img,
            "ALIGN Efficiency Across Trials",
            (x0, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            (
                f"success={success_count}/{n}  "
                f"metric=100/(1+seconds_to_success), failed trials=0"
            ),
            (x0, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "green=success  red=fail  orange=line trend",
            (x0 + 380, y1 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

        ok = bool(cv2.imwrite(str(self.chart_file_path), img))
        return {
            "ok": bool(ok),
            "path": str(self.chart_file_path),
            "trial_count": int(n),
            "success_count": int(success_count),
            "avg_efficiency_pct": float(avg_eff),
            "max_efficiency_pct": float(max(efficiencies) if efficiencies else 0.0),
        }

    def run_experiment(self):
        # WIPE DATA FILE ON START
        if Path(DATA_FILE).exists():
            print(f"[ALIGN] Wiping {DATA_FILE} for fresh learning...")
            Path(DATA_FILE).unlink()
        if self.results_file_path.exists():
            print(f"[ALIGN] Wiping {self.results_file_path} for fresh results...")
            self.results_file_path.unlink()
        if self.chart_file_path.exists():
            print(f"[ALIGN] Wiping {self.chart_file_path} for fresh chart...")
            self.chart_file_path.unlink()
        if self.act_progress_chart_file_path.exists():
            print(f"[ALIGN] Wiping {self.act_progress_chart_file_path} for fresh chart...")
            self.act_progress_chart_file_path.unlink()
        self.data_log = []
        self._unsaved_events = 0
        self._init_results_log()
        self._write_results()
        self._append_event({
            "type": "meta",
            "run_id": self.run_id,
            "timestamp": time.time(),
            "num_trials": NUM_TRIALS,
            "x_gate_metric": X_GATE_METRIC,
            "x_gate_target_mm": X_GATE_TARGET_MM,
            "x_gate_tol_mm": X_GATE_TOL_MM,
            "dist_gate_metric": DIST_GATE_METRIC,
            "dist_gate_target_mm": DIST_GATE_TARGET_MM,
            "dist_gate_tol_mm": DIST_GATE_TOL_MM,
            "x_axis_sign": X_AXIS_SIGN,
            "x_target_mm": X_TARGET_MM,
            "x_tol_mm": X_TOL_MM,
            "x_tol_relax_multiplier": X_TOL_RELAX_MULTIPLIER,
            "x_tol_growth_per_mm_back": X_TOL_GROWTH_PER_MM_BACK,
            "x_tol_max_mm": X_TOL_MAX_MM,
            "precision_score_base": PRECISION_SCORE_BASE,
            "precision_bump_pct": PRECISION_BUMP_PCT,
            "coarse_score_cap": COARSE_SCORE,
            "vision_lost_consec_frames": int(VISION_LOST_CONSEC_FRAMES),
            "vision_lost_max_s": float(VISION_LOST_MAX_S),
            "align_observe_timeout_s": float(ALIGN_OBSERVE_TIMEOUT_S),
            "post_act_observe_ratio": float(POST_ACT_OBSERVE_RATIO),
            "post_act_observe_min_s": float(POST_ACT_OBSERVE_MIN_S),
            "post_act_observe_max_s": float(POST_ACT_OBSERVE_MAX_S),
            "inter_trial_settle_s": float(INTER_TRIAL_SETTLE_S),
            "recovery_reverse_max_acts": int(RECOVERY_REVERSE_MAX_ACTS),
            "recovery_reverse_score": int(RECOVERY_REVERSE_SCORE),
            "recovery_reverse_allowed_phases": sorted(RECOVERY_REVERSE_ALLOWED_PHASES),
            "recovery_reverse_skip_reason_prefixes": list(RECOVERY_REVERSE_SKIP_REASON_PREFIXES),
            "recovery_reverse_skip_reasons": sorted(RECOVERY_REVERSE_SKIP_REASONS),
            "recovery_scan_max_attempts": int(RECOVERY_SCAN_MAX_ATTEMPTS),
            "recovery_scan_score": int(RECOVERY_SCAN_SCORE),
            "recovery_scan_burst_acts": int(RECOVERY_SCAN_BURST_ACTS),
            "recovery_scan_observe_s": float(RECOVERY_SCAN_OBSERVE_S),
            "max_consecutive_unrecovered_vision_losses": int(MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES),
            "max_adjusts_per_trial": int(MAX_ADJUSTS_PER_TRIAL),
        })

        print(f"\n--- ROGUE ALIGNMENT EXPERIMENT: CONSTANT MOTION ---")
        print(f"X Gate ({X_GATE_METRIC}): {X_GATE_TARGET_MM:+.2f}mm (tol {X_GATE_TOL_MM:.2f}mm)")
        print(f"Using X target: {X_TARGET_MM:+.2f}mm (tol {X_TOL_MM:.2f}mm)")
        print("[ALIGN] Act logging: colorful + verbose output for every independent act.")
        print(f"[ALIGN] Results summary file: {self.results_file_path}")
        print(
            "[ALIGN] Early stop guard: "
            f"{int(MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES)} consecutive unrecovered vision losses."
        )
        print(f"[ALIGN] Trial guard: give up after {int(MAX_ADJUSTS_PER_TRIAL)} adjust acts.")
        print(
            "[ALIGN] Fresh-observation guard: post-act samples wait "
            f"{POST_ACT_OBSERVE_RATIO:.2f}x duration (min {POST_ACT_OBSERVE_MIN_S:.2f}s, "
            f"max {POST_ACT_OBSERVE_MAX_S:.2f}s)."
        )
        
        for i in range(1, NUM_TRIALS + 1):
            if not self.running:
                break
            if self.abort_run_early:
                break
            self.current_trial = i
            self.success_frames = 0
            self.offset_history.clear()
            self.recent_acts.clear()
            self.precision_bump_pct = PRECISION_BUMP_PCT
            self.precision_stall_acts = 0
            self.trial_vision_lost_events = 0
            self.trial_recovery_attempts = 0
            self.trial_recovery_successes = 0
            self.trial_adjust_count = 0
            self.trial_progress_acts = 0
            self.trial_worsening_acts = 0
            self._last_logged_abs_x_err = None
            print(f"\n\n\nTRIAL {i}/{NUM_TRIALS}")
            print(
                "[TRIAL] success gate: "
                "visible=true and |x_err| <= x_tol_active for 3 consecutive frames."
            )
            print(
                "[TRIAL] x_tol_active(dist) = "
                f"clamp(({X_TOL_MM:.2f}*{X_TOL_RELAX_MULTIPLIER:.2f}) + "
                f"max(0, dist - ({DIST_GATE_TARGET_MM:.2f}+{DIST_GATE_TOL_MM:.2f})) * "
                f"{X_TOL_GROWTH_PER_MM_BACK:.3f}, max {X_TOL_MAX_MM:.2f})"
            )
            trial_success = False
            trial_reason = "unknown"
            
            # 1. SETUP: Deliberate vision-driven reset away from the X gate target
            self.current_phase = "reset"
            pose_reset = self.run_reset_phase()
            if not pose_reset:
                print("  [ERROR] Cannot find brick for reset setup.")
                trial_reason = "reset_failed_vision"
                self.print_trial_recovery_summary(i, success=False, reason=trial_reason)
                if self.abort_run_early:
                    break
                continue

            # Trial-start snapshot: show the active (distance-normalized) X tolerance.
            try:
                start_offset = float(pose_reset.get("offset_x"))
            except (TypeError, ValueError):
                start_offset = None
            try:
                start_dist = float(pose_reset.get("dist"))
            except (TypeError, ValueError):
                start_dist = None
            start_active_tol = self._active_x_tol_mm(start_dist)
            start_x_err = self._x_err_mm(start_offset) if start_offset is not None else None
            dist_text = f"{start_dist:.1f}mm" if start_dist is not None else "--"
            err_text = f"{start_x_err:+.2f}mm" if start_x_err is not None else "--"
            print(
                "[TRIAL] target snapshot: "
                f"x_target={X_TARGET_MM:+.2f}mm | "
                f"x_tol_base={X_TOL_MM:.2f}mm x{X_TOL_RELAX_MULTIPLIER:.2f} -> x_tol_active={start_active_tol:.2f}mm | "
                f"dist={dist_text} (dist_target={DIST_GATE_TARGET_MM:.2f}±{DIST_GATE_TOL_MM:.2f}mm) | "
                f"start_x_err={err_text}"
            )
            self._append_event(
                {
                    "type": "trial_target_snapshot",
                    "run_id": self.run_id,
                    "trial": int(i),
                    "timestamp": time.time(),
                    "x_target_mm": float(X_TARGET_MM),
                    "x_tol_base_mm": float(X_TOL_MM),
                    "x_tol_relax_multiplier": float(X_TOL_RELAX_MULTIPLIER),
                    "x_tol_active_mm": float(start_active_tol),
                    "dist_mm": (None if start_dist is None else float(start_dist)),
                    "dist_gate_target_mm": float(DIST_GATE_TARGET_MM),
                    "dist_gate_tol_mm": float(DIST_GATE_TOL_MM),
                    "start_x_err_mm": (None if start_x_err is None else float(start_x_err)),
                }
            )

            # No robot.stop() here - we want to transition immediately into ALIGN
            
            # 2. CONTINUOUS ALIGNMENT
            self.current_phase = "align"
            trial_start_t = time.time()
            trial_start_offset = None
            prev_align_abs_err = None
            
            while self.running:
                # Continuous observation (1 sample for speed, short timeout).
                pose = self.get_stable_pose(
                    samples=1,
                    timeout_s=float(ALIGN_OBSERVE_TIMEOUT_S),
                    min_sample_time=self._observe_not_before_ts(),
                )
                if not pose:
                    print(f"  [LOST] Vision lost, attempting recovery...")
                    self.current_phase = "recovery"
                    if not self.recover_visibility(source="align_observe"):
                        trial_reason = "lost_vision"
                        break
                    self.current_phase = "align"
                    pose = self.get_stable_pose(
                        samples=1,
                        timeout_s=float(ALIGN_OBSERVE_TIMEOUT_S),
                        min_sample_time=self._observe_not_before_ts(),
                    )
                    if not pose:
                        trial_reason = "lost_vision"
                        break
                    prev_align_abs_err = None
                    trial_start_t = time.time() # Reset timer if we lost track
                
                off_now = pose["offset_x"]
                dist_now = pose.get("dist")
                self.offset_history.append(off_now)
                if trial_start_offset is None: trial_start_offset = off_now
                
                x_err = self._x_err_mm(off_now)
                abs_error_from_target = abs(x_err)
                active_x_tol = self._active_x_tol_mm(dist_now)
                if prev_align_abs_err is not None:
                    delta_abs_err = float(prev_align_abs_err) - float(abs_error_from_target)
                    if delta_abs_err > 0.02:
                        self.trial_progress_acts = int(self.trial_progress_acts + 1)
                    elif delta_abs_err < -0.02:
                        self.trial_worsening_acts = int(self.trial_worsening_acts + 1)
                prev_align_abs_err = float(abs_error_from_target)
                
                # Check Gate Success: Requirement = 3 consecutive frames
                # We use the official gate_satisfied check for parity with the overlay
                official_satisfied = gate_satisfied(self.world.brick, ALIGN_GATES) if self.world.brick else False
                exp_satisfied = abs_error_from_target <= float(active_x_tol)
                
                if exp_satisfied:
                    self.success_frames += 1
                    if self.success_frames >= self.REQUIRED_SUCCESS_FRAMES:
                        total_time = time.time() - trial_start_t
                        avg_recent = None
                        if self.error_history:
                            recent = list(self.error_history)[-3:]
                            avg_recent = statistics.mean(recent)
                        self.print_trial_summary(i, off_now, total_time)
                        self.log_trial(i, trial_start_offset, off_now, total_time)
                        self.update_learning_speed(total_time, avg_recent=avg_recent)
                        print(
                            "[VICTORY] "
                            f"after {int(self.trial_adjust_count)} acts "
                            f"({int(self.trial_progress_acts)} progress acts + "
                            f"{int(self.trial_worsening_acts)} worsening acts)"
                        )
                        # Stop between trials; next motion should come from an explicit reset/align decision.
                        self.robot.stop()
                        trial_success = True
                        trial_reason = "success"
                        break
                else:
                    self.success_frames = 0
                
                # PREDICT DIRECTION & SPEED
                # Very simple X-axis correction: brick right => turn right.
                align_dir = 'r' if x_err > 0 else 'l'
                self.keepalive_dir = align_dir
                
                # Two-Stage Speed: learned coarse score if far, precision bump if near.
                near_threshold = float(active_x_tol) + 4.0
                # If we aren't making progress after ~2 acts, nudge speed up a bit.
                stall = False
                if len(self.offset_history) == STALL_WINDOW:
                    span = max(self.offset_history) - min(self.offset_history)
                    if span < NOISE_FLOOR_MM:
                        stall = True

                # Precision-mode stall escalation (only when near target).
                if abs_error_from_target <= near_threshold:
                    if stall:
                        self.precision_stall_acts += 1
                        if self.precision_stall_acts >= PRECISION_ESCALATE_AFTER:
                            self.precision_bump_pct = min(
                                self.precision_bump_pct + PRECISION_ESCALATE_STEP,
                                PRECISION_BUMP_MAX_PCT,
                            )
                            self.precision_stall_acts = 0
                            print(f"[STALL] No progress -> bump +{self.precision_bump_pct:.0f}%")
                    else:
                        if self.precision_stall_acts or self.precision_bump_pct != PRECISION_BUMP_PCT:
                            self.precision_stall_acts = 0
                            self.precision_bump_pct = PRECISION_BUMP_PCT

                # HEARTBEAT: Always send command to feed the 100ms watchdog.
                if int(self.trial_adjust_count) >= int(MAX_ADJUSTS_PER_TRIAL):
                    trial_reason = "max_adjusts"
                    print(
                        f"  [STOP] Reached max adjusts for trial: "
                        f"{int(self.trial_adjust_count)}/{int(MAX_ADJUSTS_PER_TRIAL)}."
                    )
                    break

                if abs_error_from_target > near_threshold:
                    score = min(self.learned_speed_score, COARSE_SCORE)
                    if stall:
                        score = min(score + STALL_BUMP_SCORE, COARSE_SCORE)
                    event = self.send_coarse_command(
                        align_dir,
                        score=score,
                        reason="x_axis_far",
                        extra={"stall": stall, "near_threshold_mm": near_threshold, "official_gate": official_satisfied, "exp_gate": exp_satisfied},
                    )
                    if isinstance(event, dict):
                        self.trial_adjust_count = int(self.trial_adjust_count + 1)
                else:
                    bump_pct = self.precision_bump_pct + (STALL_BUMP_PCT if stall else 0.0)
                    event = self.send_precision_command(
                        align_dir,
                        bump_pct=bump_pct,
                        reason="x_axis_near",
                        extra={"stall": stall, "near_threshold_mm": near_threshold, "official_gate": official_satisfied, "exp_gate": exp_satisfied},
                    )
                    if isinstance(event, dict):
                        self.trial_adjust_count = int(self.trial_adjust_count + 1)
                
                # Short sleep to not hammer the CPU/Serial
                time.sleep(0.04)

            if not trial_success and trial_reason == "unknown":
                if not self.running:
                    trial_reason = "stopped"
                elif self.abort_run_early:
                    trial_reason = "early_stop_unrecovered_vision"
                else:
                    trial_reason = "loop_exit"
            self.print_trial_recovery_summary(i, success=trial_success, reason=trial_reason)
            if self.abort_run_early:
                break
            if self.running and i < int(NUM_TRIALS):
                self.robot.stop()
                self._last_act_sent_at = None
                self._last_act_duration_s = 0.0
                if float(INTER_TRIAL_SETTLE_S) > 0.0:
                    time.sleep(float(INTER_TRIAL_SETTLE_S))

        recovery_summary = self.print_recovery_summary()
        chart_result = self.write_efficiency_chart()
        act_chart_result = self.write_act_progress_chart()
        if bool(chart_result.get("ok")):
            print(f"[ALIGN] Efficiency chart: {chart_result.get('path')}")
        else:
            print(f"[ALIGN] Efficiency chart skipped: {chart_result.get('reason')}")
        if bool(act_chart_result.get("ok")):
            print(f"[ALIGN] Per-act progress chart: {act_chart_result.get('path')}")
        else:
            print(f"[ALIGN] Per-act progress chart skipped: {act_chart_result.get('reason')}")
        if isinstance(self.results_log, dict):
            self.results_log["recovery_summary"] = dict(recovery_summary) if isinstance(recovery_summary, dict) else None
            self.results_log["efficiency_chart"] = dict(chart_result) if isinstance(chart_result, dict) else None
            self.results_log["act_progress_chart"] = (
                dict(act_chart_result) if isinstance(act_chart_result, dict) else None
            )
            self.results_log["early_stop"] = bool(self.abort_run_early)
            self.results_log["early_stop_reason"] = (
                "consecutive_unrecovered_vision_losses" if self.abort_run_early else None
            )
            self.results_log["status"] = "finished"
            self.results_log["finished_at"] = float(time.time())
            self._write_results()
        print(f"\n--- Complete. Final K: L={self.K_L:.1f}, R={self.K_R:.1f} ---")

    def print_trial_summary(self, trial_idx, error, total_time):
        avg_3 = None
        comparison = ""
        if len(self.error_history) >= 3:
            last_3 = list(self.error_history)[-3:]
            avg_3 = statistics.mean(last_3)
            diff_pct = ((total_time - avg_3) / avg_3) * 100
            
            if diff_pct < 0:
                comparison = f" ( {abs(diff_pct):.1f}% better than avg3 🟢 )"
            else:
                comparison = f" ( {diff_pct:.1f}% worse than avg3 🔴 )"
        
        success_emo = "🤩" if total_time < 2.0 else "✅"
        print(f"[RESULT] Trial {trial_idx} Success in {total_time:.2f}s (Final Off: {error:.2f}mm) {success_emo}{comparison}")
        self.error_history.append(total_time)

    def log_trial(self, i, initial, final, total_time):
        self._append_event({
            "type": "trial",
            "run_id": self.run_id,
            "trial": i, "timestamp": time.time(), "initial_offset": initial,
            "final_offset": final, "seconds_to_success": total_time,
            "K_L": self.K_L, "K_R": self.K_R
        })
        self._flush_logs(force=True)

    def close(self):
        self.running = False
        self._flush_logs(force=True)
        if isinstance(self.results_log, dict) and self.results_log:
            if self.results_log.get("status") == "running":
                self.results_log["status"] = "closed_early"
                self.results_log["finished_at"] = float(time.time())
            chart_result = self.write_efficiency_chart()
            act_chart_result = self.write_act_progress_chart()
            if isinstance(chart_result, dict):
                self.results_log["efficiency_chart"] = dict(chart_result)
            if isinstance(act_chart_result, dict):
                self.results_log["act_progress_chart"] = dict(act_chart_result)
            self._write_results()
        self.robot.close()
        self.vision.close()
        self.server.stop()

if __name__ == "__main__":
    calibrator = AlignCalibrator()
    try: calibrator.run_experiment()
    except KeyboardInterrupt: print("\nInterrupted.")
    finally: calibrator.close()
