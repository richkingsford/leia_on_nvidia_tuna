#!/usr/bin/env python3
import json
import math
import random
import time
import statistics
import subprocess
import shutil
import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from collections import deque
from functools import partial
import tempfile

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    _MATPLOTLIB_AVAILABLE = False

from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
from helper_next import (
    compute_alignment_decision,
    align_local_gate_status,
    align_brick_dist_error_speed_score,
    align_brick_x_axis_one_shot_score,
    format_align_pre_observation_text,
    build_align_result_delta_obj,
)
import telemetry_robot as telemetry_robot_module
from telemetry_robot import WorldModel, StepState
from telemetry_process import (
    COLOR_CYAN,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_ORANGE_BRIGHT,
    COLOR_RED,
    COLOR_RESET,
    COLOR_WHITE,
    COLOR_YELLOW,
    format_control_action_line,
    format_shorthand_calibration_line,
    send_robot_command,
    send_robot_command_pwm,
)
from helper_gate_utils import load_process_steps, gate_satisfied, SuccessGateTracker, update_gatecheck
from telemetry_process import evaluate_gate_status
from helper_align_profile import build_align_profile_from_results_payload, promote_align_profile_if_better

# Experiment Config
# Goal reminder: minimize total acts and avoid overshoots/worse acts while staying robust to distance.
NUM_TRIALS = 20
EXPERIMENT_TRIALS = 15  # Only first N trials should explore/adapt learning knobs
DATA_FILE = "world_model_align.json"
RESULTS_FILE = str(
    Path(DATA_FILE).with_name(f"{Path(DATA_FILE).stem}_results{Path(DATA_FILE).suffix}")
)
ALPHA = 0.25         # Learning rate
SPEED_CURVE_PNG = "world_model_align_speed_curve.png"
SPEED_CURVE_X_BINS_MM = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 22.0]

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
DIST_TARGET_MM = float(DIST_GATE_TARGET_MM)
DIST_TOL_MM = float(DIST_GATE_TOL_MM)
LOG_FLUSH_EVERY_EVENTS = 50

# *** PRODUCTION LOGIC: Use fixed gates, not normalized ones ***
# Remove distance normalization hack - gates are absolute
# Alignment must satisfy BOTH x_err and dist gates simultaneously
# Legacy constants (no longer used, kept for backward compatibility in JSON logs):
X_TOL_RELAX_MULTIPLIER = 1.0  # Deprecated (was 1.4)
X_TOL_GROWTH_PER_MM_BACK = 0.0  # Deprecated (was 0.05)
X_TOL_MAX_MM = 4.0  # Deprecated

# Distance-aware act scaling (affects decisions, not just success gate).
DIST_ACT_FAR_DELTA_MM = 22.0
DIST_ACT_NEAR_DELTA_MM = 14.0
DIST_ACT_FAR_SCORE_SCALE = 1.15
DIST_ACT_NEAR_SCORE_SCALE = 0.85
DIST_ACT_FAR_BUMP_SCALE = 1.10
DIST_ACT_NEAR_BUMP_SCALE = 0.85
DIST_ACT_FAR_THRESHOLD_SCALE = 1.20
DIST_ACT_NEAR_THRESHOLD_SCALE = 0.85
DIST_ACT_SCORE_MIN = 1
OVERSHOOT_SIGN_FLIP_DEADZONE_MM = 0.30
MAX_OVERSHOOTS_PER_TRIAL = 8
OVERSHOOT_DAMPENING_FACTOR = 0.65  # Multiply speed by this after each overshoot (exponential decay)
DIRECTION_CHANGE_DAMPENING = 0.50  # Reduce speed by 50% when reversing direction (more aggressive)
OVERSHOOT_FIRST_PENALTY_PCT = 30.0  # One-strike rule: cut learned_speed_score by 30% after first overshoot
SETTLING_ZONE_MULTIPLIER = 2.0     # Enter ultra-precision when within tolerance * this
PROPORTIONAL_CONTROL_ENABLED = True  # Scale precision speed based on error magnitude

# Advanced Control Tunings
BACKLASH_S = 0.08         
COARSE_SCORE = 50         # Max coarse speed score (cap)
RESET_MIN_AWAY_MM = 2.0   # Min distance beyond target; was 10.0 (unnecessary extra backing)
RESET_MAX_AWAY_MM = 70.0
RESET_X_MIN_MM = -40.0  # Min x-axis offset during reset (doubled again)
RESET_X_MAX_MM = 40.0   # Max x-axis offset during reset (doubled again)
RESET_SCORE = 20
RESET_MAX_ACTS = 80
RESET_MAX_DISTANCE_MM = 170.0  # Safety limit: never exceed this distance during reset
RESET_OBSERVE_TIMEOUT_S = 1.5
PRE_RESET_VISION_ATTEMPTS = 5
PRE_RESET_OBSERVE_TIMEOUT_S = 2.5
PRE_RESET_PAUSE_S = 0.3
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
VISION_LOST_CONSEC_FRAMES = 6     # Trigger loss quickly when no brick frames arrive.
VISION_LOST_MAX_S = 0.7           # Hard cap on blind sampling window.
ALIGN_OBSERVE_TIMEOUT_S = 0.6      # Faster loop cadence for alignment control.
RECOVERY_REVERSE_MAX_ACTS = 6      # How many recent acts to invert for recovery.
RECOVERY_REVERSE_OBSERVE_S = 0.8
RECOVERY_SCAN_MAX_ATTEMPTS = 8
RECOVERY_REVERSE_ALLOWED_PHASES = {"align", "reset"}
RECOVERY_REVERSE_SKIP_REASON_PREFIXES = ("recovery_",)
RECOVERY_REVERSE_SKIP_REASONS = {"between_trials"}
MAX_CONSECUTIVE_UNRECOVERED_VISION_LOSSES = 3
MAX_ADJUSTS_PER_TRIAL = 100  # Raised to allow longer convergence attempts per trial
MAX_GATE_RECHECK_ATTEMPTS = 3  # Immediate mid-gate-check retry attempts
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
        # Load production gate rules for gate checking
        self.world.process_rules = STEPS
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
        pass  # Charts removed
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
        self.trial_overshoot_acts = 0
        self._last_logged_abs_x_err = None
        self._last_act_sent_at = None
        self._prev_act_x_err = None
        self._prev_act_cmd = None
        self._prev_act_dist_mm = None
        self._prev_act_correction_type = None  # 'distance' or 'x_axis'
        self._prev_act_trial = None
        self._prev_act_phase = None
        self._last_act_duration_s = 0.0
        self.speed_curve_alpha = 1.7
        self.speed_curve_cap = 26.0
        self.speed_curve_history = []
        self._speed_curve_plot_warned = False
        self.best_policy = None
        self.best_policy_score = None
        
        # Histories for smarter learning and metrics
        self.error_history = deque(maxlen=10)
        self.offset_history = deque(maxlen=STALL_WINDOW)
        
        # Production gatecheck: consecutive OR majority mode
        # Default: 12 consecutive frames OR 14 of last 26 frames
        self.success_tracker = SuccessGateTracker(
            consecutive_required=12,
            majority_window=26,
            majority_required=14
        )
        
        # Asymmetrical K (L vs R)
        self.K_L = 300.0
        self.K_R = 300.0
        self.last_dir = None
        self.last_align_dir = None  # Track direction changes for dampening
        
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

        self.running = True
        print(
            f"[ALIGN] X gate ({X_GATE_METRIC}): target {X_GATE_TARGET_MM:+.2f}mm tol {X_GATE_TOL_MM:.2f}mm "
            f"(using target {X_TARGET_MM:+.2f} tol {X_TOL_MM:.2f}, sign {X_AXIS_SIGN:+.0f})"
        )
        print(
            f"[ALIGN] Distance gate ({DIST_GATE_METRIC}): target {DIST_GATE_TARGET_MM:.2f}mm tol {DIST_TOL_MM:.2f}mm "
            f"(both gates must be satisfied)"
        )

    @staticmethod
    def _clamp(value, low, high):
        return max(float(low), min(float(high), float(value)))

    def _build_speed_curve_points(self):
        x_bins = list(SPEED_CURVE_X_BINS_MM)
        alpha = float(self._clamp(self.speed_curve_alpha, 0.8, 4.0))
        cap = float(self._clamp(self.speed_curve_cap, 5.0, 100.0))
        points = []
        max_x = max(1e-6, float(x_bins[-1]))
        for x_mm in x_bins:
            ratio = float(self._clamp(float(x_mm) / max_x, 0.0, 1.0))
            curved = ratio ** alpha
            score = 1.0 + (cap - 1.0) * curved
            points.append((float(x_mm), float(self._clamp(round(score), 1.0, 100.0))))
        return points

    def _record_speed_curve_snapshot(self, trial_idx):
        snapshot = {
            "trial": int(trial_idx),
            "alpha": float(self.speed_curve_alpha),
            "cap": float(self.speed_curve_cap),
            "points": self._build_speed_curve_points(),
        }
        self.speed_curve_history.append(snapshot)
        return snapshot

    def _render_speed_curve_png(self):
        if not bool(_MATPLOTLIB_AVAILABLE):
            if not self._speed_curve_plot_warned:
                print("[LEARN] matplotlib unavailable; skipping speed-curve PNG rendering.")
                self._speed_curve_plot_warned = True
            return
        if not self.speed_curve_history:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for idx, snap in enumerate(self.speed_curve_history):
            xs = [float(x) for x, _ in snap["points"]]
            ys = [float(y) for _, y in snap["points"]]
            color = cmap(idx % 10)
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.0, color=color, label=f"Trial {int(snap['trial'])}")

        ax.set_title("Speed Score Decision Curve by Trial")
        ax.set_xlabel("|x-axis offset error| (mm)")
        ax.set_ylabel("Speed score (1-100)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(1.0, 100.0)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(SPEED_CURVE_PNG, dpi=160)
        plt.close(fig)

    def update_speed_curve_after_trial(self, trial_idx, success, reason):
        old_alpha = float(self.speed_curve_alpha)
        old_cap = float(self.speed_curve_cap)

        progress = int(self.trial_progress_acts)
        worsening = int(self.trial_worsening_acts)
        overshoot = int(self.trial_overshoot_acts)
        adjusts = int(self.trial_adjust_count)
        reason_key = str(reason or "").strip().lower()

        lesson_bits = []

        # Conservative when unstable, modestly more assertive when clean.
        if overshoot > 0:
            self.speed_curve_alpha = float(self.speed_curve_alpha + (0.12 + 0.03 * min(6, overshoot)))
            self.speed_curve_cap = float(self.speed_curve_cap - (1.0 + 0.25 * min(8, overshoot)))
            lesson_bits.append(f"overshoot={overshoot} -> slower near target")

        if worsening > progress + 3:
            self.speed_curve_alpha = float(self.speed_curve_alpha + 0.08)
            self.speed_curve_cap = float(self.speed_curve_cap - 0.5)
            lesson_bits.append(f"worsening({worsening}>{progress}) -> dampen curve")
        elif progress > worsening + 5 and overshoot == 0:
            self.speed_curve_alpha = float(self.speed_curve_alpha - 0.06)
            self.speed_curve_cap = float(self.speed_curve_cap + 0.8)
            lesson_bits.append(f"progress({progress}>{worsening}) clean -> slightly bolder")

        if bool(success) and adjusts <= 25 and overshoot == 0:
            self.speed_curve_cap = float(self.speed_curve_cap + 0.8)
            lesson_bits.append("quick clean success -> raise far-offset cap")
        if reason_key == "max_adjusts" and adjusts >= int(MAX_ADJUSTS_PER_TRIAL):
            self.speed_curve_alpha = float(self.speed_curve_alpha + 0.06)
            lesson_bits.append("hit max_adjusts -> increase conservatism")

        # Tie cap gently to experiment-learned coarse speed while staying bounded.
        target_cap = float(self._clamp(8.0 + (0.60 * float(self.learned_speed_score)), 8.0, 65.0))
        self.speed_curve_cap = float((0.75 * float(self.speed_curve_cap)) + (0.25 * target_cap))

        self.speed_curve_alpha = float(self._clamp(self.speed_curve_alpha, 0.8, 4.0))
        self.speed_curve_cap = float(self._clamp(self.speed_curve_cap, 5.0, 100.0))

        snap = self._record_speed_curve_snapshot(trial_idx)
        self._render_speed_curve_png()

        lesson_text = "; ".join(lesson_bits) if lesson_bits else "stable trial -> minor smoothing update"
        print(
            f"[LEARN] Trial {int(trial_idx)} curve update: "
            f"alpha {old_alpha:.2f}->{self.speed_curve_alpha:.2f}, "
            f"cap {old_cap:.1f}->{self.speed_curve_cap:.1f} | {lesson_text}"
        )
        sample_points = ", ".join(
            f"{float(x):.1f}mm->{int(y)}" for x, y in snap["points"][:6]
        )
        print(
            f"[LEARN] Trial {int(trial_idx)} speed-score decision curve (sample): {sample_points} ... "
            f"(png: {SPEED_CURVE_PNG})"
        )
        self._append_event(
            {
                "type": "speed_curve_update",
                "run_id": self.run_id,
                "trial": int(trial_idx),
                "timestamp": time.time(),
                "alpha_before": float(old_alpha),
                "alpha_after": float(self.speed_curve_alpha),
                "cap_before": float(old_cap),
                "cap_after": float(self.speed_curve_cap),
                "lesson": str(lesson_text),
                "points": [{"x_mm": float(x), "score": float(y)} for x, y in snap["points"]],
            }
        )

    def _x_err_mm(self, offset_x):
        try:
            return (float(offset_x) * float(X_AXIS_SIGN)) - float(X_TARGET_MM)
        except (TypeError, ValueError):
            return 0.0

    def _active_x_tol_mm(self, dist_mm=None):
        """Return the fixed production gate tolerance (no distance normalization)."""
        return float(X_TOL_MM)

    def _distance_act_scale(self, dist_mm):
        if dist_mm is None:
            return 1.0, 1.0, "unknown"
        try:
            delta = float(dist_mm) - float(DIST_GATE_TARGET_MM)
        except (TypeError, ValueError):
            return 1.0, 1.0, "unknown"
        if delta >= float(DIST_ACT_FAR_DELTA_MM):
            return float(DIST_ACT_FAR_SCORE_SCALE), float(DIST_ACT_FAR_BUMP_SCALE), "far"
        if delta <= float(DIST_ACT_NEAR_DELTA_MM):
            return float(DIST_ACT_NEAR_SCORE_SCALE), float(DIST_ACT_NEAR_BUMP_SCALE), "near"
        return 1.0, 1.0, "mid"

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

            "config": {
                "num_trials": int(NUM_TRIALS),
                "experiment_trials": int(EXPERIMENT_TRIALS),
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
                "dist_act_far_threshold_scale": float(DIST_ACT_FAR_THRESHOLD_SCALE),
                "dist_act_near_threshold_scale": float(DIST_ACT_NEAR_THRESHOLD_SCALE),
                "max_overshoots_per_trial": int(MAX_OVERSHOOTS_PER_TRIAL),
                "reset_min_away_mm": float(RESET_MIN_AWAY_MM),
                "reset_max_away_mm": float(RESET_MAX_AWAY_MM),
                "reset_x_min_mm": float(RESET_X_MIN_MM),
                "reset_x_max_mm": float(RESET_X_MAX_MM),
                "learn_speed_start": int(LEARN_SPEED_START),
                "learn_speed_step": int(LEARN_SPEED_STEP),
                "learn_speed_bonus": int(LEARN_SPEED_BONUS),
                "pre_reset_vision_attempts": int(PRE_RESET_VISION_ATTEMPTS),
                "pre_reset_observe_timeout_s": float(PRE_RESET_OBSERVE_TIMEOUT_S),
                "pre_reset_pause_s": float(PRE_RESET_PAUSE_S),
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
                "max_gate_recheck_attempts": int(MAX_GATE_RECHECK_ATTEMPTS),
                "speed_curve_png": str(SPEED_CURVE_PNG),
            },
            "trials": [],
            "recovery_summary": None,

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
        # Old-style logging removed - using new formatted logs in align loop instead
        # self._emit_act_console(event)
        self._append_event(event)
        self.act_idx += 1
        return event

    def _emit_act_console(self, event):
        if not isinstance(event, dict):
            return
        phase = str(event.get("phase") or "?")
        reason = str(event.get("reason") or "")
        phase_key = phase.strip().lower()
        reason_key = reason.strip().lower()
        # Suppress noisy setup logs between trials; keep data in JSON events.
        if phase_key in {"reset", "reset_recovery"} or reason_key in {"between_trials"}:
            return
        
        trial = event.get("trial")
        act_idx = event.get("act")
        cmd = event.get("cmd")
        
        # Determine score to display
        score_for_display = event.get("score_requested")
        if score_for_display is None:
            score_for_display = event.get("score")
        if score_for_display is None:
            score_for_display = event.get("score_model")
        
        approx_score = None
        try:
            if score_for_display is not None:
                approx_score = int(round(float(score_for_display)))
        except (TypeError, ValueError):
            approx_score = None
        
        if approx_score is None:
            try:
                power_val = float(event.get("power"))
                cmd_for_quant = str(event.get("cmd_sent") or cmd).strip().lower()
                if cmd_for_quant:
                    _, q_score = telemetry_robot_module.quantize_speed(cmd_for_quant, speed=power_val)
                    approx_score = int(round(float(q_score)))
            except Exception:
                approx_score = None
        
        score_pct = int(approx_score) if approx_score is not None else 0
        
        # Extract current (pre-action) observation
        try:
            x_err_mm = float(event.get("x_err_mm"))
        except (TypeError, ValueError):
            x_err_mm = None
        try:
            x_tol_active = float(event.get("x_tol_active_mm"))
        except (TypeError, ValueError):
            x_tol_active = float(X_TOL_MM)
        try:
            dist_mm = float(event.get("dist"))
        except (TypeError, ValueError):
            dist_mm = None
        try:
            offset_x = float(event.get("offset_x"))
        except (TypeError, ValueError):
            offset_x = None
        
        result_delta_obj = None
        if (
            self._prev_act_x_err is not None
            and x_err_mm is not None
            and trial == self._prev_act_trial
            and phase == self._prev_act_phase
        ):
            result_delta_obj = build_align_result_delta_obj(
                correction_type=self._prev_act_correction_type,
                prev_x_err_mm=self._prev_act_x_err,
                curr_x_err_mm=x_err_mm,
                prev_dist_mm=self._prev_act_dist_mm,
                curr_dist_mm=dist_mm,
                dist_target_mm=float(DIST_TARGET_MM),
            )
        
        # --- SECOND: Show current act (pre-action state and command to send) ---
        pre_obs_text = format_align_pre_observation_text(
            x_err_mm=x_err_mm,
            x_tol_mm=x_tol_active,
            x_target_mm=float(X_TARGET_MM),
            offset_x=offset_x,
            dist_mm=dist_mm,
        )
        
        # Emit single-line diagnostic
        cmd_upper = str(cmd).upper() if cmd else "?"
        phase_color = COLOR_RED if phase.lower().startswith("recovery") else COLOR_WHITE
        
        # Extract motor command variables
        motor_power = event.get("power")
        motor_pwm = event.get("pwm")
        motor_duration_ms = event.get("duration_ms")
        
        line = format_shorthand_calibration_line(
            trial=trial,
            phase=phase,
            pre_obs_text=pre_obs_text,
            action_cmd=cmd_upper,
            action_score=score_pct,
            result_delta_obj=result_delta_obj,
            phase_color=phase_color,
            motor_power=motor_power,
            motor_pwm=motor_pwm,
            motor_duration_ms=motor_duration_ms,
        )
        print(line)
        
        # Store current act info for showing result on next iteration
        self._prev_act_x_err = x_err_mm
        self._prev_act_cmd = cmd
        self._prev_act_trial = trial
        self._prev_act_phase = phase
        self._prev_act_dist_mm = dist_mm
        self._prev_act_correction_type = getattr(self, '_current_correction_type', None)


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
            speed_score=PRECISION_SCORE_BASE,  # Report as score 1, not PWM-quantized
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
            # Update world model from the latest camera observation
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
            score_scale, _, dist_bucket = self._distance_act_scale(self._last_obs_dist)
            recovery_score = int(round(float(recovery_score) * float(score_scale)))
            recovery_score = max(int(DIST_ACT_SCORE_MIN), min(int(recovery_score), int(COARSE_SCORE)))
            if emit_logs:
                print(
                    f"[RECOVERY] reverse step {idx}/{len(plan)}: cmd={cmd.upper()} "
                    f"score={int(recovery_score)}% dist={dist_bucket}"
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
                    "dist_bucket": dist_bucket,
                    "dist_score_scale": float(score_scale),
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

    def ensure_pre_reset_visibility(self):
        attempts = 0
        while self.running and attempts < int(PRE_RESET_VISION_ATTEMPTS):
            pose = self.get_stable_pose(
                samples=1,
                timeout_s=float(PRE_RESET_OBSERVE_TIMEOUT_S),
                min_sample_time=self._observe_not_before_ts(),
            )
            if pose:
                if attempts:
                    self._append_event(
                        {
                            "type": "pre_reset_visibility_recovered",
                            "run_id": self.run_id,
                            "trial": self.current_trial,
                            "timestamp": time.time(),
                            "attempts": int(attempts),
                        }
                    )
                try:
                    offset = float(pose.get("offset_x"))
                except (TypeError, ValueError):
                    offset = None
                try:
                    dist = float(pose.get("dist"))
                except (TypeError, ValueError):
                    dist = None
                offset_text = f"{offset:+.2f}mm" if offset is not None else "--"
                dist_text = f"{dist:.1f}mm" if dist is not None else "--"
                print(f"[RESET] pre-check ok: offset={offset_text} dist={dist_text}")
                return pose
            attempts += 1
            self._append_event(
                {
                    "type": "pre_reset_visibility_retry",
                    "run_id": self.run_id,
                    "trial": self.current_trial,
                    "timestamp": time.time(),
                    "attempt": int(attempts),
                }
            )
            time.sleep(float(PRE_RESET_PAUSE_S))
        self._append_event(
            {
                "type": "pre_reset_visibility_failed",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "attempts": int(PRE_RESET_VISION_ATTEMPTS),
            }
        )
        if self.current_phase != "reset":
            self.current_phase = "reset"
        if self.recover_visibility(source="pre_reset"):
            return self.get_stable_pose(
                samples=1,
                timeout_s=float(PRE_RESET_OBSERVE_TIMEOUT_S),
                min_sample_time=self._observe_not_before_ts(),
            )
        return None

    def run_reset_phase(self, pose_start=None):
        """
        Simple two-act reset phase:
          - Act 1: Random distance move (f or b with random magnitude)
          - Act 2: Random x-axis move (l or r with random magnitude)
        No distance targeting, no direction validation - just move away from start position.
        Only recover if vision is lost.
        """
        if pose_start is None:
            pose_start = self.get_stable_pose(
                samples=1,
                timeout_s=2.0,
                min_sample_time=self._observe_not_before_ts(),
            )
        if not pose_start:
            return None
        start_off = float(pose_start["offset_x"])
        start_dist = float(pose_start.get("dist", 0.0))
        last_pose = pose_start
        reset_failed_vision = False

        # Pick random offsets for the two reset moves
        dist_magnitude = float(random.uniform(float(RESET_MIN_AWAY_MM), float(RESET_MAX_AWAY_MM)))
        x_offset = float(random.uniform(float(RESET_X_MIN_MM), float(RESET_X_MAX_MM)))
        
        # Pick directions: distance always backward, x-axis based on offset sign
        dist_dir = "b"  # Always move backward for distance reset
        x_dir = "l" if x_offset > 0 else "r"

        self._append_event(
            {
                "type": "reset_start",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "start_offset_x": start_off,
                "start_dist": start_dist,
                "dist_offset_goal": dist_magnitude,
                "x_offset_goal": x_offset,
                "reset_score": int(RESET_SCORE),
            }
        )

        # ACT 1: Distance move
        print(f"  [RESET] Act 1: Distance move {dist_dir} (offset ~{abs(dist_magnitude):.1f}mm)")
        self.keepalive_dir = dist_dir
        event1 = self.send_coarse_command(dist_dir, score=int(RESET_SCORE), reason="reset_distance_offset", extra={})
        time.sleep(max(0.02, float(BACKLASH_S)))
        
        pose_after_act1 = self.get_stable_pose(
            samples=1,
            timeout_s=RESET_OBSERVE_TIMEOUT_S,
            min_sample_time=self._observe_not_before_ts(),
        )
        if not pose_after_act1:
            print(f"           Vision lost after act 1, recovering...")
            self.current_phase = "reset_recovery"
            if not self.recover_visibility(source="reset_act1"):
                reset_failed_vision = True
                self.current_phase = "reset"
                return None
            self.current_phase = "reset"
            pose_after_act1 = self.get_stable_pose(
                samples=1,
                timeout_s=RESET_OBSERVE_TIMEOUT_S,
                min_sample_time=self._observe_not_before_ts(),
            )
            if not pose_after_act1:
                return None
        
        dist_after_act1 = float(pose_after_act1.get("dist", 0.0))
        off_after_act1 = float(pose_after_act1.get("offset_x", 0.0))
        print(f"           Result: dist={dist_after_act1:.1f}mm, offset={off_after_act1:+.2f}mm")

        # ACT 2: X-axis move
        # Scale bump percentage based on random offset magnitude (3x minimum for better visibility)
        x_offset_abs = abs(float(x_offset))
        x_range = float(RESET_X_MAX_MM) - float(RESET_X_MIN_MM)
        if x_range > 0:
            x_offset_ratio = x_offset_abs / (x_range / 2.0)  # Ratio of 0 to 1
            x_bump = 15.0 + (30.0 * min(x_offset_ratio, 1.0))  # 15% to 45% based on offset magnitude (3x previous)
        else:
            x_bump = 30.0  # Fallback (3x previous 10%)
        print(f"  [RESET] Act 2: X-axis move {x_dir} (offset ~{x_offset_abs:.1f}mm, bump={x_bump:.1f}%)")
        self.keepalive_dir = x_dir
        event2 = self.send_precision_command(x_dir, bump_pct=x_bump, reason="reset_x_offset", extra={})
        time.sleep(max(0.02, float(BACKLASH_S)))
        
        pose_after_act2 = self.get_stable_pose(
            samples=1,
            timeout_s=RESET_OBSERVE_TIMEOUT_S,
            min_sample_time=self._observe_not_before_ts(),
        )
        if not pose_after_act2:
            print(f"           Vision lost after act 2, recovering...")
            self.current_phase = "reset_recovery"
            if not self.recover_visibility(source="reset_act2"):
                reset_failed_vision = True
                self.current_phase = "reset"
                return None
            self.current_phase = "reset"
            pose_after_act2 = self.get_stable_pose(
                samples=1,
                timeout_s=RESET_OBSERVE_TIMEOUT_S,
                min_sample_time=self._observe_not_before_ts(),
            )
            if not pose_after_act2:
                return None
        
        dist_after_act2 = float(pose_after_act2.get("dist", 0.0))
        off_after_act2 = float(pose_after_act2.get("offset_x", 0.0))
        print(f"           Result: dist={dist_after_act2:.1f}mm, offset={off_after_act2:+.2f}mm")

        last_pose = pose_after_act2

        self._append_event(
            {
                "type": "reset_end",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "acts": 2,
                "final_offset_x": float(last_pose.get("offset_x")) if last_pose else None,
                "final_dist": float(last_pose.get("dist", 0.0)) if last_pose else None,
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
            score_scale, _, dist_bucket = self._distance_act_scale(self._last_obs_dist)
            scan_score = int(round(float(RECOVERY_SCAN_SCORE) * float(score_scale)))
            scan_score = max(int(DIST_ACT_SCORE_MIN), min(int(scan_score), int(COARSE_SCORE)))
            if emit_logs:
                print(
                    f"[RECOVERY] scan act {i}/{max_attempts}: cmd={scan_dir.upper()} "
                    f"score={int(scan_score)}% dist={dist_bucket} "
                    f"burst={burst_idx}/{int(max(1, RECOVERY_SCAN_BURST_ACTS))}"
                )
            event = self.send_coarse_command(
                scan_dir,
                score=int(scan_score),
                reason="recovery_scan",
                extra={
                    "scan_attempt": int(i),
                    "scan_total": int(max_attempts),
                    "scan_burst_index": int(burst_idx),
                    "scan_burst_size": int(max(1, RECOVERY_SCAN_BURST_ACTS)),
                    "dist_bucket": dist_bucket,
                    "dist_score_scale": float(score_scale),
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
            f"progress={int(self.trial_progress_acts)} worsening={int(self.trial_worsening_acts)} "
            f"overshoot={int(self.trial_overshoot_acts)}"
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
                "overshoot_acts": int(self.trial_overshoot_acts),
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
                "overshoot_acts": int(self.trial_overshoot_acts),
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

    def _policy_snapshot(self):
        return {
            "learned_speed_score": int(self.learned_speed_score),
            "speed_curve_alpha": float(self.speed_curve_alpha),
            "speed_curve_cap": float(self.speed_curve_cap),
        }

    def _apply_policy_snapshot(self, snap):
        if not isinstance(snap, dict):
            return False
        try:
            self.learned_speed_score = int(snap.get("learned_speed_score", self.learned_speed_score))
            self.speed_curve_alpha = float(snap.get("speed_curve_alpha", self.speed_curve_alpha))
            self.speed_curve_cap = float(snap.get("speed_curve_cap", self.speed_curve_cap))
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _policy_candidate_key(*, success, overshoot, adjust_count, worsening, progress, total_time):
        try:
            t_val = float(total_time) if total_time is not None else 1e9
        except (TypeError, ValueError):
            t_val = 1e9
        return (
            1 if bool(success) else 0,
            -int(overshoot),
            -int(adjust_count),
            -(int(worsening) - int(progress)),
            -float(t_val),
        )

    def _consider_best_policy(self, *, trial_idx, trial_success, trial_reason, total_time, policy_before_trial):
        if not isinstance(policy_before_trial, dict):
            return
        candidate_key = self._policy_candidate_key(
            success=bool(trial_success),
            overshoot=int(self.trial_overshoot_acts),
            adjust_count=int(self.trial_adjust_count),
            worsening=int(self.trial_worsening_acts),
            progress=int(self.trial_progress_acts),
            total_time=total_time,
        )
        if self.best_policy_score is None or candidate_key > self.best_policy_score:
            self.best_policy_score = candidate_key
            self.best_policy = {
                **policy_before_trial,
                "trial": int(trial_idx),
                "success": bool(trial_success),
                "reason": str(trial_reason or ""),
                "adjust_count": int(self.trial_adjust_count),
                "overshoot_acts": int(self.trial_overshoot_acts),
                "worsening_acts": int(self.trial_worsening_acts),
                "progress_acts": int(self.trial_progress_acts),
                "seconds_to_success": (None if total_time is None else float(total_time)),
            }
            print(
                f"[LEARN] New champion policy from trial {int(trial_idx)}: "
                f"success={bool(trial_success)} reason={str(trial_reason)} "
                f"adj={int(self.trial_adjust_count)} over={int(self.trial_overshoot_acts)}"
            )

    def run_pretrial_random_turn(self):
        """One random heading turn before reset so L/R orientation changes are visible."""
        turn_dir = random.choice(["l", "r"])
        print(f"  [RESET] Pre-turn: random heading move {turn_dir.upper()} @ {int(RESET_SCORE)}%")
        self.keepalive_dir = turn_dir
        self.send_coarse_command(
            turn_dir,
            score=int(RESET_SCORE),
            reason="pretrial_random_turn",
            extra={"turn_dir": str(turn_dir), "reset_score": int(RESET_SCORE)},
        )
        time.sleep(max(0.02, float(BACKLASH_S)))
        pose_after_turn = self.get_stable_pose(
            samples=1,
            timeout_s=RESET_OBSERVE_TIMEOUT_S,
            min_sample_time=self._observe_not_before_ts(),
        )
        if not pose_after_turn:
            print("           Vision lost after pre-turn, recovering...")
            self.current_phase = "reset_recovery"
            if not self.recover_visibility(source="pretrial_random_turn"):
                self.current_phase = "reset"
                return None
            self.current_phase = "reset"
            pose_after_turn = self.get_stable_pose(
                samples=1,
                timeout_s=RESET_OBSERVE_TIMEOUT_S,
                min_sample_time=self._observe_not_before_ts(),
            )
            if not pose_after_turn:
                return None
        self._append_event(
            {
                "type": "reset_pretrial_turn",
                "run_id": self.run_id,
                "trial": self.current_trial,
                "timestamp": time.time(),
                "cmd": str(turn_dir),
                "score": int(RESET_SCORE),
                "offset_x": float(pose_after_turn.get("offset_x")) if pose_after_turn.get("offset_x") is not None else None,
                "dist": float(pose_after_turn.get("dist")) if pose_after_turn.get("dist") is not None else None,
            }
        )
        return pose_after_turn

    @staticmethod
    def _grade_letter(score):
        try:
            val = float(score)
        except (TypeError, ValueError):
            return "F"
        if val >= 90.0:
            return "A"
        if val >= 80.0:
            return "B"
        if val >= 70.0:
            return "C"
        if val >= 60.0:
            return "D"
        return "F"

    def _trial_grade_rows(self):
        if not isinstance(self.results_log, dict):
            return []
        trials = self.results_log.get("trials")
        if not isinstance(trials, list):
            return []

        rows = []
        for row in trials:
            if not isinstance(row, dict):
                continue
            try:
                trial_idx = int(row.get("trial"))
            except (TypeError, ValueError):
                continue

            success = bool(row.get("success"))
            reason = str(row.get("reason") or "")
            adjusts = int(row.get("adjust_count") or 0)
            progress = int(row.get("progress_acts") or 0)
            worsening = int(row.get("worsening_acts") or 0)
            overshoot = int(row.get("overshoot_acts") or 0)

            # Safety clamp: each of these are act-level counters and should never exceed adjusts.
            # Keeps grading robust even when reading older logs produced before metric fixes.
            if adjusts >= 0:
                progress = max(0, min(progress, adjusts))
                worsening = max(0, min(worsening, adjusts))
                overshoot = max(0, min(overshoot, adjusts))

            # Efficiency-weighted score: rewards fast/clean convergence and penalizes drift/overshoot.
            acts = max(1, int(adjusts))
            progress_rate = float(progress) / float(acts)
            worsening_rate = float(worsening) / float(acts)
            overshoot_rate = float(overshoot) / float(acts)
            act_load = float(self._clamp(float(adjusts) / float(max(1, int(MAX_ADJUSTS_PER_TRIAL))), 0.0, 1.0))

            # Ratio-based scoring keeps grades meaningful across short/long trials.
            score = 80.0
            if success:
                score += 15.0
            else:
                score -= 20.0

            score += (12.0 * progress_rate)
            score -= (16.0 * worsening_rate)
            score -= (26.0 * overshoot_rate)
            score -= (30.0 * act_load)

            reason_key = str(reason or "").strip().lower()
            if reason_key == "max_adjusts":
                score -= 12.0
            if reason_key in {"lost_vision", "pre_reset_failed_vision", "reset_failed_vision"}:
                score -= 8.0
            if success and reason_key == "success":
                score = max(60.0, float(score))
            score = max(0.0, min(100.0, float(score)))

            rows.append(
                {
                    "trial": int(trial_idx),
                    "reason": reason,
                    "success": bool(success),
                    "adjust_count": int(adjusts),
                    "progress_acts": int(progress),
                    "worsening_acts": int(worsening),
                    "overshoot_acts": int(overshoot),
                    "score": float(score),
                    "grade": str(self._grade_letter(score)),
                }
            )
        return rows

    def print_grade_summary(self):
        rows = self._trial_grade_rows()
        if not rows:
            print("[GRADE] No trial rows available.")
            return None

        print("\n[GRADE] Trial grades:")
        for row in rows:
            print(
                f"  Trial {int(row['trial']):2d}: "
                f"{row['grade']} ({float(row['score']):5.1f}) | "
                f"reason={row['reason']} | "
                f"adj={int(row['adjust_count'])} prog={int(row['progress_acts'])} "
                f"worse={int(row['worsening_acts'])} over={int(row['overshoot_acts'])}"
            )

        avg_score = statistics.mean(float(row["score"]) for row in rows)
        avg_grade = self._grade_letter(avg_score)
        print(f"[GRADE] Average: {avg_grade} ({float(avg_score):.1f})")
        return {"rows": rows, "avg_score": float(avg_score), "avg_grade": str(avg_grade)}

    def adapt_learning_after_trial(self, *, success, reason):
        """
        Apply learning update from every trial.
        Success trials use update_learning_speed(); this handles failed-trial adaptation.
        """
        if bool(success):
            return

        old_score = int(self.learned_speed_score)
        delta = 0

        overshoot = int(self.trial_overshoot_acts)
        progress = int(self.trial_progress_acts)
        worsening = int(self.trial_worsening_acts)
        adjusts = int(self.trial_adjust_count)
        reason_key = str(reason or "").strip().lower()

        # Strongly penalize repeated overshoot.
        if overshoot > 0:
            delta -= min(4, 1 + (overshoot // 2))

        # Reward/penalize trend quality.
        if progress > (worsening + 3) and overshoot == 0:
            delta += 1
        elif worsening > (progress + 3):
            delta -= 1

        # Endpoint penalties for known bad outcomes.
        if reason_key == "max_adjusts" and adjusts >= int(MAX_ADJUSTS_PER_TRIAL):
            delta -= 1
        if reason_key in {"lost_vision", "pre_reset_failed_vision", "reset_failed_vision"}:
            delta -= 1

        new_score = max(1, min(int(COARSE_SCORE), int(old_score + delta)))
        self.learned_speed_score = int(new_score)

        if int(new_score) != int(old_score):
            direction = "up" if int(new_score) > int(old_score) else "down"
            print(
                f"[LEARN] Failed-trial adaptation ({reason_key or 'unknown'}): "
                f"speed {old_score}% -> {new_score}% ({direction})."
            )
        else:
            print(
                f"[LEARN] Failed-trial adaptation ({reason_key or 'unknown'}): "
                f"speed unchanged at {new_score}%."
            )

    def run_manual_test_mode(self):
        """Manual testing mode: observe vision and suggest acts without moving robot."""
        print("\n" + "="*80)
        print("MANUAL TEST MODE - Gate Checker & Act Logic Tester")
        print("="*80)
        print("This mode observes vision and suggests actions WITHOUT moving the robot.")
        print("Move the brick manually with your hands to test gate checker and act logic.")
        print("Press Ctrl+C to exit.")
        print("="*80 + "\n")
        
        self.running = True
        print(f"[MANUAL] X gate: target {X_TARGET_MM:.2f}mm tol {X_TOL_MM:.2f}mm")
        print(f"[MANUAL] Distance gate: target {DIST_GATE_TARGET_MM:.2f}mm tol {DIST_GATE_TOL_MM:.2f}mm")
        print()
        
        # Initialize success tracker for gate checking
        success_tracker = SuccessGateTracker(
            consecutive_required=12,
            majority_window=26,
            majority_required=14
        )
        
        frame_count = 0
        last_offset_x = None
        last_x_err = None
        
        try:
            while self.running:
                # Observe current state
                found, angle, dist, offset_x, conf, cam_h, above, below = self.vision.read()
                frame_count += 1
                self.world._frame_id = frame_count
                
                if not found:
                    print(f"\r[MANUAL] Frame {frame_count}: No brick visible" + " "*30, end="", flush=True)
                    time.sleep(0.1)
                    continue
                
                # Update world brick state
                self.world.brick = {
                    "visible": True,
                    "offset_x": float(offset_x),
                    "dist": float(dist),
                    "angle": float(angle),
                    "confidence": float(conf),
                }
                
                # Extract current measurements
                offset_x = float(offset_x)
                dist = float(dist)
                x_err = self._x_err_mm(offset_x)
                abs_error = abs(x_err)
                active_x_tol = self._active_x_tol_mm(dist)
                
                # Run gate checker
                gate_eval_ok, confidence = evaluate_gate_status(self.world, "ALIGN_BRICK")
                success_ok = bool(gate_eval_ok)
                success_met = update_gatecheck(
                    self.world,
                    "ALIGN_BRICK",
                    success_tracker,
                    success_ok,
                    phase="manual_test"
                )
                
                status = getattr(self.world, "_gatecheck_status", {}) or {}
                streak = status.get("streak", 0)
                window_pass = status.get("window_pass", 0)
                window_size = status.get("window_size", 0)
                truth_by = status.get("truth_by", "none")
                
                # Calculate what action we would suggest
                align_dir = 'r' if x_err > 0 else 'l'
                direction_changed = (last_x_err is not None and 
                                   (last_x_err < 0 and x_err > 0 or last_x_err > 0 and x_err < 0))
                
                score_scale, bump_scale, dist_bucket = self._distance_act_scale(dist)
                
                base_near_threshold = float(active_x_tol) + 4.0
                if dist_bucket == "far":
                    near_threshold = float(base_near_threshold) * float(DIST_ACT_FAR_THRESHOLD_SCALE)
                elif dist_bucket == "near":
                    near_threshold = float(base_near_threshold) * float(DIST_ACT_NEAR_THRESHOLD_SCALE)
                else:
                    near_threshold = float(base_near_threshold)
                
                in_settling_zone = abs_error <= (float(active_x_tol) * float(SETTLING_ZONE_MULTIPLIER))
                
                # Build suggestion
                if abs_error > near_threshold:
                    mode = "COARSE"
                    score = min(self.learned_speed_score, COARSE_SCORE)
                    overshoot_scale = 1.0
                    dir_scale = float(DIRECTION_CHANGE_DAMPENING) if direction_changed else 1.0
                    score = int(round(float(score) * float(score_scale) * float(overshoot_scale) * float(dir_scale)))
                    suggestion = f"{align_dir.upper()} score={score} (speed ~{score}%)"
                else:
                    mode = "PRECISION"
                    bump_pct = self.precision_bump_pct
                    bump_pct = float(bump_pct) * float(bump_scale)
                    
                    proportional_scale = 1.0
                    if PROPORTIONAL_CONTROL_ENABLED:
                        error_ratio = abs_error / max(0.1, float(near_threshold))
                        proportional_scale = 0.5 + (0.5 * min(1.0, error_ratio))
                        bump_pct = float(bump_pct) * float(proportional_scale)
                    
                    settling_scale = 1.0
                    if in_settling_zone:
                        settling_scale = 0.4
                        bump_pct = float(bump_pct) * settling_scale
                    
                    dir_scale = 1.0
                    if direction_changed:
                        dir_scale = float(DIRECTION_CHANGE_DAMPENING)
                        bump_pct = float(bump_pct) * dir_scale
                    
                    features = []
                    if in_settling_zone:
                        features.append(f"settling({settling_scale:.2f}x)")
                    if direction_changed:
                        features.append(f"dir_change({dir_scale:.2f}x)")
                    if PROPORTIONAL_CONTROL_ENABLED and proportional_scale != 1.0:
                        features.append(f"prop({proportional_scale:.2f}x)")
                    
                    suggestion = f"{align_dir.upper()} bump={bump_pct:.1f}%"
                    if features:
                        suggestion += f" [{', '.join(features)}]"
                
                # Color code offset for display
                if abs_error > 10.0:
                    offset_color = COLOR_RED
                elif abs_error > 3.0:
                    offset_color = COLOR_ORANGE_BRIGHT
                else:
                    offset_color = COLOR_GREEN
                
                # Success gate status color
                if success_met:
                    gate_color = COLOR_GREEN
                    gate_status = f"✓ SUCCESS ({truth_by})"
                elif success_ok:
                    gate_color = COLOR_WHITE
                    gate_status = f"streak={streak}/{12} win={window_pass}/{window_size}"
                else:
                    gate_color = COLOR_GRAY
                    gate_status = f"not satisfied"
                
                # Display line
                print(
                    f"\r[{mode:9s}] "
                    f"{offset_color}offset={offset_x:+7.2f}mm{COLOR_RESET} "
                    f"x_err={x_err:+6.2f}mm "
                    f"dist={dist:6.1f}mm "
                    f"tol={active_x_tol:.2f}mm "
                    f"{dist_bucket:3s} | "
                    f"{gate_color}{gate_status:30s}{COLOR_RESET} | "
                    f"→ {suggestion}",
                    end="",
                    flush=True
                )
                
                # Update world state (updates overlay internally)
                self.update_world(found, angle, dist, offset_x, conf, cam_h, above, below, self.vision.current_frame)
                
                last_offset_x = offset_x
                last_x_err = x_err
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\n[MANUAL] Stopping manual test mode...")
        finally:
            print("\n[MANUAL] Final gate status:")
            status = getattr(self.world, "_gatecheck_status", {}) or {}
            print(f"  Consecutive streak: {status.get('streak', 0)}/{status.get('need', 12)}")
            print(f"  Majority window: {status.get('window_pass', 0)}/{status.get('window_size', 0)} (need {status.get('window_need', 14)})")
            print(f"  Truth by: {status.get('truth_by', 'none')}")
            print()

    def run_experiment(self):
        # WIPE DATA FILE ON START
        if Path(DATA_FILE).exists():
            print(f"[ALIGN] Wiping {DATA_FILE} for fresh learning...")
            Path(DATA_FILE).unlink()
        if self.results_file_path.exists():
            print(f"[ALIGN] Wiping {self.results_file_path} for fresh results...")
            self.results_file_path.unlink()
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
            "dist_act_far_threshold_scale": float(DIST_ACT_FAR_THRESHOLD_SCALE),
            "dist_act_near_threshold_scale": float(DIST_ACT_NEAR_THRESHOLD_SCALE),
            "max_overshoots_per_trial": int(MAX_OVERSHOOTS_PER_TRIAL),
            "reset_min_away_mm": float(RESET_MIN_AWAY_MM),
            "reset_max_away_mm": float(RESET_MAX_AWAY_MM),
            "reset_x_min_mm": float(RESET_X_MIN_MM),
            "reset_x_max_mm": float(RESET_X_MAX_MM),
            "pre_reset_vision_attempts": int(PRE_RESET_VISION_ATTEMPTS),
            "pre_reset_observe_timeout_s": float(PRE_RESET_OBSERVE_TIMEOUT_S),
            "pre_reset_pause_s": float(PRE_RESET_PAUSE_S),
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
        self._record_speed_curve_snapshot(trial_idx=0)
        self._render_speed_curve_png()
        print(f"[LEARN] Initialized baseline speed-score curve PNG: {SPEED_CURVE_PNG}")
        
        for i in range(1, NUM_TRIALS + 1):
            if not self.running:
                break
            if self.abort_run_early:
                break
            if int(i) == int(NUM_TRIALS) and isinstance(self.best_policy, dict):
                if self._apply_policy_snapshot(self.best_policy):
                    print(
                        f"[LEARN] Trial {int(i)} champion replay from trial {int(self.best_policy.get('trial', -1))}: "
                        f"score={int(self.learned_speed_score)} alpha={float(self.speed_curve_alpha):.2f} cap={float(self.speed_curve_cap):.1f}"
                    )
            self.current_trial = i
            policy_before_trial = self._policy_snapshot()
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
            self.trial_overshoot_acts = 0
            self.trial_first_overshoot_penalty_applied = False  # One-strike rule flag
            self._align_frame_count = 0
            self._last_logged_abs_x_err = None
            self._prev_act_x_err = None
            self._prev_act_cmd = None
            self._prev_act_trial = None
            self._prev_act_phase = None
            
            # Reset gatecheck tracker and tracking variables for new trial
            self.success_tracker = SuccessGateTracker(
                consecutive_required=12,
                majority_window=26,
                majority_required=14
            )
            self._prev_act_correction_type = None
            self._current_correction_type = None
            
            print(f"\n\n\nTRIAL {i}/{NUM_TRIALS}")
            print(
                "[TRIAL] success gate: "
                "visible=true AND |x_err| <= x_tol AND |dist - dist_target| <= dist_tol "
                "for 12 consecutive frames OR 14/26 majority (PRODUCTION GATES)"
            )
            print(
                "[TRIAL] Gate criteria: "
                f"X: {X_TARGET_MM:+.2f}mm ± {X_TOL_MM:.2f}mm | "
                f"Dist: {DIST_TARGET_MM:.2f}mm ± {DIST_TOL_MM:.2f}mm (BOTH must be satisfied)"
            )
            trial_success = False
            trial_reason = "unknown"
            
            # 1. CHECK GATES BEFORE RESET
            # Only perform reset if we're NOT already in success gates
            # Reset tracking variables for a fresh trial
            self._prev_act_x_err = None
            self._prev_act_cmd = None
            self._prev_act_dist_mm = None
            self._prev_act_correction_type = None
            self._current_correction_type = None
            
            self.current_phase = "reset"
            pre_reset_pose = self.ensure_pre_reset_visibility()
            if not pre_reset_pose:
                print("  [ERROR] Vision check failed before reset.")
                trial_reason = "pre_reset_failed_vision"
                self.print_trial_recovery_summary(i, success=False, reason=trial_reason)
                if self.abort_run_early:
                    break
                continue

            pose_after_turn = self.run_pretrial_random_turn()
            if not pose_after_turn:
                print("  [ERROR] Pre-trial random turn failed (vision recovery).")
                trial_reason = "reset_failed_vision"
                self.print_trial_recovery_summary(i, success=False, reason=trial_reason)
                if self.abort_run_early:
                    break
                continue
            
            print("  [RESET] Executing reset...")
            pose_reset = self.run_reset_phase(pose_start=pose_after_turn)
            if not pose_reset:
                print("  [ERROR] Cannot find brick for reset setup.")
                trial_reason = "reset_failed_vision"
                self.print_trial_recovery_summary(i, success=False, reason=trial_reason)
                if self.abort_run_early:
                    break
                continue
            
            # Clear any previous correction type for fresh align phase
            self._current_correction_type = None
            
            # Show reset completion
            reset_dist = float(pose_reset.get("dist", 0.0))
            reset_off = float(pose_reset.get("offset_x", 0.0))
            print(
                f"  [RESET OK] Repositioned to dist={reset_dist:.1f}mm, offset={reset_off:+.2f}mm"
            )

            # Reset result tracking for this trial
            self._prev_act_x_err = None
            self._prev_act_cmd = None
            self._prev_act_dist_mm = None
            self._prev_act_correction_type = None
            
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
                f"x_target={X_TARGET_MM:+.2f}mm ± {X_TOL_MM:.2f}mm | "
                f"dist_target={DIST_TARGET_MM:.2f}mm ± {DIST_TOL_MM:.2f}mm | "
                f"start_dist={dist_text} start_x_err={err_text}"
            )
            try:
                axis_sign = float(X_AXIS_SIGN)
                axis_sign = axis_sign if axis_sign != 0.0 else 1.0
            except (TypeError, ValueError):
                axis_sign = 1.0
            center_offset = float(X_TARGET_MM) / float(axis_sign)
            tol_span = float(start_active_tol) / float(abs(axis_sign))
            gate_left = center_offset - tol_span
            gate_right = center_offset + tol_span
            print(
                "[TRIAL] gate diagram: "
                f"SuccessGateEdge_left (offset={gate_left:+.2f}mm, x_err={-tol_span:+.2f}mm) ---- "
                f"center (offset={center_offset:+.2f}mm, x_err={0.0:+.2f}mm) ---- "
                f"SuccessGateEdge_right (offset={gate_right:+.2f}mm, x_err={tol_span:+.2f}mm)"
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
            prev_align_x_err = None
            prev_dist_err = None  # Track previous distance error for change highlighting
            pending_eval_after_act = False
            pending_pre_act_x_err = None
            pending_pre_act_abs_x_err = None
            _in_gate_observation = False  # Track if we're currently observing gates
            _gate_observation_frame_count = 0  # How many frames since last adjustment
            _gate_verbose_active = False  # Mirrors full gate-check active state for log visibility
            _full_gate_check_active = False  # Retry loop only after first true/true trigger
            _gate_recheck_attempts = 0
            
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
                    prev_dist_err = None  # Reset distance error tracking on recovery
                    trial_start_t = time.time() # Reset timer if we lost track
                
                off_now = pose["offset_x"]
                dist_now = pose.get("dist")
                self.offset_history.append(off_now)
                if trial_start_offset is None: trial_start_offset = off_now
                
                x_err = self._x_err_mm(off_now)
                abs_error_from_target = abs(x_err)
                active_x_tol = self._active_x_tol_mm(dist_now)
                dist_err = abs(float(dist_now) - float(DIST_TARGET_MM)) if dist_now else float('inf')
                # Evaluate exactly one act outcome on the first fresh observation after a command.
                if bool(pending_eval_after_act):
                    if pending_pre_act_abs_x_err is not None:
                        delta_abs_err = float(pending_pre_act_abs_x_err) - float(abs_error_from_target)
                        if delta_abs_err > 0.02:
                            self.trial_progress_acts = int(self.trial_progress_acts + 1)
                        elif delta_abs_err < -0.02:
                            self.trial_worsening_acts = int(self.trial_worsening_acts + 1)

                    overshoot_occurred = False
                    if (
                        pending_pre_act_x_err is not None
                        and (float(pending_pre_act_x_err) * float(x_err)) < 0.0
                        and abs(float(pending_pre_act_x_err)) > float(OVERSHOOT_SIGN_FLIP_DEADZONE_MM)
                        and abs(float(x_err)) > float(OVERSHOOT_SIGN_FLIP_DEADZONE_MM)
                    ):
                        overshoot_occurred = True
                        self.trial_overshoot_acts = int(self.trial_overshoot_acts + 1)

                        overshoot_num = int(self.trial_overshoot_acts)
                        print(
                            f"  [OVERSHOOT #{overshoot_num}] Crossed target: "
                            f"prev_x_err={float(pending_pre_act_x_err):+.2f}mm → curr_x_err={float(x_err):+.2f}mm | "
                            f"dist={float(dist_now):.1f}mm | "
                            f"Error magnitude: {abs(float(pending_pre_act_x_err)):.2f}mm → {abs_error_from_target:.2f}mm"
                        )

                        # ONE-STRIKE RULE: After first overshoot, cut learned_speed_score.
                        if int(i) <= int(EXPERIMENT_TRIALS) and not self.trial_first_overshoot_penalty_applied:
                            penalty_reduction = float(OVERSHOOT_FIRST_PENALTY_PCT) / 100.0
                            old_score = self.learned_speed_score
                            self.learned_speed_score = int(round(float(self.learned_speed_score) * (1.0 - penalty_reduction)))
                            self.learned_speed_score = max(1, self.learned_speed_score)
                            self.trial_first_overshoot_penalty_applied = True
                            print(
                                f"  [ONE-STRIKE] First overshoot penalty applied: "
                                f"learned_speed_score {old_score} → {self.learned_speed_score} "
                                f"(-{OVERSHOOT_FIRST_PENALTY_PCT:.0f}%)"
                            )

                    pending_eval_after_act = False
                    pending_pre_act_x_err = None
                    pending_pre_act_abs_x_err = None

                prev_align_abs_err = float(abs_error_from_target)
                # Note: prev_align_x_err and prev_dist_err are updated ONLY when taking actions
                # (after send_coarse_command or send_precision_command below)
                # This ensures delta calculations show true act-to-act changes, not frame-to-frame drift
                
                # Increment gate observation frame count
                _gate_observation_frame_count += 1
                
                # Check Gate Success using production SuccessGateTracker
                # Evaluates: consecutive frames OR majority window
                # Gates: visible=true AND |x_err| <= x_tol AND |dist - dist_target| <= dist_tol
                gate_eval_ok, confidence = evaluate_gate_status(self.world, "ALIGN_BRICK")
                
                # Detailed gate analysis for debugging and strict local truth check.
                local_gate = align_local_gate_status(
                    self.world,
                    "ALIGN_BRICK",
                    process_rules=self.world.process_rules,
                )
                x_within_tol = bool(local_gate.get("x_within_tol"))
                dist_within_tol = bool(local_gate.get("dist_within_tol"))
                local_gate_ok = bool(local_gate.get("ok"))
                success_ok = bool(gate_eval_ok and local_gate_ok)
                
                # Colorize boolean results
                x_status = f"{COLOR_GREEN}true{COLOR_RESET}" if x_within_tol else f"{COLOR_RED}false{COLOR_RESET}"
                dist_status = f"{COLOR_GREEN}true{COLOR_RESET}" if dist_within_tol else f"{COLOR_RED}false{COLOR_RESET}"
                x_err_gate = local_gate.get("x_err")
                if x_err_gate is None:
                    x_err_gate = x_err
                x_tol_gate = float(local_gate.get("x_tol") or X_TOL_MM)
                dist_gate = local_gate.get("dist")
                if dist_gate is None:
                    dist_gate = dist_now
                dist_target_gate = float(local_gate.get("dist_target") or DIST_TARGET_MM)
                dist_err_gate = local_gate.get("dist_err")
                if dist_err_gate is None:
                    dist_err_gate = dist_err
                dist_tol_gate = float(local_gate.get("dist_tol") or DIST_TOL_MM)
                gate_analysis = (
                    f"X:[{float(x_err_gate):+.2f}mm<={x_tol_gate:.2f}? {x_status}] | "
                    f"Dist:[{float(dist_gate):.1f}mm≠{dist_target_gate:.2f}mm={float(dist_err_gate):.2f}"
                    f"±{dist_tol_gate:.2f}? {dist_status}]"
                )
                
                # Track frame count for periodic reporting
                if not hasattr(self, '_align_frame_count'):
                    self._align_frame_count = 0
                self._align_frame_count += 1
                
                success_met = update_gatecheck(
                    self.world,
                    "ALIGN_BRICK",
                    self.success_tracker,
                    success_ok,
                    phase="align"
                )
                
                was_in_gate_observation = bool(_in_gate_observation)
                # Update gate observation state based on success_ok
                if local_gate_ok:
                    _in_gate_observation = True  # Start/continue observation phase
                    _gate_recheck_attempts = 0
                else:
                    # Before first true/true, observe briefly before deciding to act.
                    # After a true/true session begins, retry handling below controls exit timing.
                    if _full_gate_check_active and was_in_gate_observation:
                        _in_gate_observation = True
                    else:
                        _in_gate_observation = _gate_observation_frame_count < 5
                
                # Once we first pass true/true, keep verbose gate logs on every frame.
                if local_gate_ok and not _full_gate_check_active:
                    _gate_verbose_active = True
                    _full_gate_check_active = True
                    trial_num = int(self.current_trial)
                    print(
                        f"  [T{trial_num} GATE] First true/true observed. "
                        "Enabling full per-frame gate logs."
                    )

                # Report gate status only while full gate-check session is active.
                if _full_gate_check_active:
                    status = getattr(self.world, "_gatecheck_status", {}) or {}
                    streak = status.get("streak", 0)
                    streak_need = status.get("need", 12)
                    window_pass = status.get("window_pass", 0)
                    window_size = status.get("window_size", 0)
                    window_need = status.get("window_need", 14)
                    truth_by = status.get("truth_by", "none")
                    trial_num = int(self.current_trial)
                    print(
                        f"  [T{trial_num} GATE] Frame #{self._align_frame_count}: {gate_analysis} | "
                        f"Streak: {streak}/{streak_need} | Majority: {window_pass}/{window_size}/{window_need} | "
                        f"truth_by={truth_by}"
                    )

                # Mid-gate-check failure: immediate re-check loop up to MAX_GATE_RECHECK_ATTEMPTS.
                if _full_gate_check_active and was_in_gate_observation and not local_gate_ok:
                    _gate_recheck_attempts = int(_gate_recheck_attempts + 1)
                    trial_num = int(self.current_trial)
                    if _gate_recheck_attempts <= int(MAX_GATE_RECHECK_ATTEMPTS):
                        print(
                            f"  [T{trial_num} GATE] Mid-check failure "
                            f"(retry {_gate_recheck_attempts}/{int(MAX_GATE_RECHECK_ATTEMPTS)}): "
                            "condition turned false; immediate re-check."
                        )
                        time.sleep(0.01)
                        continue
                    print(
                        f"  [T{trial_num} GATE] Re-check retries exhausted "
                        f"({_gate_recheck_attempts}/{int(MAX_GATE_RECHECK_ATTEMPTS)}). "
                        "Resuming adjustment actions."
                    )
                    _in_gate_observation = False
                    _gate_verbose_active = False
                    _full_gate_check_active = False
                    _gate_recheck_attempts = 0
                
                if success_met:
                    total_time = time.time() - trial_start_t
                    avg_recent = None
                    if self.error_history:
                        recent = list(self.error_history)[-3:]
                        avg_recent = statistics.mean(recent)
                    
                    status = getattr(self.world, "_gatecheck_status", {}) or {}
                    truth_by = status.get("truth_by", "unknown")
                    streak = status.get("streak", 0)
                    window_pass = status.get("window_pass", 0)
                    window_size = status.get("window_size", 0)
                    
                    self.print_trial_summary(i, off_now, total_time)
                    self.log_trial(i, trial_start_offset, off_now, total_time)
                    if int(i) <= int(EXPERIMENT_TRIALS):
                        self.update_learning_speed(total_time, avg_recent=avg_recent)
                    else:
                        print(
                            f"[LEARN] Trial {int(i)} in exploit-only mode "
                            f"(no exploratory speed update; locked after trial {int(EXPERIMENT_TRIALS)})."
                        )
                    
                    # *** PROMINENT VICTORY MESSAGE ***
                    print(f"\n{'='*80}")
                    print(f"{COLOR_GREEN}{'████ VICTORY ████':^80}{COLOR_RESET}")
                    print(f"{'='*80}")
                    print(
                        f"{COLOR_GREEN}[VICTORY] {truth_by} gate achieved:{COLOR_RESET}\n"
                        f"  Consecutive streak: {streak} frames\n"
                        f"  Majority window: {window_pass}/{window_size}\n"
                        f"  Success after {int(self.trial_adjust_count)} total acts:\n"
                        f"    - {int(self.trial_progress_acts)} progress acts (improved error)\n"
                        f"    - {int(self.trial_worsening_acts)} worsening acts (made error worse)\n"
                        f"    - {int(self.trial_overshoot_acts)} overshoot acts (sign flip)"
                    )
                    print(f"{'='*80}\n")
                    
                    # Stop between trials; next motion should come from an explicit reset/align decision.
                    self.robot.stop()
                    trial_success = True
                    trial_reason = "success"
                    break
                
                # ONLY take a new adjustment if we're NOT in gate observation phase
                if _in_gate_observation:
                    # Keep observing, don't take a new adjustment
                    pass
                else:
                    # We can take a new adjustment
                    # Check trial limits before attempting another adjustment
                    if int(self.trial_adjust_count) >= int(MAX_ADJUSTS_PER_TRIAL):
                        trial_reason = "max_adjusts"
                        print(
                            f"  [STOP] Reached max adjusts for trial: "
                            f"{int(self.trial_adjust_count)}/{int(MAX_ADJUSTS_PER_TRIAL)}."
                        )
                        break
                    
                    # Reset gate observation counter when taking new adjustment
                    _gate_observation_frame_count = 0
                    _gate_recheck_attempts = 0
                    
                    # PREDICT DIRECTION & SPEED using shared production helper logic.
                    analytics = compute_alignment_decision(
                        world=None,
                        step="ALIGN_BRICK",
                        process_rules=self.world.process_rules,
                        learned_rules=None,
                        duration_s=0.05,
                        visible=True,
                        x_axis=off_now,
                        angle=0.0,
                        dist=dist_now,
                    )
                    prod_cmd = analytics.get("cmd")  # 'f', 'b', 'l', 'r', or None
                    prod_worst_metric = analytics.get("worst_metric")  # 'dist', 'xAxis_offset_abs', or None
                    
                    # Map production command to our direction
                    if prod_cmd in ('f', 'b', 'l', 'r'):
                        align_dir = prod_cmd
                        correction_type = "distance" if prod_cmd in ('f', 'b') else "x_axis"
                    else:
                        # Fallback (shouldn't happen if gates work)
                        align_dir = 'l' if x_err > 0 else 'r'
                        correction_type = "x_axis"
                    
                    self.keepalive_dir = align_dir
                    
                    # Store correction type for result display
                    self._current_correction_type = correction_type
                    
                    # SIMPLIFIED CONSERVATIVE APPROACH FOR ACCURACY
                    # Enhanced print: trial/act numbers, target values, and formatted metrics
                    trial_num = int(self.current_trial)
                    act_num = int(self.trial_adjust_count) + 1
                    if correction_type == "distance":
                        # Calculate granular speed score based on distance error
                        # This uses the single source of truth: align_brick_dist_error_speed_score
                        dist_score_float = align_brick_dist_error_speed_score(dist_err)
                        dist_score = int(round(dist_score_float))  # Round to nearest int for command
                        event = self.send_coarse_command(
                            align_dir,
                            score=dist_score,
                            reason="distance_alignment",
                            extra={
                                "worst_metric": prod_worst_metric,
                                "dist_error_mm": dist_err,
                                "dist_score": dist_score,
                                "dist_score_float": dist_score_float,
                            },
                        )
                        # Print act with motor details
                        motor_details = ""
                        if isinstance(event, dict):
                            parts = []
                            if event.get("pwm") is not None:
                                parts.append(f"pwm={int(event['pwm'])}")
                            if event.get("power") is not None:
                                parts.append(f"pwr={float(event['power']):.3f}")
                            if event.get("duration_ms") is not None:
                                parts.append(f"t={int(event['duration_ms'])}ms")
                            if parts:
                                motor_details = f" {COLOR_GRAY}({', '.join(parts)}){COLOR_RESET}"
                        # Format distance error with color and delta
                        if prev_dist_err is not None:
                            delta_dist = float(prev_dist_err) - float(dist_err)
                            if abs(delta_dist) < 0.05:
                                # Negligible change, use yellow
                                dist_err_color = COLOR_YELLOW
                                dist_err_text = f"{dist_err:+.2f}mm"
                                change_text = f"(~unchanged)"
                            elif delta_dist > 0:
                                # Progress - getting closer
                                dist_err_color = COLOR_GREEN
                                dist_err_text = f"{dist_err:+.2f}mm"
                                change_text = f"({delta_dist:+.2f}mm better)"
                            else:
                                # Worsening - getting farther
                                dist_err_color = COLOR_RED
                                dist_err_text = f"{dist_err:+.2f}mm"
                                change_text = f"({delta_dist:+.2f}mm worse)"
                        else:
                            # First act of trial
                            dist_err_color = COLOR_YELLOW
                            dist_err_text = f"{dist_err:+.2f}mm"
                            change_text = "(initial)"
                        
                        print(
                            f"{COLOR_CYAN}[T{trial_num}.{act_num} ALIGN]{COLOR_RESET} "
                            f"dist_err={dist_err_color}{dist_err_text}{COLOR_RESET} {change_text} "
                            f"(target {DIST_TARGET_MM:.2f}mm ±{DIST_TOL_MM:.2f}mm) → "
                            f"{COLOR_WHITE}{align_dir} {dist_score}%{COLOR_RESET}{motor_details}"
                        )
                        # Update previous distance error after taking this action
                        prev_dist_err = float(dist_err)
                    else:
                        # One-shot x-axis correction: use score sized from current x-gap.
                        x_turn_score = int(align_brick_x_axis_one_shot_score(x_err))
                        event = self.send_coarse_command(
                            align_dir,
                            score=x_turn_score,
                            reason="x_axis_alignment",
                            extra={
                                "worst_metric": prod_worst_metric,
                                "x_error_mm": x_err,
                                "x_turn_score": int(x_turn_score),
                            },
                        )
                        # Print act with motor details
                        motor_details = ""
                        if isinstance(event, dict):
                            parts = []
                            if event.get("pwm") is not None:
                                parts.append(f"pwm={int(event['pwm'])}")
                            if event.get("power") is not None:
                                parts.append(f"pwr={float(event['power']):.3f}")
                            if event.get("duration_ms") is not None:
                                parts.append(f"t={int(event['duration_ms'])}ms")
                            if parts:
                                motor_details = f" {COLOR_GRAY}({', '.join(parts)}){COLOR_RESET}"
                        # Format x-axis error with color and delta
                        overshoot_detected = False
                        if prev_align_x_err is not None:
                            # Check for overshoot (sign flip beyond deadzone)
                            if (float(prev_align_x_err) * float(x_err)) < 0.0:
                                if (abs(float(prev_align_x_err)) > float(OVERSHOOT_SIGN_FLIP_DEADZONE_MM) and
                                    abs(float(x_err)) > float(OVERSHOOT_SIGN_FLIP_DEADZONE_MM)):
                                    overshoot_detected = True
                            
                            if overshoot_detected:
                                # Overshot - crossed the target
                                x_err_color = COLOR_RED
                                x_err_text = f"{x_err:+.2f}mm"
                                change_text = f"(overshot - sailed past {X_TARGET_MM:+.2f}mm target)"
                            else:
                                delta_x = abs(float(prev_align_x_err)) - abs(float(x_err))
                                if abs(delta_x) < 0.05:
                                    # Negligible change
                                    x_err_color = COLOR_YELLOW
                                    x_err_text = f"{x_err:+.2f}mm"
                                    change_text = f"(~unchanged)"
                                elif delta_x > 0:
                                    # Progress - getting closer
                                    x_err_color = COLOR_GREEN
                                    x_err_text = f"{x_err:+.2f}mm"
                                    change_text = f"({delta_x:+.2f}mm better)"
                                else:
                                    # Worsening - getting farther
                                    x_err_color = COLOR_RED
                                    x_err_text = f"{x_err:+.2f}mm"
                                    change_text = f"({delta_x:+.2f}mm worse)"
                        else:
                            # First act of trial
                            x_err_color = COLOR_YELLOW
                            x_err_text = f"{x_err:+.2f}mm"
                            change_text = "(initial)"
                        
                        print(
                            f"{COLOR_CYAN}[T{trial_num}.{act_num} ALIGN]{COLOR_RESET} "
                            f"x_err={x_err_color}{x_err_text}{COLOR_RESET} {change_text} "
                            f"(target {X_TARGET_MM:+.2f}mm ±{X_TOL_MM:.2f}mm) → "
                            f"{COLOR_WHITE}{align_dir} {int(x_turn_score)}%{COLOR_RESET}{motor_details}"
                        )
                        # Update previous x-axis error after taking this action
                        prev_align_x_err = float(x_err)
                    
                    # Track successful commands
                    if isinstance(event, dict):
                        self.trial_adjust_count = int(self.trial_adjust_count + 1)
                        pending_eval_after_act = True
                        pending_pre_act_x_err = float(x_err)
                        pending_pre_act_abs_x_err = float(abs_error_from_target)
                    
                    # Freeze robot for gate observation phase
                    self.robot.stop()
                
                # Short sleep to not hammer the CPU/Serial
                time.sleep(0.04)

            if not trial_success and trial_reason == "unknown":
                if not self.running:
                    trial_reason = "stopped"
                elif self.abort_run_early:
                    trial_reason = "early_stop_unrecovered_vision"
                else:
                    trial_reason = "loop_exit"

            trial_total_time = None
            if bool(trial_success):
                trial_total_time = float(time.time() - trial_start_t)

            # Learn from failures only during experiment trials.
            if int(i) <= int(EXPERIMENT_TRIALS):
                self.adapt_learning_after_trial(success=trial_success, reason=trial_reason)
            elif not trial_success:
                print(
                    f"[LEARN] Trial {int(i)} failure in exploit-only mode "
                    f"(no exploratory adaptation; locked after trial {int(EXPERIMENT_TRIALS)})."
                )

            # Keep best policy for exploit/champion replay selection.
            self._consider_best_policy(
                trial_idx=i,
                trial_success=trial_success,
                trial_reason=trial_reason,
                total_time=trial_total_time,
                policy_before_trial=policy_before_trial,
            )

            # Update and redraw speed-score decision curve only during experiment trials.
            if int(i) <= int(EXPERIMENT_TRIALS):
                self.update_speed_curve_after_trial(i, success=trial_success, reason=trial_reason)
            else:
                print(
                    f"[LEARN] Trial {int(i)} in exploit-only mode "
                    f"(speed-curve frozen after trial {int(EXPERIMENT_TRIALS)})."
                )

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
        if isinstance(self.results_log, dict):
            self.results_log["recovery_summary"] = dict(recovery_summary) if isinstance(recovery_summary, dict) else None
            self.results_log["early_stop"] = bool(self.abort_run_early)
            self.results_log["early_stop_reason"] = (
                "consecutive_unrecovered_vision_losses" if self.abort_run_early else None
            )
            grade_summary = self.print_grade_summary()
            if isinstance(grade_summary, dict):
                self.results_log["grade_summary"] = grade_summary
            self.results_log["status"] = "finished"
            self.results_log["finished_at"] = float(time.time())
            self._write_results()
            try:
                profile = build_align_profile_from_results_payload(self.results_log)
                outcome = promote_align_profile_if_better(profile, Path(__file__).resolve().parent)
                if bool(outcome.get("promoted")):
                    print(
                        "[ALIGN_PROFILE] Promoted new champion profile "
                        f"(reason={outcome.get('reason')}, score={outcome.get('candidate_score'):.2f})."
                    )
                else:
                    print(
                        "[ALIGN_PROFILE] Candidate profile saved but champion retained "
                        f"(reason={outcome.get('reason')}, "
                        f"candidate={outcome.get('candidate_score'):.2f}, "
                        f"champion={outcome.get('champion_score')})."
                    )
            except Exception as exc:
                print(f"[ALIGN_PROFILE] Failed to save calibration profile: {exc}")
        else:
            self.print_grade_summary()
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
            self._write_results()
        self.robot.close()
        self.vision.close()


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if bool(default) else "[y/N]"
    while True:
        try:
            raw = input(f"{prompt} {suffix}: ").strip().lower()
        except EOFError:
            return bool(default)
        if raw == "":
            return bool(default)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer y/yes or n/no.")


def _run_checked(cmd: list[str], label: str) -> bool:
    try:
        subprocess.run(cmd, check=True)
        print(f"[ALIGN][MEDIA] {label}: done")
        return True
    except Exception as exc:
        print(f"[ALIGN][MEDIA] {label}: failed ({exc})")
        return False


def _export_align_png(workspace_root: Path) -> bool:
    viz_script = workspace_root / "calibrate_align_visualize.py"
    input_json = workspace_root / DATA_FILE
    output_png = workspace_root / "align_trials_final.png"
    if not viz_script.exists():
        print(f"[ALIGN][MEDIA] Missing visualizer script: {viz_script}")
        return False
    if not input_json.exists():
        print(f"[ALIGN][MEDIA] Missing event log: {input_json}")
        return False
    cmd = [
        sys.executable,
        str(viz_script),
        "--input",
        str(input_json),
        "--static-only",
        "--save-png",
        str(output_png),
    ]
    return _run_checked(cmd, f"PNG export ({output_png.name})")


def _export_align_trial_gifs(workspace_root: Path) -> tuple[bool, Path]:
    viz_script = workspace_root / "calibrate_align_visualize.py"
    input_json = workspace_root / DATA_FILE
    output_dir = workspace_root / "align_trials_by_trial"
    if not viz_script.exists():
        print(f"[ALIGN][MEDIA] Missing visualizer script: {viz_script}")
        return False, output_dir
    if not input_json.exists():
        print(f"[ALIGN][MEDIA] Missing event log: {input_json}")
        return False, output_dir
    cmd = [
        sys.executable,
        str(viz_script),
        "--input",
        str(input_json),
        "--per-trial-gifs",
        "--save-gif-dir",
        str(output_dir),
        "--no-save-png",
    ]
    return _run_checked(cmd, f"Per-trial GIF export ({output_dir.name})"), output_dir


def _stitch_trial_gifs_to_mp4(gif_dir: Path) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[ALIGN][MEDIA] ffmpeg not found on PATH; skipping MP4 stitch.")
        return False

    gifs = sorted(gif_dir.glob("align_trial_*.gif"))
    if not gifs:
        print(f"[ALIGN][MEDIA] No trial GIFs found in {gif_dir}")
        return False

    output_mp4 = gif_dir / "align_trials_by_trial_stitched.mp4"
    with tempfile.TemporaryDirectory(prefix="align_stitch_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        concat_file = tmp_dir / "segments.txt"
        lines = []

        for idx, gif in enumerate(gifs, start=1):
            seg = tmp_dir / f"seg_{idx:03d}.mp4"
            ok = _run_checked(
                [
                    ffmpeg_bin,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(gif),
                    "-vf",
                    "fps=30,format=yuv420p",
                    "-an",
                    str(seg),
                ],
                f"GIF→segment {gif.name}",
            )
            if not ok:
                return False
            lines.append(f"file '{str(seg)}'\n")

        concat_file.write_text("".join(lines))

        return _run_checked(
            [
                ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_mp4),
            ],
            f"Stitched MP4 export ({output_mp4.name})",
        )


def maybe_prompt_post_run_media_exports(all_trials_completed: bool) -> None:
    if not bool(all_trials_completed):
        return

    print("\n[ALIGN][MEDIA] Trials complete. Optional exports:")
    if _ask_yes_no("Generate all media outputs (PNG + GIFs + MP4)?", default=False):
        want_png = True
        want_gifs = True
        want_mp4 = True
    else:
        want_png = _ask_yes_no("1) Generate PNG of all trials?", default=False)
        want_gifs = _ask_yes_no("2) Generate per-trial GIFs?", default=False)
        want_mp4 = _ask_yes_no("3) Stitch trial GIFs into MP4?", default=False)

    if not any([want_png, want_gifs, want_mp4]):
        print("[ALIGN][MEDIA] Skipping media exports.")
        return

    workspace_root = Path(__file__).resolve().parent
    gif_dir = workspace_root / "align_trials_by_trial"
    gifs_ready = False

    if bool(want_png):
        _export_align_png(workspace_root)

    if bool(want_gifs):
        gifs_ready, gif_dir = _export_align_trial_gifs(workspace_root)

    if bool(want_mp4):
        if not gifs_ready:
            print("[ALIGN][MEDIA] MP4 requires trial GIFs; generating GIFs now.")
            gifs_ready, gif_dir = _export_align_trial_gifs(workspace_root)
        if gifs_ready:
            _stitch_trial_gifs_to_mp4(gif_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALIGN_BRICK calibration - learn optimal speeds")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual testing mode: observe vision and suggest acts without moving robot"
    )
    args = parser.parse_args()
    
    # Print startup info
    print(f"\n{COLOR_CYAN}[ALIGN_BRICK] Calibration starting...{COLOR_RESET}")
    print(f"  10 trials with accuracy-first gate validation")
    print(f"  Target: X_offset={X_TARGET_MM}mm (±{X_TOL_MM}mm), Dist={DIST_TARGET_MM}mm (±{DIST_TOL_MM}mm)\n")
    
    calibrator = AlignCalibrator()
    try:
        if args.manual:
            calibrator.run_manual_test_mode()
        else:
            calibrator.run_experiment()
            all_trials_completed = (
                (not bool(calibrator.abort_run_early))
                and bool(calibrator.running)
                and int(getattr(calibrator, "current_trial", 0) or 0) >= int(NUM_TRIALS)
            )
            maybe_prompt_post_run_media_exports(all_trials_completed=all_trials_completed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        calibrator.close()
