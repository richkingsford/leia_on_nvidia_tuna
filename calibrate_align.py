#!/usr/bin/env python3
import json
import random
import time
import statistics
import threading
import cv2
import os
from pathlib import Path
from collections import deque

from helper_robot_control import Robot
from helper_vision_aruco import ArucoBrickVision
import telemetry_robot as telemetry_robot_module
from telemetry_robot import WorldModel, draw_telemetry_overlay, StepState
from telemetry_process import compute_stream_gate_summary, send_robot_command, send_robot_command_pwm
from helper_stream_server import StreamServer, format_stream_url
from helper_gate_utils import load_process_steps, gate_satisfied

# Experiment Config
NUM_TRIALS = 10
DATA_FILE = "world_model_align.json"
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

# Experiment-local X-axis control tuning.
# If left/right feel backwards, flip X_AXIS_SIGN to -1.0 (no other files need changes).
X_AXIS_SIGN = 1.0
X_TARGET_MM = float(X_GATE_TARGET_MM)
X_TOL_MM = float(X_GATE_TOL_MM)
LOG_FLUSH_EVERY_EVENTS = 50

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
LEARN_SPEED_START = 15    # Starting coarse speed (will increase per trial)
LEARN_SPEED_STEP = 2      # Always increase every trial
LEARN_SPEED_BONUS = 3     # Extra increase if trial beats recent average
OBSERVE_SLEEP_S = 0.02    # Small sleep while sampling to keep CPU down
STALL_WINDOW = 3          # Detect stalling after ~2 acts (3 samples)
STALL_BUMP_SCORE = 2      # Add a small speed bump when stalling (coarse mode)
STALL_BUMP_PCT = 5.0      # Extra bump for precision mode when stalling

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
        self.learned_speed_score = LEARN_SPEED_START
        self.keepalive_dir = "l"
        self.precision_bump_pct = PRECISION_BUMP_PCT
        self.precision_stall_acts = 0
        
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

    def _get_frame(self):
        with self.lock: return self.current_frame

    def _x_err_mm(self, offset_x):
        try:
            return (float(offset_x) * float(X_AXIS_SIGN)) - float(X_TARGET_MM)
        except (TypeError, ValueError):
            return 0.0

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

    def log_act(self, cmd, mode, score=None, bump_pct=None, reason=None, extra=None):
        offset_x = self._last_obs_offset_x
        dist = self._last_obs_dist
        err_mm = self._x_err_mm(offset_x)
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
            "cmd": cmd,
            "hw_cmd": hw_cmd,
            "mode": mode,
            "score": score,
            "precision_bump_pct": bump_pct,
            "reason": reason,
        }
        if isinstance(extra, dict):
            event.update(extra)
        self._append_event(event)
        self.act_idx += 1
        return event

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
        
        # Draw Overlay
        gate_summary, analytics = compute_stream_gate_summary(self.world, "ALIGN_BRICK", active=True)
        
        # In-Line Enhancement for Livestream
        x_err = self._x_err_mm(offset_x)
        status = "OK" if abs(x_err) <= X_TOL_MM else "PENDING"
        extra = [
            f"X_OFF: {offset_x:+.1f}mm",
            f"X_ERR: {x_err:+.1f}mm",
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

    def get_stable_pose(self, samples=3, timeout_s=2.0, keepalive_dir=None, keepalive_bump_pct=None):
        """
        Reads 'samples' unique frames and returns averaged (offset_x, angle).
        Ensures consistency between samples.
        """
        poses = []
        start_t = time.time()
        while len(poses) < samples and (time.time() - start_t < timeout_s) and self.running:
            found, angle, dist, offset_x, conf, cam_h, above, below = self.vision.read()
            # Update world model for the livestream
            self.update_world(found, angle, dist, offset_x, conf, cam_h, above, below, self.vision.current_frame)
            # Keep the robot moving slowly while sampling for less blur.
            if keepalive_dir:
                self.send_precision_command(keepalive_dir, bump_pct=keepalive_bump_pct, reason="sample_keepalive")
            
            if found:
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
                if not poses:
                    poses.append({"offset_x": off_val, "dist": dist_val, "angle": angle, "confidence": conf})
                else:
                    last = poses[-1]
                    if abs(off_val - last.get("offset_x", off_val)) > 0.1 or abs(dist_val - last.get("dist", dist_val)) > 0.5:
                        poses.append({"offset_x": off_val, "dist": dist_val, "angle": angle, "confidence": conf})
                if len(poses) < samples:
                    time.sleep(OBSERVE_SLEEP_S) 
            else: 
                if len(poses) < samples:
                    time.sleep(OBSERVE_SLEEP_S)
        
        if not poses or not self.running: return None

        # Average results
        avg_offset = statistics.mean([p["offset_x"] for p in poses])
        avg_dist = statistics.mean([p["dist"] for p in poses])
        
        return {"offset_x": avg_offset, "dist": avg_dist}

    def run_reset_phase(self):
        """
        Deliberate, vision-driven reset:
          - Pick a random desired abs(x_err) away from the ALIGN gate target.
          - Turn in a direction and validate via vision after each act.
          - Log each reset act with before/after observations (offset + dist).
        """
        pose_start = self.get_stable_pose(samples=1, timeout_s=2.0)
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
        while self.running and reset_acts < int(RESET_MAX_ACTS):
            # Observe BEFORE deciding the next act (no keepalive so we don't double-command).
            pre = self.get_stable_pose(samples=1, timeout_s=RESET_OBSERVE_TIMEOUT_S)
            if not pre:
                print("  [RESET] Lost vision, attempting recovery...")
                self.current_phase = "reset_recovery"
                if not self.recovery_scan():
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

            post = self.get_stable_pose(samples=1, timeout_s=RESET_OBSERVE_TIMEOUT_S)
            if not post:
                if isinstance(event, dict):
                    event.update(
                        {
                            "obs_after_found": False,
                            "result": "vision_lost_after_act",
                        }
                    )
                print("  [RESET] Vision lost after act, attempting recovery...")
                self.current_phase = "reset_recovery"
                if not self.recovery_scan():
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
            }
        )

        return last_pose

    def recovery_scan(self, max_attempts=15):
        """Slowly scan nearby area to find the brick again."""
        print("[RECOVERY] Sight lost. Scanning nearby area...")
        # Wiggle pattern: alternating directions, keep moving while sampling.
        for i in range(1, max_attempts + 1):
            scan_dir = 'l' if i % 2 == 0 else 'r'
            self.keepalive_dir = scan_dir
            self.send_precision_command(scan_dir, reason="recovery_scan")
            if self.get_stable_pose(samples=1, timeout_s=0.8, keepalive_dir=scan_dir): # Quick check
                print(f"[RECOVERY] Found brick again on attempt {i}!")
                return True
        return False

    def update_learning_speed(self, total_time, avg_recent=None):
        delta = LEARN_SPEED_STEP
        if avg_recent is not None and total_time < avg_recent:
            delta += LEARN_SPEED_BONUS
        self.learned_speed_score = min(COARSE_SCORE, self.learned_speed_score + delta)
        print(f"[LEARN] Next trial coarse speed: {self.learned_speed_score}%")

    def run_experiment(self):
        # WIPE DATA FILE ON START
        if Path(DATA_FILE).exists():
            print(f"[ALIGN] Wiping {DATA_FILE} for fresh learning...")
            Path(DATA_FILE).unlink()
        self.data_log = []
        self._unsaved_events = 0
        self._append_event({
            "type": "meta",
            "run_id": self.run_id,
            "timestamp": time.time(),
            "num_trials": NUM_TRIALS,
            "x_gate_metric": X_GATE_METRIC,
            "x_gate_target_mm": X_GATE_TARGET_MM,
            "x_gate_tol_mm": X_GATE_TOL_MM,
            "x_axis_sign": X_AXIS_SIGN,
            "x_target_mm": X_TARGET_MM,
            "x_tol_mm": X_TOL_MM,
            "precision_score_base": PRECISION_SCORE_BASE,
            "precision_bump_pct": PRECISION_BUMP_PCT,
            "coarse_score_cap": COARSE_SCORE,
        })

        print(f"\n--- ROGUE ALIGNMENT EXPERIMENT: CONSTANT MOTION ---")
        print(f"X Gate ({X_GATE_METRIC}): {X_GATE_TARGET_MM:+.2f}mm (tol {X_GATE_TOL_MM:.2f}mm)")
        print(f"Using X target: {X_TARGET_MM:+.2f}mm (tol {X_TOL_MM:.2f}mm)")
        
        for i in range(1, NUM_TRIALS + 1):
            if not self.running: break
            self.current_trial = i
            self.success_frames = 0
            self.offset_history.clear()
            self.precision_bump_pct = PRECISION_BUMP_PCT
            self.precision_stall_acts = 0
            print(f"\nTRIAL {i}/{NUM_TRIALS}")
            
            # 1. SETUP: Deliberate vision-driven reset away from the X gate target
            self.current_phase = "reset"
            pose_reset = self.run_reset_phase()
            if not pose_reset:
                print("  [ERROR] Cannot find brick for reset setup.")
                continue

            # No robot.stop() here - we want to transition immediately into ALIGN
            
            # 2. CONTINUOUS ALIGNMENT
            self.current_phase = "align"
            trial_start_t = time.time()
            trial_start_offset = None
            last_cmd = None
            last_sig = None
            
            while self.running:
                # Continuous observation (1 sample for speed)
                # Increased timeout to 1.0s to handle blur during 50% movement
                pose = self.get_stable_pose(samples=1, timeout_s=1.0, keepalive_dir=self.keepalive_dir)
                if not pose:
                    print(f"  [LOST] Vision lost, attempting recovery...")
                    self.current_phase = "recovery"
                    if not self.recovery_scan(): break 
                    self.current_phase = "align"
                    pose = self.get_stable_pose(samples=1, keepalive_dir=self.keepalive_dir)
                    if not pose: break
                    trial_start_t = time.time() # Reset timer if we lost track
                
                off_now = pose["offset_x"]
                self.offset_history.append(off_now)
                if trial_start_offset is None: trial_start_offset = off_now
                
                x_err = self._x_err_mm(off_now)
                abs_error_from_target = abs(x_err)
                
                # Check Gate Success: Requirement = 3 consecutive frames
                # We use the official gate_satisfied check for parity with the overlay
                official_satisfied = gate_satisfied(self.world.brick, ALIGN_GATES) if self.world.brick else False
                exp_satisfied = abs_error_from_target <= float(X_TOL_MM)
                
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
                        # Keep moving slowly while we transition to the next trial.
                        self.send_precision_command(self.keepalive_dir, reason="between_trials")
                        break
                else:
                    self.success_frames = 0
                
                # PREDICT DIRECTION & SPEED
                # Very simple X-axis correction: brick right => turn right.
                align_dir = 'r' if x_err > 0 else 'l'
                self.keepalive_dir = align_dir
                
                # Two-Stage Speed: learned coarse score if far, precision bump if near.
                near_threshold = float(X_TOL_MM) + 4.0
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
                            print(f"[STALL] No progress -> precision bump +{self.precision_bump_pct:.0f}%")
                    else:
                        if self.precision_stall_acts or self.precision_bump_pct != PRECISION_BUMP_PCT:
                            self.precision_stall_acts = 0
                            self.precision_bump_pct = PRECISION_BUMP_PCT

                # HEARTBEAT: Always send command to feed the 100ms watchdog
                if abs_error_from_target > near_threshold:
                    score = min(self.learned_speed_score, COARSE_SCORE)
                    if stall:
                        score = min(score + STALL_BUMP_SCORE, COARSE_SCORE)
                    self.send_coarse_command(
                        align_dir,
                        score=score,
                        reason="x_axis_far",
                        extra={"stall": stall, "near_threshold_mm": near_threshold, "official_gate": official_satisfied, "exp_gate": exp_satisfied},
                    )
                    sig = ("score", score)
                    if align_dir != last_cmd or sig != last_sig:
                        print(f"  [ADJUST] x={off_now:+.1f}mm err={x_err:+.1f} -> {align_dir}@{score}%")
                        last_cmd = align_dir
                        last_sig = sig
                else:
                    bump_pct = self.precision_bump_pct + (STALL_BUMP_PCT if stall else 0.0)
                    self.send_precision_command(
                        align_dir,
                        bump_pct=bump_pct,
                        reason="x_axis_near",
                        extra={"stall": stall, "near_threshold_mm": near_threshold, "official_gate": official_satisfied, "exp_gate": exp_satisfied},
                    )
                    sig = ("precision", bump_pct)
                    if align_dir != last_cmd or sig != last_sig:
                        print(f"  [ADJUST] x={off_now:+.1f}mm err={x_err:+.1f} -> {align_dir}@p+{bump_pct:.0f}%")
                        last_cmd = align_dir
                        last_sig = sig
                
                # Short sleep to not hammer the CPU/Serial
                time.sleep(0.04)

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
        self.robot.close()
        self.vision.close()
        self.server.stop()

if __name__ == "__main__":
    calibrator = AlignCalibrator()
    try: calibrator.run_experiment()
    except KeyboardInterrupt: print("\nInterrupted.")
    finally: calibrator.close()
