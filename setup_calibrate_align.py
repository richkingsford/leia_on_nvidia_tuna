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
from telemetry_robot import WorldModel, draw_telemetry_overlay, manual_speed_for_cmd, StepState
from telemetry_process import compute_stream_gate_summary
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
SUCCESS_TOL_MM = ALIGN_GATES.get("xAxis_offset", {}).get("tol", 1.5)
SUCCESS_TARGET_MM = ALIGN_GATES.get("xAxis_offset", {}).get("target", -2.24)

# Advanced Control Tunings
BACKLASH_S = 0.08         
COARSE_SCORE = 50         # Restored to 50% for speed
PRECISION_SCORE = 1       # Use the 0.290 power you set
SETTLE_TIME_S = 0.0       # Pause removed for constant motion
MIN_K = 4.0               
NOISE_FLOOR_MM = 0.2      
POWER_INCREMENT = 10      

class AlignCalibrator:
    def __init__(self):
        self.robot = Robot()
        self.vision = ArucoBrickVision(debug=False)
        self.world = WorldModel()
        self.world.step_state = StepState.ALIGN_BRICK
        self.data_log = []
        
        # Histories for smarter learning and metrics
        self.error_history = deque(maxlen=10)
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

    def _get_frame(self):
        with self.lock: return self.current_frame

    def update_world(self, found, angle, dist, offset_x, conf, cam_h, above, below, frame):
        self.world.update_vision(found, dist, angle, conf, offset_x, cam_h, above, below)
        
        # Draw Overlay
        gate_summary, analytics = compute_stream_gate_summary(self.world, "ALIGN_BRICK", active=True)
        
        # In-Line Enhancement for Livestream
        status = "OK" if abs(offset_x) <= SUCCESS_TOL_MM else "PENDING"
        extra = [
            f"X_OFF: {offset_x:+.1f}mm",
            f"GATE: {status}"
        ]
        
        display = frame.copy()
        draw_telemetry_overlay(
            display, 
            self.world, 
            gate_summary=gate_summary,
            extra_messages=extra,
            highlight_metric="xAxis_offset"
        )
        
        with self.lock: self.current_frame = display

    def get_stable_pose(self, samples=3, timeout_s=2.0):
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
            
            if found:
                # Ensure we have a unique frame (or significantly different)
                if not poses or abs(offset_x - poses[-1]["offset_x"]) > 0.1:
                    poses.append({"offset_x": offset_x, "angle": angle})
                if len(poses) < samples:
                    time.sleep(0.05) 
            else: 
                if len(poses) < samples:
                    time.sleep(0.05)
        
        if not poses or not self.running: return None

        # Average results
        avg_offset = statistics.mean([p["offset_x"] for p in poses])
        
        return {"offset_x": avg_offset}

    def recovery_scan(self, max_attempts=15):
        """Slowly scan nearby area to find the brick again."""
        print("[RECOVERY] Sight lost. Scanning nearby area...")
        # Wiggle pattern: small pulses, alternating directions with increasing duration
        for i in range(1, max_attempts + 1):
            scan_dir = 'l' if i % 2 == 0 else 'r'
            duration = 0.1 + (i * 0.05)
            speed = manual_speed_for_cmd(scan_dir, PRECISION_SCORE)
            
            self.robot.send_command(scan_dir, speed)
            time.sleep(duration)
            # No robot.stop() - keep moving during experiment as much as possible
            # but for recovery we might need to pause to check.
            self.robot.stop()
            time.sleep(0.8)
            
            if self.get_stable_pose(samples=1): # Quick check
                print(f"[RECOVERY] Found brick again on attempt {i}!")
                return True
        return False

    def run_experiment(self):
        # WIPE DATA FILE ON START
        if Path(DATA_FILE).exists():
            print(f"[ALIGN] Wiping {DATA_FILE} for fresh learning...")
            Path(DATA_FILE).unlink()
        self.data_log = []

        print(f"\n--- ROGUE ALIGNMENT EXPERIMENT: CONSTANT MOTION ---")
        print(f"Gate: {SUCCESS_TARGET_MM}mm (tol {SUCCESS_TOL_MM}mm)")
        
        for i in range(1, NUM_TRIALS + 1):
            if not self.running: break
            print(f"\nTRIAL {i}/{NUM_TRIALS}")
            
            # 1. SETUP: Randomized offset - Move to a specific random target
            pose_start = self.get_stable_pose(samples=1, timeout_s=2.0)
            if not pose_start: 
                print("  [ERROR] Cannot find brick for reset setup.")
                continue
            
            off_start = pose_start["offset_x"]
            target_reset_off = random.uniform(15.0, 35.0) * random.choice([-1, 1])
            reset_dir = 'r' if target_reset_off > off_start else 'l'
            reset_speed = manual_speed_for_cmd(reset_dir, 1) # Slowest effective turn (1%)
            
            print(f"  [RESET] Starting at {off_start:+.1f}mm. Moving to {target_reset_off:+.1f}mm target...")
            
            while self.running:
                pose = self.get_stable_pose(samples=1, timeout_s=1.0)
                if not pose: break
                off_now = pose["offset_x"]
                
                # Check if we've reached or passed the target
                if (reset_dir == 'r' and off_now >= target_reset_off) or \
                   (reset_dir == 'l' and off_now <= target_reset_off):
                    break
                
                self.robot.send_command(reset_dir, reset_speed)
                time.sleep(0.04)

            # No robot.stop() here - we want to transition immediately into ALIGN
            
            # 2. CONTINUOUS ALIGNMENT
            trial_start_t = time.time()
            trial_start_offset = None
            last_cmd = None
            last_score = None
            
            while self.running:
                # Continuous observation (1 sample for speed)
                # Increased timeout to 1.0s to handle blur during 50% movement
                pose = self.get_stable_pose(samples=1, timeout_s=1.0)
                if not pose:
                    print(f"  [LOST] Vision lost, stopping...")
                    self.robot.stop()
                    if not self.recovery_scan(): break 
                    pose = self.get_stable_pose(samples=1)
                    if not pose: break
                    trial_start_t = time.time() # Reset timer if we lost track
                
                off_now = pose["offset_x"]
                if trial_start_offset is None: trial_start_offset = off_now
                
                abs_error_from_target = abs(off_now - SUCCESS_TARGET_MM)
                
                # Check Gate Success: Requirement = 3 consecutive frames
                # We use the official gate_satisfied check for parity with the overlay
                is_satisfied = gate_satisfied(self.world.brick, ALIGN_GATES) if self.world.brick else False
                
                if is_satisfied:
                    self.success_frames += 1
                    if self.success_frames >= self.REQUIRED_SUCCESS_FRAMES:
                        self.robot.stop()
                        total_time = time.time() - trial_start_t
                        self.print_trial_summary(i, off_now, total_time)
                        self.log_trial(i, trial_start_offset, off_now, total_time)
                        break
                else:
                    self.success_frames = 0
                
                # PREDICT DIRECTION & SPEED
                # If brick is RIGHT of target, we want robot to turn RIGHT to move it LEFT
                align_dir = 'r' if off_now > SUCCESS_TARGET_MM else 'l'
                
                # Two-Stage Speed: 50% if far (gap > 4mm or angle > 10deg)
                near_threshold = (ALIGN_GATES.get("xAxis_offset", {}).get("tol", 1.5)) + 4.0
                if abs_error_from_target > near_threshold:
                    score = COARSE_SCORE
                else:
                    score = 1 # 1% speed
                
                speed = manual_speed_for_cmd(align_dir, score)
                
                # HEARTBEAT: Always send command to feed the 100ms watchdog
                self.robot.send_command(align_dir, speed)
                if align_dir != last_cmd or score != last_score:
                    print(f"  [ADJUST] {off_now:+.1f}mm -> {align_dir}@{score}%")
                    last_cmd = align_dir
                    last_score = score
                
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
        self.data_log.append({
            "trial": i, "timestamp": time.time(), "initial_offset": initial,
            "final_offset": final, "seconds_to_success": total_time,
            "K_L": self.K_L, "K_R": self.K_R
        })
        with open(DATA_FILE, 'w') as f: json.dump(self.data_log, f, indent=2)

    def close(self):
        self.running = False
        self.robot.close()
        self.vision.close()
        self.server.stop()

if __name__ == "__main__":
    calibrator = AlignCalibrator()
    try: calibrator.run_experiment()
    except KeyboardInterrupt: print("\nInterrupted.")
    finally: calibrator.close()
