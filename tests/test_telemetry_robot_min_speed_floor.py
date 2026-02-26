import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class TestTelemetryRobotMinSpeedFloor(unittest.TestCase):
    def test_speed_power_pwm_for_cmd_uses_exact_drive_anchor_when_present(self):
        prev_drive = telemetry_robot.SCORE_POWER_PWM_DRIVE
        prev_dur = telemetry_robot.SPEED_SCORE_DURATION_MS
        try:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = {
                telemetry_robot.SPEED_SCORE_MIN: {"pwm": 93, "power": telemetry_robot.pwm_to_power(93), "duration_ms": 250},
                20: {"pwm": 163, "power": telemetry_robot.pwm_to_power(163), "duration_ms": 275},
                telemetry_robot.SPEED_SCORE_MAX: {"pwm": 255, "power": telemetry_robot.pwm_to_power(255), "duration_ms": 300},
            }
            telemetry_robot.SPEED_SCORE_DURATION_MS = {
                telemetry_robot.SPEED_SCORE_MIN: 250,
                20: 275,
                telemetry_robot.SPEED_SCORE_MAX: 300,
            }
            power, pwm, score_used, duration_ms = telemetry_robot.speed_power_pwm_for_cmd("f", 20)
            self.assertEqual(int(score_used), 20)
            self.assertEqual(int(pwm), 163)
            self.assertEqual(int(duration_ms), 275)
            self.assertAlmostEqual(float(power), float(telemetry_robot.pwm_to_power(163)), places=6)
        finally:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = prev_drive
            telemetry_robot.SPEED_SCORE_DURATION_MS = prev_dur

    def test_speed_power_pwm_for_cmd_clamps_turn_below_baseline(self):
        baseline = int(telemetry_robot.baseline_pwm_floor_for_cmd("l") or 0)
        self.assertGreater(baseline, 0)

        entry = telemetry_robot.SCORE_POWER_PWM_TURN.get(telemetry_robot.SPEED_SCORE_MIN)
        self.assertIsInstance(entry, dict)
        prev = dict(entry)
        try:
            entry["pwm"] = max(0, baseline - 20)
            _, pwm, _, _ = telemetry_robot.speed_power_pwm_for_cmd("l", telemetry_robot.SPEED_SCORE_MIN)
            self.assertGreaterEqual(int(pwm), baseline)
        finally:
            entry.clear()
            entry.update(prev)

    def test_speed_power_pwm_for_cmd_clamps_drive_below_baseline(self):
        baseline = int(telemetry_robot.baseline_pwm_floor_for_cmd("f") or 0)
        self.assertGreater(baseline, 0)

        entry = telemetry_robot.SCORE_POWER_PWM_DRIVE.get(telemetry_robot.SPEED_SCORE_MIN)
        self.assertIsInstance(entry, dict)
        prev = dict(entry)
        try:
            entry["pwm"] = max(0, baseline - 20)
            _, pwm, _, _ = telemetry_robot.speed_power_pwm_for_cmd("f", telemetry_robot.SPEED_SCORE_MIN)
            self.assertGreaterEqual(int(pwm), baseline)
        finally:
            entry.clear()
            entry.update(prev)


if __name__ == "__main__":
    unittest.main()
