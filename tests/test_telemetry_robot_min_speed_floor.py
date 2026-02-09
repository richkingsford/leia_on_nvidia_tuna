import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class TestTelemetryRobotMinSpeedFloor(unittest.TestCase):
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

