import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process
import telemetry_robot


class _DummyRobot:
    def __init__(self):
        self.sent = []
        self._last_turn_cmd = None

    def send_command_pwm(self, cmd, pwm, duration_ms=0):
        self.sent.append((cmd, pwm, duration_ms))


class _DummyWorld:
    pass


class TestManualHotkeySpeedScoreMin(unittest.TestCase):
    def test_manual_score_min_uses_model_duration_and_cmd_maps(self):
        orig_drive = telemetry_robot.SCORE_POWER_PWM_DRIVE
        orig_turn = telemetry_robot.SCORE_POWER_PWM_TURN
        orig_duration = telemetry_robot.SPEED_SCORE_DURATION_MS
        try:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = {
                telemetry_robot.SPEED_SCORE_MIN: {"pwm": 80},
                telemetry_robot.SPEED_SCORE_MAX: {"pwm": 200},
            }
            telemetry_robot.SCORE_POWER_PWM_TURN = {
                telemetry_robot.SPEED_SCORE_MIN: {"pwm": 60},
                telemetry_robot.SPEED_SCORE_MAX: {"pwm": 220},
            }
            telemetry_robot.SPEED_SCORE_DURATION_MS = {
                telemetry_robot.SPEED_SCORE_MIN: 250,
                telemetry_robot.SPEED_SCORE_MAX: 300,
            }
            telemetry_robot.refresh_speed_model_baseline()

            world = _DummyWorld()
            robot = _DummyRobot()

            meta_drive = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="f",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=False,
            )
            self.assertTrue(robot.sent)
            self.assertIsInstance(meta_drive, dict)
            cmd, pwm, duration_ms = robot.sent[-1]
            self.assertEqual(cmd, "f")
            self.assertEqual(pwm, 80)
            self.assertEqual(duration_ms, 250)
            self.assertEqual(meta_drive.get("duration_ms"), 250)

            # Sequential same-direction turn should use the full model duration.
            robot._last_turn_cmd = "l"
            meta_turn = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="l",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=False,
            )
            self.assertIsInstance(meta_turn, dict)
            cmd, pwm, duration_ms = robot.sent[-1]
            self.assertEqual(cmd, "l")
            self.assertEqual(pwm, 60)
            self.assertEqual(duration_ms, 250)
            self.assertEqual(meta_turn.get("duration_ms"), 250)
        finally:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = orig_drive
            telemetry_robot.SCORE_POWER_PWM_TURN = orig_turn
            telemetry_robot.SPEED_SCORE_DURATION_MS = orig_duration
            telemetry_robot.refresh_speed_model_baseline()

    def test_manual_first_turn_uses_half_duration_of_sequential_turn(self):
        orig_drive = telemetry_robot.SCORE_POWER_PWM_DRIVE
        orig_turn = telemetry_robot.SCORE_POWER_PWM_TURN
        orig_duration = telemetry_robot.SPEED_SCORE_DURATION_MS
        try:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = {
                telemetry_robot.SPEED_SCORE_MIN: {"pwm": 80},
                telemetry_robot.SPEED_SCORE_MAX: {"pwm": 200},
            }
            telemetry_robot.SCORE_POWER_PWM_TURN = {
                telemetry_robot.SPEED_SCORE_MIN: {"pwm": 60},
                telemetry_robot.SPEED_SCORE_MAX: {"pwm": 220},
            }
            telemetry_robot.SPEED_SCORE_DURATION_MS = {
                telemetry_robot.SPEED_SCORE_MIN: 250,
                telemetry_robot.SPEED_SCORE_MAX: 300,
            }
            telemetry_robot.refresh_speed_model_baseline()

            world = _DummyWorld()
            robot = _DummyRobot()

            first = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="l",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=False,
            )
            second = telemetry_process.send_robot_command(
                robot,
                world,
                step="MANUAL",
                cmd="l",
                speed=0.0,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=False,
            )
            self.assertIsInstance(first, dict)
            self.assertIsInstance(second, dict)
            first_ms = int(first.get("duration_ms") or 0)
            second_ms = int(second.get("duration_ms") or 0)
            self.assertEqual(first_ms, 125)
            self.assertEqual(second_ms, 250)
        finally:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = orig_drive
            telemetry_robot.SCORE_POWER_PWM_TURN = orig_turn
            telemetry_robot.SPEED_SCORE_DURATION_MS = orig_duration
            telemetry_robot.refresh_speed_model_baseline()


if __name__ == "__main__":
    unittest.main()
