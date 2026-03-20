import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process
import telemetry_robot


class _DummyRobot:
    def __init__(self):
        self.sent = []
        self._last_turn_cmd = "r"

    def send_command_pwm(self, cmd, pwm, duration_ms=0):
        self.sent.append((cmd, pwm, duration_ms))


class TestTelemetryProcessAutoTurnDurationFloor(unittest.TestCase):
    def test_auto_turn_duration_never_below_min_score_duration(self):
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

            robot = _DummyRobot()
            meta = telemetry_process.send_robot_command_pwm(
                robot,
                world=None,
                step="ALIGN_BRICK",
                cmd="l",
                power=0.0,
                pwm=60,
                duration_ms=250,
                speed_score=telemetry_robot.SPEED_SCORE_MIN,
                auto_mode=True,
            )
            self.assertIsInstance(meta, dict)
            self.assertTrue(robot.sent)
            cmd, _, duration_ms = robot.sent[-1]
            self.assertEqual(cmd, "l")
            self.assertEqual(duration_ms, 250)
            self.assertEqual(meta.get("duration_ms"), 250)
        finally:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = orig_drive
            telemetry_robot.SCORE_POWER_PWM_TURN = orig_turn
            telemetry_robot.SPEED_SCORE_DURATION_MS = orig_duration
            telemetry_robot.refresh_speed_model_baseline()

    def test_auto_turn_pwm_and_duration_respect_requested_score_floor(self):
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

            robot = _DummyRobot()
            req_score = 2
            _, req_pwm, _, req_ms = telemetry_robot.speed_power_pwm_for_cmd("r", req_score)
            meta = telemetry_process.send_robot_command_pwm(
                robot,
                world=None,
                step="ALIGN_BRICK",
                cmd="r",
                power=0.0,
                pwm=1,
                duration_ms=10,
                speed_score=req_score,
                auto_mode=True,
            )
            self.assertIsInstance(meta, dict)
            self.assertTrue(robot.sent)
            cmd, pwm_sent, duration_sent = robot.sent[-1]
            self.assertEqual(cmd, "r")
            self.assertGreaterEqual(int(pwm_sent), int(req_pwm))
            self.assertGreaterEqual(int(duration_sent), int(req_ms))
            self.assertGreaterEqual(int(duration_sent), 250)
        finally:
            telemetry_robot.SCORE_POWER_PWM_DRIVE = orig_drive
            telemetry_robot.SCORE_POWER_PWM_TURN = orig_turn
            telemetry_robot.SPEED_SCORE_DURATION_MS = orig_duration
            telemetry_robot.refresh_speed_model_baseline()

    def test_auto_turn_duration_override_is_literal(self):
        robot = _DummyRobot()

        meta = telemetry_process.send_robot_command(
            robot,
            world=None,
            step="FIND_WALL2",
            cmd="l",
            speed=0.0,
            speed_score=3,
            auto_mode=True,
            duration_override_ms=5000,
        )

        self.assertIsInstance(meta, dict)
        self.assertTrue(robot.sent)
        cmd, _pwm, duration_ms = robot.sent[-1]
        self.assertEqual(cmd, "l")
        self.assertEqual(int(duration_ms), 5000)
        self.assertEqual(int(meta.get("duration_ms") or 0), 5000)


if __name__ == "__main__":
    unittest.main()
