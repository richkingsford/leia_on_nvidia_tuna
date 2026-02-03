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


class TestTelemetryProcessActionDisplay(unittest.TestCase):
    def test_send_robot_command_records_display_from_effective_command(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        telemetry_process.send_robot_command(
            robot,
            world,
            step="ALIGN_BRICK",
            cmd="l",
            speed=0.0,
            speed_score=telemetry_robot.SPEED_SCORE_MIN,
            auto_mode=True,
        )
        self.assertTrue(robot.sent)
        self.assertEqual(getattr(world, "_last_action_cmd", None), "l")
        expected_score = telemetry_robot.quantize_speed("l", speed=getattr(world, "_last_action_speed", 0.0))[1]
        expected = telemetry_process.action_display_text("l", expected_score)
        self.assertEqual(getattr(world, "_last_action_display", None), expected)


if __name__ == "__main__":
    unittest.main()
