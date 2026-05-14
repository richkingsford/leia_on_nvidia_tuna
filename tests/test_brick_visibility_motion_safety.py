import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_brick_visibility_safety as safety


class _FakeRobot:
    def __init__(self):
        self.commands = []
        self.custom_commands = []
        self.stops = 0

    def send_command_pwm(self, cmd, pwm, duration_ms=None):
        self.commands.append((cmd, pwm, duration_ms))
        return {"cmd": cmd, "pwm": pwm, "duration_ms": duration_ms}

    def send_custom_actions_pwm(self, cmd, actions, duration_ms=None):
        self.custom_commands.append((cmd, list(actions), duration_ms))
        return {"cmd": cmd, "actions": list(actions), "duration_ms": duration_ms}

    def stop(self):
        self.stops += 1


class TestBrickVisibilityMotionSafety(unittest.TestCase):
    def test_tuple_reading_requires_visible_and_min_confidence(self):
        confident = safety.brick_motion_measurement_from_result(
            (True, 0.0, 180.0, 4.0, 88.0, 0.0, False, False),
            min_confidence_pct=75.0,
        )
        self.assertTrue(confident["visible"])
        self.assertTrue(confident["confident"])
        self.assertEqual(confident["reason"], "confident_visible")

        low_conf = safety.brick_motion_measurement_from_result(
            (True, 0.0, 180.0, 4.0, 60.0, 0.0, False, False),
            min_confidence_pct=75.0,
        )
        self.assertTrue(low_conf["visible"])
        self.assertFalse(low_conf["confident"])
        self.assertEqual(low_conf["reason"], "low_confidence")

    def test_guarded_command_blocks_motion_without_confident_brick(self):
        robot = _FakeRobot()
        reading = safety.brick_motion_measurement_from_result(
            (True, 0.0, 180.0, 4.0, 60.0, 0.0, False, False),
            min_confidence_pct=75.0,
        )

        result = safety.guarded_send_command_pwm(
            robot,
            "f",
            120,
            duration_ms=100,
            reading=reading,
            context="test_forward",
        )

        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "low_confidence")
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.stops, 1)

    def test_guarded_command_allows_stop_without_visibility(self):
        robot = _FakeRobot()

        result = safety.guarded_send_command_pwm(
            robot,
            "f",
            0,
            duration_ms=100,
            reading=None,
            context="test_stop",
        )

        self.assertFalse(result.get("blocked", False))
        self.assertEqual(robot.commands, [("f", 0, 100)])
        self.assertEqual(robot.stops, 0)

    def test_guarded_custom_actions_blocks_moving_tread_without_visibility(self):
        robot = _FakeRobot()

        result = safety.guarded_send_custom_actions_pwm(
            robot,
            "r",
            [{"target": "l", "action": "f", "pwm": 120}, {"target": "r", "action": "s", "pwm": 0}],
            duration_ms=100,
            reading={"visible": False, "conf": 0.0, "min_confidence_pct": 75.0},
            context="test_custom",
        )

        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "not_confidently_visible")
        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.stops, 1)


if __name__ == "__main__":
    unittest.main()
