import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process
import telemetry_robot


class TestTelemetryProcessFindBrickTurnSpeedPolicy(unittest.TestCase):
    def test_find_brick_turn_check_phase_caps_to_min_score(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "l",
            25,
            phase="check",
        )
        expected = telemetry_robot.normalize_speed_score(telemetry_robot.SPEED_SCORE_MIN)
        self.assertEqual(score, expected)

    def test_find_brick_turn_move_phase_caps_to_two_percent(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "r",
            25,
            phase="move",
        )
        self.assertEqual(score, telemetry_robot.normalize_speed_score(2))

    def test_find_brick_turn_demo_phase_preserves_demo_score(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "r",
            9,
            phase="demo",
        )
        self.assertEqual(score, telemetry_robot.normalize_speed_score(9))

    def test_non_turn_command_is_unchanged(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "FIND_BRICK",
            "f",
            9,
            phase="move",
        )
        self.assertEqual(score, 9)

    def test_other_steps_are_unchanged(self):
        score = telemetry_process._apply_find_brick_turn_speed_policy(
            "ALIGN_BRICK",
            "l",
            9,
            phase="move",
        )
        self.assertEqual(score, 9)


if __name__ == "__main__":
    unittest.main()
