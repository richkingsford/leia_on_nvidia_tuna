import copy
import unittest
from pathlib import Path

import telemetry_robot


class TestDirectionalTurnDuration(unittest.TestCase):
    def setUp(self):
        self.orig_use = telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS
        self.orig_shared = telemetry_robot.SPEED_SCORE_DURATION_MS
        self.orig_turn = telemetry_robot.SPEED_SCORE_DURATION_MS_TURN
        self.orig_left = telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_LEFT
        self.orig_right = telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_RIGHT
        self.orig_turn_breakaway_path = telemetry_robot.TURN_BREAKAWAY_TEST_FILE
        self.orig_shared_values = copy.deepcopy(self.orig_shared) if isinstance(self.orig_shared, dict) else None
        self.orig_turn_values = copy.deepcopy(self.orig_turn) if isinstance(self.orig_turn, dict) else None
        self.orig_left_values = copy.deepcopy(self.orig_left) if isinstance(self.orig_left, dict) else None
        self.orig_right_values = copy.deepcopy(self.orig_right) if isinstance(self.orig_right, dict) else None
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = Path(__file__).resolve().parents[1] / "_missing_turn_breakaway_test.json"

    def tearDown(self):
        telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS = self.orig_use
        telemetry_robot.SPEED_SCORE_DURATION_MS = self.orig_shared
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN = self.orig_turn
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_LEFT = self.orig_left
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_RIGHT = self.orig_right
        telemetry_robot.TURN_BREAKAWAY_TEST_FILE = self.orig_turn_breakaway_path
        if isinstance(self.orig_shared, dict) and isinstance(self.orig_shared_values, dict):
            self.orig_shared.clear()
            self.orig_shared.update(self.orig_shared_values)
        if isinstance(self.orig_turn, dict) and isinstance(self.orig_turn_values, dict):
            self.orig_turn.clear()
            self.orig_turn.update(self.orig_turn_values)
        if isinstance(self.orig_left, dict) and isinstance(self.orig_left_values, dict):
            self.orig_left.clear()
            self.orig_left.update(self.orig_left_values)
        if isinstance(self.orig_right, dict) and isinstance(self.orig_right_values, dict):
            self.orig_right.clear()
            self.orig_right.update(self.orig_right_values)

    def test_turn_cmd_uses_directional_duration_map_when_enabled(self):
        telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS = True

        self.orig_shared.clear()
        self.orig_shared.update({1: 250, 100: 300})
        self.orig_turn.clear()
        self.orig_turn.update({1: 260, 100: 300})
        self.orig_left.clear()
        self.orig_left.update({1: 111, 100: 300})
        self.orig_right.clear()
        self.orig_right.update({1: 222, 100: 300})

        _, _, _, l_ms = telemetry_robot.speed_power_pwm_for_cmd("l", 1)
        _, _, _, r_ms = telemetry_robot.speed_power_pwm_for_cmd("r", 1)
        _, _, _, f_ms = telemetry_robot.speed_power_pwm_for_cmd("f", 1)

        self.assertEqual(l_ms, 111)
        self.assertEqual(r_ms, 222)
        self.assertEqual(f_ms, 250)

    def test_legacy_shared_duration_override_applies_to_turns(self):
        telemetry_robot.USE_DIRECTIONAL_TURN_DURATION_MAPS = True
        telemetry_robot.SPEED_SCORE_DURATION_MS = {1: 333, 100: 300}
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN = {1: 260, 100: 300}
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_LEFT = {1: 111, 100: 300}
        telemetry_robot.SPEED_SCORE_DURATION_MS_TURN_RIGHT = {1: 222, 100: 300}

        _, _, _, l_ms = telemetry_robot.speed_power_pwm_for_cmd("l", 1)
        _, _, _, r_ms = telemetry_robot.speed_power_pwm_for_cmd("r", 1)

        self.assertEqual(l_ms, 333)
        self.assertEqual(r_ms, 333)


if __name__ == "__main__":
    unittest.main()
