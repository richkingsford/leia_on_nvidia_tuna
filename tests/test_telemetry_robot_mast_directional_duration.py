import copy
import unittest

import telemetry_robot


class TestDirectionalMastDuration(unittest.TestCase):
    def setUp(self):
        self.orig_use = telemetry_robot.USE_DIRECTIONAL_MAST_DURATION_MAPS
        self.orig_shared = telemetry_robot.SPEED_SCORE_DURATION_MS
        self.orig_up = telemetry_robot.SPEED_SCORE_DURATION_MS_MAST_UP
        self.orig_down = telemetry_robot.SPEED_SCORE_DURATION_MS_MAST_DOWN
        self.orig_shared_values = copy.deepcopy(self.orig_shared) if isinstance(self.orig_shared, dict) else None
        self.orig_up_values = copy.deepcopy(self.orig_up) if isinstance(self.orig_up, dict) else None
        self.orig_down_values = copy.deepcopy(self.orig_down) if isinstance(self.orig_down, dict) else None

    def tearDown(self):
        telemetry_robot.USE_DIRECTIONAL_MAST_DURATION_MAPS = self.orig_use
        telemetry_robot.SPEED_SCORE_DURATION_MS = self.orig_shared
        telemetry_robot.SPEED_SCORE_DURATION_MS_MAST_UP = self.orig_up
        telemetry_robot.SPEED_SCORE_DURATION_MS_MAST_DOWN = self.orig_down
        if isinstance(self.orig_shared, dict) and isinstance(self.orig_shared_values, dict):
            self.orig_shared.clear()
            self.orig_shared.update(self.orig_shared_values)
        if isinstance(self.orig_up, dict) and isinstance(self.orig_up_values, dict):
            self.orig_up.clear()
            self.orig_up.update(self.orig_up_values)
        if isinstance(self.orig_down, dict) and isinstance(self.orig_down_values, dict):
            self.orig_down.clear()
            self.orig_down.update(self.orig_down_values)

    def test_mast_cmd_uses_directional_duration_map_when_enabled(self):
        telemetry_robot.USE_DIRECTIONAL_MAST_DURATION_MAPS = True

        self.orig_shared.clear()
        self.orig_shared.update({1: 250, 20: 275, 100: 300})
        self.orig_up.clear()
        self.orig_up.update({1: 230, 20: 245, 100: 270})
        self.orig_down.clear()
        self.orig_down.update({1: 180, 20: 200, 100: 230})

        _, _, _, u_ms = telemetry_robot.speed_power_pwm_for_cmd("u", 20)
        _, _, _, d_ms = telemetry_robot.speed_power_pwm_for_cmd("d", 20)
        _, _, _, f_ms = telemetry_robot.speed_power_pwm_for_cmd("f", 20)

        self.assertEqual(u_ms, 245)
        self.assertEqual(d_ms, 200)
        self.assertEqual(f_ms, 275)

    def test_legacy_shared_duration_override_applies_to_mast(self):
        telemetry_robot.USE_DIRECTIONAL_MAST_DURATION_MAPS = True
        telemetry_robot.SPEED_SCORE_DURATION_MS = {1: 333, 20: 333, 100: 300}
        telemetry_robot.SPEED_SCORE_DURATION_MS_MAST_UP = {1: 230, 20: 245, 100: 270}
        telemetry_robot.SPEED_SCORE_DURATION_MS_MAST_DOWN = {1: 180, 20: 200, 100: 230}

        _, _, _, u_ms = telemetry_robot.speed_power_pwm_for_cmd("u", 20)
        _, _, _, d_ms = telemetry_robot.speed_power_pwm_for_cmd("d", 20)

        self.assertEqual(u_ms, 333)
        self.assertEqual(d_ms, 333)


if __name__ == "__main__":
    unittest.main()
