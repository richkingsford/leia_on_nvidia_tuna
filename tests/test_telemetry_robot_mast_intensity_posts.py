import copy
import unittest

import telemetry_robot


class TestTelemetryRobotMastIntensityPosts(unittest.TestCase):
    def setUp(self):
        self.orig_up = telemetry_robot.MAST_INTENSITY_POSTS_UP
        self.orig_down = telemetry_robot.MAST_INTENSITY_POSTS_DOWN
        self.orig_up_values = copy.deepcopy(self.orig_up) if isinstance(self.orig_up, dict) else None
        self.orig_down_values = copy.deepcopy(self.orig_down) if isinstance(self.orig_down, dict) else None

    def tearDown(self):
        telemetry_robot.MAST_INTENSITY_POSTS_UP = self.orig_up
        telemetry_robot.MAST_INTENSITY_POSTS_DOWN = self.orig_down
        if isinstance(self.orig_up, dict) and isinstance(self.orig_up_values, dict):
            self.orig_up.clear()
            self.orig_up.update(self.orig_up_values)
        if isinstance(self.orig_down, dict) and isinstance(self.orig_down_values, dict):
            self.orig_down.clear()
            self.orig_down.update(self.orig_down_values)

    def test_motion_intensity_uses_mast_down_posts(self):
        telemetry_robot.MAST_INTENSITY_POSTS_DOWN = {
            4.0: {"pwm": 40, "duration_ms": 260},
            12.0: {"pwm": 42, "duration_ms": 800},
            18.0: {"pwm": 46, "duration_ms": 950},
        }

        power, pwm, score, duration_ms, intensity_eff = telemetry_robot.speed_power_pwm_for_motion_intensity("d", 12.0)

        self.assertEqual(pwm, 42)
        self.assertEqual(duration_ms, 800)
        self.assertEqual(float(intensity_eff), 12.0)
        self.assertGreater(float(power), 0.0)
        self.assertGreaterEqual(int(score), 1)

    def test_motion_intensity_interpolates_between_mast_posts(self):
        telemetry_robot.MAST_INTENSITY_POSTS_DOWN = {
            8.0: {"pwm": 40, "duration_ms": 420},
            18.0: {"pwm": 46, "duration_ms": 950},
        }

        _power, pwm, _score, duration_ms, intensity_eff = telemetry_robot.speed_power_pwm_for_motion_intensity("d", 13.0)

        self.assertEqual(float(intensity_eff), 13.0)
        self.assertEqual(pwm, 43)
        self.assertEqual(duration_ms, 685)


if __name__ == "__main__":
    unittest.main()
