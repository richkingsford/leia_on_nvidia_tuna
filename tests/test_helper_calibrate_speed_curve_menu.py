import unittest

import helper_calibrate_speed_curve


class HelperCalibrateSpeedCurveMenuTests(unittest.TestCase):
    def test_telemetry_option_is_exposed_in_calibration_menu(self):
        option = next(
            (item for item in helper_calibrate_speed_curve.OPTIONS if str(item.key) == "telemetry"),
            None,
        )

        self.assertIsNotNone(option)
        self.assertIn("telemetry", str(option.label).lower())
        self.assertIs(helper_calibrate_speed_curve._resolve_choice("telemetry"), option)

    def test_breakaway_option_is_exposed_in_calibration_menu(self):
        option = next(
            (item for item in helper_calibrate_speed_curve.OPTIONS if str(item.key) == "breakaway"),
            None,
        )

        self.assertIsNotNone(option)
        self.assertIn("breakaway", str(option.label).lower())
        self.assertIs(helper_calibrate_speed_curve._resolve_choice("breakaway"), option)

    def test_turn_breakaway_option_is_exposed_in_calibration_menu(self):
        option = next(
            (item for item in helper_calibrate_speed_curve.OPTIONS if str(item.key) == "turn-breakaway"),
            None,
        )

        self.assertIsNotNone(option)
        self.assertIn("turn breakaway", str(option.label).lower())
        self.assertIs(helper_calibrate_speed_curve._resolve_choice("turn-breakaway"), option)


if __name__ == "__main__":
    unittest.main()
