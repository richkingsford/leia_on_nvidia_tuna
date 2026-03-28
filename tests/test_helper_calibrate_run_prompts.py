import unittest
from unittest.mock import patch

from calibration import helper_calibrate


class HelperCalibrateRunPromptTests(unittest.TestCase):
    def test_prompt_calibration_run_settings_returns_defaults_without_interactive_stdin(self):
        with patch.object(helper_calibrate, "stdin_supports_interactive_input", return_value=False):
            settings = helper_calibrate.prompt_calibration_run_settings(
                prefix="CALIBRATE_TEST",
                observed_distance_mm=111.0,
                default_speed_score=5,
                default_min_duration_ms=200,
                default_max_duration_ms=400,
                duration_ceiling_ms=helper_calibrate.CALIBRATION_DURATION_LIMIT_MS,
            )

        self.assertEqual(
            settings,
            {
                "speed_score": 5,
                "min_duration_ms": 200,
                "max_duration_ms": 400,
                "prompted_speed_score": False,
                "prompted_duration_bounds": False,
                "prompted_any": False,
            },
        )

    def test_prompt_calibration_run_settings_logs_observed_distance_before_prompting(self):
        logs = []
        prompts = []
        responses = iter(["7", "150", "300"])

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return next(responses)

        with patch.object(helper_calibrate, "stdin_supports_interactive_input", return_value=True), patch(
            "builtins.input",
            side_effect=fake_input,
        ):
            settings = helper_calibrate.prompt_calibration_run_settings(
                prefix="CALIBRATE_TEST",
                observed_distance_mm=111.0,
                default_speed_score=5,
                default_min_duration_ms=200,
                default_max_duration_ms=400,
                duration_ceiling_ms=helper_calibrate.CALIBRATION_DURATION_LIMIT_MS,
                log=logs.append,
            )

        self.assertEqual(logs[0], "[CALIBRATE_TEST] Observed dist before prompts: 111.00mm.")
        self.assertEqual(logs[1], "[CALIBRATE_TEST] Enter run settings for this calibration.")
        self.assertEqual(
            prompts,
            [
                "  Speed score % [5]: ",
                "  Min duration ms [200]: ",
                "  Max duration ms [400]: ",
            ],
        )
        self.assertEqual(
            settings,
            {
                "speed_score": 7,
                "min_duration_ms": 150,
                "max_duration_ms": 300,
                "prompted_speed_score": True,
                "prompted_duration_bounds": True,
                "prompted_any": True,
            },
        )

    def test_prompt_calibration_run_settings_allows_large_duration_values_below_10000(self):
        with patch.object(helper_calibrate, "stdin_supports_interactive_input", return_value=True), patch(
            "builtins.input",
            side_effect=["5", "200", "1500"],
        ):
            settings = helper_calibrate.prompt_calibration_run_settings(
                prefix="CALIBRATE_TEST",
                observed_distance_mm=111.0,
                default_speed_score=5,
                default_min_duration_ms=200,
                default_max_duration_ms=400,
                duration_ceiling_ms=helper_calibrate.CALIBRATION_DURATION_LIMIT_MS,
            )

        self.assertEqual(settings["max_duration_ms"], 1500)

    def test_prompt_calibration_run_settings_reprompts_invalid_values(self):
        logs = []
        responses = iter(["0", "8", "-1", "150", "149", "320"])

        with patch.object(helper_calibrate, "stdin_supports_interactive_input", return_value=True), patch(
            "builtins.input",
            side_effect=lambda _prompt: next(responses),
        ):
            settings = helper_calibrate.prompt_calibration_run_settings(
                prefix="CALIBRATE_TEST",
                observed_distance_mm=None,
                default_speed_score=5,
                default_min_duration_ms=200,
                default_max_duration_ms=400,
                duration_ceiling_ms=helper_calibrate.CALIBRATION_DURATION_LIMIT_MS,
                log=logs.append,
            )

        self.assertEqual(logs[0], "[CALIBRATE_TEST] Observed dist before prompts: unknown.")
        self.assertIn("Please enter a whole number between 1 and 100.", logs)
        self.assertIn("Please enter a whole number between 1 and 9999.", logs)
        self.assertIn("Please enter a whole number between 150 and 9999.", logs)
        self.assertEqual(settings["speed_score"], 8)
        self.assertEqual(settings["min_duration_ms"], 150)
        self.assertEqual(settings["max_duration_ms"], 320)


if __name__ == "__main__":
    unittest.main()
