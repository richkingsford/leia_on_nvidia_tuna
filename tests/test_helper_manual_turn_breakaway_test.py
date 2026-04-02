import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_manual_turn_breakaway_test


class TestHelperManualTurnBreakawayTest(unittest.TestCase):
    def test_run_interactive_turn_breakaway_test_collects_new_defaults(self):
        prompts = iter(["7", "3.00"])
        captured = {}
        original_runner = helper_manual_turn_breakaway_test.run_turn_breakaway_test
        messages = []
        try:
            def _fake_runner(**kwargs):
                captured.update(kwargs)
                return {"ok": True, "summary": {}}

            helper_manual_turn_breakaway_test.run_turn_breakaway_test = _fake_runner
            result = helper_manual_turn_breakaway_test.run_interactive_turn_breakaway_test(
                robot=object(),
                vision=object(),
                world=object(),
                prompt_fn=lambda _prompt: next(prompts),
                log_fn=messages.append,
            )
        finally:
            helper_manual_turn_breakaway_test.run_turn_breakaway_test = original_runner

        self.assertTrue(bool(result.get("ok")))
        self.assertEqual(captured["hotkeys"], ("q", "e"))
        self.assertEqual(captured["ping_pong_max_trials"], 7)
        self.assertEqual(captured["duration_ms"], 130)
        self.assertAlmostEqual(captured["movement_threshold_mm"], 3.0)
        self.assertTrue(any("raw turn pwm floor search" in str(line).lower() for line in messages))
        self.assertTrue(any("fixed 130ms acts" in str(line).lower() for line in messages))

    def test_turn_curve_candidate_uses_raw_score_one_row(self):
        candidate = helper_manual_turn_breakaway_test._turn_curve_candidate_for_hotkey(
            "q",
            duration_ms=130,
        )
        self.assertEqual(candidate["curve_score"], 1)
        self.assertEqual(candidate["curve_pwm"], 102)
        self.assertAlmostEqual(candidate["curve_power"], 0.3013698630136986)
        self.assertEqual(candidate["curve_duration_ms"], 130)
        self.assertEqual(candidate["duration_ms"], 130)

    def test_summarize_candidate_rows_uses_median_threshold(self):
        rows = [
            {"ok": True, "abs_delta_mm": 0.01, "raw_delta_mm": 0.01, "moved": False},
            {"ok": True, "abs_delta_mm": 0.08, "raw_delta_mm": 0.08, "moved": True},
            {"ok": True, "abs_delta_mm": 0.09, "raw_delta_mm": 0.09, "moved": True},
        ]
        summary = helper_manual_turn_breakaway_test._summarize_candidate_rows(
            rows,
            movement_threshold_mm=0.05,
        )
        self.assertEqual(summary["valid_trial_count"], 3)
        self.assertEqual(summary["movement_count"], 2)
        self.assertAlmostEqual(summary["median_abs_delta_mm"], 0.08)
        self.assertTrue(bool(summary["moved"]))

        rows[1]["abs_delta_mm"] = 0.02
        rows[1]["raw_delta_mm"] = 0.02
        rows[1]["moved"] = False
        summary = helper_manual_turn_breakaway_test._summarize_candidate_rows(
            rows,
            movement_threshold_mm=0.05,
        )
        self.assertAlmostEqual(summary["median_abs_delta_mm"], 0.02)
        self.assertFalse(bool(summary["moved"]))

    def test_log_turn_trial_highlights_no_movement_and_movement(self):
        lines = []
        helper_manual_turn_breakaway_test._log_turn_trial(
            lines.append,
            {
                "trial_number": 1,
                "display_label": "Q/LEFT",
                "pwm": 102,
                "power": 0.301,
                "duration_ms": 130,
                "raw_delta_mm": 0.0,
                "result_label": "no movement",
                "moved": False,
            },
        )
        self.assertIn(helper_manual_turn_breakaway_test.ANSI_RED, lines[0])
        self.assertIn("no movement", lines[0])

        helper_manual_turn_breakaway_test._log_turn_trial(
            lines.append,
            {
                "trial_number": 2,
                "display_label": "E/RIGHT",
                "pwm": 104,
                "power": 0.311,
                "duration_ms": 130,
                "raw_delta_mm": -0.083,
                "result_label": "movement",
                "moved": True,
            },
        )
        self.assertIn(helper_manual_turn_breakaway_test.ANSI_GREEN, lines[1])
        self.assertIn("-0.083mm", lines[1])

    def test_dead_direction_abort_requires_consecutive_dead_candidates_and_drift(self):
        candidate_result = {
            "ok": True,
            "moved": False,
            "drift_summary": {
                "timed_out_count": 1,
                "max_abs_x_axis_error_mm": 14.0,
                "max_abs_dist_error_mm": 2.0,
            },
        }
        abort_info = helper_manual_turn_breakaway_test._dead_direction_abort_info(
            candidate_result,
            helper_manual_turn_breakaway_test.TURN_ABORT_CONSECUTIVE_DEAD_CANDIDATES,
        )
        self.assertEqual(abort_info["reason"], "dead_direction_abort")
        self.assertEqual(abort_info["consecutive_dead_candidates"], 2)

        no_abort = helper_manual_turn_breakaway_test._dead_direction_abort_info(
            {
                "ok": True,
                "moved": False,
                "drift_summary": {
                    "timed_out_count": 0,
                    "max_abs_x_axis_error_mm": 30.0,
                    "max_abs_dist_error_mm": 30.0,
                },
            },
            helper_manual_turn_breakaway_test.TURN_ABORT_CONSECUTIVE_DEAD_CANDIDATES,
        )
        self.assertIsNone(no_abort)


if __name__ == "__main__":
    unittest.main()
