import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import practiceStep12EmptyHalfReset as step12


class TestStep12PracticeSafety(unittest.TestCase):
    def test_85_percent_requires_5_of_5_trials(self):
        self.assertEqual(step12._required_wins_for_rate(5, 0.85), 5)

    def test_block_summary_marks_partial_clean_run_incomplete(self):
        summary = step12._block_summary_counts(
            [
                {
                    "clean": True,
                    "s1_won": True,
                    "s2_won": True,
                    "wrong_way_count": 0,
                    "severe_overshoot_count": 0,
                }
            ],
            requested_trials=5,
            min_win_rate=0.85,
        )

        self.assertEqual(summary["completed_trials"], 1)
        self.assertEqual(summary["required_wins"], 5)
        self.assertFalse(summary["meets_trial_count"])
        self.assertFalse(summary["meets_min_win_rate"])
        self.assertAlmostEqual(summary["completed_s1_win_rate"], 1.0)
        self.assertAlmostEqual(summary["s1_win_rate"], 0.2)

    def test_block_summary_flags_100x_clean_proof(self):
        rows = [
            {
                "clean": True,
                "s1_won": True,
                "s2_won": True,
                "wrong_way_count": 0,
                "severe_overshoot_count": 0,
            }
            for _ in range(100)
        ]

        summary = step12._block_summary_counts(rows, requested_trials=100, min_win_rate=0.85)

        self.assertEqual(summary["longest_clean_streak"], 100)
        self.assertTrue(summary["meets_trial_count"])
        self.assertTrue(summary["meets_min_win_rate"])
        self.assertTrue(summary["meets_safety"])
        self.assertTrue(summary["meets_100x_proof"])

    def test_far_low_confident_reading_is_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 299.5, "y_mm": -85.4}

        self.assertTrue(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )

    def test_shifted_live_far_low_reading_is_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 299.5, "y_mm": -56.7}

        self.assertTrue(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )

    def test_normal_far_start_is_not_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 323.8, "y_mm": -32.2}

        self.assertFalse(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )

    def test_not_confident_reading_uses_existing_visibility_stop_path(self):
        reading = {"confident": False, "dist_mm": 299.5, "y_mm": -85.4}

        self.assertFalse(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-50.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
