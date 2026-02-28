import math
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessSuccessTrackerVisibilityFalse(unittest.TestCase):
    def test_visible_false_success_gate_uses_standard_visible_only_thresholds(self):
        process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        tracker = telemetry_process.new_success_tracker("EXIT_WALL", process_rules)
        base_consec = int(telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED)
        negative_frames = max(1, base_consec // 2)
        expected_window = max(1, base_consec + negative_frames)
        expected_majority_need = max(1, int(math.ceil(float(expected_window) / 2.0)))

        self.assertEqual(int(tracker.consecutive_required), base_consec)
        self.assertEqual(int(tracker.majority_window), expected_window)
        self.assertEqual(int(tracker.majority_required), expected_majority_need)
        self.assertEqual(
            int(getattr(tracker, "consecutive_pass_required")),
            max(1, int(math.ceil(float(base_consec) / 2.0))),
        )
        self.assertEqual(
            int(getattr(tracker, "majority_pass_required")),
            expected_majority_need,
        )

    def test_visible_true_only_retains_relaxed_pass_thresholds(self):
        process_rules = {
            "FIND_WALL2": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        tracker = telemetry_process.new_success_tracker("FIND_WALL2", process_rules)
        base_consec = int(telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED)
        negative_frames = max(1, base_consec // 2)
        expected_window = max(1, base_consec + negative_frames)
        expected_majority_need = max(1, int(math.ceil(float(expected_window) / 2.0)))

        self.assertEqual(int(tracker.consecutive_required), base_consec)
        self.assertEqual(int(tracker.majority_window), expected_window)
        self.assertEqual(int(tracker.majority_required), expected_majority_need)
        self.assertEqual(
            int(getattr(tracker, "consecutive_pass_required")),
            max(1, int(math.ceil(float(base_consec) / 2.0))),
        )
        self.assertEqual(
            int(getattr(tracker, "majority_pass_required")),
            expected_majority_need,
        )


if __name__ == "__main__":
    unittest.main()
