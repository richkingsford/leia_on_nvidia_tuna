import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessSuccessTrackerVisibilityFalse(unittest.TestCase):
    def test_visible_false_success_gate_uses_strict_global_thresholds(self):
        process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        with mock.patch.object(
            telemetry_process,
            "_active_success_gate_mode",
            return_value=telemetry_process.VISION_MODE_CYAN,
        ):
            tracker = telemetry_process.new_success_tracker("EXIT_WALL", process_rules)
        base_consec = int(telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED)
        expected_window = int(telemetry_process.GATECHECK_MAJORITY_WINDOW)
        expected_majority_need = int(telemetry_process.GATECHECK_MAJORITY_REQUIRED)

        self.assertEqual(int(tracker.consecutive_required), base_consec)
        self.assertEqual(int(tracker.majority_window), expected_window)
        self.assertEqual(int(tracker.majority_required), expected_majority_need)
        self.assertEqual(
            int(getattr(tracker, "consecutive_pass_required")),
            base_consec,
        )
        self.assertEqual(
            int(getattr(tracker, "majority_pass_required")),
            expected_majority_need,
        )

    def test_visible_true_only_uses_strict_global_thresholds(self):
        process_rules = {
            "FIND_WALL2": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        with mock.patch.object(
            telemetry_process,
            "_active_success_gate_mode",
            return_value=telemetry_process.VISION_MODE_CYAN,
        ):
            tracker = telemetry_process.new_success_tracker("FIND_WALL2", process_rules)
        base_consec = int(telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED)
        expected_window = int(telemetry_process.GATECHECK_MAJORITY_WINDOW)
        expected_majority_need = int(telemetry_process.GATECHECK_MAJORITY_REQUIRED)

        self.assertEqual(int(tracker.consecutive_required), base_consec)
        self.assertEqual(int(tracker.majority_window), expected_window)
        self.assertEqual(int(tracker.majority_required), expected_majority_need)
        self.assertEqual(
            int(getattr(tracker, "consecutive_pass_required")),
            base_consec,
        )
        self.assertEqual(
            int(getattr(tracker, "majority_pass_required")),
            expected_majority_need,
        )


if __name__ == "__main__":
    unittest.main()
