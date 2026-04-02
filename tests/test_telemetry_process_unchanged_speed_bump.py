import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessUnchangedSpeedBump(unittest.TestCase):
    def test_should_fail_required_gap_unknown_only_in_close_gap_phase(self):
        self.assertFalse(
            telemetry_process._should_fail_required_gap_unknown(
                {"visible": False},
                is_search_action=False,
            )
        )
        self.assertFalse(
            telemetry_process._should_fail_required_gap_unknown(
                {"visible": True},
                is_search_action=True,
            )
        )
        self.assertTrue(
            telemetry_process._should_fail_required_gap_unknown(
                {"visible": True},
                is_search_action=False,
            )
        )

    def test_required_align_unknown_gaps_flags_required_missing_fields(self):
        missing = telemetry_process._required_align_unknown_gaps(
            {
                "x_required": True,
                "x_err": None,
                "y_required": True,
                "y_err": 1.2,
                "dist_required": True,
                "dist": None,
                "dist_target": 50.0,
            }
        )
        self.assertEqual(missing, ("x_err", "dist"))

    def test_required_align_unknown_gaps_ignores_non_required_missing_fields(self):
        missing = telemetry_process._required_align_unknown_gaps(
            {
                "x_required": False,
                "x_err": None,
                "y_required": False,
                "y_err": None,
                "dist_required": False,
                "dist": None,
                "dist_target": None,
            }
        )
        self.assertEqual(missing, ())

    def test_required_align_unknown_gaps_flags_missing_dist_target(self):
        missing = telemetry_process._required_align_unknown_gaps(
            {
                "x_required": True,
                "x_err": 0.0,
                "y_required": False,
                "y_err": None,
                "dist_required": True,
                "dist": 42.0,
                "dist_target": None,
            }
        )
        self.assertEqual(missing, ("dist_target",))

    def test_lite_fail_fallback_supports_seat_brick_step(self):
        class _World:
            process_rules = {
                "SEAT_BRICK": {
                    "success_gates": {
                        "visible": {"min": True},
                        "dist": {"target": 48.0, "tol": 1.5},
                    }
                }
            }
            brick = {"visible": True, "dist": 45.2}

        orig_lite_measure = telemetry_process._lite_gate_measurement_for_step
        try:
            telemetry_process._lite_gate_measurement_for_step = (
                lambda *_a, **_k: (
                    {"visible": True, "dist": 45.2},
                    {"enabled": True, "required": 1, "collected": 1},
                )
            )
            act = telemetry_process._align_brick_lite_fail_fallback_action(_World(), "SEAT_BRICK")
        finally:
            telemetry_process._lite_gate_measurement_for_step = orig_lite_measure

        self.assertIsInstance(act, dict)
        self.assertEqual(act.get("reason"), "lite_fail_fallback_dist")
        self.assertEqual(act.get("cmd"), "b")
        self.assertEqual(act.get("score"), 1)

    def test_lite_fail_fallback_can_use_current_effective_y_axis_failure(self):
        class _World:
            process_rules = {
                "ALIGN_BRICK": {
                    "success_gates": {
                        "visible": {"min": True},
                        "xAxis_offset_abs": {"target": 7.1, "tol": 1.4},
                        "yAxis_offset_abs": {"target": 3.7, "tol": 2.3},
                        "dist": {"target": 103.6, "tol": 2.3},
                    }
                }
            }
            brick = {
                "visible": True,
                "dist": 104.7,
                "x_axis": 8.4,
                "offset_x": 8.4,
                "y_axis": 6.2,
                "offset_y": 6.2,
            }

        orig_lite_measure = telemetry_process._lite_gate_measurement_for_step
        try:
            telemetry_process._lite_gate_measurement_for_step = (
                lambda *_a, **_k: (
                    {"visible": True, "dist": 104.7, "x_axis": 8.4, "offset_x": 8.4, "y_axis": 5.9, "offset_y": 5.9},
                    {"enabled": True, "required": 1, "collected": 1},
                )
            )
            act = telemetry_process._align_brick_lite_fail_fallback_action(
                _World(),
                "ALIGN_BRICK",
                prefer_effective_sample=True,
            )
        finally:
            telemetry_process._lite_gate_measurement_for_step = orig_lite_measure

        self.assertIsInstance(act, dict)
        self.assertEqual(act.get("reason"), "lite_fail_fallback_y_axis")
        self.assertEqual(act.get("cmd"), "d")
        self.assertEqual(act.get("score"), 1)


if __name__ == "__main__":
    unittest.main()
