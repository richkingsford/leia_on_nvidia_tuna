import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class TestTelemetryRobotStreamOverlayMetrics(unittest.TestCase):
    def setUp(self):
        self.rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -2.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 1.0, "tol": 1.0},
                    "dist": {"target": 90.0, "tol": 2.0},
                }
            },
            "APPROACH_VECTOR_BRICK_SUPPLY": {
                "start_gates": {
                    "visible": {"min": True},
                }
            },
        }

    def test_uses_success_gate_metric_order_for_selected_step(self):
        metrics = telemetry_robot._stream_overlay_metric_keys_for_step(
            self.rules,
            "align_brick",
        )
        self.assertEqual(
            metrics,
            ["visible", "xAxis_offset_abs", "yAxis_offset_abs", "dist"],
        )

    def test_falls_back_to_start_gates_when_success_gates_absent(self):
        metrics = telemetry_robot._stream_overlay_metric_keys_for_step(
            self.rules,
            "approach_vector_brick_supply",
        )
        self.assertEqual(metrics, ["visible"])

    def test_unknown_step_returns_empty_list(self):
        metrics = telemetry_robot._stream_overlay_metric_keys_for_step(
            self.rules,
            "missing_step",
        )
        self.assertEqual(metrics, [])


if __name__ == "__main__":
    unittest.main()
