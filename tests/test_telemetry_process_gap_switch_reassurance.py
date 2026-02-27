import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.process_rules = {
            "POSITION_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -4.0, "tol": 1.4},
                    "yAxis_offset_abs": {"target": 2.0, "tol": 1.5},
                    "dist": {"target": 90.0, "tol": 1.5},
                    "brick_above": {"target": False, "tol": 0.0},
                }
            }
        }
        self._smoothed_frame_history = [
            {
                "frame_id": 101,
                "visible": True,
                "dist": 84.0,
                "angle": 0.0,
                "x_axis": -1.0,
                "offset_x": -1.0,
                "y_axis": 2.0,
                "offset_y": 2.0,
                "confidence": 90.0,
                "brick_above": True,
                "brick_below": False,
            },
            {
                "frame_id": 102,
                "visible": True,
                "dist": 84.0,
                "angle": 0.0,
                "x_axis": -1.0,
                "offset_x": -1.0,
                "y_axis": 2.0,
                "offset_y": 2.0,
                "confidence": 90.0,
                "brick_above": True,
                "brick_below": False,
            },
            {
                "frame_id": 103,
                "visible": True,
                "dist": 84.0,
                "angle": 0.0,
                "x_axis": -1.0,
                "offset_x": -1.0,
                "y_axis": 2.0,
                "offset_y": 2.0,
                "confidence": 90.0,
                "brick_above": True,
                "brick_below": False,
            },
        ]


class TestTelemetryProcessGapSwitchReassurance(unittest.TestCase):
    def test_reassurance_reports_switched_metric_outside_lite_gate(self):
        world = _DummyWorld()
        result = telemetry_process._gap_switch_lite_gate_reassurance(
            world,
            "POSITION_BRICK",
            "x_err",
        )
        metric_line = result.get("metric_plain") or ""
        self.assertIn("Reassurance", metric_line)
        self.assertIn("xAxis_offset_abs", metric_line)
        self.assertIn("outside success gate", metric_line)
        self.assertIn("state=", metric_line)
        self.assertIn("gate=", metric_line)

        snapshot_lines = result.get("snapshot_plain_lines") or []
        snapshot_text = "\n".join(snapshot_lines)
        self.assertIn("Success gate state/gate snapshot", snapshot_text)
        self.assertIn("visible state=true", snapshot_text)
        self.assertIn("xAxis_offset_abs state=-1.0mm", snapshot_text)
        self.assertIn("dist state=84.0mm", snapshot_text)
        self.assertIn("brick_above state=true", snapshot_text)

    def test_reassurance_handles_missing_success_gates(self):
        world = _DummyWorld()
        world.process_rules["POSITION_BRICK"] = {}
        result = telemetry_process._gap_switch_lite_gate_reassurance(
            world,
            "POSITION_BRICK",
            "x_err",
        )
        self.assertIn("no success gates configured", (result.get("metric_plain") or "").lower())
        self.assertEqual(result.get("snapshot_plain_lines"), ["Success gate state/gate snapshot: none"])


if __name__ == "__main__":
    unittest.main()
