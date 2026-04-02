import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_brick


class _DummyWorld:
    def __init__(self):
        self.brick = {
            "visible": True,
            "confidence": 90.0,
            "dist": 103.6,
            "raw_dist": 219.8,
            "angle": 0.0,
            "offset_x": 7.1,
            "x_axis": 7.1,
            "offset_y": 3.8,
            "y_axis": 3.8,
            "brickAbove": False,
            "brickBelow": False,
            "inCrosshairs": True,
        }
        self._brick_frame_buffer = []
        self.learned_rules = {}
        self.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 7.13, "tol": 1.4},
                    "yAxis_offset_abs": {"target": 3.74, "tol": 2.3},
                    "dist": {"target": 103.61, "tol": 2.3},
                }
            }
        }
        self.last_visible_time = None


class TestTelemetryBrickMetricValue(unittest.TestCase):
    def test_dist_metric_uses_corrected_distance_not_raw_distance(self):
        brick = {"dist": 103.6, "raw_dist": 219.8}

        value = telemetry_brick.metric_value(brick, "dist")

        self.assertAlmostEqual(value, 103.6, places=3)

    def test_success_gate_eval_uses_corrected_distance_when_raw_distance_present(self):
        world = _DummyWorld()

        check = telemetry_brick.evaluate_success_gates(
            world,
            "ALIGN_BRICK",
            learned_rules={},
            process_rules=world.process_rules,
        )
        entries = telemetry_brick.success_gate_entries(
            world,
            "ALIGN_BRICK",
            learned_rules={},
            process_rules=world.process_rules,
        )
        dist_entry = next(entry for entry in entries if entry.get("metric") == "dist")

        self.assertTrue(check.ok, msg=check.reason_str())
        self.assertAlmostEqual(float(dist_entry.get("value")), 103.6, places=3)
        self.assertAlmostEqual(float(dist_entry.get("raw_value")), 103.6, places=3)


if __name__ == "__main__":
    unittest.main()
