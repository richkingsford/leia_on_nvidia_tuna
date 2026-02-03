import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_next
import telemetry_robot


class _DummyWorld:
    def __init__(self, x_axis):
        self.brick = {
            "visible": True,
            "x_axis": float(x_axis),
            "offset_x": float(x_axis),
            "angle": 0.0,
            "dist": 80.0,
        }


class TestHelperNextAlignTurnSpeed(unittest.TestCase):
    def _rules(self):
        return {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -2.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 1.0},
                }
            }
        }

    def test_align_turn_uses_3pct_when_offset_above_9mm(self):
        world = _DummyWorld(10.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(3))

    def test_align_turn_uses_1pct_when_offset_at_or_below_9mm(self):
        world = _DummyWorld(9.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(1))


if __name__ == "__main__":
    unittest.main()
