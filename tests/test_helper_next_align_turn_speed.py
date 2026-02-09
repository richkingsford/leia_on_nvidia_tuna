import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_next
import telemetry_robot


class _DummyWorld:
    def __init__(self, x_axis, dist=80.0):
        self.brick = {
            "visible": True,
            "x_axis": float(x_axis),
            "offset_x": float(x_axis),
            "angle": 0.0,
            "dist": float(dist),
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

    def test_align_turn_uses_1pct_when_abs_x_axis_at_or_below_9mm(self):
        world = _DummyWorld(4.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(1))

    def test_align_turn_uses_1pct_when_abs_x_axis_is_9mm(self):
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

    def test_align_turn_uses_2pct_when_abs_x_axis_between_9_and_16mm(self):
        world = _DummyWorld(10.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(2))

    def test_align_turn_uses_3pct_when_abs_x_axis_above_16mm(self):
        world = _DummyWorld(20.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(analytics.get("speed_score"), telemetry_robot.normalize_speed_score(3))

    def test_align_loosened_x_axis_tolerance_when_far_keeps_dist_priority_for_minor_x_error(self):
        world = _DummyWorld(1.5, dist=420.0)  # slightly outside tol=1.0
        rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 400.0, "tol": 5.0},
                }
            }
        }
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertFalse(getattr(world, "_align_focus_dist", False))

    def test_align_prioritizes_dist_when_gap_exceeds_150mm_even_if_x_axis_off(self):
        world = _DummyWorld(20.0, dist=300.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertEqual(analytics.get("cmd"), "f")
        self.assertTrue(getattr(world, "_align_focus_dist", False))

    def test_align_dist_focus_persists_until_gap_below_100mm(self):
        world = _DummyWorld(20.0, dist=300.0)

        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertTrue(getattr(world, "_align_focus_dist", False))

        world.brick["dist"] = 190.0  # gap ~110mm: keep focusing dist
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "dist")
        self.assertTrue(getattr(world, "_align_focus_dist", False))

        world.brick["dist"] = 170.0  # gap ~90mm: release focus and return to x-axis priority
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(analytics.get("worst_metric"), "xAxis_offset")
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertFalse(getattr(world, "_align_focus_dist", False))


if __name__ == "__main__":
    unittest.main()
