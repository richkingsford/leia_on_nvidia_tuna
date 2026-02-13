import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_next
import telemetry_robot


class _DummyWorld:
    def __init__(self, x_axis, dist):
        self.brick = {
            "visible": True,
            "x_axis": float(x_axis),
            "offset_x": float(x_axis),
            "angle": 0.0,
            "dist": float(dist),
        }


class TestHelperNextTurnDistCautionPolicy(unittest.TestCase):
    def _align_rules(self):
        return {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -2.0, "tol": 1.0},
                    "dist": {"target": 98.0, "tol": 1.5},
                }
            }
        }

    def _position_rules(self):
        return {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": 0.0, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }

    def test_step4_turn_caps_to_2pct_when_dist_is_near_success_gate(self):
        world = _DummyWorld(17.0, dist=98.0)  # x-gap drives turn; dist is at gate
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._align_rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), telemetry_robot.normalize_speed_score(2))

    def test_step4_turn_uses_1pct_more_often_when_near_dist_and_modest_x_gap(self):
        # Mirrors ping-pong pattern: target=-2.3, observed x=+2.5 -> x-gap=4.8mm.
        rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -2.3, "tol": 0.7},
                    "dist": {"target": 98.0, "tol": 1.5},
                }
            }
        }
        world = _DummyWorld(2.5, dist=98.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), telemetry_robot.normalize_speed_score(1))

    def test_step4_turn_forces_1pct_when_x_gap_is_under_12mm(self):
        world = _DummyWorld(9.0, dist=108.0)  # target=-2.0 => gap=11mm
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._align_rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), telemetry_robot.normalize_speed_score(1))

    def test_step7_turn_caps_to_2pct_when_dist_is_near_success_gate(self):
        world = _DummyWorld(19.0, dist=48.0)  # x-gap drives turn; dist is at gate
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._position_rules(),
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), telemetry_robot.normalize_speed_score(2))

    def test_step7_turn_uses_1pct_more_often_when_near_dist_and_modest_x_gap(self):
        # Mirrors ping-pong pattern: target=-2.3, observed x=+2.5 -> x-gap=4.8mm.
        rules = {
            "POSITION_BRICK": {
                "max_speed_score": 10,
                "success_gates": {
                    "xAxis_offset_abs": {"target": -2.3, "tol": 0.7},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
            }
        }
        world = _DummyWorld(2.5, dist=48.0)
        analytics = helper_next.compute_alignment_analytics(
            world,
            rules,
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), telemetry_robot.normalize_speed_score(1))

    def test_step7_turn_forces_1pct_when_x_gap_is_under_12mm(self):
        world = _DummyWorld(11.0, dist=58.0)  # target=0.0 => gap=11mm
        analytics = helper_next.compute_alignment_analytics(
            world,
            self._position_rules(),
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertIn(analytics.get("cmd"), ("l", "r"))
        self.assertEqual(int(analytics.get("speed_score") or 0), telemetry_robot.normalize_speed_score(1))

    def test_step4_and_step7_share_same_turn_dist_caution_policy(self):
        align_near = helper_next.compute_alignment_analytics(
            _DummyWorld(17.0, dist=98.0),
            self._align_rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        position_near = helper_next.compute_alignment_analytics(
            _DummyWorld(19.0, dist=48.0),
            self._position_rules(),
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(int(align_near.get("speed_score") or 0), 2)
        self.assertEqual(int(position_near.get("speed_score") or 0), 2)

        align_far = helper_next.compute_alignment_analytics(
            _DummyWorld(17.0, dist=108.0),
            self._align_rules(),
            learned_rules={},
            step="ALIGN_BRICK",
            duration_s=0.05,
        )
        position_far = helper_next.compute_alignment_analytics(
            _DummyWorld(19.0, dist=58.0),
            self._position_rules(),
            learned_rules={},
            step="POSITION_BRICK",
            duration_s=0.05,
        )
        self.assertEqual(int(align_far.get("speed_score") or 0), 3)
        self.assertEqual(int(position_far.get("speed_score") or 0), 3)


if __name__ == "__main__":
    unittest.main()
