import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_brick


class _DummyWorld:
    def __init__(self):
        self.brick = {
            "visible": False,
            "confidence": 0.0,
            "dist": 0.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "offset_y": 0.0,
            "y_axis": 0.0,
            "brickAbove": None,
            "brickBelow": None,
        }
        self._brick_frame_buffer = []
        self._smoothed_frame_history = []
        self.learned_rules = {}
        self.process_rules = {
            "BRICK_LOCK_WALL": {
                "start_gates": {"visible": {"min": True}},
            }
        }


class TestTelemetryBrickStartGateVisibleFallback(unittest.TestCase):
    def test_visible_start_gate_uses_recent_raw_frame_when_smoothed_visible_false(self):
        world = _DummyWorld()
        world._brick_frame_buffer = [
            {
                "found": True,
                "conf": 90.0,
                "dist": 100.0,
                "angle": 0.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "cam_h": 0.0,
                "brick_above": False,
                "brick_below": False,
            }
        ]
        check = telemetry_brick.evaluate_start_gates(
            world,
            "BRICK_LOCK_WALL",
            learned_rules={},
            process_rules=world.process_rules,
        )
        self.assertTrue(check.ok, msg=check.reason_str())

    def test_visible_start_gate_still_fails_without_recent_raw_visibility(self):
        world = _DummyWorld()
        world._brick_frame_buffer = [
            {
                "found": False,
                "conf": 0.0,
                "dist": 0.0,
                "angle": 0.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "cam_h": 0.0,
                "brick_above": False,
                "brick_below": False,
            }
        ]
        check = telemetry_brick.evaluate_start_gates(
            world,
            "BRICK_LOCK_WALL",
            learned_rules={},
            process_rules=world.process_rules,
        )
        self.assertFalse(check.ok)
        self.assertIn("visible gate", check.reasons)

    def test_smoothed_snapshot_x_axis_matches_world_sign_convention(self):
        world = _DummyWorld()
        world._brick_frame_buffer = [
            {
                "found": True,
                "conf": 90.0,
                "dist": 120.0,
                "angle": 0.0,
                "offset_x": 8.0,
                "offset_y": 0.0,
                "cam_h": 0.0,
                "brick_above": False,
                "brick_below": False,
            },
            {
                "found": True,
                "conf": 92.0,
                "dist": 121.0,
                "angle": 0.0,
                "offset_x": 10.0,
                "offset_y": 0.0,
                "cam_h": 0.0,
                "brick_above": False,
                "brick_below": False,
            },
            {
                "found": True,
                "conf": 93.0,
                "dist": 119.5,
                "angle": 0.0,
                "offset_x": 9.0,
                "offset_y": 0.0,
                "cam_h": 0.0,
                "brick_above": False,
                "brick_below": False,
            },
            {
                "found": True,
                "conf": 91.0,
                "dist": 120.5,
                "angle": 0.0,
                "offset_x": 9.0,
                "offset_y": 0.0,
                "cam_h": 0.0,
                "brick_above": False,
                "brick_below": False,
            },
        ]

        snapshot = telemetry_brick.smoothed_brick_snapshot(world)

        self.assertAlmostEqual(float(snapshot["x_axis"]), -9.0, places=3)
        self.assertAlmostEqual(float(snapshot["offset_x"]), -9.0, places=3)

    def test_smoothed_snapshot_prefers_corrected_smoothed_history_over_raw_frame_buffer(self):
        world = _DummyWorld()
        world.brick.update(
            {
                "visible": True,
                "dist": 103.6,
                "x_axis": 7.1,
                "offset_x": 7.1,
                "y_axis": 3.8,
                "offset_y": 3.8,
                "inCrosshairs": True,
            }
        )
        world._brick_frame_buffer = [
            {
                "found": True,
                "conf": 90.0,
                "dist": 220.0,
                "angle": 0.0,
                "offset_x": 8.0,
                "offset_y": 110.0,
                "cam_h": 110.0,
                "brick_above": False,
                "brick_below": False,
            }
        ]
        world._smoothed_frame_history = [
            {
                "frame_id": 1,
                "visible": True,
                "dist": 103.5,
                "angle": 0.0,
                "x_axis": 7.0,
                "offset_x": 7.0,
                "y_axis": 3.7,
                "offset_y": 3.7,
                "confidence": 91.0,
                "brick_above": False,
                "brick_below": False,
            },
            {
                "frame_id": 2,
                "visible": True,
                "dist": 103.7,
                "angle": 0.0,
                "x_axis": 7.2,
                "offset_x": 7.2,
                "y_axis": 3.9,
                "offset_y": 3.9,
                "confidence": 92.0,
                "brick_above": False,
                "brick_below": False,
            },
        ]

        snapshot = telemetry_brick.smoothed_brick_snapshot(world)

        self.assertAlmostEqual(float(snapshot["dist"]), 103.6, places=3)
        self.assertAlmostEqual(float(snapshot["x_axis"]), 7.1, places=3)
        self.assertAlmostEqual(float(snapshot["y_axis"]), 3.8, places=3)
        self.assertTrue(bool(snapshot["inCrosshairs"]))


if __name__ == "__main__":
    unittest.main()
