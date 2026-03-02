import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class _DummyWorld:
    def __init__(self):
        self.lift_height = 10.0
        self.lift_height_anchor = None
        self.lift_height_source = "dead_reckon"
        self.lift_height_quality = 0.0


class TestTelemetryRobotLiftFloorRoi(unittest.TestCase):
    def test_floor_roi_updates_lift_when_quality_is_good(self):
        world = _DummyWorld()
        telemetry_robot.update_lift_from_vision(
            world,
            cam_h=0.0,
            brick_height=0.0,
            conf=0.0,
            floor_lift_mm=30.0,
            floor_lift_quality=0.8,
        )
        self.assertAlmostEqual(float(world.lift_height), 14.0, places=3)
        self.assertEqual(str(world.lift_height_source), "tiny_roi_floor")
        self.assertGreaterEqual(float(world.lift_height_quality), 0.8)

    def test_floor_roi_ignored_when_quality_too_low(self):
        world = _DummyWorld()
        telemetry_robot.update_lift_from_vision(
            world,
            cam_h=0.0,
            brick_height=0.0,
            conf=0.0,
            floor_lift_mm=30.0,
            floor_lift_quality=0.1,
        )
        self.assertAlmostEqual(float(world.lift_height), 10.0, places=6)
        self.assertEqual(str(world.lift_height_source), "dead_reckon")

    def test_marker_path_still_available_without_floor_roi(self):
        world = _DummyWorld()
        telemetry_robot.update_lift_from_vision(
            world,
            cam_h=100.0,
            brick_height=0.0,
            conf=80.0,
        )
        self.assertIsNotNone(world.lift_height_anchor)
        telemetry_robot.update_lift_from_vision(
            world,
            cam_h=110.0,
            brick_height=0.0,
            conf=80.0,
        )
        self.assertGreater(float(world.lift_height), 10.0)
        self.assertEqual(str(world.lift_height_source), "aruco_cam_h")


if __name__ == "__main__":
    unittest.main()
