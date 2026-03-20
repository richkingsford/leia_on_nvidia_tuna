import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class _DummyWorld:
    def __init__(self):
        self.lift_height = 10.0
        self.lift_height_anchor = None
        self.lift_height_source = "dead_reckon"
        self.lift_height_quality = 0.0


class _DummyBrickWorld:
    def __init__(self):
        self.process_rules = {}
        self.learned_rules = {}
        self.step_state = None
        self.wall_envelope = None
        self.align_tol_offset = 12.0
        self.align_tol_angle = 5.0
        self.align_tol_dist_min = 30.0
        self.align_tol_dist_max = 500.0
        self.scoop_success_offset_factor = 1.2
        self.stability_count = 0
        self.brick = {
            "visible": False,
            "dist": 0.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "offset_y": 0.0,
            "y_axis": 0.0,
            "inCrosshairs": False,
            "confidence": 0.0,
            "held": False,
            "brickAbove": None,
            "brickBelow": None,
        }
        self.last_visible_time = None
        self.last_seen_angle = None
        self.last_seen_offset_x = None
        self.last_seen_x_axis = None
        self.last_seen_offset_y = None
        self.last_seen_y_axis = None
        self.last_seen_dist = None
        self.last_seen_confidence = None
        self.scoop_forward_preferred = False
        self.scoop_desired_offset_x = 0.0
        self.scoop_lateral_drift = 0.0
        self.camera_height_anchor = None
        self.height_mm = None


class _DummyWallModule:
    @staticmethod
    def update_from_vision(_world, _found, _dist, _angle, _conf, _wall_envelope):
        return None


class TestTelemetryRobotLiftUpdate(unittest.TestCase):
    def test_update_vision_passes_cam_h_into_brick_y_axis(self):
        world = _DummyBrickWorld()

        with patch.object(telemetry_robot, "update_lift_from_vision", return_value=None), \
             patch.object(telemetry_robot, "_wall_module", return_value=_DummyWallModule()), \
             patch.object(telemetry_robot.helper_xyz_coords, "sync_from_world", return_value=None):
            telemetry_robot.WorldModel.update_vision(
                world,
                found=True,
                dist=120.0,
                angle=0.0,
                conf=90.0,
                offset_x=0.0,
                cam_h=26.5,
                brick_above=True,
                brick_below=False,
            )

        self.assertAlmostEqual(float(world.brick["y_axis"]), 26.5, places=6)
        self.assertAlmostEqual(float(world.brick["offset_y"]), 26.5, places=6)
        self.assertAlmostEqual(float(world.last_seen_y_axis), 26.5, places=6)
        self.assertFalse(world.brick["inCrosshairs"])

    def test_low_confidence_path_keeps_dead_reckon_source(self):
        world = _DummyWorld()
        telemetry_robot.update_lift_from_vision(
            world,
            cam_h=0.0,
            brick_height=0.0,
            conf=0.0,
        )
        self.assertAlmostEqual(float(world.lift_height), 10.0, places=6)
        self.assertEqual(str(world.lift_height_source), "dead_reckon")
        self.assertEqual(float(world.lift_height_quality), 0.0)

    def test_marker_path_updates_lift_height(self):
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
