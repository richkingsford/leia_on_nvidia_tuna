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
            "inCrosshairs": False,
            "brickAbove": None,
            "brickBelow": None,
        }
        self.learned_rules = {}
        self.process_rules = {}
        self.step_state = "FIND_BRICK"
        self.scoop_forward_preferred = False
        self.scoop_desired_offset_x = 0.0
        self.scoop_lateral_drift = 0.0
        self.scoop_success_offset_factor = 1.0
        self.camera_height_anchor = None
        self.height_mm = None
        self.align_tol_offset = 12.0
        self.align_tol_angle = 5.0
        self.align_tol_dist_min = 30.0
        self.align_tol_dist_max = 500.0
        self.stability_count = 0
        self.last_visible_time = None


class TestTelemetryBrickInCrosshairs(unittest.TestCase):
    def test_sets_true_when_visible_and_offsets_within_half_brick_extents(self):
        world = _DummyWorld()
        telemetry_brick.update_from_vision(
            world,
            found=True,
            dist=100.0,
            angle=0.0,
            conf=90.0,
            offset_x=5.0,
            cam_h=4.0,
            brick_above=False,
            brick_below=False,
        )
        self.assertTrue(world.brick.get("inCrosshairs"))

    def test_sets_false_when_x_offset_outside_half_brick_width(self):
        world = _DummyWorld()
        telemetry_brick.update_from_vision(
            world,
            found=True,
            dist=100.0,
            angle=0.0,
            conf=90.0,
            offset_x=50.0,
            cam_h=0.0,
            brick_above=False,
            brick_below=False,
        )
        self.assertFalse(world.brick.get("inCrosshairs"))

    def test_sets_false_when_marker_center_is_outside_marker_edge_overlap(self):
        world = _DummyWorld()
        telemetry_brick.update_from_vision(
            world,
            found=True,
            dist=100.0,
            angle=0.0,
            conf=90.0,
            offset_x=20.0,
            cam_h=0.0,
            brick_above=False,
            brick_below=False,
        )
        self.assertFalse(world.brick.get("inCrosshairs"))

    def test_sets_false_when_not_visible(self):
        world = _DummyWorld()
        telemetry_brick.update_from_vision(
            world,
            found=False,
            dist=0.0,
            angle=0.0,
            conf=0.0,
            offset_x=0.0,
            cam_h=0.0,
            brick_above=False,
            brick_below=False,
        )
        self.assertFalse(world.brick.get("inCrosshairs"))

    def test_step_specific_in_crosshairs_center_override_is_applied(self):
        world = _DummyWorld()
        world.step_state = "FIND_TOPMOST_BRICK"
        world.process_rules = {
            "FIND_TOPMOST_BRICK": {
                "in_crosshairs_center_mm": {"x": 0.0, "y": -12.0}
            }
        }
        telemetry_brick.update_from_vision(
            world,
            found=True,
            dist=100.0,
            angle=0.0,
            conf=90.0,
            offset_x=0.0,
            cam_h=0.0,
            brick_above=False,
            brick_below=False,
        )
        self.assertFalse(world.brick.get("inCrosshairs"))


if __name__ == "__main__":
    unittest.main()
