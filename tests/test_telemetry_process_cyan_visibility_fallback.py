import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.brick = {
            "visible": False,
            "dist": 0.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "offset_y": 0.0,
            "y_axis": 0.0,
            "confidence": 0.0,
            "brickAbove": None,
            "brickBelow": None,
        }
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.lift_height = 0.0
        self.height_mm = 145.0
        self.step_state = None
        self.wall = {"origin": None, "valid": False}

    def update_vision(
        self,
        found,
        dist,
        angle,
        conf,
        offset_x=0.0,
        cam_h=0.0,
        brick_above=False,
        brick_below=False,
    ):
        self.brick["visible"] = bool(found)
        self.brick["dist"] = float(dist)
        self.brick["angle"] = float(angle)
        self.brick["confidence"] = float(conf)
        self.brick["offset_x"] = float(offset_x)
        self.brick["x_axis"] = float(offset_x)
        self.brick["offset_y"] = float(cam_h)
        self.brick["y_axis"] = float(cam_h)
        self.brick["brickAbove"] = bool(brick_above)
        self.brick["brickBelow"] = bool(brick_below)


class _DummyCyanVision:
    def __init__(self, *, conf=12.0, found=True):
        self.conf_threshold = 0.15
        self.current_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self._found = bool(found)
        self._conf = float(conf)

    def read(self):
        return (
            bool(self._found),
            0.0,   # angle
            100.0, # dist
            0.0,   # offset_x
            float(self._conf),
            0.0,   # cam_h
            False,
            False,
        )


class TestTelemetryProcessCyanVisibilityFallback(unittest.TestCase):
    def test_cyan_raw_fallback_keeps_visible_during_warmup(self):
        world = _DummyWorld()
        vision = _DummyCyanVision(conf=12.0, found=True)
        telemetry_process.update_world_from_vision(world, vision, log=False)
        self.assertEqual(getattr(world, "_vision_backend", None), "cyan")
        self.assertTrue(bool(world.brick.get("visible")))
        self.assertGreater(float(world.brick.get("confidence", 0.0) or 0.0), 0.0)

    def test_cyan_raw_fallback_respects_relaxed_confidence_floor(self):
        world = _DummyWorld()
        vision = _DummyCyanVision(conf=3.0, found=True)
        telemetry_process.update_world_from_vision(world, vision, log=False)
        self.assertFalse(bool(world.brick.get("visible")))

    def test_aruco_raw_fallback_keeps_visible_during_warmup(self):
        world = _DummyWorld()
        vision = object.__new__(telemetry_process.ArucoBrickVision)
        vision.current_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        vision.read = lambda: (True, 0.0, 100.0, 0.0, 100.0, 0.0, False, False)
        telemetry_process.update_world_from_vision(world, vision, log=False)
        self.assertEqual(getattr(world, "_vision_backend", None), "aruco")
        self.assertTrue(bool(world.brick.get("visible")))

    def test_aruco_raw_fallback_respects_confidence_floor(self):
        world = _DummyWorld()
        vision = object.__new__(telemetry_process.ArucoBrickVision)
        vision.current_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        vision.read = lambda: (True, 0.0, 100.0, 0.0, 20.0, 0.0, False, False)
        telemetry_process.update_world_from_vision(world, vision, log=False)
        self.assertEqual(getattr(world, "_vision_backend", None), "aruco")
        self.assertFalse(bool(world.brick.get("visible")))

    def test_low_confidence_fallback_path_still_syncs_xyz_workspace(self):
        world = _DummyWorld()
        vision = _DummyCyanVision(conf=3.0, found=True)
        with mock.patch.object(telemetry_process.helper_xyz_coords, "sync_from_world") as sync_mock:
            telemetry_process.update_world_from_vision(world, vision, log=False)
        sync_mock.assert_called_once_with(world, reason="vision")


if __name__ == "__main__":
    unittest.main()
