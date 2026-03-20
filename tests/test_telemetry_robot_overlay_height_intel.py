import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


class TestTelemetryRobotOverlayHeightIntel(unittest.TestCase):
    def test_draw_overlay_omits_redundant_supply_stack_count_in_brick_telemetry(self):
        world = telemetry_robot.WorldModel()
        world.brick["visible"] = True
        world.brick["confidence"] = 95.0
        world.brick["x_axis"] = 0.0
        world.brick["offset_x"] = 0.0
        world.brick["y_axis"] = 0.0
        world.brick["offset_y"] = 0.0
        world.brick["angle"] = 0.0
        world.brick["dist"] = 120.0
        world.brick["brickAbove"] = False
        world.brick["brickBelow"] = False
        world.brick["inCrosshairs"] = False
        world.brick_supply_height_bricks = 7

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        lines = []
        telemetry_robot.draw_telemetry_overlay(
            frame,
            world,
            draw_text=False,
            line_sink=lines,
            gate_summary=[],
            gate_checker_summary=[],
            show_center_line=False,
            telemetry_step="FIND_TOPMOST_BRICK",
        )

        line_text = [entry.get("text") for entry in lines if isinstance(entry, dict)]
        self.assertNotIn("Supply stack: 7 bricks", line_text)
        self.assertIn("inCrosshairs: false", line_text)


if __name__ == "__main__":
    unittest.main()
