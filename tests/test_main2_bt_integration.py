"""Integration test: NvidiaCameraLivestream._tick_bt drives the alignment BT.

Tests the full path from camera telemetry → blackboard → BT decision → robot command.
Uses object.__new__ to build the camera without opening real hardware.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import py_trees

from helper_bt_align import (
    ALIGN_X_THRESHOLD_MM,
    _BB_KEY,
    _BB_NAMESPACE,
    build_x_align_tree,
)
from main2 import NvidiaCameraLivestream

_TURN_STEP_MM = 8.0


class _MockRobot:
    def __init__(self, x_mm: float) -> None:
        self.x_mm = float(x_mm)
        self.commands: list[tuple] = []

    def send_command(self, cmd_char: str, speed: float, duration_ms=None):
        self.commands.append((cmd_char, speed, duration_ms))
        if cmd_char == "r":
            self.x_mm -= _TURN_STEP_MM
        elif cmd_char == "l":
            self.x_mm += _TURN_STEP_MM
        return {"cmd_sent": cmd_char}


def _make_camera(robot) -> NvidiaCameraLivestream:
    """Construct NvidiaCameraLivestream with a robot but no real camera hardware."""
    cam = object.__new__(NvidiaCameraLivestream)
    cam._robot = robot
    cam._bricks_telemetry = []
    cam._bt_status = ""
    if robot is not None:
        cam._bt_tree = build_x_align_tree(robot)
        cam._bt_writer = py_trees.blackboard.Client(
            name="leia_camera_test", namespace=_BB_NAMESPACE
        )
        cam._bt_writer.register_key(_BB_KEY, access=py_trees.common.Access.WRITE)
    else:
        cam._bt_tree = None
        cam._bt_writer = None
    cam._bt_tick_interval = 0.0  # no cooldown in tests
    cam._bt_last_tick_at = 0.0
    return cam


class TestBTLiveIntegration(unittest.TestCase):
    def setUp(self) -> None:
        py_trees.blackboard.Blackboard.clear()

    # ------------------------------------------------------------------
    # Single-tick decisions
    # ------------------------------------------------------------------

    def test_tick_issues_right_turn_for_positive_x(self) -> None:
        """x_mm=+40 (bot left of brick) → _tick_bt issues 'r'."""
        robot = _MockRobot(x_mm=40.0)
        cam = _make_camera(robot)
        cam._bricks_telemetry = [{"x_mm": 40.0, "y_mm": 0.0, "dist_mm": 600.0}]

        cam._tick_bt()

        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0][0], "r")
        self.assertEqual(cam._bt_status, "RUNNING")

    def test_tick_issues_left_turn_for_negative_x(self) -> None:
        """x_mm=-40 (bot right of brick) → _tick_bt issues 'l'."""
        robot = _MockRobot(x_mm=-40.0)
        cam = _make_camera(robot)
        cam._bricks_telemetry = [{"x_mm": -40.0, "y_mm": 0.0, "dist_mm": 600.0}]

        cam._tick_bt()

        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0][0], "l")
        self.assertEqual(cam._bt_status, "RUNNING")

    def test_tick_no_command_when_already_aligned(self) -> None:
        """x_mm within threshold → no command, status SUCCESS."""
        robot = _MockRobot(x_mm=5.0)
        cam = _make_camera(robot)
        cam._bricks_telemetry = [{"x_mm": 5.0, "y_mm": 0.0, "dist_mm": 600.0}]

        cam._tick_bt()

        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(cam._bt_status, "SUCCESS")

    def test_tick_no_command_when_no_brick_visible(self) -> None:
        """Empty telemetry → FAILURE, no command."""
        robot = _MockRobot(x_mm=40.0)
        cam = _make_camera(robot)
        cam._bricks_telemetry = []

        cam._tick_bt()

        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(cam._bt_status, "FAILURE")

    # ------------------------------------------------------------------
    # Multi-tick loop (simulates camera frame loop)
    # ------------------------------------------------------------------

    def test_loop_aligns_from_left_of_brick(self) -> None:
        """Simulated frame loop: start at x_mm=+50, tick until SUCCESS, only 'r' turns."""
        robot = _MockRobot(x_mm=50.0)
        cam = _make_camera(robot)

        for _ in range(20):
            cam._bricks_telemetry = [{"x_mm": robot.x_mm, "y_mm": 0.0, "dist_mm": 600.0}]
            cam._tick_bt()
            if cam._bt_status == "SUCCESS":
                break

        self.assertEqual(cam._bt_status, "SUCCESS")
        self.assertLessEqual(abs(robot.x_mm), ALIGN_X_THRESHOLD_MM)
        self.assertTrue(all(c == "r" for c, _, _ in robot.commands))

    # ------------------------------------------------------------------
    # No-robot path
    # ------------------------------------------------------------------

    def test_tick_is_noop_without_robot(self) -> None:
        """Camera created without robot must not raise and status stays empty."""
        cam = _make_camera(robot=None)
        cam._bricks_telemetry = [{"x_mm": 50.0, "y_mm": 0.0, "dist_mm": 600.0}]

        cam._tick_bt()  # must not raise

        self.assertEqual(cam._bt_status, "")


if __name__ == "__main__":
    unittest.main()
