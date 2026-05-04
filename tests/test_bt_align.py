"""Test the x-axis alignment behaviour tree (BehaviorTree.CPP / Nav2 style).

Scenario exercised here:
  - Robot starts slightly to the LEFT of a brick.
  - Camera sees x_mm = +50  (brick appears 50 mm right of frame centre).
  - BT should issue 'r' (turn right) pulses until |x_mm| <= threshold.
  - Tree must report SUCCESS once aligned.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import py_trees

from helper_bt_align import (
    ALIGN_TURN_DURATION_MS,
    ALIGN_TURN_SPEED,
    ALIGN_X_THRESHOLD_MM,
    build_x_align_tree,
)

# Simulated mm moved per 80 ms turn pulse (conservative estimate)
_TURN_STEP_MM = 8.0


class _MockRobot:
    """Minimal robot stub that records commands and simulates x_mm drift."""

    def __init__(self, x_mm: float) -> None:
        self.x_mm = float(x_mm)
        self.commands: list[tuple] = []

    def send_command(self, cmd_char: str, speed: float, duration_ms=None):
        self.commands.append((cmd_char, speed, duration_ms))
        if cmd_char == "r":
            self.x_mm -= _TURN_STEP_MM
        elif cmd_char == "l":
            self.x_mm += _TURN_STEP_MM
        return {"cmd_sent": cmd_char, "pwm": 100, "power": speed, "duration_ms": duration_ms}


def _make_writer() -> py_trees.blackboard.Client:
    writer = py_trees.blackboard.Client(name="test_writer", namespace="/")
    writer.register_key("x_mm", access=py_trees.common.Access.WRITE)
    return writer


def _tick_until_done(
    tree: py_trees.trees.BehaviourTree,
    robot: _MockRobot,
    writer: py_trees.blackboard.Client,
    max_ticks: int = 50,
) -> list[py_trees.common.Status]:
    statuses = []
    for _ in range(max_ticks):
        writer.x_mm = robot.x_mm
        tree.tick()
        status = tree.root.status
        statuses.append(status)
        if status != py_trees.common.Status.RUNNING:
            break
    return statuses


class TestXAlignBT(unittest.TestCase):
    def setUp(self) -> None:
        # Wipe all keys, values, and client registrations between tests.
        py_trees.blackboard.Blackboard.clear()

    # ------------------------------------------------------------------
    # Core scenario: robot left of brick
    # ------------------------------------------------------------------

    def test_turns_right_when_left_of_brick(self) -> None:
        """x_mm=+50 (bot left of brick) → BT issues only 'r' commands."""
        robot = _MockRobot(x_mm=50.0)
        tree = build_x_align_tree(robot)
        writer = _make_writer()

        statuses = _tick_until_done(tree, robot, writer)

        self.assertEqual(statuses[-1], py_trees.common.Status.SUCCESS)
        cmds = [c for c, _, _ in robot.commands]
        self.assertTrue(
            all(c == "r" for c in cmds),
            f"Expected only 'r' turns, got: {cmds}",
        )
        self.assertLessEqual(abs(robot.x_mm), ALIGN_X_THRESHOLD_MM)

    def test_aligns_within_reasonable_tick_count(self) -> None:
        """50 mm offset at 8 mm/step needs ~6 steps; must finish in ≤10 ticks."""
        robot = _MockRobot(x_mm=50.0)
        tree = build_x_align_tree(robot)
        writer = _make_writer()

        statuses = _tick_until_done(tree, robot, writer, max_ticks=20)

        self.assertEqual(statuses[-1], py_trees.common.Status.SUCCESS)
        # 50 / 8 = 6.25 → 7 turn ticks + 1 final SUCCESS tick = 8 ticks max
        self.assertLessEqual(len(robot.commands), 10)

    # ------------------------------------------------------------------
    # Symmetric / boundary cases
    # ------------------------------------------------------------------

    def test_no_turn_needed_when_already_aligned(self) -> None:
        """x_mm within threshold → immediate SUCCESS, zero commands issued."""
        robot = _MockRobot(x_mm=5.0)
        tree = build_x_align_tree(robot)
        writer = _make_writer()

        statuses = _tick_until_done(tree, robot, writer, max_ticks=3)

        self.assertEqual(statuses[0], py_trees.common.Status.SUCCESS)
        self.assertEqual(len(robot.commands), 0)

    def test_turns_left_when_right_of_brick(self) -> None:
        """x_mm=-30 (bot right of brick) → all commands are 'l'."""
        robot = _MockRobot(x_mm=-30.0)
        tree = build_x_align_tree(robot)
        writer = _make_writer()

        statuses = _tick_until_done(tree, robot, writer)

        self.assertEqual(statuses[-1], py_trees.common.Status.SUCCESS)
        cmds = [c for c, _, _ in robot.commands]
        self.assertTrue(
            all(c == "l" for c in cmds),
            f"Expected only 'l' turns, got: {cmds}",
        )

    def test_failure_when_no_brick_visible(self) -> None:
        """x_mm=None (brick not detected) → tree returns FAILURE immediately."""
        robot = _MockRobot(x_mm=50.0)
        tree = build_x_align_tree(robot)
        writer = _make_writer()

        writer.x_mm = None
        tree.tick()

        self.assertEqual(tree.root.status, py_trees.common.Status.FAILURE)
        self.assertEqual(len(robot.commands), 0)

    # ------------------------------------------------------------------
    # Command parameter contract
    # ------------------------------------------------------------------

    def test_turn_command_uses_correct_speed_and_duration(self) -> None:
        """First turn pulse must use ALIGN_TURN_SPEED and ALIGN_TURN_DURATION_MS."""
        robot = _MockRobot(x_mm=50.0)
        tree = build_x_align_tree(robot)
        writer = _make_writer()

        writer.x_mm = robot.x_mm
        tree.tick()

        self.assertEqual(len(robot.commands), 1)
        _cmd, speed, duration_ms = robot.commands[0]
        self.assertAlmostEqual(speed, ALIGN_TURN_SPEED)
        self.assertEqual(duration_ms, ALIGN_TURN_DURATION_MS)


if __name__ == "__main__":
    unittest.main()
