"""Behaviour-tree helpers for aligning Leia to a brick on the x-axis.

Uses py_trees (the Python BehaviorTree.CPP / Nav2 equivalent) so navigation
decisions live in a declarative tree rather than custom if/else chains.

Tree structure (built by build_x_align_tree):

  Selector(AlignOrTurn, memory=False)
  ├── IsAligned            ← immediate SUCCESS when |x_mm| <= threshold
  └── Sequence(AlignSeq, memory=False)
      ├── IsBrickVisible   ← FAILURE if no brick reading on blackboard
      └── TurnTowardBrick  ← issues one turn pulse, RUNNING until aligned

Update the shared blackboard each camera frame before ticking:
    write_blackboard_x_mm(camera.bricks_telemetry[0]["x_mm"])
    tree.tick()
"""
from __future__ import annotations

import py_trees

ALIGN_X_THRESHOLD_MM = 10.0   # ±mm from centre counts as aligned
ALIGN_TURN_SPEED = 0.15       # conservative turn power (0–1 scale)
ALIGN_TURN_DURATION_MS = 300  # 0.3 s nudge per BT tick

_BB_NAMESPACE = "/"
_BB_KEY = "x_mm"


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

class IsBrickVisible(py_trees.behaviour.Behaviour):
    """Succeeds when x_mm is present and non-None on the blackboard."""

    def __init__(self) -> None:
        super().__init__(name="IsBrickVisible")
        self._bb = self.attach_blackboard_client(
            name=self.name, namespace=_BB_NAMESPACE
        )
        self._bb.register_key(_BB_KEY, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            val = self._bb.x_mm
        except KeyError:
            return py_trees.common.Status.FAILURE
        return (
            py_trees.common.Status.FAILURE
            if val is None
            else py_trees.common.Status.SUCCESS
        )


class IsAligned(py_trees.behaviour.Behaviour):
    """Succeeds when |x_mm| <= ALIGN_X_THRESHOLD_MM."""

    def __init__(self) -> None:
        super().__init__(name="IsAligned")
        self._bb = self.attach_blackboard_client(
            name=self.name, namespace=_BB_NAMESPACE
        )
        self._bb.register_key(_BB_KEY, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            val = self._bb.x_mm
        except KeyError:
            return py_trees.common.Status.FAILURE
        if val is None:
            return py_trees.common.Status.FAILURE
        return (
            py_trees.common.Status.SUCCESS
            if abs(float(val)) <= ALIGN_X_THRESHOLD_MM
            else py_trees.common.Status.FAILURE
        )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class TurnTowardBrick(py_trees.behaviour.Behaviour):
    """Issues one short turn pulse per tick until the brick is centred.

    Returns RUNNING while still misaligned, SUCCESS once within threshold,
    FAILURE if the brick reading disappears mid-turn.
    """

    def __init__(self, robot) -> None:
        super().__init__(name="TurnTowardBrick")
        self._robot = robot
        self._bb = self.attach_blackboard_client(
            name=self.name, namespace=_BB_NAMESPACE
        )
        self._bb.register_key(_BB_KEY, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            val = self._bb.x_mm
        except KeyError:
            return py_trees.common.Status.FAILURE
        if val is None:
            return py_trees.common.Status.FAILURE
        x = float(val)
        if abs(x) <= ALIGN_X_THRESHOLD_MM:
            return py_trees.common.Status.SUCCESS
        # x > 0 → brick right of centre → bot is LEFT of brick → turn RIGHT
        # x < 0 → brick left of centre → bot is RIGHT of brick → turn LEFT
        cmd = "r" if x > 0 else "l"
        self._robot.send_command(cmd, ALIGN_TURN_SPEED, duration_ms=ALIGN_TURN_DURATION_MS)
        return py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# Tree factory
# ---------------------------------------------------------------------------

def build_x_align_tree(robot) -> py_trees.trees.BehaviourTree:
    """Return a BehaviourTree that steers Leia's x-axis onto 0.

    Caller must update the blackboard with write_blackboard_x_mm() before
    each call to tree.tick().
    """
    align_seq = py_trees.composites.Sequence(name="AlignSeq", memory=False)
    align_seq.add_children([IsBrickVisible(), TurnTowardBrick(robot)])

    root = py_trees.composites.Selector(name="AlignOrTurn", memory=False)
    root.add_children([IsAligned(), align_seq])

    return py_trees.trees.BehaviourTree(root)


# ---------------------------------------------------------------------------
# Blackboard writer (call from camera loop before each tick)
# ---------------------------------------------------------------------------

_bb_writer: py_trees.blackboard.Client | None = None


def write_blackboard_x_mm(x_mm) -> None:
    """Write the latest camera x_mm reading to the shared blackboard."""
    global _bb_writer
    if _bb_writer is None:
        _bb_writer = py_trees.blackboard.Client(
            name="x_align_source", namespace=_BB_NAMESPACE
        )
        _bb_writer.register_key(_BB_KEY, access=py_trees.common.Access.WRITE)
    _bb_writer.x_mm = x_mm
