import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_wall


class _DummyWorld:
    def __init__(self, *, wall_steps=None):
        self.wall_model = {"steps": wall_steps or {}}
        self.wall = {
            "origin": None,
            "valid": False,
            "contradiction_reason": None,
        }
        self.brick = {
            "visible": True,
            "dist": 90.0,
            "angle": 0.0,
        }
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def compute_brick_world_xy(self, dist, angle):
        return float(dist or 0.0), float(angle or 0.0)


class TestTelemetryWallGateRequirements(unittest.TestCase):
    def setUp(self):
        self.envelope = telemetry_wall.WallEnvelope(
            angle_deg=0.0,
            min_confidence=80.0,
            max_origin_drift_mm=75.0,
            max_angle_drift_deg=10.0,
            place_offset_mm=25.0,
            allow_auto_origin=True,
            lock_step="FIND_WALL",
            origin=None,
        )

    def test_retreat_requires_wall_origin_by_default(self):
        world = _DummyWorld()
        check = telemetry_wall.evaluate_start_gates(world, "RETREAT", self.envelope)
        self.assertFalse(check.ok)
        self.assertIn("wall origin unset", check.reasons)

    def test_retreat_can_disable_wall_origin_requirement_in_wall_model(self):
        world = _DummyWorld(wall_steps={"RETREAT": {"requires_wall_origin": False}})
        start_check = telemetry_wall.evaluate_start_gates(world, "RETREAT", self.envelope)
        failure_check = telemetry_wall.evaluate_failure_gates(world, "RETREAT", self.envelope)
        success_check = telemetry_wall.evaluate_success_gates(world, "RETREAT", self.envelope)
        self.assertTrue(start_check.ok)
        self.assertTrue(failure_check.ok)
        self.assertTrue(success_check.ok)

    def test_position_brick_can_disable_wall_origin_requirement_in_wall_model(self):
        world = _DummyWorld(wall_steps={"POSITION_BRICK": {"requires_wall_origin": False}})
        start_check = telemetry_wall.evaluate_start_gates(world, "POSITION_BRICK", self.envelope)
        failure_check = telemetry_wall.evaluate_failure_gates(world, "POSITION_BRICK", self.envelope)
        success_check = telemetry_wall.evaluate_success_gates(world, "POSITION_BRICK", self.envelope)
        self.assertTrue(start_check.ok)
        self.assertTrue(failure_check.ok)
        self.assertTrue(success_check.ok)

    def test_wall_origin_requirement_note_reports_disabled_override(self):
        world = _DummyWorld(wall_steps={"POSITION_BRICK": {"requires_wall_origin": False}})
        note = telemetry_wall.wall_origin_requirement_note(world, "POSITION_BRICK")
        self.assertEqual(note, "wall origin requirement disabled by wall model")

    def test_wall_origin_requirement_note_reports_none_when_not_overridden(self):
        world = _DummyWorld()
        note = telemetry_wall.wall_origin_requirement_note(world, "POSITION_BRICK")
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()
