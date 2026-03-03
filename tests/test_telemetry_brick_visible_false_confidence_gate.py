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
        self.last_visible_time = None


class TestTelemetryBrickVisibleFalseConfidenceGate(unittest.TestCase):
    def test_visible_false_gate_fails_when_brick_seen_at_or_above_70_confidence(self):
        world = _DummyWorld()
        world.brick["visible"] = True
        world.brick["confidence"] = 70.0
        process_rules = {"EXIT_WALL": {"success_gates": {"visible": {"min": False}}}}

        check = telemetry_brick.evaluate_success_gates(
            world,
            "EXIT_WALL",
            learned_rules={},
            process_rules=process_rules,
        )

        self.assertFalse(check.ok)
        self.assertIn("visible gate", check.reasons)

    def test_visible_false_gate_fails_when_seen_below_70_confidence(self):
        world = _DummyWorld()
        world.brick["visible"] = True
        world.brick["confidence"] = 69.9
        process_rules = {"EXIT_WALL": {"success_gates": {"visible": {"min": False}}}}

        check = telemetry_brick.evaluate_success_gates(
            world,
            "EXIT_WALL",
            learned_rules={},
            process_rules=process_rules,
        )

        self.assertFalse(check.ok)
        self.assertIn("visible gate", check.reasons)

    def test_visible_false_gate_passes_when_not_visible(self):
        world = _DummyWorld()
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        process_rules = {"EXIT_WALL": {"success_gates": {"visible": {"min": False}}}}

        check = telemetry_brick.evaluate_success_gates(
            world,
            "EXIT_WALL",
            learned_rules={},
            process_rules=process_rules,
        )

        self.assertTrue(check.ok, msg=check.reason_str())

    def test_visible_false_gate_fails_when_recent_raw_streak_is_confident(self):
        world = _DummyWorld()
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world._raw_brick_visibility_history = [
            {"frame_id": 1, "found": True, "conf": 90.0},
            {"frame_id": 2, "found": True, "conf": 92.0},
            {"frame_id": 3, "found": True, "conf": 95.0},
        ]
        process_rules = {"EXIT_WALL": {"success_gates": {"visible": {"min": False}}}}

        check = telemetry_brick.evaluate_success_gates(
            world,
            "EXIT_WALL",
            learned_rules={},
            process_rules=process_rules,
        )

        self.assertFalse(check.ok)
        self.assertIn("visible gate", check.reasons)

    def test_visible_false_gate_fails_with_three_recent_confident_hits_even_with_one_miss(self):
        world = _DummyWorld()
        world.brick["visible"] = False
        world.brick["confidence"] = 0.0
        world._raw_brick_visibility_history = [
            {"frame_id": 1, "found": True, "conf": 91.0},
            {"frame_id": 2, "found": False, "conf": 0.0},
            {"frame_id": 3, "found": True, "conf": 93.0},
            {"frame_id": 4, "found": True, "conf": 95.0},
        ]
        process_rules = {"EXIT_WALL": {"success_gates": {"visible": {"min": False}}}}

        check = telemetry_brick.evaluate_success_gates(
            world,
            "EXIT_WALL",
            learned_rules={},
            process_rules=process_rules,
        )

        self.assertFalse(check.ok)
        self.assertIn("visible gate", check.reasons)

    def test_visible_true_gate_remains_strict(self):
        world = _DummyWorld()
        world.brick["visible"] = False
        world.brick["confidence"] = 99.0
        process_rules = {"FIND_WALL2": {"success_gates": {"visible": {"min": True}}}}

        check = telemetry_brick.evaluate_success_gates(
            world,
            "FIND_WALL2",
            learned_rules={},
            process_rules=process_rules,
        )

        self.assertFalse(check.ok)
        self.assertIn("visible gate", check.reasons)

    def test_find_brick_visible_only_gate_does_not_use_hidden_brick_below_failure(self):
        world = _DummyWorld()
        world.brick["visible"] = True
        world.brick["confidence"] = 88.0
        world.brick["brickBelow"] = True
        process_rules = {"FIND_BRICK": {"success_gates": {"visible": {"min": True}}}}

        check = telemetry_brick.evaluate_success_gates(
            world,
            "FIND_BRICK",
            learned_rules={},
            process_rules=process_rules,
        )

        self.assertTrue(check.ok, msg=check.reason_str())


if __name__ == "__main__":
    unittest.main()
