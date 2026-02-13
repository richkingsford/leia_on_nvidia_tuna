import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessStartGateSkipFind(unittest.TestCase):
    def test_find_wall_steps_do_not_require_start_gates(self):
        self.assertFalse(telemetry_process.step_requires_start_gates("FIND_WALL", {}))
        self.assertFalse(telemetry_process.step_requires_start_gates("FIND_WALL2", {}))

    def test_wait_for_start_gates_short_circuits_for_find_steps(self):
        status = telemetry_process.wait_for_start_gates(
            None,
            None,
            "FIND_WALL2",
            log=False,
        )
        self.assertEqual(status, "start")

    def test_wait_for_start_gates_observes_once_before_skip(self):
        class _DummyWorld:
            def __init__(self):
                self.process_rules = {}

        class _DummyRobot:
            def __init__(self):
                self.stop_calls = 0
                self._last_turn_cmd = "r"

            def stop(self):
                self.stop_calls += 1

        world = _DummyWorld()
        robot = _DummyRobot()
        vision = object()
        update_calls = []
        observer_events = []

        def _fake_update(world_obj, vision_obj, log=True):
            _ = log
            self.assertIs(world_obj, world)
            self.assertIs(vision_obj, vision)
            update_calls.append("update")

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update):
            status = telemetry_process.wait_for_start_gates(
                world,
                vision,
                "FIND_WALL2",
                robot=robot,
                log=False,
                allow_success=False,
                observer=lambda phase, *_args: observer_events.append(phase),
            )

        self.assertEqual(status, "start")
        self.assertEqual(update_calls, ["update"])
        self.assertIn("frame", observer_events)
        self.assertEqual(robot.stop_calls, 0)
        self.assertIsNone(robot._last_turn_cmd)

    def test_non_find_steps_still_require_start_gates(self):
        self.assertTrue(telemetry_process.step_requires_start_gates("ALIGN_BRICK", {}))

    def test_wait_for_start_gates_can_return_success_when_skip_step_already_meets_gates(self):
        class _DummyWorld:
            def __init__(self):
                self.process_rules = {
                    "FIND_WALL": {
                        "success_gates": {
                            "visible": {"min": True},
                        }
                    }
                }

        class _DummyRobot:
            def __init__(self):
                self.stop_calls = 0
                self._last_turn_cmd = "l"

            def stop(self):
                self.stop_calls += 1

        world = _DummyWorld()
        robot = _DummyRobot()
        vision = object()
        update_calls = []

        def _fake_update(world_obj, vision_obj, log=True):
            _ = log
            self.assertIs(world_obj, world)
            self.assertIs(vision_obj, vision)
            update_calls.append("update")

        with patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update):
            with patch.object(telemetry_process, "evaluate_gate_status", return_value=(True, 100.0)):
                status = telemetry_process.wait_for_start_gates(
                    world,
                    vision,
                    "FIND_WALL",
                    robot=robot,
                    log=False,
                    allow_success=True,
                )

        self.assertEqual(status, "success")
        self.assertGreaterEqual(len(update_calls), 1)
        self.assertGreaterEqual(robot.stop_calls, 1)
        self.assertIsNone(robot._last_turn_cmd)


if __name__ == "__main__":
    unittest.main()
