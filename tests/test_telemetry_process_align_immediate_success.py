import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _DummyWorld:
    def __init__(self):
        self.process_rules = {
            "FIND_WALL": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "dist": {"target": 120.0, "tol": 2.0},
                }
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False


class TestTelemetryProcessAlignImmediateSuccess(unittest.TestCase):
    def test_alignment_exits_immediately_when_success_is_true(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        send_calls = []

        orig_wait_for_start = telemetry_process.wait_for_start_gates
        orig_update_world = telemetry_process.update_world_from_vision
        orig_eval_gate = telemetry_process.evaluate_gate_status
        orig_send = telemetry_process.send_robot_command
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        try:
            telemetry_process.wait_for_start_gates = lambda *args, **kwargs: "start"
            telemetry_process.update_world_from_vision = lambda *args, **kwargs: None
            telemetry_process.evaluate_gate_status = lambda *args, **kwargs: (True, 1.0)
            telemetry_process.send_robot_command = lambda *args, **kwargs: send_calls.append(1)
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *args, **kwargs: {}

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_WALL",
                robot=robot,
                vision=None,
                world=world,
                steps=[],
                raw_steps=[],
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait_for_start
            telemetry_process.update_world_from_vision = orig_update_world
            telemetry_process.evaluate_gate_status = orig_eval_gate
            telemetry_process.send_robot_command = orig_send
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_calls, [])
        self.assertGreaterEqual(robot.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
