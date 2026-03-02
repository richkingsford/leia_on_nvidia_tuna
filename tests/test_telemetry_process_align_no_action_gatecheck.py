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
            "SEAT_BRICK2": {
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.5},
                    "dist": {"target": 48.0, "tol": 4.0},
                }
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 48.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class TestTelemetryProcessAlignNoActionGatecheck(unittest.TestCase):
    def test_no_actionable_gap_runs_gatecheck_immediately(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        send_calls = []
        gatecheck_calls = []
        observer_events = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gatecheck_calls.append(1)
            return True

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls.append(1)
            return {}

        def _fake_observer(stage, _world, _vision, cmd, speed, reason):
            observer_events.append((stage, cmd, speed, reason))

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_send = telemetry_process.send_robot_command
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.next_module.select_alignment_next_act = (
                lambda *_a, **_k: {
                    "planner": "gap",
                    "cmd": None,
                    "speed": 0.0,
                    "reason": "all_gaps_within_gate",
                    "score": None,
                }
            )
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="SEAT_BRICK2",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=_fake_observer,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.send_robot_command = orig_send
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(gatecheck_calls, [1])
        self.assertEqual(send_calls, [])
        self.assertGreaterEqual(robot.stop_calls, 1)
        self.assertTrue(
            any(
                stage == "action" and cmd is None and speed == 0.0 and reason == "gatecheck hold"
                for stage, cmd, speed, reason in observer_events
            )
        )

    def test_no_actionable_gap_retries_gatecheck_until_success(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        send_calls = []
        sleep_calls = []
        gatecheck_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gatecheck_calls["n"] += 1
            return gatecheck_calls["n"] >= 2

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls.append(1)
            return {}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_send = telemetry_process.send_robot_command
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.next_module.select_alignment_next_act = (
                lambda *_a, **_k: {
                    "planner": "gap",
                    "cmd": None,
                    "speed": 0.0,
                    "reason": "all_gaps_within_gate",
                    "score": None,
                }
            )
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *args, **_kwargs: sleep_calls.append(
                float(args[0]) if args else 0.0
            )

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="SEAT_BRICK2",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.send_robot_command = orig_send
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(gatecheck_calls["n"], 2)
        self.assertEqual(send_calls, [])
        self.assertGreaterEqual(len(sleep_calls), 1)
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_none_reason_still_runs_immediate_gatecheck_hold(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        send_calls = []
        gatecheck_calls = []
        observer_events = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gatecheck_calls.append(1)
            return True

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls.append(1)
            return {}

        def _fake_observer(stage, _world, _vision, cmd, speed, reason):
            observer_events.append((stage, cmd, speed, reason))

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_send = telemetry_process.send_robot_command
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.next_module.select_alignment_next_act = (
                lambda *_a, **_k: {
                    "planner": "gap",
                    "cmd": None,
                    "speed": 0.0,
                    "reason": "none",
                    "score": None,
                }
            )
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="SEAT_BRICK2",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=_fake_observer,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.send_robot_command = orig_send
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(gatecheck_calls, [1])
        self.assertEqual(send_calls, [])
        self.assertGreaterEqual(robot.stop_calls, 1)
        self.assertTrue(
            any(
                stage == "action" and cmd is None and speed == 0.0 and reason == "gatecheck hold"
                for stage, cmd, speed, reason in observer_events
            )
        )


if __name__ == "__main__":
    unittest.main()
