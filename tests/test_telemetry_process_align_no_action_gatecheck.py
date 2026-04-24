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
    def test_align_actions_send_single_pulse_no_ease_queue(self):
        world = _DummyWorld()
        world.process_rules = {
            "SEAT_BRICK2": {
                "align_policy": {
                    "target_lock_enabled": False,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.5},
                    "dist": {"target": 48.0, "tol": 4.0},
                },
                "start_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 52.0,
            "angle": 0.0,
            "offset_x": 2.0,
            "offset_y": 1.0,
            "x_axis": 2.0,
            "y_axis": 1.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        send_kwargs = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_planner(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "l",
                "score": 15,
                "speed": 0.0,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*_args, **_kwargs):
            send_kwargs.append(dict(_kwargs))
            return {}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.pre_action_success_observation = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.run_full_gatecheck_after_act = lambda *_a, **_k: True
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
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.pre_action_success_observation = orig_pre_action
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.send_robot_command = orig_send
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(len(send_kwargs), 1)
        self.assertIs(send_kwargs[0].get("ease_in_out_enabled"), False)

    def test_align_skips_send_when_gate_passes_at_pre_send_edge(self):
        world = _DummyWorld()
        world.process_rules = {
            "SEAT_BRICK2": {
                "align_policy": {
                    "target_lock_enabled": False,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.5},
                    "dist": {"target": 48.0, "tol": 4.0},
                },
                "start_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 50.0,
            "angle": 0.0,
            "offset_x": 2.0,
            "offset_y": 1.0,
            "x_axis": 2.0,
            "y_axis": 1.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        send_calls = {"n": 0}
        obs_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_observe_success(*_args, **_kwargs):
            obs_calls["n"] += 1
            # 1st call = loop gatecheck fail, 2nd call = pre-send edge pass.
            if obs_calls["n"] == 1:
                return {"success_met": False, "hold_for_confirm": False}
            if obs_calls["n"] == 2:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_pre_action_obs(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "l",
                "score": 15,
                "speed": 0.0,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls["n"] += 1
            return {}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.pre_action_success_observation = _fake_pre_action_obs
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
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
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.pre_action_success_observation = orig_pre_action
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.send_robot_command = orig_send
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_calls["n"], 0)

    def test_brick_lock_in_crosshairs_true_soft_bypasses_failed_lock_acquire(self):
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "align_policy": {
                    "target_lock_enabled": True,
                    "target_lock_confirm_frames": 2,
                    "target_lock_acquire_timeout_s": 0.02,
                    "target_lock_hold_bad_frames": 1,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -3.86, "tol": 1.4},
                    "dist": {"target": 115.49, "tol": 1.5},
                },
                "start_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 141.7,
            "angle": 0.0,
            "offset_x": 20.0,
            "offset_y": 0.0,
            "x_axis": 20.0,
            "y_axis": 0.0,
            "inCrosshairs": True,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        planner_calls = {"n": 0}
        sent_cmds = []
        gate_calls = {"n": 0}
        update_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            update_calls["n"] += 1
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            # Alternate left/right to prevent stable lock acquisition.
            x_val = 20.0 if (update_calls["n"] % 2 == 0) else -20.0
            _world.brick["visible"] = True
            _world.brick["inCrosshairs"] = True
            _world.brick["x_axis"] = x_val
            _world.brick["offset_x"] = x_val
            _world.brick["y_axis"] = 0.0
            _world.brick["offset_y"] = 0.0
            _world.brick["dist"] = 141.7

        def _fake_observe_success(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            planner_calls["n"] += 1
            return {
                "planner": "gap",
                "cmd": "l",
                "score": 20,
                "speed": 0.0,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            return gate_calls["n"] >= 1

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
            telemetry_process.send_robot_command = orig_send
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(planner_calls["n"], 1)

    def test_brick_lock_in_crosshairs_true_bypasses_lock_y_band_reject(self):
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "align_policy": {
                    "target_lock_enabled": True,
                    "target_lock_confirm_frames": 1,
                    "target_lock_acquire_timeout_s": 0.05,
                    "target_lock_hold_bad_frames": 1,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -3.86, "tol": 1.4},
                    "dist": {"target": 115.49, "tol": 1.5},
                },
                "start_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 141.7,
            "angle": 0.0,
            "offset_x": 6.7,
            "offset_y": 30.0,  # intentionally far in y
            "x_axis": 6.7,
            "y_axis": 30.0,
            "inCrosshairs": True,  # operator says this should be sufficient
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        planner_calls = {"n": 0}
        sent_cmds = []
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["inCrosshairs"] = True
            _world.brick["x_axis"] = 6.7
            _world.brick["offset_x"] = 6.7
            _world.brick["y_axis"] = 30.0
            _world.brick["offset_y"] = 30.0
            _world.brick["dist"] = 141.7

        def _fake_observe_success(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            planner_calls["n"] += 1
            return {
                "planner": "gap",
                "cmd": "l",
                "score": 20,
                "speed": 0.0,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            return gate_calls["n"] >= 1

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="BRICK_LOCK",
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
            telemetry_process.send_robot_command = orig_send
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(planner_calls["n"], 1)

    def test_gap_alignment_does_not_block_on_center_focus_lock(self):
        world = _DummyWorld()
        world.process_rules = {
            "SEAT_BRICK2": {
                "align_policy": {
                    "target_lock_enabled": True,
                    "target_lock_confirm_frames": 2,
                    "target_lock_acquire_timeout_s": 0.01,
                    "target_lock_hold_bad_frames": 1,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.5},
                    "dist": {"target": 48.0, "tol": 4.0},
                },
                "start_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 48.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        planner_calls = {"n": 0}
        sent_cmds = []
        gate_obs_calls = {"n": 0}
        update_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            update_calls["n"] += 1
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            # Alternate x-axis between centered and far-right to prevent
            # consecutive lock-confirm frames.
            if update_calls["n"] % 2 == 0:
                _world.brick["x_axis"] = 0.0
                _world.brick["offset_x"] = 0.0
            else:
                _world.brick["x_axis"] = 20.0
                _world.brick["offset_x"] = 20.0
            _world.brick["y_axis"] = 0.0
            _world.brick["offset_y"] = 0.0
            _world.brick["dist"] = 48.0
            _world.brick["visible"] = True

        def _fake_observe_success(*_args, **_kwargs):
            gate_obs_calls["n"] += 1
            # Exit after a few loops.
            return {"success_met": gate_obs_calls["n"] >= 3, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            planner_calls["n"] += 1
            return {
                "planner": "gap",
                "cmd": "r",
                "score": 15,
                "speed": 0.0,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            return {}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
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
            telemetry_process.send_robot_command = orig_send
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(planner_calls["n"], 1)

    def test_find_brick_no_action_invisible_scans_right(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_BRICK": {
                "scan_direction": "r",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 25,
                    "low_score": 25,
                    "start_with_high": True,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": False,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
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
            def _planner_should_not_run(*_a, **_k):
                planner_calls["n"] += 1
                raise AssertionError("planner should not run before FIND_BRICK is visible")

            telemetry_process.next_module.select_alignment_next_act = _planner_should_not_run
            telemetry_process.run_full_gatecheck_after_act = lambda *_a, **_k: True
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_BRICK",
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
        self.assertIn("r", sent_cmds)
        self.assertEqual(sent_scores[0], 25)
        self.assertEqual(planner_calls["n"], 0)

    def test_find_brick_invisible_search_alternates_20_then_5(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_BRICK": {
                "scan_direction": "r",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 20,
                    "low_score": 5,
                    "start_with_high": True,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": False,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            return gate_calls["n"] >= 2

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

            def _planner_should_not_run(*_a, **_k):
                planner_calls["n"] += 1
                raise AssertionError("planner should not run before FIND_BRICK is visible")

            telemetry_process.next_module.select_alignment_next_act = _planner_should_not_run
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_BRICK",
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
        self.assertEqual(sent_cmds[:2], ["r", "r"])
        self.assertEqual(sent_scores[:2], [20, 5])
        self.assertEqual(planner_calls["n"], 0)

    def test_find_brick_invisible_search_turns_then_backs_off(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_BRICK": {
                "scan_direction": "r",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 25,
                    "low_score": 25,
                    "start_with_high": True,
                    "commands": ["r", "b"],
                    "command_scores": {"b": 10},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": False,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            return gate_calls["n"] >= 2

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

            def _planner_should_not_run(*_a, **_k):
                planner_calls["n"] += 1
                raise AssertionError("planner should not run before FIND_BRICK is visible")

            telemetry_process.next_module.select_alignment_next_act = _planner_should_not_run
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_BRICK",
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
        self.assertEqual(sent_cmds[:2], ["r", "b"])
        self.assertEqual(sent_scores[:2], [25, 10])
        self.assertEqual(planner_calls["n"], 0)

    def test_find_wall2_no_action_invisible_scans_left_then_aligns_later(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "scan_direction": "l",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 20,
                    "low_score": 5,
                    "start_with_high": True,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": False,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
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

            def _planner_should_not_run(*_a, **_k):
                planner_calls["n"] += 1
                raise AssertionError("planner should not run before FIND_WALL2 is visible")

            telemetry_process.next_module.select_alignment_next_act = _planner_should_not_run
            telemetry_process.run_full_gatecheck_after_act = lambda *_a, **_k: True
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_WALL2",
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
        self.assertIn("l", sent_cmds)
        self.assertEqual(sent_scores[0], 20)
        self.assertEqual(planner_calls["n"], 0)

    def test_find_wall2_invisible_search_alternates_left_then_back(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "scan_direction": "l",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 25,
                    "low_score": 25,
                    "start_with_high": True,
                    "commands": ["l", "b"],
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": False,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            return gate_calls["n"] >= 2

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

            def _planner_should_not_run(*_a, **_k):
                planner_calls["n"] += 1
                raise AssertionError("planner should not run before FIND_WALL2 is visible")

            telemetry_process.next_module.select_alignment_next_act = _planner_should_not_run
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_WALL2",
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
        self.assertEqual(sent_cmds[:2], ["l", "b"])
        self.assertEqual(sent_scores[:2], [25, 25])
        self.assertEqual(planner_calls["n"], 0)

    def test_find_wall2_startup_action_runs_before_success_shortcut_and_triggers_ground_reset(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "startup_action_exception": {
                    "enabled": True,
                    "command": "l",
                    "score": 100,
                    "acts": 7,
                    "pause_s": 0.25,
                },
                "start_ground_reset_exception": {
                    "enabled": True,
                    "run_after_startup_action": True,
                },
                "success_gates": {
                    "visible": {"min": True},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        ground_reset_calls = {"n": 0}
        sleep_calls = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            return {}

        def _fake_ground_reset(*_args, **_kwargs):
            ground_reset_calls["n"] += 1
            return {"enabled": True, "success": True, "reason": "ground reset pass"}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_send = telemetry_process.send_robot_command
        orig_ground_reset = telemetry_process._run_start_ground_reset_exception
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            # Simulate the early-success precheck case that previously skipped startup exceptions.
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "success"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = (
                lambda *_a, **_k: {"success_met": True, "hold_for_confirm": False}
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
            telemetry_process.run_full_gatecheck_after_act = lambda *_a, **_k: True
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process._run_start_ground_reset_exception = _fake_ground_reset
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda secs=0.0: sleep_calls.append(float(secs or 0.0))

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_WALL2",
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
            telemetry_process._run_start_ground_reset_exception = orig_ground_reset
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(sent_cmds[:7], ["l"] * 7)
        self.assertEqual(ground_reset_calls["n"], 1)
        pause_hits = [val for val in sleep_calls if abs(float(val) - 0.25) < 1e-6]
        self.assertEqual(len(pause_hits), 6)

    def test_find_wall2_visible_true_with_unknown_xy_uses_search_cycle(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "scan_direction": "l",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 25,
                    "low_score": 25,
                    "start_with_high": True,
                    "commands": ["l", "b"],
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": None,
            "offset_y": None,
            "x_axis": None,
            "y_axis": None,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}
        gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            return gate_calls["n"] >= 2

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

            def _planner_should_not_run(*_a, **_k):
                planner_calls["n"] += 1
                raise AssertionError("planner should not run before FIND_WALL2 has x/y observations")

            telemetry_process.next_module.select_alignment_next_act = _planner_should_not_run
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_WALL2",
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
        self.assertEqual(sent_cmds[:2], ["l", "b"])
        self.assertEqual(sent_scores[:2], [25, 25])
        self.assertEqual(planner_calls["n"], 0)

    def test_approach_vector_wall_skips_turn_when_x_err_unknown(self):
        world = _DummyWorld()
        world.process_rules = {
            "APPROACH_VECTOR_WALL": {
                "controller": "align",
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "angle_abs": {"target": 6.61, "tol": 0.0},
                },
            }
        }
        world.brick = {
            "visible": True,
            "dist": 171.0,
            "angle": 5.1,
            "offset_x": None,
            "offset_y": -8.9,
            "x_axis": None,
            "y_axis": -8.9,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_x_values = []
        planner_calls = {"n": 0}
        full_gate_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            # Keep x unknown on first loop so the unknown-measurement guard blocks action.
            if int(_world._frame_id) >= 2:
                _world.brick["x_axis"] = 4.0
                _world.brick["offset_x"] = 4.0

        def _fake_observe_success(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_pre_action_obs(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            planner_calls["n"] += 1
            return {
                "planner": "generic",
                "cmd": "r",
                "speed": 0.2,
                "score": 20,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_x_values.append(world.brick.get("x_axis"))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            full_gate_calls["n"] += 1
            return full_gate_calls["n"] >= 1

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.pre_action_success_observation = _fake_pre_action_obs
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="APPROACH_VECTOR_WALL",
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
            telemetry_process.pre_action_success_observation = orig_pre_action
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.send_robot_command = orig_send
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(planner_calls["n"], 2)
        self.assertEqual(sent_cmds, ["r"])
        self.assertIsNotNone(sent_x_values[0])
        self.assertGreaterEqual(robot.stop_calls, 1)

    def test_find_wall2_phase2_planner_runs_only_after_phase1_visible_pass(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "scan_direction": "l",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 25,
                    "low_score": 25,
                    "start_with_high": True,
                    "commands": ["l", "b"],
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": False,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 3.0,
            "offset_y": 4.0,
            "x_axis": 3.0,
            "y_axis": 4.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}
        planner_sent = {"done": False}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            # Phase 1: invisible for first two loop observations, then visible.
            _world.brick["visible"] = bool(int(_world._frame_id) >= 3)

        def _fake_observe_success(*_args, **_kwargs):
            return {"success_met": bool(planner_sent["done"]), "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            planner_calls["n"] += 1
            if not bool(world.brick.get("visible")):
                raise AssertionError("phase2 planner must not run before phase1 visible pass")
            return {
                "planner": "gap",
                "cmd": "d",
                "speed": 0.0,
                "score": 10,
                "reason": "y_axis_alignment",
                "correction_type": "y_axis",
            }

        def _fake_send_robot_command(*args, **_kwargs):
            cmd = str(args[3])
            sent_cmds.append(cmd)
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
            if cmd == "d":
                planner_sent["done"] = True
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
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.run_full_gatecheck_after_act = lambda *_a, **_k: False
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_WALL2",
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
        self.assertGreaterEqual(planner_calls["n"], 1)
        self.assertGreaterEqual(len(sent_cmds), 1)
        self.assertEqual(sent_cmds[0], "l")
        self.assertEqual(sent_scores[0], 25)

    def test_find_wall2_search_has_no_extra_control_dt_sleep_between_acts(self):
        world = _DummyWorld()
        world.process_rules = {
            "FIND_WALL2": {
                "scan_direction": "l",
                "search_visible_false_speed_cycle": {
                    "enabled": True,
                    "high_score": 25,
                    "low_score": 25,
                    "start_with_high": True,
                    "commands": ["l", "b"],
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 6.0},
                },
            }
        }
        world.brick = {
            "visible": False,
            "dist": 120.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 90.0,
        }
        world._frame_id = 0
        robot = _DummyRobot()
        sent_cmds = []
        sent_scores = []
        planner_calls = {"n": 0}
        gate_calls = {"n": 0}
        sleep_calls = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_send_robot_command(*args, **_kwargs):
            sent_cmds.append(str(args[3]))
            sent_scores.append(int(_kwargs.get("speed_score") or 0))
            return {}

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            return gate_calls["n"] >= 2

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

            def _planner_should_not_run(*_a, **_k):
                planner_calls["n"] += 1
                raise AssertionError("planner should not run before FIND_WALL2 is visible")

            telemetry_process.next_module.select_alignment_next_act = _planner_should_not_run
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda secs=0.0: sleep_calls.append(float(secs or 0.0))

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="FIND_WALL2",
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
        self.assertEqual(sent_cmds[:2], ["l", "b"])
        self.assertEqual(sent_scores[:2], [25, 25])
        self.assertEqual(planner_calls["n"], 0)
        control_dt_hits = [
            val
            for val in sleep_calls
            if abs(float(val) - float(telemetry_process.CONTROL_DT)) < 1e-9
        ]
        self.assertEqual(control_dt_hits, [])

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

    def test_align_action_waits_for_post_act_settle_before_gatecheck(self):
        world = _DummyWorld()
        world.process_rules = {
            "SEAT_BRICK2": {
                "align_policy": {
                    "target_lock_enabled": False,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": 0.0, "tol": 1.0},
                    "yAxis_offset_abs": {"target": 0.0, "tol": 1.5},
                    "dist": {"target": 48.0, "tol": 4.0},
                },
                "start_gates": {
                    "visible": {"min": True},
                },
            }
        }
        robot = _DummyRobot()
        order = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_planner(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "f",
                "score": 1,
                "speed": 0.0,
                "reason": "distance_alignment",
                "correction_type": "distance",
            }

        def _fake_send_robot_command(*_args, **_kwargs):
            order.append("send")
            return {"duration_ms": 700}

        def _fake_wait_for_frame_settle(_world, _vision, frames, log=False):
            _ = log
            order.append(f"wait:{int(frames)}")

        def _fake_run_full_gatecheck(*_args, **_kwargs):
            order.append("gatecheck")
            return True

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_wait_frames = telemetry_process.wait_for_frame_settle
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.pre_action_success_observation = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.wait_for_frame_settle = _fake_wait_for_frame_settle
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
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.pre_action_success_observation = orig_pre_action
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.send_robot_command = orig_send
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.wait_for_frame_settle = orig_wait_frames
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        expected_wait_frames = telemetry_process._frames_from_seconds(
            0.7,
            telemetry_process.POST_ACT_PAUSE_FRAMES,
        )
        self.assertEqual(order[0], "send")
        self.assertIn(f"wait:{int(expected_wait_frames)}", order)
        self.assertEqual(order[-1], "gatecheck")
        self.assertLess(
            order.index(f"wait:{int(expected_wait_frames)}"),
            order.index("gatecheck"),
        )

    def test_align_reuses_single_post_act_gatecheck_before_second_micro_adjust(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        observe_calls = {"n": 0}
        pre_action_calls = {"n": 0}
        send_calls = {"n": 0}
        gatecheck_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_observe_success(*_args, **_kwargs):
            observe_calls["n"] += 1
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_pre_action_obs(*_args, **_kwargs):
            pre_action_calls["n"] += 1
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "l",
                "score": 7,
                "speed": 0.0,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls["n"] += 1
            return {}

        def _fake_run_full_gatecheck(_world, _vision, _step, *_args, **_kwargs):
            gatecheck_calls["n"] += 1
            if gatecheck_calls["n"] == 1:
                _world._recent_post_act_gatecheck = {
                    "step": "SEAT_BRICK2",
                    "phase": "align",
                    "success_ok": False,
                    "instant_success_ok": False,
                    "effective_success_ok": False,
                    "success_met": False,
                    "hold_for_confirm": False,
                    "truth_ok": False,
                    "frame_id": int(getattr(_world, "_frame_id", 0) or 0),
                }
                return False
            return True

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_post = telemetry_process.post_act_analysis
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.pre_action_success_observation = _fake_pre_action_obs
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.post_act_analysis = lambda *_a, **_k: None
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
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.pre_action_success_observation = orig_pre_action
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.send_robot_command = orig_send
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.post_act_analysis = orig_post
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_calls["n"], 2)
        self.assertEqual(pre_action_calls["n"], 1)
        self.assertEqual(observe_calls["n"], 2)

    def test_align_holds_on_cached_positive_post_act_gatecheck_instead_of_sending_again(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        observe_calls = {"n": 0}
        send_calls = {"n": 0}
        gatecheck_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_observe_success(*_args, **_kwargs):
            observe_calls["n"] += 1
            if observe_calls["n"] >= 3:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "l",
                "score": 7,
                "speed": 0.0,
                "reason": "x_axis_alignment",
                "correction_type": "x_axis",
            }

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls["n"] += 1
            return {}

        def _fake_run_full_gatecheck(_world, _vision, _step, *_args, **_kwargs):
            gatecheck_calls["n"] += 1
            _world._recent_post_act_gatecheck = {
                "step": "SEAT_BRICK2",
                "phase": "align",
                "success_ok": True,
                "instant_success_ok": True,
                "effective_success_ok": True,
                "success_met": False,
                "hold_for_confirm": True,
                "truth_ok": False,
                "frame_id": int(getattr(_world, "_frame_id", 0) or 0),
            }
            return False

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_run_full = telemetry_process.run_full_gatecheck_after_act
        orig_post = telemetry_process.post_act_analysis
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.pre_action_success_observation = (
                lambda *_a, **_k: {"success_met": False, "hold_for_confirm": False}
            )
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.run_full_gatecheck_after_act = _fake_run_full_gatecheck
            telemetry_process.post_act_analysis = lambda *_a, **_k: None
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
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.pre_action_success_observation = orig_pre_action
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.send_robot_command = orig_send
            telemetry_process.run_full_gatecheck_after_act = orig_run_full
            telemetry_process.post_act_analysis = orig_post
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_calls["n"], 1)
        self.assertEqual(observe_calls["n"], 3)


if __name__ == "__main__":
    unittest.main()
