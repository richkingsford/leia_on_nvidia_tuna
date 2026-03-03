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
        self.assertGreaterEqual(len(sent_cmds), 1)

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
        self.assertGreaterEqual(len(sent_cmds), 1)

    def test_gap_alignment_waits_for_center_focus_lock_before_planning(self):
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
            # Exit after a few loops; planner should never run before this.
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
        self.assertEqual(planner_calls["n"], 0)
        self.assertEqual(sent_cmds, [])

    def test_find_brick_no_action_invisible_scans_right(self):
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
        self.assertEqual(sent_scores[0], 20)
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
