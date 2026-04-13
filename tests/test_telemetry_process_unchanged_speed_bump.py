import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessUnchangedSpeedBump(unittest.TestCase):
    def test_should_fail_required_gap_unknown_only_in_close_gap_phase(self):
        self.assertFalse(
            telemetry_process._should_fail_required_gap_unknown(
                {"visible": False},
                is_search_action=False,
            )
        )
        self.assertFalse(
            telemetry_process._should_fail_required_gap_unknown(
                {"visible": True},
                is_search_action=True,
            )
        )
        self.assertTrue(
            telemetry_process._should_fail_required_gap_unknown(
                {"visible": True},
                is_search_action=False,
            )
        )

    def test_required_align_unknown_gaps_flags_required_missing_fields(self):
        missing = telemetry_process._required_align_unknown_gaps(
            {
                "x_required": True,
                "x_err": None,
                "y_required": True,
                "y_err": 1.2,
                "dist_required": True,
                "dist": None,
                "dist_target": 50.0,
            }
        )
        self.assertEqual(missing, ("x_err", "dist"))

    def test_required_align_unknown_gaps_ignores_non_required_missing_fields(self):
        missing = telemetry_process._required_align_unknown_gaps(
            {
                "x_required": False,
                "x_err": None,
                "y_required": False,
                "y_err": None,
                "dist_required": False,
                "dist": None,
                "dist_target": None,
            }
        )
        self.assertEqual(missing, ())

    def test_required_align_unknown_gaps_flags_missing_dist_target(self):
        missing = telemetry_process._required_align_unknown_gaps(
            {
                "x_required": True,
                "x_err": 0.0,
                "y_required": False,
                "y_err": None,
                "dist_required": True,
                "dist": 42.0,
                "dist_target": None,
            }
        )
        self.assertEqual(missing, ("dist_target",))

    def test_lite_fail_fallback_supports_seat_brick_step(self):
        class _World:
            process_rules = {
                "SEAT_BRICK": {
                    "success_gates": {
                        "visible": {"min": True},
                        "dist": {"target": 48.0, "tol": 1.5},
                    }
                }
            }
            brick = {"visible": True, "dist": 45.2}

        orig_lite_measure = telemetry_process._lite_gate_measurement_for_step
        try:
            telemetry_process._lite_gate_measurement_for_step = (
                lambda *_a, **_k: (
                    {"visible": True, "dist": 45.2},
                    {"enabled": True, "required": 1, "collected": 1},
                )
            )
            act = telemetry_process._align_brick_lite_fail_fallback_action(_World(), "SEAT_BRICK")
        finally:
            telemetry_process._lite_gate_measurement_for_step = orig_lite_measure

        self.assertIsInstance(act, dict)
        self.assertEqual(act.get("reason"), "lite_fail_fallback_dist")
        self.assertEqual(act.get("cmd"), "b")
        self.assertEqual(act.get("score"), 1)

    def test_lite_fail_fallback_can_use_current_effective_y_axis_failure(self):
        class _World:
            process_rules = {
                "ALIGN_BRICK": {
                    "success_gates": {
                        "visible": {"min": True},
                        "xAxis_offset_abs": {"target": 7.1, "tol": 1.4},
                        "yAxis_offset_abs": {"target": 3.7, "tol": 2.3},
                        "dist": {"target": 103.6, "tol": 2.3},
                    }
                }
            }
            brick = {
                "visible": True,
                "dist": 104.7,
                "x_axis": 8.4,
                "offset_x": 8.4,
                "y_axis": 6.2,
                "offset_y": 6.2,
            }

        orig_lite_measure = telemetry_process._lite_gate_measurement_for_step
        try:
            telemetry_process._lite_gate_measurement_for_step = (
                lambda *_a, **_k: (
                    {"visible": True, "dist": 104.7, "x_axis": 8.4, "offset_x": 8.4, "y_axis": 5.9, "offset_y": 5.9},
                    {"enabled": True, "required": 1, "collected": 1},
                )
            )
            act = telemetry_process._align_brick_lite_fail_fallback_action(
                _World(),
                "ALIGN_BRICK",
                prefer_effective_sample=True,
            )
        finally:
            telemetry_process._lite_gate_measurement_for_step = orig_lite_measure

        self.assertIsInstance(act, dict)
        self.assertEqual(act.get("reason"), "lite_fail_fallback_y_axis")
        self.assertEqual(act.get("cmd"), "d")
        self.assertEqual(act.get("score"), 1)

    def test_repeated_unchanged_action_escalates_speed_after_three_same_acts(self):
        class _World:
            def __init__(self):
                self.process_rules = {
                    "TEST_STEP": {
                        "align_policy": {
                            "target_lock_enabled": False,
                        },
                        "success_gates": {
                            "visible": {"min": True},
                            "xAxis_offset_abs": {"target": 0.0, "tol": 2.0},
                            "yAxis_offset_abs": {"target": 0.0, "tol": 2.0},
                            "dist": {"target": 50.0, "tol": 5.0},
                        },
                        "start_gates": {
                            "visible": {"min": True},
                        },
                    }
                }
                self.learned_rules = {}
                self.wall_envelope = None
                self.brick = {
                    "visible": True,
                    "dist": 52.0,
                    "angle": 0.0,
                    "offset_x": 1.0,
                    "offset_y": 0.0,
                    "x_axis": 1.0,
                    "y_axis": 0.0,
                    "confidence": 90.0,
                }
                self._frame_id = 0
                self._success_confirm_frames = 0
                self._success_confirm_progress = None
                self._success_confirm_logged = False

            def update_from_motion(self, _evt):
                return None

        world = _World()
        robot = type("_DummyRobot", (), {"stop": lambda self: None})()
        send_calls = []
        observe_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_observe_success(*_args, **_kwargs):
            observe_calls["n"] += 1
            if observe_calls["n"] >= 20:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_pre_action_obs(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "f",
                "score": 15,
                "speed": 0.1,
                "reason": "dist_alignment",
                "correction_type": "distance",
            }

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls.append(dict(_kwargs))
            return {}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_capture = telemetry_process._capture_auto_diag_focus
        orig_delta = telemetry_process._auto_diag_delta_phrase
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.pre_action_success_observation = _fake_pre_action_obs
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process._capture_auto_diag_focus = lambda *_a, **_k: {"metric": "x_err", "value": 2.0}
            telemetry_process._auto_diag_delta_phrase = lambda *_a, **_k: (None, None, "unchanged")
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="TEST_STEP",
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
            telemetry_process._capture_auto_diag_focus = orig_capture
            telemetry_process._auto_diag_delta_phrase = orig_delta
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(len(send_calls), 4)
        self.assertEqual(send_calls[0].get("speed_score"), 15)
        self.assertEqual(send_calls[1].get("speed_score"), 15)
        self.assertEqual(send_calls[2].get("speed_score"), 15)
        self.assertEqual(send_calls[3].get("speed_score"), int(round(min(100, 15 * 1.15))))

    def test_no_breakway_contingency_switches_to_duration_after_speed_tries(self):
        class _World:
            def __init__(self):
                self.process_rules = {
                    "TEST_STEP": {
                        "align_policy": {
                            "target_lock_enabled": False,
                        },
                        "success_gates": {
                            "visible": {"min": True},
                            "xAxis_offset_abs": {"target": 0.0, "tol": 2.0},
                            "yAxis_offset_abs": {"target": 0.0, "tol": 2.0},
                            "dist": {"target": 50.0, "tol": 5.0},
                        },
                        "start_gates": {
                            "visible": {"min": True},
                        },
                    }
                }
                self.learned_rules = {}
                self.wall_envelope = None
                self.brick = {
                    "visible": True,
                    "dist": 52.0,
                    "angle": 0.0,
                    "offset_x": 1.0,
                    "offset_y": 0.0,
                    "x_axis": 1.0,
                    "y_axis": 0.0,
                    "confidence": 90.0,
                }
                self._frame_id = 0
                self._success_confirm_frames = 0
                self._success_confirm_progress = None
                self._success_confirm_logged = False

            def update_from_motion(self, _evt):
                return None

        class _DummyRobot:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls = int(self.stop_calls) + 1

        world = _World()
        robot = _DummyRobot()
        send_calls = []
        observe_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1

        def _fake_observe_success(*_args, **_kwargs):
            observe_calls["n"] += 1
            if observe_calls["n"] >= 40:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_pre_action_obs(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "f",
                "score": 15,
                "speed": 0.1,
                "reason": "dist_alignment",
                "correction_type": "distance",
            }

        def _fake_send_robot_command(*_args, **_kwargs):
            send_calls.append(dict(_kwargs))
            return {}

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_pre_action = telemetry_process.pre_action_success_observation
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_send = telemetry_process.send_robot_command
        orig_capture = telemetry_process._capture_auto_diag_focus
        orig_delta = telemetry_process._auto_diag_delta_phrase
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success
            telemetry_process.pre_action_success_observation = _fake_pre_action_obs
            telemetry_process.next_module.select_alignment_next_act = _fake_planner
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process._capture_auto_diag_focus = lambda *_a, **_k: {"metric": "x_err", "value": 2.0}
            telemetry_process._auto_diag_delta_phrase = lambda *_a, **_k: (None, None, "unchanged")
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *_a, **_k: None

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="TEST_STEP",
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
            telemetry_process._capture_auto_diag_focus = orig_capture
            telemetry_process._auto_diag_delta_phrase = orig_delta
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(len(send_calls), 9)
        self.assertGreaterEqual(int(robot.stop_calls), 1)
        self.assertTrue(any((call.get("duration_override_ms") or 0) > 0 for call in send_calls))


if __name__ == "__main__":
    unittest.main()
