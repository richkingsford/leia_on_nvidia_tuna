import sys
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self):
        self.process_rules = {}
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": False,
            "dist": 0.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "confidence": 0.0,
        }
        self._smoothed_frame_history = []
        self._frame_id = 0
        self.last_visible_time = time.time()
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False


class TestTelemetryProcessLiteGate(unittest.TestCase):
    def setUp(self):
        self.prev_default = getattr(telemetry_process, "LITE_GATE_DEFAULT_UNIQUE_FRAMES", 3)
        self.prev_steps = dict(getattr(telemetry_process, "LITE_GATE_STEP_UNIQUE_FRAMES", {}))
        self.prev_lite_only_aruco_experiment = bool(
            getattr(telemetry_process, "GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT", False)
        )
        self.prev_aruco_full_pass_scale = float(
            getattr(telemetry_process, "GATECHECK_ARUCO_FULL_PASS_SCALE", 1.0)
        )
        telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = False
        telemetry_process.GATECHECK_ARUCO_FULL_PASS_SCALE = 1.0

    def tearDown(self):
        telemetry_process.LITE_GATE_DEFAULT_UNIQUE_FRAMES = self.prev_default
        telemetry_process.LITE_GATE_STEP_UNIQUE_FRAMES = self.prev_steps
        telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = self.prev_lite_only_aruco_experiment
        telemetry_process.GATECHECK_ARUCO_FULL_PASS_SCALE = self.prev_aruco_full_pass_scale

    def test_apply_lite_gate_check_config_parses_step_frames(self):
        telemetry_process.apply_lite_gate_check_config(
            {
                "default_unique_smoothed_frames": 3,
                "steps": {
                    "ALIGN_BRICK": {"enabled": True, "unique_smoothed_frames": 3},
                    "POSITION_BRICK": {"enabled": True, "unique_smoothed_frames": 4},
                },
            }
        )
        self.assertEqual(telemetry_process.lite_gate_unique_frames("ALIGN_BRICK"), 3)
        self.assertEqual(telemetry_process.lite_gate_unique_frames("POSITION_BRICK"), 4)
        self.assertEqual(telemetry_process.lite_gate_unique_frames("FIND_BRICK"), 3)

    def test_apply_gate_checker_config_parses_aruco_full_pass_scale(self):
        prev_consec = telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED
        prev_majority_window = telemetry_process.GATECHECK_MAJORITY_WINDOW
        prev_majority_required = telemetry_process.GATECHECK_MAJORITY_REQUIRED
        prev_lite_only = telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT
        prev_scale = telemetry_process.GATECHECK_ARUCO_FULL_PASS_SCALE
        try:
            telemetry_process.apply_gate_checker_config(
                {
                    "consecutive_required": 6,
                    "majority_window": 13,
                    "majority_required": 5,
                    "lite_only_aruco_experiment": False,
                    "aruco_full_gatecheck_pass_scale": 0.5,
                }
            )
            self.assertEqual(int(telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED), 6)
            self.assertEqual(int(telemetry_process.GATECHECK_MAJORITY_WINDOW), 13)
            self.assertEqual(int(telemetry_process.GATECHECK_MAJORITY_REQUIRED), 5)
            self.assertFalse(bool(telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT))
            self.assertAlmostEqual(float(telemetry_process.GATECHECK_ARUCO_FULL_PASS_SCALE), 0.5, places=3)
        finally:
            telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED = prev_consec
            telemetry_process.GATECHECK_MAJORITY_WINDOW = prev_majority_window
            telemetry_process.GATECHECK_MAJORITY_REQUIRED = prev_majority_required
            telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = prev_lite_only
            telemetry_process.GATECHECK_ARUCO_FULL_PASS_SCALE = prev_scale

    def test_lite_gate_defaults_apply_to_all_modeled_steps(self):
        root = Path(__file__).resolve().parents[1]
        process_model = json.loads((root / "world_model_process.json").read_text())
        wall_model = json.loads((root / "world_model_wall.json").read_text())
        process_steps = (process_model.get("steps") or {}) if isinstance(process_model, dict) else {}
        wall_steps = (wall_model.get("steps") or {}) if isinstance(wall_model, dict) else {}
        all_steps = set()
        all_steps.update(str(name) for name in process_steps.keys())
        all_steps.update(str(name) for name in wall_steps.keys())
        self.assertTrue(bool(all_steps))

        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3, "steps": {}})
        for step_name in sorted(all_steps):
            frames = telemetry_process.lite_gate_unique_frames(step_name)
            self.assertIsNotNone(frames, step_name)
            self.assertGreaterEqual(int(frames), 1, step_name)

    def test_evaluate_gate_status_uses_lite_average_for_configured_step(self):
        telemetry_process.apply_lite_gate_check_config(
            {
                "steps": {
                    "ALIGN_BRICK": {"enabled": True, "unique_smoothed_frames": 3},
                }
            }
        )
        world = _DummyWorld()
        world.process_rules = {
            "ALIGN_BRICK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -2.0, "tol": 1.0},
                    "dist": {"target": 80.0, "tol": 2.0},
                }
            }
        }
        world.brick.update(
            {
                "visible": True,
                "dist": 80.0,
                "angle": 0.0,
                "offset_x": -2.0,
                "x_axis": -2.0,
                "confidence": 92.0,
            }
        )
        world._smoothed_frame_history = [
            {
                "frame_id": 11,
                "visible": True,
                "dist": 79.0,
                "angle": 0.0,
                "x_axis": -2.1,
                "offset_x": -2.1,
                "confidence": 90.0,
            },
            {
                "frame_id": 12,
                "visible": True,
                "dist": 80.0,
                "angle": 0.0,
                "x_axis": -2.0,
                "offset_x": -2.0,
                "confidence": 92.0,
            },
            {
                "frame_id": 13,
                "visible": True,
                "dist": 81.0,
                "angle": 0.0,
                "x_axis": -1.9,
                "offset_x": -1.9,
                "confidence": 94.0,
            },
        ]
        ok, _ = telemetry_process.evaluate_gate_status(world, "ALIGN_BRICK")
        self.assertTrue(ok)
        self.assertEqual(world._gatecheck_mode, "traditional")
        self.assertEqual(world._gatecheck_lite_required, 3)
        self.assertEqual(world._gatecheck_lite_collected, 3)

    def test_evaluate_gate_status_falls_back_to_traditional_for_other_steps(self):
        telemetry_process.apply_lite_gate_check_config(
            {
                "steps": {
                    "ALIGN_BRICK": {"enabled": True, "unique_smoothed_frames": 3},
                }
            }
        )
        world = _DummyWorld()
        world.process_rules = {
            "FIND_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                }
            }
        }
        world.brick["visible"] = True
        world.last_visible_time = time.time()
        world._smoothed_frame_history = [
            {"frame_id": 1, "visible": True},
            {"frame_id": 2, "visible": True},
            {"frame_id": 3, "visible": True},
        ]
        ok, _ = telemetry_process.evaluate_gate_status(world, "FIND_BRICK")
        self.assertTrue(ok)
        self.assertEqual(world._gatecheck_mode, "traditional")
        self.assertEqual(world._gatecheck_lite_required, 3)
        self.assertEqual(world._gatecheck_lite_collected, 3)

    def test_full_gate_tracker_does_not_start_before_first_post_lite_success_sample(self):
        world = _DummyWorld()
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 3
        world._gatecheck_lite_checks = 0
        world._frame_id = 42

        tracker = telemetry_process.gate_utils.SuccessGateTracker(12, 26, 13)
        tracker.total_checks = 0

        success_met = telemetry_process.update_gatecheck_with_precheck(
            world,
            "BRICK_LOCK",
            tracker,
            False,  # effective success gates still failing (e.g. x-axis gap remains)
            phase="align",
            log=False,
        )

        self.assertFalse(success_met)
        self.assertEqual(tracker.total_checks, 0)
        self.assertEqual(world._gatecheck_mode, "lite")
        status = getattr(world, "_gatecheck_status", None)
        self.assertIsInstance(status, dict)
        self.assertEqual(status.get("mode"), "lite")

    def test_lite_mode_gatecheck_progress_is_logged(self):
        world = _DummyWorld()
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 1
        world._gatecheck_lite_checks = 0
        world._frame_id = 44

        tracker = telemetry_process.gate_utils.SuccessGateTracker(12, 26, 13)
        tracker.total_checks = 0

        with mock.patch("builtins.print") as mock_print:
            success_met = telemetry_process.update_gatecheck_with_precheck(
                world,
                "SEAT_BRICK2",
                tracker,
                False,
                phase="align",
                log=True,
            )

        self.assertFalse(success_met)
        printed = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        )
        self.assertIn("[GATECHECK] SEAT_BRICK2 align: lite", printed)
        self.assertIn("lite 1/3", printed)

    def test_full_gate_tracker_starts_on_first_post_lite_success_sample(self):
        world = _DummyWorld()
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 3
        world._gatecheck_lite_checks = 0
        world._frame_id = 43

        tracker = telemetry_process.gate_utils.SuccessGateTracker(12, 26, 13)
        tracker.total_checks = 0

        success_met = telemetry_process.update_gatecheck_with_precheck(
            world,
            "BRICK_LOCK",
            tracker,
            True,  # first effective-success sample after lite pass
            phase="align",
            log=False,
        )

        self.assertFalse(success_met)  # one sample is not enough to confirm
        self.assertEqual(tracker.total_checks, 1)
        status = getattr(world, "_gatecheck_status", None)
        self.assertIsInstance(status, dict)
        self.assertEqual(status.get("mode"), "traditional")

    def test_run_full_gatecheck_after_act_keeps_align_brick_to_one_check(self):
        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3})
        world = _DummyWorld()
        vision = object()
        tracker = telemetry_process.gate_utils.SuccessGateTracker(12, 26, 13)
        wait_calls = []

        def _fake_wait_for_fresh_frames(*args, **kwargs):
            wait_calls.append((args, kwargs))
            world._frame_id += 1
            return {"advanced": 1}

        with mock.patch.object(telemetry_process.gate_utils, "wait_for_fresh_frames", side_effect=_fake_wait_for_fresh_frames), \
             mock.patch.object(telemetry_process, "update_world_from_vision", return_value=None), \
             mock.patch.object(telemetry_process, "gatecheck_after_move", return_value=False):
            ok = telemetry_process.run_full_gatecheck_after_act(
                world,
                vision,
                "ALIGN_BRICK",
                tracker,
                phase="align",
                log=False,
                observer=None,
            )

        self.assertFalse(ok)
        self.assertEqual(len(wait_calls), 1)

    def test_evaluate_gate_status_lite_only_aruco_skips_traditional_brick_check(self):
        world = _DummyWorld()
        telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = True
        with mock.patch.object(
            telemetry_process,
            "_evaluate_lite_gate_brick_success",
            return_value=True,
        ), mock.patch.object(
            telemetry_process,
            "_evaluate_traditional_brick_success",
            side_effect=AssertionError("traditional brick check should be skipped"),
        ), mock.patch.object(
            telemetry_process,
            "_active_success_gate_mode",
            return_value=telemetry_process.VISION_MODE_ARUCO,
        ), mock.patch.object(
            telemetry_process.telemetry_wall,
            "evaluate_success_gates",
            return_value=SimpleNamespace(ok=True),
        ), mock.patch.object(
            telemetry_process.telemetry_robot_module,
            "evaluate_success_gates",
            return_value=SimpleNamespace(ok=True),
        ):
            ok, _ = telemetry_process.evaluate_gate_status(world, "BRICK_LOCK")

        self.assertTrue(ok)
        self.assertEqual(world._gatecheck_mode, "traditional")
        self.assertTrue(bool(getattr(world, "_gatecheck_lite_passed", False)))

    def test_update_gatecheck_with_precheck_lite_only_aruco_returns_immediate_truth(self):
        world = _DummyWorld()
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 3
        world._gatecheck_lite_passed = True
        world._frame_id = 77
        tracker = telemetry_process.gate_utils.SuccessGateTracker(12, 26, 13)
        telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = True

        with mock.patch.object(
            telemetry_process,
            "_active_success_gate_mode",
            return_value=telemetry_process.VISION_MODE_ARUCO,
        ):
            success_met = telemetry_process.update_gatecheck_with_precheck(
                world,
                "BRICK_LOCK",
                tracker,
                True,
                phase="align",
                log=False,
            )

        self.assertTrue(success_met)
        self.assertEqual(tracker.total_checks, 0)
        status = getattr(world, "_gatecheck_status", None)
        self.assertIsInstance(status, dict)
        self.assertEqual(status.get("truth_by"), "lite_only_aruco")
        self.assertEqual(status.get("need"), 1)
        self.assertEqual(status.get("window_total"), 1)

    def test_update_gatecheck_with_precheck_lite_only_aruco_logs_warning_on_lite_pass(self):
        world = _DummyWorld()
        world._gatecheck_mode = "traditional"
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 3
        world._gatecheck_lite_passed = True
        world._frame_id = 78
        tracker = telemetry_process.gate_utils.SuccessGateTracker(12, 26, 13)
        telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = True

        with mock.patch.object(
            telemetry_process,
            "_active_success_gate_mode",
            return_value=telemetry_process.VISION_MODE_ARUCO,
        ), mock.patch("builtins.print") as print_mock:
            telemetry_process.update_gatecheck_with_precheck(
                world,
                "BRICK_LOCK",
                tracker,
                True,
                phase="align",
                log=True,
            )

        printed = [str(call.args[0]) for call in print_mock.call_args_list if call.args]
        self.assertTrue(
            any("TEMPORARY ARUCO LITE-ONLY EXPERIMENT" in line for line in printed),
            printed,
        )

    def test_run_full_gatecheck_after_act_lite_only_aruco_uses_one_check(self):
        world = _DummyWorld()
        world._gatecheck_lite_required = 3
        vision = object()
        tracker = telemetry_process.gate_utils.SuccessGateTracker(12, 26, 13)
        telemetry_process.GATECHECK_LITE_ONLY_ARUCO_EXPERIMENT = True
        wait_calls = []

        def _fake_wait_for_fresh_frames(*args, **kwargs):
            wait_calls.append((args, kwargs))
            world._frame_id += 1
            return {"advanced": 1}

        with mock.patch.object(
            telemetry_process,
            "_active_success_gate_mode",
            return_value=telemetry_process.VISION_MODE_ARUCO,
        ), mock.patch.object(
            telemetry_process.gate_utils,
            "wait_for_fresh_frames",
            side_effect=_fake_wait_for_fresh_frames,
        ), mock.patch.object(
            telemetry_process,
            "update_world_from_vision",
            return_value=None,
        ), mock.patch.object(
            telemetry_process,
            "gatecheck_after_move",
            return_value=False,
        ):
            ok = telemetry_process.run_full_gatecheck_after_act(
                world,
                vision,
                "APPROACH_VECTOR_WALL",
                tracker,
                phase="align",
                log=False,
                observer=None,
            )

        self.assertFalse(ok)
        self.assertEqual(len(wait_calls), 1)

    def test_new_success_tracker_halves_pass_thresholds_in_aruco_only(self):
        prev_consec = telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED
        prev_majority_window = telemetry_process.GATECHECK_MAJORITY_WINDOW
        prev_majority_required = telemetry_process.GATECHECK_MAJORITY_REQUIRED
        telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED = 6
        telemetry_process.GATECHECK_MAJORITY_WINDOW = 13
        telemetry_process.GATECHECK_MAJORITY_REQUIRED = 5
        telemetry_process.GATECHECK_ARUCO_FULL_PASS_SCALE = 0.5
        try:
            with mock.patch.object(
                telemetry_process,
                "_active_success_gate_mode",
                return_value=telemetry_process.VISION_MODE_ARUCO,
            ):
                tracker = telemetry_process.new_success_tracker("BRICK_LOCK", process_rules={})
        finally:
            telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED = prev_consec
            telemetry_process.GATECHECK_MAJORITY_WINDOW = prev_majority_window
            telemetry_process.GATECHECK_MAJORITY_REQUIRED = prev_majority_required

        self.assertEqual(int(tracker.consecutive_required), 6)
        self.assertEqual(int(tracker.majority_required), 5)
        self.assertEqual(int(getattr(tracker, "consecutive_pass_required", tracker.consecutive_required)), 3)
        self.assertEqual(int(getattr(tracker, "majority_pass_required", tracker.majority_required)), 3)

    def test_new_success_tracker_keeps_pass_thresholds_for_cyan(self):
        prev_consec = telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED
        prev_majority_window = telemetry_process.GATECHECK_MAJORITY_WINDOW
        prev_majority_required = telemetry_process.GATECHECK_MAJORITY_REQUIRED
        telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED = 6
        telemetry_process.GATECHECK_MAJORITY_WINDOW = 13
        telemetry_process.GATECHECK_MAJORITY_REQUIRED = 5
        telemetry_process.GATECHECK_ARUCO_FULL_PASS_SCALE = 0.5
        try:
            with mock.patch.object(
                telemetry_process,
                "_active_success_gate_mode",
                return_value=telemetry_process.VISION_MODE_CYAN,
            ):
                tracker = telemetry_process.new_success_tracker("BRICK_LOCK", process_rules={})
        finally:
            telemetry_process.GATECHECK_CONSECUTIVE_REQUIRED = prev_consec
            telemetry_process.GATECHECK_MAJORITY_WINDOW = prev_majority_window
            telemetry_process.GATECHECK_MAJORITY_REQUIRED = prev_majority_required

        self.assertEqual(int(tracker.consecutive_required), 6)
        self.assertEqual(int(tracker.majority_required), 5)
        self.assertEqual(int(getattr(tracker, "consecutive_pass_required", tracker.consecutive_required)), 6)
        self.assertEqual(int(getattr(tracker, "majority_pass_required", tracker.majority_required)), 5)

    def test_lite_visible_false_gate_fails_when_recent_raw_frames_confidently_see_brick(self):
        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3})
        world = _DummyWorld()
        world.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        world._smoothed_frame_history = [
            {"frame_id": 11, "visible": False, "confidence": 0.0},
            {"frame_id": 12, "visible": False, "confidence": 0.0},
            {"frame_id": 13, "visible": False, "confidence": 0.0},
        ]
        world._raw_brick_visibility_history = [
            {"frame_id": 11, "found": True, "conf": 90.0},
            {"frame_id": 12, "found": True, "conf": 92.0},
            {"frame_id": 13, "found": True, "conf": 95.0},
        ]

        ok, _ = telemetry_process.evaluate_gate_status(world, "EXIT_WALL")
        self.assertFalse(ok)
        self.assertEqual(world._gatecheck_mode, "lite")
        self.assertTrue(bool(getattr(world, "_lite_gate_visible_false_confident_seen", False)))

    def test_lite_visible_false_gate_fails_with_recent_confident_hits_even_with_one_miss(self):
        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3})
        world = _DummyWorld()
        world.process_rules = {
            "EXIT_WALL": {
                "success_gates": {
                    "visible": {"min": False},
                }
            }
        }
        world._smoothed_frame_history = [
            {"frame_id": 21, "visible": False, "confidence": 0.0},
            {"frame_id": 22, "visible": False, "confidence": 0.0},
            {"frame_id": 23, "visible": False, "confidence": 0.0},
        ]
        world._raw_brick_visibility_history = [
            {"frame_id": 20, "found": True, "conf": 90.0},
            {"frame_id": 21, "found": False, "conf": 0.0},
            {"frame_id": 22, "found": True, "conf": 92.0},
            {"frame_id": 23, "found": True, "conf": 95.0},
        ]

        ok, _ = telemetry_process.evaluate_gate_status(world, "EXIT_WALL")
        self.assertFalse(ok)
        self.assertEqual(world._gatecheck_mode, "lite")
        self.assertTrue(bool(getattr(world, "_lite_gate_visible_false_confident_seen", False)))

    def test_lite_gate_dist_target_tol_matches_runtime_directional_truth(self):
        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3})
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                }
            }
        }
        world.brick.update(
            {
                "visible": True,
                "dist": 148.6,
                "x_axis": -6.0,
                "offset_x": -6.0,
                "confidence": 95.0,
            }
        )
        world._smoothed_frame_history = [
            {"frame_id": 31, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 32, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 33, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
        ]

        ok, _ = telemetry_process.evaluate_gate_status(world, "BRICK_LOCK")
        self.assertTrue(ok)
        self.assertEqual(world._gatecheck_mode, "traditional")

    def test_observe_success_gatecheck_starts_full_tracker_once_lite_passes(self):
        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3})
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                }
            }
        }
        world.brick.update(
            {
                "visible": True,
                "dist": 148.6,
                "x_axis": -6.0,
                "offset_x": -6.0,
                "confidence": 95.0,
            }
        )
        world._smoothed_frame_history = [
            {"frame_id": 41, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 42, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 43, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
        ]

        tracker = telemetry_process.new_success_tracker("BRICK_LOCK", world.process_rules)
        result = telemetry_process.observe_success_gatecheck(
            world,
            "BRICK_LOCK",
            tracker,
            phase="align",
            log=False,
        )

        self.assertTrue(bool(result.get("effective_success_ok")))
        self.assertEqual(int(getattr(tracker, "total_checks", 0) or 0), 1)
        status = getattr(world, "_gatecheck_status", {}) or {}
        self.assertEqual(status.get("mode"), "traditional")

    def test_observe_success_gatecheck_holds_when_lite_rows_pass_but_full_not_started(self):
        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3})
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                }
            }
        }
        world._gatecheck_mode = "lite"
        world._gatecheck_lite_required = 3
        world._gatecheck_lite_collected = 3
        world._smoothed_frame_history = [
            {"frame_id": 71, "visible": True, "dist": 154.0, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 72, "visible": True, "dist": 154.5, "x_axis": -6.1, "offset_x": -6.1, "confidence": 95.0},
            {"frame_id": 73, "visible": True, "dist": 154.7, "x_axis": -6.2, "offset_x": -6.2, "confidence": 95.0},
        ]
        tracker = telemetry_process.new_success_tracker("BRICK_LOCK", world.process_rules)

        with mock.patch.object(telemetry_process, "evaluate_gate_status", return_value=(False, 0.0)), \
             mock.patch.object(telemetry_process, "_evaluate_instant_success_truth", return_value=False), \
             mock.patch.object(telemetry_process, "_update_success_gate_metric_tallies", return_value={}), \
             mock.patch.object(telemetry_process, "update_gatecheck_with_precheck", return_value=False):
            result = telemetry_process.observe_success_gatecheck(
                world,
                "BRICK_LOCK",
                tracker,
                phase="align",
                log=False,
            )

        self.assertFalse(bool(result.get("effective_success_ok")))
        self.assertFalse(bool(result.get("success_met")))
        self.assertTrue(bool(result.get("hold_for_confirm")))

    def test_result_lite_gate_detail_uses_equal_for_pass_and_not_equal_for_fail(self):
        telemetry_process.apply_lite_gate_check_config({"default_unique_smoothed_frames": 3})
        world = _DummyWorld()
        world.process_rules = {
            "BRICK_LOCK": {
                "success_gates": {
                    "xAxis_offset_abs": {"target": -6.2, "tol": 4.0},
                    "dist": {"target": 154.6, "tol": 4.0},
                }
            }
        }
        world._smoothed_frame_history = [
            {"frame_id": 51, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 52, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 53, "visible": True, "dist": 148.6, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
        ]
        detail_pass = telemetry_process._result_lite_gate_detail(world, "BRICK_LOCK")
        plain_pass = str((detail_pass or {}).get("plain") or "")
        self.assertIn("xAxis_offset_abs (-6.0mm)=-6.2+/-4.0", plain_pass)
        self.assertIn("dist (148.6mm)=154.6+/-4.0", plain_pass)

        world._smoothed_frame_history = [
            {"frame_id": 61, "visible": True, "dist": 170.0, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 62, "visible": True, "dist": 170.0, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
            {"frame_id": 63, "visible": True, "dist": 170.0, "x_axis": -6.0, "offset_x": -6.0, "confidence": 95.0},
        ]
        detail_fail = telemetry_process._result_lite_gate_detail(world, "BRICK_LOCK")
        plain_fail = str((detail_fail or {}).get("plain") or "")
        self.assertIn("xAxis_offset_abs (-6.0mm)=-6.2+/-4.0", plain_fail)
        self.assertIn("dist (170.0mm)!=154.6+/-4.0", plain_fail)


if __name__ == "__main__":
    unittest.main()
