import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_random_back_turn_experiment as experiment


class TestHelperRandomBackTurnExperiment(unittest.TestCase):
    def test_choose_random_plan_uses_requested_ranges(self):
        plan = experiment.choose_random_back_turn_plan(
            rng=random.Random(7),
            back_range_ms=(500, 1500),
            turn_range_ms=(500, 800),
        )

        self.assertEqual(plan["sequence"][0]["cmd"], "b")
        self.assertIn(plan["turn_cmd"], {"l", "r"})
        self.assertGreaterEqual(plan["back_duration_ms"], 500)
        self.assertLessEqual(plan["back_duration_ms"], 1500)
        self.assertGreaterEqual(plan["turn_duration_ms"], 500)
        self.assertLessEqual(plan["turn_duration_ms"], 800)

    def test_analyze_iteration_reports_improvement_and_hold(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        before_pose = {
            "offset_x": 15.0,
            "offset_y": 8.0,
            "dist": 118.0,
            "angle": 0.0,
            "confidence": 82.0,
            "obs_ts": 10.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        after_pose = {
            "offset_x": 7.2,
            "offset_y": 3.9,
            "dist": 103.8,
            "angle": 0.0,
            "confidence": 84.0,
            "obs_ts": 11.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        analysis, suggestion = experiment.analyze_iteration(
            world=world,
            before_pose=before_pose,
            after_pose=after_pose,
            before_meta={"mode": "primary_full"},
            after_meta={"mode": "primary_full"},
        )

        self.assertTrue(analysis["observation_usable"])
        self.assertEqual(analysis["primary_effect"], "improved_alignment")
        self.assertGreater(float(analysis["progress_after_pct"]), float(analysis["progress_before_pct"]))
        self.assertIn("x_axis", analysis["improved_metrics"])
        self.assertIn("dist", analysis["improved_metrics"])
        self.assertEqual(suggestion["action"], "HOLD")

    def test_run_random_back_turn_experiment_logs_observation_analysis_and_suggestion(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        before_pose = {
            "offset_x": 15.0,
            "offset_y": 6.0,
            "dist": 116.0,
            "angle": 0.0,
            "confidence": 81.0,
            "obs_ts": 20.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        after_pose = {
            "offset_x": 2.0,
            "offset_y": 4.2,
            "dist": 104.0,
            "angle": 0.0,
            "confidence": 83.0,
            "obs_ts": 21.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        def _fake_send_robot_command(*_args, **kwargs):
            cmd = str(_args[3]).lower()
            sent_map = {"b": "f", "l": "r", "r": "l"}
            return {
                "cmd_sent": sent_map[cmd],
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(
            experiment,
            "observe_alignment_pose",
            side_effect=[
                (before_pose, {"mode": "primary_full", "reobserved": False}),
                (after_pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(experiment, "send_robot_command", side_effect=_fake_send_robot_command), patch.object(
            experiment.time,
            "sleep",
            return_value=None,
        ):
            result = experiment.run_random_back_turn_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                score=1,
                rng=random.Random(7),
                seed=7,
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["seed"], 7)
        self.assertEqual(result["vision_mode"], experiment.VISION_MODE_CYAN)
        self.assertIn("observation", result)
        self.assertEqual(result["observation"]["before"]["offset_x"], 15.0)
        self.assertEqual(result["observation"]["after"]["offset_x"], 2.0)
        self.assertIn("analysis", result)
        self.assertEqual(result["analysis"]["before_mode"], "primary_full")
        self.assertEqual(result["analysis"]["after_mode"], "primary_full")
        self.assertIn("suggestion", result)
        self.assertEqual(result["suggestion"]["action"], "L 1%")
        self.assertEqual(len(result["pulses"]), 2)

    def test_x_axis_turn_plan_prefers_turn_and_blocks_other_gate_errors(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        analytics = {
            "cmd": "r",
            "speed_score": 14,
            "worst_metric": "xAxis_offset_abs",
        }
        pose = {
            "offset_x": 22.6,
            "offset_y": 4.1,
            "dist": 105.4,
            "angle": 0.0,
            "confidence": 82.0,
            "obs_ts": 30.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        plan = experiment._x_axis_turn_plan(
            world=world,
            pose=pose,
            analytics=analytics,
            intensity_scale=1.0,
            max_turn_intensity_pct=20.0,
        )

        self.assertEqual(plan["cmd"], "r")
        self.assertEqual(plan["reason"], "targeted_x_axis_turn_tracking")
        self.assertEqual(plan["analytics_speed_score_pct"], 14)
        self.assertTrue(plan["secondary_focus"]["secondary_focus_ready"])
        self.assertGreater(float(plan["turn_intensity_pct"]), 0.0)
        self.assertLessEqual(float(plan["turn_intensity_pct"]), 20.0)

        bplus_pose = dict(pose)
        bplus_pose["dist"] = 106.25
        bplus_plan = experiment._x_axis_turn_plan(
            world=world,
            pose=bplus_pose,
            analytics=analytics,
        )
        self.assertEqual(bplus_plan["cmd"], "r")
        self.assertTrue(bplus_plan["secondary_focus"]["secondary_focus_ready"])

        blocked_pose = dict(pose)
        blocked_pose["dist"] = 110.5
        blocked_plan = experiment._x_axis_turn_plan(
            world=world,
            pose=blocked_pose,
            analytics=analytics,
        )
        self.assertIsNone(blocked_plan["cmd"])
        self.assertEqual(
            blocked_plan["reason"],
            "x_axis_only_experiment_blocked_by_other_gate_errors",
        )

    def test_dist_recovery_plan_prefers_drive_when_y_is_bplus(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        analytics = {
            "cmd": "l",
            "speed_score": 25,
            "worst_metric": "xAxis_offset_abs",
        }
        pose = {
            "offset_x": 4.0,
            "offset_y": 4.0,
            "dist": 108.9,
            "angle": 0.0,
            "confidence": 82.0,
            "obs_ts": 31.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        plan = experiment._dist_recovery_plan(
            world=world,
            pose=pose,
            analytics=analytics,
        )

        self.assertEqual(plan["stage"], "dist_recovery")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["reason"], "targeted_dist_recovery")
        self.assertTrue(plan["secondary_focus"]["y_bplus_ok"])
        self.assertFalse(plan["secondary_focus"]["dist_bplus_ok"])
        self.assertGreaterEqual(int(plan["predicted_score_pct"]), 1)
        self.assertLessEqual(
            int(plan["predicted_score_pct"]),
            int(experiment.DEFAULT_MAX_DRIVE_RECOVERY_SCORE),
        )

    def test_x_turn_plan_uses_real_effective_minimum(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        analytics = {
            "cmd": "l",
            "speed_score": 1,
            "worst_metric": "xAxis_offset_abs",
        }
        pose = {
            "offset_x": 1.9,
            "offset_y": 5.7,
            "dist": 104.3,
            "angle": 0.0,
            "confidence": 88.0,
            "obs_ts": 32.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        plan = experiment._x_axis_turn_plan(
            world=world,
            pose=pose,
            analytics=analytics,
            intensity_scale=0.4,
        )

        self.assertEqual(plan["cmd"], "l")
        self.assertEqual(plan["turn_intensity_pct"], 1.0)

    def test_x_turn_plan_can_force_floor_breakout_probe(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        analytics = {
            "cmd": "l",
            "speed_score": 1,
            "worst_metric": "xAxis_offset_abs",
        }
        pose = {
            "offset_x": 1.9,
            "offset_y": 5.7,
            "dist": 104.3,
            "angle": 0.0,
            "confidence": 88.0,
            "obs_ts": 32.5,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        plan = experiment._x_axis_turn_plan(
            world=world,
            pose=pose,
            analytics=analytics,
            intensity_scale=0.4,
            forced_min_turn_intensity_pct=3.0,
        )

        self.assertEqual(plan["cmd"], "l")
        self.assertEqual(plan["turn_intensity_pct"], 3.0)
        self.assertEqual(plan["forced_floor_breakout_intensity_pct"], 3.0)

    def test_x_turn_breakout_intensity_scales_with_error(self):
        self.assertEqual(
            experiment._x_turn_breakout_intensity_pct(3.5, max_turn_intensity_pct=8.0),
            2.0,
        )
        self.assertEqual(
            experiment._x_turn_breakout_intensity_pct(6.0, max_turn_intensity_pct=8.0),
            3.0,
        )
        self.assertEqual(
            experiment._x_turn_breakout_intensity_pct(12.0, max_turn_intensity_pct=8.0),
            4.0,
        )

    def test_select_turn_cmd_holds_after_single_small_worsen(self):
        controller = experiment._init_x_lock_controller()
        experiment._update_x_lock_controller(
            controller,
            cmd="l",
            x_improvement_mm=0.574,
            min_progress_mm=experiment.DEFAULT_MIN_PROGRESS_MM,
            worsen_mm=experiment.DEFAULT_DIRECTION_FLIP_WORSEN_MM,
        )
        experiment._update_x_lock_controller(
            controller,
            cmd="l",
            x_improvement_mm=-0.355,
            min_progress_mm=experiment.DEFAULT_MIN_PROGRESS_MM,
            worsen_mm=experiment.DEFAULT_DIRECTION_FLIP_WORSEN_MM,
        )

        cmd, source = experiment._select_turn_cmd(controller, analytics_cmd="l")

        self.assertEqual(cmd, "l")
        self.assertEqual(source, "hold_positive_mean")

    def test_select_turn_cmd_flips_after_single_hard_worsen(self):
        controller = experiment._init_x_lock_controller()
        experiment._update_x_lock_controller(
            controller,
            cmd="l",
            x_improvement_mm=-0.9,
            min_progress_mm=experiment.DEFAULT_MIN_PROGRESS_MM,
            worsen_mm=experiment.DEFAULT_DIRECTION_FLIP_WORSEN_MM,
        )

        cmd, source = experiment._select_turn_cmd(controller, analytics_cmd="l")

        self.assertEqual(cmd, "r")
        self.assertEqual(source, "bias_away_hard")

    def test_run_x_axis_lock_experiment_tracks_cycles_until_hold(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        pose_1 = {
            "offset_x": 22.8,
            "offset_y": 4.1,
            "dist": 105.5,
            "angle": 0.4,
            "confidence": 86.0,
            "obs_ts": 40.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        pose_2 = {
            "offset_x": 8.8,
            "offset_y": 4.0,
            "dist": 105.3,
            "angle": 0.2,
            "confidence": 85.0,
            "obs_ts": 41.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        pose_3 = {
            "offset_x": 7.2,
            "offset_y": 3.9,
            "dist": 103.8,
            "angle": 0.1,
            "confidence": 84.0,
            "obs_ts": 42.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        def _fake_send_robot_command(*_args, **kwargs):
            cmd = str(_args[3]).lower()
            sent_map = {"l": "r", "r": "l"}
            return {
                "cmd_sent": sent_map.get(cmd, cmd),
                "duration_ms": 90,
                "score_effective": int(kwargs.get("speed_score") or 0),
                "turn_intensity_requested": kwargs.get("turn_intensity"),
                "turn_intensity_effective": kwargs.get("turn_intensity"),
            }

        with patch.object(
            experiment,
            "observe_alignment_pose",
            side_effect=[
                (pose_1, {"mode": "primary_full", "reobserved": False}),
                (pose_2, {"mode": "primary_full", "reobserved": False}),
                (pose_3, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(experiment, "send_robot_command", side_effect=_fake_send_robot_command), patch.object(
            experiment.time,
            "sleep",
            return_value=None,
        ):
            result = experiment.run_x_axis_lock_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                max_cycles=4,
                log_path=None,
                memory_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["experiment_type"], "targeted_x_axis_lock_tracking")
        self.assertEqual(result["stop_reason"], "locked")
        self.assertEqual(result["cycles_completed"], 2)
        self.assertEqual(len(result["cycles"]), 2)
        self.assertEqual(result["suggestion"]["action"], "HOLD")
        self.assertEqual(result["observation"]["before"]["offset_x"], 22.8)
        self.assertEqual(result["observation"]["after"]["offset_x"], 7.2)
        self.assertGreater(
            float(result["analysis"]["gate_error_before_mm"]["x_axis"]),
            float(result["analysis"]["gate_error_after_mm"]["x_axis"]),
        )

    def test_run_x_axis_lock_experiment_hands_off_from_dist_recovery_to_hold(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        pose_1 = {
            "offset_x": 4.0,
            "offset_y": 4.0,
            "dist": 108.9,
            "angle": 0.0,
            "confidence": 86.0,
            "obs_ts": 50.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        pose_2 = {
            "offset_x": 7.0,
            "offset_y": 4.0,
            "dist": 104.0,
            "angle": 0.0,
            "confidence": 85.0,
            "obs_ts": 51.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        def _fake_send_robot_command(*_args, **kwargs):
            return {
                "cmd_sent": str(_args[3]).lower(),
                "duration_ms": 90,
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(
            experiment,
            "observe_alignment_pose",
            side_effect=[
                (pose_1, {"mode": "primary_full", "reobserved": False}),
                (pose_2, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(experiment, "send_robot_command", side_effect=_fake_send_robot_command), patch.object(
            experiment.time,
            "sleep",
            return_value=None,
        ):
            result = experiment.run_x_axis_lock_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                max_cycles=4,
                log_path=None,
                memory_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stop_reason"], "locked")
        self.assertEqual(result["cycles_completed"], 1)
        self.assertEqual(result["pulses"][0]["cmd"], "f")
        self.assertEqual(result["pulses"][0]["stage"], "dist_recovery")
        self.assertEqual(result["suggestion"]["action"], "HOLD")

    def test_x_turn_uses_stronger_post_move_reacquire_policy(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        pose_1 = {
            "offset_x": 22.8,
            "offset_y": 4.1,
            "dist": 105.5,
            "angle": 0.4,
            "confidence": 86.0,
            "obs_ts": 60.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        pose_2 = {
            "offset_x": 7.2,
            "offset_y": 3.9,
            "dist": 103.8,
            "angle": 0.1,
            "confidence": 84.0,
            "obs_ts": 61.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        observe_calls = []

        def _fake_observe_alignment_pose(*args, **kwargs):
            observe_calls.append(
                {
                    "timeout_s": kwargs.get("timeout_s"),
                    "relaxed_timeout_s": kwargs.get("relaxed_timeout_s"),
                    "reobserve_rounds": kwargs.get("reobserve_rounds"),
                }
            )
            if len(observe_calls) == 1:
                return pose_1, {"mode": "primary_full", "reobserved": False}
            return pose_2, {"mode": "primary_full", "reobserved": False}

        def _fake_send_robot_command(*_args, **kwargs):
            cmd = str(_args[3]).lower()
            sent_map = {"l": "r", "r": "l"}
            return {
                "cmd_sent": sent_map.get(cmd, cmd),
                "duration_ms": 90,
                "score_effective": int(kwargs.get("speed_score") or 0),
                "turn_intensity_requested": kwargs.get("turn_intensity"),
                "turn_intensity_effective": kwargs.get("turn_intensity"),
            }

        with patch.object(
            experiment,
            "observe_alignment_pose",
            side_effect=_fake_observe_alignment_pose,
        ), patch.object(experiment, "send_robot_command", side_effect=_fake_send_robot_command), patch.object(
            experiment.time,
            "sleep",
            return_value=None,
        ):
            result = experiment.run_x_axis_lock_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                max_cycles=2,
                log_path=None,
                memory_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(observe_calls), 2)
        self.assertEqual(
            observe_calls[1]["reobserve_rounds"],
            experiment.DEFAULT_X_TURN_POST_MOVE_REOBSERVE_ROUNDS,
        )
        self.assertEqual(
            observe_calls[1]["timeout_s"],
            experiment.DEFAULT_X_TURN_POST_MOVE_OBSERVE_TIMEOUT_S,
        )
        self.assertEqual(
            observe_calls[1]["relaxed_timeout_s"],
            experiment.DEFAULT_X_TURN_POST_MOVE_RELAXED_TIMEOUT_S,
        )
        self.assertEqual(
            result["cycles"][0]["post_move_observe_policy"]["policy"],
            "x_turn_stronger_reacquire",
        )

    def test_controller_memory_round_trip_preserves_trusted_bias(self):
        controller = experiment._init_x_lock_controller()
        experiment._update_x_lock_controller(
            controller,
            cmd="r",
            x_improvement_mm=0.8,
            min_progress_mm=experiment.DEFAULT_MIN_PROGRESS_MM,
            worsen_mm=experiment.DEFAULT_DIRECTION_FLIP_WORSEN_MM,
        )
        experiment._update_x_lock_controller(
            controller,
            cmd="r",
            x_improvement_mm=0.7,
            min_progress_mm=experiment.DEFAULT_MIN_PROGRESS_MM,
            worsen_mm=experiment.DEFAULT_DIRECTION_FLIP_WORSEN_MM,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.json"
            saved_path = experiment._save_x_lock_memory(
                memory_path,
                controller=controller,
                stop_reason="test",
                final_pose={"offset_x": 10.0, "offset_y": 4.0, "dist": 104.0},
            )
            self.assertEqual(saved_path, str(memory_path))
            restored = experiment._load_x_lock_memory(memory_path)

        restored_snapshot = experiment._controller_snapshot(restored)
        self.assertEqual(restored_snapshot["trusted_cmd"], "r")
        self.assertGreaterEqual(restored_snapshot["trusted_confirmations"], 2)
        self.assertGreater(
            float(restored_snapshot["cmd_stats"]["r"]["recent_mean_improvement_mm"]),
            0.0,
        )
        self.assertEqual(restored_snapshot["cmd_stats"]["r"]["improve_streak"], 0)
        self.assertEqual(restored_snapshot["cmd_stats"]["r"]["worsen_streak"], 0)
        self.assertIsNone(restored_snapshot["cmd_stats"]["r"]["last_outcome"])

    def test_summarize_motion_phase_reports_best_errors(self):
        summary = experiment._summarize_motion_phase(
            [
                {
                    "visible": True,
                    "confidence": 81.0,
                    "progress_pct": 22.0,
                    "gate_error_mm": {"x_axis": 6.0, "y_axis": 0.2, "dist": 1.4},
                },
                {
                    "visible": True,
                    "confidence": 85.0,
                    "progress_pct": 37.0,
                    "gate_error_mm": {"x_axis": 4.0, "y_axis": 0.0, "dist": 0.5},
                },
                {
                    "visible": False,
                    "confidence": None,
                    "progress_pct": None,
                    "gate_error_mm": None,
                },
            ]
        )

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["visible_sample_count"], 2)
        self.assertEqual(summary["visible_rate"], 0.667)
        self.assertEqual(summary["x_gate_error_mm"]["best"], 4.0)
        self.assertEqual(summary["dist_gate_error_mm"]["end"], 0.5)
        self.assertEqual(summary["confidence"]["median"], 83.0)

    def test_run_observe_while_moving_experiment_records_motion_samples(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        initial_pose = {
            "offset_x": 10.0,
            "offset_y": 4.0,
            "dist": 106.5,
            "angle": 0.0,
            "confidence": 82.0,
            "obs_ts": 70.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        mid_pose = {
            "offset_x": 8.0,
            "offset_y": 4.1,
            "dist": 104.5,
            "angle": 0.1,
            "confidence": 83.0,
            "obs_ts": 71.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }
        final_pose = {
            "offset_x": 9.0,
            "offset_y": 4.0,
            "dist": 105.0,
            "angle": 0.2,
            "confidence": 84.0,
            "obs_ts": 72.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 1000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        fake_clock = FakeClock()
        sample_states = [
            {"visible": True, "offset_x": 9.8, "offset_y": 4.0, "dist": 106.0, "angle": 0.0, "confidence": 81.0},
            {"visible": True, "offset_x": 9.1, "offset_y": 4.0, "dist": 105.2, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": 8.7, "offset_y": 4.1, "dist": 104.8, "angle": 0.1, "confidence": 83.0},
            {"visible": True, "offset_x": 8.9, "offset_y": 4.0, "dist": 105.0, "angle": 0.1, "confidence": 84.0},
        ]
        update_count = {"value": 0}

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            idx = min(update_count["value"], len(sample_states) - 1)
            state = sample_states[idx]
            update_count["value"] += 1
            world_obj.brick = {
                "visible": bool(state["visible"]),
                "offset_x": float(state["offset_x"]),
                "x_axis": float(state["offset_x"]),
                "offset_y": float(state["offset_y"]),
                "y_axis": float(state["offset_y"]),
                "dist": float(state["dist"]),
                "angle": float(state["angle"]),
                "confidence": float(state["confidence"]),
                "pose_source": "brick_state",
            }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        def _fake_send_robot_command(*_args, **kwargs):
            return {
                "cmd_sent": str(kwargs.get("cmd")).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(
            experiment,
            "observe_alignment_pose",
            side_effect=[
                (initial_pose, {"mode": "primary_full", "reobserved": False}),
                (mid_pose, {"mode": "primary_full", "reobserved": False}),
                (final_pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment.run_observe_while_moving_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                score=1,
                phase_duration_ms=200,
                sample_hz=10.0,
                trials=1,
                cmd_sequence=("f", "b"),
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["experiment_type"], "observe_while_moving_probe")
        self.assertEqual(result["cmd_sequence"], ["f", "b"])
        self.assertEqual(len(result["phases"]), 2)
        self.assertGreater(len(result["motion_samples"]), 0)
        self.assertEqual(result["phases"][0]["cmd"], "f")
        self.assertGreater(result["phases"][0]["motion_sample_count"], 0)
        self.assertEqual(result["observation"]["before"]["offset_x"], 10.0)
        self.assertEqual(result["observation"]["after"]["offset_x"], 9.0)

    def test_run_observe_while_moving_accepts_all_motion_axes(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        pose = {
            "offset_x": 1.0,
            "offset_y": 4.0,
            "dist": 105.0,
            "angle": 0.0,
            "confidence": 82.0,
            "obs_ts": 80.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 2000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        fake_clock = FakeClock()

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            world_obj.brick = {
                "visible": True,
                "offset_x": 1.0,
                "x_axis": 1.0,
                "offset_y": 4.0,
                "y_axis": 4.0,
                "dist": 105.0,
                "angle": 0.0,
                "confidence": 82.0,
                "pose_source": "brick_state",
            }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        def _fake_send_robot_command(*_args, **kwargs):
            return {
                "cmd_sent": str(kwargs.get("cmd")).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(
            experiment,
            "observe_alignment_pose",
            side_effect=[
                (pose, {"mode": "primary_full", "reobserved": False}),
                (pose, {"mode": "primary_full", "reobserved": False}),
                (pose, {"mode": "primary_full", "reobserved": False}),
                (pose, {"mode": "primary_full", "reobserved": False}),
                (pose, {"mode": "primary_full", "reobserved": False}),
                (pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment.run_observe_while_moving_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                score=1,
                phase_duration_ms=100,
                sample_hz=10.0,
                trials=1,
                cmd_sequence=("l", "u", "f"),
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["cmd_sequence"], ["l", "u", "f"])
        self.assertEqual([row["axis"] for row in result["phases"]], ["x", "y", "dist"])

    def test_build_moving_phase_specs_groups_by_axis(self):
        phase_specs = experiment._build_moving_phase_specs(
            cmd_sequence=("l", "r", "u", "d", "f", "b"),
            trials=2,
            group_by_axis=True,
        )

        self.assertEqual(
            [(row["axis"], row["cmd"], row["trial"]) for row in phase_specs[:6]],
            [
                ("x", "l", 1),
                ("x", "r", 1),
                ("x", "l", 2),
                ("x", "r", 2),
                ("y", "u", 1),
                ("y", "d", 1),
            ],
        )
        self.assertEqual(phase_specs[-1]["axis"], "dist")

    def test_turn_phase_retriggers_multiple_command_events(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK
        pose = {
            "offset_x": 1.0,
            "offset_y": 4.0,
            "dist": 105.0,
            "angle": 0.0,
            "confidence": 82.0,
            "obs_ts": 90.0,
            "pose_source": "lite_smoothed",
            "samples_used": 3,
            "lite_required_frames": 3,
        }

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 3000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        fake_clock = FakeClock()

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            world_obj.brick = {
                "visible": True,
                "offset_x": 1.0,
                "x_axis": 1.0,
                "offset_y": 4.0,
                "y_axis": 4.0,
                "dist": 105.0,
                "angle": 0.0,
                "confidence": 82.0,
                "pose_source": "brick_state",
            }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        send_calls = []

        def _fake_send_robot_command(*_args, **kwargs):
            send_calls.append(kwargs.get("duration_override_ms"))
            return {
                "cmd_sent": str(kwargs.get("cmd")).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(
            experiment,
            "observe_alignment_pose",
            side_effect=[
                (pose, {"mode": "primary_full", "reobserved": False}),
                (pose, {"mode": "primary_full", "reobserved": False}),
            ],
        ), patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment.run_observe_while_moving_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                score=1,
                phase_duration_ms=700,
                sample_hz=10.0,
                trials=1,
                cmd_sequence=("l",),
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(send_calls), 2)
        self.assertGreaterEqual(result["phases"][0]["command_send_count"], 2)

    def test_run_single_goal_right_turn_experiment_stops_on_zero_band(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 4000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        fake_clock = FakeClock()
        states = [
            {"visible": True, "offset_x": -12.0, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -14.0, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": False, "offset_x": None, "offset_y": None, "dist": None, "angle": None, "confidence": None},
            {"visible": False, "offset_x": None, "offset_y": None, "dist": None, "angle": None, "confidence": None},
            {"visible": False, "offset_x": None, "offset_y": None, "dist": None, "angle": None, "confidence": None},
            {"visible": True, "offset_x": -0.8, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 83.0},
            {"visible": True, "offset_x": 0.2, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
            {"visible": True, "offset_x": 0.2, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
        ]
        update_idx = {"value": 0}

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            idx = min(update_idx["value"], len(states) - 1)
            state = states[idx]
            update_idx["value"] += 1
            if bool(state["visible"]):
                world_obj.brick = {
                    "visible": True,
                    "offset_x": float(state["offset_x"]),
                    "x_axis": float(state["offset_x"]),
                    "offset_y": float(state["offset_y"]),
                    "y_axis": float(state["offset_y"]),
                    "dist": float(state["dist"]),
                    "angle": float(state["angle"]),
                    "confidence": float(state["confidence"]),
                    "pose_source": "brick_state",
                }
            else:
                world_obj.brick = {
                    "visible": False,
                    "offset_x": 0.0,
                    "x_axis": 0.0,
                    "offset_y": 0.0,
                    "y_axis": 0.0,
                    "dist": 0.0,
                    "angle": 0.0,
                    "confidence": 0.0,
                    "pose_source": "brick_state",
                }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        def _fake_send_robot_command(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": 1,
                "turn_intensity_effective": kwargs.get("turn_intensity"),
            }

        with patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment.run_single_goal_right_turn_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                trials=1,
                sample_hz=10.0,
                reset_timeout_s=0.4,
                attempt_timeout_s=0.5,
                zero_band_mm=0.5,
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["experiment_type"], "single_goal_right_turn_x_zero")
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["trials"][0]["reset"]["status"], "lost_visibility")
        self.assertEqual(result["trials"][0]["attempt"]["status"], "x_axis_zero_band")
        self.assertEqual(result["trials"][0]["attempt"]["best_visible_abs_x_mm"], 0.2)
        self.assertEqual(len(result["pulses"]), 2)

    def test_run_single_goal_right_turn_experiment_flags_first_visible_positive_overshoot(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 5000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        fake_clock = FakeClock()
        states = [
            {"visible": True, "offset_x": -9.0, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": False, "offset_x": None, "offset_y": None, "dist": None, "angle": None, "confidence": None},
            {"visible": False, "offset_x": None, "offset_y": None, "dist": None, "angle": None, "confidence": None},
            {"visible": True, "offset_x": 2.4, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
            {"visible": True, "offset_x": 2.4, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
        ]
        update_idx = {"value": 0}

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            idx = min(update_idx["value"], len(states) - 1)
            state = states[idx]
            update_idx["value"] += 1
            if bool(state["visible"]):
                world_obj.brick = {
                    "visible": True,
                    "offset_x": float(state["offset_x"]),
                    "x_axis": float(state["offset_x"]),
                    "offset_y": float(state["offset_y"]),
                    "y_axis": float(state["offset_y"]),
                    "dist": float(state["dist"]),
                    "angle": float(state["angle"]),
                    "confidence": float(state["confidence"]),
                    "pose_source": "brick_state",
                }
            else:
                world_obj.brick = {
                    "visible": False,
                    "offset_x": 0.0,
                    "x_axis": 0.0,
                    "offset_y": 0.0,
                    "y_axis": 0.0,
                    "dist": 0.0,
                    "angle": 0.0,
                    "confidence": 0.0,
                    "pose_source": "brick_state",
                }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        def _fake_send_robot_command(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": 1,
                "turn_intensity_effective": kwargs.get("turn_intensity"),
            }

        with patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment.run_single_goal_right_turn_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                trials=1,
                sample_hz=10.0,
                reset_timeout_s=0.3,
                attempt_timeout_s=0.4,
                zero_band_mm=0.5,
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["success_count"], 0)
        self.assertEqual(result["trials"][0]["attempt"]["status"], "first_visible_positive_overshoot")
        self.assertEqual(result["analysis"]["status_counts"]["first_visible_positive_overshoot"], 1)
        self.assertEqual(
            result["suggestion"]["action"],
            "increase_sampling_or_reduce_reset_depth",
        )

    def test_run_single_goal_right_turn_experiment_accepts_left_edge_plateau_reset(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 6000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        fake_clock = FakeClock()
        states = [
            {"visible": True, "offset_x": -34.7, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -34.9, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -35.10, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -35.18, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -35.16, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -35.14, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -35.15, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": True, "offset_x": -9.0, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 83.0},
            {"visible": True, "offset_x": -0.1, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
            {"visible": True, "offset_x": -0.1, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
        ]
        update_idx = {"value": 0}

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            idx = min(update_idx["value"], len(states) - 1)
            state = states[idx]
            update_idx["value"] += 1
            world_obj.brick = {
                "visible": bool(state["visible"]),
                "offset_x": float(state["offset_x"]),
                "x_axis": float(state["offset_x"]),
                "offset_y": float(state["offset_y"]),
                "y_axis": float(state["offset_y"]),
                "dist": float(state["dist"]),
                "angle": float(state["angle"]),
                "confidence": float(state["confidence"]),
                "pose_source": "brick_state",
            }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        def _fake_send_robot_command(*args, **kwargs):
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": 1,
                "turn_intensity_effective": kwargs.get("turn_intensity"),
            }

        with patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment.run_single_goal_right_turn_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                trials=1,
                sample_hz=10.0,
                reset_timeout_s=0.8,
                attempt_timeout_s=0.5,
                zero_band_mm=0.5,
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["trials"][0]["reset"]["status"], "left_edge_plateau")
        self.assertEqual(result["trials"][0]["attempt"]["status"], "x_axis_zero_band")
        self.assertEqual(result["pulses"][0]["stage"], "reset_left_to_edge")
        self.assertEqual(result["pulses"][1]["stage"], "right_turn_to_x_zero")

    def test_x_zero_turn_stage_uses_speed_score_for_integer_one_percent_turns(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 7000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        fake_clock = FakeClock()
        states = [
            {"visible": True, "offset_x": -9.0, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 82.0},
            {"visible": False, "offset_x": None, "offset_y": None, "dist": None, "angle": None, "confidence": None},
            {"visible": False, "offset_x": None, "offset_y": None, "dist": None, "angle": None, "confidence": None},
            {"visible": True, "offset_x": -0.2, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
            {"visible": True, "offset_x": -0.2, "offset_y": 5.0, "dist": 105.0, "angle": 0.0, "confidence": 84.0},
        ]
        update_idx = {"value": 0}
        send_calls = []

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            idx = min(update_idx["value"], len(states) - 1)
            state = states[idx]
            update_idx["value"] += 1
            if bool(state["visible"]):
                world_obj.brick = {
                    "visible": True,
                    "offset_x": float(state["offset_x"]),
                    "x_axis": float(state["offset_x"]),
                    "offset_y": float(state["offset_y"]),
                    "y_axis": float(state["offset_y"]),
                    "dist": float(state["dist"]),
                    "angle": float(state["angle"]),
                    "confidence": float(state["confidence"]),
                    "pose_source": "brick_state",
                }
            else:
                world_obj.brick = {
                    "visible": False,
                    "offset_x": 0.0,
                    "x_axis": 0.0,
                    "offset_y": 0.0,
                    "y_axis": 0.0,
                    "dist": 0.0,
                    "angle": 0.0,
                    "confidence": 0.0,
                    "pose_source": "brick_state",
                }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        def _fake_send_robot_command(*args, **kwargs):
            send_calls.append(dict(kwargs))
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(kwargs.get("duration_override_ms") or 0),
                "score_effective": int(kwargs.get("speed_score") or 0),
                "turn_intensity_effective": kwargs.get("turn_intensity"),
            }

        with patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment.run_single_goal_right_turn_experiment(
                robot=object(),
                world=world,
                vision=object(),
                vision_mode=experiment.VISION_MODE_CYAN,
                trials=1,
                sample_hz=10.0,
                reset_timeout_s=0.3,
                attempt_timeout_s=0.4,
                zero_band_mm=0.5,
                log_path=None,
                log_fn=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(len(send_calls), 2)
        self.assertEqual(send_calls[0].get("speed_score"), 1)
        self.assertEqual(send_calls[1].get("speed_score"), 1)
        self.assertNotIn("turn_intensity", send_calls[0])
        self.assertNotIn("turn_intensity", send_calls[1])

    def test_x_zero_turn_stage_uses_modeled_score_duration_with_888ms_ceiling(self):
        world = experiment.WorldModel()
        world.step_state = experiment.StepState.ALIGN_BRICK

        class FakeClock:
            def __init__(self):
                self.now = 0.0
                self.epoch = 9000.0

            def monotonic(self):
                self.now += 0.05
                return self.now

            def time(self):
                self.epoch += 0.05
                return self.epoch

            def sleep(self, seconds):
                self.now += max(0.0, float(seconds))
                self.epoch += max(0.0, float(seconds))

        class FakeRobot:
            def stop(self):
                return None

        fake_clock = FakeClock()
        send_calls = []

        def _fake_update_world_from_vision(world_obj, _vision_obj, log=False):
            world_obj.brick = {
                "visible": False,
                "offset_x": 0.0,
                "x_axis": 0.0,
                "offset_y": 0.0,
                "y_axis": 0.0,
                "dist": 0.0,
                "angle": 0.0,
                "confidence": 0.0,
                "pose_source": "brick_state",
            }
            world_obj._vision_backend = experiment.VISION_MODE_CYAN
            world_obj._camera_fps = 20.0

        def _fake_send_robot_command(*args, **kwargs):
            duration_ms = int(kwargs.get("duration_override_ms") or 0)
            send_calls.append(duration_ms)
            return {
                "cmd_sent": str(args[3]).lower(),
                "duration_ms": int(duration_ms),
                "score_effective": int(kwargs.get("speed_score") or 0),
            }

        with patch.object(experiment, "update_world_from_vision", side_effect=_fake_update_world_from_vision), patch.object(
            experiment,
            "send_robot_command",
            side_effect=_fake_send_robot_command,
        ), patch.object(
            experiment.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), patch.object(
            experiment.time,
            "time",
            side_effect=fake_clock.time,
        ), patch.object(
            experiment.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            result = experiment._run_turn_stage(
                robot=FakeRobot(),
                world=world,
                vision=object(),
                trial_index=1,
                stage="right_turn_to_x_zero",
                cmd="r",
                turn_intensity_pct=1.0,
                timeout_s=2.2,
                sample_hz=5.0,
                run_started_monotonic=0.0,
                stop_evaluator=None,
            )

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(len(send_calls), 3)
        self.assertTrue(all(int(ms) <= 130 for ms in send_calls))
        self.assertIn(130, send_calls)
        self.assertEqual(len(result.get("send_results") or []), len(send_calls))


if __name__ == "__main__":
    unittest.main()
