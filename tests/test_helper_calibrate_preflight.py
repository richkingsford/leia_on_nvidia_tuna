import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from calibration import helper_calibrate


class _VisionSequence:
    def __init__(self, values, *, metric="cam_h"):
        self._values = list(values)
        self._metric = str(metric)

    def read(self):
        if not self._values:
            value = 0.0
        else:
            value = self._values.pop(0)
        dist = 100.0
        offset_x = 0.0
        cam_h = 0.0
        if self._metric == "dist":
            dist = float(value)
        elif self._metric == "x_axis":
            offset_x = float(value)
        else:
            cam_h = float(value)
        return True, 0.0, dist, offset_x, 90.0, cam_h, 0.0, 0.0


class _WorldStub:
    def update_vision(self, *_args, **_kwargs):
        return None


class _WorldPoseStub:
    def __init__(self):
        self.step_state = "ALIGN_BRICK"
        self.brick = None

    def update_vision(self, *_args, **_kwargs):
        return None


class Check1PctSpeedMovementTests(unittest.TestCase):
    def test_preflight_escalates_scores_and_keeps_fixed_duration(self):
        vision = _VisionSequence(
            [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.2,
                10.25,
                10.3,
            ]
        )
        world = _WorldStub()
        sent = []

        def _fake_speed_power_pwm_for_cmd(cmd, score):
            return 0.1 * float(score), 30 + int(score), int(score), 300

        def _fake_send_robot_command_pwm(_robot, _world, _step, cmd, power, pwm, duration_ms, **kwargs):
            sent.append(
                {
                    "cmd": cmd,
                    "power": power,
                    "pwm": pwm,
                    "duration_ms": duration_ms,
                    "speed_score": kwargs.get("speed_score"),
                }
            )

        with patch("telemetry_robot.speed_power_pwm_for_cmd", side_effect=_fake_speed_power_pwm_for_cmd), patch(
            "telemetry_process.send_robot_command_pwm", side_effect=_fake_send_robot_command_pwm
        ), patch.object(helper_calibrate.time, "sleep"):
            result = helper_calibrate.check_1pct_speed_movement(
                robot=object(),
                vision=vision,
                world=world,
                cmd="u",
                movement_threshold_mm=0.15,
                sample_frames=3,
                sample_timeout_s=1.5,
                observe_sleep_s=0.0,
                control_sleep_s=0.0,
                score_candidates=[1, 2],
                duration_override_ms=250,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["score_used"], 2)
        self.assertEqual(result["duration_ms"], 250)
        self.assertEqual(result["attempt_idx"], 2)
        self.assertEqual([item["speed_score"] for item in sent], [1, 2])
        self.assertEqual([item["duration_ms"] for item in sent], [250, 250])

    def test_preflight_returns_none_after_exhausting_candidates(self):
        vision = _VisionSequence(
            [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ]
        )
        world = _WorldStub()
        sent_scores = []

        def _fake_speed_power_pwm_for_cmd(cmd, score):
            return 0.1 * float(score), 30 + int(score), int(score), 300

        def _fake_send_robot_command_pwm(_robot, _world, _step, _cmd, _power, _pwm, _duration_ms, **kwargs):
            sent_scores.append(int(kwargs.get("speed_score") or 0))

        with patch("telemetry_robot.speed_power_pwm_for_cmd", side_effect=_fake_speed_power_pwm_for_cmd), patch(
            "telemetry_process.send_robot_command_pwm", side_effect=_fake_send_robot_command_pwm
        ), patch.object(helper_calibrate.time, "sleep"):
            result = helper_calibrate.check_1pct_speed_movement(
                robot=object(),
                vision=vision,
                world=world,
                cmd="u",
                movement_threshold_mm=0.15,
                sample_frames=3,
                sample_timeout_s=1.5,
                observe_sleep_s=0.0,
                control_sleep_s=0.0,
                score_candidates=[1, 2],
                duration_override_ms=250,
            )

        self.assertIsNone(result)
        self.assertEqual(sent_scores, [1, 2])

    def test_prediction_closeness_percentage_reports_match_quality(self):
        self.assertAlmostEqual(
            helper_calibrate.prediction_closeness_percentage(
                actual_distance_mm=1.8,
                predicted_distance_mm=2.0,
            ),
            90.0,
        )
        self.assertAlmostEqual(
            helper_calibrate.prediction_closeness_percentage(
                actual_distance_mm=4.5,
                predicted_distance_mm=2.0,
            ),
            0.0,
        )
        self.assertIsNone(
            helper_calibrate.prediction_closeness_percentage(
                actual_distance_mm=1.0,
                predicted_distance_mm=0.0,
            )
        )

    def test_load_calibration_trial_speed_profile_reads_y_axis_curve(self):
        payload = {
            "calibration_trial_speed_profiles": {
                "y_axis": {
                    "metric": "brick_distance_mm",
                    "curve_points": [
                        {"distance_mm": 120.0, "speed_score": 1},
                        {"distance_mm": 250.0, "speed_score": 100},
                    ],
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            curve_path = Path(tmpdir) / "world_model_robot.json"
            curve_path.write_text(json.dumps(payload))
            profile = helper_calibrate.load_calibration_trial_speed_profile("y_axis", path=curve_path)
        self.assertEqual(profile["metric"], "brick_distance_mm")
        self.assertEqual(
            profile["curve_points"],
            [
                {"distance_mm": 120.0, "speed_score": 1},
                {"distance_mm": 250.0, "speed_score": 100},
            ],
        )

    def test_repo_y_axis_trial_speed_profile_ramps_above_1pct_by_105mm(self):
        profile = helper_calibrate.load_calibration_trial_speed_profile("y_axis")
        self.assertIsNotNone(profile)
        score, meta = helper_calibrate.resolve_calibration_trial_speed_score(
            observed_distance_mm=105.17169186564179,
            requested_score=1,
            speed_profile=profile,
        )
        self.assertGreater(score, 1)
        self.assertEqual(meta["source"], "distance_curve")

    def test_resolve_calibration_trial_speed_score_clamps_and_interpolates(self):
        profile = {
            "metric": "brick_distance_mm",
            "curve_points": [
                {"distance_mm": 120.0, "speed_score": 1},
                {"distance_mm": 250.0, "speed_score": 100},
            ],
        }
        near_score, near_meta = helper_calibrate.resolve_calibration_trial_speed_score(
            observed_distance_mm=100.0,
            requested_score=5,
            speed_profile=profile,
        )
        mid_score, mid_meta = helper_calibrate.resolve_calibration_trial_speed_score(
            observed_distance_mm=186.0,
            requested_score=5,
            speed_profile=profile,
        )
        far_score, far_meta = helper_calibrate.resolve_calibration_trial_speed_score(
            observed_distance_mm=300.0,
            requested_score=5,
            speed_profile=profile,
        )
        fallback_score, fallback_meta = helper_calibrate.resolve_calibration_trial_speed_score(
            observed_distance_mm=None,
            requested_score=5,
            speed_profile=profile,
        )
        self.assertEqual(near_score, 1)
        self.assertEqual(mid_score, 51)
        self.assertEqual(far_score, 100)
        self.assertEqual(fallback_score, 5)
        self.assertEqual(near_meta["source"], "distance_curve")
        self.assertEqual(mid_meta["source"], "distance_curve")
        self.assertEqual(far_meta["source"], "distance_curve")
        self.assertEqual(fallback_meta["source"], "arg")

    def test_prepare_shared_stream_state_sets_defaults(self):
        stream_state = {}
        prepared = helper_calibrate.prepare_shared_stream_state(stream_state, vision_mode="cyan")
        self.assertIs(prepared, stream_state)
        self.assertIn("lock", stream_state)
        self.assertTrue(isinstance(stream_state["lock"], threading.Lock().__class__))
        self.assertIsNone(stream_state.get("frame"))
        self.assertEqual(stream_state.get("text_lines"), [])
        self.assertIsNone(stream_state.get("xyz_workspace"))
        self.assertTrue(stream_state.get("show_center_line"))
        self.assertEqual(stream_state.get("vision_mode"), "cyan")

    def test_use_shared_stream_runtime_restores_previous_runtime(self):
        original_state = {"lock": threading.Lock(), "frame": None, "text_lines": []}
        nested_state = {"lock": threading.Lock(), "frame": None, "text_lines": []}
        helper_calibrate.set_shared_stream_runtime(
            stream_state=original_state,
            stream_url="http://127.0.0.1:5000",
        )
        try:
            with helper_calibrate.use_shared_stream_runtime(
                stream_state=nested_state,
                stream_url="http://127.0.0.1:5001",
            ):
                active_state, active_url = helper_calibrate.get_shared_stream_runtime()
                self.assertIs(active_state, nested_state)
                self.assertEqual(active_url, "http://127.0.0.1:5001")
            restored_state, restored_url = helper_calibrate.get_shared_stream_runtime()
            self.assertIs(restored_state, original_state)
            self.assertEqual(restored_url, "http://127.0.0.1:5000")
        finally:
            helper_calibrate.set_shared_stream_runtime(stream_state=None, stream_url=None)

    def test_shared_read_pose_calls_stream_refresh_callback(self):
        world = _WorldPoseStub()
        callback_hits = []

        def _fake_update_world_from_vision(world_obj, _vision, log=False):
            self.assertFalse(log)
            return None

        pose = helper_calibrate.read_pose(
            vision=object(),
            world=world,
            samples=1,
            timeout_s=0.1,
            min_sample_time=None,
            min_samples_required=1,
            observe_sleep_s=0.0,
            fallback_step_label="ALIGN_BRICK",
            update_world_from_vision=_fake_update_world_from_vision,
            latest_unique_smoothed_frames=lambda *_args, **_kwargs: [
                {
                    "frame_id": 1,
                    "visible": True,
                    "offset_y": 1.0,
                    "offset_x": 2.0,
                    "dist": 100.0,
                    "angle": 0.0,
                    "confidence": 90.0,
                }
            ],
            average_smoothed_frames=lambda frames, **_kwargs: dict(frames[-1]),
            lite_gate_unique_frames=lambda _step: 1,
            on_vision_update=lambda: callback_hits.append("tick"),
        )

        self.assertIsNotNone(pose)
        self.assertEqual(callback_hits, ["tick"])

    def test_shared_pose_meets_multiframe_requirement_rejects_raw_and_partial(self):
        self.assertTrue(
            helper_calibrate.pose_meets_multiframe_requirement(
                {
                    "pose_source": "lite_smoothed",
                    "samples_used": 3,
                    "lite_required_frames": 3,
                },
                required_samples=3,
            )
        )
        self.assertFalse(
            helper_calibrate.pose_meets_multiframe_requirement(
                {
                    "pose_source": "raw_visible",
                    "samples_used": 3,
                    "lite_required_frames": None,
                },
                required_samples=3,
            )
        )
        self.assertFalse(
            helper_calibrate.pose_meets_multiframe_requirement(
                {
                    "pose_source": "lite_smoothed",
                    "samples_used": 1,
                    "lite_required_frames": 3,
                },
                required_samples=3,
            )
        )


if __name__ == "__main__":
    unittest.main()
