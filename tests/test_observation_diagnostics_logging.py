import sys
import unittest
from itertools import chain, repeat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from calibration import helper_calibrate
import telemetry_process


class _FakeRobot:
    def __init__(self):
        self.last_command = None

    def send_command_pwm(self, cmd, pwm, duration_ms=None):
        self.last_command = f"{cmd} {int(pwm)} {int(duration_ms or 0)}"
        return {
            "cmd_sent": str(cmd),
            "pwm": int(pwm),
            "duration_ms": int(duration_ms or 0),
            "wire_text": str(self.last_command),
        }


class TestObservationDiagnosticsLogging(unittest.TestCase):
    def test_aggregate_pose_samples_tracks_unique_frame_ids_and_spans(self):
        poses = [
            {
                "offset_y": 1.0,
                "offset_x": 5.0,
                "dist": 100.0,
                "angle": 0.0,
                "confidence": 80.0,
                "obs_ts": 10.0,
                "pose_source": "lite_smoothed",
                "lite_required_frames": 3,
                "lite_frame_ids": [101, 102, 103],
                "lite_frame_ts_start": 9.70,
                "lite_frame_ts_end": 9.82,
            },
            {
                "offset_y": 1.2,
                "offset_x": 4.6,
                "dist": 99.5,
                "angle": 0.2,
                "confidence": 82.0,
                "obs_ts": 10.1,
                "pose_source": "lite_smoothed",
                "lite_required_frames": 3,
                "lite_frame_ids": [102, 103, 104],
                "lite_frame_ts_start": 9.76,
                "lite_frame_ts_end": 9.88,
            },
            {
                "offset_y": 1.1,
                "offset_x": 4.8,
                "dist": 99.8,
                "angle": 0.1,
                "confidence": 81.0,
                "obs_ts": 10.2,
                "pose_source": "lite_smoothed",
                "lite_required_frames": 3,
                "lite_frame_ids": [103, 104, 105],
                "lite_frame_ts_start": 9.81,
                "lite_frame_ts_end": 9.93,
            },
        ]

        aggregated = helper_calibrate.aggregate_pose_samples(poses)

        self.assertEqual(aggregated["samples_used"], 3)
        self.assertEqual(aggregated["lite_required_frames"], 3)
        self.assertEqual(aggregated["lite_frame_count"], 5)
        self.assertEqual(aggregated["lite_frame_first_id"], 101)
        self.assertEqual(aggregated["lite_frame_last_id"], 105)
        self.assertAlmostEqual(aggregated["lite_frame_span_s"], 0.23)
        self.assertAlmostEqual(aggregated["sample_obs_ts_start"], 10.0)
        self.assertAlmostEqual(aggregated["sample_obs_ts_end"], 10.2)
        self.assertAlmostEqual(aggregated["sample_obs_span_s"], 0.2)

    def test_send_robot_command_pwm_reports_send_timestamps(self):
        world = SimpleNamespace()
        robot = _FakeRobot()

        with patch.object(
            telemetry_process,
            "_duration_used_ms_for_cmd",
            return_value=180,
        ), patch.object(
            telemetry_process,
            "_repeat_act_guard_before_send",
        ), patch.object(
            telemetry_process,
            "_repeat_act_guard_after_send",
        ), patch.object(
            telemetry_process,
            "record_action_display",
        ):
            result = telemetry_process.send_robot_command_pwm(
                robot,
                world,
                "ALIGN_BRICK",
                "l",
                power=0.25,
                pwm=80,
                duration_ms=180,
                speed_score=5,
                auto_mode=False,
            )

        self.assertIsNotNone(result["send_started_ts"])
        self.assertIsNotNone(result["send_completed_ts"])
        self.assertGreaterEqual(result["send_completed_ts"], result["send_started_ts"])
        self.assertEqual(getattr(world, "_last_action_wire_time", None), result["send_completed_ts"])

    def test_read_pose_filters_smoothed_frames_before_min_sample_time(self):
        world = SimpleNamespace(step_state=None, process_rules={})
        history = []
        world._smoothed_frame_history = history
        pending_batches = [
            [
                {"frame_id": 1, "timestamp": 9.91, "visible": True, "dist": 100.0, "angle": 0.0, "offset_x": 5.0, "x_axis": 5.0, "offset_y": 1.0, "y_axis": 1.0, "confidence": 80.0},
                {"frame_id": 2, "timestamp": 9.95, "visible": True, "dist": 101.0, "angle": 0.0, "offset_x": 4.5, "x_axis": 4.5, "offset_y": 1.0, "y_axis": 1.0, "confidence": 80.0},
                {"frame_id": 3, "timestamp": 9.99, "visible": True, "dist": 102.0, "angle": 0.0, "offset_x": 4.0, "x_axis": 4.0, "offset_y": 1.0, "y_axis": 1.0, "confidence": 80.0},
            ],
            [
                {"frame_id": 4, "timestamp": 10.01, "visible": True, "dist": 103.0, "angle": 0.0, "offset_x": 3.5, "x_axis": 3.5, "offset_y": 1.0, "y_axis": 1.0, "confidence": 80.0},
            ],
            [
                {"frame_id": 5, "timestamp": 10.02, "visible": True, "dist": 104.0, "angle": 0.0, "offset_x": 3.0, "x_axis": 3.0, "offset_y": 1.0, "y_axis": 1.0, "confidence": 80.0},
            ],
            [
                {"frame_id": 6, "timestamp": 10.03, "visible": True, "dist": 105.0, "angle": 0.0, "offset_x": 2.5, "x_axis": 2.5, "offset_y": 1.0, "y_axis": 1.0, "confidence": 80.0},
            ],
        ]

        def _update_world_from_vision(_world, _vision, log=False):
            if pending_batches:
                history.extend(pending_batches.pop(0))

        def _latest_unique_smoothed_frames(_world, required_frames, min_timestamp=None):
            selected = []
            seen = set()
            for entry in reversed(history):
                frame_id = int(entry.get("frame_id", 0) or 0)
                if frame_id <= 0 or frame_id in seen:
                    continue
                if min_timestamp is not None and float(entry.get("timestamp", 0.0) or 0.0) < float(min_timestamp):
                    continue
                seen.add(frame_id)
                selected.append(entry)
                if len(selected) >= int(required_frames):
                    break
            selected.reverse()
            return selected

        def _average_smoothed_frames(frames, *, step=None, process_rules=None):
            return {
                "visible": True,
                "dist": sum(float(frame["dist"]) for frame in frames) / len(frames),
                "angle": 0.0,
                "offset_x": sum(float(frame["offset_x"]) for frame in frames) / len(frames),
                "x_axis": sum(float(frame["x_axis"]) for frame in frames) / len(frames),
                "offset_y": 1.0,
                "y_axis": 1.0,
                "confidence": 80.0,
            }

        fake_times = chain([10.00, 10.00, 10.00, 10.02, 10.03, 10.04, 10.05], repeat(10.05))

        with patch.object(
            helper_calibrate.time,
            "time",
            side_effect=lambda: next(fake_times),
        ), patch.object(
            helper_calibrate.time,
            "sleep",
        ):
            pose = helper_calibrate.read_pose(
                object(),
                world,
                samples=1,
                timeout_s=0.1,
                min_sample_time=10.01,
                min_samples_required=1,
                observe_sleep_s=0.0,
                fallback_step_label="ALIGN_BRICK",
                update_world_from_vision=_update_world_from_vision,
                latest_unique_smoothed_frames=_latest_unique_smoothed_frames,
                average_smoothed_frames=_average_smoothed_frames,
                lite_gate_unique_frames=lambda _step: 3,
                min_lite_unique_frames=3,
            )

        self.assertIsNotNone(pose)
        self.assertEqual(pose["lite_frame_first_id"], 4)
        self.assertEqual(pose["lite_frame_last_id"], 6)
        self.assertAlmostEqual(pose["lite_frame_ts_start"], 10.01)
        self.assertEqual(len(history), 6)

    def test_observe_pose_with_reobserve_keeps_min_sample_time_in_rescue_round(self):
        call_times = []

        def _read_pose_fn(_vision, _world, *, samples, timeout_s, min_sample_time=None, min_samples_required=None, on_vision_update=None):
            call_times.append(min_sample_time)
            if len(call_times) == 1:
                return None
            return {
                "pose_source": "lite_smoothed",
                "samples_used": 3,
                "lite_required_frames": 3,
            }

        pose, meta = helper_calibrate.observe_pose_with_reobserve(
            read_pose_fn=_read_pose_fn,
            log_fn=lambda *_args, **_kwargs: None,
            log_prefix="[TEST]",
            vision=object(),
            world=SimpleNamespace(step_state=None),
            samples=3,
            timeout_s=0.1,
            min_sample_time=12.5,
            hold_s=0.0,
            reobserve_rounds=1,
            relaxed_timeout_s=0.2,
        )

        self.assertIsNotNone(pose)
        self.assertEqual(meta["mode"], "hold_reobserve_full")
        self.assertEqual(call_times, [12.5, 12.5])


if __name__ == "__main__":
    unittest.main()