import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from calibration import helper_calibrate
import helper_calibrate_telemetry


class _DummyWorld:
    def __init__(self):
        self.step_state = None
        self._post_action_observe_delay_s = 0.0


class _DummyVision:
    pass


class HelperCalibrateTelemetrySharedStreamTests(unittest.TestCase):
    def test_run_uses_shared_stream_runtime_instead_of_starting_local_server(self):
        shared_stream_state = {
            "lock": threading.Lock(),
            "frame": None,
            "text_lines": [],
        }

        pose = {
            "dist": 118.0,
            "offset_x": 2.0,
            "offset_y": 4.0,
            "confidence": 0.9,
            "samples_used": 3,
            "observed_spread_mm": 0.8,
            "dropped_outliers": 0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir, helper_calibrate.use_shared_stream_runtime(
            stream_state=shared_stream_state,
            stream_url="http://127.0.0.1:5000",
        ):
            tmp_path = Path(tmp_dir)
            results_path = tmp_path / "results.json"
            plot_path = tmp_path / "plot.png"
            staging_path = tmp_path / "staging.json"
            world_model_path = tmp_path / "world_model.json"

            argv = [
                "helper_calibrate_telemetry.py",
                "--variable",
                "dist",
                "--points",
                "1",
                "--results-file",
                str(results_path),
                "--plot-path",
                str(plot_path),
                "--staging-file",
                str(staging_path),
                "--world-model-file",
                str(world_model_path),
                "--no-commit-world-model",
            ]

            with patch.object(sys, "argv", argv), \
                 patch.object(helper_calibrate_telemetry, "WorldModel", side_effect=_DummyWorld), \
                 patch.object(helper_calibrate_telemetry, "Robot", return_value=object()), \
                 patch.object(helper_calibrate_telemetry, "_create_vision", return_value=_DummyVision()), \
                 patch.object(helper_calibrate_telemetry.helper_xyz_coords, "sync_from_world"), \
                 patch.object(helper_calibrate_telemetry, "_refresh_stream_state"), \
                 patch.object(helper_calibrate_telemetry, "_prompt_with_hotkeys_and_live_refresh", return_value=""), \
                 patch.object(helper_calibrate_telemetry, "_prompt_expected_value_text", return_value="120"), \
                 patch.object(helper_calibrate_telemetry, "_capture_tight_pose", return_value=(pose, {"reason": "ok"})), \
                 patch.object(helper_calibrate_telemetry, "_write_curve_plot", return_value=(False, "disabled")), \
                 patch.object(helper_calibrate_telemetry, "start_stream_server") as start_stream_server:
                exit_code = helper_calibrate_telemetry.run_telemetry_variable_calibration("dist")
                results_written = results_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertFalse(start_stream_server.called)
        self.assertEqual(shared_stream_state.get("vision_mode"), "cyan")
        self.assertTrue(results_written)

    def test_run_reuses_shared_live_runtime_when_context_is_provided(self):
        shared_stream_state = {
            "lock": threading.Lock(),
            "frame": None,
            "text_lines": [],
            "telemetry_lines": [],
            "extra_text_lines": [],
        }
        shared_world = _DummyWorld()
        shared_robot = object()
        shared_vision = _DummyVision()
        pose = {
            "dist": 90.0,
            "offset_x": 1.0,
            "offset_y": 2.0,
            "confidence": 0.8,
            "samples_used": 3,
            "observed_spread_mm": 0.6,
            "dropped_outliers": 0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir, helper_calibrate.use_shared_stream_runtime(
            stream_state=shared_stream_state,
            stream_url="http://127.0.0.1:5000",
            context={
                "world": shared_world,
                "robot": shared_robot,
                "vision": shared_vision,
            },
        ):
            tmp_path = Path(tmp_dir)
            argv = [
                "helper_calibrate_telemetry.py",
                "--variable",
                "dist",
                "--points",
                "1",
                "--results-file",
                str(tmp_path / "results.json"),
                "--plot-path",
                str(tmp_path / "plot.png"),
                "--staging-file",
                str(tmp_path / "staging.json"),
                "--world-model-file",
                str(tmp_path / "world_model.json"),
                "--no-commit-world-model",
            ]

            with patch.object(sys, "argv", argv), \
                 patch.object(helper_calibrate_telemetry, "_prompt_with_hotkeys_and_live_refresh", return_value=""), \
                 patch.object(helper_calibrate_telemetry, "_prompt_expected_value_text", return_value="95"), \
                 patch.object(helper_calibrate_telemetry, "_capture_tight_pose", return_value=(pose, {"reason": "ok"})), \
                 patch.object(helper_calibrate_telemetry, "_write_curve_plot", return_value=(False, "disabled")), \
                 patch.object(helper_calibrate_telemetry, "_create_vision") as create_vision, \
                 patch.object(helper_calibrate_telemetry, "Robot") as robot_ctor, \
                 patch.object(helper_calibrate_telemetry, "start_stream_server") as start_stream_server:
                exit_code = helper_calibrate_telemetry.run_telemetry_variable_calibration("dist")

        self.assertEqual(exit_code, 0)
        self.assertFalse(create_vision.called)
        self.assertFalse(robot_ctor.called)
        self.assertFalse(start_stream_server.called)


if __name__ == "__main__":
    unittest.main()
