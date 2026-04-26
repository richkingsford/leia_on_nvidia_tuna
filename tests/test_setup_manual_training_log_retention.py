import sys
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_MAIN as setup_manual_training
from telemetry_robot import TelemetryLogger


class _DummyApp:
    def __init__(self, log_path):
        self.logger = TelemetryLogger(log_path)
        self.logger_closed = False
        self.log_path = Path(log_path)


class _DummyOpenLogApp:
    def __init__(self, runs_dir):
        self.runs_dir = Path(runs_dir)
        self.demos_dir = Path(runs_dir)
        self.logger = None
        self.logger_closed = True
        self.log_path = None
        self.session_log_paths = []
        self.world = SimpleNamespace(run_id=None, attempt_id=None)


class _NoopLock:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _StopRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class TestSetupManualTrainingLogRetention(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(setup_manual_training, name)
        setattr(setup_manual_training, name, value)
        self.addCleanup(setattr, setup_manual_training, name, original)

    def _patch_time_now(self, fn):
        original = setup_manual_training.time.time
        setup_manual_training.time.time = fn
        self.addCleanup(setattr, setup_manual_training.time, "time", original)

    def test_close_log_keeps_file_when_no_completed_attempt_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "kbd_test.json"
            app = _DummyApp(log_path)
            messages = []

            self._patch_module_attr("log_line", lambda msg: messages.append(str(msg)))

            setup_manual_training.close_log(app, marker=None)

            self.assertTrue(log_path.exists())
            self.assertTrue(app.logger_closed)
            self.assertTrue(any("No completed attempts" in msg for msg in messages))

    def test_cleanup_stale_run_logs_hard_deletes_files_older_than_one_hour(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_aruco = Path(tmp) / "Runs - aruco"
            run_cyan = Path(tmp) / "Runs - cyan"
            run_aruco.mkdir()
            run_cyan.mkdir()
            old_aruco = run_aruco / "old_aruco.json"
            old_cyan = run_cyan / "old_cyan.json"
            fresh_aruco = run_aruco / "fresh_aruco.json"
            old_aruco.write_text("{}\n")
            old_cyan.write_text("{}\n")
            fresh_aruco.write_text("{}\n")
            now_s = 10_000.0
            stale_s = now_s - 3601.0
            fresh_s = now_s - 30.0
            os.utime(old_aruco, (stale_s, stale_s))
            os.utime(old_cyan, (stale_s, stale_s))
            os.utime(fresh_aruco, (fresh_s, fresh_s))
            self._patch_time_now(lambda: now_s)

            setup_manual_training._cleanup_stale_run_logs(
                run_folders=(run_aruco, run_cyan),
                cutoff_age_s=3600.0,
            )

            self.assertFalse(old_aruco.exists())
            self.assertFalse(old_cyan.exists())
            self.assertTrue(fresh_aruco.exists())

    def test_cleanup_stale_run_files_preserves_run_view_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_views = Path(tmp) / "trials" / "run_views"
            run_views.mkdir(parents=True)
            old_view = run_views / "old_run.html"
            index = run_views / "index.html"
            old_view.write_text("<html></html>\n")
            index.write_text("<html></html>\n")
            now_s = 20_000.0
            stale_s = now_s - 3601.0
            os.utime(old_view, (stale_s, stale_s))
            os.utime(index, (stale_s, stale_s))
            self._patch_time_now(lambda: now_s)

            setup_manual_training._cleanup_stale_run_logs(
                run_folders=(run_views,),
                cutoff_age_s=3600.0,
            )

            self.assertFalse(old_view.exists())
            self.assertTrue(index.exists())

    def test_run_log_retention_default_is_ten_hours(self):
        self.assertEqual(float(setup_manual_training.RUN_LOG_RETENTION_S), 10 * 60 * 60)

    def test_run_auto_step_runs_retention_cleanup_before_loading_demos(self):
        cleanup_calls = []
        call_order = []
        app = SimpleNamespace(
            world=SimpleNamespace(),
            robot=SimpleNamespace(),
            demos_dir=Path("demos"),
            log_path=Path("Runs - cyan") / "active.json",
        )
        step = SimpleNamespace(value="ALIGN_BRICK")

        self._patch_module_attr("log_line", lambda msg: None)
        self._patch_module_attr("_set_stream_state_success_gate_step", lambda *_args, **_kwargs: None)
        self._patch_module_attr("reset_repeat_act_guard", lambda *_args, **_kwargs: None)
        self._patch_module_attr(
            "_cleanup_stale_run_logs",
            lambda **kwargs: (call_order.append("cleanup"), cleanup_calls.append(dict(kwargs))),
        )

        def _load_demo_logs_empty(*_args, **_kwargs):
            call_order.append("load_demo_logs")
            return []

        self._patch_module_attr("load_demo_logs", _load_demo_logs_empty)
        self._patch_module_attr("update_process_model_from_demos", lambda *_args, **_kwargs: None)
        self._patch_module_attr("refresh_autobuild_config", lambda *_args, **_kwargs: None)

        ok = setup_manual_training.run_auto_step(app, step)

        self.assertFalse(ok)
        self.assertEqual(call_order[:2], ["cleanup", "load_demo_logs"])
        self.assertEqual(len(cleanup_calls), 1)
        self.assertEqual(cleanup_calls[0]["preserve_live_files"], ("active.json",))

    def test_auto_observer_hard_stops_after_fifteenth_logged_action(self):
        app = SimpleNamespace(
            auto_cancel_event=None,
            auto_demo_logging_active=True,
            current_auto_step="ALIGN_BRICK",
            current_auto_action_count=14,
            auto_step_act_limit=15,
            robot=_StopRobot(),
            lock=_NoopLock(),
        )
        world = SimpleNamespace(_last_action_score=25, _last_action_duration_ms=100, _last_action_speed=0.4)
        refresh_calls = []
        messages = []

        def _fake_log_motion(app_state, *_args, **_kwargs):
            app_state.current_auto_action_count += 1

        self._patch_module_attr("_log_motion_event_and_state", _fake_log_motion)
        self._patch_module_attr("refresh_brick_telemetry", lambda *_args, **_kwargs: refresh_calls.append(True))
        self._patch_module_attr("log_line", lambda msg: messages.append(str(msg)))

        observer = setup_manual_training.make_auto_observer(app)

        with self.assertRaises(setup_manual_training.AutoStepActLimit):
            observer("action", world, None, "f", 0.4, "test")

        self.assertEqual(app.current_auto_action_count, 15)
        self.assertEqual(app.robot.stop_calls, 1)
        self.assertEqual(len(refresh_calls), 1)
        self.assertTrue(any("15/15" in msg for msg in messages))

    def test_open_new_log_runs_retention_cleanup_before_creating_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = _DummyOpenLogApp(tmp)
            cleanup_calls = []

            self._patch_module_attr("log_line", lambda msg: None)
            self._patch_module_attr(
                "_cleanup_stale_run_logs",
                lambda **kwargs: cleanup_calls.append(dict(kwargs)),
            )

            setup_manual_training.open_new_log(app)

            self.assertEqual(len(cleanup_calls), 1)
            self.assertIsNotNone(app.log_path)
            self.assertTrue(Path(app.log_path).parent.exists())
            self.assertTrue(str(app.world.run_id).startswith("run_"))
            self.assertEqual(int(app.world.attempt_id), 1)


if __name__ == "__main__":
    unittest.main()
