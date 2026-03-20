import sys
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training
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
