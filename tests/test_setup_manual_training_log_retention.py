import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training
from telemetry_robot import TelemetryLogger


class _DummyApp:
    def __init__(self, log_path):
        self.logger = TelemetryLogger(log_path)
        self.logger_closed = False
        self.log_path = Path(log_path)


class TestSetupManualTrainingLogRetention(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(setup_manual_training, name)
        setattr(setup_manual_training, name, value)
        self.addCleanup(setattr, setup_manual_training, name, original)

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


if __name__ == "__main__":
    unittest.main()
