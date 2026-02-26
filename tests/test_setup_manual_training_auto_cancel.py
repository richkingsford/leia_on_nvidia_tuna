import sys
import threading
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class _DummyCancelApp:
    def __init__(self):
        self.auto_cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.auto_confirm_event = threading.Event()
        self.auto_confirm_needed = False
        self.last_enter_time = 0.0
        self.running = True


class TestSetupManualTrainingAutoCancel(unittest.TestCase):
    def test_auto_observer_raises_when_cancel_requested(self):
        app = _DummyCancelApp()
        app.auto_cancel_event.set()
        observer = setup_manual_training.make_auto_observer(app)
        with self.assertRaises(setup_manual_training.AutoStepCancelled):
            observer("frame", None, None, None, None, None)

    def test_auto_confirm_raises_when_cancel_requested(self):
        app = _DummyCancelApp()
        app.auto_cancel_event.set()
        confirm = setup_manual_training.make_auto_confirm(app)
        with self.assertRaises(setup_manual_training.AutoStepCancelled):
            confirm(object(), object())
        self.assertFalse(app.auto_confirm_needed)


if __name__ == "__main__":
    unittest.main()
