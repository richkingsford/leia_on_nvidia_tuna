import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class _DummyApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.stream_state = {
            "lock": threading.Lock(),
            "step_success_seq": 0,
            "step_success_step": None,
            "step_success_at": 0.0,
        }


class _DummyLogger:
    def __init__(self):
        self.enabled = True
        self.calls = []

    def log_keyframe(self, marker, obj_label):
        self.calls.append((marker, obj_label))


class _DummyWorld:
    def __init__(self):
        self.step_state = setup_manual_training.StepState.SEAT_BRICK2
        self.attempt_status = "SUCCESS"
        self.recording_active = True

    def reset_mission(self):
        return None


class _DummyAttemptApp(_DummyApp):
    def __init__(self):
        super().__init__()
        self.world = _DummyWorld()
        self.logger = _DummyLogger()
        self.active_attempt = "SUCCESS"
        self.step_open = True
        self.open_step = self.world.step_state


class TestSetupManualTrainingStreamStepSuccessEvent(unittest.TestCase):
    def test_emit_stream_step_success_event_updates_state(self):
        app = _DummyApp()
        setup_manual_training._emit_stream_step_success_event(app, "seat_brick2")
        with app.stream_state["lock"]:
            self.assertEqual(app.stream_state.get("step_success_seq"), 1)
            self.assertEqual(app.stream_state.get("step_success_step"), "SEAT_BRICK2")
            self.assertGreater(float(app.stream_state.get("step_success_at") or 0.0), 0.0)

    def test_emit_stream_step_success_event_increments_sequence(self):
        app = _DummyApp()
        setup_manual_training._emit_stream_step_success_event(app, "ALIGN_BRICK")
        setup_manual_training._emit_stream_step_success_event(app, "SEAT_BRICK2")
        with app.stream_state["lock"]:
            self.assertEqual(app.stream_state.get("step_success_seq"), 2)
            self.assertEqual(app.stream_state.get("step_success_step"), "SEAT_BRICK2")

    def test_end_attempt_success_emits_stream_step_success_event(self):
        app = _DummyAttemptApp()
        with patch("setup_manual_training.apply_height_snapshot_from_step", return_value={"applied": True}):
            ok, _msg, _obj, _attempt, _closed = setup_manual_training.end_attempt(app, complete_step=True)
        self.assertTrue(ok)
        with app.stream_state["lock"]:
            self.assertEqual(app.stream_state.get("step_success_seq"), 1)
            self.assertEqual(app.stream_state.get("step_success_step"), "SEAT_BRICK2")


if __name__ == "__main__":
    unittest.main()
