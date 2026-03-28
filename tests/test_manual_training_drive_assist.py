import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_MAIN


class _DummyWorld:
    step_state = "MANUAL"


class _DummyAppState:
    def __init__(self):
        self.robot = object()
        self.world = _DummyWorld()


class TestManualTrainingDriveAssist(unittest.TestCase):
    def test_send_manual_drive_hotkey_with_assist_returns_none_when_repo_default_is_disabled(self):
        app_state = _DummyAppState()
        result = a_MAIN._send_manual_drive_hotkey_with_assist(
            app_state,
            cmd="b",
            score_used=1,
            hotkey="f",
            duration_override_ms=None,
            pwm_override=None,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
