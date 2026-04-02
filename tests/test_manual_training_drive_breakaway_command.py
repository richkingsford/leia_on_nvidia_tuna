import sys
import threading
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_MAIN


class _DummyAppState:
    def __init__(self):
        self.lock = threading.Lock()


class TestManualTrainingDriveBreakawayCommand(unittest.TestCase):
    def test_handle_command_line_runs_drive_breakaway_command(self):
        app_state = _DummyAppState()
        calls = []
        original_runner = a_MAIN.run_drive_breakaway_test_command
        try:
            def _fake_runner(app):
                calls.append(app)
                return {"ok": True}

            a_MAIN.run_drive_breakaway_test_command = _fake_runner
            exit_mode, do_help, messages, ended_info = a_MAIN.handle_command_line(app_state, "drivebreak")
        finally:
            a_MAIN.run_drive_breakaway_test_command = original_runner

        self.assertFalse(exit_mode)
        self.assertFalse(do_help)
        self.assertIsNone(ended_info)
        self.assertEqual(calls, [app_state])
        self.assertIn("[BREAKAWAY TEST] Complete.", messages)

    def test_print_command_help_lists_drive_breakaway_shortcut(self):
        lines = []
        original_log_line = a_MAIN.log_line
        try:
            a_MAIN.log_line = lines.append
            a_MAIN.print_command_help()
        finally:
            a_MAIN.log_line = original_log_line

        self.assertTrue(any(":drivebreak" in str(line) for line in lines))
        self.assertTrue(any(":turnbreak" in str(line) for line in lines))

    def test_handle_command_line_runs_turn_breakaway_command(self):
        app_state = _DummyAppState()
        calls = []
        original_runner = a_MAIN.run_turn_breakaway_test_command
        try:
            def _fake_runner(app):
                calls.append(app)
                return {"ok": True}

            a_MAIN.run_turn_breakaway_test_command = _fake_runner
            exit_mode, do_help, messages, ended_info = a_MAIN.handle_command_line(app_state, "turnbreak")
        finally:
            a_MAIN.run_turn_breakaway_test_command = original_runner

        self.assertFalse(exit_mode)
        self.assertFalse(do_help)
        self.assertIsNone(ended_info)
        self.assertEqual(calls, [app_state])
        self.assertIn("[TURN BREAKAWAY TEST] Complete.", messages)


if __name__ == "__main__":
    unittest.main()
