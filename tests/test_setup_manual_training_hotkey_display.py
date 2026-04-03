import re
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_MAIN as setup_manual_training


class _DummyAppState:
    def __init__(self):
        self.running = True
        self.robot = None
        self.hotkey_edit_target = None


class TestSetupManualTrainingHotkeyDisplay(unittest.TestCase):
    def test_hotkey_preview_log_shows_logical_command_with_remap_present(self):
        app = _DummyAppState()
        messages = []

        orig_log_line = setup_manual_training.log_line
        orig_hotkeys = setup_manual_training.HOTKEY_SPEED_SCORES
        orig_remap = getattr(setup_manual_training.telemetry_robot_module, "COMMAND_REMAP", None)
        try:
            setup_manual_training.log_line = messages.append
            setup_manual_training.HOTKEY_SPEED_SCORES = {
                "x": {"cmd": "f", "score": 1},
            }
            setup_manual_training.telemetry_robot_module.COMMAND_REMAP = {"f": "b"}

            score_used = setup_manual_training.maybe_prompt_hotkey_var_update(
                app,
                "f",
                1,
                enter_edit=False,
                hotkey="x",
            )
        finally:
            setup_manual_training.log_line = orig_log_line
            setup_manual_training.HOTKEY_SPEED_SCORES = orig_hotkeys
            setup_manual_training.telemetry_robot_module.COMMAND_REMAP = orig_remap

        self.assertEqual(score_used, 1)
        self.assertEqual(len(messages), 1)
        line = str(messages[0])
        self.assertRegex(
            line,
            r"^\[HOTKEY\] X -> f 1% \(pwm=\d+, pwr=\d+\.\d{3}, t=\d+ms; shared 1% floor(?:; breakaway \d+/\d+)?\) - press y to edit vars$",
        )
        self.assertNotIn("wire=", line)
        self.assertNotIn("power=", line)
        self.assertFalse(re.search(r"\bB\b", line))

    def test_format_hotkey_wire_log_includes_pressed_key_and_wire_text(self):
        line = setup_manual_training._format_hotkey_wire_log(
            "w",
            "f",
            50,
            {
                "cmd_sent": "f",
                "pwm": 128,
                "power": 0.5,
                "duration_ms": 284,
                "wire_text": "l.f.50.284,r.f.50.284",
            },
        )

        self.assertIsInstance(line, str)
        self.assertRegex(
            str(line),
            r"^\[HOTKEY WIRE\] W -> f 50% \(pwm=128, pwr=0\.500, t=284ms\) -> l\.f\.50\.284,r\.f\.50\.284$",
        )

    def test_ctrl_help_line_lists_explicit_lift_up_and_down_hotkeys(self):
        line = str(setup_manual_training.CTRL_HELP_LINE)
        self.assertIn("Lift up: O 1%, P 100%.", line)
        self.assertIn("Lift down: K 1%, L 100%.", line)


if __name__ == "__main__":
    unittest.main()
