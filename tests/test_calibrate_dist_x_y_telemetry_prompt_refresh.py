import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_calibrate_telemetry


class _FakeStdin:
    def isatty(self):
        return True

    def fileno(self):
        return 123


class TestPromptHotkeyRefresh(unittest.TestCase):
    def test_expected_value_prompt_is_short_terminal_friendly(self):
        prompt = helper_calibrate_telemetry._expected_value_prompt("dist", 65.0)
        self.assertEqual(prompt, "Expected dist mm [Enter=65.0, q=quit] > ")
        self.assertLess(len(prompt), 50)

    def test_ready_for_capture_prompt_is_short_terminal_friendly(self):
        prompt = helper_calibrate_telemetry._ready_for_capture_prompt("dist")
        self.assertIn("Enter=type expected dist", prompt)
        self.assertLess(len(prompt), 60)

    def test_hotkey_triggers_immediate_refresh(self):
        calls = []

        def hotkey_handler(row):
            calls.append(("hotkey", row.get("hotkey")))

        def live_refresh_callback():
            calls.append(("refresh", None))

        with patch.object(sys, "stdin", _FakeStdin()), \
             patch.object(helper_calibrate_telemetry.termios, "tcgetattr", return_value=[0, 0, 0, 0, 0, 0, [0, 0]]), \
             patch.object(helper_calibrate_telemetry.termios, "tcsetattr"), \
             patch.object(helper_calibrate_telemetry, "_enable_single_char_noecho_mode"), \
             patch.object(helper_calibrate_telemetry.select, "select", side_effect=[([123], [], []), ([123], [], [])]), \
             patch.object(helper_calibrate_telemetry.os, "read", side_effect=[b"o", b"\n"]), \
             patch.object(helper_calibrate_telemetry, "update_world_from_vision") as update_world:
            result = helper_calibrate_telemetry._prompt_with_hotkeys_and_live_refresh(
                prompt="prompt: ",
                vision=object(),
                world=object(),
                enable_live_refresh=True,
                refresh_interval_s=0.1,
                hotkey_rows={"o": {"hotkey": "o", "cmd": "u", "score": 1}},
                hotkey_handler=hotkey_handler,
                live_refresh_callback=live_refresh_callback,
            )

        self.assertEqual(result, "")
        self.assertEqual(calls, [("hotkey", "o"), ("refresh", None)])
        update_world.assert_called()

    def test_escape_exits_ready_prompt(self):
        with patch.object(sys, "stdin", _FakeStdin()), \
             patch.object(helper_calibrate_telemetry.termios, "tcgetattr", return_value=[0, 0, 0, 0, 0, 0, [0, 0]]), \
             patch.object(helper_calibrate_telemetry.termios, "tcsetattr"), \
             patch.object(helper_calibrate_telemetry, "_enable_single_char_noecho_mode"), \
             patch.object(
                 helper_calibrate_telemetry.select,
                 "select",
                 side_effect=[([123], [], [])],
             ), \
             patch.object(helper_calibrate_telemetry.os, "read", side_effect=[b"\x1b"]):
            result = helper_calibrate_telemetry._prompt_with_hotkeys_and_live_refresh(
                prompt="prompt: ",
                vision=object(),
                world=object(),
                enable_live_refresh=True,
                refresh_interval_s=0.1,
                hotkey_rows={"o": {"hotkey": "o", "cmd": "u", "score": 1}},
                hotkey_handler=lambda _row: None,
                live_refresh_callback=None,
            )

        self.assertEqual(result, "quit")

    def test_expected_value_text_uses_plain_line_input(self):
        with patch.object(helper_calibrate_telemetry, "_restore_tty_line_input_mode"), \
             patch("builtins.input", return_value="120"), \
             patch.object(helper_calibrate_telemetry.termios, "tcflush"):
            result = helper_calibrate_telemetry._prompt_expected_value_text("Expected dist mm > ")

        self.assertEqual(result, "120")


if __name__ == "__main__":
    unittest.main()
