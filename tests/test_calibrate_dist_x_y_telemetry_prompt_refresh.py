import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import calibrate_dist_x_y_telemetry


class _FakeStdin:
    def isatty(self):
        return True

    def fileno(self):
        return 123


class TestPromptHotkeyRefresh(unittest.TestCase):
    def test_hotkey_triggers_immediate_refresh(self):
        calls = []

        def hotkey_handler(row):
            calls.append(("hotkey", row.get("hotkey")))

        def live_refresh_callback():
            calls.append(("refresh", None))

        with patch.object(sys, "stdin", _FakeStdin()), \
             patch.object(calibrate_dist_x_y_telemetry.termios, "tcgetattr", return_value=[0, 0, 0, 0, 0, 0, [0, 0]]), \
             patch.object(calibrate_dist_x_y_telemetry.termios, "tcsetattr"), \
             patch.object(calibrate_dist_x_y_telemetry.tty, "setcbreak"), \
             patch.object(calibrate_dist_x_y_telemetry.select, "select", side_effect=[([123], [], []), ([123], [], [])]), \
             patch.object(calibrate_dist_x_y_telemetry.os, "read", side_effect=[b"o", b"\n"]), \
             patch.object(calibrate_dist_x_y_telemetry, "update_world_from_vision") as update_world:
            result = calibrate_dist_x_y_telemetry._prompt_with_hotkeys_and_live_refresh(
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


if __name__ == "__main__":
    unittest.main()
