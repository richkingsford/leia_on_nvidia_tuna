import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class TestSetupManualTrainingHotkeyEaseAllowlist(unittest.TestCase):
    def test_only_t_g_z_c_enable_ease_in_out(self):
        for key in ("t", "g", "z", "c", "T", "G", "Z", "C"):
            self.assertTrue(setup_manual_training.hotkey_uses_ease_in_out(key))
        for key in ("w", "s", "a", "d", "q", "e", "r", "f", "o", "k", "u", "p", "l", None, ""):
            self.assertFalse(setup_manual_training.hotkey_uses_ease_in_out(key))


if __name__ == "__main__":
    unittest.main()
