import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class _DummyApp:
    def __init__(self, demos_dir):
        self.demos_dir = Path(demos_dir)
        self.config_mtime = 123.0
        self.last_config_check = 456.0
        self.logger = None
        self.logger_closed = True
        self.log_path = None


class TestSetupManualTrainingVisionModeDirs(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(setup_manual_training, name)
        setattr(setup_manual_training, name, value)
        self.addCleanup(setattr, setup_manual_training, name, original)

    def test_switch_demos_dir_when_vision_mode_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "old"
            new_dir = root / "new"
            old_dir.mkdir(parents=True, exist_ok=True)

            app = _DummyApp(old_dir)
            calls = {"close": 0, "open": 0, "log": []}

            self._patch_module_attr("_demos_dir_for_vision_mode", lambda _mode: new_dir)
            self._patch_module_attr("close_log", lambda *_args, **_kwargs: calls.__setitem__("close", calls["close"] + 1))
            self._patch_module_attr("open_new_log", lambda *_args, **_kwargs: calls.__setitem__("open", calls["open"] + 1))
            self._patch_module_attr("log_line", lambda msg: calls["log"].append(str(msg)))

            changed = setup_manual_training._set_active_demos_dir_for_mode(app, "cyan")

            self.assertTrue(changed)
            self.assertEqual(app.demos_dir, new_dir)
            self.assertTrue(new_dir.exists())
            self.assertEqual(calls["close"], 1)
            self.assertEqual(calls["open"], 1)
            self.assertEqual(app.config_mtime, 0)
            self.assertEqual(app.last_config_check, 0)
            self.assertTrue(any("Active folder" in line for line in calls["log"]))

    def test_no_switch_when_mode_maps_to_same_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            same_dir = root / "same"
            same_dir.mkdir(parents=True, exist_ok=True)

            app = _DummyApp(same_dir)
            calls = {"close": 0, "open": 0}

            self._patch_module_attr("_demos_dir_for_vision_mode", lambda _mode: same_dir)
            self._patch_module_attr("close_log", lambda *_args, **_kwargs: calls.__setitem__("close", calls["close"] + 1))
            self._patch_module_attr("open_new_log", lambda *_args, **_kwargs: calls.__setitem__("open", calls["open"] + 1))

            changed = setup_manual_training._set_active_demos_dir_for_mode(app, "aruco")

            self.assertFalse(changed)
            self.assertEqual(calls["close"], 0)
            self.assertEqual(calls["open"], 0)
            self.assertEqual(app.demos_dir, same_dir)
            self.assertEqual(app.config_mtime, 123.0)
            self.assertEqual(app.last_config_check, 456.0)


if __name__ == "__main__":
    unittest.main()
