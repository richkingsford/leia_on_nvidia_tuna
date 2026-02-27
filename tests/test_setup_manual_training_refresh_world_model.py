import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class _DummyWorld:
    def __init__(self):
        self.process_rules = {}
        self.rules = {}
        self.learned_rules = {}


class _DummyAppState:
    def __init__(self, demos_dir):
        self.demos_dir = Path(demos_dir)
        self.last_config_check = 0.0
        self.config_mtime = 0.0
        self.world = _DummyWorld()
        self._last_align_profile_signature = None


class TestSetupManualTrainingRefreshWorldModel(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(setup_manual_training, name)
        setattr(setup_manual_training, name, value)
        self.addCleanup(setattr, setup_manual_training, name, original)

    def _patch_time_now(self, fn):
        original = setup_manual_training.time.time
        setup_manual_training.time.time = fn
        self.addCleanup(setattr, setup_manual_training.time, "time", original)

    def test_refresh_preserves_existing_process_model_when_no_demo_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demos_dir = root / "demos"
            demos_dir.mkdir(parents=True, exist_ok=True)
            process_file = root / "world_model_process.json"
            model = {
                "steps": {
                    "ALIGN_BRICK": {
                        "success_gates": {
                            "visible": {"min": True},
                            "dist": {"target": 90.0, "tol": 1.5},
                        }
                    }
                }
            }
            process_file.write_text(json.dumps(model, indent=2) + "\n")
            now = float(int(time.time()))
            os.utime(process_file, (now - 20.0, now - 20.0))

            app = _DummyAppState(demos_dir)
            update_calls = {"count": 0}

            self._patch_module_attr("PROCESS_MODEL_FILE", process_file)
            self._patch_module_attr("_latest_demo_mtime", lambda _demos: 0.0)
            self._patch_module_attr("load_demo_logs", lambda _demos: [])
            self._patch_module_attr(
                "update_process_model_from_demos",
                lambda *_args, **_kwargs: update_calls.__setitem__("count", update_calls["count"] + 1),
            )
            self._patch_module_attr("refresh_autobuild_config", lambda _path: None)
            self._patch_module_attr("load_process_model", lambda path: json.loads(Path(path).read_text()))
            self._patch_module_attr("load_align_profile", lambda _root: {})
            self._patch_module_attr(
                "inject_align_profile_into_learned_rules",
                lambda learned, _profile: dict(learned or {}),
            )
            self._patch_time_now(lambda: now)

            setup_manual_training.refresh_world_model_from_demos(app, force=True, min_interval_s=0.0)

            self.assertEqual(update_calls["count"], 0)
            self.assertEqual(app.world.process_rules, model["steps"])
            self.assertEqual(app.world.rules, model["steps"])
            self.assertAlmostEqual(app.config_mtime, now - 20.0, places=2)

    def test_refresh_uses_post_update_process_mtime_to_prevent_repeated_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demos_dir = root / "demos"
            demos_dir.mkdir(parents=True, exist_ok=True)
            process_file = root / "world_model_process.json"
            process_file.write_text(json.dumps({"steps": {}}, indent=2) + "\n")

            base = float(int(time.time()))
            initial_process_mtime = base - 30.0
            demo_mtime = base - 20.0
            updated_process_mtime = base - 10.0
            os.utime(process_file, (initial_process_mtime, initial_process_mtime))

            app = _DummyAppState(demos_dir)
            update_calls = {"count": 0}

            def _fake_update(_logs, path):
                update_calls["count"] += 1
                Path(path).write_text(
                    json.dumps(
                        {"steps": {"ALIGN_BRICK": {"success_gates": {"visible": {"min": True}}}}},
                        indent=2,
                    )
                    + "\n"
                )
                os.utime(path, (updated_process_mtime, updated_process_mtime))

            clock = {"now": base}

            def _fake_time():
                clock["now"] += 1.0
                return clock["now"]

            self._patch_module_attr("PROCESS_MODEL_FILE", process_file)
            self._patch_module_attr("_latest_demo_mtime", lambda _demos: demo_mtime)
            self._patch_module_attr("load_demo_logs", lambda _demos: [(Path("demo.json"), [{"type": "keyframe"}])])
            self._patch_module_attr("update_process_model_from_demos", _fake_update)
            self._patch_module_attr("refresh_autobuild_config", lambda _path: None)
            self._patch_module_attr("load_process_model", lambda path: json.loads(Path(path).read_text()))
            self._patch_module_attr("load_align_profile", lambda _root: {})
            self._patch_module_attr(
                "inject_align_profile_into_learned_rules",
                lambda learned, _profile: dict(learned or {}),
            )
            self._patch_time_now(_fake_time)

            setup_manual_training.refresh_world_model_from_demos(app, force=False, min_interval_s=0.0)
            self.assertEqual(update_calls["count"], 1)
            self.assertAlmostEqual(app.config_mtime, updated_process_mtime, places=2)
            self.assertIn("ALIGN_BRICK", app.world.process_rules)

            setup_manual_training.refresh_world_model_from_demos(app, force=False, min_interval_s=0.0)
            self.assertEqual(update_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
