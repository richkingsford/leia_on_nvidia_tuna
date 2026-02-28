import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_vision_config


class TestHelperVisionConfig(unittest.TestCase):
    def test_mode_aliases_normalize_to_expected_values(self):
        self.assertEqual(helper_vision_config.normalize_vision_mode("aruco"), helper_vision_config.VISION_MODE_ARUCO)
        self.assertEqual(helper_vision_config.normalize_vision_mode("yolo"), helper_vision_config.VISION_MODE_CYAN)
        self.assertEqual(helper_vision_config.normalize_vision_mode("markerless"), helper_vision_config.VISION_MODE_CYAN)

    def test_load_and_resolve_demos_dirs_from_world_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aruco_dir = root / "aruco-demos"
            cyan_dir = root / "cyan-demos"
            aruco_dir.mkdir(parents=True, exist_ok=True)
            cyan_dir.mkdir(parents=True, exist_ok=True)
            model_path = root / "world_model_vision.json"
            model_path.write_text(
                json.dumps(
                    {
                        "active_mode": "markerless",
                        "demos_by_mode": {
                            "aruco": str(aruco_dir),
                            "cyan": str(cyan_dir),
                        },
                    },
                    indent=2,
                )
                + "\n"
            )

            cfg = helper_vision_config.load_vision_model(path=model_path)
            self.assertEqual(cfg["active_mode"], helper_vision_config.VISION_MODE_CYAN)
            self.assertEqual(helper_vision_config.active_vision_mode(path=model_path), helper_vision_config.VISION_MODE_CYAN)
            self.assertEqual(helper_vision_config.demos_dir_for_mode("aruco", path=model_path), aruco_dir)
            self.assertEqual(helper_vision_config.demos_dir_for_mode("cyan", path=model_path), cyan_dir)

    def test_aruco_can_fallback_to_legacy_demos_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_dir = root / "legacy-demos"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            model_path = root / "world_model_vision.json"
            model_path.write_text(
                json.dumps(
                    {
                        "active_mode": "aruco",
                        "demos_by_mode": {
                            "aruco": str(root / "missing-aruco"),
                            "cyan": str(root / "missing-cyan"),
                        },
                    },
                    indent=2,
                )
                + "\n"
            )

            original_legacy = helper_vision_config.LEGACY_DEMOS_DIR
            try:
                helper_vision_config.LEGACY_DEMOS_DIR = legacy_dir
                demos_dir = helper_vision_config.demos_dir_for_mode("aruco", path=model_path)
            finally:
                helper_vision_config.LEGACY_DEMOS_DIR = original_legacy
            self.assertEqual(demos_dir, legacy_dir)


if __name__ == "__main__":
    unittest.main()
