import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import calibrate_dist_x_y_telemetry


class TestTightCapture(unittest.TestCase):
    def test_capture_tight_pose_accepts_small_spread(self):
        poses = [
            {"dist": 100.0, "offset_x": 1.0, "offset_y": 20.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 1.0},
            {"dist": 101.0, "offset_x": 1.2, "offset_y": 20.2, "angle": 0.0, "confidence": 0.8, "obs_ts": 2.0},
            {"dist": 99.5, "offset_x": 1.1, "offset_y": 20.1, "angle": 0.0, "confidence": 0.85, "obs_ts": 3.0},
        ]
        with patch.object(calibrate_dist_x_y_telemetry, "_read_pose", side_effect=poses):
            pose, meta = calibrate_dist_x_y_telemetry._capture_tight_pose(
                vision=None,
                world=None,
                variable="dist",
                samples=3,
                timeout_s=1.5,
                max_spread_mm=3.0,
                trim_outliers=0,
            )
        self.assertIsNotNone(pose)
        self.assertEqual(meta.get("reason"), "ok")
        self.assertAlmostEqual(float(pose.get("dist")), (100.0 + 101.0 + 99.5) / 3.0, places=6)
        self.assertAlmostEqual(float(pose.get("observed_spread_mm")), 1.5, places=6)
        self.assertEqual(meta.get("observed_values"), [100.0, 101.0, 99.5])

    def test_capture_tight_pose_rejects_large_spread(self):
        poses = [
            {"dist": 100.0, "offset_x": 1.0, "offset_y": 20.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 1.0},
            {"dist": 110.0, "offset_x": 1.0, "offset_y": 20.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 2.0},
            {"dist": 108.0, "offset_x": 1.0, "offset_y": 20.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 3.0},
        ]
        with patch.object(calibrate_dist_x_y_telemetry, "_read_pose", side_effect=poses):
            pose, meta = calibrate_dist_x_y_telemetry._capture_tight_pose(
                vision=None,
                world=None,
                variable="dist",
                samples=3,
                timeout_s=1.5,
                max_spread_mm=3.0,
                trim_outliers=0,
            )
        self.assertIsNone(pose)
        self.assertEqual(meta.get("reason"), "spread_too_high")
        self.assertAlmostEqual(float(meta.get("spread_mm")), 10.0, places=6)
        self.assertEqual(meta.get("observed_values"), [100.0, 110.0, 108.0])

    def test_capture_tight_pose_rejects_missing_visibility(self):
        with patch.object(calibrate_dist_x_y_telemetry, "_read_pose", side_effect=[None]):
            pose, meta = calibrate_dist_x_y_telemetry._capture_tight_pose(
                vision=None,
                world=None,
                variable="y",
                samples=3,
                timeout_s=1.5,
                max_spread_mm=3.0,
                trim_outliers=0,
            )
        self.assertIsNone(pose)
        self.assertEqual(meta.get("reason"), "no_visible")

    def test_capture_tight_pose_accepts_with_trimmed_outliers(self):
        poses = [
            {"dist": 100.0, "offset_x": 0.0, "offset_y": 70.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 1.0},
            {"dist": 100.0, "offset_x": 0.0, "offset_y": 71.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 2.0},
            {"dist": 100.0, "offset_x": 0.0, "offset_y": 69.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 3.0},
            {"dist": 100.0, "offset_x": 0.0, "offset_y": 90.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 4.0},
            {"dist": 100.0, "offset_x": 0.0, "offset_y": 50.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 5.0},
        ]
        with patch.object(calibrate_dist_x_y_telemetry, "_read_pose", side_effect=poses):
            pose, meta = calibrate_dist_x_y_telemetry._capture_tight_pose(
                vision=None,
                world=None,
                variable="y",
                samples=5,
                timeout_s=2.5,
                max_spread_mm=3.0,
                trim_outliers=2,
            )
        self.assertIsNotNone(pose)
        self.assertEqual(meta.get("reason"), "ok")
        self.assertEqual(meta.get("selected_values"), [69.0, 70.0, 71.0])
        self.assertEqual(sorted(meta.get("dropped_values")), [50.0, 90.0])
        self.assertEqual(int(meta.get("dropped_count") or 0), 2)
        self.assertAlmostEqual(float(pose.get("offset_y")), 70.0, places=6)


if __name__ == "__main__":
    unittest.main()
