import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_calibrate_telemetry


class TestTightCapture(unittest.TestCase):
    def test_capture_tight_pose_accepts_small_spread(self):
        poses = [
            {"dist": 100.0, "offset_x": 1.0, "offset_y": 20.0, "angle": 0.0, "confidence": 0.9, "obs_ts": 1.0},
            {"dist": 101.0, "offset_x": 1.2, "offset_y": 20.2, "angle": 0.0, "confidence": 0.8, "obs_ts": 2.0},
            {"dist": 99.5, "offset_x": 1.1, "offset_y": 20.1, "angle": 0.0, "confidence": 0.85, "obs_ts": 3.0},
        ]
        with patch.object(helper_calibrate_telemetry, "_read_pose", side_effect=poses):
            pose, meta = helper_calibrate_telemetry._capture_tight_pose(
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
        with patch.object(helper_calibrate_telemetry, "_read_pose", side_effect=poses):
            pose, meta = helper_calibrate_telemetry._capture_tight_pose(
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
        with patch.object(helper_calibrate_telemetry, "_read_pose", side_effect=[None]):
            pose, meta = helper_calibrate_telemetry._capture_tight_pose(
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
        with patch.object(helper_calibrate_telemetry, "_read_pose", side_effect=poses):
            pose, meta = helper_calibrate_telemetry._capture_tight_pose(
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


class TestTelemetryCalibrationMerge(unittest.TestCase):
    def test_single_point_run_appends_into_existing_active_curve(self):
        existing_payload = helper_calibrate_telemetry._payload_from_rows(
            rows=[
                helper_calibrate_telemetry.SampleRow(1, 73.0, 64.7775, 0.0, 0.0, 40.7, None, None, None, 1.0),
                helper_calibrate_telemetry.SampleRow(2, 130.0, 113.5083, 0.0, 0.0, 76.5, None, None, None, 2.0),
                helper_calibrate_telemetry.SampleRow(3, 155.0, 136.3103, 0.0, 0.0, 80.9, None, None, None, 3.0),
            ],
            variable="dist",
            config={"variable": "dist", "captured_points": 3},
            generated_at=10.0,
        )
        existing_record = {
            "schema_version": 1,
            "run_id": 111,
            "timestamp": 10.0,
            "variable": "dist",
            "config": dict(existing_payload.get("config") or {}),
            "fit": dict(existing_payload.get("fit") or {}),
            "summary": dict(existing_payload.get("summary") or {}),
            "curve_points": [
                {
                    "index": int(item.get("index") or 0),
                    "expected_value": float(item.get("expected_value") or 0.0),
                    "observed_variable": float(item.get("observed_variable") or 0.0),
                    "observed_dist": float(item.get("observed_dist") or 0.0),
                    "observed_x": float(item.get("observed_x") or 0.0),
                    "observed_y": float(item.get("observed_y") or 0.0),
                    "error_mm": float(item.get("error_mm") or 0.0),
                    "confidence": float(item.get("confidence") or 0.0),
                }
                for item in list(existing_payload.get("samples") or [])
            ],
        }
        new_run_payload = {
            "source": "calibrate_dist_x_y_telemetry",
            "generated_at": 20.0,
            "config": {"variable": "dist", "captured_points": 1},
            "samples": [
                {
                    "index": 1,
                    "expected_value": 107.0,
                    "observed_dist": 107.3373,
                    "observed_x": 0.0,
                    "observed_y": 0.0,
                    "observed_variable": 107.3373,
                    "error_mm": 0.3373,
                    "confidence": 84.0,
                    "timestamp": 20.0,
                }
            ],
        }

        merged = helper_calibrate_telemetry._merge_incremental_run_payload(
            existing_record,
            new_run_payload,
            "dist",
        )

        self.assertEqual(int(merged["config"]["captured_points"]), 4)
        self.assertEqual(
            [round(float(item["expected_value"]), 1) for item in merged["curve_points"]],
            [73.0, 107.0, 130.0, 155.0],
        )
        self.assertNotEqual(float(merged["fit"]["slope"]), 1.0)


if __name__ == "__main__":
    unittest.main()
