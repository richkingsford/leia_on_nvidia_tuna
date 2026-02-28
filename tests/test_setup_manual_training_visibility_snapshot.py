import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class TestSetupManualTrainingVisibilitySnapshot(unittest.TestCase):
    def test_smoothed_snapshot_uses_current_raw_visibility(self):
        study = {
            "average": {
                "found": True,
                "dist": 240.0,
                "angle": 12.0,
                "offset_x": -120.0,
                "conf": 88.0,
                "cam_h": 10.0,
                "brick_above": False,
                "brick_below": True,
            }
        }
        snapshot = setup_manual_training._study_snapshot_or_raw(
            study,
            found=False,
            angle=1.0,
            dist=2.0,
            offset_x=3.0,
            conf=4.0,
            cam_h=5.0,
            brick_above=True,
            brick_below=False,
        )
        self.assertFalse(snapshot["found"])
        self.assertEqual(snapshot["dist"], 240.0)
        self.assertEqual(snapshot["angle"], 12.0)
        self.assertEqual(snapshot["offset_x"], -120.0)

    def test_no_average_falls_back_to_raw_frame(self):
        snapshot = setup_manual_training._study_snapshot_or_raw(
            {"average": None},
            found=False,
            angle=7.0,
            dist=300.0,
            offset_x=-10.0,
            conf=15.0,
            cam_h=8.0,
            brick_above=False,
            brick_below=False,
        )
        self.assertFalse(snapshot["found"])
        self.assertEqual(snapshot["dist"], 300.0)
        self.assertEqual(snapshot["angle"], 7.0)
        self.assertEqual(snapshot["offset_x"], -10.0)

    def test_no_study_uses_raw_frame(self):
        snapshot = setup_manual_training._study_snapshot_or_raw(
            None,
            found=True,
            angle=5.0,
            dist=180.0,
            offset_x=22.0,
            conf=91.0,
            cam_h=11.0,
            brick_above=True,
            brick_below=False,
        )
        self.assertTrue(snapshot["found"])
        self.assertEqual(snapshot["dist"], 180.0)
        self.assertEqual(snapshot["angle"], 5.0)
        self.assertEqual(snapshot["offset_x"], 22.0)


if __name__ == "__main__":
    unittest.main()
