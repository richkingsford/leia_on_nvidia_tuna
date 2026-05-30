import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import practiceStep12EmptyHalfReset as step12


class TestStep12PracticeSafety(unittest.TestCase):
    def test_far_low_confident_reading_is_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 299.5, "y_mm": -85.4}

        self.assertTrue(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-70.0,
            )
        )

    def test_normal_far_start_is_not_pickup_suspect(self):
        reading = {"confident": True, "dist_mm": 323.8, "y_mm": -32.2}

        self.assertFalse(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-70.0,
            )
        )

    def test_not_confident_reading_uses_existing_visibility_stop_path(self):
        reading = {"confident": False, "dist_mm": 299.5, "y_mm": -85.4}

        self.assertFalse(
            step12._pregame_pickup_suspected(
                reading,
                min_dist_mm=260.0,
                max_y_mm=-70.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
