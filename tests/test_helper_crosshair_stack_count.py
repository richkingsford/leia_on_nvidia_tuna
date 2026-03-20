import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_crosshair_stack_count


class HelperCrosshairStackCountTests(unittest.TestCase):
    def test_summarize_crosshair_stack_samples_supports_initial_bricks_offset(self):
        samples = [False, True, False, True, False, True, False, False]

        summary = helper_crosshair_stack_count.summarize_crosshair_stack_samples(
            samples,
            initial_bricks=2,
        )

        self.assertEqual(int(summary["brick_bands"]), 3)
        self.assertEqual(int(summary["internal_gap_bands"]), 2)
        self.assertEqual(int(summary["initial_bricks"]), 2)
        self.assertEqual(int(summary["estimated_bricks"]), 5)


if __name__ == "__main__":
    unittest.main()
