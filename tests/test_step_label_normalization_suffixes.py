import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_demo_log_utils import normalize_step_label
import telemetry_brick


class TestStepLabelNormalizationSuffixes(unittest.TestCase):
    def test_normalize_step_label_strips_align_suffix(self):
        self.assertEqual(normalize_step_label("FIND_WALL2_ALIGN"), "FIND_WALL2")
        self.assertEqual(normalize_step_label("carry_align"), "FIND_WALL2")
        self.assertEqual(normalize_step_label("ALIGN_BRICK"), "ALIGN_BRICK")

    def test_telemetry_brick_step_name_strips_align_suffix(self):
        self.assertEqual(telemetry_brick._step_name("FIND_WALL2_ALIGN"), "FIND_WALL2")
        self.assertEqual(telemetry_brick._step_name("carry_align"), "FIND_WALL2")


if __name__ == "__main__":
    unittest.main()
