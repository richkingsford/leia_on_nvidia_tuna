import unittest

import livestream_crown_vision as livestream


class TestLivestreamVisionContext(unittest.TestCase):
    def test_context_aliases_are_normalized_for_display_only(self):
        self.assertEqual(livestream._normalize_vision_context("empty"), "empty_s1")
        self.assertEqual(livestream._normalize_vision_context("empty step 2"), "empty_s2")
        self.assertEqual(livestream._normalize_vision_context("holding"), "holding_s1")
        self.assertEqual(livestream._normalize_vision_context("holding-step3"), "holding_s3")
        self.assertEqual(livestream._normalize_vision_context("unknown"), "empty_s1")

    def test_alternate_holding_measurement_paths_are_removed(self):
        self.assertFalse(hasattr(livestream, "_holding_result_for_vision_context"))
        self.assertFalse(hasattr(livestream, "_choose_holding_target_result"))
        self.assertNotIn("HoldingMaskLock", livestream.__dict__)


if __name__ == "__main__":
    unittest.main()
