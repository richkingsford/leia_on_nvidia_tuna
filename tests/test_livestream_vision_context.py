import unittest

from helper_holding_brick import HoldingMaskLock, HoldingMaskLockConfig
from livestream_crown_vision import (
    _holding_result_for_vision_context,
    _normalize_vision_context,
    _vision_context_allows_holding_model,
)


class TestLivestreamVisionContext(unittest.TestCase):
    def test_empty_context_aliases_block_holding_model(self):
        for raw in ("empty", "empty_s1", "empty-step1", "empty step 2", "empty_step3", "unknown"):
            context = _normalize_vision_context(raw)

            self.assertFalse(_vision_context_allows_holding_model(context), raw)

    def test_holding_context_aliases_allow_holding_model(self):
        for raw in ("holding", "held", "holding_s1", "holding-step2", "holding step 3"):
            context = _normalize_vision_context(raw)

            self.assertTrue(_vision_context_allows_holding_model(context), raw)

    def test_empty_context_preserves_raw_holding_diagnostic_but_blocks_model(self):
        lock = HoldingMaskLock(HoldingMaskLockConfig(release_after_misses=3))
        raw = {"holding": True, "reason": "held_brick_top_nubs", "mask_regions": [(10, 0, 20, 12)]}

        result = _holding_result_for_vision_context(raw, "empty_s1", lock)

        self.assertFalse(result["holding"])
        self.assertFalse(result["holding_model_allowed"])
        self.assertTrue(result["raw_holding"])
        self.assertEqual(result["raw_holding_reason"], "held_brick_top_nubs")
        self.assertEqual(result["reason"], "empty_context_uses_normal_model")
        self.assertEqual(result["holding_mask_blocked_by_context"], "empty_s1")

    def test_empty_context_resets_stale_holding_mask_lock(self):
        lock = HoldingMaskLock(HoldingMaskLockConfig(release_after_misses=3))
        raw = {"holding": True, "reason": "held", "mask_regions": [(10, 0, 20, 12)]}

        first = _holding_result_for_vision_context(raw, "holding_s1", lock)
        blocked = _holding_result_for_vision_context(raw, "empty_s1", lock)
        after = _holding_result_for_vision_context({"holding": False, "reason": "clear"}, "holding_s1", lock)

        self.assertTrue(first["holding"])
        self.assertFalse(blocked["holding"])
        self.assertFalse(after["holding"])
        self.assertEqual(after["reason"], "clear")

    def test_holding_context_uses_mask_lock_for_brief_flicker(self):
        lock = HoldingMaskLock(HoldingMaskLockConfig(release_after_misses=3))
        raw = {"holding": True, "reason": "held", "mask_regions": [(10, 0, 20, 12)]}

        first = _holding_result_for_vision_context(raw, "holding_s1", lock)
        second = _holding_result_for_vision_context({"holding": False, "reason": "clear"}, "holding_s1", lock)

        self.assertTrue(first["holding"])
        self.assertTrue(second["holding"])
        self.assertTrue(second["holding_mask_locked"])
        self.assertEqual(second["reason"], "held_brick_mask_lock")


if __name__ == "__main__":
    unittest.main()
