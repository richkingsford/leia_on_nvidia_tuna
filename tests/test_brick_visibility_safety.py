import unittest

from helper_brick_visibility_safety import brick_motion_allowed


class TestBrickVisibilitySafety(unittest.TestCase):
    def test_forward_recovery_can_start_beyond_virtual_wall(self):
        reading = {
            "visible": True,
            "confident": True,
            "conf": 98.0,
            "min_confidence_pct": 75.0,
            "dist_mm": 382.0,
            "reason": "confident_visible",
        }

        self.assertFalse(brick_motion_allowed(reading))
        self.assertTrue(
            brick_motion_allowed(
                reading,
                allow_virtual_safety_forward_recovery=True,
            )
        )

    def test_forward_recovery_still_requires_confidence(self):
        reading = {
            "visible": True,
            "confident": False,
            "conf": 40.0,
            "min_confidence_pct": 75.0,
            "dist_mm": 382.0,
            "reason": "low_confidence",
        }

        self.assertFalse(
            brick_motion_allowed(
                reading,
                allow_virtual_safety_forward_recovery=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
