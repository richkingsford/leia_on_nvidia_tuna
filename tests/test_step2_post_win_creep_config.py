import unittest

import a_follow_the_brick as follow


class TestStep2PostWinCreepConfig(unittest.TestCase):
    def test_empty_profile_preserves_post_win_forward_creep(self):
        follow._set_game_profile("empty")
        cfg = follow._follow_step2_config()

        self.assertEqual(cfg.get("post_win_forward_creep_ms"), 300)
        self.assertEqual(cfg.get("post_win_forward_creep_pwm"), 103)


if __name__ == "__main__":
    unittest.main()
