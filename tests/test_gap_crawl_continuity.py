import unittest
from unittest.mock import patch

import a_follow_the_brick as follow


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.001, float(seconds))


def _crawl_config():
    return {
        "crawl_pwm": 115,
        "turn_pwm": 133,
        "turn_first_enabled": True,
        "turn_phase_ms": 200,
        "turn_straight_phase_ms": 200,
        "turn_min_seg_ms": 220,
        "both_ms": 200,
        "hold_gentle_ms": 200,
        "hold_sharp_ms": 400,
        "sharp_x_mm": 14.0,
        "straight_ms": 300,
        "micro_x_deadband_mm": 1.0,
        "poll_s": 0.055,
        "command_overlap_ms": 100,
        "lost_confident_frames_before_stop": 10,
        "lost_confident_grace_s": 3.0,
        "sustained_jump_pause_s": 0.25,
    }


def _confident_reading():
    return {
        "confident": True,
        "visible": True,
        "conf": 98.0,
        "dist_mm": 140.0,
        "x_mm": 0.0,
        "y_mm": 0.0,
    }


class TestGapCrawlContinuity(unittest.TestCase):
    def setUp(self):
        follow._set_game_profile("empty")

    def _run(self, read, *, duration_s=0.8):
        clock = _Clock()
        sends = []
        stops = []
        stats = {}

        def send(_robot, _cmd, actions, **kwargs):
            sends.append((clock.now, list(actions), dict(kwargs)))
            return {"cmd_sent": "f", "blocked": False}

        with (
            patch.object(follow, "_gap_crawl_config", side_effect=_crawl_config),
            patch.object(follow, "_vision_jump_guard_config", return_value={"confirm_frames": 3}),
            patch.object(follow, "_crawl_forward_pwm", return_value=103),
            patch.object(follow, "_read_brick_measurement", side_effect=read),
            patch.object(follow, "guarded_send_custom_actions_pwm", side_effect=send),
            patch.object(follow, "_stop_robot", side_effect=lambda _robot: stops.append(clock.now)),
            patch.object(follow, "_reset_follow_reading_history"),
            patch.object(follow.time, "monotonic", side_effect=clock.monotonic),
            patch.object(follow.time, "sleep", side_effect=clock.sleep),
        ):
            won, last = follow._gap_closing_crawl(
                object(),
                object(),
                win_predicate=lambda _reading: False,
                x_target_mm=0.0,
                duration_s=duration_s,
                stats=stats,
            )
        return won, last, stats, sends, stops

    def test_confident_segments_overlap_without_intermediate_stop(self):
        won, _last, _stats, sends, stops = self._run(lambda *_args, **_kwargs: _confident_reading())

        self.assertFalse(won)
        self.assertGreaterEqual(len(sends), 3)
        self.assertEqual(stops, [])
        send_gaps = [later[0] - earlier[0] for earlier, later in zip(sends, sends[1:])]
        self.assertTrue(all(gap < 0.3 for gap in send_gaps))

    def test_confidence_loss_does_not_stop_on_tenth_frame_before_grace(self):
        read_count = 0
        readings = [_confident_reading()] + [
            {"confident": False, "visible": False, "reason": "not_visible"}
            for _ in range(10)
        ] + [_confident_reading()]

        def read(*_args, **_kwargs):
            nonlocal read_count
            read_count += 1
            return readings.pop(0) if readings else _confident_reading()

        recovery_calls = []
        with patch.object(
            follow,
            "_wait_for_visibility_recovery",
            side_effect=lambda *_args, **_kwargs: recovery_calls.append(read_count) or _confident_reading(),
        ):
            _won, _last, stats, _sends, stops = self._run(read, duration_s=1.2)

        self.assertEqual(stops, [])
        self.assertEqual(recovery_calls, [])
        self.assertIsNone(stats.get("gapcrawl_confidence_pause_count"))

    def test_confidence_loss_waits_full_grace_before_visibility_recovery(self):
        def read(*_args, **_kwargs):
            return {"confident": False, "visible": False, "reason": "not_visible"}

        recovery_times = []

        def recover(*_args, **_kwargs):
            recovery_times.append(follow.time.monotonic())
            return _confident_reading()

        with patch.object(follow, "_wait_for_visibility_recovery", side_effect=recover):
            _won, _last, stats, _sends, stops = self._run(read, duration_s=3.8)

        self.assertTrue(stops)
        self.assertTrue(recovery_times)
        self.assertGreaterEqual(recovery_times[0], 3.0)
        self.assertEqual(stats.get("gapcrawl_confidence_pause_count"), 1)

    def test_three_jump_guard_frames_trigger_brief_reobserve(self):
        read_count = 0
        jump = {
            "confident": False,
            "visible": True,
            "reason": "ghost_jump_hard_rejected",
            "ghost_jump_hard_rejected": True,
            "ghost_jump_delta": {"dist": 80.0, "x": 25.0, "vector": 83.8},
        }
        readings = [_confident_reading(), dict(jump), dict(jump), dict(jump)]

        def read(*_args, **_kwargs):
            nonlocal read_count
            read_count += 1
            return readings.pop(0) if readings else _confident_reading()

        _won, _last, stats, _sends, stops = self._run(read)

        self.assertTrue(stops)
        self.assertEqual(stats.get("gapcrawl_jump_pause_count"), 1)
        self.assertEqual(read_count >= 4, True)

    def test_holding_blind_lower_uses_active_profile_duration(self):
        follow._set_game_profile("holding")

        self.assertEqual(follow._follow_step2_config().get("seat_mast_duration_ms"), 3773)


if __name__ == "__main__":
    unittest.main()
