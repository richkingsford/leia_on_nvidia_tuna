import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_follow_the_brick as follow


class _FakeVision:
    def __init__(self, result):
        self._result = result

    def read(self):
        return self._result


class _SequenceVision:
    def __init__(self, results):
        self._results = list(results)
        self._last = self._results[-1] if self._results else (False,)

    def read(self):
        if self._results:
            self._last = self._results.pop(0)
        return self._last


class _FakeRobot:
    def __init__(self):
        self.commands = []
        self.custom_commands = []
        self.stops = 0

    def send_command_pwm(self, cmd, pwm, duration_ms=None):
        self.commands.append((cmd, pwm, duration_ms))

    def send_custom_actions_pwm(self, cmd, actions, duration_ms=None):
        self.custom_commands.append((cmd, list(actions), duration_ms))

    def stop(self):
        self.stops += 1


class _FakeRng:
    def __init__(self, choice_value, uniform_value=None):
        self.choice_value = choice_value
        self.uniform_value = uniform_value

    def choice(self, values):
        assert self.choice_value in values
        return self.choice_value

    def uniform(self, min_value, max_value):
        if self.uniform_value is None:
            return (float(min_value) + float(max_value)) / 2.0
        assert float(min_value) <= float(self.uniform_value) <= float(max_value)
        return float(self.uniform_value)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, float(seconds))


def _configured_reset_target_reading(x_mm=None):
    cfg = follow._reset_motion_config()["reverse_turn"]
    return {
        "visible": True,
        "dist_mm": float(cfg["dist_target_mm"]),
        "x_mm": float(cfg["target_abs_x_mm"] if x_mm is None else x_mm),
        "y_mm": float(cfg["y_target_mm"]),
    }


class TestFollowTheBrickReset(unittest.TestCase):
    def test_tegra_lfb_parser_reads_largest_free_block(self):
        text = (
            "RAM 5955/7620MB (lfb 4x4MB) SWAP 328/3810MB\n"
            "RAM 5900/7620MB (lfb 2x32MB)"
        )

        self.assertEqual(follow._parse_tegra_lfb_mb(text), 32.0)

    def test_worker_argv_preserves_supervised_run_options(self):
        args = follow._parse_args(["--duration-s", "12", "--reset-only", "--skip-vision-preflight"])

        argv = follow._worker_argv(args)

        self.assertIn("--worker", argv)
        self.assertIn("--reset-only", argv)
        self.assertIn("--skip-vision-preflight", argv)
        self.assertIn("--duration-s", argv)
        self.assertIn("12.0", argv)

    def test_default_game_duration_is_forty_seconds(self):
        args = follow._parse_args([])

        self.assertEqual(args.duration_s, 40.0)

    def test_distance_tolerances_are_configured_for_current_game(self):
        self.assertEqual(follow.TARGET_DIST_MM, 65.0)
        self.assertEqual(follow.DIST_TOL_MM, 9.0)
        self.assertEqual(follow.X_TOL_MM, 9.0)
        self.assertEqual(follow.Y_TOL_MM, 9.0)
        self.assertAlmostEqual(follow.RESET_DIST_TARGET_MM, follow.TARGET_DIST_MM * 1.75)
        self.assertAlmostEqual(follow.RESET_DIST_TOL_MM, 9.0)

    def test_reset_x_offset_ready_uses_configured_min_abs_offset(self):
        cfg = {"x_offset_min_mm": 25.0, "x_offset_max_mm": 45.0}

        self.assertFalse(follow._reset_x_offset_ready(24.9, cfg))
        self.assertTrue(follow._reset_x_offset_ready(25.0, cfg))
        self.assertTrue(follow._reset_x_offset_ready(-35.0, cfg))
        self.assertTrue(follow._reset_x_offset_ready(45.0, cfg))
        self.assertFalse(follow._reset_x_offset_ready(45.1, cfg))

    def test_read_brick_measurement_uses_fresh_visibility(self):
        visible = follow._read_brick_measurement(
            _FakeVision((True, 0.0, 261.5, -3.0, 88.0, 0.0, False, False))
        )
        self.assertTrue(visible["visible"])
        self.assertTrue(visible["confident"])
        self.assertEqual(visible["dist_mm"], 261.5)
        self.assertEqual(visible["x_mm"], -3.0)
        self.assertEqual(visible["conf"], 88.0)

        missing = follow._read_brick_measurement(_FakeVision((False,)))
        self.assertFalse(missing["visible"])
        self.assertFalse(missing["confident"])

    def test_wait_for_confident_brick_blocks_without_visibility(self):
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            reading = follow._wait_for_confident_brick(
                _FakeVision((False,)),
                timeout_s=0.2,
                sample_s=0.1,
            )

        self.assertFalse(reading["confident"])

    def test_wait_for_confident_brick_returns_visible_sample(self):
        reading = follow._wait_for_confident_brick(
            _FakeVision((True, 0.0, 150.0, 2.0, 88.0, 0.0, False, False)),
            timeout_s=0.0,
            sample_s=0.01,
        )

        self.assertTrue(reading["confident"])
        self.assertEqual(reading["dist_mm"], 150.0)

    def test_reset_reverse_turn_uses_gap_algorithm_for_one_backward_arc(self):
        robot = _FakeRobot()

        reading = {
            "visible": True,
            "confident": True,
            "conf": 88.0,
            "min_confidence_pct": 75.0,
            "x_mm": 0.0,
        }

        with mock.patch.object(follow, "_motion_power_scale", return_value=1.0):
            result = follow._reset_reverse_turn(robot, "r", reading, rng=_FakeRng("r", uniform_value=2000.0))

        reset_cfg = follow._reset_motion_config()["reverse_turn"]
        reset_curve = follow._reset_arc_curve_for_reading(reading, reset_cfg)
        expected_actions = follow._turn_curve_actions(
            drive_mode="backward",
            cmd="r",
            curve=reset_curve,
        )
        expected_actions = [dict(action, duration_ms=reset_cfg["pulse_ms"]) for action in expected_actions]
        expected_actions.append({"target": "m", "action": "d", "pwm": 255, "duration_ms": 2000})
        self.assertEqual(result["wheel_ms"], reset_cfg["pulse_ms"])
        self.assertEqual(result["mast_up_ms"], 2000)
        self.assertEqual(result["duration_ms"], 2000)
        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, sent_duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "r")
        self.assertEqual(sent_duration_ms, reset_cfg["pulse_ms"])
        self.assertEqual(actions, expected_actions)
        self.assertEqual(reset_curve["x_gap_mm"], reset_cfg["target_abs_x_mm"])
        self.assertEqual(reset_curve["slower_pwm"], 111)
        self.assertEqual(reset_curve["faster_pwm"], 171)

    def test_reset_arc_algorithm_interpolates_for_smaller_x_gap(self):
        robot = _FakeRobot()

        reading = {
            "visible": True,
            "confident": True,
            "conf": 88.0,
            "min_confidence_pct": 75.0,
            "x_mm": follow._reset_motion_config()["reverse_turn"]["target_abs_x_mm"] - 20.0,
        }

        with mock.patch.object(follow, "_motion_power_scale", return_value=1.0):
            result = follow._reset_reverse_turn(robot, "l", reading, rng=_FakeRng("l", uniform_value=2000.0))

        reset_cfg = follow._reset_motion_config()["reverse_turn"]
        reset_curve = follow._reset_arc_curve_for_reading(reading, reset_cfg)
        expected_actions = follow._turn_curve_actions(
            drive_mode="backward",
            cmd="l",
            curve=reset_curve,
        )
        expected_actions = [dict(action, duration_ms=reset_cfg["pulse_ms"]) for action in expected_actions]
        expected_actions.append({"target": "m", "action": "d", "pwm": 255, "duration_ms": 2000})
        self.assertEqual(result["wheel_ms"], reset_cfg["pulse_ms"])
        self.assertEqual(result["mast_up_ms"], 2000)
        self.assertEqual(result["duration_ms"], 2000)
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, sent_duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "l")
        self.assertEqual(sent_duration_ms, reset_cfg["pulse_ms"])
        self.assertEqual(actions, expected_actions)
        self.assertEqual(reset_curve["x_gap_mm"], 20.0)
        self.assertEqual(reset_curve["slower_pwm"], 108)
        self.assertEqual(reset_curve["faster_pwm"], 156)

    def test_reverse_turn_until_x_offset_refuses_to_move_without_visible_brick(self):
        robot = _FakeRobot()

        ok, reason, reading = follow._reverse_turn_until_x_offset(
            _FakeVision((False,)),
            robot,
            direction="l",
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "brick_not_confident_before_reset_motion")
        self.assertFalse(reading["visible"])
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.stops, 1)

    def test_reverse_turn_until_x_offset_sends_backward_turn_and_reset_mast_up(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 4.0, 88.0, 0.0, False, False),
                (True, 0.0, 176.0, 6.0, 88.0, 0.0, False, False),
            ]
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "sleep", side_effect=fake_clock.sleep):
            ok, reason, reading = follow._reverse_turn_until_x_offset(
                vision,
                robot,
                direction="l",
                rng=_FakeRng("l", uniform_value=2000.0),
            )

        expected_actions = follow._turn_curve_actions(
            drive_mode="backward",
            cmd="l",
            curve=follow._reset_arc_curve_for_reading(
                {
                    "visible": True,
                    "confident": True,
                    "conf": 88.0,
                    "min_confidence_pct": 75.0,
                    "dist_mm": follow.TARGET_DIST_MM,
                    "x_mm": 4.0,
                },
                follow._reset_motion_config()["reverse_turn"],
            ),
        )
        expected_actions = [
            dict(action, duration_ms=follow._reset_motion_config()["reverse_turn"]["pulse_ms"])
            for action in expected_actions
        ]
        expected_actions.append({"target": "m", "action": "d", "pwm": 255, "duration_ms": 2000})
        self.assertTrue(ok)
        self.assertEqual(reason, "one_act_complete")
        self.assertEqual(reading["dist_mm"], 176.0)
        self.assertEqual(reading["x_mm"], 6.0)
        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, _duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "l")
        self.assertEqual(actions, expected_actions)
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_stops_on_first_missing_frame_without_stale_motion(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 200.0, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            follow._follow_loop(vision, robot, duration_s=0.5)

        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0][0], "f")
        self.assertEqual(robot.custom_commands, [])
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_uses_micro_bias_when_x_gap_is_open_and_dist_is_far(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 45.0,
                follow.X_TOL_MM + 0.8,
                88.0,
                follow.Y_TARGET_MM,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "f")
        self.assertEqual(duration_ms, follow.PULSE_MS)
        self.assertEqual(
            actions,
            [
                {"target": "l", "action": "b", "pwm": 106},
                {"target": "r", "action": "f", "pwm": 103},
            ],
        )

    def test_follow_loop_creeps_dist_in_discrete_readback_acts(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 45.0,
                0.0,
                88.0,
                follow.Y_TARGET_MM,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            follow._follow_loop(vision, robot, duration_s=0.3)

        self.assertEqual(robot.commands, [("f", follow._normal_drive_pwm("f"), follow.PULSE_MS)])
        self.assertEqual(robot.custom_commands, [])

    def test_follow_loop_uses_micro_bias_for_small_x_and_dist_gaps(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                follow.Y_TARGET_MM,
                follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 1.0,
                follow.X_TOL_MM + 1.5,
                88.0,
                follow.Y_TARGET_MM,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "f")
        self.assertEqual(duration_ms, follow.PULSE_MS)
        self.assertEqual(
            actions,
            [
                {"target": "l", "action": "b", "pwm": 106},
                {"target": "r", "action": "f", "pwm": 103},
            ],
        )

    def test_follow_loop_uses_turn_curve_when_only_x_gap_is_open(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM,
                follow.X_TOL_MM + 0.5,
                88.0,
                follow.Y_TARGET_MM,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "r")
        self.assertEqual(duration_ms, follow.PULSE_MS)
        self.assertEqual(
            actions,
            follow._turn_curve_actions(
                drive_mode="forward",
                cmd="r",
                curve=follow._turn_curve_for_drive_mode("forward", "gentle"),
            ),
        )

    def test_follow_loop_prioritizes_y_before_forward_dist_when_x_aligned(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 35.0,
                0.0,
                88.0,
                follow.Y_TARGET_MM + follow.Y_TOL_MM + 2.0,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("d", 40, 220)])
        self.assertEqual(stats["act_counts"], {"MAST_D": 1})

    def test_follow_loop_prioritizes_mast_only_when_y_gap_is_large(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 35.0,
                0.0,
                88.0,
                follow.Y_TARGET_MM + 40.0,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0], ("d", 40, 220))
        self.assertEqual(stats["act_counts"], {"MAST_D_PROTECT": 1})

    def test_follow_loop_prioritizes_x_when_x_and_y_gaps_are_large(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 35.0,
                follow.X_TOL_MM + 30.0,
                88.0,
                follow.Y_TARGET_MM + 40.0,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, _duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "f")
        self.assertEqual(actions[-1], {"target": "m", "action": "u", "pwm": 40})
        self.assertEqual(stats["act_counts"], {"BIAS_R_STRONG_MAST_D": 1})

    def test_follow_loop_runs_y_lock_on_near_ninety_mm(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                90.0,
                0.0,
                88.0,
                follow.Y_TARGET_MM + 20.0,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("d", 255, 1000)])
        self.assertEqual(stats["act_counts"], {"Y_LOCK_MAST_D": 1})
        self.assertFalse(stats["y_lock_on_armed"])

    def test_follow_loop_uses_stronger_escape_when_too_close(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 25.0,
                0.0,
                88.0,
                follow.Y_TARGET_MM,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("b", follow._approved_straight_drive_pwm("b"), 500)])
        self.assertEqual(stats["act_counts"], {"BCK_ESCAPE": 1})

    def test_approved_straight_drive_pwm_matches_uno_effective_score_one_floor(self):
        self.assertEqual(follow._speed_pwm("b", follow.SPEED_SCORE), 103)
        self.assertEqual(follow._approved_straight_drive_pwm("b"), 104)

    def test_follow_loop_combines_too_close_escape_and_y_correction_with_wire_polarity(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 25.0,
                0.0,
                88.0,
                follow.Y_TARGET_MM + follow.Y_TOL_MM + 20.0,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "b")
        self.assertEqual(duration_ms, 500)
        self.assertEqual(actions[0], {"target": "l", "action": "f", "pwm": follow._approved_straight_drive_pwm("b")})
        self.assertEqual(actions[1], {"target": "r", "action": "b", "pwm": follow._approved_straight_drive_pwm("b")})
        self.assertEqual(actions[-1], {"target": "m", "action": "u", "pwm": 40})
        self.assertEqual(stats["act_counts"], {"BCK_ESCAPE_MAST_D": 1})

    def test_too_close_escape_clamps_configured_pwm_to_approved_top_speed(self):
        with mock.patch.object(
            follow,
            "_follow_motion_config",
            return_value={
                "too_close_escape": {
                    "pwm": follow._approved_straight_drive_pwm("b") + 80,
                    "pulse_ms": 500,
                    "attach_mast": False,
                }
            },
        ):
            policy = follow._too_close_escape_policy()

        self.assertEqual(policy["pwm"], follow._approved_straight_drive_pwm("b"))
        self.assertEqual(policy["pulse_ms"], 500)

    def test_follow_act_duration_is_capped_at_half_second(self):
        self.assertEqual(follow._bounded_act_duration_ms(999), 500)

    def test_follow_loop_blocks_low_confidence_visible_brick(self):
        robot = _FakeRobot()
        vision = _FakeVision((True, 0.0, 200.0, 0.0, 60.0, follow.Y_TARGET_MM, False, False))
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            follow._follow_loop(vision, robot, duration_s=0.075)

        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.custom_commands, [])
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_does_not_record_win_closeness_without_win(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + (follow.DIST_TOL_MM / 2.0),
                follow.X_TOL_MM * 2.0,
                88.0,
                follow.Y_TARGET_MM,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(stats["win_count"], 0)
        self.assertEqual(stats["sample_count"], 1)
        self.assertEqual(stats["confident_sample_count"], 1)
        self.assertEqual(stats["follow_attempt_count"], 1)
        self.assertEqual(stats["act_counts"], {"TURN_R": 1})
        self.assertEqual(stats["miss_reasons"], {"x_outside": 1})
        self.assertEqual(stats["win_dist_target_closeness_pct"], [])
        self.assertEqual(stats["win_x_target_closeness_pct"], [])
        self.assertEqual(stats["win_target_closeness_pct"], [])
        self.assertEqual(len(stats["non_win_dist_target_closeness_pct"]), 1)
        self.assertAlmostEqual(stats["non_win_dist_target_closeness_pct"][0], 50.0)
        self.assertEqual(stats["non_win_x_target_closeness_pct"], [0.0])
        self.assertEqual(len(stats["non_win_target_closeness_pct"]), 1)
        self.assertAlmostEqual(stats["non_win_target_closeness_pct"][0], 25.0)

    def test_follow_loop_counts_win_then_runs_reset_and_tracks_x_after_reset(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()
        reset_reading = _configured_reset_target_reading(x_mm=-follow._reset_motion_config()["reverse_turn"]["target_abs_x_mm"])
        reset_result = {
            "success": True,
            "phase": "reverse_turn",
            "reason": "x_offset_confirmed",
            "turn_cmd": "l",
            "mast_up_sent": True,
            "reading": reset_reading,
            "target_met": True,
        }

        with mock.patch.object(follow, "_run_reset_sequence", return_value=reset_result) as reset_mock, mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.26)

        reset_mock.assert_called_once_with(vision, robot)
        self.assertEqual(stats["win_count"], 1)
        self.assertEqual(stats["reset_attempt_count"], 1)
        self.assertEqual(stats["reset_count"], 1)
        self.assertEqual(stats["reset_target_met_count"], 1)
        self.assertEqual(stats["sample_count"], 3)
        self.assertEqual(stats["confident_sample_count"], 2)
        self.assertEqual(stats["not_confident_count"], 1)
        self.assertEqual(stats["follow_attempt_count"], 0)
        self.assertEqual(stats["act_counts"], {"RESET_BACK_TURN_L": 1, "RESET_MAST_U": 1})
        self.assertEqual(stats["miss_reasons"], {"brick_not_confident": 1})
        target_abs_x = follow._reset_motion_config()["reverse_turn"]["target_abs_x_mm"]
        self.assertEqual(stats["reset_x_after_mm"], [-target_abs_x])
        self.assertEqual(stats["reset_abs_x_after_mm"], [target_abs_x])
        self.assertEqual(follow._avg_reset_abs_x_after_mm(stats), target_abs_x)
        self.assertEqual(stats["win_dist_target_closeness_pct"], [100.0])
        self.assertEqual(stats["win_x_target_closeness_pct"], [100.0])
        self.assertEqual(stats["win_y_target_closeness_pct"], [100.0])
        self.assertEqual(stats["win_target_closeness_pct"], [100.0])
        self.assertEqual(stats["reset_dist_target_closeness_pct"], [100.0])
        self.assertEqual(stats["reset_x_target_closeness_pct"], [100.0])
        self.assertEqual(stats["reset_y_target_closeness_pct"], [100.0])
        self.assertEqual(stats["reset_target_closeness_pct"], [100.0])

    def test_follow_loop_rejects_skimmed_happy_zone_after_stop(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 12.0,
                    0.0,
                    88.0,
                    follow.Y_TARGET_MM,
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow, "_run_reset_sequence") as reset_mock, mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.2)

        reset_mock.assert_not_called()
        self.assertEqual(stats["win_count"], 0)
        self.assertEqual(stats["sample_count"], 2)
        self.assertEqual(stats["confident_sample_count"], 2)
        self.assertEqual(stats["follow_attempt_count"], 0)
        self.assertEqual(stats["miss_reasons"], {"happy_not_stopped": 1})
        self.assertEqual(stats["win_target_closeness_pct"], [])
        self.assertEqual(stats["closest_non_win"]["action"], "HAPPY_REJECT")
        self.assertEqual(robot.commands, [])
        self.assertGreaterEqual(robot.stops, 1)

    def test_results_table_includes_win_and_reset_closeness_averages(self):
        stats = follow._new_game_stats()
        stats["win_count"] = 2
        stats["reset_attempt_count"] = 1
        stats["reset_count"] = 1
        stats["reset_target_met_count"] = 1
        stats["sample_count"] = 12
        stats["confident_sample_count"] = 10
        stats["not_confident_count"] = 2
        stats["follow_attempt_count"] = 8
        stats["act_counts"]["TURN_R"] = 5
        stats["act_counts"]["FWD"] = 3
        stats["miss_reasons"]["x_outside"] = 5
        stats["miss_reasons"]["too_far"] = 3
        stats["closest_non_win"] = {
            "action": "TURN_R",
            "reason": "x_outside",
            "dist_err": 1.0,
            "x_err": 6.0,
            "dist_closeness_pct": 95.0,
            "x_closeness_pct": 80.0,
            "closeness_pct": 87.5,
        }
        stats["last_non_win"] = {
            "action": "FWD",
            "reason": "too_far",
            "dist_err": 25.0,
            "x_err": 2.0,
            "dist_closeness_pct": 0.0,
            "x_closeness_pct": 60.0,
            "closeness_pct": 30.0,
        }
        stats["win_dist_target_closeness_pct"].append(80.0)
        stats["win_dist_target_closeness_pct"].append(60.0)
        stats["win_x_target_closeness_pct"].append(60.0)
        stats["win_x_target_closeness_pct"].append(20.0)
        stats["win_y_target_closeness_pct"].append(100.0)
        stats["win_y_target_closeness_pct"].append(80.0)
        stats["win_target_closeness_pct"].append(70.0)
        stats["win_target_closeness_pct"].append(40.0)
        stats["reset_dist_target_closeness_pct"].append(100.0)
        stats["reset_x_target_closeness_pct"].append(90.0)
        stats["reset_y_target_closeness_pct"].append(80.0)
        stats["reset_target_closeness_pct"].append(95.0)
        stats["reset_abs_x_after_mm"].append(36.0)
        stats["reset_dist_after_mm"].append(follow.RESET_DIST_TARGET_MM)
        stats["reset_y_after_mm"].append(follow._reset_motion_config()["reverse_turn"]["y_target_mm"])

        table = follow._format_game_results_table(stats)

        self.assertIn("Close avg±sd", table)
        self.assertIn("| Win | 2 | 2 | 55%±15% [######----] | 70%±10% | 40%±20% | 90%±10% | target dist 65.0mm, x 0.0mm, y -5.8mm |", table)
        self.assertIn("| Reset | 1/1 | 1/1 | 95%±0% [##########] | 100%±0% | 90%±0% | 80%±0% | dist 113.8mm, |x| 36.0mm, y -9.4mm |", table)
        self.assertIn("| Movement attempts | 8 |", table)
        self.assertIn("| Planned: TURN_R=5, FWD=3 | 8 |", table)
        self.assertIn("| Sent: none | 0 |", table)
        self.assertIn("| Miss reasons | x_outside=5, too_far=3 |", table)
        self.assertIn("Closest non-win | 88% (dist=95% x=80%, dist_err=+1.0mm x_err=+6.0mm, TURN_R)", table)

    def test_follow_drive_never_requests_above_approved_top_speed(self):
        robot = _FakeRobot()
        reading = {"visible": True, "confident": True, "conf": 88.0, "min_confidence_pct": 75.0}

        with mock.patch.object(follow, "_motion_power_scale", return_value=1.05):
            follow._drive(robot, "b", reading)

        self.assertEqual(robot.commands, [("b", follow._approved_straight_drive_pwm("b"), follow.PULSE_MS)])

    def test_follow_drive_never_requests_below_floor_pwm(self):
        robot = _FakeRobot()
        reading = {"visible": True, "confident": True, "conf": 88.0, "min_confidence_pct": 75.0}

        with mock.patch.object(follow, "_motion_power_scale", return_value=0.1):
            follow._drive(robot, "b", reading)

        floor_pwm = follow._pwm_floor_for_cmd("b")
        self.assertEqual(robot.commands, [("b", floor_pwm, follow.PULSE_MS)])

    def test_follow_curve_scales_custom_action_pwm_from_world_model_config(self):
        robot = _FakeRobot()
        reading = {
            "visible": True,
            "confident": True,
            "conf": 88.0,
            "min_confidence_pct": 75.0,
            "x_mm": 12.0,
        }

        with mock.patch.object(follow, "_motion_power_scale", return_value=1.05):
            follow._curve_forward(robot, "r", reading)

        expected_actions = follow._turn_curve_actions(
            drive_mode="forward",
            cmd="r",
            curve=follow._turn_curve_for_drive_mode("forward", "medium"),
        )
        self.assertEqual(len(robot.custom_commands), 1)
        _cmd, actions, _duration_ms = robot.custom_commands[0]
        scaled_expected_actions = []
        for action in expected_actions:
            row = dict(action)
            scaled_pwm = follow._telemetry_robot.clamp_pwm(round(float(row.get("pwm") or 0) * 1.05))
            row["pwm"] = max(follow._pwm_floor_for_cmd(row.get("action")), scaled_pwm)
            scaled_expected_actions.append(row)
        self.assertEqual(actions, scaled_expected_actions)

    def test_follow_curve_never_requests_custom_action_below_floor_pwm(self):
        robot = _FakeRobot()
        reading = {
            "visible": True,
            "confident": True,
            "conf": 88.0,
            "min_confidence_pct": 75.0,
            "x_mm": 24.0,
        }

        with mock.patch.object(follow, "_motion_power_scale", return_value=0.1):
            follow._curve_forward(robot, "r", reading)

        self.assertEqual(len(robot.custom_commands), 1)
        _cmd, actions, _duration_ms = robot.custom_commands[0]
        for action in actions:
            self.assertGreaterEqual(action["pwm"], follow._pwm_floor_for_cmd(action["action"]))

    def test_follow_strong_curve_uses_demonstrated_forward_turn_pair(self):
        robot = _FakeRobot()
        reading = {
            "visible": True,
            "confident": True,
            "conf": 88.0,
            "min_confidence_pct": 75.0,
            "x_mm": 24.0,
        }

        with mock.patch.object(follow, "_motion_power_scale", return_value=1.0):
            follow._curve_forward(robot, "r", reading)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "r")
        self.assertEqual(duration_ms, follow.PULSE_MS)
        self.assertEqual(
            actions,
            [
                {"target": "l", "action": "b", "pwm": 209},
                {"target": "r", "action": "f", "pwm": 104},
            ],
        )

    def test_follow_turn_curves_load_six_world_model_values(self):
        expected_outer = {
            "gentle": 155,
            "medium": 181,
            "strong": 209,
        }

        for drive_mode in ("forward", "backward"):
            for strength, outer_pwm in expected_outer.items():
                curve = follow._turn_curve_for_drive_mode(drive_mode, strength)
                self.assertEqual(curve["drive_mode"], drive_mode)
                self.assertEqual(curve["strength"], strength)
                self.assertEqual(curve["inner_pwm"], 104)
                self.assertEqual(curve["outer_pwm"], outer_pwm)

    def test_backward_turn_curves_mirror_forward_tread_actions(self):
        forward = follow._turn_curve_for_drive_mode("forward", "gentle")
        backward = follow._turn_curve_for_drive_mode("backward", "gentle")

        self.assertEqual(
            follow._turn_curve_actions(drive_mode="forward", cmd="r", curve=forward),
            [
                {"target": "l", "action": "b", "pwm": 155},
                {"target": "r", "action": "f", "pwm": 104},
            ],
        )
        self.assertEqual(
            follow._turn_curve_actions(drive_mode="backward", cmd="r", curve=backward),
            [
                {"target": "l", "action": "f", "pwm": 155},
                {"target": "r", "action": "b", "pwm": 104},
            ],
        )
        self.assertEqual(
            follow._turn_curve_actions(drive_mode="forward", cmd="l", curve=forward),
            [
                {"target": "l", "action": "b", "pwm": 104},
                {"target": "r", "action": "f", "pwm": 155},
            ],
        )
        self.assertEqual(
            follow._turn_curve_actions(drive_mode="backward", cmd="l", curve=backward),
            [
                {"target": "l", "action": "f", "pwm": 104},
                {"target": "r", "action": "b", "pwm": 155},
            ],
        )

    def test_reset_sequence_randomizes_reverse_turn_direction(self):
        robot = _FakeRobot()
        vision = _FakeVision((False,))
        offset_reading = _configured_reset_target_reading(
            x_mm=-follow._reset_motion_config()["reverse_turn"]["target_abs_x_mm"]
        )

        with mock.patch.object(
            follow,
            "_reverse_turn_until_x_offset",
            return_value=(True, "x_offset_confirmed", offset_reading),
        ) as reset_mock, mock.patch.object(
            follow,
            "_reset_post_pause_s",
            return_value=0.0,
        ):
            result = follow._run_reset_sequence(
                vision,
                robot,
                rng=_FakeRng("l"),
            )

        reset_mock.assert_called_once_with(vision, robot, direction="l", rng=mock.ANY)
        self.assertTrue(result["success"])
        self.assertEqual(result["turn_cmd"], "l")
        self.assertEqual(result["reason"], "x_offset_confirmed")
        self.assertTrue(result["target_met"])

    def test_reverse_turn_until_x_offset_pauses_before_scoring_reset(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 4.0, 88.0, 0.0, False, False),
                (
                    True,
                    0.0,
                    follow._reset_motion_config()["reverse_turn"]["dist_target_mm"],
                    follow._reset_motion_config()["reverse_turn"]["target_abs_x_mm"],
                    88.0,
                    follow._reset_motion_config()["reverse_turn"]["y_target_mm"],
                    False,
                    False,
                ),
            ]
        )

        with mock.patch.object(
            follow,
            "_reset_post_pause_s",
            return_value=2.0,
        ), mock.patch.object(
            follow.time,
            "sleep",
        ) as sleep_mock:
            ok, reason, _reading = follow._reverse_turn_until_x_offset(
                vision,
                robot,
                direction="r",
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "target_hit")
        sleep_mock.assert_any_call(2.0)


if __name__ == "__main__":
    unittest.main()
