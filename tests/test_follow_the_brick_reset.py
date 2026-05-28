import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_follow_the_brick as follow


class _FakeVision:
    def __init__(self, result):
        self._result = result

    def set_runtime_tuning(self, **kwargs):
        self.runtime_tuning = dict(kwargs)

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


class _FakeHoldingVision:
    def __init__(self):
        self.raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.raw_frame[40:200, 120:540] = (0, 180, 80)
        self.masked_frame = None

    def read(self):
        return (True, 0.0, 37.0, -2.0, 85.0, -8.0, False, False)

    def read_frame(self, frame):
        self.masked_frame = frame.copy()
        return (True, 0.0, 120.0, 4.0, 85.0, -4.0, False, False)


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
    def setUp(self):
        follow._set_game_profile("empty")

    def test_tegra_lfb_parser_reads_largest_free_block(self):
        text = (
            "RAM 5955/7620MB (lfb 4x4MB) SWAP 328/3810MB\n"
            "RAM 5900/7620MB (lfb 2x32MB)"
        )

        self.assertEqual(follow._parse_tegra_lfb_mb(text), 32.0)

    def test_worker_argv_preserves_supervised_run_options(self):
        args = follow._parse_args(["--duration-s", "12", "--reset-only", "--park-step2", "--skip-vision-preflight"])

        argv = follow._worker_argv(args)

        self.assertIn("--worker", argv)
        self.assertIn("--reset-only", argv)
        self.assertIn("--park-step2", argv)
        self.assertIn("--skip-vision-preflight", argv)
        self.assertIn("--game-profile", argv)
        self.assertIn("auto", argv)
        self.assertIn("--duration-s", argv)
        self.assertIn("12.0", argv)

    def test_worker_does_not_reset_when_pregame_visibility_fails(self):
        args = follow._parse_args(["--skip-vision-preflight"])
        args.worker = True
        vision = _FakeVision((False,))
        robot = _FakeRobot()

        with mock.patch.object(follow, "BrickDetector", return_value=vision), mock.patch.object(
            follow,
            "Robot",
            return_value=robot,
        ), mock.patch.object(follow, "_warmup"), mock.patch.object(
            follow,
            "_wait_for_confident_brick",
            return_value=follow.brick_motion_measurement_from_result((False,)),
        ), mock.patch.object(
            follow,
            "_run_reset_sequence",
        ) as reset_mock, mock.patch.object(
            follow,
            "_run_step3_no_visibility_fallback",
        ) as fallback_mock:
            code = follow._run_worker(args)

        self.assertEqual(code, follow.PREGAME_VISIBILITY_BLOCK_EXIT)
        reset_mock.assert_not_called()
        fallback_mock.assert_not_called()
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.custom_commands, [])

    def test_supervisor_restarts_after_pregame_visibility_timeout(self):
        args = follow._parse_args(["--skip-vision-preflight"])
        returns = [follow.PREGAME_VISIBILITY_BLOCK_EXIT, 0]
        popen_calls = []

        class _FakeProcess:
            def __init__(self, argv, cwd=None):
                popen_calls.append((list(argv), cwd))
                self._returncode = returns.pop(0)

            def wait(self, timeout=None):
                return self._returncode

            def poll(self):
                return self._returncode

        with mock.patch.object(follow.subprocess, "Popen", side_effect=_FakeProcess), mock.patch.object(
            follow,
            "_emergency_stop_robot",
        ) as stop_mock, mock.patch.object(
            follow,
            "_recover_pregame_visibility",
        ) as recover_mock:
            code = follow._supervise_run(args)

        self.assertEqual(code, 0)
        self.assertEqual(len(popen_calls), 2)
        stop_mock.assert_called_once()
        recover_mock.assert_called_once()

    def test_auto_select_game_profile_uses_holding_majority(self):
        class _Vision:
            def __init__(self):
                self.raw_frame = None

            def read(self):
                return None

        vision = _Vision()
        votes = [
            {"holding": True, "reason": "held_brick_detected"},
            {"holding": False, "reason": "below_threshold"},
            {"holding": True, "reason": "held_brick_detected"},
        ]

        with mock.patch.object(follow, "detect_holding_brick", side_effect=votes):
            profile, detail = follow._auto_select_game_profile(vision, sample_s=0.0)

        self.assertEqual(profile, "holding")
        self.assertEqual(detail["holding_count"], 2)
        self.assertEqual(detail["samples"], 3)

    def test_read_brick_measurement_masks_held_brick_when_holding(self):
        vision = _FakeHoldingVision()
        holding_result = {
            "holding": True,
            "reason": "held_brick_detected",
            "best": {"bbox": (120, 40, 420, 100)},
        }

        with mock.patch.object(follow, "detect_holding_brick", return_value=holding_result):
            reading = follow._read_brick_measurement(vision)

        self.assertTrue(reading["holding"])
        self.assertTrue(reading["target_masked_for_holding"])
        self.assertEqual(reading["dist_mm"], 120.0)
        self.assertEqual(reading["unmasked_target_reading"]["dist_mm"], 37.0)
        self.assertIsNotNone(vision.masked_frame)
        self.assertEqual(int(vision.masked_frame[50, 130].sum()), 0)
        self.assertGreater(int(vision.masked_frame[50, 200].sum()), 0)
        self.assertGreater(int(vision.masked_frame[60, 200].sum()), 0)
        self.assertGreater(int(vision.masked_frame[180, 320].sum()), 0)
        self.assertGreater(int(vision.masked_frame[180, 125].sum()), 0)

    def test_read_brick_measurement_relaxes_confidence_for_holding_target(self):
        class _LowConfHoldingVision(_FakeHoldingVision):
            def __init__(self):
                super().__init__()
                self.tuning_calls = []

            def set_runtime_tuning(self, **kwargs):
                self.tuning_calls.append(dict(kwargs))

            def read_frame(self, frame):
                self.masked_frame = frame.copy()
                return (True, 0.0, 120.0, 4.0, 50.0, -4.0, False, False)

        vision = _LowConfHoldingVision()
        holding_result = {
            "holding": True,
            "reason": "held_brick_detected",
            "best": {"bbox": (120, 40, 420, 160)},
        }

        with mock.patch.object(follow, "detect_holding_brick", return_value=holding_result):
            reading = follow._read_brick_measurement(vision)

        self.assertTrue(reading["holding"])
        self.assertTrue(reading["target_masked_for_holding"])
        self.assertTrue(reading["holding_target_relaxed_confidence"])
        self.assertTrue(reading["confident"])
        self.assertEqual(reading["conf"], 50.0)
        self.assertEqual(reading["min_confidence_pct"], 40.0)
        self.assertEqual(reading["dist_mm"], 120.0)
        self.assertGreaterEqual(len(vision.tuning_calls), 2)

    def test_default_game_duration_is_twenty_five_seconds(self):
        args = follow._parse_args([])

        self.assertEqual(args.duration_s, 25.0)

    def test_distance_tolerances_are_configured_for_current_game(self):
        self.assertAlmostEqual(follow.TARGET_DIST_MM, 143.23551791647205)
        self.assertEqual(follow.DIST_TOL_MM, 8.0)
        self.assertAlmostEqual(follow._dist_target_mm(), 162.7)
        self.assertEqual(follow._dist_tol_mm(), 8.0)
        self.assertAlmostEqual(follow.X_TARGET_MM, -0.8462156212848165)
        self.assertEqual(follow.X_TOL_MM, 3.0)
        self.assertAlmostEqual(follow._x_target_mm(), 1.0)
        self.assertEqual(follow._x_tol_mm(), 4.0)
        self.assertAlmostEqual(follow.Y_TARGET_MM, -11.952850590581479)
        self.assertAlmostEqual(follow._y_win_target_mm(), -19.4)
        self.assertEqual(follow.Y_TOL_MM, 0.6)
        self.assertEqual(follow._follow_y_axis_config()["win_tol_mm"], 4.0)
        self.assertEqual(follow._win_confirmation_config()["single_win_allowance_s"], 10.0)

    def test_holding_step1_targets_can_override_dist_x_and_y_from_world_model(self):
        follow._set_game_profile("holding")

        self.assertAlmostEqual(follow._dist_target_mm(), 82.87841288248698)
        self.assertEqual(follow._dist_tol_mm(), 4.0)
        self.assertAlmostEqual(follow._x_target_mm(), -2.322967529296875)
        self.assertEqual(follow._x_tol_mm(), 3.0)
        self.assertTrue(follow._follow_y_axis_config()["enabled"])
        self.assertFalse(follow._follow_y_axis_config()["lock_on_enabled"])
        self.assertAlmostEqual(follow._y_win_target_mm(), 5.250551859537761)
        self.assertEqual(follow._y_win_tol_mm(), 2.5)
        self.assertEqual(follow._cautious_visibility_config()["motion_min_confidence_pct"], 75.0)
        self.assertAlmostEqual(follow.RESET_DIST_TARGET_MM, follow.TARGET_DIST_MM * 1.75)
        self.assertAlmostEqual(follow.RESET_DIST_TOL_MM, 9.0)

    def test_holding_step2_runs_retreat_then_hands_off_to_empty_after_reset(self):
        follow._set_game_profile("holding")
        cfg = follow._follow_motion_config()
        self.assertTrue(cfg["complete_after_step2"])
        self.assertEqual(cfg["step2"]["nickname"], "place")
        self.assertEqual(cfg["step3"]["kind"], "retreat")
        self.assertEqual(cfg["step3"]["nickname"], "retreat")
        self.assertAlmostEqual(follow._step3_retreat_target_dist_mm(cfg["step3"]), 162.7)

        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (
                    True,
                    0.0,
                    follow._dist_target_mm(),
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow._dist_target_mm(),
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow._dist_target_mm(),
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()
        step2_result = {
            "success": True,
            "target_met": True,
            "reason": "step2_targets_scored",
            "reading": {"confident": True, "dist_mm": 82.0, "x_mm": -2.0, "y_mm": -10.0},
        }
        reset_result = {
            "success": True,
            "phase": "reverse_turn",
            "reason": "x_offset_confirmed",
            "turn_cmd": "l",
            "mast_up_sent": True,
            "reading": _configured_reset_target_reading(),
            "target_met": True,
        }
        retreat_result = {
            "success": True,
            "target_met": True,
            "reason": "step3_retreat_target_dist_seen",
            "reading": {"confident": True, "dist_mm": 162.7, "x_mm": -2.0, "y_mm": -10.0},
            "drive_duration_ms": 750,
        }

        with mock.patch.object(follow, "_run_step2_seat_sequence", return_value=step2_result), mock.patch.object(
            follow,
            "_run_step3_seat_sequence",
        ) as step3_seat_mock, mock.patch.object(
            follow,
            "_run_step3_retreat_sequence",
            return_value=retreat_result,
        ) as step3_retreat_mock, mock.patch.object(
            follow,
            "_run_reset_sequence",
            return_value=reset_result,
        ) as reset_mock, mock.patch.object(
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
        step3_seat_mock.assert_not_called()
        step3_retreat_mock.assert_called_once_with(vision, robot)
        self.assertEqual(follow._active_game_profile(), "empty")
        self.assertEqual(stats["reset_count"], 1)

    def test_step3_retreat_sequence_stops_at_empty_step1_distance(self):
        follow._set_game_profile("holding")
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 120.0, -2.0, 88.0, -10.0, False, False),
                (True, 0.0, 163.0, -2.0, 88.0, -10.0, False, False),
            ]
        )

        with mock.patch.object(follow, "_min_motion_duration_ms", return_value=1), mock.patch.object(
            follow.time,
            "sleep",
            return_value=None,
        ):
            result = follow._run_step3_retreat_sequence(vision, robot)

        self.assertTrue(result["success"])
        self.assertTrue(result["target_met"])
        self.assertEqual(result["reason"], "step3_retreat_target_dist_seen")
        self.assertEqual(result["drive_duration_ms"], 250)
        self.assertEqual(robot.commands, [("b", 103, 250)])
        self.assertGreaterEqual(robot.stops, 1)

    def test_holding_large_three_gap_uses_simultaneous_bias_with_mast(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "dist_mm": 221.5,
                    "x_mm": 46.0,
                    "y_mm": -11.5,
                }
            )
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["action"], "BIAS_R_ADAPTIVE_MAST_D")
        self.assertEqual(plan["reason"], "x_polish_while_creeping_dist")
        self.assertEqual(plan["strength"], "adaptive")
        self.assertEqual(plan["mast_cmd"], "d")

    def test_holding_bias_regression_profile_uses_short_soft_observed_acts(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            reading = {
                "visible": True,
                "confident": True,
                "dist_mm": 117.7,
                "x_mm": 46.6,
                "y_mm": 13.3,
            }
            plan = follow._follow_action_plan(reading)
            curve = follow._adaptive_turn_bias_curve_for_drive_mode("forward", abs(plan["x_err"]))
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["action"], "BIAS_R_ADAPTIVE")
        self.assertEqual(plan["duration_ms"], 200)
        self.assertNotIn("mast_cmd", plan)
        self.assertEqual(curve["outer_pwm"], 116)

    def test_negative_y_error_moves_mast_up_toward_holding_step1_target(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "dist_mm": follow._dist_target_mm(),
                    "x_mm": follow._x_target_mm(),
                    "y_mm": -11.5,
                }
            )
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "u")
        self.assertEqual(plan["action"], "MAST_U")

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

    def test_read_brick_measurement_rejects_placeholder_even_with_history(self):
        vision = _SequenceVision(
            [
                (True, 0.0, 130.0, 3.0, 85.0, -2.0, False, False),
                (True, 0.0, 132.0, 4.0, 86.0, -3.0, False, False),
                (True, 0.0, 500.0, 0.0, 50.0, 0.0, False, False),
            ]
        )

        first = follow._read_brick_measurement(vision)
        second = follow._read_brick_measurement(vision)
        rejected = follow._read_brick_measurement(vision)

        self.assertTrue(first["confident"])
        self.assertTrue(second["confident"])
        self.assertFalse(rejected["confident"])
        self.assertEqual(rejected["reason"], "placeholder_500_0_0_rejected")
        self.assertTrue(rejected["placeholder_rejected"])
        self.assertEqual(rejected["dist_mm"], 500.0)

    def test_read_brick_measurement_rejects_placeholder_without_history(self):
        reading = follow._read_brick_measurement(
            _FakeVision((True, 0.0, 500.0, 0.0, 50.0, 0.0, False, False))
        )

        self.assertTrue(reading["visible"])
        self.assertFalse(reading["confident"])
        self.assertEqual(reading["reason"], "placeholder_500_0_0_rejected")

    def test_temporal_filter_rejects_single_frame_ghost_jump_until_confirmed(self):
        vision = _FakeVision((False,))
        cfg = dict(follow.DEFAULT_VISION_JUMP_GUARD_CONFIG)
        cfg.update(
            {
                "enabled": True,
                "confirm_frames": 2,
                "max_dist_jump_mm": 20.0,
                "max_x_jump_mm": 20.0,
                "max_y_jump_mm": 8.0,
                "max_vector_jump_mm": 25.0,
                "confirm_window_mm": 6.0,
            }
        )
        stable = {
            "visible": True,
            "confident": True,
            "dist_mm": 143.0,
            "x_mm": -1.0,
            "y_mm": -12.0,
            "conf": 90.0,
        }
        jump = {
            "visible": True,
            "confident": True,
            "dist_mm": 114.0,
            "x_mm": -1.5,
            "y_mm": -11.5,
            "conf": 90.0,
        }
        repeat = dict(jump)
        repeat["dist_mm"] = 115.5

        with mock.patch.object(follow, "_vision_jump_guard_config", return_value=cfg):
            accepted = follow._temporal_filter_brick_reading(vision, stable, jump_guard=True)
            rejected = follow._temporal_filter_brick_reading(vision, jump, jump_guard=True)
            confirmed = follow._temporal_filter_brick_reading(vision, repeat, jump_guard=True)

        self.assertTrue(accepted["confident"])
        self.assertFalse(rejected["confident"])
        self.assertEqual(rejected["reason"], "ghost_jump_unconfirmed")
        self.assertTrue(rejected["ghost_jump_unconfirmed"])
        self.assertTrue(confirmed["confident"])
        self.assertTrue(confirmed["jump_confirmed"])
        self.assertAlmostEqual(confirmed["dist_mm"], 115.5)

    def test_temporal_filter_can_reject_confirmed_ghost_jumps(self):
        vision = _FakeVision((False,))
        cfg = dict(follow.DEFAULT_VISION_JUMP_GUARD_CONFIG)
        cfg.update(
            {
                "enabled": True,
                "accept_confirmed_jumps": False,
                "confirm_frames": 2,
                "max_dist_jump_mm": 20.0,
                "max_x_jump_mm": 20.0,
                "max_y_jump_mm": 8.0,
                "max_vector_jump_mm": 25.0,
                "confirm_window_mm": 6.0,
            }
        )
        stable = {
            "visible": True,
            "confident": True,
            "dist_mm": 143.0,
            "x_mm": -1.0,
            "y_mm": -12.0,
            "conf": 90.0,
        }
        jump = {
            "visible": True,
            "confident": True,
            "dist_mm": 114.0,
            "x_mm": -1.5,
            "y_mm": -11.5,
            "conf": 90.0,
        }
        repeat = dict(jump)
        repeat["dist_mm"] = 115.5

        with mock.patch.object(follow, "_vision_jump_guard_config", return_value=cfg):
            accepted = follow._temporal_filter_brick_reading(vision, stable, jump_guard=True)
            rejected = follow._temporal_filter_brick_reading(vision, jump, jump_guard=True)
            confirmed_rejected = follow._temporal_filter_brick_reading(vision, repeat, jump_guard=True)

        self.assertTrue(accepted["confident"])
        self.assertFalse(rejected["confident"])
        self.assertEqual(rejected["reason"], "ghost_jump_unconfirmed")
        self.assertFalse(confirmed_rejected["confident"])
        self.assertEqual(confirmed_rejected["reason"], "ghost_jump_confirmed_rejected")
        self.assertTrue(confirmed_rejected["ghost_jump_confirmed_rejected"])

    def test_temporal_filter_reacquires_stable_lock_after_motion(self):
        vision = _FakeVision((False,))
        cfg = dict(follow.DEFAULT_VISION_JUMP_GUARD_CONFIG)
        cfg.update(
            {
                "enabled": True,
                "reacquire_frames_after_motion": 3,
                "reacquire_window_mm": 6.0,
            }
        )
        readings = [
            {"visible": True, "confident": True, "dist_mm": 160.0, "x_mm": 2.0, "y_mm": -20.0, "conf": 90.0},
            {"visible": True, "confident": True, "dist_mm": 161.0, "x_mm": 2.4, "y_mm": -20.5, "conf": 90.0},
            {"visible": True, "confident": True, "dist_mm": 160.5, "x_mm": 2.2, "y_mm": -20.2, "conf": 90.0},
        ]

        with mock.patch.object(follow, "_vision_jump_guard_config", return_value=cfg):
            follow._reset_follow_reading_history(vision)
            first = follow._temporal_filter_brick_reading(vision, readings[0], jump_guard=True)
            second = follow._temporal_filter_brick_reading(vision, readings[1], jump_guard=True)
            third = follow._temporal_filter_brick_reading(vision, readings[2], jump_guard=True)

        self.assertFalse(first["confident"])
        self.assertEqual(first["reason"], "reacquiring_stable_brick_lock")
        self.assertFalse(second["confident"])
        self.assertEqual(second["reason"], "reacquiring_stable_brick_lock")
        self.assertTrue(third["confident"])
        self.assertFalse(getattr(vision, "_follow_reacquire_after_motion", True))

    def test_temporal_filter_hard_rejects_large_still_dist_jump_during_reacquire(self):
        vision = _FakeVision((False,))
        cfg = dict(follow.DEFAULT_VISION_JUMP_GUARD_CONFIG)
        cfg.update(
            {
                "enabled": True,
                "reacquire_frames_after_motion": 3,
                "hard_reject_dist_jump_mm": 75.0,
            }
        )
        stable = {
            "visible": True,
            "confident": True,
            "dist_mm": 160.0,
            "x_mm": 2.0,
            "y_mm": -20.0,
            "conf": 90.0,
        }
        far_candidate = {
            "visible": True,
            "confident": True,
            "dist_mm": 260.0,
            "x_mm": 2.5,
            "y_mm": -21.0,
            "conf": 90.0,
        }

        with mock.patch.object(follow, "_vision_jump_guard_config", return_value=cfg):
            accepted = follow._temporal_filter_brick_reading(vision, stable, jump_guard=True)
            follow._reset_follow_reading_history(vision, allow_large_dist_jump=False)
            rejected = follow._temporal_filter_brick_reading(vision, far_candidate, jump_guard=True)

        self.assertTrue(accepted["confident"])
        self.assertFalse(rejected["confident"])
        self.assertEqual(rejected["reason"], "ghost_jump_hard_rejected")
        self.assertTrue(rejected["ghost_jump_hard_rejected"])
        self.assertAlmostEqual(rejected["ghost_jump_delta"]["dist"], 100.0)

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

    def test_visibility_recovery_waits_and_returns_recovered_sample(self):
        fake_clock = _FakeClock()
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (False,),
                (False,),
                (True, 0.0, 144.0, 3.0, 88.0, -2.0, False, False),
            ]
        )

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            reading = follow._wait_for_visibility_recovery(
                vision,
                robot,
                follow.brick_motion_measurement_from_result((False,)),
                timeout_s=3.0,
                sample_s=0.15,
                context="test",
            )

        self.assertTrue(reading["confident"])
        self.assertEqual(reading["dist_mm"], 144.0)
        self.assertGreaterEqual(robot.stops, 1)

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
            result = follow._reset_reverse_turn(robot, "r", reading, rng=_FakeRng("r", uniform_value=2750.0))

        reset_cfg = follow._reset_motion_config()["reverse_turn"]
        reset_curve = follow._reset_arc_curve_for_reading(reading, reset_cfg)
        expected_actions = follow._turn_curve_actions(
            drive_mode="backward",
            cmd="r",
            curve=reset_curve,
        )
        expected_actions, sharp_finish_ms = follow._reset_segmented_turn_actions(
            expected_actions,
            reset_cfg,
            reset_curve,
            reset_cfg["pulse_ms"],
        )
        expected_actions.append({"target": "m", "action": "u", "pwm": 255, "duration_ms": 2750})
        self.assertEqual(result["wheel_ms"], reset_cfg["pulse_ms"])
        self.assertEqual(result["gentle_ms"], reset_cfg["pulse_ms"] - 135)
        self.assertEqual(result["sharp_finish_ms"], 135)
        self.assertEqual(sharp_finish_ms, 135)
        self.assertEqual(result["mast_up_ms"], 2750)
        self.assertEqual(result["duration_ms"], reset_cfg["pulse_ms"])
        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, sent_duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "r")
        self.assertEqual(sent_duration_ms, reset_cfg["pulse_ms"])
        self.assertEqual(actions, expected_actions)
        self.assertEqual(reset_curve["x_gap_mm"], reset_cfg["target_abs_x_mm"])
        self.assertEqual(reset_curve["slower_pwm"], 104)
        self.assertEqual(reset_curve["faster_pwm"], 122)

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
            result = follow._reset_reverse_turn(robot, "l", reading, rng=_FakeRng("l", uniform_value=2750.0))

        reset_cfg = follow._reset_motion_config()["reverse_turn"]
        reset_curve = follow._reset_arc_curve_for_reading(reading, reset_cfg)
        expected_actions = follow._turn_curve_actions(
            drive_mode="backward",
            cmd="l",
            curve=reset_curve,
        )
        expected_actions, sharp_finish_ms = follow._reset_segmented_turn_actions(
            expected_actions,
            reset_cfg,
            reset_curve,
            reset_cfg["pulse_ms"],
        )
        expected_actions.append({"target": "m", "action": "u", "pwm": 255, "duration_ms": 2750})
        self.assertEqual(result["wheel_ms"], reset_cfg["pulse_ms"])
        self.assertEqual(result["gentle_ms"], reset_cfg["pulse_ms"] - 135)
        self.assertEqual(result["sharp_finish_ms"], 135)
        self.assertEqual(sharp_finish_ms, 135)
        self.assertEqual(result["mast_up_ms"], 2750)
        self.assertEqual(result["duration_ms"], reset_cfg["pulse_ms"])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, sent_duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "l")
        self.assertEqual(sent_duration_ms, reset_cfg["pulse_ms"])
        self.assertEqual(actions, expected_actions)
        self.assertEqual(reset_curve["x_gap_mm"], 10.0)
        self.assertEqual(reset_curve["slower_pwm"], 104)
        self.assertEqual(reset_curve["faster_pwm"], 118)

    def test_reverse_turn_until_x_offset_refuses_to_move_without_visible_brick(self):
        robot = _FakeRobot()

        with mock.patch.object(
            follow,
            "_wait_for_visibility_recovery",
            side_effect=lambda _vision, _robot, reading, **_kwargs: reading,
        ):
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
                (True, 0.0, 176.0, 36.0, 88.0, 0.0, False, False),
            ]
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "sleep", side_effect=fake_clock.sleep), mock.patch.object(
            follow,
            "_reset_adjustment_config",
            return_value={"enabled": False, "max_attempts": 0, "pulse_min_ms": 60, "pulse_max_ms": 260, "settle_s": 0.12},
        ):
            ok, reason, reading = follow._reverse_turn_until_x_offset(
                vision,
                robot,
                direction="l",
                rng=_FakeRng("l", uniform_value=2750.0),
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
        reset_cfg = follow._reset_motion_config()["reverse_turn"]
        reset_curve = follow._reset_arc_curve_for_reading(
            {
                "visible": True,
                "confident": True,
                "conf": 88.0,
                "min_confidence_pct": 75.0,
                "dist_mm": follow.TARGET_DIST_MM,
                "x_mm": 4.0,
            },
            reset_cfg,
        )
        expected_actions, _sharp_finish_ms = follow._reset_segmented_turn_actions(
            expected_actions,
            reset_cfg,
            reset_curve,
            reset_cfg["pulse_ms"],
        )
        expected_actions.append({"target": "m", "action": "u", "pwm": 255, "duration_ms": 2750})
        self.assertTrue(ok)
        self.assertEqual(reason, "one_act_complete")
        self.assertEqual(reading["dist_mm"], 176.0)
        self.assertEqual(reading["x_mm"], 36.0)
        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, _duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "l")
        self.assertEqual(actions, expected_actions)
        self.assertGreaterEqual(robot.stops, 1)

    def test_reverse_turn_adjusts_until_reset_x_and_dist_are_ready(self):
        robot = _FakeRobot()
        reset_cfg = follow._reset_motion_config()["reverse_turn"]
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, 0.0, False, False),
                (True, 0.0, 112.0, 45.0, 88.0, 0.0, False, False),
                (True, 0.0, 112.0, 15.0, 88.0, 0.0, False, False),
            ]
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "sleep", side_effect=fake_clock.sleep):
            ok, reason, reading = follow._reverse_turn_until_x_offset(
                vision,
                robot,
                direction="l",
                rng=_FakeRng("l", uniform_value=2750.0),
            )

        expected_adjust = follow._scaled_actions(
            follow._turn_curve_actions(
                drive_mode="backward",
                cmd="r",
                curve={
                    "inner_pwm": reset_cfg["low_x_extra_sharp_turn"]["slower_pwm"],
                    "outer_pwm": reset_cfg["low_x_extra_sharp_turn"]["faster_pwm"],
                    "slower_pwm": reset_cfg["low_x_extra_sharp_turn"]["slower_pwm"],
                    "faster_pwm": reset_cfg["low_x_extra_sharp_turn"]["faster_pwm"],
                    "x_gap_mm": 0.0,
                },
            )
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "adjusted_target_hit")
        self.assertEqual(reading["x_mm"], 15.0)
        self.assertEqual(len(robot.custom_commands), 2)
        cmd, actions, duration_ms = robot.custom_commands[1]
        self.assertEqual(cmd, "r")
        self.assertGreaterEqual(duration_ms, 60)
        self.assertEqual(actions, expected_adjust)

    def test_y_action_stall_guard_ignores_dist_and_x_jitter(self):
        stats = follow._new_game_stats()
        max_tries = int(follow._act_stall_guard_config()["y_max_no_change_tries"])
        for idx in range(max_tries):
            stats["pending_observation"] = {
                "action": "MAST_D_PROTECT",
                "dist_mm": 100.0 + float(idx),
                "x_mm": 5.0 + float(idx),
                "y_mm": 20.0,
                "y_target_mm": 0.0,
                "duration_ms": 100,
            }
            result = follow._record_observed_after_pending_act(
                stats,
                {
                    "dist_mm": 105.0 + float(idx),
                    "x_mm": 10.0 + float(idx),
                    "y_mm": 20.2,
                },
            )
            self.assertFalse(result["observed"])
            if idx < max_tries - 1:
                self.assertFalse(stats["stall_guard_triggered"])
        self.assertTrue(stats["stall_guard_triggered"])
        self.assertEqual(stats["stall_guard_reason"], "act_stall_no_observed_change:MAST_D_PROTECT")
        self.assertEqual(stats["stall_guard_detail"]["observed_mode"], "target_error")

    def test_holding_profile_halves_reset_back_duration(self):
        cfg = {"pulse_ms": 1000, "timeout_s": 1.0, "holding_pulse_scale": 0.5}
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("empty")
            self.assertEqual(follow._reset_act_duration_ms(cfg), 1000)
            follow._set_game_profile("holding")
            self.assertEqual(follow._reset_act_duration_ms(cfg), 500)
        finally:
            follow._set_game_profile(old_profile)

    def test_reset_config_uses_holding_profile_targets(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("empty")
            follow._set_game_profile("holding")
            holding_reset = follow._reset_motion_config()["reverse_turn"]
        finally:
            follow._set_game_profile(old_profile)
        self.assertAlmostEqual(holding_reset["dist_target_mm"], 105.9495, places=3)
        self.assertAlmostEqual(holding_reset["dist_tol_mm"], 30.0, places=3)
        self.assertAlmostEqual(holding_reset["target_abs_x_mm"], 17.7225, places=3)
        self.assertAlmostEqual(holding_reset["y_target_mm"], -13.716072095906299, places=3)
        self.assertAlmostEqual(holding_reset["x_offset_min_mm"], 16.2, places=3)
        self.assertAlmostEqual(holding_reset["x_offset_max_mm"], 19.2, places=3)
        self.assertAlmostEqual(holding_reset["straight_back_first"]["mast_up_delay_fraction"], 0.5, places=3)
        self.assertEqual(holding_reset["sharp_finish"]["duration_ms"], 101)
        self.assertEqual(holding_reset["low_x_extra_sharp_turn"]["duration_ms"], 150)
        self.assertEqual(holding_reset["x_goal_curve"]["max_duration_ms"], 375)
        self.assertEqual(holding_reset["x_goal_curve"]["chunk_ms"], 75)
        self.assertEqual(holding_reset["adjustment"]["pulse_min_ms"], 45)
        self.assertEqual(holding_reset["adjustment"]["pulse_max_ms"], 195)

    def test_holding_reset_delays_mast_up_until_halfway_through_straight_back(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            robot = _FakeRobot()
            reading = {
                "visible": True,
                "confident": True,
                "conf": 88.0,
                "min_confidence_pct": 75.0,
                "x_mm": 0.0,
            }

            result = follow._reset_reverse_turn(robot, "l", reading, rng=_FakeRng("l", uniform_value=500.0))
        finally:
            follow._set_game_profile(old_profile)

        self.assertIsNotNone(result)
        self.assertEqual(len(robot.custom_commands), 2)
        first_cmd, first_actions, first_duration_ms = robot.custom_commands[0]
        second_cmd, second_actions, second_duration_ms = robot.custom_commands[1]
        self.assertEqual(first_cmd, "b")
        self.assertEqual(second_cmd, "b")
        self.assertEqual(first_duration_ms, 1000)
        self.assertEqual(second_duration_ms, 1000)
        self.assertFalse(any(action.get("target") == "m" for action in first_actions))
        self.assertTrue(any(action.get("action") == "b" for action in first_actions))
        self.assertTrue(any(action.get("target") == "m" and action.get("duration_ms") == 500 for action in second_actions))

    def test_follow_loop_stops_on_first_missing_frame_without_stale_motion(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 200.0, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()
        step2_result = {
            "success": True,
            "target_met": False,
            "reason": "step2_step_timeout",
            "reading": {"confident": True, "dist_mm": 80.0, "x_mm": 0.0, "y_mm": -10.0},
        }

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            follow._follow_loop(vision, robot, duration_s=0.7)

        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0][0], "f")
        self.assertEqual(robot.custom_commands, [])
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_biases_forward_when_x_gap_is_not_polished(self):
        robot = _FakeRobot()
        forward_gap = follow._follow_motion_config()["x_only_turn"]["forward_min_dist_err_mm"] + 5.0
        x_mm = follow.X_TOL_MM + 0.8
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + forward_gap,
                x_mm,
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
        self.assertEqual(duration_ms, follow._distance_creep_duration_ms(forward_gap))
        self.assertEqual(
            actions,
            follow._scaled_actions(follow._turn_bias_actions(
                drive_mode="forward",
                turn_cmd="r",
                curve=follow._turn_bias_curve_for_drive_mode("forward", "medium"),
            )),
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

        self.assertEqual(
            robot.commands,
            [
                (
                    "f",
                    follow._normal_drive_pwm("f"),
                    follow._distance_creep_duration_ms(follow.DIST_TOL_MM + 45.0),
                )
            ],
        )
        self.assertEqual(robot.custom_commands, [])

    def test_follow_loop_drives_forward_when_only_dist_is_outside_happy_box(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                follow.Y_TARGET_MM,
                follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 1.0,
                follow._x_target_mm(),
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
        sent_cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(sent_cmd, "f")
        self.assertEqual(duration_ms, 200)
        self.assertEqual([action["pwm"] for action in actions], [0, 103])

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
                drive_mode="backward",
                cmd="r",
                curve=follow._adaptive_turn_curve_for_drive_mode(
                    "backward",
                    abs((follow.X_TOL_MM + 0.5) - follow._x_target_mm()),
                ),
            ),
        )

    def test_follow_loop_prioritizes_y_before_forward_dist_when_x_aligned(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM + (follow.DIST_TOL_MM / 2.0),
                follow._x_target_mm(),
                88.0,
                follow.Y_TARGET_MM + follow.Y_TOL_MM + 2.0,
                False,
                False,
            )
        )
        fake_clock = _FakeClock()
        y_cfg = follow._follow_y_axis_config()
        y_cfg["lock_on_enabled"] = True

        with mock.patch.object(follow, "_follow_y_axis_config", return_value=y_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("d", 255, 300)])
        self.assertEqual(stats["act_counts"], {"MAST_D": 1})

    def test_follow_loop_prioritizes_mast_only_when_y_gap_is_large(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                220.0,
                follow._x_target_mm(),
                88.0,
                follow.Y_TARGET_MM + 40.0,
                False,
                True,
            )
        )
        fake_clock = _FakeClock()
        y_cfg = follow._follow_y_axis_config()
        y_cfg["lock_on_enabled"] = True

        with mock.patch.object(follow, "_follow_y_axis_config", return_value=y_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(robot.commands[0], ("d", 255, 220))
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
        y_cfg = follow._follow_y_axis_config()
        y_cfg["lock_on_enabled"] = True

        with mock.patch.object(follow, "_follow_y_axis_config", return_value=y_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(follow.time, "sleep", side_effect=fake_clock.sleep):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(cmd, "r")
        self.assertEqual(duration_ms, follow.PULSE_MS)
        self.assertEqual(
            actions,
            follow._scaled_actions(follow._turn_curve_actions(
                drive_mode="backward",
                cmd="r",
                curve=follow._adaptive_turn_curve_for_drive_mode("backward", follow.X_TOL_MM + 30.0 - follow._x_target_mm()),
            )),
        )
        self.assertEqual(stats["act_counts"], {"TURN_R": 1})

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
        y_cfg = follow._follow_y_axis_config()
        y_cfg["lock_on_enabled"] = True

        with mock.patch.object(follow, "_follow_y_axis_config", return_value=y_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(follow.time, "sleep", side_effect=fake_clock.sleep):
            stats = follow._follow_loop(vision, robot, duration_s=0.025)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("d", 255, 400)])
        self.assertEqual(stats["act_counts"], {"Y_LOCK_MAST_D": 1})
        self.assertFalse(stats["y_lock_on_armed"])

    def test_follow_loop_releases_y_commit_when_xz_leaves_gate(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (
                    True,
                    0.0,
                    90.0,
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm() + 20.0,
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 25.0,
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm() + 10.0,
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()
        y_cfg = follow._follow_y_axis_config()
        y_cfg["lock_on_enabled"] = True

        with mock.patch.object(follow, "_follow_y_axis_config", return_value=y_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.65)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("d", 255, 400), ("b", 103, 373)])
        self.assertEqual(stats["act_counts"], {"Y_LOCK_MAST_D": 1, "BCK": 1})
        self.assertFalse(stats["y_commit_active"])
        self.assertEqual(
            stats["miss_reasons"],
            {"y_lock_on": 1, "y_commit_released_xz_outside": 1, "too_close": 1},
        )

    def test_debug_mode_does_not_terminate_on_y_commit_target_before_step_win(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (
                    True,
                    0.0,
                    90.0,
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm() + 20.0,
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 25.0,
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()
        y_cfg = follow._follow_y_axis_config()
        y_cfg["lock_on_enabled"] = True

        with mock.patch.object(follow, "_follow_y_axis_config", return_value=y_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.65, debug_mode=True)

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("d", 255, 400), ("b", 103, 373)])
        self.assertFalse(stats.get("debug_mode_terminated", False))
        self.assertNotEqual(stats.get("last_action"), "DEBUG_Y_COMMIT_TERMINATE")
        self.assertEqual(stats["y_commit_target_hit_count"], 0)
        self.assertGreaterEqual(robot.stops, 1)

    def test_debug_mode_stops_after_full_step1_win_without_step2_or_reset(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM,
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM,
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM,
                    follow._x_target_mm(),
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), mock.patch.object(
            follow,
            "_run_step2_seat_sequence",
        ) as step2_mock, mock.patch.object(
            follow,
            "_run_reset_sequence",
        ) as reset_mock:
            stats = follow._follow_loop(vision, robot, duration_s=2.0, debug_mode=True)

        self.assertTrue(stats.get("debug_mode_terminated", False))
        self.assertEqual(stats.get("last_action"), "DEBUG_STEP1_TERMINATE")
        self.assertGreaterEqual(stats["win_count"], 1)
        step2_mock.assert_not_called()
        reset_mock.assert_not_called()
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_backs_up_from_signed_distance_error_when_too_close(self):
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
        self.assertEqual(
            robot.commands,
            [("b", 103, follow._distance_correction_duration_ms(-(follow.DIST_TOL_MM + 25.0)))],
        )
        self.assertEqual(stats["act_counts"], {"BCK": 1})

    def test_approved_straight_drive_pwm_matches_uno_effective_score_one_floor(self):
        self.assertEqual(follow._speed_pwm("b", follow.SPEED_SCORE), 103)
        self.assertEqual(follow._approved_straight_drive_pwm("b"), 104)

    def test_follow_loop_backs_up_without_mast_attachment_when_too_close(self):
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

        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(
            robot.commands,
            [("b", 103, follow._distance_correction_duration_ms(-(follow.DIST_TOL_MM + 25.0)))],
        )
        self.assertEqual(stats["act_counts"], {"BCK": 1})

    def test_too_close_escape_clamps_configured_pwm_to_approved_top_speed(self):
        with mock.patch.object(
            follow,
            "_follow_motion_config",
            return_value={
                "too_close_escape": {
                    "pwm": follow._approved_straight_drive_pwm("b") + 80,
                    "pulse_ms": 400,
                    "attach_mast": False,
                }
            },
        ):
            policy = follow._too_close_escape_policy()

        self.assertEqual(policy["pwm"], follow._approved_straight_drive_pwm("b"))
        self.assertEqual(policy["pulse_ms"], 400)

    def test_follow_act_duration_allows_two_second_far_distance_acts(self):
        self.assertEqual(follow._bounded_act_duration_ms(2500), 2000)

    def test_post_action_wait_respects_timed_turn_duration(self):
        wait_s = follow._post_action_wait_s(
            {"kind": "turn", "action": "TURN_L"},
            {"duration_ms": 255},
        )

        self.assertAlmostEqual(wait_s, 0.655, places=3)
        self.assertGreater(wait_s, follow.LOOP_S)

    def test_post_action_wait_keeps_distance_creep_settle(self):
        wait_s = follow._post_action_wait_s(
            {"kind": "drive", "action": "FWD", "distance_creep": True},
            {"duration_ms": 300},
        )

        self.assertAlmostEqual(wait_s, 0.34, places=3)

    def test_empty_post_turn_wait_adds_observation_settle(self):
        wait_s = follow._post_action_wait_s(
            {"kind": "turn", "action": "TURN_R", "duration_ms": 200},
            {"duration_ms": 200},
        )

        self.assertAlmostEqual(wait_s, 0.6, places=3)

    def test_attached_mast_action_keeps_its_own_duration(self):
        actions = follow._actions_with_mast(
            [{"target": "l", "action": "b", "pwm": 103}],
            "d",
            mast_pwm=255,
            mast_duration_ms=300,
        )

        self.assertEqual(actions[-1], {"target": "m", "action": "u", "pwm": 255, "duration_ms": 300})

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

    def test_follow_loop_waits_for_timed_turn_before_next_sample(self):
        class _TimedTurnRobot(_FakeRobot):
            def send_custom_actions_pwm(self, cmd, actions, duration_ms=None):
                super().send_custom_actions_pwm(cmd, actions, duration_ms)
                return {
                    "cmd_sent": cmd,
                    "duration_ms": 255,
                    "actions": list(actions),
                }

        robot = _TimedTurnRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                185.6,
                -11.2,
                86.0,
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
            stats = follow._follow_loop(vision, robot, duration_s=0.24)

        self.assertEqual(len(robot.custom_commands), 1)
        self.assertEqual(stats["sample_count"], 1)
        self.assertEqual(stats["act_counts"], {"BIAS_L_MEDIUM": 1})

    def test_follow_loop_hard_stops_after_six_no_change_acts(self):
        robot = _FakeRobot()
        vision = _FakeVision((True, 0.0, 500.0, 0.0, 85.0, follow.Y_TARGET_MM, False, False))
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=13.0)

        self.assertTrue(stats["stall_guard_triggered"])
        self.assertEqual(stats["stall_guard_reason"], "act_stall_no_observed_change:FWD")
        self.assertEqual(stats["no_observed_change_streak_count"], 6)
        self.assertEqual(stats["no_observed_after_act_counts"], {"FWD": 6})
        self.assertEqual(len(robot.commands), 6)
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_gives_mast_stall_guard_y_specific_tries_or_duration(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM,
                follow._x_target_mm(),
                85.0,
                follow.Y_TARGET_MM - 20.0,
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
            stats = follow._follow_loop(vision, robot, duration_s=10.0)

        self.assertTrue(stats["stall_guard_triggered"])
        self.assertEqual(stats["stall_guard_reason"], "act_stall_no_observed_change:MAST_U")
        detail = stats["stall_guard_detail"]
        cfg = follow._act_stall_guard_config()
        self.assertEqual(detail["max_no_change_tries"], cfg["y_max_no_change_tries"])
        self.assertEqual(detail["max_no_change_duration_ms"], cfg["y_max_no_change_duration_ms"])
        self.assertGreaterEqual(detail["streak_duration_ms"], cfg["y_max_no_change_duration_ms"])
        self.assertGreater(int(stats["no_observed_change_streak_count"]), 6)

    def test_follow_loop_counts_win_then_runs_reset_and_tracks_x_after_reset(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, follow._x_target_mm(), 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, follow._x_target_mm(), 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, follow._x_target_mm(), 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()
        def _reset_result_for_active_profile(_vision, _robot):
            cfg = follow._reset_motion_config()["reverse_turn"]
            return {
                "success": True,
                "phase": "reverse_turn",
                "reason": "x_offset_confirmed",
                "turn_cmd": "l",
                "mast_up_sent": True,
                "reading": _configured_reset_target_reading(x_mm=-cfg["target_abs_x_mm"]),
                "target_met": True,
            }
        step2_result = {
            "success": True,
            "target_met": True,
            "reason": "step2_targets_scored",
            "reading": {"confident": True, "dist_mm": 45.0, "x_mm": 0.0, "y_mm": -6.0},
        }
        step3_result = {
            "success": True,
            "target_met": True,
            "holding": True,
            "reason": "step3_targets_scored",
            "reading": {"confident": True, "dist_mm": 45.0, "x_mm": 0.0, "y_mm": -3.5},
        }

        with mock.patch.object(follow, "_run_step2_seat_sequence", return_value=step2_result), mock.patch.object(
            follow,
            "_run_step3_lift_sequence",
            return_value=step3_result,
        ), mock.patch.object(follow, "_run_reset_sequence", side_effect=_reset_result_for_active_profile) as reset_mock, mock.patch.object(
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
        self.assertEqual(stats["confident_sample_count"], 3)
        self.assertEqual(stats["not_confident_count"], 0)
        self.assertEqual(stats["follow_attempt_count"], 0)
        self.assertEqual(stats["act_counts"], {"STEP2_SEAT": 1, "RESET_BACK_TURN_L": 1, "RESET_MAST_U": 1})
        self.assertEqual(stats["miss_reasons"], {})
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

    def test_follow_loop_resets_only_after_honest_step3_target_hit(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()
        step2_result = {
            "success": True,
            "target_met": True,
            "reason": "step2_targets_scored",
            "reading": {"confident": True, "dist_mm": 60.0, "x_mm": 0.0, "y_mm": 3.0},
        }
        step3_result = {
            "success": True,
            "target_met": False,
            "holding": True,
            "reason": "step3_already_past_y_target",
            "reading": {"confident": True, "dist_mm": 60.0, "x_mm": 0.0, "y_mm": 7.0},
        }

        with mock.patch.object(follow, "_run_step2_seat_sequence", return_value=step2_result), mock.patch.object(
            follow,
            "_run_step3_lift_sequence",
            return_value=step3_result,
        ), mock.patch.object(follow, "_run_reset_sequence") as reset_mock, mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.26)

        reset_mock.assert_not_called()
        self.assertEqual(stats["win_count"], 1)
        self.assertEqual(stats["step2_target_met_count"], 1)
        self.assertEqual(stats["reset_attempt_count"], 0)

    def test_follow_loop_parks_without_reset_after_frozen_xz_step2_hit(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()
        step2_result = {
            "success": True,
            "target_met": True,
            "reason": "step2_targets_scored",
            "reading": {"confident": True, "dist_mm": 60.0, "x_mm": 0.0, "y_mm": 3.0, "xz_frozen": True},
            "precision_counts": {"xz_freeze": 1, "mast_d": 2, "fwd": 0, "bck": 0},
        }
        step3_result = {
            "success": True,
            "target_met": True,
            "holding": True,
            "reason": "step3_targets_scored",
            "reading": {"confident": True, "dist_mm": 60.0, "x_mm": 0.0, "y_mm": 3.0},
        }

        with mock.patch.object(follow, "_run_step2_seat_sequence", return_value=step2_result), mock.patch.object(
            follow,
            "_run_step3_lift_sequence",
            return_value=step3_result,
        ), mock.patch.object(follow, "_run_reset_sequence") as reset_mock, mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.26)

        reset_mock.assert_not_called()
        self.assertEqual(stats["win_count"], 1)
        self.assertEqual(stats["step2_target_met_count"], 1)
        self.assertEqual(stats["reset_attempt_count"], 0)
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_resets_after_step3_no_visibility_fallback(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()
        step2_result = {
            "success": True,
            "target_met": True,
            "reason": "step2_targets_scored",
            "reading": {"confident": True, "dist_mm": 60.0, "x_mm": 0.0, "y_mm": -7.0},
        }
        step3_result = {
            "success": True,
            "target_met": False,
            "fallback_reset_ok": True,
            "holding": False,
            "reason": "step3_no_visibility_fallback_lift",
            "fallback_cmd": "u",
            "fallback_duration_ms": 3000,
            "reading": {"confident": False},
        }
        reset_result = {
            "success": True,
            "phase": "reverse_turn",
            "reason": "fallback_reset_done",
            "turn_cmd": "r",
            "mast_up_sent": True,
            "reading": _configured_reset_target_reading(),
            "target_met": True,
        }

        with mock.patch.object(follow, "_run_step2_seat_sequence", return_value=step2_result), mock.patch.object(
            follow,
            "_run_step3_lift_sequence",
            return_value=step3_result,
        ), mock.patch.object(follow, "_run_reset_sequence", return_value=reset_result) as reset_mock, mock.patch.object(
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
        self.assertEqual(stats["step2_target_met_count"], 1)
        self.assertEqual(stats["reset_attempt_count"], 1)

    def test_follow_loop_resets_after_step3_target_hit(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
            ]
        )
        fake_clock = _FakeClock()
        step2_result = {
            "success": True,
            "target_met": True,
            "reason": "step2_targets_scored",
            "reading": {"confident": True, "dist_mm": 60.0, "x_mm": 0.0, "y_mm": 3.0},
        }
        step3_result = {
            "success": True,
            "target_met": True,
            "holding": True,
            "reason": "step3_targets_scored",
            "reading": {"confident": True, "dist_mm": 60.0, "x_mm": 0.0, "y_mm": 3.0},
        }
        reset_reading = _configured_reset_target_reading(
            x_mm=follow._reset_motion_config()["reverse_turn"]["target_abs_x_mm"]
        )
        reset_result = {
            "success": True,
            "phase": "reverse_turn",
            "reason": "target_hit",
            "turn_cmd": "r",
            "mast_up_sent": True,
            "reading": reset_reading,
            "target_met": True,
        }

        with mock.patch.object(follow, "_run_step2_seat_sequence", return_value=step2_result), mock.patch.object(
            follow,
            "_run_step3_lift_sequence",
            return_value=step3_result,
        ), mock.patch.object(follow, "_run_reset_sequence", return_value=reset_result) as reset_mock, mock.patch.object(
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
        self.assertEqual(stats["step2_target_met_count"], 1)
        self.assertEqual(stats["reset_attempt_count"], 1)
        self.assertEqual(stats["reset_target_met_count"], 1)

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
        self.assertGreaterEqual(stats["sample_count"], 2)
        self.assertGreaterEqual(stats["confident_sample_count"], 2)
        self.assertEqual(stats["follow_attempt_count"], 0)
        self.assertEqual(stats["miss_reasons"], {"happy_not_stopped": 1})
        self.assertEqual(stats["win_target_closeness_pct"], [])
        self.assertEqual(stats["closest_non_win"]["action"], "HAPPY_REJECT")
        self.assertEqual(robot.commands, [])
        self.assertGreaterEqual(robot.stops, 1)

    def test_debug_mode_continues_observing_after_live_happy_reject(self):
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
        win_cfg = {
            "settle_s": 0.0,
            "confirm_frames": 2,
            "single_win_allowance_s": 0.1,
            "min_axis_closeness_pct": 0.0,
            "min_confidence_pct": 75.0,
        }

        with mock.patch.object(follow, "_win_confirmation_config", return_value=win_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.05, debug_mode=True)

        self.assertEqual(robot.commands, [])
        self.assertEqual(stats["win_count"], 0)
        self.assertFalse(stats.get("debug_mode_terminated", False))
        self.assertEqual(stats["last_action"], "HAPPY_REJECT")
        self.assertGreaterEqual(robot.stops, 1)

    def test_park_happy_stops_after_live_happy_reject_instead_of_backing_up(self):
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
        win_cfg = {
            "settle_s": 0.0,
            "confirm_frames": 2,
            "single_win_allowance_s": 0.1,
            "min_axis_closeness_pct": 0.0,
            "min_confidence_pct": 75.0,
        }

        with mock.patch.object(follow, "_win_confirmation_config", return_value=win_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=1.0, stop_after_win=True)

        self.assertEqual(robot.commands, [])
        self.assertEqual(stats["win_count"], 0)
        self.assertTrue(stats["happy_reject_terminated"])
        self.assertFalse(stats.get("debug_mode_terminated", False))
        self.assertEqual(stats["last_action"], "PARK_STEP1_HAPPY_REJECT_TERMINATE")
        self.assertGreaterEqual(robot.stops, 1)

    def test_park_happy_stops_for_confirmation_instead_of_reversing_after_dist_pingpong(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM + 12.0,
                    follow._x_target_mm(),
                    88.0,
                    follow.Y_TARGET_MM,
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM - 10.0,
                    follow._x_target_mm(),
                    88.0,
                    follow.Y_TARGET_MM,
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()
        win_cfg = {
            "settle_s": 0.0,
            "confirm_frames": 2,
            "single_win_allowance_s": 0.1,
            "min_axis_closeness_pct": 0.0,
            "min_confidence_pct": 75.0,
        }

        with mock.patch.object(follow, "_win_confirmation_config", return_value=win_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=1.0, stop_after_win=True)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        sent_cmd, _actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(sent_cmd, "f")
        self.assertEqual(duration_ms, 200)
        self.assertTrue(stats["dist_pingpong_guard_terminated"])
        self.assertEqual(stats["last_action"], "PARK_DIST_PINGPONG_GUARD_TERMINATE")
        self.assertGreaterEqual(robot.stops, 1)

    def test_follow_loop_waits_within_single_win_allowance_instead_of_backing_up(self):
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
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM,
                    0.0,
                    88.0,
                    follow.Y_TARGET_MM,
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), mock.patch.object(
            follow,
            "_run_step2_seat_sequence",
            return_value=step2_result,
        ) as step2_mock:
            stats = follow._follow_loop(vision, robot, duration_s=0.2, debug_mode=True)

        self.assertEqual(robot.commands, [])
        self.assertEqual(stats["win_count"], 1)
        self.assertTrue(stats.get("debug_mode_terminated", False))
        self.assertEqual(stats.get("last_action"), "DEBUG_STEP1_TERMINATE")
        step2_mock.assert_not_called()

    def test_follow_loop_accepts_nonconsecutive_stopped_happy_hits_within_allowance(self):
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
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 12.0,
                    0.0,
                    88.0,
                    follow.Y_TARGET_MM,
                    False,
                    False,
                ),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (
                    True,
                    0.0,
                    follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 12.0,
                    0.0,
                    88.0,
                    follow.Y_TARGET_MM,
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()
        step2_result = {
            "success": True,
            "target_met": False,
            "reason": "step2_step_timeout",
            "reading": {"confident": True, "dist_mm": 80.0, "x_mm": 0.0, "y_mm": -10.0},
        }

        with mock.patch.object(follow.time, "monotonic", side_effect=fake_clock.monotonic), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ), mock.patch.object(
            follow,
            "_temporal_filter_brick_reading",
            side_effect=lambda _vision, reading: reading,
        ), mock.patch.object(
            follow,
            "_run_step2_seat_sequence",
            return_value=step2_result,
        ) as step2_mock:
            stats = follow._follow_loop(vision, robot, duration_s=1.0, debug_mode=True)

        self.assertEqual(robot.commands, [])
        self.assertEqual(stats["win_count"], 1)
        self.assertTrue(stats.get("debug_mode_terminated", False))
        self.assertEqual(stats.get("last_action"), "DEBUG_STEP1_TERMINATE")
        step2_mock.assert_not_called()

    def test_follow_plan_requires_y_margin_before_claiming_happy(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM,
                "x_mm": 0.0,
                "y_mm": follow.Y_TARGET_MM + follow.Y_TOL_MM + 0.1,
            }
        )

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "d")
        self.assertEqual(plan["reason"], "final_y")
        self.assertEqual(plan["pwm"], 255)
        self.assertEqual(plan["duration_ms"], 300)

    def test_follow_plan_finishes_y_before_small_dist_backup(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM - 8.0,
                "x_mm": follow._x_target_mm(),
                "y_mm": follow.Y_TARGET_MM + follow.Y_TOL_MM + 0.1,
            }
        )

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "d")
        self.assertEqual(plan["reason"], "final_y")
        self.assertEqual(plan["pwm"], 255)
        self.assertEqual(plan["duration_ms"], 300)

    def test_follow_loop_uses_stronger_finish_mast_nudge(self):
        robot = _FakeRobot()
        vision = _FakeVision(
            (
                True,
                0.0,
                follow.TARGET_DIST_MM,
                0.0,
                88.0,
                follow.Y_TARGET_MM + follow.Y_TOL_MM + 0.1,
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
        self.assertEqual(robot.commands, [("d", 255, 300)])
        self.assertEqual(stats["act_counts"], {"MAST_D": 1})

    def test_follow_plan_finishes_y_before_near_target_x_turn(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + (follow.DIST_TOL_MM / 2.0),
                "x_mm": follow._x_target_mm() + 2.0,
                "y_mm": follow.Y_TARGET_MM + follow.Y_TOL_MM + 0.1,
            }
        )

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "d")
        self.assertEqual(plan["reason"], "final_y")

    def test_follow_plan_finishes_y_before_slightly_too_far_creep(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + (follow.DIST_TOL_MM / 2.0),
                "x_mm": follow._x_target_mm(),
                "y_mm": follow.Y_TARGET_MM + follow.Y_TOL_MM + 0.1,
            }
        )

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "d")
        self.assertEqual(plan["reason"], "final_y")

    def test_follow_plan_prioritizes_dist_over_y_until_dist_is_happy(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + follow.DIST_TOL_MM + 30.0,
                "x_mm": follow._x_target_mm(),
                "y_mm": follow.Y_TARGET_MM - 20.0,
            }
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["reason"], "dist_only_creep")

    def test_follow_plan_allows_protective_y_when_brick_below_camera(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 220.0,
                "x_mm": follow._x_target_mm(),
                "y_mm": follow.Y_TARGET_MM + 20.0,
                "brick_below": True,
            }
        )

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "d")
        self.assertEqual(plan["reason"], "protect_lower_edge")

    def test_follow_plan_vetoes_protective_y_when_close_even_if_brick_below_camera(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 156.9,
                "x_mm": follow._x_target_mm(),
                "y_mm": follow.Y_TARGET_MM + 20.0,
                "brick_below": True,
            }
        )

        self.assertNotEqual(plan.get("reason"), "protect_lower_edge")

    def test_follow_plan_does_not_protect_y_from_y_value_alone(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 220.0,
                "x_mm": follow._x_target_mm(),
                "y_mm": follow.Y_TARGET_MM + 40.0,
                "brick_below": False,
            }
        )

        self.assertNotEqual(plan.get("reason"), "protect_lower_edge")

    def test_follow_plan_still_backs_up_when_distance_is_too_close_despite_y_gap(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 1.0,
                "x_mm": follow._x_target_mm(),
                "y_mm": follow.Y_TARGET_MM + follow.Y_TOL_MM + 0.1,
            }
        )

        self.assertEqual(plan["kind"], "drive_nudge")
        self.assertEqual(plan["cmd"], "b")
        self.assertEqual(plan["reason"], "dist_micro_nudge")

    def test_follow_plan_backs_up_when_dist_overshoots_happy_margin(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 12.0,
                "x_mm": 0.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "drive")
        self.assertEqual(plan["cmd"], "b")
        self.assertEqual(plan["reason"], "dist_only_creep")
        self.assertEqual(plan["duration_ms"], follow._distance_correction_duration_ms(plan["dist_err"]))

    def test_follow_plan_uses_short_backup_for_small_dist_overshoot(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 1.0,
                "x_mm": 0.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "drive_nudge")
        self.assertEqual(plan["cmd"], "b")
        self.assertLess(plan["duration_ms"], follow._follow_dist_approach_policy()["max_forward_pulse_ms"])

    def test_follow_plan_prioritizes_x_with_backward_turn_when_too_close_and_x_wide(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM - follow.DIST_TOL_MM - 1.0,
                "x_mm": follow.X_TOL_MM + 4.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["drive_mode"], "backward")
        self.assertEqual(plan["strength"], "adaptive")
        self.assertEqual(plan["reason"], "x_first_before_dist")

    def test_empty_align_turn_can_carry_y_correction(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow._dist_target_mm() - follow._dist_tol_mm() - 1.0,
                "x_mm": follow._x_target_mm() + follow._x_tol_mm() + 4.0,
                "y_mm": follow._y_win_target_mm() - 5.0,
            }
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["cmd"], "r")
        self.assertEqual(plan["mast_cmd"], "u")
        self.assertEqual(plan["mast_reason"], "final_y")

    def test_step1_timeout_allows_bounded_corrective_y_grace(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (
                    True,
                    0.0,
                    follow._dist_target_mm() - follow._dist_tol_mm() - 1.0,
                    follow._x_target_mm() - follow._x_tol_mm() - 4.0,
                    88.0,
                    follow._y_win_target_mm(),
                    False,
                    False,
                ),
                (
                    True,
                    0.0,
                    follow._dist_target_mm() - follow._dist_tol_mm() - 1.0,
                    follow._x_target_mm() - follow._x_tol_mm() - 4.0,
                    88.0,
                    follow._y_win_target_mm() + follow._y_win_tol_mm() + 0.8,
                    False,
                    False,
                ),
            ]
        )
        fake_clock = _FakeClock()
        win_cfg = dict(follow._win_confirmation_config())
        win_cfg["single_win_allowance_s"] = 0.001
        win_cfg["timeout_y_correction_grace_acts"] = 1

        with mock.patch.object(follow, "_win_confirmation_config", return_value=win_cfg), mock.patch.object(
            follow.time,
            "monotonic",
            side_effect=fake_clock.monotonic,
        ), mock.patch.object(
            follow.time,
            "sleep",
            side_effect=fake_clock.sleep,
        ):
            stats = follow._follow_loop(vision, robot, duration_s=0.7)

        self.assertGreaterEqual(len(robot.custom_commands), 2)
        _cmd, actions, _duration_ms = robot.custom_commands[1]
        self.assertTrue(any(action.get("target") == "m" and action.get("action") == "d" for action in actions))
        self.assertEqual(stats["step1_timeout_y_correction_grace_count"], 1)
        self.assertFalse(stats.get("step1_attempt_timeout", False))
        self.assertEqual(stats["sent_act_counts"].get("TURN_L_MAST_D"), 1)

    def test_follow_plan_biases_forward_when_dist_and_x_are_both_open(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + 60.0,
                "x_mm": follow.X_TOL_MM + 1.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["turn_cmd"], "r")
        self.assertEqual(plan["reason"], "x_polish_while_creeping_dist")
        self.assertNotIn("mast_cmd", plan)

    def test_empty_step1_follow_uses_short_slow_adaptive_curve_budget(self):
        self.assertLessEqual(follow._max_act_ms(), 300)
        self.assertLessEqual(follow._follow_dist_approach_policy()["max_forward_pulse_ms"], 300)
        self.assertLessEqual(follow._too_close_escape_policy()["pulse_ms"], 220)

        gentle_curve = follow._turn_bias_curve_for_drive_mode("forward", "gentle")
        strong_curve = follow._turn_bias_curve_for_drive_mode("forward", "strong")

        self.assertEqual(gentle_curve["inner_pwm"], 103)
        self.assertLessEqual(gentle_curve["outer_pwm"], 112)
        self.assertLessEqual(strong_curve["outer_pwm"], 125)

    def test_three_gap_step1_plan_avoids_bad_forward_left_bias(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 192.3,
                "x_mm": -7.4,
                "y_mm": -6.6,
            }
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["cmd"], "l")
        self.assertEqual(plan["drive_mode"], "backward")
        self.assertEqual(plan["strength"], "adaptive")
        self.assertLessEqual(plan["duration_ms"], 300)
        self.assertEqual(plan["reason"], "forward_left_bias_untrusted_x_first")

    def test_step1_finishes_y_only_when_dist_happy_and_x_inside_finish_deadband(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow._dist_target_mm() - 3.0,
                "x_mm": follow._x_target_mm() + 5.2,
                "y_mm": follow._y_win_target_mm() + 13.5,
            }
        )

        self.assertEqual(plan["kind"], "mast")
        self.assertEqual(plan["cmd"], "d")
        self.assertEqual(plan["reason"], "final_y")

    def test_y_mast_duration_scales_down_near_happy_target(self):
        y_cfg = follow._follow_y_axis_config()

        far_duration = follow._adaptive_y_mast_duration_ms(6.0, y_cfg, near_end=False)
        finish_duration = follow._adaptive_y_mast_duration_ms(1.0, y_cfg, near_end=True)

        self.assertGreaterEqual(far_duration, 35)
        self.assertLessEqual(far_duration, 80)
        self.assertGreaterEqual(finish_duration, 10)
        self.assertLessEqual(finish_duration, 25)

    def test_follow_plan_keeps_current_stuck_pose_moving_instead_of_turn_stalling(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": 111.7,
                "x_mm": 8.8,
                "y_mm": -14.7,
            }
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["turn_cmd"], "r")
        self.assertEqual(plan["reason"], "x_polish_while_creeping_dist")
        self.assertEqual(plan["mast_cmd"], "d")
        self.assertEqual(plan["mast_reason"], "approach_high_y")

    def test_follow_plan_creeps_dist_with_tiny_polish_x_gap(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + 60.0,
                "x_mm": 5.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["reason"], "x_polish_while_creeping_dist")
        self.assertEqual(plan["duration_ms"], follow._distance_creep_duration_ms(60.0))

    def test_follow_plan_uses_x_first_turn_when_dist_small_and_x_wide(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + 8.0,
                "x_mm": follow.X_TOL_MM + 5.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["drive_mode"], "backward")
        self.assertEqual(plan["strength"], "adaptive")
        self.assertEqual(plan["reason"], "x_first_before_dist")

    def test_follow_plan_creeps_forward_near_target_when_dist_still_too_far_and_x_needs_polish(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "dist_mm": follow._dist_target_mm() + follow._dist_tol_mm() + 7.0,
                    "x_mm": follow._x_target_mm() + follow._x_tol_mm() + 1.4,
                    "y_mm": follow._y_win_target_mm(),
                }
            )
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(plan["kind"], "drive_bias")
        self.assertEqual(plan["cmd"], "f")
        self.assertEqual(plan["reason"], "x_polish_while_creeping_dist")
        self.assertEqual(plan["duration_ms"], 200)

    def test_follow_plan_uses_single_tread_nudge_for_tiny_dist_gap_when_x_is_happy(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "dist_mm": follow._dist_target_mm() - follow._dist_tol_mm() - 5.5,
                    "x_mm": follow._x_target_mm() - 0.4,
                    "y_mm": follow._y_win_target_mm(),
                }
            )
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(plan["kind"], "drive_nudge")
        self.assertEqual(plan["cmd"], "b")
        self.assertEqual(plan["turn_cmd"], "l")
        self.assertEqual(plan["reason"], "dist_micro_nudge")
        self.assertEqual(plan["duration_ms"], 200)
        self.assertEqual(plan["pwm"], 103)

    def test_execute_single_tread_nudge_sends_one_drive_tread_and_one_zero_tread(self):
        robot = _FakeRobot()
        reading = {
            "visible": True,
            "confident": True,
            "conf": 88.0,
            "dist_mm": follow._dist_target_mm() - follow._dist_tol_mm() - 5.5,
            "x_mm": follow._x_target_mm() - 0.4,
            "y_mm": follow._y_win_target_mm(),
        }
        plan = {
            "kind": "drive_nudge",
            "cmd": "b",
            "turn_cmd": "l",
            "pwm": 103,
            "duration_ms": 200,
            "dist_err": -9.5,
            "x_err": -0.4,
            "distance_creep": True,
        }

        follow._execute_follow_action(robot, plan, reading)

        self.assertEqual(robot.commands, [])
        self.assertEqual(len(robot.custom_commands), 1)
        sent_cmd, actions, duration_ms = robot.custom_commands[0]
        self.assertEqual(sent_cmd, "b")
        self.assertEqual(duration_ms, 200)
        self.assertEqual(actions[0]["pwm"], 0)
        self.assertEqual(actions[1]["pwm"], 103)

    def test_follow_plan_holds_when_all_axes_are_inside_happy_box(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + (follow.DIST_TOL_MM / 2.0),
                "x_mm": follow._x_target_mm() + 1.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "hold")
        self.assertEqual(plan["action"], "HAPPY")

    def test_follow_plan_allows_forward_x_turn_only_when_distance_gap_is_large(self):
        policy = follow._follow_x_dist_curve_policy()
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + policy["large_dist_gap_mm"] + 5.0,
                "x_mm": follow._x_target_mm() + 90.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["drive_mode"], "forward")
        self.assertEqual(plan["reason"], "x_first_before_dist")

    def test_empty_follow_plan_uses_short_x_only_turn_for_large_x_gap(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow._dist_target_mm() + 55.0,
                "x_mm": follow._x_target_mm() + 42.0,
                "y_mm": follow._y_win_target_mm(),
            }
        )

        self.assertEqual(plan["kind"], "turn")
        self.assertEqual(plan["reason"], "x_first_before_dist")
        self.assertEqual(plan["cmd"], "r")
        self.assertEqual(plan["duration_ms"], follow.PULSE_MS)

    def test_follow_plan_holds_when_dist_x_and_y_are_happy(self):
        plan = follow._follow_action_plan(
            {
                "visible": True,
                "confident": True,
                "dist_mm": follow.TARGET_DIST_MM + (follow.DIST_TOL_MM / 2.0),
                "x_mm": 0.0,
                "y_mm": follow.Y_TARGET_MM,
            }
        )

        self.assertEqual(plan["kind"], "hold")
        self.assertEqual(plan["action"], "HAPPY")

    def test_dist_aggressiveness_is_tuned_for_smaller_creep_steps(self):
        policy = follow._follow_dist_approach_policy()

        self.assertAlmostEqual(policy["closure_shots"], 1.0)
        self.assertAlmostEqual(policy["settle_after_act_s"], 0.04)
        self.assertEqual(policy["min_forward_pulse_ms"], 200)
        self.assertEqual(policy["max_forward_pulse_ms"], 2000)
        self.assertEqual(policy["full_forward_gap_mm"], 100.0)
        self.assertEqual(policy["near_target_creep_band_mm"], 30.0)
        self.assertEqual(policy["near_target_min_pulse_ms"], 200)
        self.assertEqual(policy["near_target_max_pulse_ms"], 300)
        self.assertEqual(policy["close_target_creep_band_mm"], 20.0)
        self.assertEqual(policy["close_target_max_pulse_ms"], 200)
        self.assertEqual(policy["very_close_target_creep_band_mm"], 10.0)
        self.assertEqual(policy["very_close_target_max_pulse_ms"], 200)
        self.assertEqual(policy["micro_nudge_abs_dist_err_mm"], 16.0)
        self.assertEqual(policy["micro_nudge_pulse_ms"], 200)
        self.assertEqual(policy["micro_nudge_pwm"], 103)
        self.assertEqual(policy["pingpong_confirm_abs_dist_err_mm"], 20.0)
        self.assertEqual(policy["near_target_forward_veto_mm"], 0.0)

    def test_near_target_dist_creep_scales_slowly_for_forward_and_backward(self):
        tol = follow._dist_tol_mm()
        just_outside = tol + 1.0
        very_close_band = 10.0
        close_band = 20.0
        mid_band = 25.0
        edge_band = 30.0
        far_band = 100.0

        self.assertEqual(follow._distance_correction_duration_ms(very_close_band), 200)
        self.assertLessEqual(follow._distance_correction_duration_ms(close_band), 200)

        self.assertLess(
            follow._distance_correction_duration_ms(just_outside),
            follow._distance_correction_duration_ms(mid_band),
        )
        self.assertLess(
            follow._distance_correction_duration_ms(mid_band),
            follow._distance_correction_duration_ms(edge_band),
        )
        self.assertEqual(follow._distance_correction_duration_ms(edge_band), 300)
        self.assertGreater(follow._distance_correction_duration_ms(far_band), 300)
        self.assertLessEqual(follow._distance_correction_duration_ms(far_band), 2000)
        self.assertEqual(
            follow._distance_correction_duration_ms(-edge_band),
            follow._distance_correction_duration_ms(edge_band),
        )
        self.assertEqual(
            follow._distance_creep_duration_ms(edge_band),
            follow._distance_correction_duration_ms(edge_band),
        )

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
        self.assertIn("| Step 1 Win | 2 | 2 | 55%±15% [######----] | 70%±10% | 40%±20% | 90%±10% | target dist 162.7mm, x 1.0mm, y -19.4mm |", table)
        self.assertIn("| Step 1 Reset | 1/1 | 1/1 | 95%±0% [##########] | 100%±0% | 90%±0% | 80%±0% | dist 250.7mm, |x| 36.0mm, y -5.0mm |", table)
        self.assertIn("| Step 2 Win | 0/0 | 0/0 | N/A [??????????] | N/A | N/A | N/A | N/A |", table)
        self.assertIn("| Movement attempts | 8 |", table)
        self.assertIn("| Planned: TURN_R=5, FWD=3 | 8 |", table)
        self.assertIn("| Sent: none | 0 |", table)
        self.assertIn("| Miss reasons | x_outside=5, too_far=3 |", table)
        self.assertIn("Closest non-win | 88% (dist=95% x=80%, dist_err=+1.0mm x_err=+6.0mm, TURN_R)", table)

    def test_step2_defaults_define_mast_then_visible_precision_settle(self):
        cfg = follow._follow_step2_config()

        self.assertEqual(cfg["seat_mast_cmd"], "d")
        self.assertEqual(cfg["seat_mast_duration_ms"], 200)
        self.assertEqual(cfg["seat_drive_cmd"], "f")
        self.assertEqual(cfg["seat_drive_duration_ms"], 0)
        self.assertTrue(cfg["precision_settle_enabled"])
        self.assertIsNone(cfg["targets"]["x_mm"])
        self.assertAlmostEqual(cfg["targets"]["y_mm"], -36.9)
        self.assertEqual(cfg["targets"]["y_tol_mm"], 4.0)
        self.assertEqual(cfg["precision_mast_small_gap_max_mm"], 5.0)
        self.assertEqual(cfg["precision_mast_small_gap_pwm_scale"], 0.5)
        self.assertEqual(cfg["precision_mast_small_gap_min_pulse_ms"], 150)
        self.assertEqual(cfg["precision_mast_small_gap_max_pulse_ms"], 275)
        self.assertFalse(cfg["freeze_xz_after_xz_target"])
        self.assertEqual(follow._step2_missing_target_keys(cfg), [])

    def test_step2_precision_mast_curve_halves_power_for_small_y_gap(self):
        cfg = {
            "seat_mast_pwm": 200,
            "precision_mast_pulse_ms": 400,
            "precision_mast_small_gap_max_mm": 5.0,
            "precision_mast_small_gap_pwm_scale": 0.5,
            "precision_mast_small_gap_min_pulse_ms": 150,
            "precision_mast_small_gap_max_pulse_ms": 250,
        }

        self.assertEqual(follow._step2_precision_mast_pwm("d", 2.5, cfg), 100)
        self.assertEqual(follow._step2_precision_mast_duration_ms(2.5, cfg), 200)
        self.assertEqual(follow._step2_precision_mast_pwm("d", 5.1, cfg), 200)
        self.assertEqual(follow._step2_precision_mast_duration_ms(5.1, cfg), 400)

    def test_y_mast_duration_can_use_gap_outside_tolerance(self):
        cfg = {
            "finish_mast_pulse_ms": 130,
            "finish_mast_min_pulse_ms": 100,
            "finish_mast_max_pulse_ms": 150,
            "mast_up_mm_per_100ms": 1.0,
            "mast_down_mm_per_100ms": 2.2,
            "mast_mm_per_100ms": 2.0,
            "mast_correction_fraction": 0.75,
        }

        self.assertEqual(
            follow._adaptive_y_mast_duration_ms(8.0, cfg, near_end=True, cmd="u", duration_err=4.0),
            150,
        )
        self.assertEqual(
            follow._adaptive_y_mast_duration_ms(-8.0, cfg, near_end=True, cmd="d", duration_err=4.0),
            136,
        )

    def test_holding_profile_step1_has_y_target_without_lock_on_and_step2_is_y_only(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            y_cfg = follow._follow_y_axis_config()
            cfg = follow._follow_step2_config()
        finally:
            follow._set_game_profile(old_profile)

        self.assertTrue(y_cfg["enabled"])
        self.assertFalse(y_cfg["lock_on_enabled"])
        self.assertAlmostEqual(y_cfg["win_target_mm"], 5.250551859537761)
        self.assertEqual(y_cfg["win_tol_mm"], 2.5)
        self.assertEqual(y_cfg["finish_mast_pwm"], 40)
        self.assertEqual(y_cfg["finish_mast_pulse_ms"], 260)
        self.assertFalse(cfg["freeze_xz_after_xz_target"])
        self.assertFalse(cfg["recovery_creep_enabled"])
        self.assertFalse(cfg["visibility_recovery_creep_enabled"])
        self.assertIsNone(cfg["targets"]["dist_mm"])
        self.assertIsNone(cfg["targets"]["dist_tol_mm"])
        self.assertIsNone(cfg["targets"]["x_mm"])
        self.assertIsNone(cfg["targets"]["x_tol_mm"])
        self.assertAlmostEqual(cfg["targets"]["y_mm"], -10.25129472)
        self.assertEqual(cfg["targets"]["y_tol_mm"], 2.5)
        self.assertEqual(follow._step2_missing_target_keys(cfg), [])

    def test_holding_step1_holds_before_downward_y_trek(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            plan = follow._follow_action_plan(
                {
                    "visible": True,
                    "confident": True,
                    "dist_mm": follow._dist_target_mm(),
                    "x_mm": follow._x_target_mm(),
                    "y_mm": follow._y_win_target_mm(),
                }
            )
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(plan["kind"], "hold")
        self.assertEqual(plan["action"], "HAPPY")

    def test_step3_defaults_define_lift_target_for_empty_profile(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("empty")
            cfg = follow._follow_step4_config()
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(cfg["lift_mast_cmd"], "u")
        self.assertEqual(cfg["lift_mast_pwm"], 255)
        self.assertEqual(cfg["lift_pulse_ms"], 1100)
        self.assertTrue(cfg["no_visibility_fallback_enabled"])
        self.assertEqual(cfg["no_visibility_fallback_mast_cmd"], "u")
        self.assertEqual(cfg["no_visibility_fallback_mast_pwm"], 255)
        self.assertEqual(cfg["no_visibility_fallback_duration_ms"], 3000)
        self.assertAlmostEqual(cfg["targets"]["y_mm"], 1.751941067049)
        self.assertEqual(cfg["targets"]["y_tol_mm"], 0.5)
        self.assertEqual(follow._step3_missing_target_keys(cfg), [])

    def test_empty_step4_lift_uses_physical_up_command(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("empty")
            cfg = follow._follow_step4_config()
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(cfg["lift_mast_cmd"], "u")
        self.assertEqual(cfg["no_visibility_fallback_mast_cmd"], "u")
        self.assertAlmostEqual(cfg["targets"]["y_mm"], 1.751941067049)
        self.assertEqual(cfg["targets"]["y_tol_mm"], 0.5)

    def test_step4_lift_honors_world_model_duration_above_one_second(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 70.0, 0.0, 88.0, 8.0, False, False),
                (True, 0.0, 70.0, 0.0, 88.0, 4.0, False, False),
            ]
        )
        cfg = {
            "lift_mast_cmd": "u",
            "lift_mast_pwm": 255,
            "lift_pulse_ms": 1800,
            "lift_settle_s": 0.0,
            "max_lift_attempts": 1,
            "step_timeout_s": 10.0,
            "initial_lift_before_gate": True,
            "targets": {"y_mm": 1.8, "y_tol_mm": 0.5},
        }

        with mock.patch.object(follow, "_follow_step4_config", return_value=cfg), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step3_lift_sequence(vision, robot)

        self.assertEqual(robot.commands, [("u", 255, 1800)])
        self.assertEqual(result["reason"], "step4_y_target_not_reached")

    def test_step3_no_visibility_lifts_mast_then_allows_reset(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (False,),
                (False,),
            ]
        )
        cfg = {
            "no_visibility_fallback_enabled": True,
            "no_visibility_fallback_mast_cmd": "u",
            "no_visibility_fallback_mast_pwm": 255,
            "no_visibility_fallback_duration_ms": 3000,
            "no_visibility_fallback_settle_s": 0.0,
            "targets": {"y_mm": -3.5, "y_tol_mm": 1.0},
        }

        with mock.patch.object(follow, "_follow_step3_config", return_value=cfg), mock.patch.object(
            follow,
            "_wait_for_visibility_recovery",
            side_effect=lambda _vision, _robot, reading, **_kwargs: reading,
        ), mock.patch.object(
            follow.time,
            "sleep",
        ) as sleep_mock:
            result = follow._run_step3_lift_sequence(vision, robot)

        self.assertEqual(result["reason"], "step3_no_visibility_fallback_lift")
        self.assertEqual(result["trigger_reason"], "brick_not_confident_before_step3")
        self.assertTrue(result["success"])
        self.assertFalse(result["target_met"])
        self.assertTrue(result["fallback_reset_ok"])
        self.assertEqual(robot.commands, [("u", 255, 3000)])
        self.assertGreaterEqual(robot.stops, 1)
        sleep_mock.assert_called_once_with(3.0)

    def test_step2_target_readiness_scores_all_three_axes(self):
        cfg = {
            "targets": {
                "dist_mm": 45.0,
                "dist_tol_mm": 10.0,
                "x_mm": 2.0,
                "x_tol_mm": 4.0,
                "y_mm": -6.0,
                "y_tol_mm": 8.0,
            }
        }
        reading = {"dist_mm": 50.0, "x_mm": 4.0, "y_mm": -10.0}

        ready, reason, closeness = follow._step2_targets_ready(reading, cfg)

        self.assertTrue(ready)
        self.assertEqual(reason, "step2_targets_scored")
        self.assertEqual(closeness["dist_target_closeness_pct"], 50.0)
        self.assertEqual(closeness["x_target_closeness_pct"], 50.0)
        self.assertEqual(closeness["y_target_closeness_pct"], 50.0)
        self.assertEqual(closeness["target_closeness_pct"], 50.0)

    def test_step2_target_readiness_allows_y_only_targets(self):
        cfg = {
            "targets": {
                "dist_mm": None,
                "dist_tol_mm": None,
                "x_mm": None,
                "x_tol_mm": None,
                "y_mm": -7.25,
                "y_tol_mm": 2.5,
            }
        }
        reading = {"dist_mm": 120.0, "x_mm": 40.0, "y_mm": -8.0}

        ready, reason, closeness = follow._step2_targets_ready(reading, cfg)

        self.assertTrue(ready)
        self.assertEqual(reason, "step2_targets_scored")
        self.assertIsNone(closeness["dist_target_closeness_pct"])
        self.assertIsNone(closeness["x_target_closeness_pct"])
        self.assertEqual(closeness["y_target_closeness_pct"], 70.0)
        self.assertEqual(closeness["target_closeness_pct"], 70.0)

    def test_step2_already_happy_stops_and_does_not_apply_mast_nudge(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 150.0, 0.0, 90.0, -36.0, False, False),
                (True, 0.0, 150.0, 0.0, 90.0, -56.5, False, False),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 200,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "precision_settle_s": 0.0,
            "targets": {
                "dist_mm": None,
                "dist_tol_mm": None,
                "x_mm": None,
                "x_tol_mm": None,
                "y_mm": -36.9,
                "y_tol_mm": 4.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow,
            "_y_motion_coast_settle_s",
            return_value=0.0,
        ), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        self.assertTrue(result["success"])
        self.assertFalse(result["target_met"])
        self.assertIn("target_drifted_after_stop", result["reason"])
        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.custom_commands, [])
        self.assertGreaterEqual(robot.stops, 1)

    def test_step2_precision_target_hit_stops_without_second_correction_when_settled_read_drifts(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 150.0, 0.0, 90.0, -20.0, False, False),
                (True, 0.0, 150.0, 0.0, 90.0, -36.0, False, False),
                (True, 0.0, 150.0, 0.0, 90.0, -56.5, False, False),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 0,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "recovery_creep_enabled": False,
            "visibility_recovery_creep_enabled": False,
            "post_precision_recovery_cycles": 0,
            "precision_settle_enabled": True,
            "precision_max_attempts": 4,
            "precision_mast_pulse_ms": 250,
            "precision_settle_s": 0.0,
            "targets": {
                "dist_mm": None,
                "dist_tol_mm": None,
                "x_mm": None,
                "x_tol_mm": None,
                "y_mm": -36.9,
                "y_tol_mm": 4.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow,
            "_y_motion_coast_settle_s",
            return_value=0.0,
        ), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        self.assertTrue(result["success"])
        self.assertFalse(result["target_met"])
        self.assertIn("target_drifted_after_stop", result["reason"])
        self.assertEqual(robot.commands, [("d", 255, 250)])
        self.assertEqual(result["precision_counts"]["mast_d"], 1)
        self.assertEqual(result["precision_counts"]["target_hit_stop"], 1)
        self.assertEqual(result["precision_counts"]["target_hit_confirm_failed"], 1)

    def test_holding_y_only_step2_precision_sends_only_mast(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 120.0, 40.0, 88.0, 0.0, False, False),
                (True, 0.0, 119.0, 42.0, 88.0, -5.0, False, False),
                (True, 0.0, 118.0, 44.0, 88.0, -7.4, False, False),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 0,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "recovery_creep_enabled": False,
            "visibility_recovery_creep_enabled": False,
            "precision_settle_enabled": True,
            "precision_max_attempts": 4,
            "precision_mast_pulse_ms": 250,
            "precision_settle_s": 0.0,
            "freeze_xz_after_xz_target": False,
            "targets": {
                "dist_mm": None,
                "dist_tol_mm": None,
                "x_mm": None,
                "x_tol_mm": None,
                "y_mm": -7.25,
                "y_tol_mm": 2.5,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        self.assertTrue(result["target_met"])
        self.assertEqual(robot.commands, [("d", 128, 245)])
        self.assertEqual(result["precision_counts"]["mast_d"], 1)
        self.assertEqual(result["precision_counts"]["fwd"], 0)
        self.assertEqual(result["precision_counts"]["bck"], 0)
        self.assertNotIn("xz_frozen", result["reading"])

    def test_step2_freezes_xz_and_continues_mast_only_after_lock(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 60.0, 0.0, 88.0, 8.0, False, False),
                (True, 0.0, 100.0, 30.0, 88.0, 5.5, False, False),
                (True, 0.0, 105.0, 35.0, 88.0, 3.2, False, False),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 0,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "precision_settle_enabled": True,
            "precision_max_attempts": 5,
            "precision_mast_pulse_ms": 250,
            "precision_settle_s": 0.0,
            "freeze_xz_after_xz_target": True,
            "targets": {
                "dist_mm": 60.0,
                "dist_tol_mm": 2.0,
                "x_mm": 0.0,
                "x_tol_mm": 9.0,
                "y_mm": 3.0,
                "y_tol_mm": 1.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        self.assertTrue(result["target_met"])
        self.assertTrue(result["reading"]["xz_frozen"])
        self.assertEqual(result["reading"]["dist_mm"], 60.0)
        self.assertEqual(result["reading"]["x_mm"], 0.0)
        self.assertEqual(result["reading"]["raw_dist_mm"], 105.0)
        self.assertEqual(result["reading"]["raw_x_mm"], 35.0)
        self.assertEqual(robot.commands, [("d", 255, 250), ("d", 255, 250)])
        self.assertEqual(result["precision_counts"]["mast_d"], 2)
        self.assertEqual(result["precision_counts"]["fwd"], 0)
        self.assertEqual(result["precision_counts"]["bck"], 0)

    def test_step2_xz_freeze_blocks_visibility_creeps_after_mast_visibility_loss(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, 60.0, 0.0, 88.0, 8.0, False, False),
                (False,),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 1700,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "visibility_recovery_creep_enabled": True,
            "visibility_recovery_creep_pulse_ms": 200,
            "visibility_recovery_creep_max_attempts": 3,
            "visibility_recovery_creep_settle_s": 0.0,
            "precision_settle_enabled": True,
            "precision_max_attempts": 5,
            "precision_mast_pulse_ms": 250,
            "precision_settle_s": 0.0,
            "freeze_xz_after_xz_target": True,
            "targets": {
                "dist_mm": 60.0,
                "dist_tol_mm": 2.0,
                "x_mm": 0.0,
                "x_tol_mm": 9.0,
                "y_mm": 3.0,
                "y_tol_mm": 1.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow,
            "_wait_for_visibility_recovery",
            side_effect=lambda _vision, _robot, reading, **_kwargs: reading,
        ), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        self.assertTrue(result["success"])
        self.assertFalse(result["target_met"])
        self.assertEqual(result["reason"], "step2_unconfirmed_no_final_visibility")
        self.assertTrue(result["reading"]["xz_frozen"])
        self.assertEqual(robot.commands, [("d", 255, 1700)])
        self.assertEqual(result["visibility_recovery_creeps"], 0)

    def test_step2_seat_lowers_then_uses_precision_forward_after_visibility(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, -6.0, False, False),
                (True, 0.0, 45.0, 0.0, 88.0, -6.0, False, False),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 1700,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "seat_drive_duration_ms": 2000,
            "post_seat_pause_s": 0.0,
            "recovery_creep_enabled": True,
            "recovery_creep_pulse_ms": 200,
            "recovery_creep_max_attempts": 0,
            "recovery_creep_settle_s": 0.0,
            "targets": {
                "dist_mm": 45.0,
                "dist_tol_mm": 10.0,
                "x_mm": 0.0,
                "x_tol_mm": 9.0,
                "y_mm": -6.0,
                "y_tol_mm": 9.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow.time,
            "sleep",
        ) as sleep_mock:
            result = follow._run_step2_seat_sequence(vision, robot)

        self.assertTrue(result["success"])
        self.assertTrue(result["target_met"])
        self.assertEqual(robot.custom_commands, [])
        self.assertEqual(robot.commands, [("d", 255, 1700), ("f", 103, 116)])
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [1.7, 0.236])
        self.assertEqual(robot.stops, 2)

    def test_step2_seat_uses_three_tiny_forward_creeps_if_visibility_is_lost_after_mast(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (False,),
                (False,),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 1700,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "seat_drive_duration_ms": 2000,
            "post_seat_pause_s": 0.0,
            "recovery_creep_enabled": True,
            "recovery_creep_pulse_ms": 200,
            "recovery_creep_max_attempts": 0,
            "recovery_creep_settle_s": 0.0,
            "visibility_recovery_creep_enabled": True,
            "visibility_recovery_creep_pulse_ms": 200,
            "visibility_recovery_creep_max_attempts": 3,
            "visibility_recovery_creep_settle_s": 0.0,
            "targets": {
                "dist_mm": 45.0,
                "dist_tol_mm": 10.0,
                "x_mm": 0.0,
                "x_tol_mm": 9.0,
                "y_mm": -6.0,
                "y_tol_mm": 9.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        self.assertTrue(result["success"])
        self.assertFalse(result["target_met"])
        self.assertEqual(result["reason"], "step2_unconfirmed_no_final_visibility")
        self.assertEqual(robot.commands, [("d", 255, 1700), ("f", 103, 200), ("f", 103, 200), ("f", 103, 200)])
        self.assertEqual(result["visibility_recovery_creeps"], 3)

        stats = follow._new_game_stats()
        follow._record_step2_stats(stats, result)
        self.assertEqual(stats["step2_confirmed_win_count"], 0)
        self.assertEqual(stats["step2_unconfirmed_win_count"], 1)
        self.assertEqual(stats["step2_visibility_recovery_creep_count"], 3)
        self.assertEqual(stats["act_counts"]["STEP2_VISIBILITY_CREEP_FWD"], 3)

    def test_step2_creeps_forward_when_visible_and_short_of_dist_target(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, follow.Y_TARGET_MM, False, False),
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, -6.0, False, False),
                (True, 0.0, 57.0, 0.0, 88.0, -6.0, False, False),
                (True, 0.0, 40.0, 0.0, 88.0, -6.0, False, False),
                (True, 0.0, 25.0, 0.0, 88.0, -6.0, False, False),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 1700,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "seat_drive_duration_ms": 2000,
            "post_seat_pause_s": 0.0,
            "recovery_creep_enabled": True,
            "recovery_creep_pulse_ms": 200,
            "recovery_creep_max_attempts": 6,
            "recovery_creep_settle_s": 0.0,
            "targets": {
                "dist_mm": 24.8,
                "dist_tol_mm": 2.0,
                "x_mm": 0.0,
                "x_tol_mm": 9.0,
                "y_mm": -6.0,
                "y_tol_mm": 9.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        drive_pwm = follow._clamp_to_approved_straight_drive_pwm("f", 103)
        self.assertTrue(result["target_met"])
        self.assertEqual(result["creep_attempts"], 3)
        self.assertEqual(
            robot.commands,
            [("d", 255, 1700), ("f", drive_pwm, 200), ("f", drive_pwm, 200), ("f", drive_pwm, 133)],
        )

        stats = follow._new_game_stats()
        follow._record_step2_stats(stats, result)
        self.assertEqual(stats["step2_creep_attempt_count"], 3)
        self.assertEqual(stats["act_counts"]["STEP2_CREEP_FWD"], 3)

    def test_step2_precision_settle_backs_up_when_too_close(self):
        robot = _FakeRobot()
        vision = _SequenceVision(
            [
                (True, 0.0, follow.TARGET_DIST_MM, 0.0, 88.0, -6.0, False, False),
                (True, 0.0, 20.0, 0.0, 88.0, -6.0, False, False),
                (True, 0.0, 25.0, 0.0, 88.0, -6.0, False, False),
            ]
        )
        cfg = {
            "seat_mast_cmd": "d",
            "seat_mast_pwm": 255,
            "seat_mast_duration_ms": 0,
            "seat_drive_cmd": "f",
            "seat_drive_pwm": 103,
            "seat_drive_duration_ms": 2000,
            "post_seat_pause_s": 0.0,
            "precision_settle_enabled": True,
            "precision_drive_min_pulse_ms": 80,
            "precision_drive_max_pulse_ms": 200,
            "precision_settle_s": 0.0,
            "targets": {
                "dist_mm": 24.8,
                "dist_tol_mm": 2.0,
                "x_mm": 0.0,
                "x_tol_mm": 9.0,
                "y_mm": -6.0,
                "y_tol_mm": 9.0,
            },
        }

        with mock.patch.object(follow, "_follow_step2_config", return_value=cfg), mock.patch.object(
            follow.time,
            "sleep",
        ):
            result = follow._run_step2_seat_sequence(vision, robot)

        drive_pwm = follow._clamp_to_approved_straight_drive_pwm("f", 103)
        back_pwm = follow._clamp_to_approved_straight_drive_pwm("b", 103)
        self.assertTrue(result["target_met"])
        self.assertEqual(robot.commands, [("f", drive_pwm, 200), ("b", back_pwm, 45)])
        self.assertEqual(result["precision_counts"]["bck"], 1)

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
            row["pwm"] = 0 if int(row.get("pwm") or 0) <= 0 else max(follow._pwm_floor_for_cmd(row.get("action")), scaled_pwm)
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
            if int(action["pwm"]) <= 0:
                self.assertEqual(action["pwm"], 0)
            else:
                self.assertGreaterEqual(action["pwm"], follow._pwm_floor_for_cmd(action["action"]))

    def test_empty_follow_strong_curve_holds_inside_tread_still(self):
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
                {"target": "l", "action": "b", "pwm": 155},
                {"target": "r", "action": "f", "pwm": 0},
            ],
        )

    def test_empty_follow_turn_curves_load_profile_world_model_values(self):
        expected = {
            "gentle": {"inner_pwm": 104, "outer_pwm": 155},
            "medium": {"inner_pwm": 0, "outer_pwm": 140},
            "strong": {"inner_pwm": 0, "outer_pwm": 155},
        }

        for drive_mode in ("forward", "backward"):
            for strength, values in expected.items():
                curve = follow._turn_curve_for_drive_mode(drive_mode, strength)
                self.assertEqual(curve["drive_mode"], drive_mode)
                self.assertEqual(curve["strength"], strength)
                self.assertEqual(curve["inner_pwm"], values["inner_pwm"])
                self.assertEqual(curve["outer_pwm"], values["outer_pwm"])

    def test_holding_follow_turn_curves_use_conservative_profile_values(self):
        old_profile = getattr(follow, "CURRENT_GAME_PROFILE", "empty")
        try:
            follow._set_game_profile("holding")
            curve = follow._turn_curve_for_drive_mode("forward", "strong")
        finally:
            follow._set_game_profile(old_profile)

        self.assertEqual(curve["inner_pwm"], 104)
        self.assertEqual(curve["outer_pwm"], 185)

    def test_x_only_turn_policy_loads_distance_aware_drive_modes(self):
        cfg = follow._follow_motion_config()["x_only_turn"]

        self.assertEqual(cfg["drive_mode"], "backward")
        self.assertEqual(cfg["far_drive_mode"], "forward")
        self.assertEqual(cfg["forward_min_dist_err_mm"], 45.0)
        self.assertEqual(follow._x_only_turn_drive_mode_for_dist(10.0), "backward")
        self.assertEqual(follow._x_only_turn_drive_mode_for_dist(45.0), "forward")

    def test_adaptive_turn_curve_interpolates_from_x_gap(self):
        curve = follow._adaptive_turn_curve_for_drive_mode("forward", 14.0)

        self.assertEqual(curve["drive_mode"], "forward")
        self.assertEqual(curve["inner_pwm"], 0)
        self.assertEqual(curve["outer_pwm"], 151)
        self.assertAlmostEqual(curve["adaptive_outer_pwm_scale"], 1.0)
        self.assertEqual(curve["strength"], "adaptive_14.0mm")

    def test_results_table_includes_x_curve_learning_summary(self):
        stats = follow._new_game_stats()
        stats["x_curve_samples"].append(
            {
                "action": "TURN_R",
                "drive_mode": "backward",
                "strength": "adaptive_14.0mm",
                "inner_pwm": 104,
                "outer_pwm": 195,
                "x_before_mm": 14.0,
                "x_after_mm": 3.0,
                "abs_x_before_mm": 14.0,
                "abs_x_after_mm": 3.0,
                "x_reduction_mm": 11.0,
                "x_overshot": False,
            }
        )

        table = follow._format_game_results_table(stats)

        self.assertIn("| X Curve Learning | Value |", table)
        self.assertIn("| Samples | 1 |", table)
        self.assertIn("| Avg x reduction | 11.0mm |", table)

    def test_results_table_includes_gap_duration_wishes(self):
        stats = follow._new_game_stats()
        stats["gap_closure_samples"].extend(
            [
                {
                    "axis": "dist",
                    "action": "FWD",
                    "before_err": 2.4,
                    "after_err": -5.6,
                    "before_abs": 2.4,
                    "after_abs": 5.6,
                    "reduction": -3.2,
                    "overshot": True,
                    "overshot_within_tolerance": True,
                    "tol": 8.0,
                    "duration_ms": 200,
                },
                {
                    "axis": "x",
                    "action": "TURN_R",
                    "before_err": 10.0,
                    "after_err": 6.0,
                    "before_abs": 10.0,
                    "after_abs": 6.0,
                    "reduction": 4.0,
                    "overshot": False,
                    "overshot_within_tolerance": False,
                    "tol": 4.0,
                    "duration_ms": 200,
                },
            ]
        )

        table = follow._format_game_results_table(stats)

        self.assertIn("| dist overshot | 1/1 |", table)
        self.assertIn("dist duration wish | overshot: I wish our duration was 140ms lower (200ms->60ms, FWD +2.4->-5.6mm)", table)
        self.assertIn("x duration wish | undershot: I wish our duration was 300ms higher (200ms->500ms, TURN_R +10.0->+6.0mm)", table)
        self.assertIn("| Single Biggest Problem | What must change |", table)
        self.assertIn("x under-closed after TURN_R: still 2.0mm outside tolerance", table)

    def test_results_table_captures_regression_shots(self):
        stats = follow._new_game_stats()
        stats["gap_closure_samples"].append(
            {
                "axis": "dist",
                "action": "TURN_R_MAST_D",
                "before_err": -5.3,
                "after_err": -13.3,
                "before_abs": 5.3,
                "after_abs": 13.3,
                "reduction": -8.0,
                "regressed": True,
                "regression_mm": 8.0,
                "overshot": False,
                "overshot_within_tolerance": False,
                "tol": 8.0,
                "duration_ms": 200,
            }
        )

        table = follow._format_game_results_table(stats)

        self.assertIn("| dist regressed | 1/1 |", table)
        self.assertIn("| dist worst regression | TURN_R_MAST_D -5.3->-13.3mm (8.0mm worse) |", table)
        self.assertIn("dist regressed after TURN_R_MAST_D: gap worsened by 8.0mm to 13.3mm", table)
        self.assertIn("regressed: I wish our duration was 120ms lower (200ms->80ms, TURN_R_MAST_D -5.3->-13.3mm)", table)

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

    def test_send_turn_curve_preserves_all_four_slow_directional_curves(self):
        reading = {
            "visible": True,
            "confident": True,
            "conf": 88.0,
            "min_confidence_pct": 75.0,
            "x_mm": follow._x_target_mm() + 12.0,
        }

        for drive_mode in ("forward", "backward"):
            for cmd in ("l", "r"):
                robot = _FakeRobot()
                result = follow._send_turn_curve(
                    robot,
                    cmd=cmd,
                    drive_mode=drive_mode,
                    strength="gentle",
                    duration_ms=follow.PULSE_MS,
                    reading=reading,
                    context="test_all_directional_curves",
                )
                expected_curve = follow._turn_curve_for_drive_mode(drive_mode, "gentle")

                self.assertEqual(robot.commands, [])
                self.assertEqual(len(robot.custom_commands), 1)
                sent_cmd, actions, duration_ms = robot.custom_commands[0]
                self.assertEqual(sent_cmd, cmd)
                self.assertEqual(duration_ms, follow.PULSE_MS)
                self.assertEqual(
                    actions,
                    follow._turn_curve_actions(drive_mode=drive_mode, cmd=cmd, curve=expected_curve),
                )
                self.assertEqual(result["duration_ms"], follow.PULSE_MS)

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
