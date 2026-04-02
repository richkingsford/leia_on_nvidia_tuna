import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_manual_drive_breakaway_test


class TestHelperManualDriveBreakawayTest(unittest.TestCase):
    def test_recenter_status_uses_lenient_window(self):
        baseline = {"dist_mm": 180.0, "x_axis_mm": -12.0}
        pose_ok = {"dist_mm": 184.5, "x_axis_mm": -8.5}
        pose_bad = {"dist_mm": 186.0, "x_axis_mm": -6.0}

        ok_status = helper_manual_drive_breakaway_test._recenter_status(
            pose_ok,
            baseline,
            tolerance_dist_mm=5.0,
            tolerance_x_axis_mm=5.0,
        )
        bad_status = helper_manual_drive_breakaway_test._recenter_status(
            pose_bad,
            baseline,
            tolerance_dist_mm=5.0,
            tolerance_x_axis_mm=5.0,
        )

        self.assertTrue(ok_status["ok"])
        self.assertFalse(bad_status["ok"])

    def test_resolve_requested_hotkeys_supports_drive_turn_and_all(self):
        self.assertEqual(
            helper_manual_drive_breakaway_test._resolve_requested_hotkeys(""),
            ("r", "f", "q", "e"),
        )
        self.assertEqual(
            helper_manual_drive_breakaway_test._resolve_requested_hotkeys("drive"),
            ("r", "f"),
        )
        self.assertEqual(
            helper_manual_drive_breakaway_test._resolve_requested_hotkeys("turn"),
            ("q", "e"),
        )
        self.assertEqual(
            helper_manual_drive_breakaway_test._resolve_requested_hotkeys("all"),
            ("r", "f", "q", "e"),
        )
        self.assertEqual(
            helper_manual_drive_breakaway_test._resolve_requested_hotkeys("l"),
            ("q",),
        )
        self.assertEqual(
            helper_manual_drive_breakaway_test._parse_hotkeys_csv(""),
            ("r", "f", "q", "e"),
        )

    def test_movement_flags_require_correct_direction(self):
        forward_good = helper_manual_drive_breakaway_test._movement_flags_for_hotkey(
            "r",
            -0.3,
            threshold_mm=0.1,
        )
        forward_bad = helper_manual_drive_breakaway_test._movement_flags_for_hotkey(
            "r",
            0.3,
            threshold_mm=0.1,
        )
        turn_left_good = helper_manual_drive_breakaway_test._movement_flags_for_hotkey(
            "q",
            0.3,
            threshold_mm=0.1,
        )

        self.assertTrue(forward_good["moved"])
        self.assertTrue(forward_good["direction_ok"])
        self.assertTrue(forward_good["passes"])
        self.assertEqual(forward_good["observed_effect"], "decrease")

        self.assertTrue(forward_bad["moved"])
        self.assertFalse(forward_bad["direction_ok"])
        self.assertFalse(forward_bad["passes"])
        self.assertEqual(forward_bad["observed_effect"], "increase")

        self.assertTrue(turn_left_good["passes"])
        self.assertEqual(turn_left_good["expected_effect"], "increase")

    def test_movement_flags_allow_expected_effect_override(self):
        flags = helper_manual_drive_breakaway_test._movement_flags_for_hotkey(
            "q",
            -0.3,
            threshold_mm=0.1,
            expected_effect_override="decrease",
        )
        self.assertTrue(flags["passes"])
        self.assertEqual(flags["expected_effect"], "decrease")

    def test_duration_for_trial_scales_high_probe(self):
        self.assertEqual(
            helper_manual_drive_breakaway_test._duration_for_trial(
                score=12,
                start_score=1,
                max_score=12,
                duration_ms=800,
            ),
            267,
        )
        self.assertEqual(
            helper_manual_drive_breakaway_test._duration_for_trial(
                score=6,
                start_score=1,
                max_score=12,
                duration_ms=800,
            ),
            800,
        )

    def test_build_vision_uses_cyan_detector_for_cyan_mode(self):
        original_yolo = helper_manual_drive_breakaway_test.YoloBrickDetector
        original_leia = helper_manual_drive_breakaway_test.LeiaVision
        original_aruco = helper_manual_drive_breakaway_test.ArucoBrickVision
        try:
            class _FakeYolo:
                def __init__(self, debug=False):
                    self.debug = debug

            class _FakeLeia:
                def __init__(self, debug=False):
                    self.debug = debug

            class _FakeAruco:
                def __init__(self, debug=False):
                    self.debug = debug

            helper_manual_drive_breakaway_test.YoloBrickDetector = _FakeYolo
            helper_manual_drive_breakaway_test.LeiaVision = _FakeLeia
            helper_manual_drive_breakaway_test.ArucoBrickVision = _FakeAruco

            vision = helper_manual_drive_breakaway_test._build_vision("cyan")
        finally:
            helper_manual_drive_breakaway_test.YoloBrickDetector = original_yolo
            helper_manual_drive_breakaway_test.LeiaVision = original_leia
            helper_manual_drive_breakaway_test.ArucoBrickVision = original_aruco

        self.assertIsInstance(vision, _FakeYolo)

    def test_build_vision_uses_aruco_when_requested(self):
        original_yolo = helper_manual_drive_breakaway_test.YoloBrickDetector
        original_leia = helper_manual_drive_breakaway_test.LeiaVision
        original_aruco = helper_manual_drive_breakaway_test.ArucoBrickVision
        try:
            class _FakeYolo:
                def __init__(self, debug=False):
                    self.debug = debug

            class _FakeLeia:
                def __init__(self, debug=False):
                    self.debug = debug

            class _FakeAruco:
                def __init__(self, debug=False):
                    self.debug = debug

            helper_manual_drive_breakaway_test.YoloBrickDetector = _FakeYolo
            helper_manual_drive_breakaway_test.LeiaVision = _FakeLeia
            helper_manual_drive_breakaway_test.ArucoBrickVision = _FakeAruco

            vision = helper_manual_drive_breakaway_test._build_vision("aruco")
        finally:
            helper_manual_drive_breakaway_test.YoloBrickDetector = original_yolo
            helper_manual_drive_breakaway_test.LeiaVision = original_leia
            helper_manual_drive_breakaway_test.ArucoBrickVision = original_aruco

        self.assertIsInstance(vision, _FakeAruco)

    def test_summarize_breakaway_results_recommends_first_passing_score(self):
        rows = []
        for idx in range(5):
            rows.append({"hotkey": "r", "score": 1, "ok": True, "moved": True, "passes": idx < 3, "abs_delta_mm": 0.12})
        for idx in range(5):
            rows.append({"hotkey": "r", "score": 2, "ok": True, "moved": True, "passes": idx < 4, "abs_delta_mm": 0.18})
        for idx in range(5):
            rows.append({"hotkey": "r", "score": 3, "ok": True, "moved": True, "passes": True, "abs_delta_mm": 0.24})

        summary = helper_manual_drive_breakaway_test.summarize_breakaway_results(
            rows,
            repeats_per_score=5,
            required_success_ratio=0.80,
        )

        by_hotkey = summary["by_hotkey"]["r"]
        self.assertEqual(by_hotkey["recommended_score"], 2)
        self.assertFalse(by_hotkey["scores"][0]["passes"])
        self.assertTrue(by_hotkey["scores"][1]["passes"])
        self.assertTrue(by_hotkey["scores"][2]["passes"])
        self.assertEqual(summary["required_success_count"], 4)

    def test_summarize_breakaway_results_handles_no_passing_score(self):
        rows = []
        for score in (1, 2, 3):
            for idx in range(5):
                rows.append({"hotkey": "f", "score": score, "ok": True, "moved": idx == 0, "abs_delta_mm": 0.05})

        summary = helper_manual_drive_breakaway_test.summarize_breakaway_results(
            rows,
            repeats_per_score=5,
            required_success_ratio=0.80,
        )

        self.assertIsNone(summary["by_hotkey"]["f"]["recommended_score"])

    def test_run_interactive_drive_breakaway_test_collects_prompt_inputs(self):
        prompts = iter(["all", "2", "6", "4", "320", "0.15"])
        captured = {}
        original_runner = helper_manual_drive_breakaway_test.run_drive_breakaway_test
        messages = []
        try:
            def _fake_runner(**kwargs):
                captured.update(kwargs)
                return {"ok": True, "summary": {}}

            helper_manual_drive_breakaway_test.run_drive_breakaway_test = _fake_runner
            result = helper_manual_drive_breakaway_test.run_interactive_drive_breakaway_test(
                robot=object(),
                vision=object(),
                world=object(),
                prompt_fn=lambda _prompt: next(prompts),
                log_fn=messages.append,
            )
        finally:
            helper_manual_drive_breakaway_test.run_drive_breakaway_test = original_runner

        self.assertTrue(bool(result.get("ok")))
        self.assertEqual(captured["hotkeys"], ("r", "f", "q", "e"))
        self.assertEqual(captured["start_score"], 2)
        self.assertEqual(captured["max_score"], 6)
        self.assertEqual(captured["repeats_per_score"], 4)
        self.assertEqual(captured["duration_ms"], 320)
        self.assertAlmostEqual(captured["movement_threshold_mm"], 0.15)
        self.assertTrue(bool(captured["use_manual_assist"]))
        self.assertTrue(any("assist" in str(line).lower() for line in messages))
        self.assertTrue(any("recentered" in str(line).lower() for line in messages))

    def test_run_drive_breakaway_test_adds_recenter_checkpoints_between_hotkeys(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        original_median = helper_manual_drive_breakaway_test._median_pose_or_none
        original_raw_trial = helper_manual_drive_breakaway_test._raw_trial
        original_recenter = helper_manual_drive_breakaway_test._run_recenter_checkpoint
        original_sleep = helper_manual_drive_breakaway_test.time.sleep
        logs = []
        recenter_calls = []
        try:
            helper_manual_drive_breakaway_test._collect_pose_samples = lambda *args, **kwargs: [{"dist_mm": 180.0, "x_axis_mm": -12.0}]
            helper_manual_drive_breakaway_test._median_pose_or_none = lambda values: dict(values[0]) if values else None

            def _fake_raw_trial(**kwargs):
                hotkey = kwargs["hotkey"]
                flags = helper_manual_drive_breakaway_test._movement_flags_for_hotkey(
                    hotkey,
                    -0.2 if hotkey in ("r", "e") else 0.2,
                    threshold_mm=0.1,
                )
                return {
                    "ok": True,
                    "hotkey": hotkey,
                    "score": kwargs["score"],
                    "abs_delta_mm": 0.2,
                    "moved": flags["moved"],
                    "direction_ok": flags["direction_ok"],
                    "passes": flags["passes"],
                    "metric_label": "dist",
                    "expected_effect": flags["expected_effect"],
                    "raw_delta_mm": -0.2 if hotkey in ("r", "e") else 0.2,
                }

            def _fake_recenter(**kwargs):
                recenter_calls.append(kwargs["next_hotkey"])
                return {"ok": True, "pose": {"dist_mm": 180.0, "x_axis_mm": -12.0}}

            helper_manual_drive_breakaway_test._raw_trial = _fake_raw_trial
            helper_manual_drive_breakaway_test._run_recenter_checkpoint = _fake_recenter
            helper_manual_drive_breakaway_test.time.sleep = lambda *_args, **_kwargs: None

            result = helper_manual_drive_breakaway_test.run_drive_breakaway_test(
                robot=object(),
                vision=object(),
                world=object(),
                hotkeys=("r", "f", "q"),
                start_score=1,
                max_score=1,
                repeats_per_score=1,
                duration_ms=250,
                movement_threshold_mm=0.1,
                required_success_ratio=0.8,
                pause_between_pulses_ms=0,
                log_fn=logs.append,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect
            helper_manual_drive_breakaway_test._median_pose_or_none = original_median
            helper_manual_drive_breakaway_test._raw_trial = original_raw_trial
            helper_manual_drive_breakaway_test._run_recenter_checkpoint = original_recenter
            helper_manual_drive_breakaway_test.time.sleep = original_sleep

        self.assertTrue(bool(result.get("ok")))
        self.assertEqual(recenter_calls, ["f", "q", "q", "q"])
        self.assertEqual(len(result.get("recenter_checkpoints") or []), 2)
        self.assertFalse(bool((result.get("settings") or {}).get("use_manual_assist")))

    def test_run_drive_breakaway_test_can_recenter_before_each_trial(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        original_median = helper_manual_drive_breakaway_test._median_pose_or_none
        original_raw_trial = helper_manual_drive_breakaway_test._raw_trial
        original_recenter = helper_manual_drive_breakaway_test._run_recenter_checkpoint
        original_sleep = helper_manual_drive_breakaway_test.time.sleep
        recenter_calls = []
        try:
            helper_manual_drive_breakaway_test._collect_pose_samples = lambda *args, **kwargs: [{"dist_mm": 180.0, "x_axis_mm": -12.0}]
            helper_manual_drive_breakaway_test._median_pose_or_none = lambda values: dict(values[0]) if values else None

            def _fake_raw_trial(**kwargs):
                return {
                    "ok": True,
                    "hotkey": kwargs["hotkey"],
                    "score": kwargs["score"],
                    "abs_delta_mm": 0.05,
                    "moved": False,
                    "direction_ok": False,
                    "passes": False,
                    "metric_label": "x_axis",
                    "expected_effect": "increase",
                    "raw_delta_mm": 0.05,
                }

            def _fake_recenter(**kwargs):
                recenter_calls.append(kwargs["checkpoint_label"])
                return {"ok": True, "pose": {"dist_mm": 180.0, "x_axis_mm": -12.0}}

            helper_manual_drive_breakaway_test._raw_trial = _fake_raw_trial
            helper_manual_drive_breakaway_test._run_recenter_checkpoint = _fake_recenter
            helper_manual_drive_breakaway_test.time.sleep = lambda *_args, **_kwargs: None

            result = helper_manual_drive_breakaway_test.run_drive_breakaway_test(
                robot=object(),
                vision=object(),
                world=object(),
                hotkeys=("q",),
                start_score=1,
                max_score=1,
                repeats_per_score=3,
                duration_ms=250,
                movement_threshold_mm=0.1,
                required_success_ratio=0.8,
                pause_between_pulses_ms=0,
                recenter_before_each_trial=True,
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect
            helper_manual_drive_breakaway_test._median_pose_or_none = original_median
            helper_manual_drive_breakaway_test._raw_trial = original_raw_trial
            helper_manual_drive_breakaway_test._run_recenter_checkpoint = original_recenter
            helper_manual_drive_breakaway_test.time.sleep = original_sleep

        self.assertTrue(bool(result.get("ok")))
        self.assertEqual(
            recenter_calls,
            [
                "Q/LEFT sign validation 1/2",
                "Q/LEFT sign validation 2/2",
                "Q/LEFT score=1% trial=1/3",
                "Q/LEFT score=1% trial=2/3",
            ],
        )
        self.assertEqual(len(result.get("recenter_checkpoints") or []), 2)
        self.assertTrue(bool((result.get("settings") or {}).get("recenter_before_each_trial")))

    def test_run_drive_breakaway_test_can_use_adaptive_bracket_search(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        original_median = helper_manual_drive_breakaway_test._median_pose_or_none
        original_raw_trial = helper_manual_drive_breakaway_test._raw_trial
        original_sleep = helper_manual_drive_breakaway_test.time.sleep
        seen_scores = []
        try:
            helper_manual_drive_breakaway_test._collect_pose_samples = lambda *args, **kwargs: [{"dist_mm": 180.0, "x_axis_mm": -12.0}]
            helper_manual_drive_breakaway_test._median_pose_or_none = lambda values: dict(values[0]) if values else None
            pass_scores = {7, 8}

            def _fake_raw_trial(**kwargs):
                score = int(kwargs["score"])
                hotkey = kwargs["hotkey"]
                seen_scores.append(score)
                passes = score in pass_scores
                raw_delta = 0.4 if passes else 0.05
                return {
                    "ok": True,
                    "hotkey": hotkey,
                    "score": score,
                    "abs_delta_mm": abs(raw_delta),
                    "moved": bool(passes),
                    "direction_ok": bool(passes),
                    "passes": bool(passes),
                    "metric_label": "x_axis",
                    "expected_effect": "increase",
                    "raw_delta_mm": raw_delta,
                }

            helper_manual_drive_breakaway_test._raw_trial = _fake_raw_trial
            helper_manual_drive_breakaway_test.time.sleep = lambda *_args, **_kwargs: None

            result = helper_manual_drive_breakaway_test.run_drive_breakaway_test(
                robot=object(),
                vision=object(),
                world=object(),
                hotkeys=("q",),
                start_score=1,
                max_score=8,
                repeats_per_score=5,
                duration_ms=250,
                movement_threshold_mm=0.1,
                required_success_ratio=0.8,
                pause_between_pulses_ms=0,
                score_search_strategy=helper_manual_drive_breakaway_test.SCORE_SEARCH_STRATEGY_ADAPTIVE_BRACKET,
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect
            helper_manual_drive_breakaway_test._median_pose_or_none = original_median
            helper_manual_drive_breakaway_test._raw_trial = original_raw_trial
            helper_manual_drive_breakaway_test.time.sleep = original_sleep

        unique_scores = []
        for value in seen_scores:
            if not unique_scores or unique_scores[-1] != value:
                unique_scores.append(value)
        self.assertEqual(unique_scores, [8, 1, 8, 4, 6, 7])
        self.assertEqual((result.get("summary") or {}).get("by_hotkey", {}).get("q", {}).get("recommended_score"), 7)
        self.assertEqual((result.get("score_search_paths") or {}).get("q"), [1, 8, 4, 6, 7])
        self.assertEqual(
            (result.get("settings") or {}).get("score_search_strategy"),
            helper_manual_drive_breakaway_test.SCORE_SEARCH_STRATEGY_ADAPTIVE_BRACKET,
        )
        sign_validation = result.get("sign_validation") or []
        self.assertEqual(len(sign_validation), 1)
        self.assertEqual(sign_validation[0].get("score"), 8)

    def test_raw_trial_resets_repeat_act_guard_before_send(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        original_send = helper_manual_drive_breakaway_test.send_robot_command
        original_reset = helper_manual_drive_breakaway_test.reset_repeat_act_guard
        original_sleep = helper_manual_drive_breakaway_test.time.sleep
        calls = []
        try:
            samples = iter(
                [
                    [{"dist_mm": 180.0, "x_axis_mm": -12.0}],
                    [{"dist_mm": 180.0, "x_axis_mm": -11.4}],
                ]
            )

            def _fake_collect(*args, **kwargs):
                return next(samples)

            def _fake_reset(world, robot):
                calls.append(("reset", world, robot))

            def _fake_send(robot, world, step, cmd, **kwargs):
                calls.append(("send", cmd, kwargs.get("speed_score")))
                return {"duration_ms": 250}

            helper_manual_drive_breakaway_test._collect_pose_samples = _fake_collect
            helper_manual_drive_breakaway_test.reset_repeat_act_guard = _fake_reset
            helper_manual_drive_breakaway_test.send_robot_command = _fake_send
            helper_manual_drive_breakaway_test.time.sleep = lambda *_args, **_kwargs: None

            row = helper_manual_drive_breakaway_test._raw_trial(
                robot=object(),
                vision=object(),
                world=object(),
                hotkey="q",
                score=8,
                duration_ms=3000,
                movement_threshold_mm=0.25,
                use_manual_assist=False,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect
            helper_manual_drive_breakaway_test.send_robot_command = original_send
            helper_manual_drive_breakaway_test.reset_repeat_act_guard = original_reset
            helper_manual_drive_breakaway_test.time.sleep = original_sleep

        self.assertTrue(bool(row.get("ok")))
        self.assertEqual(calls[0][0], "reset")
        self.assertEqual(calls[1], ("send", "l", 8))
        self.assertTrue(bool(row.get("passes")))

    def test_raw_trial_failure_rows_keep_hotkey_and_score(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        try:
            helper_manual_drive_breakaway_test._collect_pose_samples = lambda *args, **kwargs: []

            row = helper_manual_drive_breakaway_test._raw_trial(
                robot=object(),
                vision=object(),
                world=object(),
                hotkey="e",
                score=3,
                duration_ms=800,
                movement_threshold_mm=0.10,
                use_manual_assist=False,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect

        self.assertFalse(bool(row.get("ok")))
        self.assertEqual(row.get("error"), "pre_pose_unavailable")
        self.assertEqual(row.get("hotkey"), "e")
        self.assertEqual(row.get("score"), 3)
        self.assertEqual(row.get("display_label"), "E/RIGHT")

    def test_recover_pose_from_recent_turn_acts_replays_inverse_of_last_three_acts(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        original_send = helper_manual_drive_breakaway_test.send_robot_command
        original_reset = helper_manual_drive_breakaway_test.reset_repeat_act_guard
        original_sleep = helper_manual_drive_breakaway_test.time.sleep
        send_calls = []
        try:
            samples = iter(
                [
                    [],
                    [],
                    [{"dist_mm": 180.0, "x_axis_mm": -12.0}],
                ]
            )

            def _fake_collect(*args, **kwargs):
                return next(samples)

            def _fake_send(robot, world, step, cmd, **kwargs):
                send_calls.append((cmd, kwargs.get("speed_score"), kwargs.get("duration_override_ms")))
                return {"duration_ms": kwargs.get("duration_override_ms")}

            helper_manual_drive_breakaway_test._collect_pose_samples = _fake_collect
            helper_manual_drive_breakaway_test.send_robot_command = _fake_send
            helper_manual_drive_breakaway_test.reset_repeat_act_guard = lambda *_args, **_kwargs: None
            helper_manual_drive_breakaway_test.time.sleep = lambda *_args, **_kwargs: None

            result = helper_manual_drive_breakaway_test._recover_pose_from_recent_turn_acts(
                robot=object(),
                vision=object(),
                world=object(),
                recent_turn_acts=[
                    {"cmd": "l", "score": 2, "duration_ms": 100},
                    {"cmd": "r", "score": 3, "duration_ms": 120},
                    {"cmd": "l", "score": 4, "duration_ms": 140},
                    {"cmd": "r", "score": 5, "duration_ms": 160},
                ],
                source_label="unit_test",
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect
            helper_manual_drive_breakaway_test.send_robot_command = original_send
            helper_manual_drive_breakaway_test.reset_repeat_act_guard = original_reset
            helper_manual_drive_breakaway_test.time.sleep = original_sleep

        self.assertTrue(bool(result.get("ok")))
        self.assertEqual(
            send_calls,
            [
                ("l", 5, 160),
                ("r", 4, 140),
                ("l", 3, 120),
            ],
        )
        self.assertEqual(result.get("pose"), {"dist_mm": 180.0, "x_axis_mm": -12.0})

    def test_run_recenter_checkpoint_can_recover_with_recent_inverse_acts(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        original_recover = helper_manual_drive_breakaway_test._recover_pose_from_recent_turn_acts
        original_sleep = helper_manual_drive_breakaway_test.time.sleep
        try:
            helper_manual_drive_breakaway_test._collect_pose_samples = lambda *args, **kwargs: []
            helper_manual_drive_breakaway_test._recover_pose_from_recent_turn_acts = lambda **kwargs: {
                "ok": True,
                "pose": {"dist_mm": 180.0, "x_axis_mm": -12.0},
                "steps": [{"index": 1}],
            }
            helper_manual_drive_breakaway_test.time.sleep = lambda *_args, **_kwargs: None

            result = helper_manual_drive_breakaway_test._run_recenter_checkpoint(
                robot=object(),
                vision=object(),
                world=object(),
                baseline_pose={"dist_mm": 180.0, "x_axis_mm": -12.0},
                next_hotkey="q",
                recent_turn_acts=[{"cmd": "l", "score": 3, "duration_ms": 250}],
                timeout_s=0.01,
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect
            helper_manual_drive_breakaway_test._recover_pose_from_recent_turn_acts = original_recover
            helper_manual_drive_breakaway_test.time.sleep = original_sleep

        self.assertTrue(bool(result.get("ok")))
        self.assertEqual((result.get("recovery") or {}).get("ok"), True)
        self.assertEqual(result.get("pose"), {"dist_mm": 180.0, "x_axis_mm": -12.0})

    def test_raw_trial_can_recover_pre_pose_from_recent_inverse_acts(self):
        original_collect = helper_manual_drive_breakaway_test._collect_pose_samples
        original_send = helper_manual_drive_breakaway_test.send_robot_command
        original_reset = helper_manual_drive_breakaway_test.reset_repeat_act_guard
        original_recover = helper_manual_drive_breakaway_test._recover_pose_from_recent_turn_acts
        original_sleep = helper_manual_drive_breakaway_test.time.sleep
        send_calls = []
        try:
            samples = iter(
                [
                    [],
                    [{"dist_mm": 180.0, "x_axis_mm": -11.4}],
                ]
            )

            def _fake_collect(*args, **kwargs):
                return next(samples)

            def _fake_send(robot, world, step, cmd, **kwargs):
                send_calls.append((cmd, kwargs.get("speed_score")))
                return {"duration_ms": kwargs.get("duration_override_ms")}

            helper_manual_drive_breakaway_test._collect_pose_samples = _fake_collect
            helper_manual_drive_breakaway_test.send_robot_command = _fake_send
            helper_manual_drive_breakaway_test.reset_repeat_act_guard = lambda *_args, **_kwargs: None
            helper_manual_drive_breakaway_test._recover_pose_from_recent_turn_acts = lambda **kwargs: {
                "ok": True,
                "pose": {"dist_mm": 180.0, "x_axis_mm": -12.0},
                "steps": [{"index": 1}],
            }
            helper_manual_drive_breakaway_test.time.sleep = lambda *_args, **_kwargs: None

            row = helper_manual_drive_breakaway_test._raw_trial(
                robot=object(),
                vision=object(),
                world=object(),
                hotkey="q",
                score=3,
                duration_ms=250,
                movement_threshold_mm=0.10,
                use_manual_assist=False,
                recent_turn_acts=[],
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            helper_manual_drive_breakaway_test._collect_pose_samples = original_collect
            helper_manual_drive_breakaway_test.send_robot_command = original_send
            helper_manual_drive_breakaway_test.reset_repeat_act_guard = original_reset
            helper_manual_drive_breakaway_test._recover_pose_from_recent_turn_acts = original_recover
            helper_manual_drive_breakaway_test.time.sleep = original_sleep

        self.assertTrue(bool(row.get("ok")))
        self.assertEqual(send_calls, [("l", 3)])
        self.assertAlmostEqual(float(row.get("raw_delta_mm") or 0.0), 0.6, places=3)
        self.assertTrue(bool(row.get("passes")))

    def test_run_turn_sign_validation_returns_override_from_observed_effect(self):
        original_recenter = helper_manual_drive_breakaway_test._run_recenter_checkpoint
        original_raw_trial = helper_manual_drive_breakaway_test._raw_trial
        try:
            helper_manual_drive_breakaway_test._run_recenter_checkpoint = lambda **kwargs: {
                "ok": True,
                "pose": {"dist_mm": 180.0, "x_axis_mm": -12.0},
            }
            helper_manual_drive_breakaway_test._raw_trial = lambda **kwargs: {
                "ok": True,
                "observed_effect": "decrease",
                "moved": True,
            }
            result = helper_manual_drive_breakaway_test._run_turn_sign_validation(
                robot=object(),
                vision=object(),
                world=object(),
                hotkey="q",
                validation_score=12,
                validation_duration_ms=267,
                movement_threshold_mm=0.10,
                baseline_pose={"dist_mm": 180.0, "x_axis_mm": -12.0},
                recent_turn_acts=[],
                log_fn=lambda *_args, **_kwargs: None,
            )
        finally:
            helper_manual_drive_breakaway_test._run_recenter_checkpoint = original_recenter
            helper_manual_drive_breakaway_test._raw_trial = original_raw_trial

        self.assertTrue(bool(result.get("ok")))
        self.assertEqual(result.get("expected_effect_override"), "decrease")


if __name__ == "__main__":
    unittest.main()
