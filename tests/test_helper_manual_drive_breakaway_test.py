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
        self.assertEqual(recenter_calls, ["f", "q"])
        self.assertEqual(len(result.get("recenter_checkpoints") or []), 2)
        self.assertFalse(bool((result.get("settings") or {}).get("use_manual_assist")))


if __name__ == "__main__":
    unittest.main()
