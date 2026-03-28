import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import a_MAIN as setup_manual_training


class _DummyApp:
    def __init__(self):
        self.stream_state = {
            "cyan_profile": "",
            "markerless_profile": "",
            "cyan_visibility": setup_manual_training.CYAN_VISIBILITY_AUTO,
            "markerless_visibility": setup_manual_training.CYAN_VISIBILITY_AUTO,
            "lock": None,
        }


class _FakeYolo:
    def __init__(self):
        self.calls = []
        self._prev_angle = 1.0
        self._prev_dist = 1.0
        self._prev_offset = 1.0

    def set_runtime_tuning(self, **kwargs):
        self.calls.append(dict(kwargs))
        snapshot = {
            "conf_threshold": 0.15,
            "smooth_alpha": 0.30,
            "nms_threshold": 0.45,
            "hsv_lower": list(setup_manual_training.CYAN_HSV_BALANCED_LOWER),
            "hsv_upper": list(setup_manual_training.CYAN_HSV_BALANCED_UPPER),
            "hsv_enabled": True,
            "hsv_erode_iterations": 2,
        }
        if "confidence" in kwargs:
            snapshot["conf_threshold"] = float(kwargs["confidence"])
        if "smoothing_alpha" in kwargs:
            snapshot["smooth_alpha"] = float(kwargs["smoothing_alpha"])
        if "hsv_lower" in kwargs:
            snapshot["hsv_lower"] = list(kwargs["hsv_lower"])
        if "hsv_upper" in kwargs:
            snapshot["hsv_upper"] = list(kwargs["hsv_upper"])
        if "hsv_enabled" in kwargs:
            snapshot["hsv_enabled"] = bool(kwargs["hsv_enabled"])
        if "hsv_erode_iterations" in kwargs:
            snapshot["hsv_erode_iterations"] = int(kwargs["hsv_erode_iterations"])
        return snapshot


class TestSetupManualTrainingCyanProfiles(unittest.TestCase):
    def test_success_gate_step_dropdown_labels_are_numbered(self):
        options = setup_manual_training.STREAM_SUCCESS_GATE_STEP_OPTIONS
        self.assertTrue(bool(options))
        for idx, (value, label) in enumerate(options, start=1):
            self.assertEqual(label, f"{idx}. {value}")

    def test_cyan_profile_dropdown_contains_requested_10_configs(self):
        options = setup_manual_training.CYAN_PROFILE_OPTIONS
        self.assertEqual(len(options), 10)
        self.assertEqual(options[0][0], "config1_defaults")
        self.assertEqual(options[-1][0], "config10_hsv_disabled")
        self.assertEqual(setup_manual_training.MARKERLESS_PROFILE_OPTIONS, options)

    def test_apply_cyan_profile_uses_exact_runtime_tuning_payload(self):
        app = _DummyApp()
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            key, runtime, applied = setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config9_tight_light_smooth",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config9_tight_light_smooth")
        self.assertEqual(app.stream_state.get("cyan_profile"), "config9_tight_light_smooth")
        self.assertEqual(app.stream_state.get("markerless_profile"), "config9_tight_light_smooth")
        self.assertEqual(
            vision.calls[-1],
            {
                "confidence": 0.15,
                "smoothing_alpha": 0.20,
                "hsv_enabled": True,
                "hsv_lower": list(setup_manual_training.CYAN_HSV_TIGHT_LOWER),
                "hsv_upper": list(setup_manual_training.CYAN_HSV_TIGHT_UPPER),
                "hsv_erode_iterations": 1,
            },
        )
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.20)
        self.assertEqual(int(runtime.get("hsv_erode_iterations")), 1)
        self.assertIsNone(vision._prev_angle)
        self.assertIsNone(vision._prev_dist)
        self.assertIsNone(vision._prev_offset)

    def test_apply_cyan_profile_restores_hsv_after_disabled_profile(self):
        app = _DummyApp()
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config10_hsv_disabled",
            )
            key, runtime, applied = setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config2_no_erosion",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config2_no_erosion")
        self.assertEqual(
            vision.calls[-1],
            {
                "confidence": 0.15,
                "smoothing_alpha": 0.30,
                "hsv_enabled": True,
                "hsv_lower": list(setup_manual_training.CYAN_HSV_BALANCED_LOWER),
                "hsv_upper": list(setup_manual_training.CYAN_HSV_BALANCED_UPPER),
                "hsv_erode_iterations": 0,
            },
        )
        self.assertTrue(bool(runtime.get("hsv_enabled")))
        self.assertEqual(list(runtime.get("hsv_lower")), list(setup_manual_training.CYAN_HSV_BALANCED_LOWER))
        self.assertEqual(list(runtime.get("hsv_upper")), list(setup_manual_training.CYAN_HSV_BALANCED_UPPER))

    def test_apply_cyan_profile_switches_from_tight_profile_back_to_wide_defaults(self):
        app = _DummyApp()
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config9_tight_light_smooth",
            )
            key, runtime, applied = setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config6_wide_cyan_range",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config6_wide_cyan_range")
        self.assertEqual(
            vision.calls[-1],
            {
                "confidence": 0.15,
                "smoothing_alpha": 0.30,
                "hsv_enabled": True,
                "hsv_lower": list(setup_manual_training.CYAN_HSV_WIDE_LOWER),
                "hsv_upper": list(setup_manual_training.CYAN_HSV_WIDE_UPPER),
                "hsv_erode_iterations": 2,
            },
        )
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.30)
        self.assertEqual(int(runtime.get("hsv_erode_iterations")), 2)
        self.assertEqual(list(runtime.get("hsv_lower")), list(setup_manual_training.CYAN_HSV_WIDE_LOWER))
        self.assertEqual(list(runtime.get("hsv_upper")), list(setup_manual_training.CYAN_HSV_WIDE_UPPER))

    def test_apply_cyan_profile_legacy_detector_path_restores_full_snapshot(self):
        app = _DummyApp()

        class _LegacyYolo:
            def __init__(self):
                self.conf_threshold = 0.95
                self._smooth_alpha = 0.99
                self.nms_threshold = 0.77
                self._hsv_lower = [1, 2, 3]
                self._hsv_upper = [4, 5, 6]
                self._hsv_enabled = False
                self._hsv_erode_iterations = 9
                self._prev_angle = 1.0
                self._prev_dist = 1.0
                self._prev_offset = 1.0
                self._prev_offset_y = 1.0
                self._center_lock_prev_center = (10.0, 20.0)

        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _LegacyYolo
            vision = _LegacyYolo()
            key, runtime, applied = setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config2_no_erosion",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config2_no_erosion")
        self.assertEqual(float(runtime.get("conf_threshold")), 0.15)
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.30)
        self.assertEqual(list(runtime.get("hsv_lower")), list(setup_manual_training.CYAN_HSV_BALANCED_LOWER))
        self.assertEqual(list(runtime.get("hsv_upper")), list(setup_manual_training.CYAN_HSV_BALANCED_UPPER))
        self.assertTrue(bool(runtime.get("hsv_enabled")))
        self.assertEqual(int(runtime.get("hsv_erode_iterations")), 0)
        self.assertEqual(vision.conf_threshold, 0.15)
        self.assertEqual(vision._smooth_alpha, 0.30)
        self.assertEqual(list(vision._hsv_lower), list(setup_manual_training.CYAN_HSV_BALANCED_LOWER))
        self.assertEqual(list(vision._hsv_upper), list(setup_manual_training.CYAN_HSV_BALANCED_UPPER))
        self.assertTrue(bool(vision._hsv_enabled))
        self.assertEqual(int(vision._hsv_erode_iterations), 0)
        self.assertIsNone(vision._prev_angle)
        self.assertIsNone(vision._prev_dist)
        self.assertIsNone(vision._prev_offset)
        self.assertIsNone(vision._prev_offset_y)
        self.assertIsNone(vision._center_lock_prev_center)

    def test_every_cyan_profile_expands_to_full_runtime_settings(self):
        required_keys = {
            "confidence",
            "smoothing_alpha",
            "hsv_enabled",
            "hsv_lower",
            "hsv_upper",
            "hsv_erode_iterations",
        }
        for profile_key, _label in setup_manual_training.CYAN_PROFILE_OPTIONS:
            resolved_key, settings = setup_manual_training.cyan_profile_settings(profile_key)
            self.assertEqual(resolved_key, profile_key)
            self.assertTrue(required_keys.issubset(set(settings.keys())))
            self.assertEqual(float(settings["confidence"]), 0.15)
            self.assertEqual(len(list(settings["hsv_lower"])), 3)
            self.assertEqual(len(list(settings["hsv_upper"])), 3)

    def test_hsv_disabled_profile_still_resolves_full_baseline_settings(self):
        profile_key, settings = setup_manual_training.cyan_profile_settings("config10_hsv_disabled")
        self.assertEqual(profile_key, "config10_hsv_disabled")
        self.assertEqual(float(settings["confidence"]), 0.15)
        self.assertEqual(float(settings["smoothing_alpha"]), 0.30)
        self.assertEqual(list(settings["hsv_lower"]), list(setup_manual_training.CYAN_HSV_BALANCED_LOWER))
        self.assertEqual(list(settings["hsv_upper"]), list(setup_manual_training.CYAN_HSV_BALANCED_UPPER))
        self.assertFalse(bool(settings["hsv_enabled"]))
        self.assertEqual(int(settings["hsv_erode_iterations"]), 2)

    def test_cyan_profile_normalization_supports_numeric_and_legacy_aliases(self):
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("config2"),
            "config2_no_erosion",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("default"),
            "config2_no_erosion",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("8"),
            "config8_responsive",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("balanced"),
            "config1_defaults",
        )
        self.assertEqual(
            setup_manual_training.normalize_markerless_profile("balanced"),
            "config1_defaults",
        )

    def test_cyan_profile_default_constant_uses_config2(self):
        self.assertEqual(setup_manual_training.CYAN_PROFILE_DEFAULT, "config2_no_erosion")
        self.assertEqual(setup_manual_training._DEFAULT_CYAN_PROFILE, "config2_no_erosion")

    def test_cyan_stream_state_getter_accepts_legacy_markerless_key(self):
        app = _DummyApp()
        app.stream_state["cyan_profile"] = ""
        app.stream_state["markerless_profile"] = "config8_responsive"
        self.assertEqual(
            setup_manual_training._stream_state_cyan_profile(app, setup_manual_training.CYAN_PROFILE_DEFAULT),
            "config8_responsive",
        )

    def test_cyan_smoothing_experiment_caps_high_alpha(self):
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            runtime, applied = setup_manual_training._apply_cyan_smoothing_experiment(
                vision,
                {"smooth_alpha": 0.50},
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls
        self.assertTrue(applied)
        self.assertEqual(
            vision.calls[-1],
            {"smoothing_alpha": setup_manual_training.CYAN_EXPERIMENT_SMOOTHING_ALPHA_MAX},
        )
        self.assertEqual(
            float(runtime.get("smooth_alpha")),
            float(setup_manual_training.CYAN_EXPERIMENT_SMOOTHING_ALPHA_MAX),
        )

    def test_cyan_smoothing_experiment_keeps_already_smooth_profiles_unchanged(self):
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            runtime, applied = setup_manual_training._apply_cyan_smoothing_experiment(
                vision,
                {"smooth_alpha": 0.10},
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls
        self.assertFalse(applied)
        self.assertEqual(runtime, {"smooth_alpha": 0.10})
        self.assertEqual(vision.calls, [])

    def test_cyan_stream_debug_lines_include_orange_partial_summary(self):
        app = _DummyApp()
        app.stream_state["vision_mode"] = setup_manual_training.VISION_MODE_CYAN

        class _Vision:
            model_path = "/tmp/brick_yolo_v4.onnx"
            last_status = "target locked (HSV)"
            last_raw_prediction_count = 8
            last_candidate_count = 3
            last_nms_count = 2
            last_primary_confidence = 0.91
            last_max_confidence = 0.97
            conf_threshold = 0.15
            _smooth_alpha = 0.20
            input_size = 640
            last_partial_labels = ["TOP HALF", "BOTTOM HALF"]
            last_primary_partial_label = "TOP HALF"

        lines = setup_manual_training.cyan_stream_debug_lines(app, _Vision())

        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[-1][1], (0, 165, 255))
        self.assertIn("PRIMARY TOP HALF", lines[-1][0])
        self.assertIn("TOP HALF x1", lines[-1][0])
        self.assertIn("BOTTOM HALF x1", lines[-1][0])


if __name__ == "__main__":
    unittest.main()
