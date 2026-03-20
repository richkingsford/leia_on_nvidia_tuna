import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


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
            "hsv_lower": [75, 60, 50],
            "hsv_upper": [115, 255, 255],
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
                "hsv_lower": [80, 80, 60],
                "hsv_upper": [110, 255, 255],
                "hsv_erode_iterations": 1,
                "smoothing_alpha": 0.20,
            },
        )
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.20)
        self.assertEqual(int(runtime.get("hsv_erode_iterations")), 1)
        self.assertIsNone(vision._prev_angle)
        self.assertIsNone(vision._prev_dist)
        self.assertIsNone(vision._prev_offset)

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
