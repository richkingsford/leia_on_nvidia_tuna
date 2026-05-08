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
            "shape_gate_mode": "negative_cutouts",
            "negative_cutout_cyan_fill_max": 0.42,
            "negative_cutout_ring_cyan_min": 0.32,
            "negative_cutout_ring_dilate_px": 5,
            "negative_cutout_min_area_px": 18.0,
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
        if "shape_gate_mode" in kwargs:
            snapshot["shape_gate_mode"] = str(kwargs["shape_gate_mode"])
        if "negative_cutout_cyan_fill_max" in kwargs:
            snapshot["negative_cutout_cyan_fill_max"] = float(kwargs["negative_cutout_cyan_fill_max"])
        if "negative_cutout_ring_cyan_min" in kwargs:
            snapshot["negative_cutout_ring_cyan_min"] = float(kwargs["negative_cutout_ring_cyan_min"])
        if "negative_cutout_ring_dilate_px" in kwargs:
            snapshot["negative_cutout_ring_dilate_px"] = int(kwargs["negative_cutout_ring_dilate_px"])
        if "negative_cutout_min_area_px" in kwargs:
            snapshot["negative_cutout_min_area_px"] = float(kwargs["negative_cutout_min_area_px"])
        return snapshot


class TestSetupManualTrainingCyanProfiles(unittest.TestCase):
    def test_success_gate_step_dropdown_labels_are_numbered(self):
        options = setup_manual_training.STREAM_SUCCESS_GATE_STEP_OPTIONS
        self.assertTrue(bool(options))
        for idx, (value, label) in enumerate(options, start=1):
            self.assertEqual(label, f"{idx}. {value}")

    def test_cyan_profile_dropdown_exposes_ten_configs(self):
        options = setup_manual_training.CYAN_PROFILE_OPTIONS
        self.assertEqual(len(options), 10)
        self.assertEqual(
            options,
            [
                ("config1", "Config 1"),
                ("config2", "Config 2"),
                ("config3", "Config 3"),
                ("config4", "Config 4"),
                ("config5", "Config 5"),
                ("config6", "Config 6"),
                ("config7", "Config 7"),
                ("config8", "Config 8"),
                ("config9", "Config 9"),
                ("config10", "Config 10"),
            ],
        )
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
                "config2",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config2")
        self.assertEqual(app.stream_state.get("cyan_profile"), "config2")
        self.assertEqual(app.stream_state.get("markerless_profile"), "config2")
        self.assertEqual(
            vision.calls[-1],
            {
                "confidence": 0.15,
                "smoothing_alpha": 0.20,
                "hsv_enabled": True,
                "hsv_lower": list(setup_manual_training.CYAN_HSV_TIGHT_LOWER),
                "hsv_upper": list(setup_manual_training.CYAN_HSV_TIGHT_UPPER),
                "hsv_erode_iterations": 0,
                "shape_gate_mode": "negative_cutouts",
                "negative_cutout_cyan_fill_max": 0.42,
                "negative_cutout_ring_cyan_min": 0.32,
                "negative_cutout_ring_dilate_px": 5,
                "negative_cutout_min_area_px": 18.0,
                "depth_source_mode": "pinhole",
            },
        )
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.20)
        self.assertEqual(int(runtime.get("hsv_erode_iterations")), 0)
        self.assertIsNone(vision._prev_angle)
        self.assertIsNone(vision._prev_dist)
        self.assertIsNone(vision._prev_offset)

    def test_apply_cyan_profile_legacy_alias_restores_config10_runtime(self):
        app = _DummyApp()
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            key, runtime, applied = setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config10_hsv_disabled",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config10")
        self.assertEqual(
            vision.calls[-1],
            {
                "confidence": 0.08,
                "smoothing_alpha": 0.10,
                "hsv_enabled": True,
                "hsv_lower": list(setup_manual_training.CYAN_HSV_WIDE_LOWER),
                "hsv_upper": list(setup_manual_training.CYAN_HSV_WIDE_UPPER),
                "hsv_erode_iterations": 0,
                "shape_gate_mode": "negative_cutouts",
                "negative_cutout_cyan_fill_max": 1.00,
                "negative_cutout_ring_cyan_min": 0.00,
                "negative_cutout_ring_dilate_px": 10,
                "negative_cutout_min_area_px": 18.0,
                "depth_source_mode": "pinhole",
            },
        )
        self.assertTrue(bool(runtime.get("hsv_enabled")))
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.10)
        self.assertEqual(list(runtime.get("hsv_lower")), list(setup_manual_training.CYAN_HSV_WIDE_LOWER))
        self.assertEqual(list(runtime.get("hsv_upper")), list(setup_manual_training.CYAN_HSV_WIDE_UPPER))

    def test_apply_cyan_profile_legacy_crown_alias_resolves_to_config2(self):
        app = _DummyApp()
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            key, runtime, applied = setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "crown_brick_default",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config2")
        self.assertEqual(app.stream_state.get("cyan_profile"), "config2")
        self.assertEqual(app.stream_state.get("markerless_profile"), "config2")
        self.assertEqual(float(runtime.get("conf_threshold")), 0.15)
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.20)
        self.assertEqual(list(runtime.get("hsv_lower")), list(setup_manual_training.CYAN_HSV_TIGHT_LOWER))
        self.assertEqual(list(runtime.get("hsv_upper")), list(setup_manual_training.CYAN_HSV_TIGHT_UPPER))

    def test_apply_cyan_profile_legacy_wide_alias_resolves_to_config2(self):
        app = _DummyApp()
        vision = _FakeYolo()
        original_cls = setup_manual_training.YoloBrickDetector
        try:
            setup_manual_training.YoloBrickDetector = _FakeYolo
            key, runtime, applied = setup_manual_training._apply_cyan_profile(
                app,
                vision,
                "config6_wide_cyan_range",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config5")
        self.assertEqual(
            vision.calls[-1],
            {
                "confidence": 0.15,
                "smoothing_alpha": 0.20,
                "hsv_enabled": True,
                "hsv_lower": list(setup_manual_training.CYAN_HSV_WIDE_LOWER),
                "hsv_upper": list(setup_manual_training.CYAN_HSV_WIDE_UPPER),
                "hsv_erode_iterations": 0,
                "shape_gate_mode": "negative_cutouts",
                "negative_cutout_cyan_fill_max": 0.55,
                "negative_cutout_ring_cyan_min": 0.24,
                "negative_cutout_ring_dilate_px": 6,
                "negative_cutout_min_area_px": 18.0,
                "depth_source_mode": "pinhole",
            },
        )
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.20)
        self.assertEqual(int(runtime.get("hsv_erode_iterations")), 0)
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
                "config2",
            )
        finally:
            setup_manual_training.YoloBrickDetector = original_cls

        self.assertTrue(applied)
        self.assertEqual(key, "config2")
        self.assertEqual(float(runtime.get("conf_threshold")), 0.15)
        self.assertEqual(float(runtime.get("smooth_alpha")), 0.20)
        self.assertEqual(list(runtime.get("hsv_lower")), list(setup_manual_training.CYAN_HSV_TIGHT_LOWER))
        self.assertEqual(list(runtime.get("hsv_upper")), list(setup_manual_training.CYAN_HSV_TIGHT_UPPER))
        self.assertTrue(bool(runtime.get("hsv_enabled")))
        self.assertEqual(int(runtime.get("hsv_erode_iterations")), 0)
        self.assertEqual(runtime.get("shape_gate_mode"), "negative_cutouts")
        self.assertEqual(float(runtime.get("negative_cutout_cyan_fill_max")), 0.42)
        self.assertEqual(float(runtime.get("negative_cutout_ring_cyan_min")), 0.32)
        self.assertEqual(int(runtime.get("negative_cutout_ring_dilate_px")), 5)
        self.assertEqual(float(runtime.get("negative_cutout_min_area_px")), 18.0)
        self.assertEqual(vision.conf_threshold, 0.15)
        self.assertEqual(vision._smooth_alpha, 0.20)
        self.assertEqual(list(vision._hsv_lower), list(setup_manual_training.CYAN_HSV_TIGHT_LOWER))
        self.assertEqual(list(vision._hsv_upper), list(setup_manual_training.CYAN_HSV_TIGHT_UPPER))
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
            "shape_gate_mode",
            "negative_cutout_cyan_fill_max",
            "negative_cutout_ring_cyan_min",
            "negative_cutout_ring_dilate_px",
            "negative_cutout_min_area_px",
        }
        for profile_key, _label in setup_manual_training.CYAN_PROFILE_OPTIONS:
            resolved_key, settings = setup_manual_training.cyan_profile_settings(profile_key)
            self.assertEqual(resolved_key, profile_key)
            self.assertTrue(required_keys.issubset(set(settings.keys())))
            self.assertGreater(float(settings["confidence"]), 0.0)
            self.assertLessEqual(float(settings["confidence"]), 0.15)
            self.assertEqual(len(list(settings["hsv_lower"])), 3)
            self.assertEqual(len(list(settings["hsv_upper"])), 3)

    def test_hsv_disabled_alias_still_resolves_config2_settings(self):
        profile_key, settings = setup_manual_training.cyan_profile_settings("config10_hsv_disabled")
        self.assertEqual(profile_key, "config10")
        self.assertEqual(float(settings["confidence"]), 0.08)
        self.assertEqual(float(settings["smoothing_alpha"]), 0.10)
        self.assertEqual(list(settings["hsv_lower"]), list(setup_manual_training.CYAN_HSV_WIDE_LOWER))
        self.assertEqual(list(settings["hsv_upper"]), list(setup_manual_training.CYAN_HSV_WIDE_UPPER))
        self.assertTrue(bool(settings["hsv_enabled"]))
        self.assertEqual(int(settings["hsv_erode_iterations"]), 0)

    def test_cyan_profile_normalization_supports_numeric_and_legacy_aliases(self):
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("config2"),
            "config2",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("default"),
            "config9",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("config2_no_erosion"),
            "config2",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("crown_brick_default"),
            "config2",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("8"),
            "config8",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("balanced"),
            "config2",
        )
        self.assertEqual(
            setup_manual_training.normalize_markerless_profile("balanced"),
            "config2",
        )
        self.assertEqual(
            setup_manual_training.normalize_cyan_profile("best"),
            "config9",
        )

    def test_cyan_profile_default_constant_uses_config9(self):
        self.assertEqual(setup_manual_training.CYAN_PROFILE_DEFAULT, "config9")
        self.assertEqual(setup_manual_training._DEFAULT_CYAN_PROFILE, "config9")
        self.assertEqual(setup_manual_training.MARKERLESS_PROFILE_DEFAULT, "config9")
        self.assertEqual(setup_manual_training.MARKERLESS_PROFILE_BALANCED, "config2")

    def test_cyan_stream_state_getter_accepts_legacy_markerless_key(self):
        app = _DummyApp()
        app.stream_state["cyan_profile"] = ""
        app.stream_state["markerless_profile"] = "config8_responsive"
        self.assertEqual(
            setup_manual_training._stream_state_cyan_profile(app, setup_manual_training.CYAN_PROFILE_DEFAULT),
            "config8",
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
            model_path = "/tmp/brick_yolo_v4_fast.plan"
            last_status = "target locked (HSV)"
            last_raw_prediction_count = 8
            last_candidate_count = 3
            last_nms_count = 2
            last_primary_confidence = 0.91
            last_max_confidence = 0.97
            conf_threshold = 0.15
            _smooth_alpha = 0.20
            input_size = 640
            last_partial_labels = ["TOP HALF", "LOWER PARTIAL"]
            last_primary_partial_label = "TOP HALF"

        lines = setup_manual_training.cyan_stream_debug_lines(app, _Vision())

        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[-1][1], (0, 165, 255))
        self.assertIn("PRIMARY TOP HALF", lines[-1][0])
        self.assertIn("TOP HALF x1", lines[-1][0])
        self.assertIn("LOWER PARTIAL x1", lines[-1][0])


if __name__ == "__main__":
    unittest.main()
