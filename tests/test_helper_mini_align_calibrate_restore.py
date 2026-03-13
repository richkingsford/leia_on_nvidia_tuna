import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_mini_align_calibrate as mini_x


class _FakeCalibrator:
    last_instance = None

    def __init__(self, **_kwargs):
        _FakeCalibrator.last_instance = self
        self.log = SimpleNamespace(
            events=[
                {"type": "confidence_summary", "confident": True},
                {"type": "conclusion", "data": {}, "persist_result": {"ok": True}},
            ]
        )
        self.tiny_discovery_by_cmd = {}
        self.tiny_discovery_persist_result = {}
        self.tiny_discovery_ran_this_instance = False
        self.restore_calls = []
        self.closed = False

    def _get_pose(self, *, samples=1, timeout_s=1.5):
        _ = (samples, timeout_s)
        return {"offset_x": 12.5}

    def _recover_from_lost_vision(self, *, source_phase):
        _ = source_phase
        return None

    def run(self):
        return None

    def restore_to_offset(self, target_offset_x):
        self.restore_calls.append(float(target_offset_x))
        return {
            "ok": True,
            "acts": 2,
            "remaining_mm": 0.2,
            "target_offset_x": float(target_offset_x),
            "final_offset_x": 12.3,
        }

    def close(self):
        self.closed = True


class TestHelperMiniXAxisCalibrateRestore(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(mini_x, name)
        setattr(mini_x, name, value)
        self.addCleanup(setattr, mini_x, name, original)

    def test_run_mini_x_axis_calibration_restores_start_pose(self):
        self._patch_module_attr("XAxisCalibrator", _FakeCalibrator)
        self._patch_module_attr(
            "_persist_auto_mini_run_meta",
            lambda **_kwargs: {"ok": True},
        )

        summary = mini_x.run_mini_x_axis_calibration(
            robot=object(),
            vision=object(),
            step_key="POSITION_BRICK",
            trials=1,
            min_interval_hours=0.0,
            log_fn=lambda _msg: None,
        )

        fake = _FakeCalibrator.last_instance
        self.assertIsNotNone(fake)
        self.assertEqual(fake.restore_calls, [12.5])
        self.assertTrue(fake.closed)
        self.assertTrue(summary.get("ok"))
        restore = summary.get("restore_start_pose_result") or {}
        self.assertTrue(restore.get("ok"))
        self.assertAlmostEqual(float(restore.get("target_offset_x")), 12.5, places=3)


if __name__ == "__main__":
    unittest.main()
