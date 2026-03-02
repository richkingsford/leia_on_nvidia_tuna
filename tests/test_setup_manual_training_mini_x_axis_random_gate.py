import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class _DummyAppState:
    def __init__(self):
        self.running = True
        self.robot = object()
        self.vision = object()
        self.lock = threading.Lock()
        self.brick_frame_buffer = []
        self.demos_dir = Path(".")
        self.world = SimpleNamespace(process_rules={}, rules={}, step_state=None)
        self.stream_state = {
            "lock": threading.Lock(),
            "success_gate_step": "ALIGN_BRICK",
        }


class TestSetupManualTrainingMiniXAxisRandomGate(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(setup_manual_training, name)
        setattr(setup_manual_training, name, value)
        self.addCleanup(setattr, setup_manual_training, name, original)

    def test_mini_x_axis_skipped_when_random_roll_above_gate(self):
        app = _DummyAppState()
        calls = {"mini": 0}

        self._patch_module_attr("AUTO_MINI_X_AXIS_ENABLED", True)
        self._patch_module_attr("AUTO_MINI_X_AXIS_DISCOVERY_CHANCE", 0.05)
        self._patch_module_attr("random", SimpleNamespace(random=lambda: 0.90))
        self._patch_module_attr("load_demo_logs", lambda _d: [])
        self._patch_module_attr("update_process_model_from_demos", lambda *_a, **_k: None)
        self._patch_module_attr("refresh_autobuild_config", lambda *_a, **_k: None)
        self._patch_module_attr(
            "run_mini_x_axis_calibration",
            lambda **_kwargs: calls.__setitem__("mini", calls["mini"] + 1) or {"ok": True},
        )
        self._patch_module_attr("log_line", lambda _msg: None)

        ok = setup_manual_training.run_auto_step(app, setup_manual_training.StepState.POSITION_BRICK)
        self.assertFalse(ok)
        self.assertEqual(calls["mini"], 0)
        self.assertEqual(app.stream_state.get("success_gate_step"), "POSITION_BRICK")

    def test_mini_x_axis_runs_when_random_roll_below_gate(self):
        app = _DummyAppState()
        calls = {"mini": 0}

        self._patch_module_attr("AUTO_MINI_X_AXIS_ENABLED", True)
        self._patch_module_attr("AUTO_MINI_X_AXIS_DISCOVERY_CHANCE", 0.05)
        self._patch_module_attr("random", SimpleNamespace(random=lambda: 0.01))
        self._patch_module_attr("load_demo_logs", lambda _d: [])
        self._patch_module_attr("update_process_model_from_demos", lambda *_a, **_k: None)
        self._patch_module_attr("refresh_autobuild_config", lambda *_a, **_k: None)
        self._patch_module_attr(
            "run_mini_x_axis_calibration",
            lambda **_kwargs: calls.__setitem__("mini", calls["mini"] + 1) or {"ok": True},
        )
        self._patch_module_attr("log_line", lambda _msg: None)

        ok = setup_manual_training.run_auto_step(app, setup_manual_training.StepState.POSITION_BRICK)
        self.assertFalse(ok)
        self.assertEqual(calls["mini"], 1)
        self.assertEqual(app.stream_state.get("success_gate_step"), "POSITION_BRICK")

    def test_align_step_runs_visibility_recovery_after_mini_calibration(self):
        app = _DummyAppState()
        calls = {"mini": 0, "recover": 0}
        recover_args = []

        self._patch_module_attr("AUTO_MINI_X_AXIS_ENABLED", True)
        self._patch_module_attr("AUTO_MINI_X_AXIS_DISCOVERY_CHANCE", 0.05)
        self._patch_module_attr("random", SimpleNamespace(random=lambda: 0.01))
        self._patch_module_attr("load_demo_logs", lambda _d: [])
        self._patch_module_attr("update_process_model_from_demos", lambda *_a, **_k: None)
        self._patch_module_attr("refresh_autobuild_config", lambda *_a, **_k: None)
        self._patch_module_attr(
            "run_mini_x_axis_calibration",
            lambda **_kwargs: calls.__setitem__("mini", calls["mini"] + 1) or {"ok": True},
        )
        self._patch_module_attr(
            "run_align_mini_visibility_recovery",
            lambda *_a, **kwargs: calls.__setitem__("recover", calls["recover"] + 1)
            or recover_args.append(dict(kwargs))
            or {"ok": True, "visible": True, "acts": 0},
        )
        self._patch_module_attr("log_line", lambda _msg: None)

        ok = setup_manual_training.run_auto_step(app, setup_manual_training.StepState.ALIGN_BRICK)
        self.assertFalse(ok)
        self.assertEqual(calls["mini"], 1)
        self.assertEqual(calls["recover"], 1)
        self.assertEqual(recover_args[0].get("step_key"), "ALIGN_BRICK")


if __name__ == "__main__":
    unittest.main()
