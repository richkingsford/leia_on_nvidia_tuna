import sys
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
        self.world = SimpleNamespace()


class TestSetupManualTrainingAlignMiniVisibilityRecovery(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(setup_manual_training, name)
        setattr(setup_manual_training, name, value)
        self.addCleanup(setattr, setup_manual_training, name, original)

    def test_returns_immediately_when_already_visible(self):
        app = _DummyAppState()
        send_calls = []

        self._patch_module_attr("AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_ENABLED", True)
        self._patch_module_attr("_auto_align_visible_now", lambda _app: True)
        self._patch_module_attr(
            "send_robot_command",
            lambda *_a, **_k: send_calls.append(True) or {"duration_ms": 30},
        )
        self._patch_module_attr("log_line", lambda _msg: None)

        result = setup_manual_training.run_align_mini_visibility_recovery(app, step_key="ALIGN_BRICK")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("acts"), 0)
        self.assertEqual(len(send_calls), 0)

    def test_turns_right_until_visible(self):
        app = _DummyAppState()
        send_calls = []
        visibility_sequence = [False, False, True]

        self._patch_module_attr("AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_ENABLED", True)
        self._patch_module_attr("AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_MAX_ACTS", 8)
        self._patch_module_attr("AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_TIMEOUT_S", 5.0)
        self._patch_module_attr("AUTO_ALIGN_MINI_VISIBILITY_RECOVERY_SCORE", 1)
        self._patch_module_attr(
            "_auto_align_visible_now",
            lambda _app: visibility_sequence.pop(0) if visibility_sequence else True,
        )
        self._patch_module_attr(
            "send_robot_command",
            lambda *_a, **_k: send_calls.append(True) or {"duration_ms": 10},
        )
        self._patch_module_attr("time", SimpleNamespace(time=lambda: 100.0, sleep=lambda _s: None))
        self._patch_module_attr("log_line", lambda _msg: None)

        result = setup_manual_training.run_align_mini_visibility_recovery(app, step_key="ALIGN_BRICK")
        self.assertTrue(result.get("ok"))
        self.assertTrue(result.get("visible"))
        self.assertEqual(result.get("acts"), 2)
        self.assertEqual(len(send_calls), 2)


if __name__ == "__main__":
    unittest.main()
