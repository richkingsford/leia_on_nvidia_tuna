import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import setup_manual_training


class _DummyAutoContinueApp:
    def __init__(self):
        self.auto_continue_deadline = 0.0
        self.auto_continue_source_step = None
        self.auto_continue_target_step = None
        self.auto_running = False
        self.auto_request = None


class TestSetupManualTrainingAutoContinue(unittest.TestCase):
    def _patch_module_attr(self, name, value):
        original = getattr(setup_manual_training, name)
        setattr(setup_manual_training, name, value)
        self.addCleanup(setattr, setup_manual_training, name, original)

    def _patch_time_now(self, fn):
        original = setup_manual_training.time.time
        setup_manual_training.time.time = fn
        self.addCleanup(setattr, setup_manual_training.time, "time", original)

    def test_schedule_auto_continue_sets_deadline_and_target(self):
        app = _DummyAutoContinueApp()
        self._patch_module_attr("AUTO_CONTINUE_WAIT_S", 5.0)
        self._patch_time_now(lambda: 100.0)

        completed = setup_manual_training.DEMO_STEPS[0]
        next_expected = setup_manual_training.next_demo_step_obj(completed)

        plan = setup_manual_training._schedule_auto_continue_after_success_locked(app, completed)
        self.assertIsNotNone(plan)
        self.assertEqual(plan[0], next_expected)
        self.assertAlmostEqual(float(plan[1]), 5.0, places=3)
        self.assertEqual(app.auto_continue_source_step, completed)
        self.assertEqual(app.auto_continue_target_step, next_expected)
        self.assertAlmostEqual(float(app.auto_continue_deadline), 105.0, places=3)

    def test_schedule_auto_continue_noop_for_last_step(self):
        app = _DummyAutoContinueApp()
        self._patch_module_attr("AUTO_CONTINUE_WAIT_S", 5.0)
        self._patch_time_now(lambda: 100.0)

        plan = setup_manual_training._schedule_auto_continue_after_success_locked(
            app,
            setup_manual_training.DEMO_STEPS[-1],
        )
        self.assertIsNone(plan)
        self.assertEqual(float(app.auto_continue_deadline), 0.0)
        self.assertIsNone(app.auto_continue_target_step)

    def test_pop_due_auto_continue_returns_next_step_only_when_due(self):
        app = _DummyAutoContinueApp()
        self._patch_module_attr("AUTO_CONTINUE_WAIT_S", 5.0)
        self._patch_time_now(lambda: 100.0)
        completed = setup_manual_training.DEMO_STEPS[0]
        next_expected = setup_manual_training.next_demo_step_obj(completed)
        setup_manual_training._schedule_auto_continue_after_success_locked(app, completed)

        not_due = setup_manual_training._pop_due_auto_continue_locked(app, now_s=103.0)
        self.assertIsNone(not_due)
        self.assertEqual(app.auto_continue_target_step, next_expected)

        due = setup_manual_training._pop_due_auto_continue_locked(app, now_s=106.0)
        self.assertEqual(due, next_expected)
        self.assertEqual(float(app.auto_continue_deadline), 0.0)
        self.assertIsNone(app.auto_continue_target_step)

    def test_pop_due_auto_continue_respects_running_and_queued_states(self):
        app = _DummyAutoContinueApp()
        self._patch_module_attr("AUTO_CONTINUE_WAIT_S", 5.0)
        self._patch_time_now(lambda: 100.0)
        completed = setup_manual_training.DEMO_STEPS[0]
        setup_manual_training._schedule_auto_continue_after_success_locked(app, completed)

        app.auto_running = True
        self.assertIsNone(setup_manual_training._pop_due_auto_continue_locked(app, now_s=999.0))
        self.assertTrue(setup_manual_training._auto_continue_pending_locked(app))

        app.auto_running = False
        app.auto_request = setup_manual_training.DEMO_STEPS[1]
        self.assertIsNone(setup_manual_training._pop_due_auto_continue_locked(app, now_s=999.0))
        self.assertTrue(setup_manual_training._auto_continue_pending_locked(app))


if __name__ == "__main__":
    unittest.main()
