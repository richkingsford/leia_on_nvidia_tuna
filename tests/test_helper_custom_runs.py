import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_custom_runs


class TestHelperCustomRuns(unittest.TestCase):
    def test_parse_custom_run_steps_csv_accepts_reset_aliases(self):
        def _resolve(token):
            if token == "3":
                return "STEP3"
            return None

        steps, err = helper_custom_runs.parse_custom_run_steps_csv(
            "r,3,reset",
            resolve_step_token_fn=_resolve,
        )

        self.assertIsNone(err)
        self.assertEqual(
            steps,
            [
                helper_custom_runs.CUSTOM_RUN_RESET_STEP,
                "STEP3",
                helper_custom_runs.CUSTOM_RUN_RESET_STEP,
            ],
        )

    def test_parse_custom_run_steps_csv_rejects_unknown_tokens(self):
        steps, err = helper_custom_runs.parse_custom_run_steps_csv(
            "3,nope",
            resolve_step_token_fn=lambda token: "STEP3" if token == "3" else None,
        )

        self.assertIsNone(steps)
        self.assertEqual(err, "[CUSTOM RUNS] Unknown step token 'nope'.")

    def test_custom_run_step_labels_render_reset_specially(self):
        self.assertTrue(
            helper_custom_runs.is_custom_run_reset_step(helper_custom_runs.CUSTOM_RUN_RESET_STEP)
        )
        self.assertEqual(
            helper_custom_runs.custom_run_step_code(
                helper_custom_runs.CUSTOM_RUN_RESET_STEP,
                step_code_for_obj_fn=lambda step: "unused",
            ),
            "r",
        )
        self.assertEqual(
            helper_custom_runs.custom_run_step_name(helper_custom_runs.CUSTOM_RUN_RESET_STEP),
            "reset",
        )


if __name__ == "__main__":
    unittest.main()
