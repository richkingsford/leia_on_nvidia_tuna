import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_auto_step_cli
from telemetry_robot import StepState


class TestHelperAutoStepCli(unittest.TestCase):
    def test_resolve_step_argument_supports_numeric_step_code(self):
        resolved = helper_auto_step_cli.resolve_step_argument("7")
        self.assertEqual(resolved, StepState.ALIGN_BRICK)


if __name__ == "__main__":
    unittest.main()
