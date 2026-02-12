import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class TestTelemetryProcessStartGateSkipFind(unittest.TestCase):
    def test_find_wall_steps_do_not_require_start_gates(self):
        self.assertFalse(telemetry_process.step_requires_start_gates("FIND_WALL", {}))
        self.assertFalse(telemetry_process.step_requires_start_gates("FIND_WALL2", {}))

    def test_wait_for_start_gates_short_circuits_for_find_steps(self):
        status = telemetry_process.wait_for_start_gates(
            None,
            None,
            "FIND_WALL2",
            log=False,
        )
        self.assertEqual(status, "start")

    def test_non_find_steps_still_require_start_gates(self):
        self.assertTrue(telemetry_process.step_requires_start_gates("ALIGN_BRICK", {}))


if __name__ == "__main__":
    unittest.main()
