import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import debug_e2e_simulation as sim


class TestDebugE2ESimulation(unittest.TestCase):
    def test_simulation_runs_full_sequence_with_helper_usage(self):
        logs = sim.collect_simulation_logs(emit=False)
        self.assertTrue(logs)

        green_lines = [entry for entry in logs if entry.color == sim.COLOR_GREEN]
        red_lines = [entry for entry in logs if entry.color == sim.COLOR_RED]
        yellow_lines = [entry for entry in logs if entry.color == sim.COLOR_YELLOW]
        white_lines = [entry for entry in logs if entry.color == sim.COLOR_WHITE]

        self.assertGreaterEqual(len(green_lines), 2)
        self.assertGreaterEqual(len(white_lines), len(sim.DEFAULT_STEP_ORDER))
        self.assertGreaterEqual(len(yellow_lines), 0)
        self.assertEqual(len(red_lines), 0)

        step_lines = [entry for entry in logs if entry.text.startswith("Step ")]
        self.assertGreaterEqual(len(step_lines), len(sim.DEFAULT_STEP_ORDER))

        final = logs[-1]
        self.assertEqual(final.color, sim.COLOR_GREEN)
        self.assertIn("E2E preflight complete", final.text)

        helper_line = [entry for entry in logs if "helper_gate_utils" in entry.text and "helper_demo_log_utils" in entry.text]
        self.assertTrue(helper_line, "expected helper usage line in logs")

        self.assertTrue(sim.run_preflight(emit=False))


if __name__ == "__main__":
    unittest.main()
