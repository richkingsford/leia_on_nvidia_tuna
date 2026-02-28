import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyCyanVision:
    pass


class TestTelemetryProcessVisibilityRecheckHold(unittest.TestCase):
    def test_visibility_recheck_hold_seconds_keeps_aruco_timing(self):
        aruco = object.__new__(telemetry_process.ArucoBrickVision)
        hold_s = telemetry_process.visibility_recheck_hold_seconds(
            aruco,
            base_hold_s=0.5,
            control_dt=0.1,
        )
        expected = (
            max(0.1, 0.5)
            * float(telemetry_process.ALIGN_RECOVERY_PREMOVE_RECHECK_HOLD_MULTIPLIER)
        )
        self.assertAlmostEqual(hold_s, expected, places=6)

    def test_visibility_recheck_hold_seconds_triples_for_cyan(self):
        cyan = _DummyCyanVision()
        aruco = object.__new__(telemetry_process.ArucoBrickVision)
        hold_aruco = telemetry_process.visibility_recheck_hold_seconds(
            aruco,
            base_hold_s=0.5,
            control_dt=0.1,
        )
        hold_cyan = telemetry_process.visibility_recheck_hold_seconds(
            cyan,
            base_hold_s=0.5,
            control_dt=0.1,
        )
        self.assertAlmostEqual(
            hold_cyan,
            hold_aruco * float(telemetry_process.ALIGN_RECOVERY_PREMOVE_RECHECK_CYAN_HOLD_MULTIPLIER),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
