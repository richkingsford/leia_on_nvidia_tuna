import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import calibrate_dist_x_y_telemetry


class _DummyWorld:
    pass


class _DummyVision:
    def __init__(self):
        self.current_frame = np.zeros((12, 16, 3), dtype=np.uint8)


class TestCalibrateTelemetryStreaming(unittest.TestCase):
    def test_refresh_stream_state_populates_frame_and_lines(self):
        stream_state = {
            "lock": threading.Lock(),
            "frame": None,
            "text_lines": [],
        }
        vision = _DummyVision()
        world = _DummyWorld()

        calibrate_dist_x_y_telemetry._refresh_stream_state(
            stream_state=stream_state,
            vision=vision,
            world=world,
            variable="dist",
            rows_captured=2,
            points_total=5,
        )

        with stream_state["lock"]:
            self.assertIsNotNone(stream_state.get("frame"))
            lines = stream_state.get("text_lines", [])
            # draw_telemetry_overlay may append extra lines via line_sink;
            # the calibration header lines must appear first.
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0], "Telemetry calibration: dist")
            self.assertEqual(lines[1], "Points captured: 2/5")

    def test_refresh_stream_state_unlimited_points_label(self):
        stream_state = {
            "lock": threading.Lock(),
            "frame": None,
            "text_lines": [],
        }
        vision = _DummyVision()
        world = _DummyWorld()

        calibrate_dist_x_y_telemetry._refresh_stream_state(
            stream_state=stream_state,
            vision=vision,
            world=world,
            variable="y",
            rows_captured=4,
            points_total=None,
        )

        with stream_state["lock"]:
            lines = stream_state.get("text_lines", [])
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0], "Telemetry calibration: y")
            self.assertEqual(lines[1], "Points captured: 4/unlimited")


if __name__ == "__main__":
    unittest.main()