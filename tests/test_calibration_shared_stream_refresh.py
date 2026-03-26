import threading
import unittest

import numpy as np

from calibration import helper_calibrate_x as calibrate_x


class _DummyVision:
    def __init__(self):
        self.current_frame = np.zeros((12, 16, 3), dtype=np.uint8)


class _DummyWorld:
    def __init__(self):
        self._xyz_workspace = {"robot": {"x_mm": 1.0}}


class CalibrationSharedStreamRefreshTests(unittest.TestCase):
    def test_refresh_stream_state_carries_xyz_workspace_for_shared_side_panel(self):
        stream_state = {
            "lock": threading.Lock(),
            "frame": None,
            "text_lines": [],
        }
        vision = _DummyVision()
        world = _DummyWorld()

        calibrate_x._refresh_stream_state(
            stream_state=stream_state,
            vision=vision,
            world=world,
            title_lines=["X calibration"],
        )

        with stream_state["lock"]:
            self.assertIsNotNone(stream_state.get("frame"))
            self.assertEqual(stream_state.get("text_lines", [None])[0], "X calibration")
            self.assertEqual(stream_state.get("xyz_workspace"), world._xyz_workspace)


if __name__ == "__main__":
    unittest.main()
