import threading
import unittest
from unittest.mock import patch

from calibration import helper_calibrate
import helper_calibrate_speed_curve


class HelperCalibrateSpeedCurveSharedStreamTests(unittest.TestCase):
    def test_run_interactive_session_exposes_shared_stream_runtime_to_selected_runner(self):
        shared_stream_state = {
            "lock": threading.Lock(),
            "frame": None,
            "text_lines": [],
        }
        observed = {}

        def _runner():
            observed["runtime"] = helper_calibrate.get_shared_stream_runtime()
            return 0

        selected_option = helper_calibrate_speed_curve.CalibrateOption(
            key="x",
            label="X-axis duration curve calibration",
            runner=_runner,
        )

        with patch.object(
            helper_calibrate_speed_curve,
            "_pick_interactive",
            side_effect=[selected_option, None],
        ), patch.object(
            helper_calibrate_speed_curve,
            "_interactive_trial_args",
            return_value=[],
        ), patch.object(
            helper_calibrate_speed_curve,
            "_restore_tty_line_input_mode",
        ), patch.object(
            helper_calibrate_speed_curve,
            "_print_stream_banner",
        ):
            exit_code = helper_calibrate_speed_curve.run_interactive_session(
                passthrough_args=[],
                show_banner=False,
                shared_stream_state=shared_stream_state,
                shared_stream_url="http://127.0.0.1:5000",
            )

        self.assertEqual(exit_code, 0)
        active_state, active_url = observed["runtime"]
        self.assertIs(active_state, shared_stream_state)
        self.assertEqual(active_url, "http://127.0.0.1:5000")
        restored_state, restored_url = helper_calibrate.get_shared_stream_runtime()
        self.assertIsNone(restored_state)
        self.assertIsNone(restored_url)


if __name__ == "__main__":
    unittest.main()
