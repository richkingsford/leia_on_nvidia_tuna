from __future__ import annotations

import unittest

from direct_visual_servo import BrickMeasurement, ServoConfig, VisualServoController
from importlib.machinery import SourceFileLoader


HARNESS = SourceFileLoader(
    "direct_visual_servo_1s_test",
    "direct visual servo 1s test.py",
).load_module()


class DirectVisualServoTests(unittest.TestCase):
    def measurement(self, x=0.0, dist=150.0, timestamp=0.0, confidence=0.9):
        return BrickMeasurement(x, dist, confidence, timestamp)

    def test_requires_acquisition_and_reaches_tracking(self):
        c = VisualServoController()
        self.assertEqual(c.update(self.measurement(timestamp=0.0), 0.0).state, "ACQUIRING")
        self.assertEqual(c.update(self.measurement(timestamp=0.05), 0.05).state, "ACQUIRING")
        self.assertEqual(c.update(self.measurement(timestamp=0.10), 0.10).state, "TRACKING")

    def test_stale_measurement_stops(self):
        c = VisualServoController()
        result = c.update(self.measurement(timestamp=0.0), 1.0)
        self.assertEqual((result.left_pct, result.right_pct), (0.0, 0.0))
        self.assertEqual(result.rejection_reason, "stale")

    def test_ghost_jump_stops_and_resets_history(self):
        c = VisualServoController()
        for index in range(3):
            c.update(self.measurement(x=0.0, timestamp=index * 0.05), index * 0.05)
        result = c.update(self.measurement(x=500.0, timestamp=0.15), 0.15)
        self.assertEqual(result.rejection_reason, "x_jump")
        self.assertEqual((result.left_pct, result.right_pct), (0.0, 0.0))

    def test_aligned_region_is_zero(self):
        c = VisualServoController()
        for index in range(8):
            result = c.update(self.measurement(x=0.0, dist=100.0, timestamp=index * 0.05), index * 0.05)
        self.assertEqual(result.state, "ALIGNED")
        self.assertEqual((result.left_pct, result.right_pct), (0.0, 0.0))

    def test_saturation_is_bounded(self):
        c = VisualServoController()
        result = None
        for index in range(5):
            result = c.update(self.measurement(x=0.0, dist=1000.0, timestamp=index * 0.05), index * 0.05)
        self.assertLessEqual(max(abs(result.left_pct), abs(result.right_pct)), 35.0)

    def test_positive_camera_x_chooses_rightward_forward_arc(self):
        c = VisualServoController()
        result = None
        for index in range(15):
            result = c.update(self.measurement(x=20.0, dist=100.0, timestamp=index * 0.05), index * 0.05)
        # Positive x is image-right: left tread is outer/faster and both treads
        # remain forward, producing a rightward arc under Uno polarity.
        self.assertGreater(result.left_pct, result.right_pct)
        self.assertGreaterEqual(result.right_pct, 0.0)

    def test_mast_is_always_zero_on_wire(self):
        self.assertEqual(HARNESS.wire_packet(25.2, -31.7), b"<25,-32,0>")


if __name__ == "__main__":
    unittest.main()
