"""Tests for WorldModel telemetry observation correction curves."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_world_with_curves(curves: dict) -> telemetry_robot.WorldModel:
    """Create a WorldModel with pre-set correction curves (bypasses file I/O)."""
    with patch("telemetry_robot.ROBOT_MODEL_FILE", new=Path("/nonexistent_file_for_test.json")):
        world = telemetry_robot.WorldModel()
    world._telemetry_correction_curves = dict(curves)
    return world


# ---------------------------------------------------------------------------
# Unit tests for _apply_telemetry_correction
# ---------------------------------------------------------------------------

class TestApplyTelemetryCorrection(unittest.TestCase):

    def test_no_curve_returns_original(self):
        world = _make_world_with_curves({})
        self.assertAlmostEqual(world._apply_telemetry_correction("dist", 300.0), 300.0)

    def test_dist_curve_applied(self):
        # expected = 0.7589 * observed + (-148.5877)
        world = _make_world_with_curves({"dist": {"slope": 0.7589, "intercept": -148.5877}})
        corrected = world._apply_telemetry_correction("dist", 300.0)
        self.assertAlmostEqual(corrected, 0.7589 * 300.0 + (-148.5877), places=4)

    def test_y_curve_applied(self):
        # expected = -1.276063 * observed + 66.699071
        world = _make_world_with_curves({"y": {"slope": -1.276063, "intercept": 66.699071}})
        corrected = world._apply_telemetry_correction("y", 52.0)
        self.assertAlmostEqual(corrected, -1.276063 * 52.0 + 66.699071, places=4)

    def test_unknown_variable_returns_original(self):
        world = _make_world_with_curves({"dist": {"slope": 0.5, "intercept": 10.0}})
        self.assertAlmostEqual(world._apply_telemetry_correction("y", 40.0), 40.0)


# ---------------------------------------------------------------------------
# Integration: update_vision applies curves when found=True
# ---------------------------------------------------------------------------

class TestUpdateVisionAppliesCorrections(unittest.TestCase):

    def _call_update_vision(self, world, *, dist, cam_h, found=True):
        """Call world.update_vision() with minimal mocking around the heavy
        downstream modules (brick/wall/xyz) so we only test the correction maths."""
        import telemetry_brick
        import telemetry_wall
        import helper_xyz_coords

        def _fake_update_from_vision(w, found, d, angle, conf, offset_x, ch, ba, bb, raw_dist=None):
            w._captured_dist = d
            w._captured_cam_h = ch
            w._captured_raw_dist = raw_dist
            return None

        with patch.object(telemetry_brick, "update_from_vision", side_effect=_fake_update_from_vision), \
             patch.object(telemetry_wall, "update_from_vision"), \
             patch.object(helper_xyz_coords, "sync_from_world"):
            world.update_vision(found, dist, 0.0, 90.0, offset_x=0, cam_h=cam_h)

    def test_dist_corrected_when_found(self):
        world = _make_world_with_curves({"dist": {"slope": 0.7589, "intercept": -148.5877}})
        self._call_update_vision(world, dist=300.0, cam_h=0.0)
        expected = 0.7589 * 300.0 + (-148.5877)
        self.assertAlmostEqual(world._captured_dist, expected, places=3)

    def test_y_corrected_when_found(self):
        world = _make_world_with_curves({"y": {"slope": -1.276063, "intercept": 66.699071}})
        self._call_update_vision(world, dist=200.0, cam_h=52.0)
        expected_y = -1.276063 * 52.0 + 66.699071
        self.assertAlmostEqual(world._captured_cam_h, expected_y, places=3)

    def test_raw_dist_preserved_when_dist_corrected(self):
        world = _make_world_with_curves({"dist": {"slope": 0.7589, "intercept": -148.5877}})
        self._call_update_vision(world, dist=300.0, cam_h=0.0)
        self.assertAlmostEqual(world._captured_raw_dist, 300.0, places=3)

    def test_no_correction_when_not_found(self):
        world = _make_world_with_curves({
            "dist": {"slope": 0.5, "intercept": 0.0},
            "y": {"slope": -1.0, "intercept": 0.0},
        })
        self._call_update_vision(world, dist=300.0, cam_h=50.0, found=False)
        # When not found, raw values pass through unchanged
        self.assertAlmostEqual(world._captured_dist, 300.0, places=3)
        self.assertAlmostEqual(world._captured_cam_h, 50.0, places=3)

    def test_no_curves_passthrough(self):
        world = _make_world_with_curves({})
        self._call_update_vision(world, dist=200.0, cam_h=30.0)
        self.assertAlmostEqual(world._captured_dist, 200.0, places=3)
        self.assertAlmostEqual(world._captured_cam_h, 30.0, places=3)


if __name__ == "__main__":
    unittest.main()
