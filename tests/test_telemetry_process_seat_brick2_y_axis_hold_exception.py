import builtins
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _DummyWorld:
    def __init__(self):
        self.process_rules = {
            "SEAT_BRICK2": {
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "x_axis": {"target": 0.0, "tol": 1.0},
                    "y_axis": {"target": 1.0, "tol": 2.5},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
                "y_axis_hold_offset_exception": {
                    "enabled": True,
                    "operator_log_reminder": "Exception active: hold y-axis +7mm until end metrics pass.",
                    "hold_offset_mm": 7.0,
                    "release_when_metrics_within_tol": ["x_axis", "dist"],
                    "release_confirm_frames": 2,
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 70.0,
            "angle": 0.0,
            "offset_x": 4.0,
            "x_axis": 4.0,
            "y_axis": 8.0,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class _DummyPostMastWorld:
    def __init__(self):
        self.process_rules = {
            "SEAT_BRICK2": {
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "x_axis": {"target": 0.0, "tol": 1.0},
                    "y_axis": {"target": 1.0, "tol": 2.5},
                    "dist": {"target": 48.0, "tol": 1.5},
                },
                "post_success_mast_exception": {
                    "enabled": True,
                    "command": "d",
                    "score": 10,
                    "acts": 10,
                    "observe_between_acts": False,
                    "pause_s": 0.0,
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 48.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "y_axis": 1.0,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class TestSeatBrick2YAxisHoldException(unittest.TestCase):
    def test_y_axis_hold_offset_applies_then_releases_for_end_phase(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        planner_y_inputs = []
        print_lines = []
        update_calls = {"n": 0}
        observe_calls = {"n": 0}

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            update_calls["n"] += 1
            if update_calls["n"] <= 1:
                _world.brick["x_axis"] = 4.0
                _world.brick["dist"] = 70.0
            elif update_calls["n"] == 2:
                _world.brick["x_axis"] = 0.2
                _world.brick["dist"] = 48.4
            else:
                _world.brick["x_axis"] = 0.0
                _world.brick["dist"] = 48.0
            _world.brick["visible"] = True
            _world.brick["y_axis"] = 8.0
            _world._frame_id = int(getattr(_world, "_frame_id", 0)) + 1

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            observe_calls["n"] += 1
            if observe_calls["n"] >= 4:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_select_alignment_next_act(*_args, **kwargs):
            y_axis_mm = kwargs.get("y_axis_mm")
            planner_y_inputs.append(float(y_axis_mm) if y_axis_mm is not None else None)
            return {
                "planner": "gap",
                "cmd": None,
                "speed": 0.0,
                "reason": "all_gaps_within_gate",
                "score": None,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success_gatecheck), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_select_alignment_next_act), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="SEAT_BRICK2",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertGreaterEqual(len(planner_y_inputs), 3)
        self.assertAlmostEqual(planner_y_inputs[0], 1.0, places=3)
        self.assertAlmostEqual(planner_y_inputs[1], 1.0, places=3)
        self.assertAlmostEqual(planner_y_inputs[2], 8.0, places=3)
        self.assertTrue(
            any(
                (telemetry_process.COLOR_MAGENTA_BRIGHT in line)
                and ("Y-axis hold active" in line)
                for line in print_lines
            )
        )
        self.assertTrue(
            any(
                (telemetry_process.COLOR_MAGENTA_BRIGHT in line)
                and ("Y-axis hold released" in line)
                for line in print_lines
            )
        )

    def test_post_success_mast_exception_runs_d_10_percent_x10(self):
        world = _DummyPostMastWorld()
        robot = _DummyRobot()
        sent = []

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            _world.brick["visible"] = True
            _world._frame_id = int(getattr(_world, "_frame_id", 0)) + 1

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **kwargs):
            speed_score = kwargs.get("speed_score")
            sent.append((str(cmd), int(speed_score) if speed_score is not None else None))
            return {
                "cmd_sent": str(cmd),
                "score_effective": speed_score,
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(
                 telemetry_process,
                 "observe_success_gatecheck",
                 return_value={"success_met": True, "hold_for_confirm": False},
             ), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="SEAT_BRICK2",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=False,
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(sent, [("d", 10)] * 10)


if __name__ == "__main__":
    unittest.main()
