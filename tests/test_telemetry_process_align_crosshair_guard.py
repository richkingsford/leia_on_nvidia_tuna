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
            "ALIGN_BRICK": {
                "align_policy": {
                    "target_lock_enabled": False,
                    "in_crosshairs_continuity_guard_enabled": True,
                    "in_crosshairs_continuity_guard_confirm_frames": 1,
                    "in_crosshairs_continuity_guard_reverse_max_acts": 3,
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -4.74, "tol": 1.4},
                    "yAxis_offset_abs": {"target": 2.5, "tol": 1.5},
                    "dist": {"target": 105.63, "tol": 1.5},
                },
                "start_gates": {
                    "visible": {"min": True},
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 117.5,
            "angle": 0.0,
            "offset_x": -5.2,
            "offset_y": 0.0,
            "x_axis": -5.2,
            "y_axis": 0.0,
            "inCrosshairs": True,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


def _gap_plan(cmd, score, correction_type):
    return {
        "planner": "gap",
        "cmd": str(cmd),
        "score": int(score),
        "speed": 0.0,
        "reason": f"{str(correction_type)}_alignment",
        "correction_type": str(correction_type),
    }


class TestTelemetryProcessAlignCrosshairGuard(unittest.TestCase):
    def _run_align_crosshair_case(self, plans):
        world = _DummyWorld()
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []
        planner_calls = {"n": 0}
        plan_list = list(plans)

        def _fake_update_world(_world, _vision, log=True, **_kwargs):
            _ = log
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["x_axis"] = -5.2
            _world.brick["offset_x"] = -5.2
            _world.brick["y_axis"] = 0.0
            _world.brick["offset_y"] = 0.0
            _world.brick["dist"] = 117.5
            if len(send_cmds) < 3:
                _world.brick["inCrosshairs"] = True
            elif len(send_cmds) < 6:
                _world.brick["inCrosshairs"] = False
            else:
                _world.brick["inCrosshairs"] = True

        def _fake_observe_success(*_args, **_kwargs):
            if planner_calls["n"] >= 3 and len(send_cmds) == 6:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_pre_action(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_planner(*_args, **_kwargs):
            idx = planner_calls["n"]
            planner_calls["n"] += 1
            if idx < len(plan_list):
                return dict(plan_list[idx])
            return dict(plan_list[-1])

        def _fake_send_robot_command(*args, **kwargs):
            cmd = str(args[3])
            send_cmds.append(cmd)
            return {
                "cmd_sent": cmd,
                "score_effective": kwargs.get("speed_score"),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success), \
             patch.object(telemetry_process, "pre_action_success_observation", side_effect=_fake_pre_action), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_planner), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", return_value=False), \
             patch.object(telemetry_process, "post_act_analysis", side_effect=_fake_update_world), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None), \
             patch.object(
                 builtins,
                 "print",
                 side_effect=lambda *args, **kwargs: print_lines.append(" ".join(str(arg) for arg in args)),
             ):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="ALIGN_BRICK",
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

        return ok, reason, send_cmds, print_lines

    def test_align_brick_crosshair_guard_does_not_recover_or_fail_on_false_crosshair(self):
        ok, reason, send_cmds, print_lines = self._run_align_crosshair_case(
            [_gap_plan("l", 12, "x_axis")],
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_cmds, ["l", "l", "l", "l", "l", "l"])
        self.assertFalse(any("inCrosshairs continuity guard:" in line for line in print_lines))
        self.assertFalse(any("post-recovery disqualified gap types" in line for line in print_lines))

    def test_align_brick_false_crosshair_keeps_using_planner_until_success_gate(self):
        ok, reason, send_cmds, print_lines = self._run_align_crosshair_case(
            [
                _gap_plan("l", 12, "x_axis"),
                _gap_plan("d", 9, "y_axis"),
                _gap_plan("f", 15, "distance"),
            ],
        )

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_cmds, ["l", "d", "f", "f", "f", "f"])
        self.assertFalse(any("Crosshair lock" in line for line in print_lines))
        self.assertTrue(
            all(cmd not in send_cmds for cmd in ("b", "u", "r"))
        )
