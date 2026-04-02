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
            "POSITION_BRICK": {
                "align_policy": {
                    "target_lock_enabled": False,
                    "moving_lock": {
                        "enabled": True,
                        "axes": ["distance", "y_axis"],
                        "require_x_within_tol": True,
                        "entry_tol_add_mm": 1.0,
                        "other_axis_tol_add_mm": 1.0,
                        "pulse_duration_ms": 180,
                        "max_pulses_per_burst": 4,
                        "max_non_improving_fresh_frames": 2,
                        "min_improvement_mm": 0.15,
                    },
                },
                "start_gates": {
                    "visible": {"min": True},
                },
                "success_gates": {
                    "visible": {"min": True},
                    "xAxis_offset_abs": {"target": -4.74, "tol": 1.4},
                    "yAxis_offset_abs": {"target": 7.74, "tol": 2.3},
                    "dist": {"target": 111.45, "tol": 2.3},
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 111.40,
            "angle": 0.0,
            "offset_x": -4.80,
            "offset_y": 10.60,
            "x_axis": -4.80,
            "y_axis": 10.60,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class TestAlignMovingLockPlanning(unittest.TestCase):
    def test_plan_accepts_near_target_distance_when_x_in_gate_and_y_bplus(self):
        step_rules = {
            "align_policy": {
                "moving_lock": {
                    "enabled": True,
                    "axes": ["distance", "y_axis"],
                    "require_x_within_tol": True,
                    "entry_tol_add_mm": 1.0,
                    "other_axis_tol_add_mm": 1.0,
                    "pulse_duration_ms": 180,
                }
            }
        }
        local_gate = {
            "visible": True,
            "x_required": True,
            "x_within_tol": True,
            "x_abs_err": 0.45,
            "x_tol": 1.4,
            "y_required": True,
            "y_within_tol": False,
            "y_abs_err": 3.0,
            "y_tol": 2.3,
            "dist_required": True,
            "dist_within_tol": False,
            "dist_err": 2.9,
            "dist_tol": 2.3,
        }

        plan = telemetry_process.plan_align_moving_lock(
            local_gate,
            step_rules=step_rules,
            cmd="f",
            correction_type="distance",
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["axis"], "distance")
        self.assertEqual(plan["pulse_duration_ms"], 180)
        self.assertAlmostEqual(plan["entry_limit_mm"], 3.3)

    def test_plan_rejects_when_x_axis_not_in_gate(self):
        step_rules = {
            "align_policy": {
                "moving_lock": {
                    "enabled": True,
                    "axes": ["distance", "y_axis"],
                    "require_x_within_tol": True,
                }
            }
        }
        local_gate = {
            "visible": True,
            "x_required": True,
            "x_within_tol": False,
            "x_abs_err": 2.2,
            "x_tol": 1.4,
            "y_required": True,
            "y_within_tol": True,
            "y_abs_err": 0.4,
            "y_tol": 2.3,
            "dist_required": True,
            "dist_within_tol": False,
            "dist_err": 2.8,
            "dist_tol": 2.3,
        }

        plan = telemetry_process.plan_align_moving_lock(
            local_gate,
            step_rules=step_rules,
            cmd="f",
            correction_type="distance",
        )

        self.assertIsNone(plan)


class TestRunAlignmentSegmentMovingLock(unittest.TestCase):
    def test_position_brick_y_axis_moving_lock_chains_second_pulse_before_full_gatecheck(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        send_calls = []
        gatecheck_calls = []
        y_positions = iter([10.25, 9.95])

        def _fake_gap_plan(*_args, **_kwargs):
            return {
                "planner": "gap",
                "cmd": "d",
                "score": 1,
                "speed": 0.0,
                "reason": "y_axis_alignment",
                "correction_type": "y_axis",
            }

        def _fake_update_world(_world, _vision, log=True, **_kwargs):
            _ = log
            return None

        def _fake_observe_success(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_pre_action(*_args, **_kwargs):
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_send_robot_command(*args, **kwargs):
            cmd = str(args[3])
            duration_override_ms = kwargs.get("duration_override_ms")
            send_calls.append((cmd, duration_override_ms))
            return {
                "cmd_sent": cmd,
                "score_effective": kwargs.get("speed_score"),
                "score_model": kwargs.get("speed_score"),
                "power": 0.0,
                "pwm": 0,
                "duration_ms": int(duration_override_ms or 180),
                "duration_model_ms": int(duration_override_ms or 180),
            }

        def _fake_post_act_analysis(_world, _vision, step=None, log=True, include_pause=True, action_meta=None):
            _ = (step, log, include_pause, action_meta)
            _world._frame_id = int(getattr(_world, "_frame_id", 0) or 0) + 1
            _world.brick["visible"] = True
            _world.brick["dist"] = 111.40
            _world.brick["x_axis"] = -4.80
            _world.brick["offset_x"] = -4.80
            y_now = next(y_positions)
            _world.brick["y_axis"] = y_now
            _world.brick["offset_y"] = y_now
            return {
                "dist": _world.brick["dist"],
                "x_axis": _world.brick["x_axis"],
                "y_axis": _world.brick["y_axis"],
            }

        def _fake_run_full_gatecheck(*args, **kwargs):
            gatecheck_calls.append((args, kwargs))
            return True

        with patch.object(telemetry_process, "wait_for_start_gates", return_value="start"), \
             patch.object(telemetry_process, "update_world_from_vision", side_effect=_fake_update_world), \
             patch.object(telemetry_process, "observe_success_gatecheck", side_effect=_fake_observe_success), \
             patch.object(telemetry_process, "pre_action_success_observation", side_effect=_fake_pre_action), \
             patch.object(telemetry_process.next_module, "select_alignment_next_act", side_effect=_fake_gap_plan), \
             patch.object(telemetry_process, "send_robot_command", side_effect=_fake_send_robot_command), \
             patch.object(telemetry_process, "post_act_analysis", side_effect=_fake_post_act_analysis), \
             patch.object(telemetry_process, "run_full_gatecheck_after_act", side_effect=_fake_run_full_gatecheck), \
             patch.object(telemetry_process.telemetry_brick, "success_gate_bounds", return_value={}), \
             patch.object(telemetry_process.time, "sleep", return_value=None):
            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="POSITION_BRICK",
                robot=robot,
                vision=object(),
                world=world,
                steps=[],
                raw_steps=[],
                observer=None,
                analysis_pause_s=0.0,
                confirm_callback=None,
                align_silent=True,
            )

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_calls, [("d", 180), ("d", 180)])
        self.assertEqual(len(gatecheck_calls), 1)
        self.assertGreaterEqual(robot.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
