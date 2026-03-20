import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import telemetry_process


class _DummyWorld:
    def __init__(self, process_steps):
        self.process_rules = process_steps
        self.brick = {"visible": False}
        self.wall_height_bricks = 1
        self.wall_height_mm = 44.0
        self.brick_supply_height_bricks = 1
        self.brick_supply_height_mm = 44.0
        self.learned_rules = {}
        self.wall_envelope = None
        self.last_visible_time = 0.0
        self._frame_id = 1
        self._gatecheck_mode = "traditional"

    def update_from_motion(self, evt):
        _ = evt


class _DummyRobot:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        return None


class _TailBurstObserverStop(Exception):
    def __init__(self, speed):
        super().__init__(speed)
        self.speed = float(speed)


class TestTelemetryProcessHeightIntelTailBurst(unittest.TestCase):
    def test_find_wall2_tail_phase_override_config(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_WALL2") if isinstance(steps, dict) else {}
        cfg = telemetry_process._height_intel_cfg_for_phase(step_cfg, phase="tail")

        self.assertEqual(cfg.get("apply_on_cmds"), ["d"])
        self.assertEqual(int(cfg.get("low_bricks_max") or 0), 20)
        self.assertEqual(int(cfg.get("high_bricks_min") or 0), 999)
        self.assertEqual(int(cfg.get("score") or 0), 20)
        self.assertTrue(bool(cfg.get("burst_acts_from_source")))
        self.assertFalse(bool(cfg.get("observe_between_acts", True)))
        self.assertTrue(bool(cfg.get("skip_success_gates")))
        self.assertTrue(bool(cfg.get("complete_after_burst")))

    def test_find_wall2_tail_uses_supply_count_burst_score_but_replay_stays_unmodified(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        world = _DummyWorld(steps or {})

        cmd_replay, score_replay, _note_replay, used_replay = telemetry_process._apply_height_intel_replay_override(
            world,
            "FIND_WALL2",
            "f",
            25,
            phase="replay",
            log=False,
        )
        self.assertFalse(bool(used_replay))
        self.assertEqual(cmd_replay, "f")
        self.assertEqual(int(score_replay), 25)

        setattr(world, "_height_intel_last_override_ts", {})
        cmd_tail, score_tail, _note_tail, used_tail = telemetry_process._apply_height_intel_replay_override(
            world,
            "FIND_WALL2",
            "d",
            25,
            phase="tail",
            log=False,
        )
        self.assertTrue(bool(used_tail))
        self.assertEqual(cmd_tail, "d")
        self.assertEqual(int(score_tail), 20)

    def test_find_wall2_tail_burst_plan_uses_supply_stack_count_and_skips_gates(self):
        world = _DummyWorld(
            {
                "FIND_WALL2": {
                    "controller": "replay",
                    "height_intelligence": {
                        "enabled": True,
                        "source": "brick_supply_height",
                        "apply_when_visible_false": True,
                        "apply_on_cmds": ["l", "r", "f", "b"],
                        "low_bricks_max": 2,
                        "high_bricks_min": 4,
                        "low_cmd": "d",
                        "high_cmd": "d",
                        "score": 1,
                        "phase_overrides": {
                            "tail": {
                                "apply_on_cmds": ["d"],
                                "low_bricks_max": 20,
                                "high_bricks_min": 999,
                                "score": 20,
                                "burst_acts_from_source": True,
                                "observe_between_acts": False,
                                "skip_success_gates": True,
                                "complete_after_burst": True,
                            }
                        },
                    },
                    "success_gates": {
                        "visible": {"min": True},
                    },
                }
            }
        )
        world.brick_supply_height_bricks = 3
        world.brick_supply_height_mm = 132.0
        plan = telemetry_process._height_intel_tail_burst_plan(
            world,
            "FIND_WALL2",
            "d",
            25,
        )

        self.assertTrue(bool(plan.get("applied")))
        self.assertEqual(plan.get("chosen_cmd"), "d")
        self.assertEqual(int(plan.get("score_out") or 0), 20)
        self.assertEqual(int(plan.get("burst_acts") or 0), 3)
        self.assertTrue(bool(plan.get("skip_success_gates")))
        self.assertTrue(bool(plan.get("complete_after_burst")))
        self.assertFalse(bool(plan.get("observe_between_acts", True)))

    def test_replay_tail_burst_observer_uses_sent_power_without_name_error(self):
        world = _DummyWorld(
            {
                "TEST_TAIL_BURST": {
                    "controller": "replay",
                    "height_intelligence": {
                        "enabled": True,
                        "phase_overrides": {
                            "tail": {
                                "burst_acts": 2,
                                "observe_between_acts": False,
                            }
                        },
                    },
                    "success_gates": {
                        "visible": {"min": True},
                    },
                }
            }
        )
        world.brick.update(
            {
                "dist": 100.0,
                "angle": 0.0,
                "offset_x": 0.0,
                "x_axis": 0.0,
                "confidence": 95.0,
            }
        )
        segment = {
            "events": [
                {"type": "action", "command": "forward", "speedScore": 1},
            ]
        }

        def _observer(*args):
            if len(args) >= 6 and args[5] == "height_intel_tail_burst":
                raise _TailBurstObserverStop(args[4])

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(telemetry_process, "wait_for_start_gates", return_value="start")
            )
            stack.enter_context(
                patch.object(telemetry_process, "update_world_from_vision", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "post_act_analysis", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(
                    telemetry_process,
                    "observe_success_gatecheck",
                    return_value={"success_met": False, "hold_for_confirm": False, "effective_success_ok": False},
                )
            )
            stack.enter_context(
                patch.object(
                    telemetry_process,
                    "pre_action_success_observation",
                    return_value={"success_met": False, "hold_for_confirm": False},
                )
            )
            stack.enter_context(
                patch.object(telemetry_process, "flush_auto_step_diagnostic", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "queue_auto_step_diagnostic", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "_success_gate_entries", return_value=[])
            )
            stack.enter_context(
                patch.object(telemetry_process, "_auto_gate_snapshot_text", return_value="")
            )
            stack.enter_context(
                patch.object(telemetry_process, "_capture_auto_diag_focus", return_value=None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "_capture_sent_action_snapshot", return_value=None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "_log_gatecheck_failure", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "_log_gatecheck_next_action", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "print_gate_summary_line", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "print_success_events", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "pause_after_exception", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(telemetry_process, "pause_after_fail", side_effect=lambda *a, **k: None)
            )
            stack.enter_context(
                patch.object(
                    telemetry_process,
                    "_height_intel_tail_burst_plan",
                    return_value={
                        "applied": True,
                        "burst_acts": 2,
                        "observe_between_acts": False,
                        "burst_pause_s": None,
                        "skip_success_gates": False,
                        "complete_after_burst": False,
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    telemetry_process,
                    "_apply_height_intel_replay_override",
                    side_effect=lambda world_obj, step, cmd, score, phase=None, log=True: (
                        cmd,
                        score,
                        "tail note",
                        phase == "tail",
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    telemetry_process,
                    "send_robot_command",
                    side_effect=lambda *a, **k: {
                        "power": 0.123,
                        "duration_ms": 100,
                        "score_effective": k.get("speed_score", 1),
                        "score_model": k.get("speed_score", 1),
                        "cmd_sent": a[3],
                    },
                )
            )
            stack.enter_context(patch.object(telemetry_process, "SUCCESS_SETTLE_S", 0.0))
            stack.enter_context(
                patch.object(telemetry_process.time, "sleep", side_effect=lambda *a, **k: None)
            )
            with self.assertRaises(_TailBurstObserverStop) as raised:
                telemetry_process.replay_segment(
                    segment=segment,
                    step="TEST_TAIL_BURST",
                    robot=_DummyRobot(),
                    vision=object(),
                    world=world,
                    observer=_observer,
                    align_silent=True,
                )

        self.assertAlmostEqual(float(raised.exception.speed), 0.123, places=6)


if __name__ == "__main__":
    unittest.main()
