import builtins
import sys
import unittest
from pathlib import Path

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
            "SEAT_BRICK": {
                "success_gates": {
                    "visible": {"min": True},
                    "dist": {"target": 40.0, "tol": 1.5},
                },
                "progress_mast_exception": {
                    "enabled": True,
                    "operator_log_reminder": "Exception active: 25% progress -> D x4.",
                    "trigger_progress_fraction": 0.25,
                    "command": "d",
                    "score": 1,
                    "acts": 4,
                    "min_start_to_target_gap_mm": 1.0,
                },
            }
        }
        self.learned_rules = {}
        self.wall_envelope = None
        self.brick = {
            "visible": True,
            "dist": 100.0,
            "angle": 0.0,
            "offset_x": 0.0,
            "x_axis": 0.0,
            "y_axis": 0.0,
            "confidence": 95.0,
        }
        self._frame_id = 0
        self._success_confirm_frames = 0
        self._success_confirm_progress = None
        self._success_confirm_logged = False

    def update_from_motion(self, _evt):
        return None


class TestTelemetryProcessSeatBrickProgressException(unittest.TestCase):
    def test_seat_brick_runs_progress_mast_exception_once(self):
        world = _DummyWorld()
        robot = _DummyRobot()
        send_cmds = []
        print_lines = []
        update_calls = {"n": 0}
        gate_calls = {"n": 0}
        sleep_calls = []
        step_number = telemetry_process._step_number_for_label("SEAT_BRICK")

        def _fake_update_world(_world, _vision, log=True):
            _ = log
            update_calls["n"] += 1
            # First sample sets start dist. Second+ samples cross 25% progress
            # from 100 -> 40 target (threshold dist <= 85).
            _world.brick["visible"] = True
            _world.brick["dist"] = 100.0 if update_calls["n"] == 1 else 80.0
            _world._frame_id = int(getattr(_world, "_frame_id", 0)) + 1

        def _fake_observe_success_gatecheck(*_args, **_kwargs):
            gate_calls["n"] += 1
            if len(send_cmds) >= 4 and gate_calls["n"] >= 5:
                return {"success_met": True, "hold_for_confirm": False}
            return {"success_met": False, "hold_for_confirm": False}

        def _fake_send_robot_command(_robot, _world, _step, cmd, *_args, **_kwargs):
            send_cmds.append(str(cmd))
            return {
                "cmd_sent": str(cmd),
                "score_effective": 1,
                "power": 0.0,
                "pwm": 0,
                "duration_ms": 10,
            }

        orig_wait = telemetry_process.wait_for_start_gates
        orig_update = telemetry_process.update_world_from_vision
        orig_observe = telemetry_process.observe_success_gatecheck
        orig_select = telemetry_process.next_module.select_alignment_next_act
        orig_eval = telemetry_process.evaluate_gate_status
        orig_send = telemetry_process.send_robot_command
        orig_post = telemetry_process.post_act_analysis
        orig_success_bounds = telemetry_process.telemetry_brick.success_gate_bounds
        orig_sleep = telemetry_process.time.sleep
        orig_print = builtins.print
        try:
            telemetry_process.wait_for_start_gates = lambda *_a, **_k: "start"
            telemetry_process.update_world_from_vision = _fake_update_world
            telemetry_process.observe_success_gatecheck = _fake_observe_success_gatecheck
            telemetry_process.next_module.select_alignment_next_act = (
                lambda *_a, **_k: {"cmd": None, "speed": 0.0, "reason": "none", "score": None}
            )
            telemetry_process.evaluate_gate_status = lambda *_a, **_k: (False, 1.0)
            telemetry_process.send_robot_command = _fake_send_robot_command
            telemetry_process.post_act_analysis = lambda *_a, **_k: None
            telemetry_process.telemetry_brick.success_gate_bounds = lambda *_a, **_k: {}
            telemetry_process.time.sleep = lambda *args, **_k: sleep_calls.append(args[0] if args else None)
            builtins.print = lambda *args, **kwargs: print_lines.append(
                " ".join(str(arg) for arg in args)
            )

            ok, reason = telemetry_process.run_alignment_segment(
                segment={"events": []},
                step="SEAT_BRICK",
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
        finally:
            telemetry_process.wait_for_start_gates = orig_wait
            telemetry_process.update_world_from_vision = orig_update
            telemetry_process.observe_success_gatecheck = orig_observe
            telemetry_process.next_module.select_alignment_next_act = orig_select
            telemetry_process.evaluate_gate_status = orig_eval
            telemetry_process.send_robot_command = orig_send
            telemetry_process.post_act_analysis = orig_post
            telemetry_process.telemetry_brick.success_gate_bounds = orig_success_bounds
            telemetry_process.time.sleep = orig_sleep
            builtins.print = orig_print

        self.assertTrue(ok)
        self.assertEqual(reason, "success gate")
        self.assertEqual(send_cmds.count("d"), 4)
        self.assertTrue(
            any(
                (telemetry_process.COLOR_MAGENTA_BRIGHT in line)
                and (f"[step#{int(step_number)}]" in line)
                and ("Triggered at" in line)
                for line in print_lines
            )
        )
        self.assertIn(5.0, sleep_calls)


if __name__ == "__main__":
    unittest.main()
