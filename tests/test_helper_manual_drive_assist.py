import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_manual_drive_assist
import telemetry_robot


class TestHelperManualDriveAssist(unittest.TestCase):
    def test_repo_default_config_exposes_but_disables_low_speed_hotkey_assist(self):
        config = helper_manual_drive_assist.load_manual_drive_assist_config()

        self.assertFalse(bool(config.get("enabled")))
        self.assertEqual(tuple(config.get("hotkeys") or ()), ("r", "f", "q", "e"))
        self.assertEqual(tuple(config.get("cmds") or ()), ("f", "b", "l", "r"))
        self.assertEqual(int(config.get("max_score") or 0), 1)
        self.assertEqual(int(config.get("kick_score") or 0), 6)
        self.assertEqual(int(config.get("kick_duration_ms") or 0), 80)
        self.assertEqual(int(config.get("pause_after_kick_ms") or 0), 40)

    def test_load_manual_drive_assist_config_normalizes_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "world_model_robot.json"
            path.write_text(
                json.dumps(
                    {
                        helper_manual_drive_assist.ASSIST_CONFIG_KEY: {
                            "enabled": 0,
                            "hotkeys": [" R ", "F"],
                            "cmds": [" f ", "B"],
                            "max_score": "2",
                            "kick_score": "6",
                            "kick_duration_ms": "95",
                            "pause_after_kick_ms": "15",
                        }
                    }
                )
            )

            config = helper_manual_drive_assist.load_manual_drive_assist_config(path)

        self.assertFalse(bool(config.get("enabled")))
        self.assertEqual(tuple(config.get("hotkeys") or ()), ("r", "f"))
        self.assertEqual(tuple(config.get("cmds") or ()), ("f", "b"))
        self.assertEqual(int(config.get("max_score") or 0), 2)
        self.assertEqual(int(config.get("kick_score") or 0), 6)
        self.assertEqual(int(config.get("kick_duration_ms") or 0), 95)
        self.assertEqual(int(config.get("pause_after_kick_ms") or 0), 15)

    def test_build_plan_applies_only_to_target_hotkeys_and_scores(self):
        config = dict(helper_manual_drive_assist.DEFAULT_ASSIST_CONFIG)

        plan = helper_manual_drive_assist.build_manual_drive_assist_plan(
            hotkey="r",
            cmd="f",
            score=1,
            hold_duration_ms=250,
            config=config,
        )
        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["kick_score"], 6)
        self.assertEqual(plan["hold_score"], 1)
        self.assertEqual(plan["hold_duration_ms"], 250)

        self.assertIsNone(
            helper_manual_drive_assist.build_manual_drive_assist_plan(
                hotkey="w",
                cmd="f",
                score=1,
                hold_duration_ms=250,
                config=config,
            )
        )
        self.assertIsNone(
            helper_manual_drive_assist.build_manual_drive_assist_plan(
                hotkey="r",
                cmd="f",
                score=2,
                hold_duration_ms=250,
                config=config,
            )
        )

    def test_build_plan_falls_back_to_model_duration_when_hotkey_has_no_override_duration(self):
        expected_duration_ms = telemetry_robot.speed_power_pwm_for_cmd("b", 1)[3]

        plan = helper_manual_drive_assist.build_manual_drive_assist_plan(
            hotkey="f",
            cmd="b",
            score=1,
            hold_duration_ms=None,
            config=dict(helper_manual_drive_assist.DEFAULT_ASSIST_CONFIG),
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(int(plan["hold_duration_ms"]), int(expected_duration_ms))
        self.assertEqual(int(plan["kick_duration_ms"]), 80)

    def test_build_plan_applies_hotkey_override_from_config(self):
        config = dict(helper_manual_drive_assist.DEFAULT_ASSIST_CONFIG)
        config["hotkey_overrides"] = {"r": {"kick_score": 8}}

        plan = helper_manual_drive_assist.build_manual_drive_assist_plan(
            hotkey="r",
            cmd="f",
            score=1,
            hold_duration_ms=250,
            config=config,
        )

        self.assertIsInstance(plan, dict)
        self.assertEqual(int(plan["kick_score"]), 8)

    def test_format_manual_drive_assist_line_uses_compact_operator_text(self):
        line = helper_manual_drive_assist.format_manual_drive_assist_line(
            {
                "hotkey": "r",
                "cmd": "f",
                "kick_score": 5,
                "kick_duration_ms": 80,
                "pause_after_kick_ms": 40,
                "hold_score": 1,
                "hold_duration_ms": 250,
            }
        )

        self.assertEqual(
            line,
            "[HOTKEY ASSIST] R -> F: kick 5% for 80ms + pause 40ms, then hold 1% for 250ms.",
        )

    def test_execute_manual_drive_assist_plan_runs_kick_then_hold(self):
        send_calls = []
        sleep_calls = []

        def _fake_send_robot_command(
            robot,
            world,
            step_state,
            cmd,
            speed,
            speed_score=None,
            duration_override_ms=None,
            auto_mode=None,
            ease_in_out_enabled=None,
        ):
            send_calls.append(
                {
                    "kind": "score",
                    "cmd": cmd,
                    "score": speed_score,
                    "duration_ms": duration_override_ms,
                    "speed": speed,
                    "ease": ease_in_out_enabled,
                }
            )
            return {
                "cmd_sent": cmd,
                "power": 0.2 + (0.01 * int(speed_score or 0)),
                "pwm": 90 + int(speed_score or 0),
                "duration_ms": int(duration_override_ms or 0),
                "score_effective": speed_score,
            }

        def _fake_send_robot_command_pwm(*args, **kwargs):
            raise AssertionError("pwm override path should not be used in this test")

        result = helper_manual_drive_assist.execute_manual_drive_assist_plan(
            robot=object(),
            world=object(),
            step_state="MANUAL",
            hotkey="e",
            cmd="r",
            score=1,
            hold_duration_ms=250,
            config=dict(helper_manual_drive_assist.DEFAULT_ASSIST_CONFIG),
            send_robot_command_fn=_fake_send_robot_command,
            send_robot_command_pwm_fn=_fake_send_robot_command_pwm,
            sleep_fn=sleep_calls.append,
        )

        self.assertIsInstance(result, dict)
        self.assertEqual([call["score"] for call in send_calls], [6, 1])
        self.assertEqual(send_calls[0]["cmd"], "r")
        self.assertEqual(send_calls[1]["cmd"], "r")
        self.assertEqual(send_calls[0]["duration_ms"], 80)
        self.assertEqual(send_calls[1]["duration_ms"], 250)
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["score_model"], 6)
        self.assertEqual(result["segments"][1]["score_model"], 1)
        self.assertEqual(len(sleep_calls), 1)
        self.assertAlmostEqual(float(sleep_calls[0]), 0.12, places=6)


if __name__ == "__main__":
    unittest.main()
