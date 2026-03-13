import unittest

import telemetry_process


class TestTelemetryProcessMotionIntensity(unittest.TestCase):
    def setUp(self):
        self.orig_motion_fn = telemetry_process.telemetry_robot_module.speed_power_pwm_for_motion_intensity
        self.orig_send_pwm = telemetry_process.send_robot_command_pwm

    def tearDown(self):
        telemetry_process.telemetry_robot_module.speed_power_pwm_for_motion_intensity = self.orig_motion_fn
        telemetry_process.send_robot_command_pwm = self.orig_send_pwm

    def test_send_robot_command_uses_motion_intensity_for_mast_cmd(self):
        calls = {}

        def _fake_motion(cmd, intensity_pct):
            calls["motion"] = (cmd, intensity_pct)
            return 0.2, 40, 1, 800, 12.0

        def _fake_send_pwm(
            robot,
            world,
            step,
            cmd,
            power,
            pwm,
            duration_ms,
            **kwargs,
        ):
            calls["send_pwm"] = {
                "robot": robot,
                "world": world,
                "step": step,
                "cmd": cmd,
                "power": power,
                "pwm": pwm,
                "duration_ms": duration_ms,
                **kwargs,
            }
            return dict(calls["send_pwm"])

        telemetry_process.telemetry_robot_module.speed_power_pwm_for_motion_intensity = _fake_motion
        telemetry_process.send_robot_command_pwm = _fake_send_pwm

        result = telemetry_process.send_robot_command(
            robot=object(),
            world=object(),
            step=object(),
            cmd="d",
            speed=0.0,
            motion_intensity=12.0,
        )

        self.assertEqual(calls["motion"], ("d", 12.0))
        self.assertEqual(calls["send_pwm"]["cmd"], "d")
        self.assertEqual(calls["send_pwm"]["duration_ms"], 800)
        self.assertEqual(calls["send_pwm"]["motion_intensity_requested"], 12.0)
        self.assertEqual(calls["send_pwm"]["motion_intensity_effective"], 12.0)
        self.assertEqual(result["motion_intensity_requested"], 12.0)

    def test_send_robot_command_honors_duration_override_for_mast_motion_intensity(self):
        calls = {}

        def _fake_motion(cmd, intensity_pct):
            calls["motion"] = (cmd, intensity_pct)
            return 0.2, 40, 1, 800, 12.0

        def _fake_send_pwm(
            robot,
            world,
            step,
            cmd,
            power,
            pwm,
            duration_ms,
            **kwargs,
        ):
            calls["send_pwm"] = {
                "cmd": cmd,
                "duration_ms": duration_ms,
                **kwargs,
            }
            return dict(calls["send_pwm"])

        telemetry_process.telemetry_robot_module.speed_power_pwm_for_motion_intensity = _fake_motion
        telemetry_process.send_robot_command_pwm = _fake_send_pwm

        result = telemetry_process.send_robot_command(
            robot=object(),
            world=object(),
            step=object(),
            cmd="d",
            speed=0.0,
            motion_intensity=12.0,
            duration_override_ms=425,
        )

        self.assertEqual(calls["motion"], ("d", 12.0))
        self.assertEqual(calls["send_pwm"]["duration_ms"], 425)
        self.assertEqual(result["duration_ms"], 425)


if __name__ == "__main__":
    unittest.main()
