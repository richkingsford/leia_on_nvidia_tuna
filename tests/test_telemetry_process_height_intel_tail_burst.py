import sys
import unittest
from pathlib import Path

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


class TestTelemetryProcessHeightIntelTailBurst(unittest.TestCase):
    def test_find_wall2_tail_phase_override_config(self):
        model = telemetry_process.load_process_model()
        steps = (model or {}).get("steps") if isinstance(model, dict) else {}
        step_cfg = (steps or {}).get("FIND_WALL2") if isinstance(steps, dict) else {}
        cfg = telemetry_process._height_intel_cfg_for_phase(step_cfg, phase="tail")

        self.assertEqual(int(cfg.get("score") or 0), 100)
        self.assertTrue(bool(cfg.get("allow_uncapped_score")))
        self.assertEqual(int(cfg.get("burst_acts") or 0), 8)
        self.assertFalse(bool(cfg.get("observe_between_acts", True)))

    def test_find_wall2_tail_uses_uncapped_100_but_replay_stays_1(self):
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
        self.assertTrue(bool(used_replay))
        self.assertEqual(cmd_replay, "d")
        self.assertEqual(int(score_replay), 1)

        setattr(world, "_height_intel_last_override_ts", {})
        cmd_tail, score_tail, _note_tail, used_tail = telemetry_process._apply_height_intel_replay_override(
            world,
            "FIND_WALL2",
            "f",
            25,
            phase="tail",
            log=False,
        )
        self.assertTrue(bool(used_tail))
        self.assertEqual(cmd_tail, "d")
        self.assertEqual(int(score_tail), 100)


if __name__ == "__main__":
    unittest.main()
