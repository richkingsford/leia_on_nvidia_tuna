import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_next
import telemetry_robot


class _DummyWorld:
    def __init__(self, x_axis, dist):
        self.brick = {
            "visible": True,
            "x_axis": float(x_axis),
            "offset_x": float(x_axis),
            "angle": 0.0,
            "dist": float(dist),
        }


@dataclass(frozen=True)
class _Scenario:
    name: str
    step: str
    x_axis: float
    dist: float
    expected_cmd: str
    expected_score: int
    x_target: float = 0.0
    x_tol: float = 0.7
    dist_target: float = 48.0
    dist_tol: float = 1.5


class TestHelperNextAutoSpeedIntelligenceMatrix(unittest.TestCase):
    def _rules_for(self, case: _Scenario):
        cfg = {
            "success_gates": {
                "xAxis_offset_abs": {
                    "target": float(case.x_target),
                    "tol": float(case.x_tol),
                },
                "dist": {
                    "target": float(case.dist_target),
                    "tol": float(case.dist_tol),
                },
            }
        }
        if case.step == "POSITION_BRICK":
            cfg["max_speed_score"] = 10
        return {case.step: cfg}

    def _dist_gap_mm(self, case: _Scenario) -> float:
        return max(0.0, abs(float(case.dist) - float(case.dist_target)) - float(case.dist_tol))

    def _scenarios(self):
        return [
            # LEFT TURN: close-to-gate and farther-error cases
            _Scenario("L align near 1mm -> 1%", "ALIGN_BRICK", -1.0, 48.0, "l", 1),
            _Scenario("L align edge 2mm -> 1%", "ALIGN_BRICK", -2.0, 48.0, "l", 1),
            _Scenario("L position 3mm near dist gate -> 1%", "POSITION_BRICK", -3.0, 48.0, "l", 1),
            _Scenario("L position edge 4mm near dist gate -> 1%", "POSITION_BRICK", -4.0, 48.0, "l", 1),
            _Scenario("L align 12mm near dist gate -> 2%", "ALIGN_BRICK", -12.0, 48.0, "l", 2),
            # RIGHT TURN: close-to-gate and farther-error cases
            _Scenario("R align near 1mm -> 1%", "ALIGN_BRICK", 1.0, 48.0, "r", 1),
            _Scenario("R align edge 2mm -> 1%", "ALIGN_BRICK", 2.0, 48.0, "r", 1),
            _Scenario("R position 3mm near dist gate -> 1%", "POSITION_BRICK", 3.0, 48.0, "r", 1),
            _Scenario("R position edge 4mm near dist gate -> 1%", "POSITION_BRICK", 4.0, 48.0, "r", 1),
            _Scenario("R align 60mm near dist gate -> 2%", "ALIGN_BRICK", 60.0, 48.0, "r", 2),
            # FORWARD: near, medium, and large distance gaps
            _Scenario("F position near gate 50mm -> 1%", "POSITION_BRICK", 0.0, 50.0, "f", 1),
            _Scenario("F position mid gate 70mm -> 5%", "POSITION_BRICK", 0.0, 70.0, "f", 5),
            _Scenario("F position far gate 95mm -> capped 10%", "POSITION_BRICK", 0.0, 95.0, "f", 10),
            _Scenario("F align near gate 50mm -> 1%", "ALIGN_BRICK", 0.0, 50.0, "f", 1),
            _Scenario("F align very far 220mm -> capped 20%", "ALIGN_BRICK", 0.0, 220.0, "f", 20),
            # BACKWARD: near, medium, and large distance gaps
            _Scenario("B position near gate 45mm -> 1%", "POSITION_BRICK", 0.0, 45.0, "b", 1),
            _Scenario(
                "B position target100 dist70 -> 5%",
                "POSITION_BRICK",
                0.0,
                70.0,
                "b",
                5,
                dist_target=100.0,
            ),
            _Scenario(
                "B position target200 dist150 -> capped 10%",
                "POSITION_BRICK",
                0.0,
                150.0,
                "b",
                10,
                dist_target=200.0,
            ),
            _Scenario("B align near gate 45mm -> 1%", "ALIGN_BRICK", 0.0, 45.0, "b", 1),
            _Scenario(
                "B align target200 dist150 -> capped 20%",
                "ALIGN_BRICK",
                0.0,
                150.0,
                "b",
                20,
                dist_target=200.0,
            ),
        ]

    def test_auto_decision_matrix_has_20_intelligent_lrfb_scenarios(self):
        scenarios = self._scenarios()
        self.assertEqual(len(scenarios), 20)

        cmd_counts = {"l": 0, "r": 0, "f": 0, "b": 0}
        near_turn_scores = []

        for case in scenarios:
            with self.subTest(case=case.name):
                world = _DummyWorld(case.x_axis, case.dist)
                analytics = helper_next.compute_alignment_analytics(
                    world,
                    self._rules_for(case),
                    learned_rules={},
                    step=case.step,
                    duration_s=0.05,
                )

                cmd = analytics.get("cmd")
                score = int(analytics.get("speed_score") or 0)
                expected_score = telemetry_robot.normalize_speed_score(case.expected_score)
                worst_metric = analytics.get("worst_metric")

                self.assertEqual(cmd, case.expected_cmd, msg=f"{case.name}: command mismatch")
                self.assertEqual(score, expected_score, msg=f"{case.name}: score mismatch")
                self.assertIn(cmd, cmd_counts, msg=f"{case.name}: unexpected command")
                cmd_counts[cmd] += 1

                expected_metric = "xAxis_offset_abs" if cmd in ("l", "r") else "dist"
                self.assertEqual(worst_metric, expected_metric, msg=f"{case.name}: trigger mismatch")

                if cmd in ("l", "r"):
                    x_gap = abs(float(case.x_axis) - float(case.x_target))
                    if x_gap <= 4.0:
                        near_turn_scores.append(score)
                else:
                    self.assertGreater(
                        self._dist_gap_mm(case),
                        0.0,
                        msg=f"{case.name}: drive command should have non-zero distance gap",
                    )

        # Ensure the matrix really covers all command types evenly.
        self.assertEqual(cmd_counts["l"], 5)
        self.assertEqual(cmd_counts["r"], 5)
        self.assertEqual(cmd_counts["f"], 5)
        self.assertEqual(cmd_counts["b"], 5)

        # Guardrail for your concern: near-gate turn cases must stay 1%/2%, never 3%+.
        self.assertTrue(near_turn_scores)
        self.assertTrue(all(score in (1, 2) for score in near_turn_scores))


if __name__ == "__main__":
    unittest.main()
