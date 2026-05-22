import json
import unittest
from pathlib import Path


class TestWorldModelBrickFaceShape(unittest.TestCase):
    def test_target_face_is_short_rectangle(self):
        model_path = Path(__file__).resolve().parents[1] / "world_model_brick.json"
        payload = json.loads(model_path.read_text())
        polygon = payload["brick"]["facePolygon"]

        self.assertEqual(
            polygon,
            [
                {"x": -26.5, "y": 0.0},
                {"x": -26.5, "y": 18.48},
                {"x": 26.5, "y": 18.48},
                {"x": 26.5, "y": 0.0},
            ],
        )
        width = float(polygon[2]["x"]) - float(polygon[0]["x"])
        height = float(polygon[1]["y"]) - float(polygon[0]["y"])
        self.assertAlmostEqual(width, 53.0)
        self.assertAlmostEqual(height, 21.0 * 0.88)


if __name__ == "__main__":
    unittest.main()
