import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import helper_stream_server
from helper_stream_server import StreamServer


class TestHelperStreamServerBrickShapePanel(unittest.TestCase):
    def test_brick_shape_panel_renders_face_cutouts_from_world_model_coords(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "world_model_brick.json"
            model_path.write_text(
                json.dumps(
                    {
                        "brick": {
                            "facePolygon": [
                                {"x": -10, "y": 0},
                                {"x": -10, "y": 8},
                                {"x": -2, "y": 12},
                                {"x": 2, "y": 12},
                                {"x": 10, "y": 8},
                                {"x": 10, "y": 0},
                                {"x": 3, "y": 0},
                                {"x": 0, "y": 4},
                                {"x": -3, "y": 0},
                            ],
                            "faceCutouts": [
                                [
                                    {"x": -2, "y": 5},
                                    {"x": 0, "y": 8},
                                    {"x": 2, "y": 5},
                                ]
                            ],
                            "faceLines": [
                                {
                                    "p1": {"x": -2, "y": 5},
                                    "p2": {"x": 0, "y": 8},
                                }
                            ],
                            "shapeGate": {
                                "mode": "negative_cutouts",
                            },
                        }
                    },
                    indent=2,
                )
                + "\n"
            )

            original_model_file = helper_stream_server.DEFAULT_BRICK_MODEL_FILE
            try:
                helper_stream_server.DEFAULT_BRICK_MODEL_FILE = model_path
                server = StreamServer(frame_provider=lambda: None)
                html = server._brick_shape_panel_html()
            finally:
                helper_stream_server.DEFAULT_BRICK_MODEL_FILE = original_model_file

        self.assertNotIn("Target Brick Face Shape", html)
        self.assertNotIn("source: world_model_brick.json", html)
        self.assertIn("aria-label='Brick face shape reference from world model coordinates'", html)
        self.assertGreaterEqual(html.count("<polygon"), 2)
        self.assertIn("fill='#11181d' stroke='#def7ff'", html)


if __name__ == "__main__":
    unittest.main()
