import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_stream_server import StreamServer


class TestHelperStreamServerStepSuccessCelebration(unittest.TestCase):
    def test_index_html_places_birdseye_above_half_size_camera_in_text_layout(self):
        server = StreamServer(
            frame_provider=lambda: None,
            text_provider=lambda: [{"text": "line1"}],
            img_width=1000,
        )

        html = server._index_html()

        self.assertLess(html.index("id='xyzLayout'"), html.index("id='mastLayout'"))
        self.assertLess(html.index("id='mastLayout'"), html.index("id='videoFeed'"))
        self.assertIn("id='xyzLayout' src='/xyz_workspace_live.svg' width=\"442\"", html)
        self.assertIn("id='mastLayout' src='/xyz_mast_live.svg' width=\"442\"", html)
        self.assertRegex(html, r"id='videoFeed' src='/video_feed\?sid=[^']+' width=\"260\"")
        self.assertIn("/xyz_mast_live.svg", html)
        self.assertIn("}, 250);", html)

    def test_index_html_places_birdseye_above_half_size_camera_without_text_layout(self):
        server = StreamServer(
            frame_provider=lambda: None,
            img_width=1000,
        )

        html = server._index_html()

        self.assertLess(html.index("id='xyzLayout'"), html.index("id='mastLayout'"))
        self.assertLess(html.index("id='mastLayout'"), html.index("img src='/video_feed'"))
        self.assertIn("id='xyzLayout' src='/xyz_workspace_live.svg' width=\"850\"", html)
        self.assertIn("id='mastLayout' src='/xyz_mast_live.svg' width=\"850\"", html)
        self.assertIn("img src='/video_feed' width=\"500\"", html)
        self.assertIn("}, 250);</script>", html)

    def test_mast_svg_route_serves_live_svg(self):
        server = StreamServer(
            frame_provider=lambda: None,
            xyz_workspace_getter=lambda: None,
        )
        client = server.app.test_client()
        response = client.get("/xyz_mast_live.svg")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/svg+xml", str(response.content_type))
        self.assertIn("<svg", response.get_data(as_text=True))
        response.close()

    def test_text_endpoint_includes_step_success_payload_from_dict_provider(self):
        server = StreamServer(
            frame_provider=lambda: None,
            text_provider=lambda: {
                "lines": [{"text": "line1"}],
                "step_success": {"seq": 7, "step": "SEAT_BRICK2", "at": 123.45},
            },
        )
        client = server.app.test_client()
        response = client.get("/text")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload.get("lines"), [{"text": "line1"}])
        step_success = payload.get("step_success") or {}
        self.assertEqual(step_success.get("seq"), 7)
        self.assertEqual(step_success.get("step"), "SEAT_BRICK2")
        self.assertAlmostEqual(float(step_success.get("at")), 123.45, places=2)
        response.close()

    def test_text_endpoint_keeps_backward_compat_for_list_provider(self):
        server = StreamServer(
            frame_provider=lambda: None,
            text_provider=lambda: [{"text": "legacy"}],
        )
        client = server.app.test_client()
        response = client.get("/text")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload.get("lines"), [{"text": "legacy"}])
        self.assertIsNone(payload.get("step_success"))
        response.close()

    def test_gong_route_serves_audio_when_file_exists(self):
        with tempfile.TemporaryDirectory() as td:
            gong_path = Path(td) / "gong.mp3"
            gong_path.write_bytes(b"fake-mp3-data")
            server = StreamServer(
                frame_provider=lambda: None,
                gong_file_path=str(gong_path),
            )
            client = server.app.test_client()
            response = client.get("/gong.mp3")
            self.assertEqual(response.status_code, 200)
            self.assertIn("audio/mpeg", str(response.content_type))
            self.assertGreater(len(response.data or b""), 0)
            response.close()

    def test_gong_route_returns_404_when_missing(self):
        server = StreamServer(
            frame_provider=lambda: None,
            gong_file_path="/tmp/nonexistent-gong-file-xyz.mp3",
        )
        client = server.app.test_client()
        response = client.get("/gong.mp3")
        self.assertEqual(response.status_code, 404)
        response.close()


if __name__ == "__main__":
    unittest.main()
