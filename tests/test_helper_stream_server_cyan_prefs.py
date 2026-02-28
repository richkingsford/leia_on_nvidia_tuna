import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_stream_server import StreamServer


class TestHelperStreamServerCyanPrefs(unittest.TestCase):
    def _build_server(self):
        state = {
            "vision_mode": "aruco",
            "cyan_profile": "config1_defaults",
            "cyan_visibility": "auto",
        }

        def get_mode():
            return state["vision_mode"]

        def set_mode(value):
            state["vision_mode"] = value

        def get_profile():
            return state["cyan_profile"]

        def set_profile(value):
            state["cyan_profile"] = value

        def get_visibility():
            return state["cyan_visibility"]

        def set_visibility(value):
            state["cyan_visibility"] = value

        server = StreamServer(
            frame_provider=lambda: None,
            vision_mode_getter=get_mode,
            vision_mode_setter=set_mode,
            vision_mode_options=[("aruco", "AruCo"), ("cyan", "Cyan")],
            cyan_profile_getter=get_profile,
            cyan_profile_setter=set_profile,
            cyan_profile_options=[("config1_defaults", "Defaults"), ("config8_responsive", "Responsive")],
            cyan_visibility_getter=get_visibility,
            cyan_visibility_setter=set_visibility,
            cyan_visibility_options=[("auto", "Auto"), ("cyan", "Cyan")],
        )
        return server, state

    def test_get_prefs_returns_cyan_and_legacy_keys(self):
        server, _state = self._build_server()
        client = server.app.test_client()
        response = client.get("/stream_prefs")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["cyan_profile"], "config1_defaults")
        self.assertEqual(payload["cyan_visibility"], "auto")
        self.assertEqual(payload["markerless_profile"], "config1_defaults")
        self.assertEqual(payload["markerless_visibility"], "auto")

    def test_post_accepts_legacy_markerless_keys(self):
        server, state = self._build_server()
        client = server.app.test_client()
        response = client.post(
            "/stream_prefs",
            json={
                "vision_mode": "yolo",
                "markerless_profile": "config8_responsive",
                "markerless_visibility": "cyan",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(state["vision_mode"], "cyan")
        self.assertEqual(state["cyan_profile"], "config8_responsive")
        self.assertEqual(state["cyan_visibility"], "cyan")


if __name__ == "__main__":
    unittest.main()
