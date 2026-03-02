import sys
import threading
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_streaming import build_text_provider


class TestHelperStreamingTextProvider(unittest.TestCase):
    def test_build_text_provider_returns_lines_and_step_success(self):
        state = {
            "text_lines": [{"text": "hello"}],
            "step_success_seq": 3,
            "step_success_step": "ALIGN_BRICK",
            "step_success_at": 111.25,
        }
        provider = build_text_provider(state)
        payload = provider()
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("lines"), [{"text": "hello"}])
        step_success = payload.get("step_success") or {}
        self.assertEqual(step_success.get("seq"), 3)
        self.assertEqual(step_success.get("step"), "ALIGN_BRICK")
        self.assertEqual(step_success.get("at"), 111.25)

    def test_build_text_provider_with_lock_reads_consistent_values(self):
        lock = threading.Lock()
        state = {
            "lock": lock,
            "text_lines": [{"text": "line"}],
            "step_success_seq": 1,
            "step_success_step": "FIND_BRICK",
            "step_success_at": 10.0,
        }
        provider = build_text_provider(state)
        with lock:
            state["text_lines"] = [{"text": "updated"}]
            state["step_success_seq"] = 2
            state["step_success_step"] = "SEAT_BRICK2"
            state["step_success_at"] = 20.0
        payload = provider()
        self.assertEqual(payload.get("lines"), [{"text": "updated"}])
        step_success = payload.get("step_success") or {}
        self.assertEqual(step_success.get("seq"), 2)
        self.assertEqual(step_success.get("step"), "SEAT_BRICK2")
        self.assertEqual(step_success.get("at"), 20.0)


if __name__ == "__main__":
    unittest.main()

