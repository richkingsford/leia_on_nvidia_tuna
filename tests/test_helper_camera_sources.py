import os
import unittest
from unittest.mock import patch

import helper_camera_sources


class HelperCameraSourcesTests(unittest.TestCase):
    def test_existing_camera_nodes_sorted_numerically(self):
        with patch.object(
            helper_camera_sources.glob,
            "glob",
            return_value=[
                "/dev/video10",
                "/dev/video2",
                "/dev/video1",
                "/dev/not-a-camera",
            ],
        ):
            self.assertEqual(
                helper_camera_sources.existing_camera_nodes(),
                ["/dev/video1", "/dev/video2", "/dev/video10"],
            )
            self.assertEqual(
                helper_camera_sources.existing_camera_indices(),
                [1, 2, 10],
            )

    def test_candidate_camera_sources_prefers_detected_nodes_over_blind_fallback(self):
        with patch.object(
            helper_camera_sources,
            "existing_camera_nodes",
            return_value=["/dev/video0", "/dev/video1"],
        ), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                helper_camera_sources.candidate_camera_sources(),
                ["/dev/video0", 0, "/dev/video1", 1],
            )

    def test_candidate_camera_sources_includes_explicit_overrides_first(self):
        with patch.object(
            helper_camera_sources,
            "existing_camera_nodes",
            return_value=["/dev/video0", "/dev/video2"],
        ), patch.dict(
            os.environ,
            {
                "LEIA_CAMERA_SOURCE": "/dev/video9",
                "LEIA_CAMERA_INDEX": "7",
            },
            clear=True,
        ):
            self.assertEqual(
                helper_camera_sources.candidate_camera_sources(preferred_index=3),
                [3, "/dev/video9", 7, "/dev/video0", 0, "/dev/video2", 2],
            )

    def test_candidate_camera_sources_falls_back_to_indices_when_no_nodes_exist(self):
        with patch.object(
            helper_camera_sources,
            "existing_camera_nodes",
            return_value=[],
        ), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                helper_camera_sources.candidate_camera_sources(),
                [0, 1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
