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
        pipe0 = helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline("/dev/video0")
        pipe1 = helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline("/dev/video1")
        with patch.object(
            helper_camera_sources,
            "existing_camera_nodes",
            return_value=["/dev/video0", "/dev/video1"],
        ), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                helper_camera_sources.candidate_camera_sources(),
                [pipe0, "/dev/video0", 0, pipe1, "/dev/video1", 1],
            )

    def test_candidate_camera_sources_includes_explicit_overrides_first(self):
        pipe3 = helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline("/dev/video3")
        pipe9 = helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline("/dev/video9")
        pipe7 = helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline("/dev/video7")
        pipe0 = helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline("/dev/video0")
        pipe2 = helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline("/dev/video2")
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
                [
                    pipe3,
                    3,
                    pipe9,
                    "/dev/video9",
                    pipe7,
                    7,
                    pipe0,
                    "/dev/video0",
                    0,
                    pipe2,
                    "/dev/video2",
                    2,
                ],
            )

    def test_candidate_camera_sources_falls_back_to_indices_when_no_nodes_exist(self):
        argus = helper_camera_sources.build_nvidia_argus_gstreamer_pipeline()
        pipes = [
            helper_camera_sources.build_nvidia_v4l2_gstreamer_pipeline(f"/dev/video{idx}")
            for idx in range(4)
        ]
        with patch.object(
            helper_camera_sources,
            "existing_camera_nodes",
            return_value=[],
        ), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                helper_camera_sources.candidate_camera_sources(),
                [argus, pipes[0], 0, pipes[1], 1, pipes[2], 2, pipes[3], 3],
            )

    def test_candidate_camera_sources_can_return_legacy_sources_only(self):
        with patch.object(
            helper_camera_sources,
            "existing_camera_nodes",
            return_value=["/dev/video0", "/dev/video1"],
        ), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                helper_camera_sources.candidate_camera_sources(include_nvidia_pipelines=False),
                ["/dev/video0", 0, "/dev/video1", 1],
            )


if __name__ == "__main__":
    unittest.main()
