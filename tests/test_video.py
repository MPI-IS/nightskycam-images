"""
Tests for module: video.
"""

from pathlib import Path
import tempfile

import numpy as np
import pytest

from nightskycam_images.convert_npy import npy_file_to_numpy
from nightskycam_images.video import (
    VideoFormat,
    _count_frames,
    _setup_image_array,
    _write_to_image,
    _write_video,
    create_video,
)


# Override parameter value for fixture.
@pytest.mark.parametrize(
    "setup_data", [{"date_dir_num": 1, "image_file_format": "npy"}], indirect=True
)
def test_write_to_image(setup_data):
    """
    Test for function: videos._write_to_image.
    """

    # Use fixture.
    _, image_path_s, text_s, _ = setup_data

    for image_path, text in zip(image_path_s, text_s):

        # Load pickled array from file.
        image_array = npy_file_to_numpy(image_path)

        # Before modification of image array by tested function:
        # Conserve copy for comparison.
        image_array_before = image_array.copy()

        # Tested function: write text to image.
        # -> Modifies image_array.
        _write_to_image(image_array, text, VideoFormat().text_format)

        # Check: image array changed.
        assert not np.array_equal(image_array, image_array_before)


def test_setup_image_array(setup_data):
    """
    Test for function: videos._setup_image_array.
    """

    # Use fixture.
    _, image_path_s, text_s, _ = setup_data

    # Get width and height of video format.
    video_format = VideoFormat()
    video_width, video_height = video_format.size

    # Test by setting up video image for each image
    for image_path, text in zip(image_path_s, text_s):
        # Tested function.
        image_array = _setup_image_array(image_path, text, video_format)

        # Check: Image shape corresponds to video format size.
        assert image_array.shape == (video_height, video_width, 3)


def test_count_frames(setup_data):
    """
    Test for function: videos._count_frames.
    """

    # Use fixture.
    _, image_path_s, text_s, _ = setup_data

    # Create a temporary video file.
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp_video:
        # Needed for tested function:
        # Set up video file.
        video_path = Path(tmp_video.name)
        _write_video(video_path, image_path_s, text_s)
        # Tested function.
        num_frames = _count_frames(video_path)
        # Check: Number of video frames matches number of input images.
        assert num_frames == len(image_path_s)


def test_write_video(setup_data):
    """
    Test for function: videos._write_video.
    """

    # Use fixture.
    _, image_path_s, text_s, _ = setup_data

    # Create a temporary video file.
    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp_video:
        video_path = Path(tmp_video.name)
        _write_video(video_path, image_path_s, text_s)
        # Check: Video path points to a file.
        assert video_path.is_file()


def test_create_video(setup_data):
    """
    Test for function: videos.create_video.
    """

    # Use fixture.
    _, image_path_s, text_s, _ = setup_data

    with tempfile.NamedTemporaryFile(suffix=".webm") as tmp_video:
        video_path = Path(tmp_video.name)
        create_video(video_path, image_path_s, text_s)
        # Check: Video path points to a file.
        assert video_path.is_file()
