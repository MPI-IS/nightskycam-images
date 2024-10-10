"""
Tests for module: thumbnail.
"""

from pathlib import Path
from typing import Generator, Iterable

import cv2
import pytest

from nightskycam_images.constants import THUMBNAIL_DIR_NAME
from nightskycam_images.thumbnail import (
    _thumbnail_path,
    create_all_thumbnails,
    create_thumbnail,
    create_thumbnails,
)
from tests.conftest import FAKE

#
# fixture setup_data: see conftest.py in same directory
#


def test_create_thumbnail(setup_data):
    """
    Test for function: thumbnail.create_thumbnail.
    """

    # Use fixture.
    _, image_path_s, _, _ = setup_data

    for image_path in image_path_s:

        # Tested function.
        thumbnail_path = create_thumbnail(image_path)

        # Check: Path is file in thumbnails directory.
        assert thumbnail_path.is_file()
        assert thumbnail_path.parent.name == THUMBNAIL_DIR_NAME

        # Check: File can be parsed as image.
        thumbnail_array = cv2.imread(str(thumbnail_path))
        assert thumbnail_array is not None


def test_create_thumbnails(setup_data):
    """
    Test for function: thumbnail.create_thumbnails.
    """

    # Use fixture.
    _, image_path_s, _, _ = setup_data

    # Tested function.
    thumbnail_path_s = create_thumbnails(image_path_s)

    # Check:
    # Number of thumbnail images matches number of input HD images.
    assert len(image_path_s) == len(thumbnail_path_s)

    for thumbnail_path in thumbnail_path_s:
        # Check: Path is file in thumbnails directory.
        assert thumbnail_path.is_file()
        assert thumbnail_path.parent.name == THUMBNAIL_DIR_NAME

        # Check: File can be parsed as image.
        thumbnail_array = cv2.imread(str(thumbnail_path))
        assert thumbnail_array is not None


# Override parameter value for fixture.
@pytest.mark.parametrize(
    "setup_data",
    [
        {
            "date_dir_num": FAKE.random_int(min=0, max=10),
            "image_file_format": "jpg",
        },
        {
            "date_dir_num": FAKE.random_int(min=0, max=10),
            "image_file_format": "npy",
        },
    ],
    indirect=True,
)
def test_create_all_thumbnails(setup_data):

    # Use fixture.
    tmp_dir_path, image_path_s, _, image_file_format = setup_data

    # Argument for tested function.
    def _walk_folders() -> Generator[Path, None, None]:
        for path in tmp_dir_path.iterdir():
            yield path

    # Argument for tested function.
    def _list_images(folder: Path) -> Iterable[Path]:
        return list(folder.glob(f"*.{image_file_format}"))

    # Tested function.
    create_all_thumbnails(
        _walk_folders,
        _list_images,
    )

    for image_path in image_path_s:
        thumbnail_path = _thumbnail_path(image_path)

        # Check: Path is file in thumbnails directory.
        assert thumbnail_path.is_file()
        assert thumbnail_path.parent.name == THUMBNAIL_DIR_NAME

        # Check: File can be parsed as image.
        thumbnail_array = cv2.imread(str(thumbnail_path))
        assert thumbnail_array is not None
