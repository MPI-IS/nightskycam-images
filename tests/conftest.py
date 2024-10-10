"""
Fixtures shared by multiple test modules.
"""

import logging
from pathlib import Path
import tempfile
import time
from typing import Generator, List, Tuple

import cv2
from faker import Faker
import numpy as np
import pytest

from nightskycam_images.constants import DATE_FORMAT

# Number of image files for tests.
IMAGE_BATCH_SIZE: int = 5


FAKE = Faker()
# Seed random generator.
SEED = int(time.time() * 1000)  # Current time in ms.
FAKE.seed_instance(SEED)


# Default parameter values for fixture.
@pytest.fixture(
    params=[
        {"date_dir_num": 1, "image_file_format": "npy"},
        {"date_dir_num": 1, "image_file_format": "jpg"},
    ]
)
def setup_data(
    request: pytest.FixtureRequest,
) -> Generator[Tuple[Path, List[Path], List[str], str], None, None]:
    """
    Set up temporary directory with image files for tests.

    Parameters
    ----------
    request
        Use `request.param` to pass parameters to this fixture:
        - date_dir_num:
            Number of date directories that will contain image files.
        - image_file_format:
            Format of the image files that will be set up.

    Yields
    ------
    Path
        Path to temporary directory
        that contains the date directories (containing the image files).
    List[Path]
        Image file paths.
    List[str]
        Dummy text for each image.
    str
        Image file format.
    """
    # Parameters for fixture.
    date_dir_num = request.param["date_dir_num"]
    image_file_format = request.param["image_file_format"]

    # Initialise.
    image_path_s = []
    text_s = []

    # Create a temporary directory.
    with tempfile.TemporaryDirectory() as tmp_dir:

        for _ in range(date_dir_num):
            date_dir_name = FAKE.unique.date(pattern=DATE_FORMAT)

            # Directory for HD images.
            date_dir_path = Path(tmp_dir) / date_dir_name
            date_dir_path.mkdir()

            # Generate numpy images of random sizes.
            for i in range(IMAGE_BATCH_SIZE):
                image_array = np.random.randint(
                    0,
                    255,
                    (i + 1 + 480, i + 1 + 360, 3),  # (height, width, RGB)
                    dtype=np.uint8,
                )

                # Pickled numpy array format.
                if image_file_format == "npy":
                    # Pickle numpy array to file.
                    image_path = date_dir_path / f"image_{i}.npy"
                    np.save(image_path, image_array)

                # jpg format.
                elif image_file_format == "jpg":
                    # Convert numpy array to image.
                    image_path = date_dir_path / f"image_{i}.{image_file_format}"
                    cv2.imwrite(str(image_path), image_array)

                else:
                    raise ValueError(
                        f"image_file_format was set to '{image_file_format}', "
                        f"but should be either 'npy' or 'jpg' instead."
                    )

                # Update.
                image_path_s.append(image_path)

                # Generate dummy text for each image.
                text_s.append(f"text_{i}")

        yield Path(tmp_dir), image_path_s, text_s, image_file_format

    # If there are failed tests.
    if request.session.testsfailed:
        # Log seed.
        logging.warning("Seed for RNG: %s", SEED)
