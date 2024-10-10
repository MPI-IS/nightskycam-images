"""
Tests for module: weather.
"""

import logging
from pathlib import Path
import tempfile
from typing import Dict, Generator, Set, Tuple, cast

import pytest
import toml
import tomli_w

from nightskycam_images.constants import DATE_FORMAT, WEATHER_SUMMARY_FILE_NAME
from nightskycam_images.weather import create_weather_summaries
from tests.conftest import FAKE, SEED

VALID_WEATHER_S: Set[str] = {
    "cloudy",
    "sunny",  # Complete match.
    "storm",  # Partial match for "thunderstorm".
}
INVALID_WEATHER_S: Set[str] = {"does_not_exist_1", "does_not_exist_2"}

# Number of date directories.
DATE_BATCH_SIZE: int = 2
# Count range for pairs of image and weather files.
COUNT_MIN: int = 1
COUNT_MAX: int = 3


@pytest.fixture
def setup_data(
    request: pytest.FixtureRequest,
) -> Generator[Tuple[Path, Dict[str, Dict[str, int]]], None, None]:
    """
    Set up temporary directory with pairs of image and weather files
    for tests.

    Yields
    ------
    Path
        Path to temporary directory
        that contains the date directories
        (containing the pairs of image and weather files).
    Dict
        Information on content of temporary directory.
        (Usable as ground-truth in tests.)
        - key:
            Name of date directory.
        - value:
            - key:
                Weather type.
            - value:
                Count.
    """
    # Create a temporary directory.
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Generate random content.
        date_dir_to_weather_dict = {
            # Name of date directory.
            FAKE.unique.date(pattern=DATE_FORMAT): {
                # Weather type: count.
                weather: FAKE.random_int(min=COUNT_MIN, max=COUNT_MAX)
                for weather in cast(
                    str,
                    FAKE.random_sample(  # Cast for mypy.
                        # Use union of both sets.
                        list(VALID_WEATHER_S | INVALID_WEATHER_S)
                    ),
                )
            }
            for _ in range(DATE_BATCH_SIZE)
        }

        for date_dir_name, weather_to_count in date_dir_to_weather_dict.items():

            # Directory for HD images.
            date_dir_path = Path(tmp_dir) / date_dir_name
            date_dir_path.mkdir()

            for weather, count in weather_to_count.items():

                for _ in range(count):
                    # Dummy unique stem for file name.
                    file_stem = FAKE.unique.user_name()

                    image_file_path = Path(tmp_dir) / date_dir_name / f"{file_stem}.jpg"
                    weather_file_path = (
                        Path(tmp_dir) / date_dir_name / f"{file_stem}.toml"
                    )

                    # Create dummy (empty) image file.
                    image_file_path.touch()
                    # Create weather file (toml format).
                    with open(weather_file_path, "wb") as f:
                        tomli_w.dump({"weather": weather}, f)

        yield Path(tmp_dir), date_dir_to_weather_dict

    # If there are failed tests.
    if request.session.testsfailed:
        # Log seed.
        logging.warning("Seed for RNG: %s", SEED)


def _check_weather_summary_file(
    tmp_dir_path: Path,
    summary_file_name: str,
    date_dir_to_weather_dict: Dict[str, Dict[str, int]],
) -> None:
    """
    Check integrity of weather summary file.

    Parameters
    ----------
    tmp_dir_path
        Path to temporary directory
        containing the date directories
        that contain the pairs of image and weather files.
    summary_file_name
        Name of weather summary file.
    date_dir_to_weather_dict
        Information on content of temporary directory.
        (Used as ground-truth.)
    """
    # For each date directory.
    for date_dir_name in date_dir_to_weather_dict.keys():
        date_dir_path = tmp_dir_path / date_dir_name

        # Check: Integrity of weather summary file.
        summary_file_path = date_dir_path / summary_file_name

        # Ground-truth.
        weather_to_count = date_dir_to_weather_dict[date_dir_name]

        # Check: Weather summary is a file.
        assert summary_file_path.is_file()

        # Parse weather summary file.
        parsed_summary = toml.load(summary_file_path)
        obs_weather_to_count = parsed_summary["weathers"]
        obs_skipped = parsed_summary["skipped"]

        # For each observed weather type.
        for obs_weather, obs_count in obs_weather_to_count.items():
            # Check: Mapping of weather type to count.
            assert obs_weather in weather_to_count.keys()
            assert obs_count == weather_to_count[obs_weather]
            # Check:
            # Number of input weather files that are in toml format
            # but have been skipped because of
            # NOT containing weather information.
            assert (
                obs_skipped == 0
            )  # TODO: Expand tests so that number of skipped is not always zero.


def test_create_weather_summaries(setup_data):
    """
    Test for function: weather.create_weather_summaries.
    """
    # Use fixture.
    tmp_dir_path, date_dir_to_weather_dict = setup_data

    # Argument for tested function.
    def _walk_folders() -> Generator[Path, None, None]:
        for date_dir_path in tmp_dir_path.iterdir():
            yield date_dir_path

    summary_file_name = WEATHER_SUMMARY_FILE_NAME

    # Tested function:
    # WITHOUT existing weather summary file.
    create_weather_summaries(_walk_folders, summary_file_name=summary_file_name)

    # Check: Integrity of weather summary file in all date directories.
    _check_weather_summary_file(
        tmp_dir_path, summary_file_name, date_dir_to_weather_dict
    )

    # Tested function:
    # with existing weather summary file (from previous run).
    create_weather_summaries(_walk_folders, summary_file_name=summary_file_name)

    # Check: Integrity of weather summary file in all date directories.
    _check_weather_summary_file(
        tmp_dir_path, summary_file_name, date_dir_to_weather_dict
    )
