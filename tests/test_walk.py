"""
Tests for module: walk.
"""

import datetime as dt
import logging
from pathlib import Path
import tempfile
from typing import Dict, Generator, Set, Tuple

import pytest

from nightskycam_images import walk
from nightskycam_images.constants import (
    DATE_FORMAT,
    DATE_FORMAT_FILE,
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
    TIME_FORMAT,
    TIME_FORMAT_FILE,
    VIDEO_FILE_NAME,
    ZIP_DIR_NAME,
)
from tests.conftest import FAKE, SEED

# Number of different system names.
SYSTEM_BATCH_SIZE: int = 2
# Number of different dates for system.
DATE_BATCH_SIZE: int = 3
# Number of different times for date.
TIME_BATCH_SIZE: int = 2
# Date is at most ~2 months in the past,
# so that it is likely that there are dates of the same year and month
# for the set DATE_BATCH_SIZE.
DATE_MAX_DAYS_AGO: int = 60  # Not counting today.


@pytest.fixture
def setup_tmp_media(
    request: pytest.FixtureRequest,
) -> Generator[Tuple[Path, Dict[str, Dict[str, Set[str]]]], None, None]:
    """
    Set up temporary media directory with content for tests.

    Yields
    ------
    Path
        Path to temporary media directory.
    Dict
        Information on content of media directory.
        (Usable as ground-truth in tests.)
        - key: str
            system name
        - value: Dict
            - key: str
                date
            - value: Set[str]
                set of time values
    """
    # Reset memory of already generated fake values
    # (used for ensuring uniqueness).
    # Necessary for using uniqueness in combination with
    # pytest.mark.parametrize.
    FAKE.unique.clear()

    # Set up temporary media directory.
    with tempfile.TemporaryDirectory() as tmp_media_dir_strpath:

        # Cast string to Path object.
        tmp_media_dir_path = Path(tmp_media_dir_strpath)

        # Generate random names for content (directories and files)
        # of media directory.
        system_to_dict = {
            # System name.
            FAKE.unique.user_name(): {
                # Date (string) between today and maximum days ago.
                FAKE.unique.date_between(
                    dt.timedelta(days=-DATE_MAX_DAYS_AGO)
                ).strftime(DATE_FORMAT_FILE): {
                    # Time (string).
                    FAKE.unique.time(pattern=TIME_FORMAT_FILE)
                    for _ in range(TIME_BATCH_SIZE)
                }
                for _ in range(DATE_BATCH_SIZE)
            }
            for _ in range(SYSTEM_BATCH_SIZE)
        }

        # Set up content of temporary media directory.
        for system, date_to_time_s in system_to_dict.items():
            for date_str, time_str_s in date_to_time_s.items():
                # Set up directory structure.
                #
                # Directory for HD images.
                date_dir_path = tmp_media_dir_path / system / date_str
                date_dir_path.mkdir(parents=True)
                # Directory for thumbnail images.
                thumbnail_dir_path = date_dir_path / THUMBNAIL_DIR_NAME
                thumbnail_dir_path.mkdir()

                # Set up dummy files in directories.
                #
                # Thumbnail video.
                (thumbnail_dir_path / VIDEO_FILE_NAME).touch()
                #
                for time_str in time_str_s:
                    # Shared stem of file names.
                    filename_stem = f"{system}_{date_str}_{time_str}"
                    # HD image file.
                    (date_dir_path / f"{filename_stem}.npy").touch()
                    # Thumbnail image file.
                    (
                        thumbnail_dir_path / f"{filename_stem}.{THUMBNAIL_FILE_FORMAT}"
                    ).touch()
                    # Meta data file.
                    (date_dir_path / f"{filename_stem}.toml").touch()

        yield tmp_media_dir_path, system_to_dict

    # If there are failed tests.
    if request.session.testsfailed:
        # Log seed.
        logging.warning("Seed for RNG: %s", SEED)


def test_walk_systems(setup_tmp_media):
    """
    Test for function: walk.walk_systems.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    system_s = system_to_dict.keys()

    # Tested function.
    obs_system_path_s = list(walk.walk_systems(tmp_media_path))

    # Check: Number of systems.
    assert len(obs_system_path_s) == len(system_s)

    for obs_system_path in obs_system_path_s:
        # Check: System path is a directory inside of media directory.
        assert obs_system_path.parent == tmp_media_path
        assert obs_system_path.is_dir()
        # Check: System name.
        obs_system = str(obs_system_path.name)
        assert obs_system in system_to_dict.keys()


def test_get_system_path(setup_tmp_media):
    """
    Test for function: walk.get_system_path.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    for system in system_to_dict.keys():
        # Tested function.
        obs_system_path = walk.get_system_path(tmp_media_path, system)
        # Check: System path is a directory inside of media directory.
        assert obs_system_path.parent == tmp_media_path
        assert obs_system_path.is_dir()
        # Check: System name.
        assert obs_system_path.name == system


@pytest.mark.parametrize(
    "param_within_nb_days, exp_date_count",
    [
        (None, DATE_BATCH_SIZE),
        (DATE_MAX_DAYS_AGO + 1, DATE_BATCH_SIZE),  # +1: Count today as well.
        (0, 0),
    ],
)
def test_walk_dates(setup_tmp_media, param_within_nb_days, exp_date_count):
    """
    Test for function: walk.walk_dates.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    for system, date_to_time_s in system_to_dict.items():

        date_str_s = date_to_time_s.keys()

        # Parameter for tested function.
        system_path = walk.get_system_path(tmp_media_path, system)

        # Tested function.
        obs_date_and_path_s = list(
            walk.walk_dates(system_path, within_nb_days=param_within_nb_days)
        )

        # Check: Number of date instances/paths.
        obs_date_count = len(obs_date_and_path_s)
        assert obs_date_count == exp_date_count

        for obs_date, obs_path in obs_date_and_path_s:
            # Check: Date.
            obs_date_str = obs_date.strftime(DATE_FORMAT_FILE)
            assert obs_date_str in date_str_s
            # Check: Date path is a directory inside of system directory.
            assert obs_path.parent == system_path
            assert obs_path.is_dir()


def test_walk_all(setup_tmp_media):
    """
    Test for function: walk.walk_all.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    # Ground-truth.
    system_s = system_to_dict.keys()
    # Number of date directory paths.
    path_count = sum(len(date_to_time) for date_to_time in system_to_dict.values())

    # Tested function.
    obs_path_s = list(walk.walk_all(tmp_media_path))

    # Check: Number of date directory paths.
    assert len(obs_path_s) == path_count

    for obs_path in obs_path_s:
        # Check: Date path is a directory inside of system directory.
        assert obs_path.parent.name in system_s
        assert obs_path.is_dir()


def test_get_images_folder(setup_tmp_media):
    """
    Test for function: walk.get_images_folder.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    for system, date_to_time_s in system_to_dict.items():
        for date_str in date_to_time_s.keys():

            # Tested function.
            date = dt.datetime.strptime(date_str, DATE_FORMAT_FILE).date()
            obs_date_path = walk.get_images_folder(tmp_media_path, system, date)

            # Check: Date path is a directory inside of system directory.
            assert obs_date_path.parent.name == system
            assert obs_date_path.is_dir()
            # Check: Name of date directory.
            assert obs_date_path.name == date_str


def test_get_ordered_dates(setup_tmp_media):
    """
    Test for function: walk.get_ordered_dates.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    for system in system_to_dict.keys():

        # Ground-truth:
        #
        # Total count of dates for system.
        date_to_time_s = system_to_dict[system]
        date_count = len(date_to_time_s.keys())
        # Dates.
        date_s = [
            dt.datetime.strptime(date_str, DATE_FORMAT_FILE)
            for date_str in date_to_time_s.keys()
        ]
        year_s = [date.year for date in date_s]

        # Tested function.
        obs_year_to_month_dict = walk.get_ordered_dates(tmp_media_path, system)

        # Test result:
        # Total count of dates for system.
        obs_date_count = 0

        for obs_year, obs_month_dict in obs_year_to_month_dict.items():

            # Check: Year.
            assert obs_year in year_s

            # Ground-truth.
            month_s = [date.month for date in date_s if date.year == obs_year]

            for obs_month, obs_date_and_path_s in obs_month_dict.items():

                # Check: Month.
                assert obs_month in month_s

                obs_date_count += len(obs_date_and_path_s)

                obs_date_s = [obs_date for obs_date, _ in obs_date_and_path_s]

                # Compare adjacent date values in list.
                for obs_date_current, obs_date_next in zip(
                    obs_date_s[:-1], obs_date_s[1:]
                ):
                    # Check: Dates are in non-descending order.
                    assert obs_date_current < obs_date_next

                for _, obs_date_path in obs_date_and_path_s:
                    # Check:
                    # Date path is a directory inside of system directory.
                    assert obs_date_path.parent.name == system
                    assert obs_date_path.is_dir()

        # Check: Number of dates.
        assert obs_date_count == date_count


def test_parse_image_path(setup_tmp_media):
    """
    Test for function: walk.parse_image_path.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    for system, date_to_time_s in system_to_dict.items():
        for date_str, time_str_s in date_to_time_s.items():
            for time_str in time_str_s:
                image_file_path = Path(
                    f"{system}/{date_str}/{system}_{date_str}_{time_str}.jpeg"
                )
                datetime_ = dt.datetime.strptime(
                    f"{date_str}_{time_str}",
                    f"{DATE_FORMAT_FILE}_{TIME_FORMAT_FILE}",
                )

                # Tested function.
                obs_system, obs_datetime = walk.parse_image_path(image_file_path)

                # Check: System name.
                assert obs_system == system
                # Check: Datetime instance.
                assert obs_datetime == datetime_


def test_get_images(setup_tmp_media):
    """
    Test for function: walk.get_images.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    for system, date_to_time_s in system_to_dict.items():
        for date_str, time_str_s in date_to_time_s.items():
            date_ = dt.datetime.strptime(date_str, DATE_FORMAT_FILE).date()
            date_dir_path = walk.get_images_folder(tmp_media_path, system, date_)

            # Tested function.
            obs_image_s = walk.get_images(date_dir_path)

            # Check: Number of image instances.
            assert len(obs_image_s) == len(time_str_s)

            for obs_image in obs_image_s:

                # Check: Integrity of filename.
                assert system in obs_image.filename_stem
                assert date_str in obs_image.filename_stem

                # Check: Integrity of other attributes.
                assert obs_image.date_and_time.date() == date_
                assert obs_image.system == system
                assert obs_image.dir_path.is_dir()
                assert obs_image.hd.is_file()
                assert obs_image.thumbnail.is_file()
                assert obs_image.meta_path.is_file()
                # Check: Day is either date or the date before.
                date_before = date_ - dt.timedelta(days=1)
                assert obs_image.nightstart_date in [date_before, date_]


def test_get_monthly_nb_images(setup_tmp_media):
    """
    Test for function: walk.get_monthly_nb_images.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    for system, date_to_time_s in system_to_dict.items():
        for date_str, time_str_s in date_to_time_s.items():

            # Parse date.
            year, month, _ = date_str.split("_")
            year = int(year)
            month = int(month)

            # Tested function.
            obs_result = walk.get_monthly_nb_images(tmp_media_path, system, year, month)

            # TODO:
            #   Add dummy file "weathers.toml" to date directory in fixture
            #   setup,
            #   so that the weather data can be parsed in the test
            #   (and does not simply return None for failed parsing as in the
            #    current fixture setup).
            for (
                obs_date,
                obs_date_path,
                obs_nb_images,
                _,
            ) in obs_result:  # `_` is None for failed parsing of weather file.

                # Check: Date.
                obs_date_str = obs_date.strftime(DATE_FORMAT_FILE)
                assert obs_date_str in date_to_time_s.keys()

                # Check: Path of date directory.
                assert obs_date_path.is_dir()
                # Check: Number of image instances for date.
                assert obs_nb_images == len(time_str_s)


def test_images_zip_file(setup_tmp_media):
    """
    Test for function: walk.images_zip_file.
    """

    # Use fixture.
    tmp_media_path, system_to_dict = setup_tmp_media

    # Set up zip result directory.
    zip_dir_path = tmp_media_path / ZIP_DIR_NAME
    zip_dir_path.mkdir()

    for system, date_to_time_s in system_to_dict.items():
        for date_str in date_to_time_s.keys():

            date_ = dt.datetime.strptime(date_str, DATE_FORMAT_FILE).date()

            # Tested function.
            zip_file_path = walk.images_zip_file(
                tmp_media_path, system, date_, zip_dir_path
            )

            # Check: Zip result is a file.
            assert zip_file_path.is_file()
