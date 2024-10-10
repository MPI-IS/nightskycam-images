import datetime as dt
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    NewType,
    Optional,
    Tuple,
    Union,
    cast,
)
import zipfile

import toml
import tomli_w

from .constants import (
    DATE_FORMAT_FILE,
    DATETIME_FORMATS,
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
)
from .image import Image
from .weather import WeatherReport

Month = NewType("Month", int)
Year = NewType("Year", int)
WEATHER_FILENAME = "weathers.toml"

# TODO: Alias `iter_system_paths`?


def walk_systems(root: Path) -> Generator[Path, None, None]:
    """
    Iterate over paths of directories (representing systems)
    in media root directory.

    Parameters
    ----------
    root
        Path to media root directory.

    Yields
    ------
    Path
        Absolute path to system directory.
    """
    if not root.is_dir():
        raise FileNotFoundError(
            f"Failed to open nightskycam root {root}: not a directory"
        )

    # Directories and files in root directory.
    for path in root.iterdir():  # Note: iterdir is not ordered.
        # Only yield directories.
        if path.is_dir():
            yield path


def get_system_path(root: Path, system_name: str) -> Optional[Path]:
    """
    Get path to directory (representing system) in media root directory.

    Parameters
    ----------
    root
        Path to media root directory.
    system_name
        Name of system.

    Returns
    -------
    Path
        Absolute path to system directory.
    """
    for system_path in walk_systems(root):
        if system_path.name == system_name:
            return system_path
    return None


# TODO: improve handling of future dates?
#   @Vincent:
#     Is it intended that this function returns True for all future dates?
#     Would it make sense to raise an exception for future dates?
def _is_date_within_days(date: dt.date, nb_days: Optional[int]) -> bool:
    """
    Whether date is within nb_days in the past (counting today).

    I.e. for nb_days=1:
    - return False for any date before today.
    - return True for today (and any future date).
    """
    today = dt.datetime.now().date()

    if nb_days is None:
        return True

    return (today - date).days < nb_days


def walk_dates(
    system_dir: Path, within_nb_days: Optional[int] = None
) -> Generator[Tuple[dt.date, Path], None, None]:
    """
    Iterate over dates in system directory.

    Parameters
    ----------
    system_dir
        Path to system directory.
    within_nb_days
        If specified:
        only yield dates that are within number of days
        (counting today).

    Yields
    ------
    datetime.date
        Date instance.
    Path
        Path of date directory.
    """
    if not system_dir.is_dir():
        raise FileNotFoundError(
            f"Failed to open nightskycam folder {system_dir}: not a directory"
        )

    # Directories and files in system directory.
    for path in system_dir.iterdir():  # Note: iterdir is not ordered.
        # Only use directories.
        if path.is_dir():
            date_ = dt.datetime.strptime(path.name, DATE_FORMAT_FILE).date()
            if within_nb_days is None or _is_date_within_days(date_, within_nb_days):
                yield date_, path
    return None


def walk_all(
    root: Path,
    within_nb_days: Optional[int] = None,
    specific_date: Optional[dt.date] = None,
) -> Generator[Path, None, None]:
    """
    Iterate over paths of directories (representing dates)
    in ALL system directories present in media root directory.

    Parameters
    ----------
    root
        Path to media root directory.
    within_nb_days
        If specified:
        only yield dates that are within number of days
        (counting today).
    specific_date
        If specified:
        only return paths for the specified date.

    Yields
    ------
    Path
        Path of date directory
    """
    for system_path in walk_systems(root):
        for date_, date_path in walk_dates(system_path, within_nb_days=within_nb_days):
            if specific_date is None:
                yield date_path
            else:
                if specific_date == date_:
                    yield date_path
    return None


# TODO: Add docstring + test.
def walk_thumbnails(
    root: Path,
) -> Generator[Path, None, None]:  # TODO: Rename `root` -> `data_dir_path`.
    for folder in walk_all(root):
        f = folder / THUMBNAIL_DIR_NAME
        if f.is_dir():
            yield f
    return None


# TODO: Rename function to `get_date_path` (analogous to `get_system_path`).
def get_images_folder(root: Path, system_name: str, date: dt.date) -> Optional[Path]:
    """
    Get directory (containing the image files) for specified date and system.

    Parameters
    ----------
    root
        Path to media root directory.
    system_name
        Name of system.
    date
        Date instance.

    Returns
    -------
    Optional[Path]
        Path to date directory (containing the image files).
    """
    system_path = get_system_path(root, system_name)
    if system_path is None:
        return None
    for date_, date_path in walk_dates(system_path):
        if date_ == date:
            return date_path
    return None


def get_ordered_dates(
    root: Path, system_name: str
) -> Dict[Year, Dict[Month, List[Tuple[dt.date, Path]]]]:
    """
    Get ordered dates and their directory paths (grouped by year and month)
    for the specified system.

    Parameters
    ----------
    root
        Path to media root directory.
    system_name
        Name of system.

    Returns
    -------
    Dict
        Grouped and ordered dates.
        - key: Year
        - value: Dict
            - key: Month
            - value: List[Tuple]:
                - datetime.date:
                    Date instance.
                - Path:
                    Date directory path.
    """

    year_to_month_dict: Dict[Year, Dict[Month, List[Tuple[dt.date, Path]]]] = {}

    system_path = get_system_path(root, system_name)
    if system_path is None:
        return year_to_month_dict

    for date_, date_path in walk_dates(system_path):

        # Note:
        # `cast` is a signal to type checker
        # (returns value unchanged).
        year = cast(Year, date_.year)
        month = cast(Month, date_.month)

        try:
            month_dict = year_to_month_dict[year]
        # If dictionary does NOT have this year as key yet.
        except KeyError:
            month_dict = {}
            year_to_month_dict[year] = month_dict

        try:
            date_and_path_s = month_dict[month]
        # If dictionary does NOT have this month as key yet.
        except KeyError:
            date_and_path_s = []
            month_dict[month] = date_and_path_s

        date_and_path_s.append((date_, date_path))

    # Ensure correct order of list items.
    year_to_month_dict = {
        year: {
            # Sort tuples (date, path) in list by date,
            # otherwise order would be arbitrary on some operating systems.
            month: sorted(date_and_path_s)
            for month, date_and_path_s in month_to_date_and_path_s.items()
        }
        for year, month_to_date_and_path_s in year_to_month_dict.items()
    }

    return year_to_month_dict


def parse_image_path(
    image_file_path: Path,
    datetime_formats: Union[str, Tuple[str, ...]] = DATETIME_FORMATS,
) -> Tuple[str, dt.datetime]:
    """
    Get system name and datetime instance by parsing the name of the
    (HD or thumbnail) image file.

    Parameters
    ----------
    image
        Path of image file.
    datetime_formats
        Possible patterns of datetime format.

    Returns
    -------
    str
        System name.
    datetime.datetime
        Datetime instance.
    """
    # File name (without suffix).
    filename_stem = image_file_path.stem
    filename_parts = filename_stem.split("_")

    # If single string value was given as datetime-format.
    if isinstance(datetime_formats, str):
        # Bundle datetime-format in an iterable.
        datetime_formats = (datetime_formats,)

    for datetime_format in datetime_formats:
        # Number of parts in datetime-format.
        n = datetime_format.count("_") + 1
        # Partition.
        system_str = "_".join(filename_parts[:-n])
        datetime_str = "_".join(filename_parts[-n:])

        try:
            datetime_ = dt.datetime.strptime(datetime_str, datetime_format)
        except ValueError:
            pass
        else:
            break
    return system_str, datetime_


def _get_image_instance(
    thumbnail_file_path: Path,
    datetime_formats: Tuple[str, ...] = DATETIME_FORMATS,
) -> Image:
    """
    Set up and return an image instance for the given thumbnail file path.
    """

    def _get_folder(thumbnail_file_path: Path) -> Path:
        # FIXME FIXME FIXME
        # JC: Bug responsible for adding a leading / at the beginning of the path
        # because the first part of the path is  '/'
        # This implementation is really robust,
        # as it relies on having one and only one "thumbnails folder"
        # Also, why do we return this value in case of an error?

        path_rec = thumbnail_file_path

        try:
            # We go up until we either find "thumbnails" or reach the root
            root_path = thumbnail_file_path.absolute().anchor  # "/" in POSIX
            while thumbnail_file_path.name != "thumbnails" and thumbnail_file_path != root_path:
                thumbnail_file_path = thumbnail_file_path.parent
            return thumbnail_file_path.parent
        except ValueError:
            return path_rec.parent

    system_name, datetime = parse_image_path(
        thumbnail_file_path, datetime_formats=datetime_formats
    )

    # Set up Image instance.
    instance = Image()
    instance.filename_stem = thumbnail_file_path.stem
    instance.date_and_time = datetime
    instance.system = system_name
    instance.dir_path = _get_folder(thumbnail_file_path)

    return instance


def get_images(
    date_dir_path: Path,
    datetime_formats: Tuple[str, ...] = DATETIME_FORMATS,
) -> List[Image]:
    """
    Get image instances (contains paths of both HD and thumbnail images).

    Parameters
    ----------
    date_dir_path
        Path to date directory.
    datetime_formats
        Possible patterns of datetime format.

    Returns
    -------
    List[Image]
        List of image instances.
    """
    # Directory containing thumbnail images.
    thumbnail_dir_path = date_dir_path / THUMBNAIL_DIR_NAME

    # If thumbnail directory does NOT exist.
    if not thumbnail_dir_path.is_dir():
        return []

    thumbnail_file_paths: List[Path] = [
        file_path
        # Note: iterdir is not ordered.
        for file_path in thumbnail_dir_path.iterdir()
        if file_path.is_file() and file_path.suffix == f".{THUMBNAIL_FILE_FORMAT}"
    ]
    return [
        # Convert path to image instance.
        _get_image_instance(
            thumbnail_file_path,
            datetime_formats=datetime_formats,
        )
        for thumbnail_file_path in thumbnail_file_paths
    ]


# TODO: Add test.
# TODO: Confirm?
#   @Vincent:
#     Is this function used externally?
#     If not, it might make sense to make this function private.
def get_weather_report(
    root: Path,
    # TODO: Rename `root` -> `date_dir_path`.
    file_name: str = WEATHER_FILENAME,
) -> Optional[WeatherReport]:
    """
    Parse weather file (TOML format) in date directory.

    Parameters
    ----------
    root
        Path to media root directory.
    file_name
        Name of weather file.

    Returns
    -------
    Optional[WeatherReport]
        WeatherReport instance parsed from weather file.
        (None, if parsing failed.)
    """
    file_path = root / file_name
    if not file_path.is_file():
        return None

    try:
        report = toml.load(file_path)
    except toml.decoder.TomlDecodeError:
        return None

    return report["weathers"], report["skipped"]


# TODO: Rename to `get_ordered_dates_with_info`?
def get_monthly_nb_images(
    root: Path,
    system_name: str,
    year: Year,
    month: Month,
    datetime_formats: Tuple[str, ...] = DATETIME_FORMATS,
) -> List[Tuple[dt.date, Path, int, Optional[WeatherReport]]]:
    """
    Get the date instance, date directory path, the number of
    image instances and the weather report
    for each date (ordered) of the specified system, year and month.

    Parameters
    ----------
    root
        Path to media root directory.
    system_name
        Name of system.
    year
        Year of query.
    month
        Month of query.
    datetime_formats
        Possible patterns of datetime format.

    Returns
    -------
    List[Tuple]:
        - datetime.date:
            Date instance.
        - Path:
            Date directory path.
        - int:
            Number of image instances for date.
        - Optional[WeatherReport]:
            WeatherReport instance parsed from weather file (TOML format)
            in date directory.
            (None, if parsing failed.)
    """
    year_to_month_dict = get_ordered_dates(root, system_name)

    try:
        month_dict = year_to_month_dict[year]
    except KeyError:
        return []

    try:
        date_and_path_s = month_dict[month]
    except KeyError:
        return []

    return [
        (
            date_,
            date_path,
            len(get_images(date_path, datetime_formats=datetime_formats)),
            get_weather_report(date_path),
        )
        for date_, date_path in date_and_path_s
    ]


# TODO: Add test.
# TODO: Confirm?
#   @Vincent:
#     Is this function used externally?
#     If not, it might make sense to make this function private.
def meta_data_file(
    images: Iterable[Image],
    target_file: Path,  # TODO: Rename to `zip_file`
    datetime_format: str = DATETIME_FORMATS[0],
) -> None:
    """
    Write meta data of images to target file (in TOML format).
    """
    all_meta: Dict[str, Dict[str, Any]] = {}
    for image in images:
        if image.meta and image.date_and_time:
            all_meta[image.date_and_time.strftime(datetime_format)] = image.meta
    # TODO: Confirm.
    #   @Vincent:
    #     Does this work as intended?
    #     Is it intended to directly write to the zip archive file?
    #     In the subsequent steps in `_create_zip_file` the meta data
    #     files are also added to the zip archive file (again), but this
    #     time with ZipFile.write.
    with open(target_file, "wb") as f:
        tomli_w.dump(all_meta, f)


def _create_zip_file(
    images: Iterable[Image],
    target_file: Path,  # TODO: Rename to `zip_file`?
    meta_file: Optional[Path] = None,
    datetime_format: str = DATETIME_FORMATS[0],
) -> None:
    """
    Create zip file containing images (and optional meta data).
    """
    # All images are relevant for zipping.
    all_files = [image.hd for image in images if image.hd]

    if meta_file:
        # Add meta data file as relevant for zipping.
        meta_data_file(images, meta_file, datetime_format=datetime_format)
        all_files.append(meta_file)

    # Add all files to zip archive.
    with zipfile.ZipFile(target_file, "w") as zipf:
        for file_ in all_files:
            if file_ is not None:
                zipf.write(file_, arcname=file_.name)


# TODO: Remove argument `zip_dir_path`?
#   @Vincent:
#     Would it make sense to remove the argument `zip_dir_path` and
#     use a new constant `ZIP_DIR_NAME`
#     (similar to `THUMBNAIL_DIR_NAME`, inside date_dir_path)?
#   @Vincent:
#     Would it make sense to use a temporary directory for the zip
#     directory, as the zip file currently does not seem reusable?
def images_zip_file(
    root: Path,
    system_name: str,
    date: dt.date,
    zip_dir_path: Path,
    datetime_formats: Tuple[str, ...] = DATETIME_FORMATS,
    date_format: str = DATE_FORMAT_FILE,
    only_if_toml: bool = False,  # TODO: Never used in function -> remove?
) -> Path:
    """
    Create and get zip archive for images (and optional meta data).

    Parameters
    ----------
    root
        Path to media root directory.
    system_name
        Name of system.
    date
        Date instance of query.
    zip_dir_path
        Path of zip directory.
    datetime_formats
        Possible patterns of datetime format.
    date_format
        Patterns of date format.

    Returns
    -------
    Path
        Path of zip file.
    """
    date_dir_path = get_images_folder(root, system_name, date)
    if date_dir_path is None:
        raise ValueError(
            f"failed to find any image for system {system_name} at date {date}"
        )
    images = get_images(
        date_dir_path,
        datetime_formats=datetime_formats,
    )

    date_str = date.strftime(date_format)
    meta_file_path = zip_dir_path / f"{system_name}_{date_str}.toml"
    target_file_path = zip_dir_path / f"{system_name}_{date_str}.zip"

    _create_zip_file(
        images,
        target_file_path,
        meta_file=meta_file_path,
        datetime_format=datetime_formats[0],
    )

    return target_file_path
