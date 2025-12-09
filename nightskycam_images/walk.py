import datetime as dt
from pathlib import Path
from typing import (
    Any,
    Callable,
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

from loguru import logger
import toml
import tomli
import tomli_w

from .constants import (
    DATE_FORMAT_FILE,
    DATETIME_FORMATS,
    IMAGE_FILE_FORMATS,
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
)
from .image import Image
from .weather import WeatherReport

from .filters import create_combined_predicate

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
    try:
        for path in root.iterdir():  # Note: iterdir is not ordered.
            try:
                # Only yield directories that are accessible
                if path.is_dir():
                    # Test if we can access the directory before yielding
                    try:
                        # Quick access test
                        list(path.iterdir())
                        yield path
                    except PermissionError:
                        logger.warning(
                            f"Permission denied, skipping system directory: {path}"
                        )
                    except OSError as e:
                        logger.warning(
                            f"OS error, skipping system directory {path}: {e}"
                        )
            except PermissionError:
                logger.warning(f"Permission denied accessing: {path}")
            except OSError as e:
                logger.warning(f"OS error accessing {path}: {e}")
    except PermissionError:
        logger.error(f"Permission denied listing root directory: {root}")
        raise
    except OSError as e:
        logger.error(f"OS error listing root directory {root}: {e}")
        raise


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
    system_dir: Path,
    within_nb_days: Optional[int] = None,
    excluded: Iterable = ("snapshot",),
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
    try:
        for path in system_dir.iterdir():  # Note: iterdir is not ordered.
            try:
                # Only use directories.
                if path.is_dir():
                    if path.name not in excluded:
                        try:
                            date_ = dt.datetime.strptime(
                                path.name, DATE_FORMAT_FILE
                            ).date()
                            if within_nb_days is None or _is_date_within_days(
                                date_, within_nb_days
                            ):
                                yield date_, path
                        except ValueError:
                            logger.debug(
                                f"Skipping directory with invalid date format: {path.name}"
                            )
            except PermissionError:
                logger.warning(f"Permission denied accessing: {path}")
            except OSError as e:
                logger.warning(f"OS error accessing {path}: {e}")
    except PermissionError:
        logger.warning(
            f"Permission denied listing system directory, skipping: {system_dir}"
        )
        return None
    except OSError as e:
        logger.warning(f"OS error listing system directory, skipping {system_dir}: {e}")
        return None
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
            while (
                thumbnail_file_path.name != "thumbnails"
                and thumbnail_file_path != root_path
            ):
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


def _is_within_time_window(
    image_datetime: dt.datetime,
    time_window: Tuple[Optional[dt.time], Optional[dt.time]],
) -> bool:
    """
    Check if image time falls within the specified time window.

    Parameters
    ----------
    image_datetime
        Datetime of the image.
    time_window
        Tuple of (start_time, end_time) defining the time window.
        Either can be None for open-ended ranges.

    Returns
    -------
    bool
        True if image time is within the time window.
    """
    image_time = image_datetime.time()
    start_time, end_time = time_window

    # If both are None, accept all times
    if start_time is None and end_time is None:
        return True

    # Only start time specified (from start_time onwards)
    if start_time is not None and end_time is None:
        return image_time >= start_time

    # Only end time specified (up to end_time)
    if start_time is None and end_time is not None:
        return image_time <= end_time

    # Both specified - handle time window that crosses midnight
    if start_time <= end_time:  # type: ignore 
        # Normal case: e.g., 08:00 to 20:00
        return start_time <= image_time <= end_time  # type: ignore
    else:
        # Crosses midnight: e.g., 20:00 to 08:00
        return image_time >= start_time or image_time <= end_time  # type: ignore


def _is_within_date_range(
    image_date: dt.date,
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
) -> bool:
    """
    Check if image date falls within the specified date range.

    Parameters
    ----------
    image_date
        Date of the image.
    start_date
        Start date of the range (inclusive). None means no lower bound.
    end_date
        End date of the range (inclusive). None means no upper bound.

    Returns
    -------
    bool
        True if image date is within the date range.
    """
    if start_date is not None and image_date < start_date:
        return False
    if end_date is not None and image_date > end_date:
        return False
    return True


def _create_symlink_safe(
    source: Path,
    destination: Path,
) -> bool:
    """
    Safely create a symlink, handling conflicts and errors.

    Parameters
    ----------
    source
        Source file path.
    destination
        Destination symlink path.

    Returns
    -------
    bool
        True if symlink was created successfully, False otherwise.
    """
    try:
        # Check if source exists
        if not source.exists():
            logger.warning(f"Source file does not exist: {source}")
            return False

        # Check if destination already exists
        if destination.exists() or destination.is_symlink():
            # Silently skip existing symlinks (common when re-running)
            return False

        # Create symlink
        destination.symlink_to(source)
        return True

    except PermissionError as e:
        logger.error(f"Permission error creating symlink {destination.name}: {e}")
        return False
    except OSError as e:
        logger.error(f"OS error creating symlink {destination.name}: {e}")
        return False


def _has_weather_substring(toml_path: Optional[Path], substring: str) -> bool:
    """
    Check if weather field in TOML contains substring (case-sensitive).

    Parameters
    ----------
    toml_path
        Path to TOML metadata file.
    substring
        Substring to search for in weather field.

    Returns
    -------
    bool
        True if weather field contains substring, False otherwise.
    """
    if toml_path is None or not toml_path.exists():
        return False

    try:
        with open(toml_path, "rb") as f:
            metadata = tomli.load(f)

        weather = metadata.get("weather")
        if weather is None:
            return False

        return substring in weather

    except Exception as e:
        logger.debug(f"Error reading TOML {toml_path}: {e}")
        return False


def _get_images_from_hd(
    date_dir_path: Path,
    datetime_formats: Tuple[str, ...] = DATETIME_FORMATS,
) -> List[Image]:
    """
    Get images directly from HD files (not thumbnails).

    This function discovers images by looking for HD image files in the date
    directory, which works even when no thumbnails directory exists.

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
    # Find HD image files in the date directory
    hd_files = [
        f
        for f in date_dir_path.iterdir()
        if f.is_file() and f.suffix.lstrip(".") in IMAGE_FILE_FORMATS
    ]

    images = []
    for hd_file in hd_files:
        try:
            system_name, datetime = parse_image_path(
                hd_file, datetime_formats=datetime_formats
            )

            # Create Image instance (following pattern from _get_image_instance)
            image = Image()
            image.filename_stem = hd_file.stem
            image.date_and_time = datetime
            image.system = system_name
            image.dir_path = date_dir_path
            images.append(image)

        except Exception as e:
            logger.debug(f"Skipping {hd_file}: {e}")

    return images


def filter_and_export_images(
    root: Path,
    output_dir: Path,
    systems: Optional[List[str]] = None,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
    time_window: Optional[Tuple[Optional[dt.time], Optional[dt.time]]] = None,
    predicate: Optional[Callable[[Path, Optional[Path]], bool]] = None,
    progress_callback: Optional[Callable[[Path, int, int], bool]] = None,
    cache_process_filter: bool = False,
    process_substring: Optional[str] = None,
    process_not_substring: Optional[str] = None,
    cloud_cover_range: Optional[tuple[int, int]] = None,
    weather_values: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    Filter images based on criteria and create symlinks in output directory.

    This function traverses the nightskycam image directory structure and creates
    symlinks to images (and their associated TOML metadata files) that match
    the specified filtering criteria.

    Parameters
    ----------
    root
        Path to media root directory.
    output_dir
        Path to output directory where symlinks will be created.
    systems
        List of system names to include. If None, all systems are included.
    start_date
        Start date of the range (inclusive). None means no lower bound.
    end_date
        End date of the range (inclusive). None means no upper bound.
    time_window
        Tuple of (start_time, end_time) for time-of-day filtering.
        Example: (dt.time(20, 0), dt.time(23, 0)) for 8pm to 11pm.
        None means no time filtering.
    predicate
        Optional callable that takes (image_path, toml_path) and returns bool.
        Only images for which predicate returns True are included.
        None means no predicate filtering.
        NOTE: If individual filter parameters (process_substring, cloud_cover_range, etc.)
        are provided, they take precedence over this predicate.
    progress_callback
        Optional callable that receives (current_folder, processed_count, total_count).
        Called for each date folder processed. Should return False to cancel processing,
        True to continue.
    cache_process_filter
        Enable process caching optimization. If True, checks ONLY the process filters
        (process_substring, process_not_substring) on the first image of each folder.
        If it fails, skips the entire folder. Cloud cover and weather filters are still
        checked on every image. This is faster but assumes the process field is the
        same for all images in a folder.
    process_substring
        Substring that must be present in the 'process' field.
    process_not_substring
        Substring that must NOT be present in the 'process' field.
    cloud_cover_range
        Tuple of (min_cover, max_cover) for cloud cover filtering.
    weather_values
        Weather value(s) to match.

    Returns
    -------
    None

    Examples
    --------
    >>> # Export all images from system 'nightskycam5' between 8pm-11pm
    >>> filter_and_export_images(
    ...     root=Path("/data/nightskycam"),
    ...     output_dir=Path("/output"),
    ...     systems=["nightskycam5"],
    ...     time_window=(dt.time(20, 0), dt.time(23, 0))
    ... )
    """
    # Import the filter creation function


    # Create predicates based on provided filters
    # If individual filter parameters are provided, use them; otherwise use the predicate parameter
    if any(
        [process_substring, process_not_substring, cloud_cover_range, weather_values]
    ):
        # Create process-only predicate (for caching)
        process_predicate = create_combined_predicate(
            process_substring=process_substring,
            process_not_substring=process_not_substring,
            cloud_cover_range=None,  # Exclude from process cache
            weather_values=None,  # Exclude from process cache
        )

        # Create full predicate (for individual image filtering)
        full_predicate = create_combined_predicate(
            process_substring=process_substring,
            process_not_substring=process_not_substring,
            cloud_cover_range=cloud_cover_range,
            weather_values=weather_values,
        )
    else:
        # Use the provided predicate for both (backward compatibility)
        process_predicate = predicate
        full_predicate = predicate

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting image filtering and export to: {output_dir}")
    logger.info(f"Root directory: {root}")
    logger.info(f"Systems filter: {systems if systems else 'all'}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Time window: {time_window}")
    if cache_process_filter:
        logger.info(f"Process caching enabled (applies only to process filters)")

    # First pass: count total folders to process for progress tracking
    total_folders = 0
    if progress_callback is not None:
        for system_path in walk_systems(root):
            system_name = system_path.name
            if systems is not None and system_name not in systems:
                continue
            for date_, date_path in walk_dates(system_path):
                if _is_within_date_range(date_, start_date, end_date):
                    total_folders += 1

    # Statistics
    total_scanned = 0
    total_matched = 0
    symlinks_created = 0
    processed_folders = 0

    # Iterate through systems
    for system_path in walk_systems(root):
        system_name = system_path.name

        # Filter by system name if specified
        if systems is not None and system_name not in systems:
            logger.debug(f"Skipping system: {system_name} (not in filter list)")
            continue

        logger.info(f"Processing system: {system_name}")

        # Iterate through dates in the system
        for date_, date_path in walk_dates(system_path):
            # Filter by date range
            if not _is_within_date_range(date_, start_date, end_date):
                continue

            # Call progress callback and check for cancellation
            if progress_callback is not None:
                processed_folders += 1
                should_continue = progress_callback(
                    date_path, processed_folders, total_folders
                )
                if not should_continue:
                    logger.info("Processing cancelled by user")
                    return

            # Track statistics for this specific folder
            folder_symlinks_created = 0
            folder_missing_toml = 0
            folder_missing_keys = 0
            folder_filtered_out = 0

            # Create output subdirectory matching the source structure
            date_str = date_.strftime(DATE_FORMAT_FILE)
            output_subdir = output_dir / system_name / date_str
            output_subdir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Investigating: {system_name}/{date_str}")

            # Get all images for this date
            images = get_images(date_path)

            # Process caching optimization: Check ONLY process criteria on first image
            # If it fails, skip the entire folder
            # Note: Cloud cover and weather are NOT cached as they can change throughout the night
            if cache_process_filter and images and process_predicate is not None:
                first_image = images[0]
                first_image_path = first_image.hd
                first_toml_path = first_image.meta_path

                if first_image_path is not None and first_toml_path is not None:
                    try:
                        # Check if first image passes the PROCESS filters only
                        if not process_predicate(first_image_path, first_toml_path):
                            logger.info(f"  Skipped (folder failed process filter)")
                            continue
                    except Exception as e:
                        logger.warning(
                            f"Error checking first image in {date_path.name}: {e}"
                        )

            for image in images:
                total_scanned += 1

                # Skip if image doesn't have required attributes
                if image.date_and_time is None:
                    logger.warning(
                        f"Image missing datetime, skipping: {image.filename_stem}"
                    )
                    continue

                # Filter by time window
                if time_window is not None:
                    if not _is_within_time_window(image.date_and_time, time_window):
                        folder_filtered_out += 1
                        continue

                # Get image and toml paths
                image_path = image.hd
                toml_path = image.meta_path

                # Skip if paths are None
                if image_path is None:
                    continue

                # Check for missing TOML file
                if toml_path is None or not toml_path.exists():
                    folder_missing_toml += 1
                    logger.debug(f"Skipping {image_path.name}: missing TOML file")
                    continue

                # Check for missing keys in TOML if we have metadata filters
                if full_predicate is not None:
                    # Check if required keys are missing
                    missing_keys = False
                    try:
                        import toml

                        data = toml.load(toml_path)

                        # Check if any of the filter keys are missing
                        if cloud_cover_range is not None and "cloud_cover" not in data:
                            missing_keys = True
                        elif weather_values is not None and "weather" not in data:
                            missing_keys = True
                        elif (
                            process_substring is not None
                            or process_not_substring is not None
                        ) and "process" not in data:
                            missing_keys = True
                    except Exception:
                        # If we can't load the TOML, treat as missing
                        folder_missing_toml += 1
                        continue

                    if missing_keys:
                        folder_missing_keys += 1
                        logger.debug(
                            f"Skipping {image_path.name}: missing required metadata keys"
                        )
                        continue

                # Apply full predicate (includes ALL filters: process, cloud cover, weather)
                if full_predicate is not None:
                    try:
                        if not full_predicate(image_path, toml_path):
                            folder_filtered_out += 1
                            logger.debug(
                                f"Skipping {image_path.name}: did not match filter criteria"
                            )
                            continue
                    except Exception as e:
                        logger.error(
                            f"Predicate raised exception for {image_path.name}: {e}"
                        )
                        folder_filtered_out += 1
                        continue

                # Image matches all filters
                total_matched += 1

                # Create symlinks in the structured output directory
                # Symlink for image file
                image_dest = output_subdir / image_path.name
                if _create_symlink_safe(image_path, image_dest):
                    symlinks_created += 1
                    folder_symlinks_created += 1

                # Symlink for toml file (if exists)
                if toml_path is not None:
                    toml_dest = output_subdir / toml_path.name
                    if _create_symlink_safe(toml_path, toml_dest):
                        symlinks_created += 1
                        folder_symlinks_created += 1

            # Log results for this folder
            if folder_symlinks_created > 0:
                skip_parts = []
                if folder_missing_toml > 0:
                    skip_parts.append(f"{folder_missing_toml} missing TOML")
                if folder_missing_keys > 0:
                    skip_parts.append(f"{folder_missing_keys} missing keys")
                if folder_filtered_out > 0:
                    skip_parts.append(f"{folder_filtered_out} filtered out")

                skip_msg = f", skipped {' + '.join(skip_parts)}" if skip_parts else ""
                logger.info(f"  Created {folder_symlinks_created} symlinks{skip_msg}")
            else:
                # Count total skipped for better reporting
                total_skipped = (
                    folder_missing_toml + folder_missing_keys + folder_filtered_out
                )
                if total_skipped > 0:
                    skip_parts = []
                    if folder_missing_toml > 0:
                        skip_parts.append(f"{folder_missing_toml} missing TOML")
                    if folder_missing_keys > 0:
                        skip_parts.append(f"{folder_missing_keys} missing keys")
                    if folder_filtered_out > 0:
                        skip_parts.append(f"{folder_filtered_out} filtered out")
                    logger.info(f"  No matches (skipped {' + '.join(skip_parts)})")
                else:
                    logger.info(f"  No images found")

    # Log summary
    logger.info(f"Filtering complete!")
    logger.info(f"Total images scanned: {total_scanned}")
    logger.info(f"Total images matched: {total_matched}")
    logger.info(f"Total symlinks created: {symlinks_created}")


def move_clear_images(
    root: Path,
    target_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Move images with clear weather (22:00-04:00) to target directory.

    Filters images where:
    - Weather field contains substring "clear" (case-sensitive)
    - Time is between 22:00 and 04:00

    Images and their TOML metadata files are moved (not copied or symlinked)
    to the target directory preserving the system/date structure. Empty
    directories in the source are cleaned up after moving.

    Parameters
    ----------
    root
        Source media root directory.
    target_dir
        Target directory (preserves system/date structure).
    dry_run
        If True, preview without moving files.
    verbose
        If True, enable debug logging.

    Returns
    -------
    Dict[str, int]
        Statistics: images_scanned, images_matched, images_moved, etc.
    """
    import shutil

    # Initialize statistics
    stats = {
        "images_scanned": 0,
        "images_matched": 0,
        "images_moved": 0,
        "tomls_moved": 0,
        "images_skipped": 0,
        "errors": 0,
    }

    # Define time window: 22:00 to 04:00 (crosses midnight)
    time_window = (dt.time(22, 0), dt.time(4, 0))

    # Validate root directory
    if not root.exists() or not root.is_dir():
        logger.error(f"Root directory does not exist: {root}")
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    # Create target directory if needed
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Moving clear images from {root} to {target_dir}")
    if dry_run:
        logger.info("[DRY-RUN MODE] No files will be moved")

    # Traverse systems
    for system_path in walk_systems(root):
        system_name = system_path.name
        logger.info(f"Processing system: {system_name}")

        # Traverse dates within system
        for date, date_path in walk_dates(system_path):
            date_str = date_path.name
            if verbose:
                logger.debug(f"  Processing date: {date_str}")

            # Get images from HD files (works without thumbnails)
            images = _get_images_from_hd(date_path)

            folder_scanned = 0
            folder_matched = 0
            folder_moved = 0
            folder_skipped = 0

            # Process each image
            for image in images:
                folder_scanned += 1
                stats["images_scanned"] += 1

                # Skip if image doesn't have datetime
                if image.date_and_time is None:
                    if verbose:
                        logger.debug(
                            f"    Skipping (no datetime): {image.filename_stem}"
                        )
                    continue

                # Check time filter
                if not _is_within_time_window(image.date_and_time, time_window):
                    continue

                # Check weather filter
                if not _has_weather_substring(image.meta_path, "clear"):
                    continue

                # Both filters passed
                folder_matched += 1
                stats["images_matched"] += 1

                # Get source paths
                image_source = image.hd
                toml_source = image.meta_path

                if image_source is None:
                    logger.warning(f"Could not find HD file for {image.filename_stem}")
                    stats["errors"] += 1
                    continue

                # Construct target paths
                target_system_dir = target_dir / system_name / date_str
                image_target = target_system_dir / image_source.name
                toml_target = (
                    target_system_dir / toml_source.name if toml_source else None
                )

                # Check if image already exists in target
                if image_target.exists():
                    if verbose:
                        logger.debug(
                            f"    Skipping (already exists): {image_source.name}"
                        )
                    folder_skipped += 1
                    stats["images_skipped"] += 1
                    continue

                # Move image file
                try:
                    if dry_run:
                        logger.info(
                            f"    [DRY-RUN] Would move: {image_source.name} -> {image_target}"
                        )
                        if verbose and toml_source and toml_source.exists():
                            logger.debug(
                                f"    [DRY-RUN] Would move TOML: {toml_source.name} -> {toml_target}"
                            )
                    else:
                        # Create target directory
                        target_system_dir.mkdir(parents=True, exist_ok=True)

                        # Move image file
                        shutil.move(str(image_source), str(image_target))
                        folder_moved += 1
                        stats["images_moved"] += 1

                        if verbose:
                            logger.debug(f"    Moved: {image_source.name}")

                        # Move TOML file if it exists
                        if toml_source and toml_source.exists() and toml_target:
                            shutil.move(str(toml_source), str(toml_target))
                            stats["tomls_moved"] += 1
                            if verbose:
                                logger.debug(f"    Moved TOML: {toml_source.name}")

                except Exception as e:
                    logger.error(f"    Failed to move {image_source}: {e}")
                    stats["errors"] += 1

            # Log folder summary
            if folder_matched > 0 or verbose:
                msg_parts = []
                if folder_matched > 0:
                    msg_parts.append(f"Matched: {folder_matched}")
                    if not dry_run:
                        msg_parts.append(f"Moved: {folder_moved}")
                    if folder_skipped > 0:
                        msg_parts.append(f"Skipped: {folder_skipped}")
                elif folder_scanned > 0:
                    msg_parts.append(f"Scanned: {folder_scanned}, no matches")
                else:
                    msg_parts.append("No images found")

                logger.info(f"  {date_str}: {', '.join(msg_parts)}")

    # Cleanup empty directories (only if not dry-run)
    if not dry_run:
        logger.info("Cleaning up empty directories...")
        # Import here to avoid circular import
        from .main import _cleanup_empty_directories

        cleanup_stats = _cleanup_empty_directories(root, dry_run, verbose)
        stats.update(cleanup_stats)

    return stats


def copy_and_retarget_symlinks(
    input_dir: Path,
    output_dir: Path,
    root_dir: Path,
) -> Dict[str, int]:
    """
    Copy a symlink directory structure and retarget symlinks to a different root.

    This function takes a directory created by nightskycam-filter-export (containing
    symlinks organized in a nightskycam structure) and creates a copy with symlinks
    retargeted to point to corresponding files in a different root directory.

    Parameters
    ----------
    input_dir
        Path to the input directory containing symlinks (from nightskycam-filter-export).
    output_dir
        Path to the output directory to create (will contain retargeted symlinks).
    root_dir
        Path to the root directory containing the target files (nightskycam structure).

    Returns
    -------
    Dict[str, int]
        Statistics dictionary with keys:
        - 'processed': Number of symlinks processed
        - 'created': Number of symlinks successfully created
        - 'skipped': Number of non-symlink files skipped
        - 'warnings': Number of warnings (missing target files)
    """
    logger.info(f"Copying symlink directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target root directory: {root_dir}")

    # Initialize statistics
    stats = {
        "processed": 0,
        "created": 0,
        "skipped": 0,
        "warnings": 0,
    }

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk through input directory
    for item in input_dir.rglob("*"):
        # Skip directories
        if item.is_dir():
            continue

        # Get relative path from input_dir
        rel_path = item.relative_to(input_dir)

        # Skip non-symlink files
        if not item.is_symlink():
            logger.debug(f"Skipping non-symlink file: {rel_path}")
            stats["skipped"] += 1
            continue

        stats["processed"] += 1

        # Construct new target path in root_dir
        new_target = root_dir / rel_path

        # Check if target exists in new root
        if not new_target.exists():
            logger.warning(f"Target file does not exist in new root: {new_target}")
            stats["warnings"] += 1
            continue

        # Create output directory structure
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create new symlink
        try:
            output_path.symlink_to(new_target)
            stats["created"] += 1
            logger.debug(f"Created symlink: {rel_path} -> {new_target}")
        except FileExistsError:
            logger.debug(f"Symlink already exists: {rel_path}")
        except PermissionError as e:
            logger.error(f"Permission error creating symlink {rel_path}: {e}")
            stats["warnings"] += 1
        except OSError as e:
            logger.error(f"OS error creating symlink {rel_path}: {e}")
            stats["warnings"] += 1

    return stats
