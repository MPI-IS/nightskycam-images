import logging
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import toml
import tomli_w

from .constants import FILE_PERMISSIONS, IMAGE_FILE_FORMATS, WEATHER_SUMMARY_FILE_NAME
from .folder_change import folder_has_changed

# linking weather description from here: https://www.meteosource.com/documentation
# to icon from here: https://icons.getbootstrap.com/

WEATHER_TO_BOOTSTRAP_ICON: Dict[str, str] = {
    "not available": "question-circle",
    "sunny": "stars",
    "mostly sunny": "cloud-moon",
    "partly sunny": "cloud-moon",
    "mostly cloudy": "cloud",
    "cloudy": "cloud",
    "overcast": "cloud-fill",
    "fog": "cloud-fog",
    "light rain": "cloud-drizzle",
    "rain": "cloud-rain-heavy",
    "possible rain": "cloud-drizzle",
    "rain shower": "cloud-drizzle",
    "thunderstorm": "cloud-lightning",
    "light snow": "cloud-snow",
    "snow": "cloud-snow",
    "possible snow": "cloud-snow",
    "snow shower": "cloud-snow",
    "hail": "cloud-hail",
    "clear": "stars",
    "mostly clear": "cloud-moon",
    "partly clear": "cloud-moon",
}

# ( {weather type: nb of images of this weather} , number of images skipped)
WeatherReport = Tuple[Dict[str, int], int]


def _bootstrap_wrap(icon: str) -> str:
    # HTML command for adding bootstrap icon.
    return f'<i class="bi-{icon}"></i>'


def get_weather_icon(weather: str) -> str:
    """
    Get HTML command for adding icon for weather.
    If no icon can be matched to the weather, return raw input string of
    weather instead.

    Parameters
    ----------
    weather
        Weather description from meteosource
        (https://www.meteosource.com/documentation).

    Returns
    -------
    str
        HTML command for adding weather icon
        OR
        input weather string, if NO matching icon was found.
    """
    # If weather string is specified in dictionary.
    try:
        return _bootstrap_wrap(WEATHER_TO_BOOTSTRAP_ICON[weather.lower()])
    except KeyError:
        pass

    # If weather string is NOT specified in dictionary.
    # Approximate with most closely related key string in dictionary.
    for superkey in (
        "rain",
        "clear",
        "snow",
        "sunny",
        "overcast",
        "thunderstorm",
        "fog",
    ):
        if superkey in weather.lower():
            return _bootstrap_wrap(WEATHER_TO_BOOTSTRAP_ICON[superkey])

    # If approximation was NOT possible.
    # Fallback: Use raw input string instead of an icon.
    return weather


def weather_summary(
    folder: Path,
    summary_path: Optional[Path] = None,
    permissions: int = FILE_PERMISSIONS,
) -> WeatherReport:
    """
    Create weather summary for a date directory
    (containing pairs of image and weather files).

    Parameters
    ----------
    folder
        Path to date directory containing the image and weather files (toml format).
        There is a weather file for each image file.
        These individual weather files will be used for creating the output
        weather summary.
    toml_report
        Path to weather summary file (toml format).
    permissions
        File permissions for the weather summary file.

    Returns
    -------
    WeatherReport
        - Dict:
            - key: weather type
            - value: number of images
        - number of skipped images
    """
    # Get individual weather files (toml format).
    #
    # Get all toml files in the directory.
    toml_files: List[Path] = list(folder.glob("*.toml"))
    # If result summary file was specified.
    if summary_path is not None:
        # Exclude toml file with same stem as the summary file.
        #
        # TODO: match filename instead of stem?
        #   @Vincent:
        #     Would it be safer to match the filename instead of only the stem?
        toml_files = [tf for tf in toml_files if not tf.stem == summary_path.stem]

    # Get all image files in the directory.
    image_path_s: List[Path] = []
    for ext in IMAGE_FILE_FORMATS:
        image_path_s.extend(folder.glob(f"*.{ext}"))
    image_stem_s = [img.stem for img in image_path_s]

    # {weather type: nb of images of this weather}
    weather_to_count: Dict[str, int] = {}
    # TODO#1: confirm (connected to TODO#2).
    #   @Vincent:
    #     Do I understand this correctly?
    #
    # Number of TOML-files that can be parsed as TOML,
    # but are skipped because of NOT containing weather information.
    skipped = 0

    for tf in toml_files:
        weather: Optional[str] = None
        try:
            # Parse as dictionary.
            content = toml.load(tf)
        # TOML file could NOT be parsed.
        except Exception:
            # TODO#2: confirm (connected to TODO#1).
            #   @Vincent:
            #     Is it intended that this does NOT increase the
            #     `skipped` counter?
            continue
        try:
            weather = content["weather"]
        except KeyError:
            if tf.stem in image_stem_s:
                weather = "?"
            else:
                skipped += 1
                continue
        if weather is not None:
            try:
                weather_to_count[weather] += 1
            # New key.
            except KeyError:
                # Initialise new key with counter.
                weather_to_count[weather] = 1

    if summary_path is not None:
        with open(summary_path, "wb") as f:
            tomli_w.dump({"skipped": skipped, "weathers": weather_to_count}, f)
            # Set file permissions for weather summary.
            summary_path.chmod(permissions)
            logging.info("created weather summary for %s successfully", folder)

    return weather_to_count, skipped


def create_weather_summaries(
    walk_folders: Callable[[], Generator[Path, None, None]],
    history: Optional[dict[Path, Optional[float]]] = None,
    permissions: int = FILE_PERMISSIONS,
    summary_file_name: str = WEATHER_SUMMARY_FILE_NAME,
) -> None:
    """
    Create weather summary for one or more date directories
    (containing pairs of image and weather files).

    Parameters
    ----------
    walk_folders
        Function for iterating over date directories
        (containing image and weather files).
    history
        Last modification times found in previous check:
        - key:
            Path to directory.
        - value:
            Time of last modification of directory.
    permissions
        File permissions for the weather summary file.
    summary_filename
        Name of weather summary file.
    """
    # Iterate over date directories
    # (containing image and weather files).
    for folder in walk_folders():
        # If last-modified-time of directory changed.
        if folder_has_changed(folder, history):
            # Create weather summary file for directory.
            weather_summary(folder, summary_path=folder / summary_file_name)


def weather_report_to_str(
    report: Optional[WeatherReport], short: bool = False, html: bool = False
) -> str:
    if report is None:
        return ""
    total_images = sum(report[0].values())
    skipped = report[1]
    images_taken = total_images - skipped
    if html:
        icons = set([get_weather_icon(w) for w in report[0].keys()])
        weathers = " ".join(icons)
    else:
        weathers = ", ".join([w for w in report[0].keys()])
    if short:
        return f"{images_taken} images - {weathers}"
    if skipped > 0:
        return (
            f"{images_taken} images ({skipped} images skipped because of bad weather). "
            f"Weather: {weathers}"
        )
    else:
        return f"{images_taken} images. Weather: {weathers}"
