"""
Definition of project-level constants.
"""

from typing import Tuple

# Files in general.
FILE_PERMISSIONS: int = 0o755  # Octal integer.

# HD image.
IMAGE_FILE_FORMATS: Tuple[str, ...] = ("npy", "tiff", "jpg", "jpeg")

# Thumbnail image.
THUMBNAIL_DIR_NAME: str = "thumbnails"
THUMBNAIL_FILE_FORMAT: str = "jpeg"
THUMBNAIL_WIDTH: int = 200  # In pixels.

# Zip.
ZIP_DIR_NAME: str = "zip"

# Video.
VIDEO_FILE_NAME: str = "day_summary.webm"

# Weather summary.
WEATHER_SUMMARY_FILE_NAME: str = "weathers.toml"

# Format patterns.

# formats for filename, e.g. nightskycamX_2024_09_26_13_57_50.jpeg
DATE_FORMAT_FILE: str = "%Y_%m_%d"
TIME_FORMAT_FILE: str = "%H_%M_%S"
DATETIME_FORMATS: Tuple[str, ...] = ("%d_%m_%Y_%H_%M_%S", "%Y_%m_%d_%H_%M_%S")

# formats for displaying on the website
DATE_FORMAT: str = "%Y-%m-%d"
TIME_FORMAT: str = "%H:%M:%S"
