import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, cast

import toml

from .constants import (
    DATE_FORMAT,
    DATE_FORMAT_FILE,
    IMAGE_FILE_FORMATS,
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
    VIDEO_FILE_NAME,
)


class Image:
    def __init__(self) -> None:

        # Stem of file name (without extension).
        self.filename_stem: Optional[str] = None
        # Datetime instance.
        # NOT named `datetime` to avoid confusion with package/class.
        self.date_and_time: Optional[dt.datetime] = None
        # System name.
        self.system: Optional[str] = None

        # Absolute path to directory containing the HD images.
        # Also contains sub-directory containing the thumbnail images.
        self.dir_path: Optional[Path] = None

    @property
    def filename(self) -> Optional[str]:
        """
        Alias for filename_stem
        """
        return self.filename_stem

    @property
    def date(self) -> Optional[dt.datetime]:
        """
        Alias for date_and_time
        """
        return self.date_and_time

    @property
    def thumbnail(self) -> Optional[Path]:
        """
        Absolute path to thumbnail image file.
        """
        if self.dir_path is None:
            return None

        file_path = (
            self.dir_path
            / THUMBNAIL_DIR_NAME
            / f"{self.filename_stem}.{THUMBNAIL_FILE_FORMAT}"
        )
        if file_path.is_file():
            return file_path

        return None

    @property
    def video(self) -> Optional[Path]:
        """
        Absolute path to video file.
        """
        if self.dir_path is None:
            return None

        file_path = self.dir_path / THUMBNAIL_DIR_NAME / VIDEO_FILE_NAME
        if file_path.is_file():
            return file_path

        return None

    @property
    def hd(self) -> Optional[Path]:
        """
        Absolute path to HD image file.
        """
        if self.dir_path is None:
            return None

        for f in IMAGE_FILE_FORMATS:
            file_path = self.dir_path / f"{self.filename_stem}.{f}"
            if file_path.is_file():
                return file_path

        return None

    @property
    def path(self) -> Optional[Path]:
        """
        Alias for hd
        """
        return self.hd

    @property
    def meta_path(self) -> Optional[Path]:
        """
        Absolute path to meta data file.
        """
        if self.dir_path is None:
            return None

        file_path = self.dir_path / f"{self.filename_stem}.toml"
        if file_path.is_file():
            return file_path

        return None

    @property
    def meta(self) -> Dict[str, Any]:

        if self.dir_path is None or self.filename_stem is None:
            return {}
        meta_file = self.meta_path
        if not meta_file:
            return {}
        if not meta_file.is_file():
            return {}
        try:
            meta = toml.load(meta_file)
        except toml.decoder.TomlDecodeError as e:
            meta = {"error": f"failed to read {meta_file}: {e}"}
        return meta

    @property
    def nightstart_date(self) -> Optional[dt.date]:
        """
        Start date of the night.

        Purpose:
        Utility for bundling images per night instead of date.

        Context:
        The date switches at midnight.
        """
        if self.date_and_time is None:
            return None
        hour = self.date_and_time.hour

        # Before noon,
        # therefore assign to date of previous day.
        if hour < 12:
            return (self.date_and_time - dt.timedelta(days=1)).date()

        return self.date_and_time.date()

    @property
    def day(self) -> Optional[dt.date]:
        """
        Alias for nightstart_date
        """
        return self.nightstart_date

    @property
    def day_as_str(self) -> Optional[str]:
        if self.day is None:
            return None
        return self.day.strftime(DATE_FORMAT_FILE)

    @property
    def nightstart_date_as_str(self) -> Optional[str]:
        """
        String representation of the property `nightstart_date`.
        """
        nightstart_date = self.nightstart_date
        if nightstart_date is None:
            return None
        return nightstart_date.strftime(DATE_FORMAT)

    @staticmethod
    def date_from_str(date: str) -> dt.date:
        """
        Get date instance from its string representation.
        """
        return dt.datetime.strptime(date, DATE_FORMAT_FILE).date()

    @staticmethod
    def day_from_str(day: str) -> dt.date:
        """
        Alias for date_from_str
        """
        return dt.datetime.strptime(day, DATE_FORMAT_FILE).date()

    @property
    def datetime_as_str(
        self, datetime_format: str = "%b. %d, %Y, %-I:%M %p"
    ) -> Optional[str]:
        """
        String representation of the attribute `date_and_time`.
        """
        if self.date is None:
            return None
        return self.date.strftime(datetime_format)

    def to_dict(self) -> Dict[str, str]:
        """
        Get dictionary representation of image instance.
        """
        r: Dict[str, str] = {}
        r["path"] = str(self.hd)
        r["thumbnail"] = str(self.thumbnail)
        r["video"] = str(self.video)
        r["date_as_str"] = str(self.datetime_as_str)
        r["day_as_str"] = str(self.nightstart_date_as_str)
        r["system"] = str(self.system)
        r["meta"] = repr(self.meta)
        r["meta_path"] = str(self.meta_path)
        return r

    def __eq__(self, other: object) -> bool:
        return self.date_and_time == Image.date_and_time

    def __gt__(self, other: object) -> bool:
        other_ = cast("Image", other)
        if self.date_and_time is None:
            return False
        if other_.date_and_time is None:
            return True
        if self.date_and_time >= other_.date_and_time:
            return True
        return False
