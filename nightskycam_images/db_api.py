"""
Public API for querying the nightskycam image database.

This module provides a high-level, read-only interface for querying
image metadata stored in the SQLite database. It is the DB-backed
counterpart to ``walk.py`` and ``filters.py``, which work directly
on the filesystem.

Typical usage::

    from nightskycam_images.db_api import ImageDB

    db = ImageDB("/path/to/root/.nightskycam_images.db")
    for system in db.systems():
        for date in db.dates(system):
            for img in db.images(system, date):
                print(img.hd_path, img.classifier_scores)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .constants import IMAGE_FILE_FORMATS, THUMBNAIL_DIR_NAME, THUMBNAIL_FILE_FORMAT
from .db import (
    get_classifier_names,
    get_classifier_scores,
    get_dates,
    get_stats,
    get_systems,
    open_db,
    query_images,
)


@dataclass
class ImageRecord:
    """
    A single image record from the database with resolved file paths.

    Attributes correspond to the ``images`` table columns plus resolved
    paths derived from the ``root`` column.
    """

    filename_stem: str
    system: str
    date: str
    time: str
    datetime: str
    root: str
    image_format: Optional[str] = None
    process: Optional[str] = None
    weather: Optional[str] = None
    cloud_cover: Optional[int] = None
    stretched: bool = False
    nightstart_date: Optional[str] = None
    has_thumbnail: bool = False
    has_toml: bool = False
    classifier_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def date_dir(self) -> Path:
        return Path(self.root) / self.system / self.date

    @property
    def hd_path(self) -> Optional[Path]:
        if self.image_format:
            p = self.date_dir / f"{self.filename_stem}.{self.image_format}"
            if p.is_file():
                return p
        for fmt in IMAGE_FILE_FORMATS:
            p = self.date_dir / f"{self.filename_stem}.{fmt}"
            if p.is_file():
                return p
        return None

    @property
    def thumbnail_path(self) -> Optional[Path]:
        p = (
            self.date_dir
            / THUMBNAIL_DIR_NAME
            / f"{self.filename_stem}.{THUMBNAIL_FILE_FORMAT}"
        )
        return p if p.is_file() else None

    @property
    def toml_path(self) -> Optional[Path]:
        p = self.date_dir / f"{self.filename_stem}.toml"
        return p if p.is_file() else None


def _build_query_kwargs(
    system: Optional[str],
    date: Optional[str],
    systems: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    start_time: Optional[str],
    end_time: Optional[str],
    weather: Optional[List[str]],
    cloud_cover_min: Optional[int],
    cloud_cover_max: Optional[int],
    stretched: Optional[bool],
    image_format: Optional[str],
    has_thumbnail: Optional[bool],
    classifier_max: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """Normalize convenience args and build kwargs for ``query_images``."""
    if system is not None:
        if systems is not None:
            raise ValueError("Cannot specify both 'system' and 'systems'")
        systems = [system]

    if date is not None:
        if start_date is not None or end_date is not None:
            raise ValueError(
                "Cannot specify both 'date' and 'start_date'/'end_date'"
            )
        start_date = date
        end_date = date

    kwargs: Dict[str, Any] = {}
    if systems is not None:
        kwargs["systems"] = systems
    if start_date is not None:
        kwargs["start_date"] = start_date
    if end_date is not None:
        kwargs["end_date"] = end_date
    if start_time is not None:
        kwargs["start_time"] = start_time
    if end_time is not None:
        kwargs["end_time"] = end_time
    if weather is not None:
        kwargs["weather"] = weather
    if cloud_cover_min is not None:
        kwargs["cloud_cover_min"] = cloud_cover_min
    if cloud_cover_max is not None:
        kwargs["cloud_cover_max"] = cloud_cover_max
    if stretched is not None:
        kwargs["stretched"] = stretched
    if image_format is not None:
        kwargs["image_format"] = image_format
    if has_thumbnail is not None:
        kwargs["has_thumbnail"] = has_thumbnail
    if classifier_max is not None:
        kwargs["classifier_max"] = classifier_max
    return kwargs


def _row_to_record(
    row: Dict[str, Any], scores: Dict[str, float]
) -> ImageRecord:
    """Convert a raw DB row dict + classifier scores to an ImageRecord."""
    return ImageRecord(
        filename_stem=row["filename_stem"],
        system=row["system"],
        date=row["date"],
        time=row["time"],
        datetime=row["datetime"],
        root=row["root"],
        image_format=row["image_format"],
        process=row["process"],
        weather=row["weather"],
        cloud_cover=row["cloud_cover"],
        stretched=bool(row["stretched"]),
        nightstart_date=row["nightstart_date"],
        has_thumbnail=bool(row["has_thumbnail"]),
        has_toml=bool(row["has_toml"]),
        classifier_scores=scores,
    )


class ImageDB:
    """
    High-level read-only interface to the nightskycam image database.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self._db_path: Path = Path(db_path)

    @property
    def path(self) -> Path:
        return self._db_path

    def systems(self) -> List[str]:
        """Return sorted list of system names."""
        return get_systems(self._db_path)

    def dates(self, system: str) -> List[str]:
        """Return sorted list of date strings (``YYYY_MM_DD``) for a system."""
        return get_dates(self._db_path, system)

    def classifier_names(self) -> List[str]:
        """Return sorted list of all classifier names in the database."""
        return get_classifier_names(self._db_path)

    def stats(self) -> Dict[str, Any]:
        """Return aggregate database statistics."""
        return get_stats(self._db_path)

    def count(
        self,
        system: Optional[str] = None,
        date: Optional[str] = None,
        *,
        systems: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        weather: Optional[List[str]] = None,
        cloud_cover_min: Optional[int] = None,
        cloud_cover_max: Optional[int] = None,
        stretched: Optional[bool] = None,
        image_format: Optional[str] = None,
        has_thumbnail: Optional[bool] = None,
        classifier_max: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Count images matching the given filters.

        Accepts the same arguments as :meth:`images`.
        """
        kwargs = _build_query_kwargs(
            system=system,
            date=date,
            systems=systems,
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,
            end_time=end_time,
            weather=weather,
            cloud_cover_min=cloud_cover_min,
            cloud_cover_max=cloud_cover_max,
            stretched=stretched,
            image_format=image_format,
            has_thumbnail=has_thumbnail,
            classifier_max=classifier_max,
        )
        rows = query_images(self._db_path, **kwargs)
        return len(rows)

    def images(
        self,
        system: Optional[str] = None,
        date: Optional[str] = None,
        *,
        systems: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        weather: Optional[List[str]] = None,
        cloud_cover_min: Optional[int] = None,
        cloud_cover_max: Optional[int] = None,
        stretched: Optional[bool] = None,
        image_format: Optional[str] = None,
        has_thumbnail: Optional[bool] = None,
        classifier_max: Optional[Dict[str, float]] = None,
    ) -> List[ImageRecord]:
        """
        Query images with optional filters.

        Convenience positional arguments ``system`` and ``date`` are
        shortcuts for the common case of querying a single system/date.

        Parameters
        ----------
        system
            Single system name (shortcut for ``systems=[system]``).
        date
            Single date string (shortcut for ``start_date=date, end_date=date``).
        systems
            Filter by system names.
        start_date
            Inclusive start date (``YYYY_MM_DD``).
        end_date
            Inclusive end date (``YYYY_MM_DD``).
        start_time
            Start time (``HH_MM_SS``) for time-of-day filtering.
        end_time
            End time (``HH_MM_SS``). If ``start_time > end_time``, the
            window crosses midnight.
        weather
            Weather substrings (OR logic).
        cloud_cover_min
            Minimum cloud cover (inclusive).
        cloud_cover_max
            Maximum cloud cover (inclusive).
        stretched
            Filter by stretched status.
        image_format
            Filter by format (e.g. ``"jpg"``, ``"tiff"``).
        has_thumbnail
            Filter by thumbnail presence.
        classifier_max
            Dict of ``{classifier_name: max_probability}``.

        Returns
        -------
        List[ImageRecord]
            Matching images with resolved file paths and classifier scores.
        """
        kwargs = _build_query_kwargs(
            system=system,
            date=date,
            systems=systems,
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,
            end_time=end_time,
            weather=weather,
            cloud_cover_min=cloud_cover_min,
            cloud_cover_max=cloud_cover_max,
            stretched=stretched,
            image_format=image_format,
            has_thumbnail=has_thumbnail,
            classifier_max=classifier_max,
        )
        rows = query_images(self._db_path, **kwargs)

        results: List[ImageRecord] = []
        for row in rows:
            scores = get_classifier_scores(
                self._db_path, row["filename_stem"]
            )
            results.append(_row_to_record(row, scores))
        return results

    def image(self, filename_stem: str) -> Optional[ImageRecord]:
        """
        Look up a single image by filename stem.

        Returns
        -------
        Optional[ImageRecord]
            The image record, or None if not found.
        """
        conn = open_db(self._db_path)
        row = conn.execute(
            "SELECT * FROM images WHERE filename_stem = ?",
            (filename_stem,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        scores = get_classifier_scores(self._db_path, filename_stem)
        return _row_to_record(dict(row), scores)

    def hd_paths(
        self,
        system: Optional[str] = None,
        date: Optional[str] = None,
        *,
        systems: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        weather: Optional[List[str]] = None,
        cloud_cover_min: Optional[int] = None,
        cloud_cover_max: Optional[int] = None,
        stretched: Optional[bool] = None,
        image_format: Optional[str] = None,
        has_thumbnail: Optional[bool] = None,
        classifier_max: Optional[Dict[str, float]] = None,
    ) -> List[Path]:
        """
        Return HD image paths for images matching the filters.

        Accepts the same arguments as :meth:`images`.
        Only images whose HD file exists on disk are included.
        """
        return [
            img.hd_path
            for img in self.images(
                system=system,
                date=date,
                systems=systems,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                weather=weather,
                cloud_cover_min=cloud_cover_min,
                cloud_cover_max=cloud_cover_max,
                stretched=stretched,
                image_format=image_format,
                has_thumbnail=has_thumbnail,
                classifier_max=classifier_max,
            )
            if img.hd_path is not None
        ]

    def thumbnail_paths(
        self,
        system: Optional[str] = None,
        date: Optional[str] = None,
        *,
        systems: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        weather: Optional[List[str]] = None,
        cloud_cover_min: Optional[int] = None,
        cloud_cover_max: Optional[int] = None,
        stretched: Optional[bool] = None,
        image_format: Optional[str] = None,
        has_thumbnail: Optional[bool] = None,
        classifier_max: Optional[Dict[str, float]] = None,
    ) -> List[Path]:
        """
        Return thumbnail paths for images matching the filters.

        Accepts the same arguments as :meth:`images`.
        Only images whose thumbnail exists on disk are included.
        """
        return [
            img.thumbnail_path
            for img in self.images(
                system=system,
                date=date,
                systems=systems,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                weather=weather,
                cloud_cover_min=cloud_cover_min,
                cloud_cover_max=cloud_cover_max,
                stretched=stretched,
                image_format=image_format,
                has_thumbnail=has_thumbnail,
                classifier_max=classifier_max,
            )
            if img.thumbnail_path is not None
        ]
