"""
SQLite database for fast querying of nightskycam image metadata.

Instead of walking the filesystem and reading TOML files for every query,
this module maintains a database that mirrors the metadata. The database
is populated by scanning the filesystem and can be refreshed at any time.
"""

import datetime as dt
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tomli
from loguru import logger

from .constants import (
    DATE_FORMAT_FILE,
    IMAGE_FILE_FORMATS,
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
)
from .walk import parse_image_path, walk_dates, walk_systems


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    root            TEXT    NOT NULL,
    system          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    time            TEXT    NOT NULL,
    datetime        TEXT    NOT NULL,
    nightstart_date TEXT,
    filename_stem   TEXT    NOT NULL UNIQUE,
    image_format    TEXT,
    process         TEXT,
    weather         TEXT,
    cloud_cover     INTEGER,
    stretched       INTEGER NOT NULL DEFAULT 0,
    has_thumbnail   INTEGER NOT NULL DEFAULT 0,
    has_toml        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS classifier_scores (
    image_id        INTEGER NOT NULL,
    classifier_name TEXT    NOT NULL,
    probability     REAL    NOT NULL,
    PRIMARY KEY (image_id, classifier_name),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_images_system          ON images(system);
CREATE INDEX IF NOT EXISTS idx_images_date            ON images(date);
CREATE INDEX IF NOT EXISTS idx_images_datetime        ON images(datetime);
CREATE INDEX IF NOT EXISTS idx_images_system_date     ON images(system, date);
CREATE INDEX IF NOT EXISTS idx_images_weather         ON images(weather);
CREATE INDEX IF NOT EXISTS idx_images_cloud_cover     ON images(cloud_cover);
CREATE INDEX IF NOT EXISTS idx_images_nightstart_date ON images(nightstart_date);
CREATE INDEX IF NOT EXISTS idx_scores_classifier      ON classifier_scores(classifier_name, probability);

CREATE TABLE IF NOT EXISTS scan_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_default_db_path(root: Path) -> Path:
    """
    Return the default database path for a media root directory.

    Parameters
    ----------
    root
        Path to media root directory.

    Returns
    -------
    Path
        Default database file path: ``root / '.nightskycam_images.db'``.
    """
    return root / ".nightskycam_images.db"


def open_db(db_path: Path) -> sqlite3.Connection:
    """
    Open a database connection and ensure tables exist.

    Enables WAL mode for concurrent read access and foreign key enforcement.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        Open database connection with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def _compute_nightstart_date(datetime_obj: dt.datetime) -> str:
    """Compute nightstart date: if before noon, assign to previous day."""
    if datetime_obj.hour < 12:
        return (datetime_obj - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    return datetime_obj.strftime("%Y-%m-%d")


def _detect_image_format(dir_path: Path, filename_stem: str) -> Optional[str]:
    """Find which image format exists on disk for a given filename stem."""
    for fmt in IMAGE_FILE_FORMATS:
        if (dir_path / f"{filename_stem}.{fmt}").is_file():
            return fmt
    return None


def _has_thumbnail(dir_path: Path, filename_stem: str) -> bool:
    """Check if a thumbnail exists for the given image."""
    thumb = dir_path / THUMBNAIL_DIR_NAME / f"{filename_stem}.{THUMBNAIL_FILE_FORMAT}"
    return thumb.is_file()


def _read_toml_metadata(
    dir_path: Path, filename_stem: str
) -> Optional[Dict[str, Any]]:
    """Read the TOML metadata file for an image. Returns None if missing/unreadable."""
    toml_path = dir_path / f"{filename_stem}.toml"
    if not toml_path.is_file():
        return None
    try:
        with open(toml_path, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        logger.debug(f"Could not read TOML {toml_path}: {e}")
        return None


def _get_last_scan_timestamp(conn: sqlite3.Connection) -> Optional[float]:
    """Get the last scan timestamp from the database."""
    row = conn.execute(
        "SELECT value FROM scan_metadata WHERE key = 'last_scan_timestamp'"
    ).fetchone()
    return float(row["value"]) if row else None


def _set_last_scan_timestamp(conn: sqlite3.Connection) -> None:
    """Store the current time as the last scan timestamp."""
    now = str(time.time())
    conn.execute(
        "INSERT OR REPLACE INTO scan_metadata (key, value) "
        "VALUES ('last_scan_timestamp', ?)",
        (now,),
    )
    conn.commit()


def populate(
    roots: Union[Path, List[Path]],
    db_path: Optional[Path] = None,
    full: bool = False,
) -> Dict[str, int]:
    """
    Scan the filesystem and populate the database with image metadata.

    For each image found, parses the filename, checks for HD file / thumbnail /
    TOML metadata, and upserts into the database. Classifier scores from the
    ``[classifiers]`` TOML section are also stored.

    Safe to run multiple times — uses ``INSERT OR REPLACE`` for idempotency.

    In incremental mode (default), date directories whose mtime is older than
    the last scan timestamp are skipped. Use ``full=True`` to force a complete
    rescan.

    Supports one or two root directories. For a given system and date, images
    are expected to be in only one of the roots.

    Parameters
    ----------
    roots
        Path to media root directory, or list of root directories.
    db_path
        Path to the SQLite database file. Defaults to
        ``first_root / '.nightskycam_images.db'``.
    full
        If True, rescan all directories regardless of mtime.

    Returns
    -------
    Dict[str, int]
        Statistics: images_scanned, images_upserted, classifiers_upserted,
        errors, folders_scanned, folders_skipped.
    """
    if isinstance(roots, Path):
        root_list = [roots]
    else:
        root_list = list(roots)

    if db_path is None:
        db_path = get_default_db_path(root_list[0])

    conn = open_db(db_path)

    stats = {
        "images_scanned": 0,
        "images_upserted": 0,
        "classifiers_upserted": 0,
        "errors": 0,
        "folders_scanned": 0,
        "folders_skipped": 0,
    }

    last_scan = _get_last_scan_timestamp(conn)

    for root in root_list:
        root_str = str(root)
        logger.info(f"Scanning root: {root_str}")

        for system_path in walk_systems(root):
            system_name = system_path.name
            logger.info(f"Processing system: {system_name}")

            for date_, date_path in walk_dates(system_path):
                date_str = date_.strftime(DATE_FORMAT_FILE)

                # Incremental mode: skip unchanged directories
                if not full and last_scan is not None:
                    dir_mtime = date_path.stat().st_mtime
                    if dir_mtime < last_scan:
                        stats["folders_skipped"] += 1
                        logger.debug(f"  Skipping unchanged date: {date_str}")
                        continue

                stats["folders_scanned"] += 1
                logger.debug(f"  Processing date: {date_str}")

                # Discover all image files in this date directory
                seen_stems = set()
                for fmt in IMAGE_FILE_FORMATS:
                    for f in date_path.glob(f"*.{fmt}"):
                        seen_stems.add(f.stem)

                # Also check for TOML files without a corresponding image
                # (thumbnail-only entries)
                thumb_dir = date_path / THUMBNAIL_DIR_NAME
                if thumb_dir.is_dir():
                    for f in thumb_dir.glob(f"*.{THUMBNAIL_FILE_FORMAT}"):
                        seen_stems.add(f.stem)

                if not seen_stems:
                    continue

                # Batch upsert per date folder
                with conn:
                    for stem in seen_stems:
                        stats["images_scanned"] += 1
                        try:
                            system_parsed, datetime_obj = parse_image_path(
                                Path(f"{stem}.tmp")
                            )
                        except Exception as e:
                            logger.debug(
                                f"    Could not parse filename '{stem}': {e}"
                            )
                            stats["errors"] += 1
                            continue

                        time_str = datetime_obj.strftime("%H_%M_%S")
                        iso_datetime = datetime_obj.isoformat()
                        nightstart = _compute_nightstart_date(datetime_obj)
                        image_format = _detect_image_format(date_path, stem)
                        has_thumb = _has_thumbnail(date_path, stem)

                        meta = _read_toml_metadata(date_path, stem)
                        has_toml = meta is not None
                        process = meta.get("process") if meta else None
                        weather = meta.get("weather") if meta else None
                        cloud_cover = meta.get("cloud_cover") if meta else None
                        stretched = (
                            "stretching" in process if process else False
                        )
                        classifiers = (
                            meta.get("classifiers", {}) if meta else {}
                        )

                        # Validate cloud_cover type
                        if cloud_cover is not None and not isinstance(
                            cloud_cover, int
                        ):
                            try:
                                cloud_cover = int(cloud_cover)
                            except (ValueError, TypeError):
                                cloud_cover = None

                        conn.execute(
                            """
                            INSERT OR REPLACE INTO images
                                (root, system, date, time, datetime,
                                 nightstart_date, filename_stem, image_format,
                                 process, weather, cloud_cover, stretched,
                                 has_thumbnail, has_toml)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                root_str,
                                system_name,
                                date_str,
                                time_str,
                                iso_datetime,
                                nightstart,
                                stem,
                                image_format,
                                process,
                                weather,
                                cloud_cover,
                                int(stretched),
                                int(has_thumb),
                                int(has_toml),
                            ),
                        )
                        stats["images_upserted"] += 1

                        # Upsert classifier scores
                        if classifiers:
                            # Get the image id
                            row = conn.execute(
                                "SELECT id FROM images WHERE filename_stem = ?",
                                (stem,),
                            ).fetchone()
                            if row:
                                image_id = row["id"]
                                for name, prob in classifiers.items():
                                    try:
                                        conn.execute(
                                            """
                                            INSERT OR REPLACE INTO
                                                classifier_scores
                                                (image_id, classifier_name,
                                                 probability)
                                            VALUES (?, ?, ?)
                                            """,
                                            (image_id, name, float(prob)),
                                        )
                                        stats["classifiers_upserted"] += 1
                                    except Exception as e:
                                        logger.debug(
                                            f"    Error upserting classifier "
                                            f"score '{name}' for '{stem}': {e}"
                                        )
                                        stats["errors"] += 1

    _set_last_scan_timestamp(conn)
    conn.close()
    return stats


def query_images(
    db_path: Path,
    systems: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    weather: Optional[List[str]] = None,
    cloud_cover_min: Optional[int] = None,
    cloud_cover_max: Optional[int] = None,
    process_substring: Optional[str] = None,
    stretched: Optional[bool] = None,
    has_thumbnail: Optional[bool] = None,
    has_toml: Optional[bool] = None,
    image_format: Optional[str] = None,
    classifier_max: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Query images with flexible filtering.

    All filter parameters are optional. When multiple are provided, they are
    combined with AND logic.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.
    systems
        Filter by system names.
    start_date
        Inclusive start date in ``YYYY_MM_DD`` format.
    end_date
        Inclusive end date in ``YYYY_MM_DD`` format.
    start_time
        Start time in ``HH_MM_SS`` format for time-of-day filtering.
    end_time
        End time in ``HH_MM_SS`` format for time-of-day filtering.
        If start_time > end_time, the window is treated as crossing midnight.
    weather
        List of weather substrings (OR logic among them).
    cloud_cover_min
        Minimum cloud cover (inclusive).
    cloud_cover_max
        Maximum cloud cover (inclusive).
    process_substring
        Substring that must appear in the process field.
    stretched
        If set, filter by whether the image was stretched.
    has_thumbnail
        If set, filter by thumbnail presence.
    has_toml
        If set, filter by TOML metadata presence.
    image_format
        Filter by image format (e.g. ``"jpg"``, ``"tiff"``).
    classifier_max
        Dict of ``{classifier_name: max_probability}``. Only images with
        scores at or below the given thresholds are returned.

    Returns
    -------
    List[Dict[str, Any]]
        List of matching image rows as dictionaries.
    """
    conn = open_db(db_path)

    clauses: List[str] = []
    params: List[Any] = []

    if systems is not None:
        placeholders = ",".join("?" for _ in systems)
        clauses.append(f"system IN ({placeholders})")
        params.extend(systems)

    if start_date is not None:
        clauses.append("date >= ?")
        params.append(start_date)

    if end_date is not None:
        clauses.append("date <= ?")
        params.append(end_date)

    if start_time is not None and end_time is not None:
        if start_time <= end_time:
            clauses.append("time >= ? AND time <= ?")
            params.extend([start_time, end_time])
        else:
            # Crosses midnight
            clauses.append("(time >= ? OR time <= ?)")
            params.extend([start_time, end_time])
    elif start_time is not None:
        clauses.append("time >= ?")
        params.append(start_time)
    elif end_time is not None:
        clauses.append("time <= ?")
        params.append(end_time)

    if weather is not None:
        weather_clauses = ["weather LIKE ?" for _ in weather]
        clauses.append(f"({' OR '.join(weather_clauses)})")
        params.extend(f"%{w}%" for w in weather)

    if cloud_cover_min is not None:
        clauses.append("cloud_cover >= ?")
        params.append(cloud_cover_min)

    if cloud_cover_max is not None:
        clauses.append("cloud_cover <= ?")
        params.append(cloud_cover_max)

    if process_substring is not None:
        clauses.append("process LIKE ?")
        params.append(f"%{process_substring}%")

    if stretched is not None:
        clauses.append("stretched = ?")
        params.append(int(stretched))

    if has_thumbnail is not None:
        clauses.append("has_thumbnail = ?")
        params.append(int(has_thumbnail))

    if has_toml is not None:
        clauses.append("has_toml = ?")
        params.append(int(has_toml))

    if image_format is not None:
        clauses.append("image_format = ?")
        params.append(image_format)

    if classifier_max is not None:
        for name, max_val in classifier_max.items():
            clauses.append(
                "id IN (SELECT image_id FROM classifier_scores "
                "WHERE classifier_name = ? AND probability <= ?)"
            )
            params.extend([name, max_val])

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM images{where} ORDER BY system, date, time"

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_systems(db_path: Path) -> List[str]:
    """
    Get all distinct system names from the database.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.

    Returns
    -------
    List[str]
        Sorted list of system names.
    """
    conn = open_db(db_path)
    rows = conn.execute(
        "SELECT DISTINCT system FROM images ORDER BY system"
    ).fetchall()
    conn.close()
    return [row["system"] for row in rows]


def get_dates(db_path: Path, system: str) -> List[str]:
    """
    Get all distinct dates for a system.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.
    system
        System name.

    Returns
    -------
    List[str]
        Sorted list of dates in ``YYYY_MM_DD`` format.
    """
    conn = open_db(db_path)
    rows = conn.execute(
        "SELECT DISTINCT date FROM images WHERE system = ? ORDER BY date",
        (system,),
    ).fetchall()
    conn.close()
    return [row["date"] for row in rows]


def get_classifier_names(db_path: Path) -> List[str]:
    """Get all distinct classifier names from the database."""
    conn = open_db(db_path)
    rows = conn.execute(
        "SELECT DISTINCT classifier_name FROM classifier_scores "
        "ORDER BY classifier_name"
    ).fetchall()
    conn.close()
    return [row["classifier_name"] for row in rows]


def get_classifier_scores(
    db_path: Path, filename_stem: str
) -> Dict[str, float]:
    """
    Get all classifier scores for a single image.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.
    filename_stem
        Unique filename stem of the image.

    Returns
    -------
    Dict[str, float]
        Mapping of classifier name to probability score.
    """
    conn = open_db(db_path)
    row = conn.execute(
        "SELECT id FROM images WHERE filename_stem = ?", (filename_stem,)
    ).fetchone()
    if row is None:
        conn.close()
        return {}
    rows = conn.execute(
        "SELECT classifier_name, probability FROM classifier_scores WHERE image_id = ?",
        (row["id"],),
    ).fetchall()
    conn.close()
    return {r["classifier_name"]: r["probability"] for r in rows}


def get_stats(db_path: Path) -> Dict[str, Any]:
    """
    Get aggregate statistics from the database.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.

    Returns
    -------
    Dict[str, Any]
        Statistics including total_images, per-system counts,
        weather distribution, and missing metadata counts.
    """
    conn = open_db(db_path)

    total = conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()["cnt"]

    # Per-system stats
    systems: Dict[str, Any] = {}
    for row in conn.execute(
        """
        SELECT system, COUNT(*) as cnt,
               MIN(date) as min_date, MAX(date) as max_date
        FROM images GROUP BY system ORDER BY system
        """
    ).fetchall():
        systems[row["system"]] = {
            "image_count": row["cnt"],
            "date_range": (row["min_date"], row["max_date"]),
        }

    # Weather distribution
    weather_dist: Dict[str, int] = {}
    for row in conn.execute(
        """
        SELECT weather, COUNT(*) as cnt FROM images
        WHERE weather IS NOT NULL GROUP BY weather ORDER BY cnt DESC
        """
    ).fetchall():
        weather_dist[row["weather"]] = row["cnt"]

    # Missing metadata
    no_toml = conn.execute(
        "SELECT COUNT(*) as cnt FROM images WHERE has_toml = 0"
    ).fetchone()["cnt"]
    no_process = conn.execute(
        "SELECT COUNT(*) as cnt FROM images WHERE process IS NULL"
    ).fetchone()["cnt"]
    no_weather = conn.execute(
        "SELECT COUNT(*) as cnt FROM images WHERE weather IS NULL"
    ).fetchone()["cnt"]
    no_cloud_cover = conn.execute(
        "SELECT COUNT(*) as cnt FROM images WHERE cloud_cover IS NULL"
    ).fetchone()["cnt"]

    conn.close()

    return {
        "total_images": total,
        "systems": systems,
        "weather_distribution": weather_dist,
        "missing_metadata": {
            "no_toml": no_toml,
            "no_process": no_process,
            "no_weather": no_weather,
            "no_cloud_cover": no_cloud_cover,
        },
    }
