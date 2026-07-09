"""
Home-page dashboard data: ``ImageDB.stats()`` plus a few extra
read-only aggregations (format breakdown, classifier coverage, last
scan time), with a short in-process cache so the home page stays snappy.
"""

import datetime as dt
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Tuple

from ..db import open_db_readonly
from ..db_api import ImageDB

CACHE_TTL_SECONDS = 300

_cache_lock = threading.Lock()
_cache: Dict[Path, Tuple[float, Dict[str, Any]]] = {}


def dashboard_stats(db_path: Path) -> Dict[str, Any]:
    """Aggregate statistics for the home page (cached ~5 minutes)."""
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(db_path)
        if entry is not None and now - entry[0] < CACHE_TTL_SECONDS:
            return entry[1]

    stats = _compute(db_path)
    with _cache_lock:
        _cache[db_path] = (now, stats)
    return stats


def _compute(db_path: Path) -> Dict[str, Any]:
    stats = ImageDB(db_path).stats()

    conn = open_db_readonly(db_path)
    try:
        format_counts = {
            row["image_format"] or "unknown": row["n"]
            for row in conn.execute(
                "SELECT image_format, COUNT(*) AS n FROM images "
                "GROUP BY image_format ORDER BY n DESC"
            )
        }
        classifier_coverage = {
            row["classifier_name"]: row["n"]
            for row in conn.execute(
                "SELECT classifier_name, COUNT(*) AS n FROM classifier_scores "
                "GROUP BY classifier_name ORDER BY classifier_name"
            )
        }
        date_bounds = conn.execute(
            "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM images"
        ).fetchone()
        newest_per_system = {
            row["system"]: row["max_date"]
            for row in conn.execute(
                "SELECT system, MAX(date) AS max_date FROM images GROUP BY system"
            )
        }
        row = conn.execute(
            "SELECT value FROM scan_metadata WHERE key = 'last_scan_timestamp'"
        ).fetchone()
    finally:
        conn.close()

    stats["format_counts"] = format_counts
    stats["classifier_coverage"] = classifier_coverage
    stats["date_min"] = date_bounds["min_date"]
    stats["date_max"] = date_bounds["max_date"]
    stats["newest_per_system"] = newest_per_system
    stats["last_scan"] = _format_timestamp(row["value"] if row else None)
    return stats


_meta_cache: Dict[Path, Tuple[float, Dict[str, Any]]] = {}


def site_meta(db_path: Path) -> Dict[str, Any]:
    """
    Vocabulary for populating the filter form and the API's ``/api/meta``:
    systems, classifier names, distinct weather values, date bounds and
    the formats present. Cached ~5 minutes.
    """
    now = time.monotonic()
    with _cache_lock:
        entry = _meta_cache.get(db_path)
        if entry is not None and now - entry[0] < CACHE_TTL_SECONDS:
            return entry[1]

    db = ImageDB(db_path)
    conn = open_db_readonly(db_path)
    try:
        weather_values = [
            row["weather"]
            for row in conn.execute(
                "SELECT DISTINCT weather FROM images "
                "WHERE weather IS NOT NULL ORDER BY weather"
            )
        ]
        formats = [
            row["image_format"]
            for row in conn.execute(
                "SELECT DISTINCT image_format FROM images "
                "WHERE image_format IS NOT NULL ORDER BY image_format"
            )
        ]
        date_bounds = conn.execute(
            "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM images"
        ).fetchone()
        try:
            locations = [
                row["location"]
                for row in conn.execute(
                    "SELECT DISTINCT location FROM images "
                    "WHERE location IS NOT NULL ORDER BY location"
                )
            ]
        except sqlite3.OperationalError:
            # Database not yet migrated (no location column).
            locations = []
    finally:
        conn.close()

    meta = {
        "systems": db.systems(),
        "classifier_names": db.classifier_names(),
        "weather_values": weather_values,
        "image_formats": formats,
        "locations": locations,
        "date_min": date_bounds["min_date"],
        "date_max": date_bounds["max_date"],
    }
    with _cache_lock:
        _meta_cache[db_path] = (now, meta)
    return meta


def _format_timestamp(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        timestamp = float(value)
    except ValueError:
        return value
    return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
