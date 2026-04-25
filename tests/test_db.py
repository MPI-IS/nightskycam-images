"""
Tests for the SQLite database module.
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest
import tomli_w

from nightskycam_images.constants import (
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
)
from nightskycam_images.db import (
    get_classifier_scores,
    get_dates,
    get_default_db_path,
    get_stats,
    get_systems,
    open_db,
    populate,
    query_images,
)


def _create_media_tree(
    root: Path,
    structure: Dict[str, Dict[str, List[str]]],
    metadata: Dict[str, Dict] = None,
) -> None:
    """
    Create a nightskycam media directory tree for testing.

    Parameters
    ----------
    root
        Root directory.
    structure
        {system: {date_str: [time_str, ...]}}
    metadata
        Optional per-stem metadata dicts: {filename_stem: {key: value, ...}}
    """
    if metadata is None:
        metadata = {}

    for system, dates in structure.items():
        for date_str, times in dates.items():
            date_dir = root / system / date_str
            date_dir.mkdir(parents=True)
            thumb_dir = date_dir / THUMBNAIL_DIR_NAME
            thumb_dir.mkdir()

            for time_str in times:
                stem = f"{system}_{date_str}_{time_str}"

                # Create HD image (jpg).
                img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
                cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)

                # Create thumbnail.
                thumb = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
                cv2.imwrite(
                    str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), thumb
                )

                # Create TOML metadata.
                meta = metadata.get(stem, {"process": "raw", "weather": "clear"})
                with open(date_dir / f"{stem}.toml", "wb") as f:
                    tomli_w.dump(meta, f)


# ============================================================================
# Schema tests
# ============================================================================


def test_open_db_creates_tables():
    """open_db creates images and classifier_scores tables."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = open_db(db_path)

        # Check tables exist.
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        assert "images" in table_names
        assert "classifier_scores" in table_names

        conn.close()


def test_open_db_creates_indexes():
    """open_db creates the expected indexes."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = open_db(db_path)

        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        index_names = {row["name"] for row in indexes}
        assert "idx_images_system" in index_names
        assert "idx_images_date" in index_names
        assert "idx_images_system_date" in index_names
        assert "idx_scores_classifier" in index_names

        conn.close()


def test_get_default_db_path():
    """Default DB path is root/.nightskycam_images.db."""
    root = Path("/some/media/root")
    assert get_default_db_path(root) == root / ".nightskycam_images.db"


# ============================================================================
# Populate tests
# ============================================================================


def test_populate_basic():
    """Populate inserts correct number of rows with correct fields."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {
                "cam1": {"2025_06_15": ["20_00_00", "21_00_00"]},
                "cam2": {"2025_06_16": ["22_00_00"]},
            },
            metadata={
                "cam1_2025_06_15_20_00_00": {
                    "process": "auto-stretching",
                    "weather": "clear",
                    "cloud_cover": 15,
                },
                "cam1_2025_06_15_21_00_00": {
                    "process": "raw",
                    "weather": "partly cloudy",
                    "cloud_cover": 60,
                },
                "cam2_2025_06_16_22_00_00": {
                    "process": "raw",
                    "weather": "rain",
                    "cloud_cover": 95,
                },
            },
        )

        stats = populate(root, db_path)
        assert stats["images_upserted"] == 3
        assert stats["errors"] == 0

        # Verify row content.
        conn = open_db(db_path)
        rows = conn.execute(
            "SELECT * FROM images ORDER BY filename_stem"
        ).fetchall()
        assert len(rows) == 3

        row0 = dict(rows[0])
        assert row0["root"] == str(root)
        assert row0["system"] == "cam1"
        assert row0["date"] == "2025_06_15"
        assert row0["time"] == "20_00_00"
        assert row0["process"] == "auto-stretching"
        assert row0["weather"] == "clear"
        assert row0["cloud_cover"] == 15
        assert row0["stretched"] == 1
        assert row0["has_thumbnail"] == 1
        assert row0["has_toml"] == 1
        assert row0["image_format"] == "jpg"

        # "raw" process → not stretched
        row1 = dict(rows[1])
        assert row1["stretched"] == 0

        conn.close()


def test_populate_idempotent():
    """Running populate twice (full) does not duplicate rows."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {"cam1": {"2025_06_15": ["20_00_00", "21_00_00"]}},
        )

        stats1 = populate(root, db_path)
        stats2 = populate(root, db_path, full=True)

        assert stats1["images_upserted"] == 2
        assert stats2["images_upserted"] == 2

        conn = open_db(db_path)
        count = conn.execute("SELECT COUNT(*) as cnt FROM images").fetchone()[
            "cnt"
        ]
        assert count == 2
        conn.close()


def test_populate_classifier_scores():
    """Classifier scores from TOML [classifiers] section are stored."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {"cam1": {"2025_06_15": ["20_00_00"]}},
            metadata={
                "cam1_2025_06_15_20_00_00": {
                    "process": "raw",
                    "classifiers": {"quality": 0.87, "clouds": 0.23},
                },
            },
        )

        stats = populate(root, db_path)
        assert stats["classifiers_upserted"] == 2

        scores = get_classifier_scores(db_path, "cam1_2025_06_15_20_00_00")
        assert scores == pytest.approx({"quality": 0.87, "clouds": 0.23})


def test_populate_no_toml():
    """Images without TOML metadata still get a row with has_toml=0."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()

        # Create image without TOML.
        date_dir = root / "cam1" / "2025_06_15"
        date_dir.mkdir(parents=True)
        thumb_dir = date_dir / THUMBNAIL_DIR_NAME
        thumb_dir.mkdir()

        stem = "cam1_2025_06_15_20_00_00"
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)
        cv2.imwrite(
            str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), img
        )
        # No .toml file.

        db_path = Path(tmp) / "test.db"
        populate(root, db_path)

        conn = open_db(db_path)
        row = conn.execute(
            "SELECT * FROM images WHERE filename_stem = ?", (stem,)
        ).fetchone()
        assert row is not None
        assert row["has_toml"] == 0
        assert row["process"] is None
        assert row["weather"] is None
        conn.close()


def test_populate_nightstart_date():
    """nightstart_date is the previous day for images taken before noon."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {
                "cam1": {
                    "2025_06_15": ["22_00_00"],  # evening → same date
                    "2025_06_16": ["03_00_00"],  # early morning → prev date
                },
            },
        )

        populate(root, db_path)

        conn = open_db(db_path)
        evening = conn.execute(
            "SELECT nightstart_date FROM images WHERE time = '22_00_00'"
        ).fetchone()
        assert evening["nightstart_date"] == "2025-06-15"

        morning = conn.execute(
            "SELECT nightstart_date FROM images WHERE time = '03_00_00'"
        ).fetchone()
        assert morning["nightstart_date"] == "2025-06-15"  # prev day
        conn.close()


# ============================================================================
# Query tests
# ============================================================================


@pytest.fixture
def populated_db():
    """Create a populated DB with diverse data for query tests."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {
                "cam1": {
                    "2025_01_10": ["20_00_00", "22_30_00"],
                    "2025_03_15": ["01_00_00"],
                },
                "cam2": {
                    "2025_01_10": ["19_00_00"],
                    "2025_06_20": ["23_00_00"],
                },
            },
            metadata={
                "cam1_2025_01_10_20_00_00": {
                    "process": "auto-stretching 8bits",
                    "weather": "clear",
                    "cloud_cover": 10,
                },
                "cam1_2025_01_10_22_30_00": {
                    "process": "raw",
                    "weather": "partly cloudy",
                    "cloud_cover": 50,
                },
                "cam1_2025_03_15_01_00_00": {
                    "process": "auto-stretching",
                    "weather": "rain",
                    "cloud_cover": 90,
                },
                "cam2_2025_01_10_19_00_00": {
                    "process": "raw",
                    "weather": "clear",
                    "cloud_cover": 5,
                },
                "cam2_2025_06_20_23_00_00": {
                    "process": "auto-stretching 8bits",
                    "weather": "light rain",
                    "cloud_cover": 80,
                    "classifiers": {"quality": 0.92},
                },
            },
        )

        populate(root, db_path)
        yield db_path


def test_query_all(populated_db):
    """Query with no filters returns all images."""
    results = query_images(populated_db)
    assert len(results) == 5


def test_query_by_system(populated_db):
    """Filter by system name."""
    results = query_images(populated_db, systems=["cam1"])
    assert len(results) == 3
    assert all(r["system"] == "cam1" for r in results)


def test_query_by_multiple_systems(populated_db):
    """Filter by multiple system names."""
    results = query_images(populated_db, systems=["cam1", "cam2"])
    assert len(results) == 5


def test_query_by_date_range(populated_db):
    """Filter by date range."""
    results = query_images(
        populated_db, start_date="2025_01_10", end_date="2025_01_10"
    )
    assert len(results) == 3  # 2 from cam1 + 1 from cam2 on that date


def test_query_by_start_date_only(populated_db):
    """Filter with start_date only."""
    results = query_images(populated_db, start_date="2025_06_01")
    assert len(results) == 1
    assert results[0]["system"] == "cam2"
    assert results[0]["date"] == "2025_06_20"


def test_query_by_time_window_normal(populated_db):
    """Filter by time window (not crossing midnight)."""
    results = query_images(
        populated_db, start_time="19_00_00", end_time="21_00_00"
    )
    assert len(results) == 2  # 20:00 and 19:00


def test_query_by_time_window_crossing_midnight(populated_db):
    """Filter by time window crossing midnight (e.g., 22:00 to 04:00)."""
    results = query_images(
        populated_db, start_time="22_00_00", end_time="04_00_00"
    )
    # 22:30, 23:00, 01:00 — all match
    assert len(results) == 3


def test_query_by_weather(populated_db):
    """Filter by weather substring."""
    results = query_images(populated_db, weather=["clear"])
    assert len(results) == 2

    results_rain = query_images(populated_db, weather=["rain"])
    assert len(results_rain) == 2  # "rain" and "light rain"


def test_query_by_cloud_cover_range(populated_db):
    """Filter by cloud cover range."""
    results = query_images(
        populated_db, cloud_cover_min=0, cloud_cover_max=30
    )
    assert len(results) == 2  # 10 and 5


def test_query_by_process_substring(populated_db):
    """Filter by process substring."""
    results = query_images(populated_db, process_substring="stretching")
    assert len(results) == 3  # "auto-stretching 8bits" x2, "auto-stretching" x1


def test_query_combined_filters(populated_db):
    """Multiple filters are combined with AND logic."""
    results = query_images(
        populated_db,
        systems=["cam1"],
        weather=["clear"],
        cloud_cover_max=20,
    )
    assert len(results) == 1
    assert results[0]["filename_stem"] == "cam1_2025_01_10_20_00_00"


def test_query_stretched(populated_db):
    """Filter by stretched flag."""
    results = query_images(populated_db, stretched=True)
    assert len(results) == 3  # "auto-stretching 8bits" x2, "auto-stretching" x1
    assert all(r["stretched"] == 1 for r in results)

    results_raw = query_images(populated_db, stretched=False)
    assert len(results_raw) == 2
    assert all(r["stretched"] == 0 for r in results_raw)


def test_query_has_thumbnail(populated_db):
    """Filter by thumbnail presence."""
    results = query_images(populated_db, has_thumbnail=True)
    assert len(results) == 5  # All images in fixture have thumbnails


# ============================================================================
# Helper function tests
# ============================================================================


def test_get_systems(populated_db):
    """get_systems returns sorted distinct system names."""
    systems = get_systems(populated_db)
    assert systems == ["cam1", "cam2"]


def test_get_dates(populated_db):
    """get_dates returns sorted dates for a system."""
    dates = get_dates(populated_db, "cam1")
    assert dates == ["2025_01_10", "2025_03_15"]


def test_get_classifier_scores_found(populated_db):
    """get_classifier_scores returns scores for image with classifiers."""
    scores = get_classifier_scores(
        populated_db, "cam2_2025_06_20_23_00_00"
    )
    assert scores == pytest.approx({"quality": 0.92})


def test_get_classifier_scores_not_found(populated_db):
    """get_classifier_scores returns empty dict for image without classifiers."""
    scores = get_classifier_scores(
        populated_db, "cam1_2025_01_10_20_00_00"
    )
    assert scores == {}


def test_get_classifier_scores_unknown_image(populated_db):
    """get_classifier_scores returns empty dict for unknown filename."""
    scores = get_classifier_scores(populated_db, "nonexistent_image")
    assert scores == {}


def test_get_stats(populated_db):
    """get_stats returns correct aggregate statistics."""
    stats = get_stats(populated_db)
    assert stats["total_images"] == 5
    assert "cam1" in stats["systems"]
    assert "cam2" in stats["systems"]
    assert stats["systems"]["cam1"]["image_count"] == 3
    assert stats["systems"]["cam2"]["image_count"] == 2
    assert stats["weather_distribution"]["clear"] == 2
    assert stats["missing_metadata"]["no_toml"] == 0


# ============================================================================
# Incremental populate tests
# ============================================================================


def test_populate_first_run_scans_all():
    """First run with no stored timestamp scans all folders."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {
                "cam1": {"2025_06_15": ["20_00_00"]},
                "cam2": {"2025_06_16": ["21_00_00"]},
            },
        )

        stats = populate(root, db_path)
        assert stats["folders_scanned"] == 2
        assert stats["folders_skipped"] == 0
        assert stats["images_upserted"] == 2


def test_populate_incremental_skips_unchanged():
    """Incremental populate skips directories unchanged since last scan."""
    import os
    import time

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {
                "cam1": {
                    "2025_06_15": ["20_00_00"],
                    "2025_06_16": ["21_00_00"],
                },
            },
        )

        # First populate: scans everything.
        stats1 = populate(root, db_path)
        assert stats1["folders_scanned"] == 2
        assert stats1["folders_skipped"] == 0

        # Wait briefly so the timestamp is clearly after the first scan.
        time.sleep(0.05)

        # Modify only one date directory by adding a new image.
        date_dir = root / "cam1" / "2025_06_16"
        stem = "cam1_2025_06_16_22_00_00"
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)

        # Second populate (incremental): only the modified folder is scanned.
        stats2 = populate(root, db_path)
        assert stats2["folders_skipped"] == 1
        assert stats2["folders_scanned"] == 1

        # The new image was added.
        conn = open_db(db_path)
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM images"
        ).fetchone()["cnt"]
        assert count == 3
        conn.close()


def test_populate_full_rescans_all():
    """populate(full=True) rescans all directories regardless of timestamp."""
    import time

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root,
            {
                "cam1": {
                    "2025_06_15": ["20_00_00"],
                    "2025_06_16": ["21_00_00"],
                },
            },
        )

        # First populate sets the timestamp.
        populate(root, db_path)

        time.sleep(0.05)

        # Full rescan: no folders skipped even though nothing changed.
        stats = populate(root, db_path, full=True)
        assert stats["folders_scanned"] == 2
        assert stats["folders_skipped"] == 0
        assert stats["images_upserted"] == 2


# ============================================================================
# Multi-root tests
# ============================================================================


def test_populate_two_roots():
    """Populate with two roots indexes images from both."""
    with tempfile.TemporaryDirectory() as tmp:
        root1 = Path(tmp) / "media1"
        root1.mkdir()
        root2 = Path(tmp) / "media2"
        root2.mkdir()
        db_path = Path(tmp) / "test.db"

        # cam1 has older dates in root1, newer dates in root2.
        _create_media_tree(
            root1,
            {"cam1": {"2025_01_10": ["20_00_00", "21_00_00"]}},
        )
        _create_media_tree(
            root2,
            {"cam1": {"2025_06_15": ["22_00_00"]}},
        )

        stats = populate([root1, root2], db_path)
        assert stats["images_upserted"] == 3
        assert stats["errors"] == 0

        # Verify root column is correct for each image.
        conn = open_db(db_path)
        rows = conn.execute(
            "SELECT root, filename_stem FROM images ORDER BY filename_stem"
        ).fetchall()
        assert len(rows) == 3

        root_map = {row["filename_stem"]: row["root"] for row in rows}
        assert root_map["cam1_2025_01_10_20_00_00"] == str(root1)
        assert root_map["cam1_2025_01_10_21_00_00"] == str(root1)
        assert root_map["cam1_2025_06_15_22_00_00"] == str(root2)
        conn.close()


def test_populate_two_roots_different_systems():
    """Two roots with different systems are both indexed."""
    with tempfile.TemporaryDirectory() as tmp:
        root1 = Path(tmp) / "media1"
        root1.mkdir()
        root2 = Path(tmp) / "media2"
        root2.mkdir()
        db_path = Path(tmp) / "test.db"

        _create_media_tree(
            root1,
            {"cam1": {"2025_01_10": ["20_00_00"]}},
        )
        _create_media_tree(
            root2,
            {"cam2": {"2025_01_10": ["20_00_00"]}},
        )

        stats = populate([root1, root2], db_path)
        assert stats["images_upserted"] == 2

        systems = get_systems(db_path)
        assert systems == ["cam1", "cam2"]

        # Query returns images from both roots.
        results = query_images(db_path)
        assert len(results) == 2
        roots_in_results = {r["root"] for r in results}
        assert str(root1) in roots_in_results
        assert str(root2) in roots_in_results


def test_populate_two_roots_query_includes_root(populated_db):
    """Query results include the root column."""
    results = query_images(populated_db)
    assert len(results) > 0
    assert "root" in results[0]
