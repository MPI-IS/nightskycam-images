"""
Tests for the location mapping (locations.py) and the DB backfill
(db.apply_locations), plus the no-data-loss upsert semantics.
"""

from pathlib import Path
import sqlite3
import tempfile
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
    apply_locations,
    count_images,
    open_db,
    populate,
    query_images,
)
from nightskycam_images.locations import (
    LocationConfigError,
    LocationMap,
    LocationRange,
    load_locations,
)


def _write_mapping(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


# ============================================================================
# Parsing and validation
# ============================================================================


def test_load_locations_full_featured(tmp_path):
    mapping = _write_mapping(
        tmp_path / "locations.toml",
        """
[[cam1]]
start = "2024_01_01"
end = "2024_02_01"
location = "Tuebingen"
note = "some note"

[[cam1]]
start = "2024_03_05"
end = "2024_03_05"
location = "palma"

[[cam1]]
start = "2024_06_01"
location = "ibach"
""",
    )
    location_map = load_locations(mapping)
    assert location_map.systems == ["cam1"]
    ranges = location_map.ranges["cam1"]
    assert ranges[0].location == "tuebingen"  # lowercased
    assert ranges[0].note == "some note"
    assert ranges[1].start == ranges[1].end == "2024_03_05"
    assert ranges[2].end is None  # ongoing


def test_lookup_boundaries(tmp_path):
    mapping = _write_mapping(
        tmp_path / "locations.toml",
        """
[[cam1]]
start = "2024_01_10"
end = "2024_01_20"
location = "tuebingen"

[[cam1]]
start = "2024_06_01"
location = "palma"
""",
    )
    location_map = load_locations(mapping)
    lookup = location_map.lookup
    assert lookup("cam1", "2024_01_09") is None  # day before
    assert lookup("cam1", "2024_01_10") == "tuebingen"  # start inclusive
    assert lookup("cam1", "2024_01_20") == "tuebingen"  # end inclusive
    assert lookup("cam1", "2024_01_21") is None  # day after
    assert lookup("cam1", "2024_05_31") is None  # gap
    assert lookup("cam1", "2099_12_31") == "palma"  # ongoing
    assert lookup("cam2", "2024_01_15") is None  # unknown system


@pytest.mark.parametrize(
    "body,match",
    [
        ('[[cam1]]\nstart = "junk"\nlocation = "x"', "YYYY_MM_DD"),
        ('[[cam1]]\nstart = "2024_01_01"\nend = "bad"\nlocation = "x"', "YYYY_MM_DD"),
        (
            '[[cam1]]\nstart = "2024_02_01"\nend = "2024_01_01"\nlocation = "x"',
            "after end",
        ),
        ('[[cam1]]\nlocation = "x"', "missing required key"),
        ('[[cam1]]\nstart = "2024_01_01"', "missing required key"),
        ('[[cam1]]\nstart = "2024_01_01"\nlocation = "  "', "empty"),
        (
            '[[cam1]]\nstart = "2024_01_01"\nlocation = "x"\nplace = "y"',
            "unknown key",
        ),
        ('[cam1]\nstart = "2024_01_01"\nlocation = "x"', "array of tables"),
    ],
)
def test_load_locations_invalid(tmp_path, body, match):
    mapping = _write_mapping(tmp_path / "locations.toml", body)
    with pytest.raises(LocationConfigError, match=match):
        load_locations(mapping)


def test_overlap_different_locations_rejected(tmp_path):
    mapping = _write_mapping(
        tmp_path / "locations.toml",
        """
[[cam1]]
start = "2024_01_01"
end = "2024_03_01"
location = "tuebingen"

[[cam1]]
start = "2024_02_01"
location = "palma"
""",
    )
    with pytest.raises(LocationConfigError, match="overlap"):
        load_locations(mapping)


def test_overlap_same_location_allowed(tmp_path):
    mapping = _write_mapping(
        tmp_path / "locations.toml",
        """
[[cam1]]
start = "2024_01_05"
end = "2024_01_10"
location = "tuebingen"

[[cam1]]
start = "2024_01_01"
end = "2024_02_01"
location = "tuebingen"
""",
    )
    location_map = load_locations(mapping)  # warning only, no raise
    assert location_map.lookup("cam1", "2024_01_07") == "tuebingen"


def test_missing_file(tmp_path):
    with pytest.raises(LocationConfigError, match="cannot read"):
        load_locations(tmp_path / "nope.toml")


# ============================================================================
# Fixture DB
# ============================================================================


def _create_media_tree(root: Path, structure: Dict[str, Dict[str, List[str]]]) -> None:
    for system, dates in structure.items():
        for date_str, times in dates.items():
            date_dir = root / system / date_str
            date_dir.mkdir(parents=True)
            thumb_dir = date_dir / THUMBNAIL_DIR_NAME
            thumb_dir.mkdir()
            for time_str in times:
                stem = f"{system}_{date_str}_{time_str}"
                img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
                cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)
                cv2.imwrite(
                    str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"),
                    img[:10, :10],
                )
                with open(date_dir / f"{stem}.toml", "wb") as f:
                    tomli_w.dump(
                        {
                            "process": "raw",
                            "weather": "clear",
                            "classifiers": {"cloudy": 0.5},
                        },
                        f,
                    )


STRUCTURE = {
    "cam1": {
        "2024_01_10": ["20_00_00", "21_00_00"],  # tuebingen range
        "2024_03_01": ["20_00_00"],  # gap (no location)
        "2024_06_15": ["20_00_00"],  # ongoing palma range
    },
    "cam2": {
        "2024_01_10": ["22_00_00"],  # no mapping for cam2
    },
}

MAPPING_TOML = """
[[cam1]]
start = "2024_01_01"
end = "2024_02_01"
location = "tuebingen"

[[cam1]]
start = "2024_06_01"
location = "palma"
"""


@pytest.fixture
def media_db(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    _create_media_tree(root, STRUCTURE)
    db_path = tmp_path / "test.db"
    populate(root, db_path, full=True)
    mapping = _write_mapping(tmp_path / "locations.toml", MAPPING_TOML)
    return root, db_path, mapping


def _locations_by_stem(db_path: Path) -> Dict[str, str]:
    conn = open_db(db_path)
    rows = conn.execute("SELECT filename_stem, location FROM images").fetchall()
    conn.close()
    return {row["filename_stem"]: row["location"] for row in rows}


# ============================================================================
# apply_locations (backfill)
# ============================================================================


def test_apply_locations(media_db):
    _, db_path, mapping = media_db
    result = apply_locations(db_path, load_locations(mapping))

    assert result["per_location"] == {"tuebingen": 2, "palma": 1}
    assert result["per_system"] == {"cam1": 3}
    assert result["remaining_null"] == 2  # cam1 gap + cam2
    assert result["total"] == 5
    assert result["dry_run"] is False

    by_stem = _locations_by_stem(db_path)
    assert by_stem["cam1_2024_01_10_20_00_00"] == "tuebingen"
    assert by_stem["cam1_2024_06_15_20_00_00"] == "palma"  # ongoing range
    assert by_stem["cam1_2024_03_01_20_00_00"] is None  # gap
    assert by_stem["cam2_2024_01_10_22_00_00"] is None  # unmapped system


def test_apply_locations_dry_run(media_db):
    _, db_path, mapping = media_db
    result = apply_locations(db_path, load_locations(mapping), dry_run=True)
    assert result["per_location"] == {"tuebingen": 2, "palma": 1}
    assert result["dry_run"] is True
    # Nothing was committed.
    assert set(_locations_by_stem(db_path).values()) == {None}


def test_apply_locations_idempotent(media_db):
    _, db_path, mapping = media_db
    location_map = load_locations(mapping)
    first = apply_locations(db_path, location_map)
    second = apply_locations(db_path, location_map)
    assert first["per_location"] == second["per_location"]
    assert first["remaining_null"] == second["remaining_null"]


def test_apply_locations_never_clears(media_db):
    """Rows outside every range keep a manually set location."""
    _, db_path, mapping = media_db
    conn = open_db(db_path)
    with conn:
        conn.execute(
            "UPDATE images SET location = 'somewhere' "
            "WHERE filename_stem = 'cam2_2024_01_10_22_00_00'"
        )
    conn.close()
    apply_locations(db_path, load_locations(mapping))
    assert _locations_by_stem(db_path)["cam2_2024_01_10_22_00_00"] == "somewhere"


# ============================================================================
# No-data-loss upsert semantics (populate x backfill orderings)
# ============================================================================


def test_rescan_without_lookup_preserves_locations(media_db):
    root, db_path, mapping = media_db
    apply_locations(db_path, load_locations(mapping))
    before = _locations_by_stem(db_path)

    populate(root, db_path, full=True)  # no location_lookup

    assert _locations_by_stem(db_path) == before


def test_populate_with_lookup_sets_locations(media_db):
    root, db_path, mapping = media_db
    location_map = load_locations(mapping)
    populate(root, db_path, full=True, location_lookup=location_map.lookup)

    by_stem = _locations_by_stem(db_path)
    assert by_stem["cam1_2024_01_10_20_00_00"] == "tuebingen"
    assert by_stem["cam1_2024_06_15_20_00_00"] == "palma"
    assert by_stem["cam1_2024_03_01_20_00_00"] is None


def test_populate_with_lookup_overwrites_stale_value(media_db):
    """A non-NULL looked-up value replaces a stale stored one."""
    root, db_path, mapping = media_db
    conn = open_db(db_path)
    with conn:
        conn.execute(
            "UPDATE images SET location = 'wrong' "
            "WHERE filename_stem = 'cam1_2024_01_10_20_00_00'"
        )
    conn.close()

    location_map = load_locations(mapping)
    populate(root, db_path, full=True, location_lookup=location_map.lookup)
    assert _locations_by_stem(db_path)["cam1_2024_01_10_20_00_00"] == "tuebingen"


def test_lookup_none_preserves_existing(media_db):
    """A lookup returning None for a date never clears a stored value."""
    root, db_path, mapping = media_db
    conn = open_db(db_path)
    with conn:
        conn.execute(
            "UPDATE images SET location = 'kept' "
            "WHERE filename_stem = 'cam1_2024_03_01_20_00_00'"  # gap date
        )
    conn.close()

    location_map = load_locations(mapping)
    populate(root, db_path, full=True, location_lookup=location_map.lookup)
    assert _locations_by_stem(db_path)["cam1_2024_03_01_20_00_00"] == "kept"


def test_rescan_keeps_row_ids_and_scores(media_db):
    """The in-place upsert must not churn ids or classifier scores."""
    root, db_path, _ = media_db
    conn = open_db(db_path)
    ids_before = {
        row["filename_stem"]: row["id"]
        for row in conn.execute("SELECT filename_stem, id FROM images")
    }
    scores_before = conn.execute(
        "SELECT COUNT(*) AS n FROM classifier_scores"
    ).fetchone()["n"]
    conn.close()
    assert scores_before > 0

    populate(root, db_path, full=True)

    conn = open_db(db_path)
    ids_after = {
        row["filename_stem"]: row["id"]
        for row in conn.execute("SELECT filename_stem, id FROM images")
    }
    scores_after = conn.execute(
        "SELECT COUNT(*) AS n FROM classifier_scores"
    ).fetchone()["n"]
    conn.close()
    assert ids_after == ids_before
    assert scores_after == scores_before


# ============================================================================
# Schema migration of a legacy database
# ============================================================================

_LEGACY_SCHEMA = """
CREATE TABLE images (
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
CREATE TABLE classifier_scores (
    image_id        INTEGER NOT NULL,
    classifier_name TEXT    NOT NULL,
    probability     REAL    NOT NULL,
    PRIMARY KEY (image_id, classifier_name),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);
"""


def _make_legacy_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO images (root, system, date, time, datetime, "
        "filename_stem) VALUES ('/r', 'cam1', '2024_01_10', '20_00_00', "
        "'2024-01-10T20:00:00', 'cam1_2024_01_10_20_00_00')"
    )
    conn.execute(
        "INSERT INTO classifier_scores (image_id, classifier_name, "
        "probability) VALUES (1, 'cloudy', 0.5)"
    )
    conn.commit()
    conn.close()


def test_open_db_migrates_legacy_schema(tmp_path):
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    conn = open_db(db_path)  # performs the migration
    columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    assert "location" in columns
    indexes = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert "idx_images_location" in indexes
    # Existing data intact, location NULL.
    row = conn.execute("SELECT * FROM images").fetchone()
    assert row["filename_stem"] == "cam1_2024_01_10_20_00_00"
    assert row["location"] is None
    score = conn.execute("SELECT COUNT(*) AS n FROM classifier_scores").fetchone()
    assert score["n"] == 1
    conn.close()

    # Idempotent.
    conn = open_db(db_path)
    conn.close()


# ============================================================================
# Location query filter
# ============================================================================


def test_query_and_count_by_location(media_db):
    _, db_path, mapping = media_db
    apply_locations(db_path, load_locations(mapping))

    assert count_images(db_path, locations=["tuebingen"]) == 2
    assert count_images(db_path, locations=["tuebingen", "palma"]) == 3
    assert count_images(db_path, locations=["nowhere"]) == 0

    rows = query_images(db_path, locations=["palma"])
    assert [r["filename_stem"] for r in rows] == ["cam1_2024_06_15_20_00_00"]
    assert rows[0]["location"] == "palma"


def test_location_map_direct_construction():
    location_map = LocationMap(
        ranges={"cam1": [LocationRange(start="2024_01_01", end=None, location="palma")]}
    )
    assert location_map.lookup("cam1", "2024_05_05") == "palma"
