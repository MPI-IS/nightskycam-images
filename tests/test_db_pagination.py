"""
Tests for the query-layer extensions used by the website:
count_images, LIMIT/OFFSET pagination, datetime ordering and the
batched classifier-score fetch.
"""

from pathlib import Path
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
    count_images,
    get_classifier_scores,
    get_classifier_scores_for_ids,
    open_db_readonly,
    populate,
    query_images,
)
from nightskycam_images.db_api import ImageDB


def _create_media_tree(
    root: Path,
    structure: Dict[str, Dict[str, List[str]]],
    metadata: Dict[str, Dict] = None,
) -> None:
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
                img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
                cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)
                thumb = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
                cv2.imwrite(str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), thumb)
                meta = metadata.get(
                    stem,
                    {
                        "process": "raw",
                        "weather": "clear",
                        "classifiers": {"cloudy": 0.5},
                    },
                )
                with open(date_dir / f"{stem}.toml", "wb") as f:
                    tomli_w.dump(meta, f)


@pytest.fixture
def populated_db():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"
        structure = {
            "cam1": {
                "2025_06_01": ["20_00_00", "21_00_00", "23_30_00"],
                "2025_06_02": ["02_00_00", "20_00_00"],
            },
            "cam2": {
                "2025_06_01": ["22_00_00", "22_30_00"],
                "2025_06_03": ["01_00_00"],
            },
        }
        _create_media_tree(root, structure)
        populate(root, db_path, full=True)
        yield db_path


FILTER_COMBOS = [
    {},
    {"systems": ["cam1"]},
    {"start_date": "2025_06_02"},
    {"start_time": "22_00_00", "end_time": "04_00_00"},  # crosses midnight
    {"weather": ["clear"]},
    {"classifier_max": {"cloudy": 0.6}},
]


@pytest.mark.parametrize("filters", FILTER_COMBOS)
def test_count_images_matches_query_images(populated_db, filters):
    assert count_images(populated_db, **filters) == len(
        query_images(populated_db, **filters)
    )


def test_default_call_unchanged(populated_db):
    """Default arguments keep the historical ordering and full result."""
    rows = query_images(populated_db)
    assert len(rows) == 8
    ordering = [(r["system"], r["date"], r["time"]) for r in rows]
    assert ordering == sorted(ordering)


def test_limit_offset_windows_tile(populated_db):
    full = query_images(populated_db, order_by="datetime")
    pages = []
    for offset in range(0, len(full), 3):
        pages.extend(
            query_images(populated_db, order_by="datetime", limit=3, offset=offset)
        )
    assert [r["id"] for r in pages] == [r["id"] for r in full]


def test_datetime_ordering(populated_db):
    ascending = query_images(populated_db, order_by="datetime")
    datetimes = [r["datetime"] for r in ascending]
    assert datetimes == sorted(datetimes)

    descending = query_images(populated_db, order_by="datetime", descending=True)
    assert [r["id"] for r in descending] == [r["id"] for r in reversed(ascending)]


def test_invalid_order_by_rejected(populated_db):
    with pytest.raises(ValueError, match="order_by"):
        query_images(populated_db, order_by="weather; DROP TABLE images")


def test_batched_scores_match_per_row(populated_db):
    rows = query_images(populated_db)
    batched = get_classifier_scores_for_ids(populated_db, [r["id"] for r in rows])
    for row in rows:
        assert batched.get(row["id"], {}) == get_classifier_scores(
            populated_db, row["filename_stem"]
        )


def test_batched_scores_chunking(populated_db):
    """IDs beyond the 500-per-chunk limit are still all fetched."""
    rows = query_images(populated_db)
    real_ids = [r["id"] for r in rows]
    # Pad with nonexistent ids to force several chunks.
    padded = real_ids + list(range(10_000, 10_000 + 1200))
    batched = get_classifier_scores_for_ids(populated_db, padded)
    assert set(batched.keys()) == set(real_ids)


def test_batched_scores_empty_input(populated_db):
    assert get_classifier_scores_for_ids(populated_db, []) == {}


# ============================================================================
# ImageDB pass-through
# ============================================================================


def test_imagedb_count_uses_filters(populated_db):
    db = ImageDB(populated_db)
    assert db.count() == 8
    assert db.count(systems=["cam1"]) == 5
    assert db.count("cam2", "2025_06_01") == 2


def test_imagedb_images_pagination(populated_db):
    db = ImageDB(populated_db)
    first = db.images(order_by="datetime", limit=3)
    second = db.images(order_by="datetime", limit=3, offset=3)
    assert len(first) == 3 and len(second) == 3
    assert not {r.filename_stem for r in first} & {r.filename_stem for r in second}


def test_imagedb_images_scores(populated_db):
    db = ImageDB(populated_db)
    with_scores = db.images(order_by="datetime", limit=2)
    assert all(r.classifier_scores == {"cloudy": 0.5} for r in with_scores)
    without = db.images(order_by="datetime", limit=2, with_scores=False)
    assert all(r.classifier_scores == {} for r in without)


def test_imagedb_images_descending(populated_db):
    db = ImageDB(populated_db)
    newest = db.images(order_by="datetime", descending=True, limit=1)[0]
    assert newest.datetime == max(r.datetime for r in db.images())
