"""
Tests for the DB-backed Flask viewer (``db_view_webapp.create_app``).

Routes are exercised through Flask's test client against a populated
SQLite database; on-disk image files are real (jpg/tiff/npy) so the
JPEG-conversion paths are covered.
"""

from io import BytesIO
from pathlib import Path
import tempfile
from typing import Dict, List

import cv2
import numpy as np
import pytest
import tifffile
import tomli_w

from nightskycam_images.constants import (
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
)
from nightskycam_images.db import populate
from nightskycam_images.db_view_webapp import create_app


def _write_image(path: Path, fmt: str) -> None:
    """Write a small image in the requested format."""
    arr = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
    if fmt in ("jpg", "jpeg"):
        cv2.imwrite(str(path), arr)
    elif fmt == "tiff":
        tifffile.imwrite(str(path), arr)
    elif fmt == "npy":
        np.save(path, arr)
    else:
        raise ValueError(fmt)


def _create_media_tree(
    root: Path,
    structure: Dict[str, Dict[str, List[str]]],
    metadata: Dict[str, Dict] = None,
    image_format: str = "jpg",
) -> None:
    """Create a media hierarchy with images, thumbnails, and TOML files."""
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
                # Honor a per-stem format override in metadata.
                fmt = metadata.get(stem, {}).pop("_format", image_format)
                _write_image(date_dir / f"{stem}.{fmt}", fmt)
                # Thumbnail is always JPEG.
                thumb = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
                cv2.imwrite(
                    str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), thumb
                )
                meta = metadata.get(
                    stem, {"process": "raw", "weather": "clear"}
                )
                with open(date_dir / f"{stem}.toml", "wb") as f:
                    tomli_w.dump(meta, f)


@pytest.fixture
def app_client():
    """Build a test client over a populated DB with diverse data."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        root.mkdir()
        db_path = Path(tmp) / "test.db"

        # cam1: jpg + tiff + npy on 2025_06_01; cam2: jpg only.
        # _format is consumed by _create_media_tree (popped before TOML write).
        metadata = {
            "cam1_2025_06_01_20_00_00": {
                "_format": "jpg",
                "process": "auto-stretching 8bits",
                "weather": "clear",
                "cloud_cover": 10,
                "classifiers": {"cloudy": 0.1, "rainy": 0.05},
            },
            "cam1_2025_06_01_21_00_00": {
                "_format": "tiff",
                "process": "raw",
                "weather": "cloudy",
                "cloud_cover": 80,
                "classifiers": {"cloudy": 0.9, "rainy": 0.3},
            },
            "cam1_2025_06_01_22_00_00": {
                "_format": "npy",
                "process": "raw",
                "weather": "clear",
                "cloud_cover": 20,
            },
            "cam2_2025_06_02_22_00_00": {
                "_format": "jpg",
                "process": "raw",
                "weather": "rain",
                "cloud_cover": 95,
            },
        }
        structure = {
            "cam1": {
                "2025_06_01": ["20_00_00", "21_00_00", "22_00_00"],
            },
            "cam2": {
                "2025_06_02": ["22_00_00"],
            },
        }
        _create_media_tree(root, structure, metadata)
        populate(root, db_path, full=True)

        app = create_app(db_path)
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client, root, db_path


# ============================================================================
# Index + simple list endpoints
# ============================================================================


def test_index(app_client):
    client, _, _ = app_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower() or b"<!doctype" in resp.data.lower()


def test_systems_route(app_client):
    client, _, _ = app_client
    resp = client.get("/api/systems")
    assert resp.status_code == 200
    assert resp.get_json() == ["cam1", "cam2"]


def test_dates_route(app_client):
    client, _, _ = app_client
    resp = client.get("/api/dates/cam1")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload == [{"value": "2025_06_01", "display": "2025-06-01"}]


def test_dates_route_unknown_system(app_client):
    client, _, _ = app_client
    resp = client.get("/api/dates/nonexistent")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_classifiers_route(app_client):
    client, _, _ = app_client
    resp = client.get("/api/classifiers")
    assert resp.status_code == 200
    assert resp.get_json() == ["cloudy", "rainy"]


# ============================================================================
# /api/images/<system>/<date> + filter parameters
# ============================================================================


def test_images_route_basic(app_client):
    client, _, _ = app_client
    resp = client.get("/api/images/cam1/2025_06_01")
    assert resp.status_code == 200
    items = resp.get_json()
    assert len(items) == 3
    stems = {item["filename_stem"] for item in items}
    assert stems == {
        "cam1_2025_06_01_20_00_00",
        "cam1_2025_06_01_21_00_00",
        "cam1_2025_06_01_22_00_00",
    }
    # Each item carries the URLs the frontend uses.
    for item in items:
        assert item["thumbnail_url"].startswith("/api/thumbnail/")
        assert item["jpeg_url"].startswith("/api/image/jpeg/")
    # raw_url only present for non-jpg formats.
    by_stem = {item["filename_stem"]: item for item in items}
    assert "raw_url" not in by_stem["cam1_2025_06_01_20_00_00"]  # jpg
    assert "raw_url" in by_stem["cam1_2025_06_01_21_00_00"]  # tiff
    assert "raw_url" in by_stem["cam1_2025_06_01_22_00_00"]  # npy


def test_images_route_filter_stretched_yes(app_client):
    client, _, _ = app_client
    resp = client.get("/api/images/cam1/2025_06_01?stretched=yes")
    items = resp.get_json()
    assert len(items) == 1
    assert items[0]["stretched"] is True


def test_images_route_filter_stretched_no(app_client):
    client, _, _ = app_client
    resp = client.get("/api/images/cam1/2025_06_01?stretched=no")
    items = resp.get_json()
    assert len(items) == 2
    assert all(item["stretched"] is False for item in items)


def test_images_route_filter_format(app_client):
    client, _, _ = app_client
    resp = client.get("/api/images/cam1/2025_06_01?format=tiff")
    items = resp.get_json()
    assert len(items) == 1
    assert items[0]["image_format"] == "tiff"

    # format=all is treated as no filter.
    resp_all = client.get("/api/images/cam1/2025_06_01?format=all")
    assert len(resp_all.get_json()) == 3


def test_images_route_filter_cloud_cover_max(app_client):
    client, _, _ = app_client
    resp = client.get("/api/images/cam1/2025_06_01?cloud_cover_max=30")
    items = resp.get_json()
    assert {item["filename_stem"] for item in items} == {
        "cam1_2025_06_01_20_00_00",
        "cam1_2025_06_01_22_00_00",
    }


def test_images_route_filter_cloud_cover_max_invalid_ignored(app_client):
    client, _, _ = app_client
    # Invalid int → ValueError is swallowed, no filter applied.
    resp = client.get("/api/images/cam1/2025_06_01?cloud_cover_max=abc")
    assert resp.status_code == 200
    assert len(resp.get_json()) == 3


def test_images_route_filter_classifier_threshold(app_client):
    client, _, _ = app_client
    # Only the 20:00 image has cloudy=0.1, the 21:00 image has cloudy=0.9.
    resp = client.get("/api/images/cam1/2025_06_01?clf_cloudy=0.5")
    items = resp.get_json()
    assert len(items) == 1
    assert items[0]["filename_stem"] == "cam1_2025_06_01_20_00_00"
    assert items[0]["classifiers"]["cloudy"] == pytest.approx(0.1)


def test_images_route_filter_classifier_invalid_ignored(app_client):
    client, _, _ = app_client
    # Empty string and bad float are both skipped.
    resp = client.get(
        "/api/images/cam1/2025_06_01?clf_cloudy=&clf_rainy=notafloat"
    )
    assert resp.status_code == 200
    # No classifier filter applied → all three returned.
    assert len(resp.get_json()) == 3


# ============================================================================
# Thumbnail serving
# ============================================================================


def test_thumbnail_route_success(app_client):
    client, _, _ = app_client
    resp = client.get("/api/thumbnail/cam1_2025_06_01_20_00_00")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.cache_control.max_age == 3600
    assert len(resp.data) > 0


def test_thumbnail_route_unknown_stem(app_client):
    client, _, _ = app_client
    resp = client.get("/api/thumbnail/nonexistent_stem")
    assert resp.status_code == 404


def test_thumbnail_route_file_missing_on_disk(app_client):
    client, root, _ = app_client
    # Delete the thumbnail file but keep the DB row.
    thumb = (
        root
        / "cam1"
        / "2025_06_01"
        / THUMBNAIL_DIR_NAME
        / f"cam1_2025_06_01_20_00_00.{THUMBNAIL_FILE_FORMAT}"
    )
    thumb.unlink()
    resp = client.get("/api/thumbnail/cam1_2025_06_01_20_00_00")
    assert resp.status_code == 404


# ============================================================================
# JPEG serving (incl. on-the-fly conversion)
# ============================================================================


def test_jpeg_route_passthrough_for_jpg(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/jpeg/cam1_2025_06_01_20_00_00")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data[:3] == b"\xff\xd8\xff"  # JPEG magic


def test_jpeg_route_converts_tiff(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/jpeg/cam1_2025_06_01_21_00_00")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data[:3] == b"\xff\xd8\xff"


def test_jpeg_route_converts_grayscale_tiff_to_rgb():
    """Non-RGB TIFF (e.g. grayscale) triggers the .convert('RGB') branch."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        date_dir = root / "cam1" / "2025_06_15"
        thumb_dir = date_dir / THUMBNAIL_DIR_NAME
        thumb_dir.mkdir(parents=True)

        stem = "cam1_2025_06_15_20_00_00"
        # 2D array → grayscale TIFF → PIL mode "L" (not RGB).
        gray = np.random.randint(0, 255, (40, 40), dtype=np.uint8)
        tifffile.imwrite(str(date_dir / f"{stem}.tiff"), gray)
        cv2.imwrite(
            str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"),
            np.stack([gray] * 3, axis=-1),
        )
        with open(date_dir / f"{stem}.toml", "wb") as f:
            tomli_w.dump({"process": "raw", "weather": "clear"}, f)

        db_path = Path(tmp) / "test.db"
        populate(root, db_path, full=True)

        app = create_app(db_path)
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get(f"/api/image/jpeg/{stem}")
            assert resp.status_code == 200
            assert resp.mimetype == "image/jpeg"
            assert resp.data[:3] == b"\xff\xd8\xff"


def test_jpeg_route_converts_npy(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/jpeg/cam1_2025_06_01_22_00_00")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data[:3] == b"\xff\xd8\xff"


def test_jpeg_route_unknown_stem(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/jpeg/nonexistent_stem")
    assert resp.status_code == 404


def test_jpeg_route_file_missing_on_disk(app_client):
    client, root, _ = app_client
    img_path = root / "cam1" / "2025_06_01" / "cam1_2025_06_01_20_00_00.jpg"
    img_path.unlink()
    resp = client.get("/api/image/jpeg/cam1_2025_06_01_20_00_00")
    assert resp.status_code == 404


# ============================================================================
# Raw image serving
# ============================================================================


def test_raw_route_jpg(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/raw/cam1_2025_06_01_20_00_00")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert "attachment" in resp.headers.get("Content-Disposition", "")


def test_raw_route_tiff(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/raw/cam1_2025_06_01_21_00_00")
    assert resp.status_code == 200
    assert resp.mimetype == "image/tiff"


def test_raw_route_npy(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/raw/cam1_2025_06_01_22_00_00")
    assert resp.status_code == 200
    assert resp.mimetype == "application/octet-stream"


def test_raw_route_unknown_stem(app_client):
    client, _, _ = app_client
    resp = client.get("/api/image/raw/nonexistent_stem")
    assert resp.status_code == 404


def test_raw_route_file_missing_on_disk(app_client):
    client, root, _ = app_client
    img_path = root / "cam1" / "2025_06_01" / "cam1_2025_06_01_21_00_00.tiff"
    img_path.unlink()
    resp = client.get("/api/image/raw/cam1_2025_06_01_21_00_00")
    assert resp.status_code == 404
