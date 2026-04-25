"""
Tests for the db_api module (high-level DB query API).
"""

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
from nightskycam_images.db import populate
from nightskycam_images.db_api import ImageDB, ImageRecord


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
                cv2.imwrite(
                    str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), thumb
                )
                meta = metadata.get(
                    stem, {"process": "raw", "weather": "clear"}
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
                "2025_06_01": ["20_00_00", "21_00_00"],
                "2025_06_02": ["20_00_00"],
            },
            "cam2": {
                "2025_06_01": ["22_00_00"],
            },
        }

        stem_c1_d1_t1 = "cam1_2025_06_01_20_00_00"
        stem_c1_d1_t2 = "cam1_2025_06_01_21_00_00"
        stem_c1_d2_t1 = "cam1_2025_06_02_20_00_00"
        stem_c2_d1_t1 = "cam2_2025_06_01_22_00_00"

        metadata = {
            stem_c1_d1_t1: {
                "process": "auto-stretching 8bits",
                "weather": "clear",
                "cloud_cover": 10,
                "classifiers": {"cloudy": 0.1, "rainy": 0.05},
            },
            stem_c1_d1_t2: {
                "process": "raw",
                "weather": "cloudy",
                "cloud_cover": 80,
                "classifiers": {"cloudy": 0.9, "rainy": 0.3},
            },
            stem_c1_d2_t1: {
                "process": "auto-stretching",
                "weather": "clear",
                "cloud_cover": 5,
            },
            stem_c2_d1_t1: {
                "process": "raw",
                "weather": "rain",
                "cloud_cover": 95,
                "classifiers": {"cloudy": 0.8, "rainy": 0.95},
            },
        }

        _create_media_tree(root, structure, metadata)
        populate(root, db_path, full=True)
        yield db_path, root


def test_systems(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    assert db.systems() == ["cam1", "cam2"]


def test_dates(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    assert db.dates("cam1") == ["2025_06_01", "2025_06_02"]
    assert db.dates("cam2") == ["2025_06_01"]


def test_classifier_names(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    assert db.classifier_names() == ["cloudy", "rainy"]


def test_stats(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    s = db.stats()
    assert s["total_images"] == 4
    assert "cam1" in s["systems"]
    assert s["systems"]["cam1"]["image_count"] == 3


def test_images_all(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    all_imgs = db.images()
    assert len(all_imgs) == 4
    assert all(isinstance(img, ImageRecord) for img in all_imgs)


def test_images_by_system(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    imgs = db.images(system="cam1")
    assert len(imgs) == 3
    assert all(img.system == "cam1" for img in imgs)


def test_images_by_system_and_date(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    imgs = db.images(system="cam1", date="2025_06_01")
    assert len(imgs) == 2


def test_images_filter_stretched(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    stretched = db.images(stretched=True)
    assert len(stretched) == 2
    assert all(img.stretched for img in stretched)

    not_stretched = db.images(stretched=False)
    assert len(not_stretched) == 2
    assert all(not img.stretched for img in not_stretched)


def test_images_filter_cloud_cover(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    low_cloud = db.images(cloud_cover_max=30)
    assert len(low_cloud) == 2
    assert all(img.cloud_cover <= 30 for img in low_cloud)


def test_images_filter_weather(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    rainy = db.images(weather=["rain"])
    assert len(rainy) == 1
    assert rainy[0].system == "cam2"


def test_images_filter_classifier_max(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    low_cloudy = db.images(classifier_max={"cloudy": 0.5})
    assert len(low_cloudy) == 1
    assert low_cloudy[0].classifier_scores["cloudy"] <= 0.5


def test_images_filter_format(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    jpgs = db.images(image_format="jpg")
    assert len(jpgs) == 4


def test_image_record_paths(populated_db):
    db_path, root = populated_db
    db = ImageDB(db_path)
    imgs = db.images(system="cam1", date="2025_06_01")
    img = imgs[0]
    assert img.hd_path is not None
    assert img.hd_path.exists()
    assert img.thumbnail_path is not None
    assert img.thumbnail_path.exists()
    assert img.toml_path is not None
    assert img.toml_path.exists()


def test_image_record_classifier_scores(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    imgs = db.images(system="cam1", date="2025_06_01")
    scored = [img for img in imgs if img.classifier_scores]
    assert len(scored) == 2
    assert "cloudy" in scored[0].classifier_scores
    assert "rainy" in scored[0].classifier_scores


def test_image_single_lookup(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    img = db.image("cam1_2025_06_01_20_00_00")
    assert img is not None
    assert img.system == "cam1"
    assert img.weather == "clear"
    assert img.classifier_scores["cloudy"] == pytest.approx(0.1)


def test_image_single_not_found(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    assert db.image("nonexistent_stem") is None


def test_count(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    assert db.count() == 4
    assert db.count(system="cam1") == 3
    assert db.count(system="cam2") == 1
    assert db.count(stretched=True) == 2


def test_hd_paths(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    paths = db.hd_paths(system="cam1")
    assert len(paths) == 3
    assert all(p.exists() for p in paths)


def test_thumbnail_paths(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    paths = db.thumbnail_paths(system="cam1", date="2025_06_01")
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_system_and_systems_exclusive(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    with pytest.raises(ValueError):
        db.images(system="cam1", systems=["cam1", "cam2"])


def test_date_and_start_date_exclusive(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    with pytest.raises(ValueError):
        db.images(date="2025_06_01", start_date="2025_06_01")


def test_path_property(populated_db):
    db_path, _ = populated_db
    db = ImageDB(db_path)
    assert db.path == db_path
