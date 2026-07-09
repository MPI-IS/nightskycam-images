"""
End-to-end tests of the website Flask app over a small fixture DB with
synthetic 16-bit TIFF, uint16 NPY and 8-bit JPEG originals.
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest
import tomli_w

from nightskycam_images.constants import (
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
)
from nightskycam_images.db import open_db, populate
from nightskycam_images.website import create_app

SYSTEM = "cam1"
DATE = "2025_06_01"

STEM_TIFF = f"{SYSTEM}_{DATE}_20_00_00"
STEM_NPY = f"{SYSTEM}_{DATE}_21_00_00"
STEM_JPG = f"{SYSTEM}_{DATE}_22_00_00"
STEM_NO_THUMB = f"{SYSTEM}_{DATE}_23_00_00"

WIDTH, HEIGHT = 50, 40


def _gradient16() -> np.ndarray:
    """A 16-bit gradient image (values spread over the uint16 range)."""
    x = np.linspace(0, 60_000, WIDTH, dtype=np.uint16)
    return np.tile(x, (HEIGHT, 1))


def _write_thumbnail(thumb_dir: Path, stem: str) -> None:
    thumb = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), thumb)


@pytest.fixture(scope="module")
def website():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        date_dir = root / SYSTEM / DATE
        date_dir.mkdir(parents=True)
        thumb_dir = date_dir / THUMBNAIL_DIR_NAME
        thumb_dir.mkdir()

        cv2.imwrite(str(date_dir / f"{STEM_TIFF}.tiff"), _gradient16())
        np.save(date_dir / f"{STEM_NPY}.npy", _gradient16())
        jpg = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.imwrite(str(date_dir / f"{STEM_JPG}.jpg"), jpg)
        cv2.imwrite(str(date_dir / f"{STEM_NO_THUMB}.jpg"), jpg)

        metadata = {
            STEM_TIFF: {
                "process": "darkframe substraction",
                "weather": "clear",
                "cloud_cover": 5,
                "classifiers": {"cloudy": 0.1},
                "controllables": {"Exposure": 30000, "Gain": 300},
            },
            STEM_NPY: {
                "process": "raw",
                "weather": "overcast",
                "cloud_cover": 95,
                "classifiers": {"cloudy": 0.9},
            },
            STEM_JPG: {
                "process": "stretching (auto_stretch)",
                "weather": "partly clear",
                "cloud_cover": 30,
            },
        }
        for stem in (STEM_TIFF, STEM_NPY, STEM_JPG):
            _write_thumbnail(thumb_dir, stem)
            with open(date_dir / f"{stem}.toml", "wb") as f:
                tomli_w.dump(metadata[stem], f)
        # STEM_NO_THUMB: no thumbnail, no toml.

        db_path = Path(tmp) / "test.db"
        populate(root, db_path, full=True)

        # Give the TIFF image a deployment location (the others keep NULL).
        conn = open_db(db_path)
        with conn:
            conn.execute(
                "UPDATE images SET location = 'palma' WHERE filename_stem = ?",
                (STEM_TIFF,),
            )
        conn.close()

        cache_dir = Path(tmp) / "cache"
        app = create_app(db_path, cache_dir=cache_dir, cache_max_mb=64)
        app.config["TESTING"] = True
        yield app.test_client(), cache_dir


# ============================================================================
# Pages
# ============================================================================


def test_home(website):
    client, _ = website
    response = client.get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "4" in page  # total images
    assert SYSTEM in page


def test_browse_all(website):
    client, _ = website
    response = client.get("/browse")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    for stem in (STEM_TIFF, STEM_NPY, STEM_JPG, STEM_NO_THUMB):
        assert stem in page


def test_browse_filtered(website):
    client, _ = website
    response = client.get("/browse?image_format=tiff")
    page = response.get_data(as_text=True)
    assert STEM_TIFF in page
    assert STEM_NPY not in page


def test_browse_classifier_filter(website):
    client, _ = website
    response = client.get("/browse?classifier_max.cloudy=0.5")
    page = response.get_data(as_text=True)
    assert STEM_TIFF in page  # cloudy 0.1
    assert STEM_NPY not in page  # cloudy 0.9


def test_browse_location_filter_and_form(website):
    client, _ = website
    response = client.get("/browse?locations=palma")
    page = response.get_data(as_text=True)
    assert STEM_TIFF in page  # the one palma image
    assert STEM_NPY not in page
    # The location checkbox group renders (meta.locations non-empty).
    assert 'name="locations" value="palma"' in page


def test_browse_invalid_filter_is_400(website):
    client, _ = website
    response = client.get("/browse?start_date=junk")
    assert response.status_code == 400
    assert "start_date" in response.get_data(as_text=True)


def test_image_detail(website):
    client, _ = website
    response = client.get(f"/image/{STEM_TIFF}")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "clear" in page
    assert "Exposure" in page  # TOML extras
    assert f"/download/{STEM_TIFF}" in page
    assert "palma" in page  # location metadata row


def test_image_detail_nav(website):
    client, _ = website
    response = client.get(f"/image/{STEM_NPY}?order_by=datetime&descending=0&i=1")
    page = response.get_data(as_text=True)
    assert f"/image/{STEM_TIFF}" in page  # prev
    assert f"/image/{STEM_JPG}" in page  # next


def test_image_detail_unknown_404(website):
    client, _ = website
    assert client.get("/image/nope").status_code == 404


# ============================================================================
# Media
# ============================================================================


def test_thumbnail(website):
    client, _ = website
    response = client.get(f"/thumb/{STEM_TIFF}.jpeg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.cache_control.max_age == 86400


def test_thumbnail_missing_redirects_to_placeholder(website):
    client, _ = website
    response = client.get(f"/thumb/{STEM_NO_THUMB}.jpeg")
    assert response.status_code == 302
    assert "placeholder.svg" in response.location


def _decode(response) -> np.ndarray:
    data = np.frombuffer(response.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    assert image is not None
    return image


@pytest.mark.parametrize("stem", [STEM_TIFF, STEM_NPY])
def test_jpeg_conversion(website, stem):
    client, cache_dir = website
    plain = client.get(f"/jpeg/{stem}.jpeg?stretch=0")
    assert plain.status_code == 200
    assert plain.mimetype == "image/jpeg"
    image = _decode(plain)
    assert image.shape[:2] == (HEIGHT, WIDTH)

    stretched = client.get(f"/jpeg/{stem}.jpeg?stretch=1")
    assert stretched.status_code == 200
    assert stretched.data != plain.data

    assert (cache_dir / f"{stem}.s0.jpeg").is_file()
    assert (cache_dir / f"{stem}.s1.jpeg").is_file()

    # Second request is served from the cache (bytes identical).
    again = client.get(f"/jpeg/{stem}.jpeg?stretch=0")
    assert again.data == plain.data


def test_jpeg_original_streams_unmodified(website):
    client, cache_dir = website
    response = client.get(f"/jpeg/{STEM_JPG}.jpeg?stretch=0")
    assert response.status_code == 200
    # No conversion cached: served straight from disk.
    assert not (cache_dir / f"{STEM_JPG}.s0.jpeg").exists()


def test_download(website):
    client, _ = website
    response = client.get(f"/download/{STEM_TIFF}")
    assert response.status_code == 200
    assert response.mimetype == "image/tiff"
    disposition = response.headers["Content-Disposition"]
    assert "attachment" in disposition
    assert f"{STEM_TIFF}.tiff" in disposition

    npy = client.get(f"/download/{STEM_NPY}")
    assert npy.mimetype == "application/octet-stream"


# ============================================================================
# API
# ============================================================================


def test_api_meta(website):
    client, _ = website
    data = client.get("/api/meta").get_json()
    assert data["systems"] == [SYSTEM]
    assert data["classifier_names"] == ["cloudy"]
    assert "clear" in data["weather_values"]
    assert data["locations"] == ["palma"]
    assert data["date_min"] == DATE


def test_api_query_location_filter(website):
    client, _ = website
    response = client.post("/api/query", json={"locations": ["Palma"]})
    data = response.get_json()
    assert data["total"] == 1
    (image,) = data["images"]
    assert image["filename_stem"] == STEM_TIFF
    assert image["location"] == "palma"


def test_api_stats(website):
    client, _ = website
    data = client.get("/api/stats").get_json()
    assert data["total_images"] == 4
    assert data["format_counts"]["jpg"] == 2


def test_api_filter_schema(website):
    client, _ = website
    schema = client.get("/api/filter-schema").get_json()
    assert schema["title"] == "FilterSpec"
    assert "systems" in schema["properties"]


def test_api_query(website):
    client, _ = website
    response = client.post("/api/query", json={"image_format": "tiff"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 1
    (image,) = data["images"]
    assert image["filename_stem"] == STEM_TIFF
    assert image["classifier_scores"] == {"cloudy": 0.1}
    assert image["urls"]["thumb"] == f"/thumb/{STEM_TIFF}.jpeg"


def test_api_query_orders_and_paginates(website):
    client, _ = website
    response = client.post("/api/query", json={"descending": False, "page_size": 12})
    data = response.get_json()
    assert data["total"] == 4
    assert data["images"][0]["filename_stem"] == STEM_TIFF


def test_api_query_invalid_400(website):
    client, _ = website
    response = client.post("/api/query", json={"image_format": "png"})
    assert response.status_code == 400
    assert response.get_json()["field"] == "image_format"

    response = client.post("/api/query", data="not json", content_type="text/plain")
    assert response.status_code == 400
