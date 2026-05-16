"""
Tests for the classify-images command and Image.classifiers property.
"""

import datetime as dt
import tempfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest
import tomli
import tomli_w

from nightskycam_images.constants import (
    DATE_FORMAT_FILE,
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
    TIME_FORMAT_FILE,
)
from nightskycam_images.image import Image
from nightskycam_images.main import (
    _ClassifyImagesConfig,
    _classify_images_config_to_dict,
    _get_default_classify_images_config,
    _parse_classify_images_config,
)


# Path to the models directory at the repo root.
MODELS_DIR = Path(__file__).parent.parent / "models"


# ============================================================================
# Image.classifiers property tests
# ============================================================================


def test_classifiers_property_empty():
    """Image.classifiers returns empty dict when no classifiers section exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        stem = "cam_2025_01_01_20_00_00"

        # Create a TOML file without [classifiers] section.
        meta = {"process": "auto-stretching", "weather": "clear"}
        with open(tmp / f"{stem}.toml", "wb") as f:
            tomli_w.dump(meta, f)

        # Create a dummy HD image so Image.hd resolves.
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp / f"{stem}.jpg"), dummy_img)

        image = Image()
        image.filename_stem = stem
        image.dir_path = tmp

        assert image.classifiers == {}


def test_classifiers_property_with_scores():
    """Image.classifiers returns the scores dict when classifiers section exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        stem = "cam_2025_01_01_20_00_00"

        scores = {"quality": 0.87, "clouds": 0.23}
        meta = {"process": "auto-stretching", "classifiers": scores}
        with open(tmp / f"{stem}.toml", "wb") as f:
            tomli_w.dump(meta, f)

        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp / f"{stem}.jpg"), dummy_img)

        image = Image()
        image.filename_stem = stem
        image.dir_path = tmp

        result = image.classifiers
        assert result == pytest.approx(scores)


def test_classifiers_property_no_toml():
    """Image.classifiers returns empty dict when no TOML file exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        stem = "cam_2025_01_01_20_00_00"

        image = Image()
        image.filename_stem = stem
        image.dir_path = tmp

        assert image.classifiers == {}


# ============================================================================
# Config parsing tests
# ============================================================================


def test_parse_config_valid():
    """Valid config dict is parsed into _ClassifyImagesConfig correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy model files.
        model_a = Path(tmp_dir) / "model_a.pt"
        model_b = Path(tmp_dir) / "model_b.pt"
        model_a.touch()
        model_b.touch()

        config = {
            "root": tmp_dir,
            "models": {"quality": str(model_a), "clouds": str(model_b)},
            "systems": ["nightskycam5"],
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
        }

        parsed = _parse_classify_images_config(config)
        assert parsed.root == Path(tmp_dir)
        assert set(parsed.models.keys()) == {"quality", "clouds"}
        assert parsed.models["quality"] == model_a
        assert parsed.systems == ["nightskycam5"]
        assert parsed.start_date == dt.date(2025, 1, 1)
        assert parsed.end_date == dt.date(2025, 12, 31)


def test_parse_config_minimal():
    """Config with only required fields is parsed correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = Path(tmp_dir) / "model.pt"
        model.touch()

        config = {
            "root": tmp_dir,
            "models": {"quality": str(model)},
        }

        parsed = _parse_classify_images_config(config)
        assert parsed.systems is None
        assert parsed.start_date is None
        assert parsed.end_date is None


def test_parse_config_missing_root():
    """Config without 'root' raises ValueError."""
    config = {"models": {"quality": "/some/path.pt"}}
    with pytest.raises(ValueError, match="root"):
        _parse_classify_images_config(config)


def test_parse_config_missing_models():
    """Config without 'models' raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = {"root": tmp_dir}
        with pytest.raises(ValueError, match="models"):
            _parse_classify_images_config(config)


def test_parse_config_empty_models():
    """Config with empty 'models' table raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = {"root": tmp_dir, "models": {}}
        with pytest.raises(ValueError, match="non-empty"):
            _parse_classify_images_config(config)


def test_parse_config_nonexistent_model():
    """Config with a model path that doesn't exist raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = {
            "root": tmp_dir,
            "models": {"quality": "/nonexistent/model.pt"},
        }
        with pytest.raises(ValueError, match="does not exist"):
            _parse_classify_images_config(config)


def test_parse_config_invalid_date():
    """Config with invalid date format raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = Path(tmp_dir) / "model.pt"
        model.touch()

        config = {
            "root": tmp_dir,
            "models": {"quality": str(model)},
            "start_date": "01-01-2025",
        }
        with pytest.raises(ValueError, match="start_date"):
            _parse_classify_images_config(config)


# ============================================================================
# Config serialization round-trip test
# ============================================================================


def test_config_to_dict_round_trip():
    """Config can be serialized to dict and back."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_a = Path(tmp_dir) / "model_a.pt"
        model_b = Path(tmp_dir) / "model_b.pt"
        model_a.touch()
        model_b.touch()

        original = _ClassifyImagesConfig(
            root=Path(tmp_dir),
            models={"quality": model_a, "clouds": model_b},
            systems=["nightskycam5"],
            start_date=dt.date(2025, 1, 1),
            end_date=dt.date(2025, 12, 31),
        )

        as_dict = _classify_images_config_to_dict(original)

        # Should be serializable to TOML and back.
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb") as f:
            tomli_w.dump(as_dict, f)
            f.flush()
            with open(f.name, "rb") as rf:
                loaded = tomli.load(rf)

        parsed = _parse_classify_images_config(loaded)
        assert parsed.root == original.root
        assert set(parsed.models.keys()) == set(original.models.keys())
        assert parsed.systems == original.systems
        assert parsed.start_date == original.start_date
        assert parsed.end_date == original.end_date


def test_get_default_config():
    """Default config is a valid dict that can be serialized to TOML."""
    default = _get_default_classify_images_config()
    assert "root" in default
    assert "models" in default
    assert isinstance(default["models"], dict)
    assert len(default["models"]) > 0

    # Should be serializable to TOML without error.
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="wb") as f:
        tomli_w.dump(default, f)


# ============================================================================
# End-to-end classify_images test with real models
# ============================================================================


def _create_test_media(
    root: Path,
    system_name: str,
    date_str: str,
    time_strs: list,
) -> None:
    """
    Create a minimal nightskycam media directory structure with real images.

    Creates HD images, thumbnails, and TOML metadata files.
    """
    date_dir = root / system_name / date_str
    date_dir.mkdir(parents=True)
    thumbnail_dir = date_dir / THUMBNAIL_DIR_NAME
    thumbnail_dir.mkdir()

    for time_str in time_strs:
        stem = f"{system_name}_{date_str}_{time_str}"

        # Create a real JPEG HD image.
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)

        # Create a real JPEG thumbnail.
        thumb = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(
            str(thumbnail_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), thumb
        )

        # Create TOML metadata with some existing fields.
        meta = {"process": "auto-stretching", "weather": "clear"}
        with open(date_dir / f"{stem}.toml", "wb") as f:
            tomli_w.dump(meta, f)


@pytest.mark.skipif(
    not (MODELS_DIR / "cloudy.pt").exists()
    or not (MODELS_DIR / "rainy.pt").exists(),
    reason="Model files not found in models/ directory",
)
def test_classify_images_end_to_end():
    """
    End-to-end test: run classify_images logic on a temp media directory
    with real models, then verify TOML files are updated with [classifiers].
    """
    from nightskycam_scorer.model.infer import SkyScorer
    from nightskycam_scorer.utils import to_float_image

    from nightskycam_images.convert_npy import to_npy
    from nightskycam_images.walk import (
        _is_within_date_range,
        get_images,
        walk_dates,
        walk_systems,
    )

    system_name = "testcam"
    date_str = "2025_06_15"
    time_strs = ["20_00_00", "21_00_00"]
    model_names = {"cloudy": MODELS_DIR / "cloudy.pt", "rainy": MODELS_DIR / "rainy.pt"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _create_test_media(root, system_name, date_str, time_strs)

        # Load models.
        scorers: Dict[str, SkyScorer] = {}
        for name, model_path in model_names.items():
            scorers[name] = SkyScorer(str(model_path), validate_size=False)

        # Run classification (replicating the core loop from classify_images).
        for system_path in walk_systems(root):
            for date_, date_path in walk_dates(system_path):
                images = get_images(date_path)
                for image in images:
                    thumbnail_path = image.thumbnail
                    meta_path = image.meta_path
                    assert thumbnail_path is not None
                    assert meta_path is not None

                    thumbnail_array = to_npy(thumbnail_path)
                    rgb_float = to_float_image(thumbnail_array)

                    classifier_scores: Dict[str, float] = {}
                    for name, scorer in scorers.items():
                        result_raw = scorer.predict(rgb_float)
                        result = (
                            result_raw[0]
                            if isinstance(result_raw, list)
                            else result_raw
                        )
                        classifier_scores[name] = float(result.probability)

                    # Read, update, write TOML.
                    with open(meta_path, "rb") as f:
                        meta_data = tomli.load(f)
                    meta_data["classifiers"] = classifier_scores
                    with open(meta_path, "wb") as f:
                        tomli_w.dump(meta_data, f)

        # Verify: read back TOML files and check [classifiers] section.
        for system_path in walk_systems(root):
            for date_, date_path in walk_dates(system_path):
                images = get_images(date_path)
                assert len(images) == len(time_strs)

                for image in images:
                    # Verify via Image.classifiers property.
                    classifiers = image.classifiers
                    assert "cloudy" in classifiers
                    assert "rainy" in classifiers
                    assert 0.0 <= classifiers["cloudy"] <= 1.0
                    assert 0.0 <= classifiers["rainy"] <= 1.0

                    # Verify original metadata is preserved.
                    meta = image.meta
                    assert meta["process"] == "auto-stretching"
                    assert meta["weather"] == "clear"
                    assert "classifiers" in meta


@pytest.mark.skipif(
    not (MODELS_DIR / "cloudy.pt").exists()
    or not (MODELS_DIR / "rainy.pt").exists(),
    reason="Model files not found in models/ directory",
)
def test_classify_images_idempotent():
    """
    Running classification twice overwrites [classifiers] without duplicating keys.
    """
    from nightskycam_scorer.model.infer import SkyScorer
    from nightskycam_scorer.utils import to_float_image

    from nightskycam_images.convert_npy import to_npy
    from nightskycam_images.walk import get_images, walk_dates, walk_systems

    system_name = "testcam"
    date_str = "2025_06_15"
    time_strs = ["22_00_00"]
    model_names = {"cloudy": MODELS_DIR / "cloudy.pt"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        _create_test_media(root, system_name, date_str, time_strs)

        scorer = SkyScorer(str(MODELS_DIR / "cloudy.pt"), validate_size=False)

        # Run classification twice.
        for _ in range(2):
            for system_path in walk_systems(root):
                for date_, date_path in walk_dates(system_path):
                    images = get_images(date_path)
                    for image in images:
                        thumbnail_array = to_npy(image.thumbnail)
                        rgb_float = to_float_image(thumbnail_array)
                        result_raw = scorer.predict(rgb_float)
                        result = (
                            result_raw[0]
                            if isinstance(result_raw, list)
                            else result_raw
                        )
                        scores = {"cloudy": float(result.probability)}

                        with open(image.meta_path, "rb") as f:
                            meta_data = tomli.load(f)
                        meta_data["classifiers"] = scores
                        with open(image.meta_path, "wb") as f:
                            tomli_w.dump(meta_data, f)

        # Verify: only one [classifiers] section with expected keys.
        for system_path in walk_systems(root):
            for date_, date_path in walk_dates(system_path):
                for image in get_images(date_path):
                    classifiers = image.classifiers
                    assert list(classifiers.keys()) == ["cloudy"]
                    assert 0.0 <= classifiers["cloudy"] <= 1.0


# ============================================================================
# second_root config tests
# ============================================================================


def test_parse_config_second_root_valid():
    """Config with a valid second_root is parsed."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        model = Path(tmp_a) / "model.pt"
        model.touch()
        config = {
            "root": tmp_a,
            "second_root": tmp_b,
            "models": {"quality": str(model)},
        }
        parsed = _parse_classify_images_config(config)
        assert parsed.root == Path(tmp_a)
        assert parsed.second_root == Path(tmp_b)


def test_parse_config_second_root_missing_dir():
    """Config pointing second_root at a non-existent directory raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = Path(tmp_dir) / "model.pt"
        model.touch()
        config = {
            "root": tmp_dir,
            "second_root": "/nonexistent/path/does/not/exist",
            "models": {"quality": str(model)},
        }
        with pytest.raises(ValueError, match="Second root"):
            _parse_classify_images_config(config)


def test_parse_config_second_root_defaults_to_none():
    """Config without second_root parses successfully with second_root=None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = Path(tmp_dir) / "model.pt"
        model.touch()
        config = {
            "root": tmp_dir,
            "models": {"quality": str(model)},
        }
        parsed = _parse_classify_images_config(config)
        assert parsed.second_root is None


def test_config_to_dict_round_trip_with_second_root():
    """second_root survives serialize/parse round trip."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        model = Path(tmp_a) / "model.pt"
        model.touch()
        original = _ClassifyImagesConfig(
            root=Path(tmp_a),
            second_root=Path(tmp_b),
            models={"quality": model},
        )
        as_dict = _classify_images_config_to_dict(original)
        assert as_dict["second_root"] == str(tmp_b)
        parsed = _parse_classify_images_config(as_dict)
        assert parsed.second_root == Path(tmp_b)


# ============================================================================
# classifier_runner.classify_image_inplace tests (fake scorers, no real models)
# ============================================================================


class _FakeResult:
    def __init__(self, probability: float) -> None:
        self.probability = probability


class _FakeScorer:
    """Minimal stand-in for SkyScorer used in unit tests."""

    def __init__(self, probability: float) -> None:
        self._p = probability
        self.calls = 0

    def predict(self, image):  # noqa: D401, ARG002
        self.calls += 1
        return _FakeResult(self._p)


def _make_one_image(tmp: Path, meta: Dict) -> tuple[Path, Path, Path]:
    """Create one image (HD + thumbnail + TOML) under ``tmp`` and return paths."""
    system = "testcam"
    date_str = "2025_06_15"
    time_str = "20_00_00"
    stem = f"{system}_{date_str}_{time_str}"

    date_dir = tmp / system / date_str
    date_dir.mkdir(parents=True)
    thumb_dir = date_dir / THUMBNAIL_DIR_NAME
    thumb_dir.mkdir()

    hd = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(date_dir / f"{stem}.jpg"), hd)
    thumb = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    thumbnail_path = thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"
    cv2.imwrite(str(thumbnail_path), thumb)
    meta_path = date_dir / f"{stem}.toml"
    with open(meta_path, "wb") as f:
        tomli_w.dump(meta, f)

    return thumbnail_path, meta_path, date_dir


def test_classify_image_inplace_fills_all_when_empty():
    """No [classifiers] section in TOML: all configured scorers run."""
    from nightskycam_images.classifier_runner import classify_image_inplace

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        thumb, meta_path, _ = _make_one_image(
            tmp, {"process": "raw", "weather": "clear"}
        )
        cloudy = _FakeScorer(0.7)
        rainy = _FakeScorer(0.2)
        scorers = {"cloudy": cloudy, "rainy": rainy}

        with open(meta_path, "rb") as f:
            meta = tomli.load(f)
        updated, modified = classify_image_inplace(
            thumb, meta_path, meta, scorers, overwrite=False
        )

        assert modified is True
        assert cloudy.calls == 1
        assert rainy.calls == 1
        assert updated["classifiers"] == pytest.approx(
            {"cloudy": 0.7, "rainy": 0.2}
        )
        # TOML on disk reflects the change.
        with open(meta_path, "rb") as f:
            on_disk = tomli.load(f)
        assert on_disk["classifiers"] == pytest.approx(
            {"cloudy": 0.7, "rainy": 0.2}
        )
        assert on_disk["process"] == "raw"
        assert on_disk["weather"] == "clear"


def test_classify_image_inplace_fills_only_missing_without_overwrite():
    """Existing configured keys are preserved; only missing ones run."""
    from nightskycam_images.classifier_runner import classify_image_inplace

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        thumb, meta_path, _ = _make_one_image(
            tmp,
            {"process": "raw", "classifiers": {"cloudy": 0.99}},
        )
        cloudy = _FakeScorer(0.1)
        rainy = _FakeScorer(0.3)
        scorers = {"cloudy": cloudy, "rainy": rainy}

        with open(meta_path, "rb") as f:
            meta = tomli.load(f)
        updated, modified = classify_image_inplace(
            thumb, meta_path, meta, scorers, overwrite=False
        )

        assert modified is True
        assert cloudy.calls == 0  # already present, skipped
        assert rainy.calls == 1
        assert updated["classifiers"]["cloudy"] == pytest.approx(0.99)
        assert updated["classifiers"]["rainy"] == pytest.approx(0.3)


def test_classify_image_inplace_no_op_when_all_present_no_overwrite():
    """All configured keys already present, no overwrite: nothing runs, no write."""
    from nightskycam_images.classifier_runner import classify_image_inplace

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        original_meta = {
            "process": "raw",
            "classifiers": {"cloudy": 0.5, "rainy": 0.6},
        }
        thumb, meta_path, _ = _make_one_image(tmp, original_meta)
        mtime_before = meta_path.stat().st_mtime_ns

        cloudy = _FakeScorer(0.1)
        rainy = _FakeScorer(0.1)
        scorers = {"cloudy": cloudy, "rainy": rainy}

        with open(meta_path, "rb") as f:
            meta = tomli.load(f)
        updated, modified = classify_image_inplace(
            thumb, meta_path, meta, scorers, overwrite=False
        )

        assert modified is False
        assert cloudy.calls == 0
        assert rainy.calls == 0
        assert updated["classifiers"] == pytest.approx(
            {"cloudy": 0.5, "rainy": 0.6}
        )
        # TOML was not rewritten.
        assert meta_path.stat().st_mtime_ns == mtime_before


def test_classify_image_inplace_overwrites_all_when_overwrite():
    """overwrite=True reruns every configured scorer."""
    from nightskycam_images.classifier_runner import classify_image_inplace

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        thumb, meta_path, _ = _make_one_image(
            tmp, {"classifiers": {"cloudy": 0.99, "rainy": 0.99}}
        )
        cloudy = _FakeScorer(0.1)
        rainy = _FakeScorer(0.2)
        scorers = {"cloudy": cloudy, "rainy": rainy}

        with open(meta_path, "rb") as f:
            meta = tomli.load(f)
        updated, modified = classify_image_inplace(
            thumb, meta_path, meta, scorers, overwrite=True
        )

        assert modified is True
        assert cloudy.calls == 1
        assert rainy.calls == 1
        assert updated["classifiers"] == pytest.approx(
            {"cloudy": 0.1, "rainy": 0.2}
        )


def test_classify_image_inplace_preserves_unconfigured_keys():
    """Classifier keys in TOML that aren't named in the config survive."""
    from nightskycam_images.classifier_runner import classify_image_inplace

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        thumb, meta_path, _ = _make_one_image(
            tmp,
            {"classifiers": {"legacy_model": 0.42}},
        )
        cloudy = _FakeScorer(0.6)
        scorers = {"cloudy": cloudy}

        with open(meta_path, "rb") as f:
            meta = tomli.load(f)
        updated, modified = classify_image_inplace(
            thumb, meta_path, meta, scorers, overwrite=True
        )

        assert modified is True
        assert updated["classifiers"]["legacy_model"] == pytest.approx(0.42)
        assert updated["classifiers"]["cloudy"] == pytest.approx(0.6)


def test_make_populate_enricher_tracks_stats():
    """The enricher closure updates the stats dict as it runs."""
    from nightskycam_images.classifier_runner import make_populate_enricher

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        thumb, meta_path, _ = _make_one_image(
            tmp, {"process": "raw"}
        )
        cloudy = _FakeScorer(0.8)
        scorers = {"cloudy": cloudy}

        enricher, stats = make_populate_enricher(scorers, overwrite=False)
        assert stats == {
            "classifier_runs": 0,
            "classifier_writes": 0,
            "classifier_errors": 0,
        }

        with open(meta_path, "rb") as f:
            meta = tomli.load(f)
        enriched = enricher(thumb, meta_path, meta)
        assert enriched["classifiers"]["cloudy"] == pytest.approx(0.8)
        assert stats["classifier_runs"] == 1
        assert stats["classifier_writes"] == 1
        assert stats["classifier_errors"] == 0

        # Re-running on the same image is a no-op without overwrite.
        with open(meta_path, "rb") as f:
            meta = tomli.load(f)
        enricher(thumb, meta_path, meta)
        assert stats["classifier_runs"] == 1  # unchanged
        assert stats["classifier_writes"] == 1
