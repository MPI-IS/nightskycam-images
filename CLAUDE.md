# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nightskycam-images` is a Python package for managing images captured by nightskycam camera-RaspberryPi systems. It provides CLI tools and Flask webapps for filtering, annotating, classifying, and browsing nightsky images stored in a structured filesystem (`root/system/date/` hierarchy).

## Build & Development

Uses **Poetry** for dependency management (Python >=3.12, <3.14).

```bash
poetry install --with dev    # install with dev dependencies
poetry shell                 # activate virtual environment
```

## Common Commands

```bash
# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest tests/test_walk.py

# Run a single test
poetry run pytest tests/test_walk.py::test_walk_systems

# Formatting and import sorting
poetry run black nightskycam_images tests
poetry run isort nightskycam_images tests

# Type checking
poetry run mypy nightskycam_images

# Coverage
coverage run -m pytest && coverage report
```

## Code Style

- Formatter: **black**
- Import sorting: **isort** (profile=black, `force_sort_within_sections = true`)
- Type checking: **mypy** (ignores missing imports for pytest, tomli_w, nightskycam_scorer, auto_stretch)
- Pre-commit hook available for auto-formatting on commit (`pre-commit install`)

## Architecture

**Filesystem hierarchy** is central to the design. Images are stored as `root/system_name/YYYY_MM_DD/` with a `thumbnails/` subdirectory per date. TOML metadata files sit alongside each image. Many operations work by creating symlinks to filtered subsets rather than copying files.

**Two query paths over the same data:**
- The filesystem itself is the source of truth — `walk.py` traversal + per-image TOML reads.
- `db.py` maintains a **SQLite mirror** of the metadata (`images`, `classifier_scores` tables) for fast querying without walking. The DB is rebuilt/refreshed by scanning the filesystem; treat it as a cache, not authoritative state. `db_api.py` exposes query helpers; `db_view_webapp.py` is a Flask UI on top of it.

Key modules:
- `walk.py` — filesystem traversal: iterates systems, dates, images; core filtering logic (`filter_and_export_images`); symlink creation for filtered exports
- `main.py` — Typer CLI entry points, wires together other modules
- `db.py` / `db_api.py` — SQLite metadata cache and query API
- `filters.py` — predicate-based filtering (process, cloud cover, weather fields from TOML metadata)
- `image.py` — `Image` class holding paths and metadata for a single capture
- `thumbnail.py` — thumbnail generation with parallel processing
- `video.py` — video creation from image sequences
- `weather.py` — `WeatherReport` parsing from TOML weather summaries
- `stats.py` — collection-wide statistics (rich-formatted tables)
- `folder_change.py` — mtime-based change detection for incremental scans
- `constants.py` — formats, dimensions, directory names (e.g., `IMAGE_FILE_FORMATS`, `THUMBNAIL_DIR_NAME`)
- `convert_npy.py` — NumPy array to image conversion with auto-stretching
- `patches.py` — image patching utilities
- `annotator_webapp.py`, `view_webapp.py`, `symlink_annotator_webapp.py`, `db_view_webapp.py` — Flask web interfaces (templates in `templates/<webapp>/`)

Subpackages (each with its own CLI module, not flat files):
- `nightskycam_images.classifier/` — training/inference for the cloud/rain classifiers; CLI at `classifier.main:app` (wired to `ns.ml.train`)
- `nightskycam_images.annotator/` — annotation tooling; CLI at `annotator.cli:app` (wired to `ns.annotate`)

Supporting top-level directories:
- `models/` — trained classifier weights (e.g. `cloudy.pt`, `rainy.pt`) consumed by `ns.ml.classify` / `ns.ml.scorer`
- `scripts/` — operational shell scripts (`reorganize_roots.sh`, `nightskycam_backup.sh`, `copy_from_allsky.sh`, etc.) that complement the Python CLI but are not part of the package

**Image formats supported:** npy, tiff, jpg, jpeg. Thumbnails are always JPEG at 200px width.

## CLI Entry Points

Defined in `pyproject.toml` under `[tool.poetry.scripts]`. Commands follow a `ns.<namespace>.<action>` convention — when adding new commands, match the existing namespace:

- `ns.thumb.*` — thumbnail listing/checking/creation/copying
- `ns.files.*` — backup, deletion, move-clear, remove-selected, `stats` (filesystem-walk report) (and `ns.files.web.*` for file-based webapps)
- `ns.filter.*` — `export` (creates symlinks from a TOML config), `copy` (retargets symlinks to a new root), `scorer` (filter via trained classifier)
- `ns.ml.*` — `classify`, `scorer`, `train` (latter lives in `nightskycam_images.classifier`)
- `ns.backup.*` — backup pipeline routing
- `ns.db.*` — `update` (rebuild/refresh SQLite mirror; optional inline classifier pass), `stats` (DB-derived summary incl. classifier scores), `web.view`
- `ns.annotate` — annotation CLI (`nightskycam_images.annotator.cli`)
- `ns.util.*` — utilities (e.g., `patches`)

> **Note:** `ns.db.stats` queries the SQLite mirror (fast, requires `ns.db.update` first). `ns.files.stats` walks the filesystem directly (no DB required). Pick based on whether you want classifier-aware reporting or a fresh filesystem-truth snapshot.

### `ns.db.update --classifier-config` — interleaved classification + DB upsert

`ns.db.update` accepts an optional `--classifier-config PATH` pointing at a TOML in the same format as `ns.ml.classify`. When set, for every image with a thumbnail + TOML, the configured classifiers are run **before** the row is upserted — TOML write and DB upsert happen in the same per-image step. This avoids the directory-mtime skip pitfall a sequential `ns.ml.classify` → `ns.db.update` flow has (in-place TOML rewrites don't bump parent-dir mtimes, so incremental populate can miss them).

Only the `[models]` section of the classifier config is honored in this code path: `root`, `second_root`, `systems`, `start_date`, `end_date` are ignored — `ns.db.update`'s own `ROOT [SECOND_ROOT]` CLI args define the scope.

**Per-key overwrite semantics** (shared by `ns.ml.classify` and `ns.db.update --classifier-config`):
- Default: only fill missing classifier keys; existing scores are kept.
- With `--classifier-overwrite`: every configured classifier is rerun and its score replaced.
- Unconfigured classifier keys already present in a TOML's `[classifiers]` table are preserved either way.

The per-image work lives in `nightskycam_images/classifier_runner.py` (`load_classifiers`, `classify_image_inplace`, `make_populate_enricher`); `db.populate()` accepts an optional `enricher` callback so `db.py` stays free of any `SkyScorer` import.

`ns.ml.classify`'s config gained an optional `second_root` field; when set, both roots are walked.

Most non-trivial commands take a TOML config file; many support `--create-config` to scaffold one and `--debug` for verbose logging.

## CI

GitHub Actions on push/PR to master: installs poetry, runs `poetry run pytest` on Python 3.12.3 / ubuntu-24.04.
