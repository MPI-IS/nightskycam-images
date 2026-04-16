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

Key modules:
- `walk.py` — filesystem traversal: iterates systems, dates, images; core filtering logic (`filter_and_export_images`); symlink creation for filtered exports
- `main.py` — CLI entry points (20+ commands via typer), wires together other modules
- `filters.py` — predicate-based filtering (process, cloud cover, weather fields from TOML metadata)
- `image.py` — `Image` class holding paths and metadata for a single capture
- `thumbnail.py` — thumbnail generation with parallel processing
- `video.py` — video creation from image sequences
- `weather.py` — `WeatherReport` parsing from TOML weather summaries
- `constants.py` — formats, dimensions, directory names (e.g., `IMAGE_FILE_FORMATS`, `THUMBNAIL_DIR_NAME`)
- `convert_npy.py` — NumPy array to image conversion with auto-stretching
- `patches.py` — image patching utilities
- `annotator_webapp.py`, `view_webapp.py`, `symlink_annotator_webapp.py` — Flask web interfaces

**Image formats supported:** npy, tiff, jpg, jpeg. Thumbnails are always JPEG at 200px width.

**CLI entry points** are defined in `pyproject.toml` under `[tool.poetry.scripts]`, all pointing into `nightskycam_images.main` or subpackage CLI modules.

## CI

GitHub Actions on push/PR to master: installs poetry, runs `poetry run pytest` on Python 3.12.3 / ubuntu-24.04.
