"""
Modern web interface for querying and displaying nightskycam images.

The site is backed by the SQLite mirror maintained by ``ns.db.update``
(read-only access via :class:`nightskycam_images.db_api.ImageDB`) and
serves thumbnails from disk, full-resolution images with on-the-fly
16-bit -> 8-bit JPEG conversion (optionally auto-stretched), and the
original files for download.

Every query surface (the browse form, the JSON API and — in a future
version — an LLM translating natural language) funnels through the
canonical :class:`nightskycam_images.website.spec.FilterSpec`.

Usage::

    from nightskycam_images.website import create_app

    app = create_app("/path/to/nightskycam.db")
    app.run()
"""

import os
from pathlib import Path
from typing import Optional, Union

from flask import Flask

DEFAULT_SAIA_BASE_URL = "https://chat-ai.academiccloud.de/v1"
# Fast GWDG-hosted open-weight model with tool-calling support; verify
# the current id with `python scripts/saia_smoke.py --list-models`.
# The SAIA_MODEL environment variable always wins.
DEFAULT_SAIA_MODEL = "qwen3-30b-a3b-instruct-2507"


def create_app(
    db_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    cache_max_mb: int = 2048,
) -> Flask:
    """
    Build the website's Flask application.

    Parameters
    ----------
    db_path
        Path to the SQLite database file (opened read-only).
    cache_dir
        Directory for cached JPEG conversions of full-resolution images.
        Defaults to ``~/.cache/nightskycam-website``.
    cache_max_mb
        Approximate size cap of the conversion cache in megabytes;
        oldest entries are pruned past the cap.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"database not found: {db_path}")
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "nightskycam-website"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["DB_PATH"] = db_path
    app.config["CACHE_DIR"] = cache_dir
    app.config["CACHE_MAX_MB"] = cache_max_mb
    # LLM agent ("Ask" page): the feature is hidden unless a SAIA API
    # key is configured (or a fake client is injected for tests).
    app.config["SAIA_API_KEY"] = os.environ.get("SAIA_API_KEY")
    app.config["SAIA_BASE_URL"] = (
        os.environ.get("SAIA_BASE_URL") or DEFAULT_SAIA_BASE_URL
    )
    app.config["SAIA_MODEL"] = os.environ.get("SAIA_MODEL") or DEFAULT_SAIA_MODEL
    app.config.setdefault("SAIA_CLIENT", None)

    from .api import api_bp
    from .media import media_bp
    from .views import pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(media_bp)
    app.register_blueprint(api_bp)

    return app


def create_app_from_env() -> Flask:
    """
    Factory reading configuration from the environment, for running via
    e.g. ``gunicorn 'nightskycam_images.website:create_app_from_env()'``.

    Environment variables: ``NIGHTSKYCAM_DB_PATH`` (required),
    ``NIGHTSKYCAM_CACHE_DIR``, ``NIGHTSKYCAM_CACHE_MAX_MB``.
    """
    db_path = os.environ.get("NIGHTSKYCAM_DB_PATH")
    if not db_path:
        raise RuntimeError("NIGHTSKYCAM_DB_PATH is not set")
    return create_app(
        db_path,
        cache_dir=os.environ.get("NIGHTSKYCAM_CACHE_DIR"),
        cache_max_mb=int(os.environ.get("NIGHTSKYCAM_CACHE_MAX_MB", "2048")),
    )
