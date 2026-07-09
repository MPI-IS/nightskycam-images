"""HTML pages: home dashboard, browse grid, image detail."""

from pathlib import Path
import tomllib
from typing import Any, Dict, Optional

from flask import Blueprint, abort, current_app, render_template, request

from ..db_api import ImageDB
from ..weather import get_weather_icon
from .queries import record_at, run_query
from .spec import FilterSpec, FilterSpecError
from .stats import dashboard_stats, site_meta

pages_bp = Blueprint("pages", __name__)

# TOML keys whose values are already shown from the DB row.
_TOML_KEYS_IN_DB = {"weather", "cloud_cover", "process", "classifiers"}


def _db() -> ImageDB:
    return ImageDB(current_app.config["DB_PATH"])


@pages_bp.app_template_filter("weather_icon")
def _weather_icon(weather: Optional[str]) -> str:
    return get_weather_icon(weather) if weather else ""


@pages_bp.route("/")
def home() -> str:
    stats = dashboard_stats(current_app.config["DB_PATH"])
    return render_template("home.html", stats=stats)


@pages_bp.route("/ask")
def ask_page() -> str:
    from .api import agent_available

    if not agent_available():
        abort(404)
    return render_template("ask.html")


@pages_bp.route("/browse")
def browse() -> Any:
    meta = site_meta(current_app.config["DB_PATH"])
    try:
        spec = FilterSpec.from_query_args(request.args)
    except FilterSpecError as error:
        empty = FilterSpec()
        return (
            render_template(
                "browse.html",
                meta=meta,
                spec=empty,
                error=str(error),
                result=None,
            ),
            400,
        )
    result = run_query(_db(), spec)
    return render_template(
        "browse.html", meta=meta, spec=spec, error=None, result=result
    )


@pages_bp.route("/image/<filename_stem>")
def image_detail(filename_stem: str) -> str:
    record = _db().image(filename_stem)
    if record is None:
        abort(404)

    toml_extras = _toml_extras(record.toml_path)

    # Stateless prev/next navigation: links from the browse grid carry the
    # filter query string plus the record's absolute index `i`.
    nav = None
    index_arg = request.args.get("i")
    if index_arg is not None and index_arg.isdigit():
        try:
            spec = FilterSpec.from_query_args(request.args)
        except FilterSpecError:
            spec = None
        if spec is not None:
            index = int(index_arg)
            db = _db()
            previous = record_at(db, spec, index - 1)
            following = record_at(db, spec, index + 1)
            back_page = index // spec.page_size + 1
            nav = {
                "index": index,
                "query_string": spec.to_query_string(),
                "prev_stem": previous.filename_stem if previous else None,
                "next_stem": following.filename_stem if following else None,
                "back_query_string": spec.to_query_string(page=back_page),
            }

    return render_template(
        "image.html", record=record, toml_extras=toml_extras, nav=nav
    )


def _toml_extras(toml_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Per-image TOML metadata not already mirrored in the DB row."""
    if toml_path is None:
        return None
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    extras = {k: v for k, v in data.items() if k not in _TOML_KEYS_IN_DB}
    return extras or None
