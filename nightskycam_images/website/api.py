"""
JSON API.

``POST /api/query`` is the programmatic twin of the browse page.
``POST /api/ask`` runs the LLM agent (``agent.py``) over the same
query layer; it is enabled only when a SAIA API key is configured.
"""

from typing import Any, Dict, List, Tuple, Union

from flask import Blueprint, current_app, jsonify, request, url_for

from ..db_api import ImageDB, ImageRecord
from .agent import AgentError, run_agent
from .queries import run_query
from .spec import FilterSpec, FilterSpecError
from .stats import dashboard_stats, site_meta

api_bp = Blueprint("api", __name__, url_prefix="/api")


def agent_available() -> bool:
    """The Ask feature is on when a key (or an injected client) exists."""
    return bool(
        current_app.config.get("SAIA_API_KEY") or current_app.config.get("SAIA_CLIENT")
    )


@api_bp.route("/meta")
def meta() -> Any:
    return jsonify(site_meta(current_app.config["DB_PATH"]))


@api_bp.route("/stats")
def stats() -> Any:
    return jsonify(dashboard_stats(current_app.config["DB_PATH"]))


@api_bp.route("/filter-schema")
def filter_schema() -> Any:
    return jsonify(FilterSpec.json_schema())


@api_bp.route("/query", methods=["POST"])
def query() -> Union[Any, Tuple[Any, int]]:
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "expected a JSON body"}), 400
    try:
        spec = FilterSpec.from_json(body)
    except FilterSpecError as error:
        return jsonify({"error": str(error), "field": error.field}), 400

    db = ImageDB(current_app.config["DB_PATH"])
    result = run_query(db, spec)
    return jsonify(
        {
            "spec": spec.to_json(),
            "total": result.total,
            "page": result.page,
            "pages": result.pages,
            "page_size": result.page_size,
            "images": [_serialize(record) for record in result.records],
        }
    )


@api_bp.route("/ask", methods=["POST"])
def ask() -> Union[Any, Tuple[Any, int]]:
    if not agent_available():
        return (
            jsonify(
                {
                    "error": (
                        "natural-language querying is not configured "
                        "(SAIA_API_KEY unset)"
                    )
                }
            ),
            503,
        )
    messages = _validate_ask_body(request.get_json(silent=True))
    if isinstance(messages, str):
        return jsonify({"error": messages}), 400

    try:
        answer = run_agent(
            current_app.config["DB_PATH"],
            _saia_client(),
            current_app.config["SAIA_MODEL"],
            messages,
        )
    except AgentError as error:
        return jsonify({"error": str(error)}), 502

    db = ImageDB(current_app.config["DB_PATH"])
    images = []
    for stem in answer.image_stems:
        record = db.image(stem)
        if record is not None:  # second guard against unknown stems
            images.append(_serialize(record))
    return jsonify(
        {
            "answer": {"text": answer.text, "images": images},
            "steps": [
                {"tool": step.tool, "summary": step.summary} for step in answer.steps
            ],
            "rounds": answer.rounds,
            "truncated": answer.truncated,
            "model": current_app.config["SAIA_MODEL"],
        }
    )


def _validate_ask_body(body: Any) -> Union[List[Dict[str, Any]], str]:
    """Return the validated message list, or an error string."""
    if not isinstance(body, dict):
        return "expected a JSON body"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "expected a non-empty 'messages' list"
    for message in messages:
        if not isinstance(message, dict):
            return "each message must be an object"
        if message.get("role") not in ("user", "assistant"):
            return "message role must be 'user' or 'assistant'"
        if not isinstance(message.get("content"), str):
            return "message content must be a string"
        stems = message.get("image_stems")
        if stems is not None and (
            not isinstance(stems, list) or not all(isinstance(s, str) for s in stems)
        ):
            return "image_stems must be a list of strings"
    if messages[-1]["role"] != "user":
        return "the last message must be from the user"
    return messages


def _saia_client() -> Any:
    """The injected fake client, or a cached real OpenAI client."""
    injected = current_app.config.get("SAIA_CLIENT")
    if injected is not None:
        return injected
    cached = current_app.config.get("_SAIA_CLIENT_CACHE")
    if cached is None:
        import openai

        cached = openai.OpenAI(
            api_key=current_app.config["SAIA_API_KEY"],
            base_url=current_app.config["SAIA_BASE_URL"],
            timeout=30,
            max_retries=1,
        )
        current_app.config["_SAIA_CLIENT_CACHE"] = cached
    return cached


def _serialize(record: ImageRecord) -> Dict[str, Any]:
    stem = record.filename_stem
    return {
        "filename_stem": stem,
        "system": record.system,
        "date": record.date,
        "time": record.time,
        "datetime": record.datetime,
        "nightstart_date": record.nightstart_date,
        "location": record.location,
        "image_format": record.image_format,
        "process": record.process,
        "weather": record.weather,
        "cloud_cover": record.cloud_cover,
        "stretched": record.stretched,
        "classifier_scores": record.classifier_scores,
        "urls": {
            "thumb": url_for("media.thumb", filename_stem=stem),
            "jpeg": url_for("media.jpeg", filename_stem=stem),
            "download": url_for("media.download", filename_stem=stem),
            "page": url_for("pages.image_detail", filename_stem=stem),
        },
    }
