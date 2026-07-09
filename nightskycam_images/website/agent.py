"""
LLM agent answering natural-language questions about the image database.

The agent runs a bounded tool-calling loop against an OpenAI-compatible
chat-completions API (SAIA, GWDG's academic LLM service). All tools are
thin read-only wrappers over the existing query layer; model-produced
filters are validated by :class:`~.spec.FilterSpec` and validation
errors are fed back so the model can self-correct.

Anti-hallucination guarantee: the final answer may only reference
``filename_stem`` values that a tool actually returned during this
request (or that were carried in the validated conversation history).

This module is Flask-free so the loop is unit-testable with a fake
client; the HTTP endpoint lives in ``api.py``.
"""

from dataclasses import dataclass, field
import datetime as dt
import json
from pathlib import Path
import re
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..db_api import ImageDB
from .queries import run_query
from .spec import FilterSpec, FilterSpecError
from .stats import dashboard_stats, site_meta

DEFAULT_MAX_ROUNDS = 8
DEFAULT_TIME_BUDGET_S = 60.0
AGENT_MAX_PAGE_SIZE = 24
MAX_ANSWER_IMAGES = 24
MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARS = 4000
TOOL_RESULT_MAX_CHARS = 8000
CHAT_TIMEOUT_S = 30
TEMPERATURE = 0.1


class AgentError(Exception):
    """Base class for agent failures surfaced to the caller."""


class AgentUpstreamError(AgentError):
    """SAIA unreachable, timed out, rejected the key, or returned 5xx."""


class AgentModelError(AgentError):
    """The configured model appears not to support tool calling."""


@dataclass
class AgentStep:
    """One executed tool call, for the UI transparency trace."""

    tool: str
    arguments: Dict[str, Any]
    summary: str


@dataclass
class AgentAnswer:
    text: str
    image_stems: List[str]
    steps: List[AgentStep]
    rounds: int
    truncated: bool


@dataclass
class ToolContext:
    db: ImageDB
    db_path: Path
    # Stems returned by tools during THIS request (plus validated history
    # stems); the only stems final_answer may reference.
    seen_stems: Set[str] = field(default_factory=set)


# ----------------------------------------------------------------------
# Tool registry
# ----------------------------------------------------------------------

ToolHandler = Callable[[ToolContext, Dict[str, Any]], Any]

TOOLS: Dict[str, Tuple[Dict[str, Any], ToolHandler]] = {}


def _tool(name: str, description: str, parameters: Dict[str, Any]) -> Callable:
    def wrap(fn: ToolHandler) -> ToolHandler:
        TOOLS[name] = (
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
            fn,
        )
        return fn

    return wrap


def _filter_params() -> Dict[str, Any]:
    """FilterSpec's JSON schema, adapted for use as a tool parameter."""
    schema = FilterSpec.json_schema()
    schema.pop("$schema", None)
    schema.pop("title", None)
    return schema


def _compact_record(record: Any) -> Dict[str, Any]:
    return {
        "filename_stem": record.filename_stem,
        "system": record.system,
        "date": record.date,
        "time": record.time,
        "location": record.location,
        "weather": record.weather,
        "cloud_cover": record.cloud_cover,
        "image_format": record.image_format,
        "classifier_scores": record.classifier_scores,
    }


@_tool(
    "get_meta",
    "Vocabulary of the database: system names, classifier names, the "
    "distinct weather strings, image formats, and the min/max dates "
    "present. Call this before guessing any weather or system value.",
    {"type": "object", "properties": {}},
)
def _get_meta(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return site_meta(ctx.db_path)


@_tool(
    "get_stats",
    "Aggregate statistics: image totals per system and per format, "
    "weather distribution, classifier coverage, date ranges, newest "
    "date per system.",
    {"type": "object", "properties": {}},
)
def _get_stats(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    return dashboard_stats(ctx.db_path)


@_tool(
    "count_images",
    "Count images matching a filter. Cheap - use it to check result "
    "sizes before fetching and to answer 'how many' questions.",
    {
        "type": "object",
        "properties": {"filter": _filter_params()},
        "required": ["filter"],
    },
)
def _count_images(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    spec = FilterSpec.from_json(args.get("filter") or {})
    return {"total": ctx.db.count(**spec.to_query_kwargs())}


@_tool(
    "query_images",
    f"Fetch matching images (max {AGENT_MAX_PAGE_SIZE} per call) with "
    "key metadata. Use the returned filename_stem values when answering "
    "with images.",
    {
        "type": "object",
        "properties": {"filter": _filter_params()},
        "required": ["filter"],
    },
)
def _query_images(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    spec = FilterSpec.from_json(args.get("filter") or {})
    spec.page_size = min(spec.page_size, AGENT_MAX_PAGE_SIZE)
    result = run_query(ctx.db, spec)
    images = [_compact_record(r) for r in result.records]
    ctx.seen_stems.update(img["filename_stem"] for img in images)
    return {
        "total": result.total,
        "page": result.page,
        "pages": result.pages,
        "images": images,
    }


@_tool(
    "get_image",
    "Full metadata for one image by filename_stem.",
    {
        "type": "object",
        "properties": {"filename_stem": {"type": "string"}},
        "required": ["filename_stem"],
    },
)
def _get_image(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    record = ctx.db.image(str(args.get("filename_stem", "")))
    if record is None:
        return {"error": "no image with that filename_stem"}
    ctx.seen_stems.add(record.filename_stem)
    compact = _compact_record(record)
    compact["process"] = record.process
    compact["nightstart_date"] = record.nightstart_date
    return compact


@_tool(
    "final_answer",
    "REQUIRED last step: deliver the answer to the user. `text` is "
    "shown as-is; `image_stems` must be filename_stem values that "
    "appeared in earlier query_images/get_image results in this "
    "conversation. Use [] for text-only answers, one stem for 'show me "
    "an image', several for sets.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "image_stems": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_ANSWER_IMAGES,
            },
        },
        "required": ["text"],
    },
)
def _final_answer(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    # Intercepted by the loop; present only for its schema.
    return None


# ----------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------


def build_system_prompt(meta: Dict[str, Any], today: dt.date) -> str:
    schema = json.dumps(_filter_params(), indent=2)
    weather_values = ", ".join(meta.get("weather_values") or [])
    classifier_names = ", ".join(meta.get("classifier_names") or [])
    return f"""\
You are the query assistant of a nightskycam all-sky image database.
You answer questions about the images by calling the provided read-only
tools; NEVER answer from memory or invent data.

Database vocabulary (live):
- systems: {", ".join(meta.get("systems") or [])}
- locations (deployment sites; systems move over time): \
{", ".join(meta.get("locations") or []) or "none recorded"}
- classifiers: {classifier_names}
- weather values: {weather_values}
- image formats: {", ".join(meta.get("image_formats") or [])}
- data covers dates {meta.get("date_min")} to {meta.get("date_max")}

Today is {today.isoformat()} ({today.strftime("%Y_%m_%d")}). Resolve
relative phrases like "last October" or "this month" against today,
keeping in mind data may end at {meta.get("date_max")}.

Filter format rules:
- dates are "YYYY_MM_DD" (underscores), times are "HH_MM_SS"
- when start_time > end_time the time window crosses midnight (typical
  for nights, e.g. 22_00_00 to 04_00_00)
- weather filters are OR-combined substrings of the weather values above
- filter by site with the "locations" list (lowercase names from the
  vocabulary above); images from periods with unknown deployment have
  no location and never match a locations filter
- classifier scores are probabilities 0..1; classifier_max keeps images
  scored AT OR BELOW the threshold. {{"cloudy": 0.2}} means "likely NOT
  cloudy" (good for "clear sky"). There is no minimum threshold, so for
  "rainy images" prefer the weather substring "rain" or high cloud
  cover instead of classifier filters. Not every image has scores.

The `filter` parameter of count_images/query_images follows this JSON
schema:
{schema}

Workflow:
- call get_meta if you are unsure about any vocabulary value
- call count_images first to gauge result sizes; keep page_size small
- use order_by/descending for "newest"/"oldest"/"first"/"latest"
- prefer few, precise tool calls
- you MUST finish by calling final_answer
- to show images, put their filename_stem values in the image_stems
  ARGUMENT of final_answer — never write stems or "image_stems" into
  the text; the text should read naturally without them
- image_stems may ONLY contain filename_stem values you saw in tool
  results of this conversation; use [] for text-only answers
- pick 1 stem when asked for "an image", at most 12 for sets unless the
  user asks for more
- if nothing matches, say so and suggest how to relax the filters

Answer style: plain concise text (no markdown tables). Mention the
filters you applied when it helps the user understand the result.
"""


# ----------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------


def run_agent(
    db_path: Path,
    client: Any,
    model: str,
    messages: Sequence[Dict[str, Any]],
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    on_step: Optional[Callable[[AgentStep], None]] = None,
) -> AgentAnswer:
    """
    Run the tool-calling loop and return the final answer.

    Parameters
    ----------
    db_path
        SQLite database path (read-only access).
    client
        An ``openai.OpenAI``-shaped client (``.chat.completions.create``).
    model
        Model identifier passed to the API.
    messages
        Conversation history: dicts with ``role`` ("user"/"assistant"),
        ``content`` (str) and optionally ``image_stems`` on assistant
        turns. The caller validates roles; this function truncates.
    """
    ctx = ToolContext(db=ImageDB(db_path), db_path=db_path)
    meta = site_meta(db_path)

    llm_messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(meta, dt.date.today()),
        }
    ]
    for message in list(messages)[-MAX_HISTORY_MESSAGES:]:
        content = str(message.get("content", ""))[:MAX_MESSAGE_CHARS]
        stems = message.get("image_stems") or []
        if message.get("role") == "assistant" and stems:
            valid = [s for s in stems if isinstance(s, str)][:MAX_ANSWER_IMAGES]
            ctx.seen_stems.update(valid)
            content += "\n[images shown: " + ", ".join(valid) + "]"
        llm_messages.append({"role": message["role"], "content": content})

    tool_schemas = [schema for schema, _ in TOOLS.values()]
    steps: List[AgentStep] = []
    deadline = time.monotonic() + time_budget_s
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        if time.monotonic() > deadline:
            return _forced_finish(client, model, llm_messages, ctx, steps, rounds)

        reply = _chat(client, model, llm_messages, tools=tool_schemas)

        if not getattr(reply, "tool_calls", None):
            # Uncooperative but harmless: treat plain text as the answer.
            text, stems = _recover_stems_from_text(
                (reply.content or "").strip(), [], ctx.seen_stems
            )
            return AgentAnswer(
                text=text,
                image_stems=stems,
                steps=steps,
                rounds=rounds,
                truncated=False,
            )

        llm_messages.append(_as_assistant_message(reply))
        for call in reply.tool_calls:
            name = call.function.name
            args = _parse_arguments(call.function.arguments)

            if name == "final_answer" and "error" not in args:
                text, stems = _recover_stems_from_text(
                    str(args.get("text", "")).strip(),
                    _validate_stems(args.get("image_stems") or [], ctx.seen_stems),
                    ctx.seen_stems,
                )
                return AgentAnswer(
                    text=text,
                    image_stems=stems,
                    steps=steps,
                    rounds=rounds,
                    truncated=False,
                )

            result = _dispatch(ctx, name, args)
            step = AgentStep(
                tool=name, arguments=args, summary=_summarize(name, result)
            )
            steps.append(step)
            if on_step is not None:
                on_step(step)
            llm_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str)[:TOOL_RESULT_MAX_CHARS],
                }
            )

    return _forced_finish(client, model, llm_messages, ctx, steps, rounds)


def _forced_finish(
    client: Any,
    model: str,
    llm_messages: List[Dict[str, Any]],
    ctx: ToolContext,
    steps: List[AgentStep],
    rounds: int,
) -> AgentAnswer:
    """Budget exhausted: demand a final_answer with what we have."""
    llm_messages.append(
        {
            "role": "user",
            "content": (
                "Stop investigating. Call final_answer NOW with your "
                "best answer based on the tool results so far."
            ),
        }
    )
    try:
        reply = _chat(
            client,
            model,
            llm_messages,
            tools=[TOOLS["final_answer"][0]],
            tool_choice={
                "type": "function",
                "function": {"name": "final_answer"},
            },
        )
        for call in getattr(reply, "tool_calls", None) or []:
            if call.function.name != "final_answer":
                continue
            args = _parse_arguments(call.function.arguments)
            if "error" in args and set(args) == {"error"}:
                continue
            text, stems = _recover_stems_from_text(
                str(args.get("text", "")).strip(),
                _validate_stems(args.get("image_stems") or [], ctx.seen_stems),
                ctx.seen_stems,
            )
            return AgentAnswer(
                text=text,
                image_stems=stems,
                steps=steps,
                rounds=rounds,
                truncated=True,
            )
        if getattr(reply, "content", None):
            text, stems = _recover_stems_from_text(
                reply.content.strip(), [], ctx.seen_stems
            )
            return AgentAnswer(
                text=text,
                image_stems=stems,
                steps=steps,
                rounds=rounds,
                truncated=True,
            )
    except AgentError:
        pass
    return AgentAnswer(
        text=(
            "I could not finish answering within the allowed number of "
            "steps. Please try a more specific question."
        ),
        image_stems=[],
        steps=steps,
        rounds=rounds,
        truncated=True,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _chat(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    tools: List[Dict[str, Any]],
    tool_choice: Optional[Dict[str, Any]] = None,
) -> Any:
    """One chat-completions call; maps failures to AgentError types."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": TEMPERATURE,
        "timeout": CHAT_TIMEOUT_S,
    }
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
    except AgentError:
        raise
    except Exception as error:  # noqa: BLE001 - classified below
        raise _classify_error(error, model) from error
    return response.choices[0].message


def _classify_error(error: Exception, model: str) -> AgentError:
    import openai

    if isinstance(error, openai.AuthenticationError):
        return AgentUpstreamError("SAIA rejected the API key")
    if isinstance(error, openai.BadRequestError):
        message = str(error)
        if "tool" in message.lower() or "function" in message.lower():
            return AgentModelError(
                f"model {model!r} appears not to support tool calling; "
                "set SAIA_MODEL to a tool-capable model (see "
                "scripts/saia_smoke.py --list-models)"
            )
        return AgentUpstreamError(f"SAIA rejected the request: {message}")
    return AgentUpstreamError(f"SAIA request failed: {error}")


def _as_assistant_message(reply: Any) -> Dict[str, Any]:
    """Echo the model's tool-calling turn back into the message list."""
    return {
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in reply.tool_calls
        ],
    }


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"error": f"tool arguments are not valid JSON: {raw!r}"}
    if not isinstance(parsed, dict):
        return {"error": "tool arguments must be a JSON object"}
    return parsed


def _dispatch(ctx: ToolContext, name: str, args: Dict[str, Any]) -> Any:
    if "error" in args and set(args) == {"error"}:
        return args
    entry = TOOLS.get(name)
    if entry is None or name == "final_answer":
        available = sorted(n for n in TOOLS if n != "final_answer")
        return {"error": f"unknown tool {name!r}; available: {available}"}
    try:
        return entry[1](ctx, args)
    except FilterSpecError as error:
        return {
            "error": str(error),
            "field": error.field,
            "hint": (
                "fix the filter and retry; the FilterSpec schema is in "
                "the system prompt"
            ),
        }
    except Exception as error:  # noqa: BLE001 - tool must not kill the request
        return {"error": f"tool failed: {error}"}


# Smaller models sometimes write the stems into the answer text (e.g.
# 'image_stems: ["cam1_..."]') instead of the final_answer argument.
_STEM_LINE_RE = re.compile(
    r"^[ \t]*\W*image_stems\W*[:=].*$", re.MULTILINE | re.IGNORECASE
)


def _recover_stems_from_text(
    text: str, stems: List[str], seen: Set[str]
) -> Tuple[str, List[str]]:
    """
    If the answer text mentions tool-verified stems that are missing
    from ``stems``, promote them to the image list; strip any literal
    "image_stems: ..." line from the displayed text either way.
    """
    if not stems:
        mentioned = [s for s in seen if s in text]
        mentioned.sort(key=text.index)
        stems = mentioned[:MAX_ANSWER_IMAGES]
    cleaned = _STEM_LINE_RE.sub("", text).strip()
    return (cleaned or text, stems)


def _validate_stems(candidates: Any, seen: Set[str]) -> List[str]:
    """Keep only stems a tool actually returned, deduplicated, capped."""
    stems: List[str] = []
    added: Set[str] = set()
    if not isinstance(candidates, list):
        return stems
    for candidate in candidates:
        if isinstance(candidate, str) and candidate in seen and candidate not in added:
            stems.append(candidate)
            added.add(candidate)
        if len(stems) >= MAX_ANSWER_IMAGES:
            break
    return stems


def _summarize(name: str, result: Any) -> str:
    if isinstance(result, dict):
        if "error" in result:
            return f"{name}: error ({str(result['error'])[:80]})"
        if name == "count_images":
            return f"count_images: {result.get('total', '?'):,} match(es)"
        if name == "query_images":
            fetched = len(result.get("images", []))
            return f"query_images: fetched {fetched} of {result.get('total', '?'):,}"
        if name == "get_image":
            return f"get_image: {result.get('filename_stem', '?')}"
    return name
