"""
FilterSpec — the canonical, serializable image-filter representation.

Every way of querying the website funnels through this one dataclass:

- the browse form (GET query string) via :meth:`FilterSpec.from_query_args`,
- the JSON API via :meth:`FilterSpec.from_json`,
- a future natural-language endpoint, where an LLM is given
  :meth:`FilterSpec.json_schema` as its structured-output schema and its
  answer is validated by the same :meth:`FilterSpec.from_json`.

Nothing outside this module should know how a FilterSpec is born; the
mapping to :class:`~nightskycam_images.db_api.ImageDB` arguments happens
in exactly one place (:meth:`FilterSpec.to_query_kwargs`).
"""

from dataclasses import dataclass, fields
import re
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode

from ..constants import IMAGE_FILE_FORMATS

PAGE_SIZE_DEFAULT = 60
PAGE_SIZE_MIN = 12
PAGE_SIZE_MAX = 200

_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
_TIME_RE = re.compile(r"^\d{2}_\d{2}_\d{2}$")
_ORDER_BY_VALUES = ("datetime", "position")


class FilterSpecError(ValueError):
    """Invalid filter input; carries the offending field name."""

    def __init__(self, field_name: str, message: str) -> None:
        self.field = field_name
        super().__init__(f"{field_name}: {message}")


@dataclass
class FilterSpec:
    """A validated set of image filters plus ordering and pagination."""

    systems: Optional[List[str]] = None
    locations: Optional[List[str]] = None  # deployment sites, lowercase
    start_date: Optional[str] = None  # "YYYY_MM_DD"
    end_date: Optional[str] = None
    start_time: Optional[str] = None  # "HH_MM_SS"; start > end wraps midnight
    end_time: Optional[str] = None
    weather: Optional[List[str]] = None  # OR-combined substrings
    cloud_cover_min: Optional[int] = None  # percent, 0..100
    cloud_cover_max: Optional[int] = None
    image_format: Optional[str] = None  # "npy" | "tiff" | "jpg" | "jpeg"
    stretched: Optional[bool] = None
    classifier_max: Optional[Dict[str, float]] = None  # {"cloudy": 0.3}
    order_by: str = "datetime"
    descending: bool = True  # newest first by default
    page: int = 1
    page_size: int = PAGE_SIZE_DEFAULT

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.locations is not None:
            # DB values are canonical lowercase site names.
            self.locations = [
                v.strip().lower() for v in self.locations if v.strip()
            ] or None
        for name in ("start_date", "end_date"):
            value = getattr(self, name)
            if value is not None:
                # Accept the HTML-date/ISO "YYYY-MM-DD" spelling too.
                value = value.replace("-", "_")
                setattr(self, name, value)
                if not _DATE_RE.match(value):
                    raise FilterSpecError(name, f"expected YYYY_MM_DD, got {value!r}")
        for name in ("start_time", "end_time"):
            value = getattr(self, name)
            if value is not None:
                # Accept HTML-time "HH:MM" / "HH:MM:SS" spellings too.
                value = value.replace(":", "_")
                if _TIME_RE.match(f"{value}_00"):
                    value = f"{value}_00"
                setattr(self, name, value)
                if not _TIME_RE.match(value):
                    raise FilterSpecError(name, f"expected HH_MM_SS, got {value!r}")
        for name in ("cloud_cover_min", "cloud_cover_max"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 100:
                raise FilterSpecError(name, f"expected 0..100, got {value!r}")
        if (
            self.image_format is not None
            and self.image_format not in IMAGE_FILE_FORMATS
        ):
            raise FilterSpecError(
                "image_format",
                f"expected one of {IMAGE_FILE_FORMATS}, got {self.image_format!r}",
            )
        if self.classifier_max is not None:
            for name, value in self.classifier_max.items():
                if not 0.0 <= value <= 1.0:
                    raise FilterSpecError(
                        "classifier_max", f"{name}: expected 0..1, got {value!r}"
                    )
        if self.order_by not in _ORDER_BY_VALUES:
            raise FilterSpecError(
                "order_by",
                f"expected one of {_ORDER_BY_VALUES}, got {self.order_by!r}",
            )
        if self.page < 1:
            raise FilterSpecError("page", f"expected >= 1, got {self.page!r}")
        self.page_size = max(PAGE_SIZE_MIN, min(PAGE_SIZE_MAX, self.page_size))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_query_args(cls, args: Mapping[str, Any]) -> "FilterSpec":
        """
        Build a spec from a request query string (``request.args``).

        List fields use repeated parameters (``?systems=a&systems=b``);
        classifier thresholds use dotted keys (``?classifier_max.cloudy=0.3``).
        Empty-string values are treated as absent so that blank form
        fields do not filter.
        """

        def get(name: str) -> Optional[str]:
            value = args.get(name)
            if value is None:
                return None
            value = str(value).strip()
            return value or None

        def getlist(name: str) -> Optional[List[str]]:
            if hasattr(args, "getlist"):
                values = [str(v).strip() for v in args.getlist(name)]
            else:  # plain dict (tests, internal calls)
                raw = args.get(name)
                if raw is None:
                    values = []
                elif isinstance(raw, (list, tuple)):
                    values = [str(v).strip() for v in raw]
                else:
                    values = [str(raw).strip()]
            values = [v for v in values if v]
            return values or None

        classifier_max: Dict[str, float] = {}
        for key in args.keys():
            if key.startswith("classifier_max."):
                name = key[len("classifier_max.") :]
                value = get(key)
                if value is None:
                    continue
                classifier_max[name] = _parse_float("classifier_max", value)

        return cls(
            systems=getlist("systems"),
            locations=getlist("locations"),
            start_date=get("start_date"),
            end_date=get("end_date"),
            start_time=get("start_time"),
            end_time=get("end_time"),
            weather=getlist("weather"),
            cloud_cover_min=_parse_optional_int(
                "cloud_cover_min", get("cloud_cover_min")
            ),
            cloud_cover_max=_parse_optional_int(
                "cloud_cover_max", get("cloud_cover_max")
            ),
            image_format=get("image_format"),
            stretched=_parse_optional_bool("stretched", get("stretched")),
            classifier_max=classifier_max or None,
            order_by=get("order_by") or "datetime",
            descending=_parse_bool("descending", get("descending"), True),
            page=_first_not_none(_parse_optional_int("page", get("page")), 1),
            page_size=_first_not_none(
                _parse_optional_int("page_size", get("page_size")),
                PAGE_SIZE_DEFAULT,
            ),
        )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "FilterSpec":
        """
        Build a spec from a JSON object (the API request body, and — in a
        future version — the LLM structured output).
        """
        if not isinstance(data, Mapping):
            raise FilterSpecError("spec", "expected a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - known
        if unknown:
            raise FilterSpecError(
                "spec", f"unknown field(s): {', '.join(sorted(unknown))}"
            )

        kwargs: Dict[str, Any] = {}
        for name in ("systems", "locations", "weather"):
            if data.get(name) is not None:
                value = data[name]
                if not isinstance(value, list) or not all(
                    isinstance(v, str) for v in value
                ):
                    raise FilterSpecError(name, "expected a list of strings")
                kwargs[name] = value or None
        for name in (
            "start_date",
            "end_date",
            "start_time",
            "end_time",
            "image_format",
            "order_by",
        ):
            if data.get(name) is not None:
                if not isinstance(data[name], str):
                    raise FilterSpecError(name, "expected a string")
                kwargs[name] = data[name]
        for name in ("cloud_cover_min", "cloud_cover_max", "page", "page_size"):
            if data.get(name) is not None:
                value = data[name]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise FilterSpecError(name, "expected an integer")
                kwargs[name] = value
        for name in ("stretched", "descending"):
            if data.get(name) is not None:
                if not isinstance(data[name], bool):
                    raise FilterSpecError(name, "expected a boolean")
                kwargs[name] = data[name]
        if data.get("classifier_max") is not None:
            value = data["classifier_max"]
            if not isinstance(value, Mapping):
                raise FilterSpecError(
                    "classifier_max", "expected an object of {name: max_prob}"
                )
            parsed: Dict[str, float] = {}
            for key, threshold in value.items():
                if isinstance(threshold, bool) or not isinstance(
                    threshold, (int, float)
                ):
                    raise FilterSpecError("classifier_max", f"{key}: expected a number")
                parsed[str(key)] = float(threshold)
            kwargs["classifier_max"] = parsed or None
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_query_kwargs(self) -> Dict[str, Any]:
        """Map to ``ImageDB.images()`` / ``ImageDB.count()`` keyword args."""
        kwargs: Dict[str, Any] = {}
        for name in (
            "systems",
            "locations",
            "start_date",
            "end_date",
            "start_time",
            "end_time",
            "weather",
            "cloud_cover_min",
            "cloud_cover_max",
            "image_format",
            "stretched",
            "classifier_max",
        ):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return kwargs

    def to_page_kwargs(self) -> Dict[str, Any]:
        """Ordering + pagination keyword args for ``ImageDB.images()``."""
        return {
            "order_by": self.order_by,
            "descending": self.descending,
            "limit": self.page_size,
            "offset": (self.page - 1) * self.page_size,
        }

    def to_json(self) -> Dict[str, Any]:
        """JSON-safe dict with defaults omitted (round-trips via from_json)."""
        result: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None and value != f.default:
                result[f.name] = value
        return result

    def to_query_string(self, page: Optional[int] = None) -> str:
        """
        Encode as a URL query string (inverse of :meth:`from_query_args`),
        optionally overriding the page — used for pagination and detail links.
        """
        items: List[tuple] = []
        for name in ("systems", "locations", "weather"):
            for value in getattr(self, name) or []:
                items.append((name, value))
        for name in (
            "start_date",
            "end_date",
            "start_time",
            "end_time",
            "cloud_cover_min",
            "cloud_cover_max",
            "image_format",
        ):
            value = getattr(self, name)
            if value is not None:
                items.append((name, value))
        if self.stretched is not None:
            items.append(("stretched", "1" if self.stretched else "0"))
        for name, value in (self.classifier_max or {}).items():
            items.append((f"classifier_max.{name}", value))
        if self.order_by != "datetime":
            items.append(("order_by", self.order_by))
        if not self.descending:
            items.append(("descending", "0"))
        if self.page_size != PAGE_SIZE_DEFAULT:
            items.append(("page_size", self.page_size))
        effective_page = self.page if page is None else page
        if effective_page != 1:
            items.append(("page", effective_page))
        return urlencode(items)

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        """
        JSON Schema of the filter contract, with per-field descriptions.

        Written to double as an LLM structured-output/tool schema for the
        future natural-language filter endpoint.
        """
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "FilterSpec",
            "description": (
                "Filters selecting nightskycam images. All fields are "
                "optional; omitted fields do not filter. Multiple fields "
                "are combined with AND."
            ),
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "systems": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Camera system names to include, e.g. "
                        "['nightskycam5']. Omit for all systems."
                    ),
                },
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Deployment site(s) of the camera when the image "
                        "was taken (systems move over time), lowercase, "
                        "e.g. ['palma']. Images from periods with unknown "
                        "deployment have no location and never match this "
                        "filter. Omit for all sites."
                    ),
                },
                "start_date": {
                    "type": "string",
                    "pattern": r"^\d{4}_\d{2}_\d{2}$",
                    "description": "Inclusive first date, format YYYY_MM_DD.",
                },
                "end_date": {
                    "type": "string",
                    "pattern": r"^\d{4}_\d{2}_\d{2}$",
                    "description": "Inclusive last date, format YYYY_MM_DD.",
                },
                "start_time": {
                    "type": "string",
                    "pattern": r"^\d{2}_\d{2}_\d{2}$",
                    "description": (
                        "Time-of-day window start, format HH_MM_SS. If "
                        "start_time > end_time the window crosses midnight "
                        "(e.g. 22_00_00 to 04_00_00)."
                    ),
                },
                "end_time": {
                    "type": "string",
                    "pattern": r"^\d{2}_\d{2}_\d{2}$",
                    "description": "Time-of-day window end, format HH_MM_SS.",
                },
                "weather": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Weather description substrings (meteosource "
                        "vocabulary, e.g. 'clear', 'overcast', 'rain'); an "
                        "image matches if ANY substring occurs in its "
                        "weather field."
                    ),
                },
                "cloud_cover_min": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Minimum cloud cover percentage, inclusive.",
                },
                "cloud_cover_max": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Maximum cloud cover percentage, inclusive.",
                },
                "image_format": {
                    "type": "string",
                    "enum": list(IMAGE_FILE_FORMATS),
                    "description": "Original file format of the image.",
                },
                "stretched": {
                    "type": "boolean",
                    "description": (
                        "Whether auto-stretching was applied to the image "
                        "file on disk during processing."
                    ),
                },
                "classifier_max": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "description": (
                        "Per-classifier maximum probability, e.g. "
                        "{'cloudy': 0.2} keeps images the 'cloudy' "
                        "classifier scored at most 0.2 (i.e. likely NOT "
                        "cloudy). Lower = stricter."
                    ),
                },
                "order_by": {
                    "type": "string",
                    "enum": list(_ORDER_BY_VALUES),
                    "description": (
                        "'datetime' sorts chronologically (default); "
                        "'position' sorts by system, then date, then time."
                    ),
                },
                "descending": {
                    "type": "boolean",
                    "description": "Sort newest first (default true).",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based result page.",
                },
                "page_size": {
                    "type": "integer",
                    "minimum": PAGE_SIZE_MIN,
                    "maximum": PAGE_SIZE_MAX,
                    "description": f"Images per page (default {PAGE_SIZE_DEFAULT}).",
                },
            },
        }


def _first_not_none(value: Optional[int], default: int) -> int:
    return default if value is None else value


def _parse_optional_int(field_name: str, value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise FilterSpecError(field_name, f"expected an integer, got {value!r}")


def _parse_float(field_name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise FilterSpecError(field_name, f"expected a number, got {value!r}")


def _parse_optional_bool(field_name: str, value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise FilterSpecError(field_name, f"expected a boolean, got {value!r}")


def _parse_bool(field_name: str, value: Optional[str], default: bool) -> bool:
    parsed = _parse_optional_bool(field_name, value)
    return default if parsed is None else parsed
