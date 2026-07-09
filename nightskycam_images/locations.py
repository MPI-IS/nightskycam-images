"""
Per-system deployment-location mapping (date ranges -> site name).

Camera systems move between sites over time. A TOML mapping file
records, per system, the date ranges spent at each location::

    [[nightskycam8]]
    start = "2024_10_19"
    end = "2024_11_14"        # omit for an ongoing deployment
    location = "tuebingen"    # canonical lowercase site name
    note = "free text"        # documentation only, never stored

Dates are inclusive on both ends and use the same ``YYYY_MM_DD``
format as the database's ``date`` column, so plain string comparison
orders them correctly. Dates not covered by any range have an unknown
location (NULL in the database).

Consumed by ``ns.db.locations`` (backfill of an existing database) and
``ns.db.update --locations`` (fill for newly scanned images).
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from loguru import logger
import tomli

_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")

_ALLOWED_KEYS = {"start", "end", "location", "note"}


class LocationConfigError(ValueError):
    """Invalid location mapping file."""


@dataclass(frozen=True)
class LocationRange:
    """One deployment: a system stayed at ``location`` from ``start``
    to ``end`` (inclusive); ``end=None`` means ongoing."""

    start: str
    end: Optional[str]
    location: str
    note: Optional[str] = None

    def covers(self, date: str) -> bool:
        return self.start <= date and (self.end is None or date <= self.end)

    def overlaps(self, other: "LocationRange") -> bool:
        self_end = self.end or "9999_99_99"
        other_end = other.end or "9999_99_99"
        return self.start <= other_end and other.start <= self_end


@dataclass
class LocationMap:
    """Parsed mapping: system name -> ranges in file order."""

    ranges: Dict[str, List[LocationRange]] = field(default_factory=dict)

    @property
    def systems(self) -> List[str]:
        return sorted(self.ranges)

    def lookup(self, system: str, date: str) -> Optional[str]:
        """
        Location of ``system`` on ``date`` (``YYYY_MM_DD``), or None
        when no range covers it. The first matching range wins.
        """
        for entry in self.ranges.get(system, ()):
            if entry.covers(date):
                return entry.location
        return None


def load_locations(path: Path) -> LocationMap:
    """
    Parse and validate a location mapping TOML file.

    Raises
    ------
    LocationConfigError
        On malformed structure, bad dates, empty locations, or
        overlapping ranges that disagree on the location. Overlapping
        ranges with the SAME location only produce a warning.
    """
    try:
        with open(path, "rb") as f:
            data = tomli.load(f)
    except OSError as error:
        raise LocationConfigError(f"cannot read {path}: {error}")
    except tomli.TOMLDecodeError as error:
        raise LocationConfigError(f"invalid TOML in {path}: {error}")

    ranges: Dict[str, List[LocationRange]] = {}
    for system, entries in data.items():
        if not isinstance(entries, list):
            raise LocationConfigError(
                f"{system}: expected an array of tables "
                f"([[{system}]] entries), got {type(entries).__name__}"
            )
        parsed: List[LocationRange] = []
        for index, entry in enumerate(entries, start=1):
            parsed.append(_parse_range(system, index, entry))
        _check_overlaps(system, parsed)
        ranges[system] = parsed
    return LocationMap(ranges=ranges)


def _parse_range(system: str, index: int, entry: Any) -> LocationRange:
    where = f"{system} entry #{index}"
    if not isinstance(entry, dict):
        raise LocationConfigError(f"{where}: expected a table")
    unknown = set(entry) - _ALLOWED_KEYS
    if unknown:
        raise LocationConfigError(
            f"{where}: unknown key(s) {sorted(unknown)} "
            f"(allowed: {sorted(_ALLOWED_KEYS)})"
        )
    for key in ("start", "location"):
        if key not in entry:
            raise LocationConfigError(f"{where}: missing required key {key!r}")

    start = str(entry["start"]).strip()
    if not _DATE_RE.match(start):
        raise LocationConfigError(f"{where}: start must be YYYY_MM_DD, got {start!r}")
    end: Optional[str] = None
    if entry.get("end") is not None:
        end = str(entry["end"]).strip()
        if not _DATE_RE.match(end):
            raise LocationConfigError(f"{where}: end must be YYYY_MM_DD, got {end!r}")
        if start > end:
            raise LocationConfigError(f"{where}: start {start} is after end {end}")

    location = str(entry["location"]).strip().lower()
    if not location:
        raise LocationConfigError(f"{where}: location must not be empty")

    note = entry.get("note")
    return LocationRange(
        start=start, end=end, location=location, note=str(note) if note else None
    )


def _check_overlaps(system: str, ranges: List[LocationRange]) -> None:
    for i, a in enumerate(ranges):
        for b in ranges[i + 1 :]:
            if not a.overlaps(b):
                continue
            if a.location != b.location:
                raise LocationConfigError(
                    f"{system}: ranges {a.start}..{a.end or 'ongoing'} "
                    f"({a.location}) and {b.start}..{b.end or 'ongoing'} "
                    f"({b.location}) overlap with different locations"
                )
            logger.warning(
                f"{system}: overlapping ranges with the same location "
                f"({a.start}..{a.end or 'ongoing'} and "
                f"{b.start}..{b.end or 'ongoing'}, both {a.location})"
            )
