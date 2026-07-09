"""
The single place where a FilterSpec meets the database.

Both HTML views and JSON API endpoints (and the future natural-language
filter flow) call :func:`run_query`; none of them talk to
:class:`~nightskycam_images.db_api.ImageDB` directly for listings.
"""

from dataclasses import dataclass
import math
from typing import List, Optional

from ..db_api import ImageDB, ImageRecord
from .spec import FilterSpec


@dataclass
class QueryResult:
    total: int
    page: int
    pages: int
    page_size: int
    records: List[ImageRecord]


def run_query(db: ImageDB, spec: FilterSpec) -> QueryResult:
    """Count matches and fetch the records of the requested page."""
    filter_kwargs = spec.to_query_kwargs()
    total = db.count(**filter_kwargs)
    pages = max(1, math.ceil(total / spec.page_size))
    records = db.images(**filter_kwargs, **spec.to_page_kwargs())
    return QueryResult(
        total=total,
        page=spec.page,
        pages=pages,
        page_size=spec.page_size,
        records=records,
    )


def record_at(db: ImageDB, spec: FilterSpec, index: int) -> Optional[ImageRecord]:
    """
    Return the record at an absolute 0-based position within the spec's
    full (unpaginated) result ordering, or None when out of range.

    Used for stateless prev/next navigation on the image detail page.
    """
    if index < 0:
        return None
    records = db.images(
        **spec.to_query_kwargs(),
        order_by=spec.order_by,
        descending=spec.descending,
        limit=1,
        offset=index,
        with_scores=False,
    )
    return records[0] if records else None
