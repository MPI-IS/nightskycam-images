"""
Tests for the website FilterSpec (canonical filter contract).
"""

from dataclasses import fields

import pytest
from werkzeug.datastructures import MultiDict

from nightskycam_images.website.spec import (
    PAGE_SIZE_DEFAULT,
    PAGE_SIZE_MAX,
    PAGE_SIZE_MIN,
    FilterSpec,
    FilterSpecError,
)

# ============================================================================
# from_query_args
# ============================================================================


def test_from_query_args_empty():
    spec = FilterSpec.from_query_args(MultiDict())
    assert spec == FilterSpec()
    assert spec.page == 1
    assert spec.page_size == PAGE_SIZE_DEFAULT
    assert spec.descending is True


def test_from_query_args_repeated_and_dotted_params():
    args = MultiDict(
        [
            ("systems", "cam1"),
            ("systems", "cam2"),
            ("weather", "clear"),
            ("classifier_max.cloudy", "0.3"),
            ("classifier_max.rainy", "0.9"),
            ("cloud_cover_max", "40"),
            ("page", "3"),
        ]
    )
    spec = FilterSpec.from_query_args(args)
    assert spec.systems == ["cam1", "cam2"]
    assert spec.weather == ["clear"]
    assert spec.classifier_max == {"cloudy": 0.3, "rainy": 0.9}
    assert spec.cloud_cover_max == 40
    assert spec.page == 3


def test_from_query_args_blank_values_ignored():
    args = MultiDict(
        [("start_date", ""), ("cloud_cover_min", ""), ("image_format", "")]
    )
    spec = FilterSpec.from_query_args(args)
    assert spec == FilterSpec()


def test_from_query_args_html_date_and_time_formats():
    args = MultiDict(
        [
            ("start_date", "2025-06-01"),
            ("end_date", "2025_06_02"),
            ("start_time", "22:00"),
            ("end_time", "04:00:30"),
        ]
    )
    spec = FilterSpec.from_query_args(args)
    assert spec.start_date == "2025_06_01"
    assert spec.end_date == "2025_06_02"
    assert spec.start_time == "22_00_00"
    assert spec.end_time == "04_00_30"


@pytest.mark.parametrize(
    "args",
    [
        {"start_date": "junk"},
        {"start_time": "25h"},
        {"cloud_cover_min": "abc"},
        {"cloud_cover_max": "101"},
        {"image_format": "png"},
        {"stretched": "maybe"},
        {"page": "0"},
        {"classifier_max.cloudy": "high"},
        {"classifier_max.cloudy": "1.5"},
    ],
)
def test_from_query_args_invalid(args):
    with pytest.raises(FilterSpecError):
        FilterSpec.from_query_args(MultiDict(args))


def test_locations_parsed_and_normalized():
    args = MultiDict([("locations", "Palma"), ("locations", " tuebingen ")])
    spec = FilterSpec.from_query_args(args)
    assert spec.locations == ["palma", "tuebingen"]
    assert "locations=palma" in spec.to_query_string()

    from_json = FilterSpec.from_json({"locations": ["IBACH"]})
    assert from_json.locations == ["ibach"]
    assert from_json.to_query_kwargs() == {"locations": ["ibach"]}


def test_locations_from_json_invalid():
    with pytest.raises(FilterSpecError):
        FilterSpec.from_json({"locations": "palma"})  # not a list


def test_page_size_clamped():
    assert FilterSpec(page_size=1).page_size == PAGE_SIZE_MIN
    assert FilterSpec(page_size=10_000).page_size == PAGE_SIZE_MAX


# ============================================================================
# from_json
# ============================================================================


def test_from_json_happy_path():
    spec = FilterSpec.from_json(
        {
            "systems": ["cam1"],
            "start_date": "2025_06_01",
            "cloud_cover_max": 30,
            "stretched": False,
            "classifier_max": {"cloudy": 0.2},
            "descending": False,
        }
    )
    assert spec.systems == ["cam1"]
    assert spec.cloud_cover_max == 30
    assert spec.stretched is False
    assert spec.classifier_max == {"cloudy": 0.2}
    assert spec.descending is False


def test_from_json_rejects_unknown_fields():
    with pytest.raises(FilterSpecError, match="unknown"):
        FilterSpec.from_json({"sysstems": ["cam1"]})


@pytest.mark.parametrize(
    "body",
    [
        {"systems": "cam1"},  # not a list
        {"cloud_cover_max": "30"},  # not an int
        {"cloud_cover_max": True},  # bool is not an int here
        {"stretched": "yes"},  # not a bool
        {"classifier_max": {"cloudy": "low"}},  # not a number
        {"page": 0},
    ],
)
def test_from_json_invalid(body):
    with pytest.raises(FilterSpecError):
        FilterSpec.from_json(body)


def test_json_round_trip():
    spec = FilterSpec(
        systems=["cam1"],
        start_date="2025_06_01",
        weather=["clear"],
        classifier_max={"cloudy": 0.3},
        descending=False,
        page=2,
    )
    assert FilterSpec.from_json(spec.to_json()) == spec


# ============================================================================
# to_query_kwargs / to_page_kwargs
# ============================================================================


def test_to_query_kwargs_only_set_fields():
    spec = FilterSpec(systems=["cam1"], cloud_cover_max=40)
    assert spec.to_query_kwargs() == {
        "systems": ["cam1"],
        "cloud_cover_max": 40,
    }


def test_to_page_kwargs():
    spec = FilterSpec(page=3, page_size=50, descending=False)
    assert spec.to_page_kwargs() == {
        "order_by": "datetime",
        "descending": False,
        "limit": 50,
        "offset": 100,
    }


# ============================================================================
# to_query_string round trip
# ============================================================================


def test_query_string_round_trip():
    spec = FilterSpec(
        systems=["cam1", "cam2"],
        start_date="2025_06_01",
        start_time="22_00_00",
        end_time="04_00_00",
        weather=["clear", "overcast"],
        cloud_cover_max=40,
        image_format="tiff",
        stretched=False,
        classifier_max={"cloudy": 0.3},
        descending=False,
        page=4,
        page_size=24,
    )
    qs = spec.to_query_string()
    parsed = FilterSpec.from_query_args(
        MultiDict([pair.split("=", 1) for pair in qs.replace("%3A", ":").split("&")])
    )
    # urlencode escaping is exercised through Flask in the app tests; here
    # the values contain no reserved characters except the dotted keys.
    assert parsed.to_json() == spec.to_json()


def test_query_string_page_override():
    spec = FilterSpec(systems=["cam1"], page=5)
    assert "page=2" in spec.to_query_string(page=2)
    assert "page=5" not in spec.to_query_string(page=2)
    assert "page" not in spec.to_query_string(page=1)


# ============================================================================
# json_schema
# ============================================================================


def test_json_schema_covers_all_fields():
    schema = FilterSpec.json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    schema_fields = set(schema["properties"].keys())
    dataclass_fields = {f.name for f in fields(FilterSpec)}
    assert schema_fields == dataclass_fields
    for prop in schema["properties"].values():
        assert "description" in prop
