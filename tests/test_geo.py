import pytest
from pydantic import ValidationError

from geoevidence.geo import match
from geoevidence.models import Filters, Point


def located(document, lon, lat):
    return document.model_copy(update={"location": Point(lon=lon, lat=lat)})


@pytest.mark.parametrize(
    "lon,expected", [(179.8, True), (-179.8, True), (180, True), (-180, True), (0, False)]
)
def test_dateline_wrap(document, lon, expected):
    filters = Filters(bbox={"west": 179, "south": -1, "east": -179, "north": 1})
    assert match(located(document, lon, 0), filters)[0] is expected


@pytest.mark.parametrize("lon", [-180, 180])
def test_meridian_equivalence_at_bbox_edge(document, lon):
    filters = Filters(bbox={"west": 179, "south": -1, "east": 180, "north": 1})
    assert match(located(document, lon, 0), filters)[0]


def test_inclusive_bounds_and_dates(document):
    filters = Filters(
        bbox={"west": 3.3792, "south": 6.5244, "east": 4, "north": 7},
        published_from="2026-06-12",
        published_to="2026-06-12",
    )
    assert match(document, filters)[0]


def test_ellipsoid_distance_and_radius(document):
    # Published WGS84 equatorial one-degree arc, about 111.31949 km, not planar degrees.
    doc = located(document, 1, 0)
    accepted, distance = match(doc, Filters(radius={"center": {"lon": 0, "lat": 0}, "km": 112}))
    assert accepted
    assert distance == pytest.approx(111.3194908, abs=1e-6)
    assert not match(doc, Filters(radius={"center": {"lon": 0, "lat": 0}, "km": 111}))[0]


def test_dateline_radius(document):
    assert match(
        located(document, -179.9, 0), Filters(radius={"center": {"lon": 179.9, "lat": 0}, "km": 25})
    )[0]


def test_filters_are_intersected(document):
    assert not match(
        document, Filters(topic="water", radius={"center": {"lon": 3.3792, "lat": 6.5244}, "km": 2})
    )[0]


@pytest.mark.parametrize(
    "payload",
    [
        {"bbox": {"west": 0, "east": 1, "south": 4, "north": 2}},
        {"radius": {"center": {"lon": 181, "lat": 0}, "km": 1}},
        {"radius": {"center": {"lon": 0, "lat": 91}, "km": 1}},
        {"radius": {"center": {"lon": 0, "lat": 0}, "km": 0}},
        {"radius": {"center": {"lon": float("nan"), "lat": 0}, "km": 1}},
        {"published_from": "2026-06-01", "published_to": "2025-01-01"},
    ],
)
def test_invalid_filters_rejected(payload):
    with pytest.raises(ValidationError):
        Filters(**payload)
