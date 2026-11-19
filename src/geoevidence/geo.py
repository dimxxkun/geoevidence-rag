"""Inclusive anchor-point filtering with WGS84 ellipsoidal distances."""

from pyproj import Geod

from .models import Document, Filters

_GEOD = Geod(ellps="WGS84")


def match(document: Document, filters: Filters) -> tuple[bool, float | None]:
    point = document.location
    if filters.topic is not None and document.topic.casefold() != filters.topic.casefold():
        return False, None
    if filters.published_from and document.published < filters.published_from:
        return False, None
    if filters.published_to and document.published > filters.published_to:
        return False, None
    if bounds := filters.bbox:
        # +180 and -180 identify the same meridian; accept either representation.
        longitudes = [point.lon] if abs(point.lon) != 180 else [-180.0, 180.0]
        longitude_matches = any(
            bounds.west <= lon <= bounds.east
            if bounds.west <= bounds.east
            else lon >= bounds.west or lon <= bounds.east
            for lon in longitudes
        )
        if not longitude_matches or not bounds.south <= point.lat <= bounds.north:
            return False, None
    distance = None
    if radius := filters.radius:
        _, _, meters = _GEOD.inv(radius.center.lon, radius.center.lat, point.lon, point.lat)
        distance = meters / 1000
        if distance > radius.km + 1e-9:
            return False, distance
    return True, distance
