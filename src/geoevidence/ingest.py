"""Read bounded point GeoJSON and retain exact source offsets when chunking."""

import json
import re
from pathlib import Path

from .models import Chunk, Document, Point

MAX_CORPUS_BYTES = 10 * 1024 * 1024
MAX_DOCUMENTS = 5000


def read_geojson(path: Path) -> list[Document]:
    # Bound the read itself, not merely a previous file-size check.
    with path.open("rb") as stream:
        payload = stream.read(MAX_CORPUS_BYTES + 1)
    if len(payload) > MAX_CORPUS_BYTES:
        raise ValueError("Corpus exceeds the 10 MiB local-demo limit")
    raw = json.loads(payload)
    if not isinstance(raw, dict) or raw.get("type") != "FeatureCollection" or "crs" in raw:
        raise ValueError("Expected WGS84 GeoJSON FeatureCollection without a legacy crs member")
    features = raw.get("features")
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_DOCUMENTS:
        raise ValueError("Expected 1 to 5000 features")
    documents = []
    seen = set()
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("Each item must be a GeoJSON Feature")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") != "Point":
            raise ValueError("Document anchors must be Point geometries, not area footprints")
        coordinates = geometry.get("coordinates")
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 2
            or any(type(value) not in (int, float) for value in coordinates)
        ):
            raise ValueError("Point requires two numeric coordinates [longitude, latitude]")
        props = feature.get("properties")
        if not isinstance(props, dict):
            raise ValueError("Feature properties must be an object")
        if "location" in props:
            raise ValueError("Location must come from the Point geometry, not feature properties")
        document = Document(**props, location=Point(lon=coordinates[0], lat=coordinates[1]))
        if document.id in seen:
            raise ValueError(f"Duplicate document id: {document.id}")
        seen.add(document.id)
        documents.append(document)
    return sorted(documents, key=lambda item: item.id)


def chunk_document(document: Document, words: int = 80, overlap: int = 15) -> list[Chunk]:
    if not 16 <= words <= 200 or not 0 <= overlap < words:
        raise ValueError("Chunk words must be 16..200 and overlap must be smaller than words")
    spans = list(re.finditer(r"\S+", document.text))
    chunks = []
    offset = 0
    while offset < len(spans):
        stop = min(offset + words, len(spans))
        if stop < len(spans):
            # Prefer a late sentence boundary, without exceeding the word budget or
            # preventing forward progress when the caller chooses a large overlap.
            endings = [
                index + 1
                for index in range(max(offset + words // 2, offset + overlap), stop)
                if re.search(r"[.!?][\"')]*$", spans[index].group())
            ]
            if endings:
                stop = endings[-1]
        window = spans[offset:stop]
        start, end = window[0].start(), window[-1].end()
        chunks.append(
            Chunk(
                id=f"{document.id}:{len(chunks)}",
                document=document,
                text=document.text[start:end],
                start=start,
                end=end,
            )
        )
        if stop == len(spans):
            break
        offset = stop - overlap
    return chunks
