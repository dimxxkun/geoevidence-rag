import json

import pytest

from geoevidence.models import Document
from geoevidence.store import build_index, load_index


@pytest.fixture
def document():
    return Document(
        id="alpha",
        title="Canal maintenance",
        text="Remove sediment from the blocked canal inlet.",
        source="fixture://alpha",
        published="2026-06-12",
        topic="flood",
        location={"lon": 3.3792, "lat": 6.5244},
    )


@pytest.fixture
def corpus(tmp_path, document):
    second = document.model_copy(
        update={
            "id": "beta",
            "title": "Electricity maintenance",
            "text": "Replace cracked feeder insulators.",
            "topic": "electricity",
        }
    )
    features = []
    for doc in (document, second):
        props = doc.model_dump(mode="json", exclude={"location"})
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [doc.location.lon, doc.location.lat]},
                "properties": props,
            }
        )
    path = tmp_path / "corpus.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


@pytest.fixture
def index(tmp_path, corpus):
    path = tmp_path / "index.sqlite"
    build_index(corpus, path)
    return load_index(path)
