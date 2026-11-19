import json

import numpy as np
import pytest

from geoevidence.ingest import chunk_document, read_geojson
from geoevidence.store import build_index, load_index


def test_overlapping_chunks_keep_exact_offsets(document):
    document = document.model_copy(
        update={"text": "  " + " \n".join(f"word{i}" for i in range(51)) + " "}
    )
    chunks = chunk_document(document, words=20, overlap=5)
    assert [len(chunk.text.split()) for chunk in chunks] == [20, 20, 20, 6]
    assert all(document.text[c.start : c.end] == c.text for c in chunks)
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]


def test_chunker_prefers_a_complete_sentence(document):
    document = document.model_copy(
        update={
            "text": "One two three four five six seven eight nine ten. " + " ".join(["word"] * 20)
        }
    )
    chunks = chunk_document(document, words=16, overlap=3)
    assert chunks[0].text.endswith("ten.")
    assert chunks[1].text.split()[:3] == ["eight", "nine", "ten."]


@pytest.mark.parametrize(
    "change", ["duplicate", "polygon", "nan", "boolean", "crs", "location", "missing-source"]
)
def test_bad_corpus_rejected(corpus, change):
    raw = json.loads(corpus.read_text())
    if change == "duplicate":
        raw["features"].append(raw["features"][0])
    elif change == "polygon":
        raw["features"][0]["geometry"]["type"] = "Polygon"
    elif change == "nan":
        raw["features"][0]["geometry"]["coordinates"][0] = float("nan")
    elif change == "boolean":
        raw["features"][0]["geometry"]["coordinates"][0] = True
    elif change == "crs":
        raw["crs"] = {"type": "name", "properties": {"name": "EPSG:3857"}}
    elif change == "location":
        raw["features"][0]["properties"]["location"] = {"lon": 0, "lat": 0}
    else:
        del raw["features"][0]["properties"]["source"]
    corpus.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        read_geojson(corpus)


def test_rebuild_identity_and_source_change(corpus, tmp_path):
    target = tmp_path / "index.sqlite"
    first = build_index(corpus, target)
    assert build_index(corpus, target) == first
    raw = json.loads(corpus.read_text())
    raw["features"][0]["properties"]["text"] += " Reinspect next week."
    corpus.write_text(json.dumps(raw))
    second = build_index(corpus, target)
    assert first["fingerprint"] != second["fingerprint"]
    assert load_index(target).metadata == second


def test_failed_rebuild_keeps_previous_snapshot(corpus, tmp_path):
    target = tmp_path / "index.sqlite"
    original = build_index(corpus, target)

    class BrokenEncoder:
        fingerprint = "broken"

        def encode(self, texts):
            return np.full((len(texts), 3), np.nan)

    with pytest.raises(ValueError, match="non-finite"):
        build_index(corpus, target, BrokenEncoder())
    assert load_index(target).metadata == original


def test_loading_missing_index_does_not_create_database(tmp_path):
    path = tmp_path / "absent.sqlite"
    with pytest.raises(FileNotFoundError, match="Run geoevidence index"):
        load_index(path)
    assert not path.exists()
