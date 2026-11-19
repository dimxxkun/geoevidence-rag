import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geoevidence.api import create_app
from geoevidence.engine import Engine
from geoevidence.evaluation import evaluate
from geoevidence.models import Query
from geoevidence.retrieval import Retriever
from geoevidence.store import build_index, load_index


class FixtureEncoder:
    """Semantic contract double, not evidence that the real model performs well."""

    fingerprint = "fixture-v1"

    def encode(self, texts):
        return np.array(
            [
                [1, 0] if "canal" in text.lower() or "sediment" in text.lower() else [0, 1]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_lexical_evidence_and_absence(index):
    engine = Engine(Retriever(index))
    answer = engine.ask(Query(question="remove sediment", mode="lexical"))
    assert answer.status == "answered"
    assert answer.grounding == "verbatim"
    assert answer.evidence[0].document_id == "alpha"
    assert answer.evidence[0].quote in answer.answer
    assert engine.ask(Query(question="quantum photons", mode="lexical")).status == "no_evidence"


def test_empty_area_never_falls_back_to_global(index):
    answer = Engine(Retriever(index)).ask(
        Query(
            question="canal maintenance",
            mode="lexical",
            filters={"radius": {"center": {"lon": 0, "lat": 0}, "km": 1}},
        )
    )
    assert answer.evidence == []
    assert answer.eligible_documents == 0
    assert answer.status == "no_evidence"


def test_neural_mode_requires_neural_index(index):
    with pytest.raises(ValueError, match="neural index"):
        Retriever(index).search(Query(question="canal maintenance"))


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_neural_modes_use_only_filtered_candidates(corpus, tmp_path, mode):
    target = tmp_path / "dense.sqlite"
    build_index(corpus, target, FixtureEncoder())
    retriever = Retriever(load_index(target), FixtureEncoder())
    found, eligible = retriever.search(Query(question="canal sediment", mode=mode))
    assert found[0].document_id == "alpha"
    assert eligible == 2
    found, eligible = retriever.search(
        Query(question="canal sediment", mode=mode, filters={"topic": "electricity"})
    )
    assert found == []
    assert eligible == 1


def test_encoder_mismatch_is_not_silently_accepted(index):
    with pytest.raises(ValueError, match="does not match"):
        Retriever(index, FixtureEncoder())


def test_chunk_overlap_does_not_duplicate_documents(corpus, tmp_path):
    raw = json.loads(corpus.read_text())
    raw["features"][0]["properties"]["text"] = " ".join(["canal sediment"] * 50)
    corpus.write_text(json.dumps(raw))
    target = tmp_path / "chunks.sqlite"
    build_index(corpus, target, words=20, overlap=5)
    found, _ = Retriever(load_index(target)).search(
        Query(question="canal sediment", mode="lexical", top_k=10)
    )
    assert [item.document_id for item in found] == ["alpha"]


def test_no_generation_on_empty_result(index):
    class ShouldNotRun:
        name = "spy"

        def generate(self, *args):
            raise AssertionError("No-evidence requests must not invoke a model")

    engine = Engine(Retriever(index), ShouldNotRun())
    assert engine.ask(Query(question="quantum photons", mode="lexical")).status == "no_evidence"


def test_api_validates_and_returns_citations(index):
    with TestClient(create_app(Engine(Retriever(index)))) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/ask", json={"question": "canal sediment", "mode": "lexical"})
        assert response.status_code == 200
        assert response.json()["evidence"][0]["citation"] == "E1"
        assert client.post("/ask", json={"question": "x"}).status_code == 422
        assert (
            client.post("/ask", json={"question": "canal sediment", "top_k": 100}).status_code
            == 422
        )
        assert (
            client.post("/ask", json={"question": "canal sediment", "mode": "dense"}).status_code
            == 422
        )


def test_metrics_count_misses_and_abstentions_separately(index, tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {"id": "hit", "query": {"question": "canal sediment"}, "relevant": ["alpha"]},
                {"id": "miss", "query": {"question": "quantum photons"}, "relevant": ["alpha"]},
                {"id": "negative", "query": {"question": "quantum photons"}, "relevant": []},
            ]
        )
    )
    result = evaluate(Retriever(index), cases, ["lexical"])["summary"]["lexical"]
    assert result["mrr"] == 0.5
    assert result["recall_at_k"] == 0.5
    assert result["abstention_accuracy"] == 1


def test_invalid_gold_is_not_scored(index, tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [{"id": "bad", "query": {"question": "canal sediment"}, "relevant": ["missing"]}]
        )
    )
    with pytest.raises(ValueError, match="Unknown gold"):
        evaluate(Retriever(index), cases, ["lexical"])
