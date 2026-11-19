"""Run actual pinned CPU models and retain answer evidence, including known limitations."""

import json
from datetime import UTC, datetime
from pathlib import Path

from geoevidence.engine import Engine
from geoevidence.models import Query


def main():
    engine = Engine.open(Path("artifacts/neural.sqlite"), generator="flan")
    filters = json.loads(Path("data/lagos-filter.json").read_text())
    requests = [
        Query(question="What should crews remove before the wet-season exercise?", filters=filters),
        Query(question="What is the exact approved drainage budget in naira?", filters=filters),
        Query(
            question="What drainage maintenance is recommended?",
            filters={"radius": {"center": {"lon": 0, "lat": 0}, "km": 5}},
        ),
    ]
    answers = [engine.ask(query).model_dump(mode="json") for query in requests]
    assert answers[0]["evidence"][0]["document_id"] == "lag-drain"
    assert answers[0]["grounding"] == "unverified_generation"
    assert answers[2]["status"] == "no_evidence" and not answers[2]["evidence"]
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "scope": (
            "Three actual local-model runs. Assertions verify workflow and context, "
            "not generated factual accuracy."
        ),
        "answers": answers,
    }
    target = Path("reports/neural-smoke.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for answer in answers:
        print(
            json.dumps(
                {
                    "question": answer["question"],
                    "answer": answer["answer"],
                    "status": answer["status"],
                    "elapsed_ms": answer["elapsed_ms"],
                }
            )
        )


if __name__ == "__main__":
    main()
