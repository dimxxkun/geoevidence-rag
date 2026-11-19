"""Document-level retrieval evaluation; no LLM judge or invented quality score."""

import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np

from .models import Query
from .retrieval import Retriever


def evaluate(retriever: Retriever, cases_path: Path, modes: list[str]) -> dict:
    raw = cases_path.read_bytes()
    cases = json.loads(raw)
    rows = []
    available = {chunk.document.id for chunk in retriever.index.chunks}
    seen = set()
    for case in cases:
        if case["id"] in seen:
            raise ValueError("Duplicate evaluation case id")
        seen.add(case["id"])
        if not set(case["relevant"]) <= available:
            raise ValueError(f"Unknown gold document in {case['id']}")
        for mode in modes:
            query = Query(**case["query"], mode=mode)
            started = perf_counter()
            evidence, eligible = retriever.search(query)
            elapsed = (perf_counter() - started) * 1000
            gold = set(case["relevant"])
            found = [item.document_id for item in evidence]
            relevant_ranks = [rank for rank, doc in enumerate(found, 1) if doc in gold]
            dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
            ideal = sum(
                1 / math.log2(rank + 1) for rank in range(1, min(len(gold), query.top_k) + 1)
            )
            rows.append(
                {
                    "id": case["id"],
                    "mode": mode,
                    "gold": sorted(gold),
                    "retrieved": found,
                    "recall_at_k": len(gold.intersection(found)) / len(gold) if gold else None,
                    "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else 0,
                    "ndcg_at_k": dcg / ideal if ideal else None,
                    "correct_abstention": not found if not gold else None,
                    "eligible_documents": eligible,
                    "latency_ms": round(elapsed, 3),
                }
            )
    summary = {}
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        answerable = [row for row in subset if row["gold"]]
        negatives = [row for row in subset if not row["gold"]]
        summary[mode] = {
            "answerable_cases": len(answerable),
            "negative_cases": len(negatives),
            "recall_at_k": float(np.mean([r["recall_at_k"] for r in answerable]))
            if answerable
            else None,
            "mrr": float(np.mean([r["reciprocal_rank"] for r in answerable]))
            if answerable
            else None,
            "ndcg_at_k": float(np.mean([r["ndcg_at_k"] for r in answerable]))
            if answerable
            else None,
            "abstention_accuracy": float(np.mean([r["correct_abstention"] for r in negatives]))
            if negatives
            else None,
            "retrieval_p50_ms": float(np.median([r["latency_ms"] for r in subset])),
            "retrieval_p95_ms": float(np.percentile([r["latency_ms"] for r in subset], 95)),
        }
    packages = {}
    for package in ("numpy", "pyproj", "sentence-transformers", "transformers", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return {
        "executed_at": datetime.now(UTC).isoformat(),
        "scope": (
            "Authored synthetic retrieval cases; not a held-out real-world benchmark "
            "or generation evaluation."
        ),
        "index": retriever.index.metadata,
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "timing": (
            "Sequential CPU retrieval only; excludes index/model loading and answer generation. "
            "First query may be cold."
        ),
        "summary": summary,
        "cases": rows,
    }
