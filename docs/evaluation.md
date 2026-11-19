# Executed evaluation

This report records local execution on **19 November 2026**, using Windows, Python 3.12.16 and CPU inference. It is a small engineering experiment, not a claim of production readiness or unseen-data performance.

## Setup

- 20 fictional documents, producing 21 overlapping chunks with an 80-word maximum and 15-word overlap; late sentence boundaries are preferred.
- 20 authored query cases: 14 answerable and six negative. Each uses `top_k=3`.
- MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized 384-dimensional embeddings.
- FLAN-T5-small revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, greedy decoding, at most 120 new tokens.
- Torch 2.15.0+cpu, Transformers 4.57.6, Sentence Transformers 5.7.0, NumPy 2.5.5 and PyProj 3.8.0. Full resolved dependencies are in `uv.lock`.
- Dense minimum similarity 0.3; reciprocal rank fusion constant 60. No threshold sweep or tuning on the fixture was performed.

The corpus and queries were authored together. Many positive cases have strong geographic/topic constraints. The scores below therefore test these known scenarios and should not be presented as independent real-world accuracy.

## Retrieval comparison

| Method | Recall@3 | MRR | nDCG@3 | Correct abstention | Retrieval p50 | Retrieval p95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 lexical | 1.000 | 1.000 | 1.000 | 5/6 | 0.077 ms | 0.116 ms |
| MiniLM dense | 0.929 | 0.929 | 0.929 | 5/6 | 7.111 ms | 8.601 ms |
| Hybrid RRF | 1.000 | 1.000 | 1.000 | 5/6 | 7.240 ms | 7.800 ms |

Recall, MRR and nDCG are macro averages over the 14 answerable cases. Abstention is measured separately on the six negative cases. Recall counts distinct documents, so overlapping chunks cannot inflate the score. nDCG uses binary relevance and an ideal ranking capped at the requested `k`.

The timer includes filtering, ranking and query encoding where applicable. It excludes index/model loading, downloading and generation. Queries ran sequentially in one process with at most four Torch threads. The first query can be cold; these figures are not a controlled cold-versus-warm throughput benchmark. CPU time and peak process-family memory were not measured. Corpus size is too small for scaling claims.

**Observed failure — dense retrieval:** `damaged-insulation` retrieved nothing at the fixed threshold. The question asks which component needs replacement under isolation; the relevant brief describes cracked insulators. Keyword retrieval recovered the brief, and hybrid retrieval retained it. This supports keeping a lexical path for these domain terms; it does not establish that hybrid retrieval universally outperforms BM25. In this fixture, BM25 and hybrid tied on the positive cases, with BM25 substantially cheaper.

**Observed failure — answerability:** every method returned the mainland drainage brief for `unsupported-budget`. The brief is related to drainage but contains no exact approved budget. All four explicit empty-scope negatives correctly returned no results, and the unrelated physics query also returned no results. Semantic or keyword relevance is insufficient to infer answerability.

Raw rankings, metrics, corpus hash, model identity, dependency versions and execution timestamp are in [retrieval.json](../reports/retrieval.json).

## Actual local generation

The neural smoke script loaded the real pinned models from the local cache with `HF_HUB_OFFLINE=1`; it did not call a hosted language-model API.

| Question | Observed answer | Interpretation |
|---|---|---|
| What should crews remove before the wet-season exercise? | “plastic waste and accumulated sediment” | Matches the supplied brief. One manually inspected example, not an accuracy estimate. |
| What is the exact approved drainage budget in naira? | “No pump purchase” | Inadequate answer. The source supplies no exact budget, and the model failed to explicitly abstain. |
| Drainage maintenance in a 5 km radius around `[0, 0]` | No matching evidence | No eligible document; generation skipped. |

The first generated response took 254.121 ms after models were loaded; the second took 118.178 ms. The empty-scope response took 0.071 ms. These are individual observations, not a throughput or end-to-end startup claim. All generated prose remains labeled `unverified_generation`.

The exact supplied passages, offsets, filters' effect, locations and index fingerprint are retained in [neural-smoke.json](../reports/neural-smoke.json). Evidence labels refer to supplied context, not verified sentence-level citations. No claim-level entailment score or LLM-judge score was computed.

## Software verification

- Local pytest suite: **48 passed**, covering geography, ingestion, atomic rebuilds, retrieval, generation-budget contracts, the API, evaluation arithmetic and the Streamlit demo.
- Real MiniLM indexing and the complete three-method retrieval comparison executed successfully.
- Real FLAN-T5 execution tested the supported question, unsupported question and empty-scope path.
- Browser inspection confirmed the rendered map, a successful scoped query and the expanded source provenance.
- Ruff lint and format checks passed. Source distribution and wheel built successfully.
- FastAPI/Starlette emitted one dependency deprecation warning about its httpx-based test client; the API tests passed.

CI runs deterministic tests and the map smoke test on Windows and Linux without neural weights. The public workflow status is authoritative for those executions; local testing is not a substitute for observing a successful remote run.

## Reproduce

```bash
uv sync --locked --extra neural --extra demo
uv run geoevidence index data/demo.geojson --encoder minilm --output artifacts/neural.sqlite
uv run geoevidence evaluate data/evaluation.json --index artifacts/neural.sqlite --modes lexical dense hybrid --output artifacts/retrieval.json
uv run python scripts/run_neural_smoke.py
uv run pytest
```

The smoke script writes `reports/neural-smoke.json` with current outputs and a timestamp. Keep published historical reports when comparing a modified system. Once dependencies and weights are cached, `HF_HUB_OFFLINE=1` prevents model network access during inference. Expect timing differences between machines and runs.

## Next experiments

Evaluate on separately collected, access-approved real documents; label answerability independently of relevance; compare a stronger local generator; measure abstention/precision tradeoffs on a separate development set; and evaluate long documents where one passage per source may omit necessary context. These experiments have **not** been completed in this version.
