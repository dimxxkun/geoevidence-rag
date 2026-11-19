# GeoEvidence RAG

**Ask a place. Inspect the evidence.**

[![checks](https://github.com/dimxxkun/geoevidence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/dimxxkun/geoevidence-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11–3.13-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-14B8A6)](LICENSE)

A local geospatial retrieval-augmented generation project for asking questions about geotagged planning briefs. It combines **BM25, MiniLM embeddings, geographic constraints, and FLAN-T5 generation**, with a map workbench, a typed HTTP API, and a reproducible evaluation.

The central problem: a semantically relevant drainage report from another city is still the wrong evidence. GeoEvidence applies location, publication-date, and topic constraints **before** retrieval and generation. It keeps the exact source passage with every answer.

> The included corpus contains fictional training scenarios at approximate real-world coordinates. It is not an operational dataset. This is a RAG engineering project built around pretrained models, not a newly trained foundation model.

## A concrete example

**Question:** “What drainage maintenance is recommended?”

**Constraints:** within 8 km of `[3.3792, 6.5244]`, topic `flood`, published from January 2026.

The eligible evidence is the current mainland brief recommending inlet clearance. A similar Abuja culvert-replacement brief is outside the radius, and an older mainland pump proposal is outside the date range. Neither enters the language model's context.

```text
GeoJSON documents
       │  validate coordinates + provenance; chunk with exact offsets
       ▼
SQLite snapshot ─── model revision + corpus hash + chunk configuration
       │
       ▼
Hard filters: bounding box ∩ geodesic radius ∩ topic ∩ publication dates
       │
       ├── BM25 keyword ranking
       └── MiniLM cosine ranking
                    │
           Reciprocal rank fusion
                    │  one passage per document
                    ▼
        Verbatim evidence / local FLAN-T5 answer
                    │
          CLI · FastAPI · map workbench
```

## What is implemented

- **Geographic retrieval:** inclusive WGS84 bounding boxes, dateline wrapping, ellipsoidal radius distances in kilometres, and intersection of all supplied constraints.
- **Hybrid search:** BM25 and normalized 384-dimensional MiniLM embeddings combined using reciprocal rank fusion. Each score component is returned for inspection.
- **Traceable ingestion:** point GeoJSON validation, overlapping word chunks, exact character offsets, duplicate-ID rejection, and atomic SQLite replacement.
- **Reproducible snapshots:** corpus hashes, pinned neural-model revisions, chunk settings, and encoder identity. A mismatched encoder is rejected.
- **Two answer paths:** a lightweight verbatim baseline and CPU-local FLAN-T5 generation. Empty retrieval skips generation; context packing never silently includes evidence that the model did not receive.
- **Interfaces:** a command line, a FastAPI service with an OpenAPI schema, and a Streamlit/PyDeck workbench with map locations and downloadable evidence.
- **Evaluation:** document-level Recall@k, MRR, nDCG@k, abstention accuracy, and retrieval latency, with per-case output and explicit failed cases.
- **Tests:** geographic edge cases, invalid input, snapshot preservation, source offsets, ranker contracts, API behavior, evaluation arithmetic, and an interactive-demo smoke test.

## Quick start — no model download

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.11–3.13. The lockfile captures the dependency resolution; CI uses Python 3.12 on Windows and 3.11/3.13 on Linux.

```bash
git clone https://github.com/dimxxkun/geoevidence-rag.git
cd geoevidence-rag
uv sync --locked
uv run geoevidence index data/demo.geojson
uv run geoevidence ask "What drainage maintenance is recommended?" --filters data/lagos-filter.json
uv run geoevidence evaluate data/evaluation.json
```

This path uses keyword retrieval and quoted passages. It is an **extractive baseline**, not neural generation. Outputs include passage IDs, sources, publication dates, coordinates, offsets, distances, and the index fingerprint.

## Full neural RAG

```bash
uv sync --locked --extra neural --extra demo
uv run geoevidence index data/demo.geojson --encoder minilm --output artifacts/neural.sqlite
uv run geoevidence ask "What drainage maintenance is recommended?" --index artifacts/neural.sqlite --mode hybrid --generator flan --filters data/lagos-filter.json
uv run geoevidence evaluate data/evaluation.json --index artifacts/neural.sqlite --modes lexical dense hybrid --output artifacts/evaluation.json
```

The first neural run downloads the pinned public MiniLM and FLAN-T5-small weights into the Hugging Face cache. The model libraries and weights are larger than the core install. Computation uses the CPU, at most four Torch threads, and no paid API. Once dependencies and weights are cached, CLI/API inference can run offline. Weights and generated indexes are not committed.

### Open the map workbench

```bash
uv sync --locked --extra demo
uv run streamlit run app.py
```

By default the workbench reads `artifacts/demo.sqlite`. To use the neural snapshot, set `GEOEVIDENCE_INDEX=artifacts/neural.sqlite` before launching it:

```powershell
# PowerShell
$env:GEOEVIDENCE_INDEX = "artifacts/neural.sqlite"
uv run streamlit run app.py
```

```bash
# macOS / Linux
GEOEVIDENCE_INDEX=artifacts/neural.sqlite uv run streamlit run app.py
```

The map basemap requires online tiles; retrieval does not. Submitted filters are retained with each displayed answer, even when you subsequently move the map or change form controls.

### Serve the API

```bash
uv run uvicorn geoevidence.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive schema. The API reads the same `GEOEVIDENCE_INDEX` variable; `GEOEVIDENCE_GENERATOR=flan` enables local generation. The default is extractive. A lexical-only index must be queried with `"mode": "lexical"`.

```bash
curl http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"What drainage maintenance is recommended?","mode":"lexical","filters":{"radius":{"center":{"lon":3.3792,"lat":6.5244},"km":8},"topic":"flood","published_from":"2026-01-01"}}'
```

The service is intended for localhost. It has no authentication, tenant separation, or production request queue. Requests share a model lock to prevent simultaneous local CPU inference. Ingestion is deliberately a CLI operation.

## Bring your own geotagged documents

Use a WGS84 `FeatureCollection` containing **Point** anchors and the following properties:

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [3.3792, 6.5244]},
    "properties": {
      "id": "brief-001",
      "title": "Drainage inspection",
      "text": "The inspection recommends clearing the canal inlet.",
      "source": "https://your-organisation.example/brief-001",
      "published": "2026-06-12",
      "topic": "flood"
    }
  }]
}
```

Then run `geoevidence index your-documents.geojson`. This project indexes document anchors, **not polygon footprints**. Coordinates are always longitude then latitude. Bounds use named `west`, `south`, `east`, `north` fields; `west > east` denotes a dateline-crossing box. Dates are inclusive publication filters, not a guarantee that the newest document supersedes all earlier ones.

Local-demo limits are 10 MiB per input corpus, 5,000 documents, 25,000 chunks, 2,000 characters per question, and ten returned passages. Search scans the eligible vectors in memory; this is intentionally not a distributed vector database. See [architecture and tradeoffs](docs/architecture.md).

## Evidence and generation limits

`grounding: "verbatim"` means the displayed text is quoted from retrieved passages. It does **not** mean that the underlying source is true or sufficient to answer the question.

`grounding: "unverified_generation"` means FLAN-T5 produced a summary. The evidence list identifies the exact context supplied to the model; it is not claim-level citation verification. A small model can omit qualifications, misread a question, or invent a fact. Prompt instructions are not a complete defense against instructions embedded in documents. The generator has no tools or write access.

`status: "answered"` records that an answer path ran; it is not a factual-correctness label. `no_evidence` means no result survived retrieval and its filters. Retrieval overlap alone cannot establish answerability: the evaluation includes an unsupported budget question whose keywords still match a drainage brief.

No fine-tuning, domain calibration, multilingual evaluation, large-corpus scaling, GPU evaluation, polygon containment, automatic place-name geocoding, or real operational validation is claimed.

## Validation and results

See [the measured experiment report](docs/evaluation.md), [dataset card](data/README.md), and [raw retrieval results](reports/retrieval.json). The benchmark is a deliberately small, authored fixture with known distractors, not an independent generalization test. Neural quality and scope-filter correctness are reported separately.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Default CI exercises deterministic code and the map workbench without downloading model weights. Neural execution is separately recorded in the experiment report; a green core test suite alone does not prove model performance.

## Repository guide

| Path | Purpose |
|---|---|
| `src/geoevidence/` | ingestion, storage, filters, rankers, generation, API, CLI |
| `app.py` | map and evidence workbench |
| `data/` | reproducible fictional corpus, filters, labeled evaluation queries |
| `tests/` | behavioral and integration regressions |
| `reports/` | executed experiment outputs, without model weights |
| `docs/` | architecture, evaluation and engineering limitations |

## Models, licenses and references

Project code and authored demo data are MIT-licensed. MiniLM and FLAN-T5 are Apache-2.0 models distributed separately under their upstream licenses. The project does not claim authorship of their pretrained weights.

- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [Sentence Transformers similarity documentation](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).
- [FLAN-T5-small model card](https://huggingface.co/google/flan-t5-small).
- [PyProj geodesic API](https://pyproj4.github.io/pyproj/stable/api/geod.html).
- [Reciprocal rank fusion paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
- [RFC 7946: GeoJSON](https://www.rfc-editor.org/rfc/rfc7946).
