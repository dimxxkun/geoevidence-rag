# Design and tradeoffs

## Boundaries and ownership

`models.py` owns public schemas. `ingest.py` validates point GeoJSON and turns source text into overlapping chunks. Each chunk stores the source document and exact character offsets; neither retrieval nor generation reconstructs the original text from token lists. Tokenization for BM25 is separate from provenance text.

`store.py` builds a complete snapshot in a temporary SQLite file before replacing the previous index. A failed input validation or encoding operation leaves the current snapshot in place. The fingerprint includes normalized source content, encoder identity, vector dimensions and chunk settings. Rebuilding is explicit; the index does not watch a mutable input file. Running services hold the loaded snapshot until restarted, while the demo reloads when its file modification time changes.

SQLite was chosen for a portable single-file artifact and atomic local replacement. It is not used as a spatial SQL engine. The service loads chunks and vectors into memory. At the supported demo scale, a direct NumPy cosine scan is simpler to validate than an approximate nearest-neighbor index. A larger deployment would need measured indexing, bounding-box preselection, a vector index and a bounded request queue.

## Geographic semantics

All coordinates are EPSG:4326 longitude/latitude. Radius filtering uses the inverse WGS84 ellipsoidal geodesic supplied by PyProj. A degree is never treated as a distance unit. Rectangular bounds include their edges and support a west value larger than east for dateline wrapping. Positive and negative 180 degrees are treated as the same meridian on the boundary.

Bounds, radius, topic and publication interval intersect. Filtering happens before candidate ranking, including dense search. If nothing is eligible, the result is empty and generation is skipped. There is no automatic global fallback or hidden widening of the radius.

Point anchoring is an intentional scope choice. A document may discuss a region larger than its anchor; this version cannot establish polygon overlap or regional coverage. Place names in the question are not geocoded. The caller selects explicit coordinates and date constraints.

## Retrieval

The BM25 implementation uses corpus-level document frequencies over chunks, `k1=1.5`, `b=0.75`, Unicode word tokens and a small English stopword list. Overlapping chunks can affect these statistics, so this is a transparent baseline rather than an independently optimized search engine.

MiniLM encodes `title + passage` and the question into normalized embeddings. Query vectors must match the index's encoder fingerprint and dimensions. A fixed cosine threshold of 0.3 removes low-similarity dense candidates; this is a starting value, not a calibrated confidence threshold. Lexical candidates require positive BM25 overlap.

Hybrid search sums `1 / (60 + rank)` across the two eligible rankings. The method avoids equating BM25 magnitudes with cosine values. Ties use stable chunk IDs. After fusion, only the best chunk from each document is retained. This reduces redundant context but may miss a second useful passage from a long document. Scores are ranking signals, not probabilities.

## Answer generation

The extractive baseline displays original passages with evidence labels. The neural option uses pinned FLAN-T5-small on the CPU with deterministic greedy decoding. Lowest-ranked passages are dropped until the complete prompt fits 512 tokens; dropped passages are also removed from the response's evidence list. If not even one complete passage and the question fit, the caller receives an actionable error instead of a silently truncated source.

FLAN-T5 was chosen to make local generative execution accessible on a CPU. It is small and not a strong general-purpose instruction follower. No fine-tuning is claimed. Returned evidence describes supplied context; generated wording is not automatically entailed by that evidence. A production system would need domain evaluation, stronger answerability controls and claim verification before presenting model prose as established fact.

## Interface and resource behavior

The command line, FastAPI and Streamlit all use the same engine. Models are loaded once per engine. The local API serializes inference with a lock; it does not implement multi-user scheduling, authentication or uploads. Corpus bytes, document count, chunk count, vector dimensions, question length and top-k are bounded. These limits are for an approachable local project, not a claim of capacity testing.

Neural libraries are optional. Core CI uses deterministic encoder doubles to test ranking and isolation contracts; real model execution and its output are separately recorded. No test double is cited as evidence of neural quality.
