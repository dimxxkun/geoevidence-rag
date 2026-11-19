# Dataset card

**Name:** fictional planning briefs, version 1.

**License:** [Noncommercial Use and Commercial Permission License](../LICENSE), authored for this project. Commercial use requires a separate written agreement and an agreed fee.

**Purpose:** exercise spatial filtering, document provenance, ranker combinations, dated distractors and known missing answers.

`demo.geojson` is generated deterministically by `python data/build_demo.py`. The source contains 20 English briefs covering drainage, electricity, water, heat, transport, coast, waste, land and ocean instrumentation. Most anchors are approximate locations around Nigerian cities. Two Pacific anchors exercise the international date line. Dates and all operational details are fictional; the file includes no customer, private network or emergency-response data.

Geometry represents the **document's location anchor**, not an asset location, hazard boundary, footprint or service area. Similar documents intentionally describe different actions in different cities. An archived mainland drainage brief proposes a pump; the current exercise recommends clearing blocked inlets. The fixture tests whether location and date constraints can keep these distinct.

The evaluation contains 20 authored questions: 14 with identified relevant documents and six deliberately unanswerable requests. Four negative cases exclude all documents by explicit scope; one asks about unrelated physics; one asks for an exact budget that the matching brief does not provide. That final case deliberately exposes the difference between retrieving a related document and establishing that the question is answerable.

These are development fixtures, not randomly sampled observations. The corpus and queries were designed together. There is no train/test split and no model was trained on them. Good scores show implementation behavior on these cases; they do not establish deployment accuracy, geographic representativeness, or performance on unseen organizations' records. Do not tune the threshold on this set and then describe it as held-out performance.

Sources use `fixture://planning-briefs/<id>` to identify local authored text. They are provenance identifiers, not public web URLs. GeoEvidence preserves them rather than inventing external references. For real data, use actual source URLs or durable internal identifiers and respect access restrictions before ingestion or publication.
