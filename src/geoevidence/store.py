"""Atomic SQLite snapshots: corpus, chunks and vectors share one versioned identity."""

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encoding import Encoder, validate_vectors
from .ingest import chunk_document, read_geojson
from .models import Chunk

SCHEMA_VERSION = 1


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray | None
    metadata: dict


def build_index(
    corpus: Path,
    destination: Path,
    encoder: Encoder | None = None,
    words: int = 80,
    overlap: int = 15,
) -> dict:
    documents = read_geojson(corpus)
    chunks = [chunk for doc in documents for chunk in chunk_document(doc, words, overlap)]
    if len(chunks) > 25_000:
        raise ValueError("Index exceeds the 25000-chunk local-demo limit")
    canonical = json.dumps([doc.model_dump(mode="json") for doc in documents], sort_keys=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "encoder": encoder.fingerprint if encoder else "lexical",
        "chunk_words": words,
        "chunk_overlap": overlap,
        "chunker": "word-window-sentence-end-v1",
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }
    vectors = None
    if encoder:
        vectors = validate_vectors(
            encoder.encode([chunk.document.title + "\n" + chunk.text for chunk in chunks]),
            len(chunks),
        )
    metadata["vector_dimensions"] = vectors.shape[1] if vectors is not None else 0
    metadata["fingerprint"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".index-", suffix=".sqlite", dir=destination.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(temporary)
        try:
            with connection:
                connection.execute("CREATE TABLE metadata (payload TEXT NOT NULL)")
                connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
                connection.execute(
                    "CREATE TABLE chunks (position INTEGER PRIMARY KEY, payload TEXT NOT NULL, "
                    "vector BLOB)"
                )
                connection.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?)",
                    [
                        (
                            pos,
                            chunk.model_dump_json(),
                            vectors[pos].astype("<f4").tobytes() if vectors is not None else None,
                        )
                        for pos, chunk in enumerate(chunks)
                    ],
                )
        finally:
            connection.close()
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata


def load_index(path: Path) -> Index:
    if not path.is_file():
        raise FileNotFoundError(f"Index not found: {path}. Run geoevidence index first.")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        metadata = json.loads(connection.execute("SELECT payload FROM metadata").fetchone()[0])
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported index schema; rebuild the index")
        rows = connection.execute("SELECT payload, vector FROM chunks ORDER BY position").fetchall()
    finally:
        connection.close()
    chunks = [Chunk.model_validate_json(row[0]) for row in rows]
    if not chunks or len(chunks) != metadata["chunk_count"]:
        raise ValueError("Index chunk count is inconsistent; rebuild the index")
    vectors = None
    if metadata["encoder"] != "lexical":
        vectors = validate_vectors(
            np.stack([np.frombuffer(row[1], dtype="<f4") for row in rows]), len(chunks)
        )
        if vectors.shape[1] != metadata["vector_dimensions"]:
            raise ValueError("Index vector dimensions are inconsistent; rebuild the index")
    return Index(chunks, vectors, metadata)
