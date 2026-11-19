"""BM25 + cosine retrieval, hard spatial filtering, and reciprocal rank fusion."""

import math
import re
from collections import Counter

import numpy as np

from .encoding import Encoder, validate_vectors
from .geo import match
from .models import Evidence, Query
from .store import Index

STOPWORDS = frozenset(
    "a an and are as at be by did do does for from how i in is it of on or that the "
    "their there these this to was were what when where which who why with would".split()
)


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[\w]+", text.casefold()) if word not in STOPWORDS]


class Retriever:
    def __init__(self, index: Index, encoder: Encoder | None = None):
        if encoder and encoder.fingerprint != index.metadata["encoder"]:
            raise ValueError("Encoder does not match this index; rebuild with the selected encoder")
        self.index, self.encoder = index, encoder
        self.terms = [Counter(tokens(c.document.title + " " + c.text)) for c in index.chunks]
        self.lengths = np.array([sum(counts.values()) for counts in self.terms])
        self.average_length = max(float(self.lengths.mean()), 1.0)
        frequency = Counter(term for counts in self.terms for term in counts)
        self.idf = {
            term: math.log(1 + (len(self.terms) - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def _bm25(self, question: str, candidates: list[int]) -> dict[int, float]:
        terms = set(tokens(question))
        result = {}
        for position in candidates:
            counts = self.terms[position]
            normalizer = 1.5 * (0.25 + 0.75 * self.lengths[position] / self.average_length)
            result[position] = sum(
                self.idf.get(term, 0) * counts[term] * 2.5 / (counts[term] + normalizer)
                for term in terms
                if counts[term]
            )
        return result

    def search(self, query: Query) -> tuple[list[Evidence], int]:
        if query.mode != "lexical" and (self.encoder is None or self.index.vectors is None):
            raise ValueError("Dense/hybrid search needs a neural index and its matching encoder")
        candidates, distances, eligible = [], {}, set()
        # All constraints apply before either ranker. An empty area never falls back globally.
        document_matches = {}
        for position, chunk in enumerate(self.index.chunks):
            document_matches.setdefault(chunk.document.id, None)
            if document_matches[chunk.document.id] is None:
                document_matches[chunk.document.id] = match(chunk.document, query.filters)
            accepted, distance = document_matches[chunk.document.id]
            if accepted:
                candidates.append(position)
                distances[position] = distance
                eligible.add(chunk.document.id)
        if not candidates:
            return [], 0
        lexical = self._bm25(query.question, candidates) if query.mode != "dense" else {}
        dense = {}
        if query.mode != "lexical":
            vector = validate_vectors(self.encoder.encode([query.question]), 1)[0]
            if vector.shape[0] != self.index.vectors.shape[1]:
                raise ValueError("Query vector dimensions differ from index dimensions")
            values = self.index.vectors[candidates] @ vector
            dense = {pos: float(value) for pos, value in zip(candidates, values, strict=True)}
        lexical_ranks = sorted(
            (pos for pos in candidates if lexical.get(pos, 0) > 0),
            key=lambda pos: (-lexical[pos], self.index.chunks[pos].id),
        )
        dense_ranks = sorted(
            (pos for pos in candidates if dense.get(pos, -1) >= query.min_similarity),
            key=lambda pos: (-dense[pos], self.index.chunks[pos].id),
        )
        scores: dict[int, float] = {}
        if query.mode == "hybrid":
            for ranking in (lexical_ranks, dense_ranks):
                for rank, pos in enumerate(ranking, start=1):
                    scores[pos] = scores.get(pos, 0) + 1 / (60 + rank)
        elif query.mode == "dense":
            scores = {pos: dense[pos] for pos in dense_ranks}
        else:
            scores = {pos: lexical[pos] for pos in lexical_ranks}
        ranking = sorted(scores, key=lambda pos: (-scores[pos], self.index.chunks[pos].id))
        evidence, seen = [], set()
        for pos in ranking:
            chunk = self.index.chunks[pos]
            document = chunk.document
            if document.id in seen:
                continue  # One chunk per document keeps overlap from monopolizing context.
            seen.add(document.id)
            evidence.append(
                Evidence(
                    citation=f"E{len(evidence) + 1}",
                    chunk_id=chunk.id,
                    document_id=document.id,
                    title=document.title,
                    source=document.source,
                    published=document.published,
                    location=document.location,
                    quote=chunk.text,
                    start=chunk.start,
                    end=chunk.end,
                    score=scores[pos],
                    lexical_score=lexical.get(pos, 0),
                    cosine_similarity=dense.get(pos),
                    distance_km=distances[pos],
                )
            )
            if len(evidence) == query.top_k:
                break
        return evidence, len(eligible)
