"""Application workflow shared by CLI, HTTP API and the map demo."""

from pathlib import Path
from threading import Lock
from time import perf_counter

from .encoding import MiniLMEncoder
from .generation import ExtractiveGenerator, FlanGenerator, Generator
from .models import Answer, Query
from .retrieval import Retriever
from .store import load_index


class Engine:
    def __init__(self, retriever: Retriever, generator: Generator | None = None):
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self._lock = Lock()

    @classmethod
    def open(cls, path: Path, generator: str = "extractive"):
        index = load_index(path)
        encoder = MiniLMEncoder() if index.metadata["encoder"] != "lexical" else None
        if generator not in ("extractive", "flan"):
            raise ValueError("generator must be extractive or flan")
        return cls(Retriever(index, encoder), FlanGenerator() if generator == "flan" else None)

    def ask(self, query: Query) -> Answer:
        started = perf_counter()
        # Shared CPU models run one request at a time in the local service.
        with self._lock:
            evidence, eligible = self.retriever.search(query)
            if evidence:
                generation = self.generator.generate(query.question, evidence)
                text, used, grounding = generation.text, generation.evidence, generation.grounding
                status = "answered"
            else:
                text = "No matching evidence was found within the requested filters."
                used, grounding, status = [], "not_applicable", "no_evidence"
        return Answer(
            question=query.question,
            answer=text,
            status=status,
            generator=self.generator.name,
            grounding=grounding,
            evidence=used,
            eligible_documents=eligible,
            index_fingerprint=self.retriever.index.metadata["fingerprint"],
            elapsed_ms=round((perf_counter() - started) * 1000, 3),
        )
