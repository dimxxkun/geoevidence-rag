"""Lazy neural dependencies keep the lexical baseline lightweight."""

from typing import Protocol

import numpy as np

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class Encoder(Protocol):
    fingerprint: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEncoder:
    fingerprint = f"{MINILM}@{MINILM_REVISION}:normalized:384"

    def __init__(self):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the neural extra: uv sync --extra neural") from exc
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model = SentenceTransformer(MINILM, revision=MINILM_REVISION, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        lengths = self.model.tokenizer(texts, truncation=False, return_length=True)["length"]
        if any(length > self.model.max_seq_length for length in lengths):
            raise ValueError(
                "Text exceeds the embedding model token budget; shorten the question or "
                "rebuild with smaller chunks/titles"
            )
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)


def validate_vectors(vectors: np.ndarray, count: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != count or not 1 <= vectors.shape[1] <= 4096:
        raise ValueError("Encoder returned an incompatible vector shape")
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned non-finite vectors")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (lengths <= 0).any():
        raise ValueError("Encoder returned a zero vector")
    return vectors / lengths
