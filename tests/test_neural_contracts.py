"""Neural boundary contracts tested without downloading or claiming to evaluate models."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from geoevidence.encoding import MiniLMEncoder, validate_vectors
from geoevidence.generation import FlanGenerator
from geoevidence.models import Query
from geoevidence.retrieval import Retriever


@pytest.mark.parametrize(
    "values,count",
    [
        ([[0, 0]], 1),
        ([[float("nan"), 1]], 1),
        ([[1, 0]], 2),
        ([1, 0], 1),
    ],
)
def test_invalid_embeddings_rejected(values, count):
    with pytest.raises(ValueError):
        validate_vectors(np.array(values), count)


def test_oversized_embedding_input_is_not_silently_truncated():
    encoder = object.__new__(MiniLMEncoder)
    encoder.model = SimpleNamespace(
        max_seq_length=256,
        tokenizer=lambda *args, **kwargs: {"length": [257]},
    )
    with pytest.raises(ValueError, match="token budget"):
        encoder.encode(["long question"])


class Batch(dict):
    def __init__(self, length):
        self.input_ids = SimpleNamespace(shape=(1, length))
        super().__init__(input_ids=self.input_ids)


class BudgetTokenizer:
    def __init__(self, always_full=False):
        self.always_full = always_full

    def __call__(self, prompt, **kwargs):
        both = "Canal maintenance" in prompt and "Electricity maintenance" in prompt
        return Batch(600 if self.always_full or both else 256)

    def decode(self, output, **kwargs):
        return "A generated summary."


def test_generator_reports_only_context_that_fit(index, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    generator = object.__new__(FlanGenerator)
    generator.tokenizer = BudgetTokenizer()
    generator.model = SimpleNamespace(generate=lambda **kwargs: [[1]])
    evidence, _ = Retriever(index).search(Query(question="canal feeder", mode="lexical"))
    assert len(evidence) == 2
    result = generator.generate("What maintenance is needed?", evidence)
    assert result.evidence == evidence[:1]
    assert result.grounding == "unverified_generation"


def test_generator_rejects_unfit_question_and_context(index, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    generator = object.__new__(FlanGenerator)
    generator.tokenizer = BudgetTokenizer(always_full=True)
    evidence, _ = Retriever(index).search(Query(question="canal sediment", mode="lexical"))
    with pytest.raises(ValueError, match="512-token budget"):
        generator.generate("long question", evidence)
