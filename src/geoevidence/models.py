"""Validated public contracts. All coordinates use WGS84 longitude, latitude."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Longitude = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False, strict=True)]
Latitude = Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False, strict=True)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Point(Contract):
    lon: Longitude
    lat: Latitude


class Bounds(Contract):
    west: Longitude
    south: Latitude
    east: Longitude
    north: Latitude

    @model_validator(mode="after")
    def latitude_order(self):
        if self.south > self.north:
            raise ValueError("south must not exceed north; west > east means crossing the dateline")
        return self


class Radius(Contract):
    center: Point
    km: float = Field(gt=0, le=20040, allow_inf_nan=False)


class Filters(Contract):
    bbox: Bounds | None = None
    radius: Radius | None = None
    topic: str | None = Field(default=None, min_length=1, max_length=80)
    published_from: date | None = None
    published_to: date | None = None

    @model_validator(mode="after")
    def date_order(self):
        if self.published_from and self.published_to and self.published_from > self.published_to:
            raise ValueError("published_from must not exceed published_to")
        return self


class Document(Contract):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    title: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=100_000)
    source: str = Field(min_length=1, max_length=2000)
    published: date
    topic: str = Field(min_length=1, max_length=80)
    location: Point


class Chunk(Contract):
    id: str
    document: Document
    text: str
    start: int
    end: int


class Query(Contract):
    question: str = Field(min_length=3, max_length=2000)
    filters: Filters = Field(default_factory=Filters)
    mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=3, ge=1, le=10)
    min_similarity: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)


class Evidence(Contract):
    citation: str
    chunk_id: str
    document_id: str
    title: str
    source: str
    published: date
    location: Point
    quote: str
    start: int
    end: int
    score: float
    lexical_score: float
    cosine_similarity: float | None
    distance_km: float | None


class Answer(Contract):
    question: str
    answer: str
    status: Literal["answered", "no_evidence"]
    generator: str
    grounding: Literal["verbatim", "unverified_generation", "not_applicable"]
    evidence: list[Evidence]
    eligible_documents: int
    index_fingerprint: str
    elapsed_ms: float
