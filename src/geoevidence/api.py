"""Read-only local API. Index ingestion is an explicit CLI operation."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .engine import Engine
from .models import Answer, Query


def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or Engine.open(
            Path(os.environ.get("GEOEVIDENCE_INDEX", "artifacts/demo.sqlite")),
            os.environ.get("GEOEVIDENCE_GENERATOR", "extractive"),
        )
        yield

    app = FastAPI(title="GeoEvidence RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        metadata = app.state.engine.retriever.index.metadata
        return {"status": "ok", "index": metadata, "generator": app.state.engine.generator.name}

    @app.post("/ask", response_model=Answer)
    def ask(query: Query):
        try:
            return app.state.engine.ask(query)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503, detail="Local model could not produce an answer"
            ) from exc

    return app
