"""Small command surface for reproducible ingest, retrieval, generation and evaluation."""

import argparse
import json
from pathlib import Path

from .encoding import MiniLMEncoder
from .engine import Engine
from .evaluation import evaluate
from .models import Query
from .retrieval import Retriever
from .store import build_index, load_index


def main():
    parser = argparse.ArgumentParser(prog="geoevidence")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Atomically build an index from point GeoJSON")
    index.add_argument("corpus", type=Path)
    index.add_argument("--output", type=Path, default=Path("artifacts/demo.sqlite"))
    index.add_argument("--encoder", choices=["lexical", "minilm"], default="lexical")
    index.add_argument("--chunk-words", type=int, default=80)
    index.add_argument("--overlap", type=int, default=15)
    ask = sub.add_parser("ask", help="Return an answer and exact evidence as JSON")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=Path("artifacts/demo.sqlite"))
    ask.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="lexical")
    ask.add_argument("--generator", choices=["extractive", "flan"], default="extractive")
    ask.add_argument("--filters", type=Path, help="JSON Filters object; coordinates are lon, lat")
    ask.add_argument("--top-k", type=int, default=3)
    evaluation = sub.add_parser("evaluate", help="Measure document retrieval against labeled cases")
    evaluation.add_argument("cases", type=Path)
    evaluation.add_argument("--index", type=Path, default=Path("artifacts/demo.sqlite"))
    evaluation.add_argument(
        "--modes", nargs="+", choices=["lexical", "dense", "hybrid"], default=["lexical"]
    )
    evaluation.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    try:
        if args.command == "index":
            result = build_index(
                args.corpus,
                args.output,
                MiniLMEncoder() if args.encoder == "minilm" else None,
                args.chunk_words,
                args.overlap,
            )
        elif args.command == "ask":
            filters = json.loads(args.filters.read_text()) if args.filters else {}
            result = (
                Engine.open(args.index, args.generator)
                .ask(
                    Query(question=args.question, filters=filters, mode=args.mode, top_k=args.top_k)
                )
                .model_dump(mode="json")
            )
        else:
            index_data = load_index(args.index)
            encoder = MiniLMEncoder() if any(m != "lexical" for m in args.modes) else None
            result = evaluate(Retriever(index_data, encoder), args.cases, args.modes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            result = {"report": str(args.output), "summary": result["summary"]}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"geoevidence: {exc}\n")


if __name__ == "__main__":
    main()
