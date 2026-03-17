from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_engine.common.models import QueryRequest
from graphrag_engine.runtime import GraphRAGRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest raw corpus documents")
    ingest.add_argument("paths", nargs="*", help="Optional source paths")

    subparsers.add_parser("extract", help="Extract entities and relations")
    subparsers.add_parser("build-graph", help="Load graph artifacts and optionally populate Neo4j")
    subparsers.add_parser("reindex", help="Warm retrieval indexes from graph artifacts")

    query = subparsers.add_parser("query", help="Run a GraphRAG query")
    query.add_argument("question", help="Question to answer")
    query.add_argument("--mode", default="hybrid", choices=["hybrid", "baseline"])
    query.add_argument("--top-k", type=int, default=8)

    subparsers.add_parser("doctor", help="Show runtime configuration and backend status")
    subparsers.add_parser("run-eval", help="Run benchmark evaluation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runtime = GraphRAGRuntime()

    if args.command == "ingest":
        paths = [Path(item).resolve() for item in args.paths] if args.paths else None
        result = runtime.ingestion.ingest(paths)
        print(result.model_dump_json(indent=2))
        return

    if args.command == "extract":
        result = runtime.extraction.extract()
        print(json.dumps(result, indent=2))
        return

    if args.command == "build-graph":
        result = runtime.graph.build_graph()
        print(result.model_dump_json(indent=2))
        return

    if args.command == "reindex":
        retriever = runtime.build_retriever()
        print(json.dumps({"indexed_chunks": len(retriever.chunk_by_id)}, indent=2))
        return

    if args.command == "query":
        response = runtime.build_agent().run(
            QueryRequest(question=args.question, retrieval_mode=args.mode, top_k=args.top_k)
        )
        print(response.model_dump_json(indent=2))
        return

    if args.command == "doctor":
        provider = runtime.provider.describe()
        payload = {
            "provider": provider,
            "data_dir": str(runtime.settings.data_path),
            "raw_files": sorted(path.name for path in runtime.settings.raw_data_path.iterdir() if path.is_file()),
            "processed_dirs": sorted(path.name for path in runtime.settings.processed_data_path.iterdir()),
            "neo4j_uri": runtime.settings.neo4j_uri,
            "model_backend": runtime.settings.model_backend,
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "run-eval":
        summary = runtime.build_evaluator().run()
        print(summary.model_dump_json(indent=2))
        return


if __name__ == "__main__":
    main()
