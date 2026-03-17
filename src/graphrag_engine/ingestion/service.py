from __future__ import annotations

import logging
from pathlib import Path

from graphrag_engine.common.artifacts import ensure_dir, write_json, write_jsonl
from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import ChunkRecord, DocumentRecord, IngestionJobRecord, SectionRecord
from graphrag_engine.common.settings import Settings

from .chunking import chunk_section
from .parser import parse_document, split_into_sections

LOGGER = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ingest(self, source_paths: list[Path] | None = None) -> IngestionJobRecord:
        files = source_paths or self._discover_sources()
        job_id = stable_hash("|".join(str(path) for path in files), prefix="ingest")
        artifacts_root = ensure_dir(self.settings.processed_data_path / "ingestion")
        jobs_root = ensure_dir(self.settings.processed_data_path / "jobs")
        documents: list[DocumentRecord] = []
        sections: list[SectionRecord] = []
        chunks: list[ChunkRecord] = []
        failures: list[str] = []
        artifact_paths: list[str] = []

        for path in files:
            try:
                document, pages = parse_document(path)
                doc_sections = split_into_sections(document, pages)
                doc_chunks = [
                    chunk
                    for section in doc_sections
                    for chunk in chunk_section(
                        section,
                        max_chunk_tokens=self.settings.max_chunk_tokens,
                        overlap=self.settings.chunk_overlap,
                    )
                ]
                doc_dir = ensure_dir(artifacts_root / document.document_id)
                write_json(doc_dir / "document.json", document.model_dump())
                write_jsonl(doc_dir / "sections.jsonl", [section.model_dump() for section in doc_sections])
                write_jsonl(doc_dir / "chunks.jsonl", [chunk.model_dump() for chunk in doc_chunks])
                artifact_paths.extend(
                    [str(doc_dir / "document.json"), str(doc_dir / "sections.jsonl"), str(doc_dir / "chunks.jsonl")]
                )
                documents.append(document)
                sections.extend(doc_sections)
                chunks.extend(doc_chunks)
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.exception("document_ingestion_failed", extra={"path": str(path)})
                failures.append(f"{path}: {exc}")

        write_jsonl(artifacts_root / "documents.jsonl", [doc.model_dump() for doc in documents])
        write_jsonl(artifacts_root / "sections.jsonl", [section.model_dump() for section in sections])
        write_jsonl(artifacts_root / "chunks.jsonl", [chunk.model_dump() for chunk in chunks])

        job = IngestionJobRecord(
            job_id=job_id,
            phase="completed" if not failures else "completed_with_errors",
            source_files=[str(path) for path in files],
            artifact_paths=artifact_paths,
            counts={
                "documents": len(documents),
                "sections": len(sections),
                "chunks": len(chunks),
            },
            failures=failures,
        )
        write_json(jobs_root / f"{job_id}.json", job.model_dump())
        return job

    def _discover_sources(self) -> list[Path]:
        candidates = []
        for suffix in ("*.pdf", "*.txt", "*.md"):
            candidates.extend(self.settings.raw_data_path.glob(suffix))
        return sorted(path for path in candidates if path.is_file())

