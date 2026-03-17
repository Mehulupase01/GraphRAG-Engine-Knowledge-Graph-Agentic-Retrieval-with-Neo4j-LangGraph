from __future__ import annotations

import re

from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import ChunkRecord, SectionRecord


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))


def chunk_section(section: SectionRecord, *, max_chunk_tokens: int, overlap: int) -> list[ChunkRecord]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", section.text) if part.strip()]
    if not paragraphs:
        paragraphs = [section.text]

    chunks: list[ChunkRecord] = []
    current_parts: list[str] = []
    current_tokens = 0
    sequence = 0

    def flush() -> None:
        nonlocal current_parts, current_tokens, sequence
        if not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        chunk_id = stable_hash(f"{section.section_id}:{sequence}:{text}", prefix="chunk")
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=section.document_id,
                section_id=section.section_id,
                article_ref=section.article_ref,
                page_start=section.page_start,
                page_end=section.page_end,
                sequence=sequence,
                text=text,
                text_hash=stable_hash(text, prefix="text"),
                token_estimate=_estimate_tokens(text),
                metadata={"section_title": section.title},
            )
        )
        sequence += 1
        overlap_parts = current_parts[-1:] if overlap > 0 else []
        current_parts = overlap_parts
        current_tokens = _estimate_tokens("\n\n".join(current_parts)) if overlap_parts else 0

    for paragraph in paragraphs:
        paragraph_tokens = _estimate_tokens(paragraph)
        if current_tokens + paragraph_tokens > max_chunk_tokens and current_parts:
            flush()
        current_parts.append(paragraph)
        current_tokens += paragraph_tokens

    flush()
    return chunks

