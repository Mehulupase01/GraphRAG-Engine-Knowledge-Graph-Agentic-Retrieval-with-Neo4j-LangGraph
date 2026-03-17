from __future__ import annotations

import re

from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import ChunkRecord, SectionRecord


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _merge_wrapped_lines(lines: list[str]) -> list[str]:
    units: list[str] = []
    buffer: list[str] = []

    for raw_line in lines:
        line = _compact_whitespace(raw_line)
        if not line:
            continue
        if re.match(r"^(Article|Chapter|Section|Title|Annex)\b", line, flags=re.IGNORECASE):
            if buffer:
                units.append(" ".join(buffer))
                buffer = []
            units.append(line)
            continue

        buffer.append(line)
        if re.search(r"[.:;!?]\)?$", line):
            units.append(" ".join(buffer))
            buffer = []

    if buffer:
        units.append(" ".join(buffer))
    return [unit for unit in units if unit]


def _split_sentences(text: str) -> list[str]:
    sentences = [
        _compact_whitespace(part)
        for part in re.split(r"(?<=[.!?;:])\s+(?=[A-Z0-9(])", _compact_whitespace(text))
        if part.strip()
    ]
    return sentences or [_compact_whitespace(text)]


def _split_word_windows(text: str, *, max_chunk_tokens: int) -> list[str]:
    words = text.split()
    max_words = max(1, int(max_chunk_tokens / 1.25))
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def _prepare_units(text: str, *, max_chunk_tokens: int) -> list[str]:
    paragraphs = [_compact_whitespace(part) for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(paragraphs) <= 1:
        lines = [line for line in text.splitlines() if line.strip()]
        paragraphs = _merge_wrapped_lines(lines) if lines else [_compact_whitespace(text)]

    units: list[str] = []
    for paragraph in paragraphs:
        if _estimate_tokens(paragraph) <= max_chunk_tokens:
            units.append(paragraph)
            continue

        sentences = _split_sentences(paragraph)
        if len(sentences) > 1:
            for sentence in sentences:
                if _estimate_tokens(sentence) <= max_chunk_tokens:
                    units.append(sentence)
                else:
                    units.extend(_split_word_windows(sentence, max_chunk_tokens=max_chunk_tokens))
            continue

        units.extend(_split_word_windows(paragraph, max_chunk_tokens=max_chunk_tokens))

    return [unit for unit in units if unit]


def chunk_section(section: SectionRecord, *, max_chunk_tokens: int, overlap: int) -> list[ChunkRecord]:
    units = _prepare_units(section.text, max_chunk_tokens=max_chunk_tokens)
    if not units:
        units = [_compact_whitespace(section.text)]

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
                metadata={
                    "section_title": section.title,
                    "unit_count": len(current_parts),
                },
            )
        )
        sequence += 1
        overlap_parts = current_parts[-1:] if overlap > 0 else []
        current_parts = overlap_parts
        current_tokens = _estimate_tokens("\n\n".join(current_parts)) if overlap_parts else 0

    for unit in units:
        unit_tokens = _estimate_tokens(unit)
        if current_tokens + unit_tokens > max_chunk_tokens and current_parts:
            flush()
        current_parts.append(unit)
        current_tokens += unit_tokens

    flush()
    return chunks
