from __future__ import annotations

import re
from pathlib import Path

from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import DocumentRecord, SectionRecord

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None


SECTION_HEADING_RE = re.compile(r"^(Article\s+\d+[A-Za-z-]*|Chapter\s+\w+|Section\s+\w+|Title\s+\w+)", re.IGNORECASE)


def parse_document(path: Path) -> tuple[DocumentRecord, list[str]]:
    if path.suffix.lower() == ".pdf" and PdfReader is not None:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rough_pages = [page.strip() for page in text.split("\f")]
        if len(rough_pages) == 1:
            rough_pages = [text[index : index + 3000] for index in range(0, len(text), 3000)]
        pages = [page for page in rough_pages if page]

    joined = "\n".join(pages)
    document = DocumentRecord(
        document_id=stable_hash(str(path.resolve()), prefix="doc"),
        name=path.stem,
        source_path=str(path.resolve()),
        checksum=stable_hash(joined, prefix="sha"),
        page_count=max(len(pages), 1),
        metadata={"suffix": path.suffix.lower()},
    )
    return document, pages


def split_into_sections(document: DocumentRecord, pages: list[str]) -> list[SectionRecord]:
    sections: list[SectionRecord] = []
    current_title = "Preamble"
    current_lines: list[str] = []
    current_article: str | None = None
    current_page = 1
    start_page = 1

    def flush() -> None:
        nonlocal current_lines, current_title, current_article, start_page, current_page
        text = "\n".join(line for line in current_lines if line.strip()).strip()
        if not text:
            current_lines = []
            return
        section_id = stable_hash(f"{document.document_id}:{current_title}:{start_page}", prefix="sec")
        sections.append(
            SectionRecord(
                section_id=section_id,
                document_id=document.document_id,
                title=current_title,
                article_ref=current_article,
                page_start=start_page,
                page_end=current_page,
                text=text,
            )
        )
        current_lines = []

    for page_number, page_text in enumerate(pages, start=1):
        current_page = page_number
        for line in page_text.splitlines():
            line = line.strip()
            if not line:
                continue
            heading_match = SECTION_HEADING_RE.match(line)
            if heading_match:
                flush()
                current_title = heading_match.group(1)
                current_article = current_title if current_title.lower().startswith("article") else None
                start_page = page_number
            current_lines.append(line)

    flush()
    if not sections:
        sections.append(
            SectionRecord(
                section_id=stable_hash(document.document_id, prefix="sec"),
                document_id=document.document_id,
                title="Document",
                article_ref=None,
                page_start=1,
                page_end=document.page_count,
                text="\n".join(pages),
            )
        )
    return sections

