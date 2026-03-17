from __future__ import annotations

import unittest

from graphrag_engine.common.models import DocumentRecord
from graphrag_engine.ingestion.parser import split_into_sections


class ParserTests(unittest.TestCase):
    def test_split_into_sections_normalizes_ocr_style_headings(self) -> None:
        document = DocumentRecord(
            document_id="doc-1",
            name="sample",
            source_path="sample.pdf",
            checksum="sha-1",
            page_count=2,
        )
        pages = [
            "\n".join(
                [
                    "Preamble line",
                    "Ar ticle 13",
                    "Information to be provided",
                    "The controller shall provide information.",
                ]
            ),
            "\n".join(
                [
                    "Section 2",
                    "Rectification and erasure",
                    "Ar ticle 16",
                    "Right to rectification",
                ]
            ),
        ]

        sections = split_into_sections(document, pages)

        self.assertGreaterEqual(len(sections), 3)
        self.assertEqual(sections[1].title, "Article 13")
        self.assertEqual(sections[1].article_ref, "Article 13")
        self.assertEqual(sections[-1].title, "Article 16")


if __name__ == "__main__":
    unittest.main()
