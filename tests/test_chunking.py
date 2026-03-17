from __future__ import annotations

import unittest

from graphrag_engine.common.models import SectionRecord
from graphrag_engine.ingestion.chunking import chunk_section


class ChunkingTests(unittest.TestCase):
    def test_chunk_metadata_is_stable(self) -> None:
        section = SectionRecord(
            section_id="sec-1",
            document_id="doc-1",
            title="Article 6",
            article_ref="Article 6",
            page_start=1,
            page_end=2,
            text="\n\n".join(["alpha beta gamma"] * 30),
        )
        chunks = chunk_section(section, max_chunk_tokens=20, overlap=0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].article_ref, "Article 6")
        self.assertEqual(chunks[0].page_start, 1)

    def test_chunking_handles_line_wrapped_legal_text(self) -> None:
        text = "\n".join(
            [
                "Article 13",
                "Information to be provided where personal data are collected from the data subject",
                "Where personal data relating to a data subject are collected from the data subject,",
                "the controller shall, at the time when personal data are obtained, provide the data subject",
                "with all of the following information:",
                "(a) the identity and the contact details of the controller;",
                "(b) the contact details of the data protection officer, where applicable;",
                "(c) the purposes of the processing for which the personal data are intended;",
            ]
            * 6
        )
        section = SectionRecord(
            section_id="sec-2",
            document_id="doc-1",
            title="Article 13",
            article_ref="Article 13",
            page_start=10,
            page_end=12,
            text=text,
        )

        chunks = chunk_section(section, max_chunk_tokens=120, overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_estimate <= 140 for chunk in chunks))
        self.assertTrue(all(chunk.article_ref == "Article 13" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
