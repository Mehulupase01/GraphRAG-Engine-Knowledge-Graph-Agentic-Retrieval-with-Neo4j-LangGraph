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


if __name__ == "__main__":
    unittest.main()

