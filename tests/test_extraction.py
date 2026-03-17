from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from graphrag_engine.common.artifacts import write_jsonl
from graphrag_engine.common.models import ChunkRecord
from graphrag_engine.common.providers import HeuristicLLMProvider
from graphrag_engine.common.settings import Settings
from graphrag_engine.extraction.service import ExtractionService


class ExtractionTests(unittest.TestCase):
    def test_entity_resolution_merges_aliases(self) -> None:
        tmp = Path.cwd() / "data" / "cache" / "test_extraction_case"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = Settings(data_dir=str(tmp))
            ingestion_dir = settings.processed_data_path / "ingestion"
            chunk = ChunkRecord(
                chunk_id="chunk-1",
                document_id="doc-1",
                section_id="sec-1",
                article_ref="Article 6",
                text="The AI Act requires providers to document conformity assessment under Article 6.",
                text_hash="hash",
            )
            write_jsonl(ingestion_dir / "chunks.jsonl", [chunk.model_dump()])
            service = ExtractionService(settings, HeuristicLLMProvider(settings))
            result = service.extract()
            self.assertGreaterEqual(result["entities"], 2)
            self.assertGreaterEqual(result["relations"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
