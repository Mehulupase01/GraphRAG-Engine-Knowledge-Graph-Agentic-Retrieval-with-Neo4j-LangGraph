from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphrag_engine.common.artifacts import ensure_dir, read_json, write_json
from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import CacheEntry
from graphrag_engine.common.settings import Settings


class PathCacheStore:
    CACHE_SCHEMA_VERSION = 2

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = ensure_dir(settings.processed_data_path / "path_cache")

    def build_cache_key(
        self,
        *,
        question: str,
        retrieval_mode: str,
        article_refs: list[str],
        document_hints: dict[str, float],
        matched_entity_ids: list[str],
        hop_limit: int,
    ) -> str:
        payload = {
            "cache_schema_version": self.CACHE_SCHEMA_VERSION,
            "question": question.strip().lower(),
            "retrieval_mode": retrieval_mode,
            "article_refs": article_refs,
            "document_hints": document_hints,
            "matched_entity_ids": matched_entity_ids,
            "hop_limit": hop_limit,
        }
        return stable_hash(json.dumps(payload, sort_keys=True), prefix="pathcache")

    def load(self, cache_key: str) -> CacheEntry | None:
        path = self.path_for_key(cache_key)
        if not path.exists():
            return None
        entry = CacheEntry.model_validate(read_json(path))
        if int(entry.metadata.get("cache_schema_version", 0)) != self.CACHE_SCHEMA_VERSION:
            return None
        return entry

    def save(self, entry: CacheEntry) -> Path:
        return write_json(self.path_for_key(entry.cache_key), entry.model_dump())

    def path_for_key(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.json"

    def clear(self) -> dict[str, Any]:
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return {
            "removed_entries": removed,
            "cache_root": str(self.root),
            "schema_version": self.CACHE_SCHEMA_VERSION,
        }

    def stats(self) -> dict[str, Any]:
        entries = list(self.root.glob("*.json"))
        return {
            "entries": len(entries),
            "cache_root": str(self.root),
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "total_size_kb": round(sum(path.stat().st_size for path in entries) / 1024, 1),
        }
