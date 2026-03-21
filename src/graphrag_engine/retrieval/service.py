from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass

from graphrag_engine.common.artifacts import read_json
from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import CacheEntry, ChunkRecord, GraphPath, PathRecord, RetrievalHit
from graphrag_engine.common.providers import LLMProvider, tokenize
from graphrag_engine.common.settings import Settings

from .fusion import reciprocal_rank_fusion
from .path_cache import PathCacheStore

DOCUMENT_NAME_MAP = {
    "celex 32016r0679 en txt": "GDPR",
    "celex 32022r2065 en txt": "Digital Services Act",
    "oj l 202401689 en txt": "AI Act",
}
ARTICLE_REF_RE = re.compile(r"\bArticle\s+\d+[A-Za-z-]*\b", flags=re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "what",
    "which",
}
DOCUMENT_HINT_RULES: dict[str, tuple[tuple[str, float], ...]] = {
    "AI Act": (
        ("ai act", 1.0),
        ("high-risk ai", 0.95),
        ("high-risk systems", 0.9),
        ("ai systems", 0.7),
        ("provider", 0.55),
        ("deployer", 0.55),
        ("conformity assessment", 0.85),
        ("ce marking", 0.8),
        ("general-purpose ai", 0.8),
        ("gpaI", 0.8),
    ),
    "GDPR": (
        ("gdpr", 1.0),
        ("personal data", 0.9),
        ("data subject", 0.85),
        ("controller", 0.7),
        ("processor", 0.7),
        ("consent", 0.65),
        ("lawfulness of processing", 0.8),
    ),
    "Digital Services Act": (
        ("digital services act", 1.0),
        ("dsa", 1.0),
        ("online platform", 0.9),
        ("hosting service", 0.8),
        ("intermediary service", 0.8),
        ("search engine", 0.75),
        ("illegal content", 0.8),
    ),
}
MULTI_HOP_HINTS = (
    "how do",
    "how does",
    "connect",
    "relationship",
    "relate",
    "between",
    "compare",
    "depends on",
    "relevant when",
    "which article",
    "linked to",
    "overlap",
    "conflict",
    "across",
    "interaction",
)


@dataclass(slots=True)
class QuerySignals:
    normalized_question: str
    question_tokens: set[str]
    content_tokens: set[str]
    article_refs: set[str]
    document_hints: dict[str, float]
    matched_entity_scores: dict[str, float]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (
        math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)) or 1.0
    )


class HybridRetriever:
    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self.settings = settings
        self.provider = provider
        self.last_retrieval_meta: dict[str, object] = {}
        self.catalog = self._load_catalog()
        self.chunk_by_id = {
            chunk["chunk_id"]: ChunkRecord.model_validate(chunk)
            for chunk in self.catalog.get("chunks", [])
        }
        self.document_name_by_id = {doc["document_id"]: doc["name"] for doc in self.catalog.get("documents", [])}
        self.document_label_by_id = {
            document_id: self._canonical_document_name(name)
            for document_id, name in self.document_name_by_id.items()
        }
        self.entities = self.catalog.get("entities", [])
        self.relations = self.catalog.get("relations", [])
        self.entity_by_id = {entity["entity_id"]: entity for entity in self.entities}
        self.relations_by_subject: dict[str, list[dict]] = defaultdict(list)
        self.relations_by_object: dict[str, list[dict]] = defaultdict(list)
        for relation in self.relations:
            self.relations_by_subject[relation["subject_entity_id"]].append(relation)
            self.relations_by_object[relation["object_entity_id"]].append(relation)
        self.chunk_ids_by_entity_id = self._build_chunk_mentions()
        self.chunk_search_text = {
            chunk_id: self._chunk_search_text(chunk)
            for chunk_id, chunk in self.chunk_by_id.items()
        }
        self.chunk_tokens = {
            chunk_id: self._content_tokens(self.chunk_search_text[chunk_id])
            for chunk_id in self.chunk_by_id
        }
        self.embeddings = {
            chunk_id: provider.embed_text(chunk.text)
            for chunk_id, chunk in self.chunk_by_id.items()
        }
        self.path_cache = PathCacheStore(settings)

    def retrieve(self, question: str, *, top_k: int | None = None, mode: str = "hybrid") -> list[RetrievalHit]:
        retrieval_started = time.perf_counter()
        top_k = top_k or self.settings.default_retrieval_k
        signals = self._analyze_question(question)
        if mode in {"path_hybrid", "path_cache"}:
            hits = self._retrieve_path_mode(question, signals, top_k=top_k, mode=mode)
            self.last_retrieval_meta["total_latency_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 2)
            return hits

        query_embedding = self.provider.embed_text(question)
        vector_scores = {
            chunk_id: cosine_similarity(query_embedding, embedding)
            for chunk_id, embedding in self.embeddings.items()
        }
        lexical_scores = self._lexical_scores(question, signals)
        metadata_scores = self._metadata_scores(signals)
        graph_scores, graph_paths = self._graph_scores(signals)

        vector_rank = [chunk_id for chunk_id, _ in sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)]
        lexical_rank = [chunk_id for chunk_id, _ in sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)]
        metadata_rank = [
            chunk_id for chunk_id, _ in sorted(metadata_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        graph_rank = [chunk_id for chunk_id, _ in sorted(graph_scores.items(), key=lambda item: item[1], reverse=True)]

        if mode == "baseline":
            fused = {
                chunk_id: score for chunk_id, score in sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)
            }
        else:
            fused = reciprocal_rank_fusion([metadata_rank, lexical_rank, vector_rank, graph_rank])
            for chunk_id in self.chunk_by_id:
                fused[chunk_id] = (
                    fused.get(chunk_id, 0.0)
                    + metadata_scores.get(chunk_id, 0.0) * 1.75
                    + lexical_scores.get(chunk_id, 0.0) * 0.2
                    + graph_scores.get(chunk_id, 0.0) * 0.15
                )

        hits = self._build_hits(
            fused,
            lexical_scores=lexical_scores,
            vector_scores=vector_scores,
            graph_scores=graph_scores,
            metadata_scores=metadata_scores,
            graph_paths=graph_paths,
            top_k=top_k,
        )
        self.last_retrieval_meta = {
            "mode": mode,
            "matched_entities": len(signals.matched_entity_scores),
            "article_refs": sorted(signals.article_refs),
            "document_hints": signals.document_hints,
            "cache_hit": False,
            "path_count": 0,
            "total_latency_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
        }
        return hits

    def retrieve_adaptive(self, question: str, *, top_k: int | None = None) -> tuple[list[RetrievalHit], dict[str, object]]:
        top_k = top_k or self.settings.default_retrieval_k
        signals = self._analyze_question(question)
        routing = self._resolve_mode_from_signals(question, signals, requested_mode="adaptive")
        candidate_modes = list(
            dict.fromkeys(
                [
                    str(routing.get("resolved_mode", "hybrid")),
                    *[str(mode) for mode in routing.get("candidate_modes", [])],
                ]
            )
        )
        if not candidate_modes:
            candidate_modes = ["hybrid"]

        candidate_summaries: list[dict[str, object]] = []
        best_hits: list[RetrievalHit] = []
        best_meta: dict[str, object] = {}
        best_mode = str(routing.get("resolved_mode", "hybrid"))
        best_score = float("-inf")
        best_priority = float("-inf")
        heuristic_mode = best_mode
        heuristic_margin = 0.12

        for mode in candidate_modes:
            hits = self.retrieve(question, top_k=top_k, mode=mode)
            retrieval_meta = dict(self.last_retrieval_meta)
            candidate = self._adaptive_candidate_summary(
                question,
                signals,
                mode=mode,
                hits=hits,
                retrieval_meta=retrieval_meta,
            )
            candidate_summaries.append(candidate)
            arbitration_score = float(candidate["arbitration_score"])
            priority = float(candidate["selection_priority"])
            if (
                arbitration_score > best_score + 0.03
                or (
                    abs(arbitration_score - best_score) <= heuristic_margin
                    and mode == heuristic_mode
                )
                or (
                    abs(arbitration_score - best_score) <= 0.03
                    and priority > best_priority
                )
            ):
                best_hits = hits
                best_meta = retrieval_meta
                best_mode = mode
                best_score = arbitration_score
                best_priority = priority

        self.last_retrieval_meta = {
            **best_meta,
            "adaptive_selected_mode": best_mode,
            "adaptive_selection_score": round(best_score, 4),
            "adaptive_preselected_mode": heuristic_mode,
            "adaptive_candidate_scores": candidate_summaries,
        }
        return best_hits, {
            **routing,
            "strategy": "adaptive_compare",
            "resolved_mode": best_mode,
            "preselected_mode": heuristic_mode,
            "candidate_modes": candidate_modes,
            "candidate_scores": candidate_summaries,
        }

    def resolve_mode(self, question: str, *, requested_mode: str = "adaptive") -> dict[str, object]:
        signals = self._analyze_question(question)
        return self._resolve_mode_from_signals(question, signals, requested_mode=requested_mode)

    def _resolve_mode_from_signals(
        self,
        question: str,
        signals: QuerySignals,
        *,
        requested_mode: str,
    ) -> dict[str, object]:
        if requested_mode != "adaptive":
            return {
                "requested_mode": requested_mode,
                "resolved_mode": requested_mode,
                "strategy": "manual",
                "reasons": ["explicit_mode_selected"],
                "article_refs": sorted(signals.article_refs),
                "document_hints": signals.document_hints,
                "matched_entities": len(signals.matched_entity_scores),
                "path_signal_score": 0,
                "cache_available": False,
                "cache_key": None,
                "candidate_modes": [requested_mode],
            }
        reasons: list[str] = []
        path_signal_score = 0
        if signals.article_refs:
            path_signal_score += 2
            reasons.append("article_anchor_detected")
        if self._is_multi_hop_question(signals.normalized_question):
            path_signal_score += 2
            reasons.append("multi_hop_reasoning_detected")
        if self._is_cross_regulation_question(signals):
            path_signal_score += 2
            reasons.append("cross_regulation_scope_detected")
        if len(signals.matched_entity_scores) >= 4:
            path_signal_score += 1
            reasons.append("dense_entity_match")

        if path_signal_score >= 2:
            cache_key = self.path_cache.build_cache_key(
                question=question,
                retrieval_mode="path_cache",
                article_refs=sorted(signals.article_refs),
                document_hints=signals.document_hints,
                matched_entity_ids=sorted(signals.matched_entity_scores),
                hop_limit=self.settings.agent_max_graph_hops,
            )
            cache_available = self.path_cache.load(cache_key) is not None
            reasons.append("cache_hit_available" if cache_available else "cache_will_be_populated")
            return {
                "requested_mode": requested_mode,
                "resolved_mode": "path_cache",
                "strategy": "adaptive",
                "reasons": reasons,
                "article_refs": sorted(signals.article_refs),
                "document_hints": signals.document_hints,
                "matched_entities": len(signals.matched_entity_scores),
                "path_signal_score": path_signal_score,
                "cache_available": cache_available,
                "cache_key": cache_key,
                "candidate_modes": ["hybrid", "path_cache"],
            }

        reasons.append("direct_lookup_prefers_hybrid")
        return {
            "requested_mode": requested_mode,
            "resolved_mode": "hybrid",
            "strategy": "adaptive",
            "reasons": reasons,
            "article_refs": sorted(signals.article_refs),
            "document_hints": signals.document_hints,
            "matched_entities": len(signals.matched_entity_scores),
            "path_signal_score": path_signal_score,
            "cache_available": False,
            "cache_key": None,
            "candidate_modes": ["hybrid"],
        }

    def _adaptive_candidate_summary(
        self,
        question: str,
        signals: QuerySignals,
        *,
        mode: str,
        hits: list[RetrievalHit],
        retrieval_meta: dict[str, object],
    ) -> dict[str, object]:
        top_hits = hits[: min(4, len(hits))]
        avg_fused = sum(hit.fused_score for hit in top_hits) / max(len(top_hits), 1)
        article_alignment = self._article_hit_alignment(top_hits, signals)
        article_density = self._article_density(top_hits)
        document_alignment = self._document_hit_alignment(top_hits, signals)
        graph_density = self._graph_support_density(top_hits)
        evidence_sufficient = self.provider.judge_evidence(question, [hit.chunk.text for hit in hits])
        multi_hop = self._is_multi_hop_question(signals.normalized_question)

        arbitration_score = avg_fused * 0.45
        arbitration_score += article_alignment * 1.0
        arbitration_score += document_alignment * 0.8
        arbitration_score += article_density * (
            0.6 if ("which article" in signals.normalized_question or "provision" in signals.normalized_question) else 0.2
        )
        arbitration_score += graph_density * (0.85 if multi_hop else 0.25)
        if evidence_sufficient:
            arbitration_score += 0.35
        if retrieval_meta.get("cache_hit"):
            arbitration_score += 0.08

        selection_priority = 0.0
        if evidence_sufficient:
            selection_priority += 1.0
        if mode == "path_cache":
            selection_priority += article_alignment + graph_density
        if mode == "hybrid":
            selection_priority += avg_fused * 0.1

        return {
            "mode": mode,
            "avg_fused": round(avg_fused, 4),
            "article_alignment": round(article_alignment, 4),
            "article_density": round(article_density, 4),
            "document_alignment": round(document_alignment, 4),
            "graph_support_density": round(graph_density, 4),
            "evidence_sufficient": evidence_sufficient,
            "cache_hit": bool(retrieval_meta.get("cache_hit")),
            "latency_ms": round(float(retrieval_meta.get("total_latency_ms", 0.0)), 2),
            "arbitration_score": round(arbitration_score, 4),
            "selection_priority": round(selection_priority, 4),
            "top_chunk_ids": [hit.chunk.chunk_id for hit in top_hits[:3]],
        }

    def _load_catalog(self) -> dict:
        path = self.settings.processed_data_path / "graph" / "graph_catalog.json"
        if not path.exists():
            return {
                "documents": [],
                "chunks": [],
                "entities": [],
                "relations": [],
                "communities": {},
                "mentions": [],
            }
        return read_json(path)

    def _build_chunk_mentions(self) -> dict[str, set[str]]:
        chunk_ids_by_entity_id: dict[str, set[str]] = defaultdict(set)
        for mention in self.catalog.get("mentions", []):
            entity_id = mention.get("entity_id")
            chunk_id = mention.get("chunk_id")
            if entity_id and chunk_id and chunk_id in self.chunk_by_id:
                chunk_ids_by_entity_id[entity_id].add(chunk_id)
        for entity in self.entities:
            entity_id = entity.get("entity_id")
            chunk_id = entity.get("source_chunk_id")
            if entity_id and chunk_id and chunk_id in self.chunk_by_id:
                chunk_ids_by_entity_id[entity_id].add(chunk_id)
        return chunk_ids_by_entity_id

    def _lexical_scores(self, question: str, signals: QuerySignals) -> dict[str, float]:
        question_tokens = signals.content_tokens or signals.question_tokens
        scores: dict[str, float] = {}
        for chunk_id, chunk in self.chunk_by_id.items():
            chunk_tokens = self.chunk_tokens[chunk_id]
            overlap = len(question_tokens & chunk_tokens)
            score = overlap / max(len(question_tokens), 1)
            if chunk.article_ref and self._normalize_article_ref(chunk.article_ref) in signals.article_refs:
                score += 0.35
            scores[chunk_id] = score
        return scores

    def _metadata_scores(self, signals: QuerySignals) -> dict[str, float]:
        scores: dict[str, float] = {}
        has_article_hint = bool(signals.article_refs)
        has_document_hint = bool(signals.document_hints)
        for chunk_id, chunk in self.chunk_by_id.items():
            score = 0.0
            normalized_article_ref = self._normalize_article_ref(chunk.article_ref)
            if normalized_article_ref and normalized_article_ref in signals.article_refs:
                score += 1.7
            elif has_article_hint and chunk.article_ref:
                score -= 0.55

            document_name = self.document_label_by_id.get(chunk.document_id, "")
            document_hint = signals.document_hints.get(document_name, 0.0)
            if document_hint:
                score += 1.45 * document_hint
            elif has_document_hint:
                score -= 0.25

            section_title = str(chunk.metadata.get("section_title", ""))
            section_tokens = self._content_tokens(section_title)
            if section_tokens:
                score += 0.45 * (len(section_tokens & signals.content_tokens) / max(len(section_tokens), 1))

            if "high-risk" in signals.normalized_question and "high risk" in self._normalize_text(self.chunk_search_text[chunk_id]):
                score += 0.3
            scores[chunk_id] = score
        return scores

    def _graph_scores(self, signals: QuerySignals) -> tuple[dict[str, float], dict[str, list[GraphPath]]]:
        score_by_chunk: dict[str, float] = defaultdict(float)
        paths: dict[str, list[GraphPath]] = defaultdict(list)

        for entity_id, match_score in signals.matched_entity_scores.items():
            entity = self.entity_by_id.get(entity_id)
            if not entity:
                continue

            mention_score = 0.35 + match_score * 0.55
            for chunk_id in self.chunk_ids_by_entity_id.get(entity_id, {entity.get("source_chunk_id", "")}):
                if chunk_id not in self.chunk_by_id:
                    continue
                adjusted_score = mention_score * self._document_alignment(chunk_id, signals)
                score_by_chunk[chunk_id] += adjusted_score
                paths[chunk_id].append(
                    GraphPath(
                        seed_entity=entity["canonical_name"],
                        traversed_entities=[entity["canonical_name"]],
                        relation_chain=["mentions"],
                        score=adjusted_score,
                    )
                )

            frontier: list[tuple[str, list[str], list[str], int, float]] = [
                (entity_id, [entity["canonical_name"]], [], 0, match_score)
            ]
            visited: set[tuple[str, int]] = {(entity_id, 0)}
            while frontier:
                current_id, traversed_entities, relation_chain, depth, carry_score = frontier.pop(0)
                if depth >= self.settings.agent_max_graph_hops:
                    continue
                for relation in self.relations_by_subject.get(current_id, []):
                    target = self.entity_by_id.get(relation["object_entity_id"])
                    source_chunk_id = relation.get("source_chunk_id")
                    if not target or source_chunk_id not in self.chunk_by_id:
                        continue
                    target_name = str(target.get("canonical_name", ""))
                    target_match = self._entity_overlap_score(target, signals)
                    hop_score = (
                        float(relation.get("confidence", 0.4)) * max(carry_score, 0.3)
                        + target_match * 0.35
                    ) / (depth + 1)
                    hop_score *= self._document_alignment(source_chunk_id, signals)
                    score_by_chunk[source_chunk_id] += hop_score
                    paths[source_chunk_id].append(
                        GraphPath(
                            seed_entity=entity["canonical_name"],
                            traversed_entities=traversed_entities + [target_name],
                            relation_chain=relation_chain + [relation["relation_type"]],
                            score=hop_score,
                        )
                    )
                    state = (relation["object_entity_id"], depth + 1)
                    if state not in visited:
                        visited.add(state)
                        frontier.append(
                            (
                                relation["object_entity_id"],
                                traversed_entities + [target_name],
                                relation_chain + [relation["relation_type"]],
                                depth + 1,
                                max(target_match, carry_score * 0.85),
                            )
                        )
        return dict(score_by_chunk), dict(paths)

    def _retrieve_path_mode(self, question: str, signals: QuerySignals, *, top_k: int, mode: str) -> list[RetrievalHit]:
        cache_lookup_started = time.perf_counter()
        query_embedding = self.provider.embed_text(question)
        vector_scores = {
            chunk_id: cosine_similarity(query_embedding, embedding)
            for chunk_id, embedding in self.embeddings.items()
        }
        lexical_scores = self._lexical_scores(question, signals)
        metadata_scores = self._metadata_scores(signals)

        cache_key = self.path_cache.build_cache_key(
            question=question,
            retrieval_mode=mode,
            article_refs=sorted(signals.article_refs),
            document_hints=signals.document_hints,
            matched_entity_ids=sorted(signals.matched_entity_scores),
            hop_limit=self.settings.agent_max_graph_hops,
        )
        cache_entry = self.path_cache.load(cache_key) if mode == "path_cache" else None
        cache_hit = cache_entry is not None
        cache_lookup_ms = round((time.perf_counter() - cache_lookup_started) * 1000, 2)
        path_enumeration_ms = 0.0
        if cache_entry is None:
            enumeration_started = time.perf_counter()
            path_records = self._enumerate_path_records(signals)
            path_enumeration_ms = round((time.perf_counter() - enumeration_started) * 1000, 2)
            cache_entry = CacheEntry(
                cache_key=cache_key,
                retrieval_mode=mode,
                question_signature=stable_hash(question.strip().lower(), prefix="q"),
                top_chunk_ids=[],
                paths=path_records,
                metadata={
                    "cache_schema_version": self.path_cache.CACHE_SCHEMA_VERSION,
                    "matched_entities": len(signals.matched_entity_scores),
                    "article_refs": sorted(signals.article_refs),
                    "document_hints": signals.document_hints,
                },
            )
            if mode == "path_cache":
                self.path_cache.save(cache_entry)

        path_scores, graph_paths = self._aggregate_path_scores(cache_entry.paths, signals)
        path_rank = [chunk_id for chunk_id, _ in sorted(path_scores.items(), key=lambda item: item[1], reverse=True)]
        vector_rank = [chunk_id for chunk_id, _ in sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)]
        lexical_rank = [chunk_id for chunk_id, _ in sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)]
        metadata_rank = [
            chunk_id for chunk_id, _ in sorted(metadata_scores.items(), key=lambda item: item[1], reverse=True)
        ]

        fused = reciprocal_rank_fusion([metadata_rank, lexical_rank, vector_rank, path_rank])
        for chunk_id in self.chunk_by_id:
            fused[chunk_id] = (
                fused.get(chunk_id, 0.0)
                + metadata_scores.get(chunk_id, 0.0) * 1.55
                + lexical_scores.get(chunk_id, 0.0) * 0.18
                + path_scores.get(chunk_id, 0.0) * 0.35
            )

        hits = self._build_hits(
            fused,
            lexical_scores=lexical_scores,
            vector_scores=vector_scores,
            graph_scores=path_scores,
            metadata_scores=metadata_scores,
            graph_paths=graph_paths,
            top_k=top_k,
        )
        if mode == "path_cache" and hits:
            cache_entry.top_chunk_ids = [hit.chunk.chunk_id for hit in hits]
            self.path_cache.save(cache_entry)
        self.last_retrieval_meta = {
            "mode": mode,
            "matched_entities": len(signals.matched_entity_scores),
            "article_refs": sorted(signals.article_refs),
            "document_hints": signals.document_hints,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "path_count": len(cache_entry.paths),
            "cached_entries": self.path_cache.stats()["entries"],
            "cache_schema_version": self.path_cache.CACHE_SCHEMA_VERSION,
            "cache_lookup_ms": cache_lookup_ms,
            "path_enumeration_ms": path_enumeration_ms,
            "top_paths": [record.model_dump() for record in cache_entry.paths[:12]],
        }
        return hits

    def _enumerate_path_records(self, signals: QuerySignals) -> list[PathRecord]:
        path_records: list[PathRecord] = []
        path_ids_seen: set[str] = set()
        ranked_seeds = sorted(signals.matched_entity_scores.items(), key=lambda item: item[1], reverse=True)[:8]
        for seed_entity_id, seed_score in ranked_seeds:
            seed_entity = self.entity_by_id.get(seed_entity_id)
            if not seed_entity:
                continue
            seed_name = str(seed_entity.get("canonical_name", seed_entity_id))

            seed_chunk_ids = self._supporting_chunks_for_path(
                signals,
                source_chunk_id=seed_entity.get("source_chunk_id", ""),
                entity_id=seed_entity_id,
            )
            for chunk_id in seed_chunk_ids:
                if chunk_id not in self.chunk_by_id:
                    continue
                record = PathRecord(
                    path_id=stable_hash(f"{seed_entity_id}:{chunk_id}:seed", prefix="path"),
                    seed_entity_id=seed_entity_id,
                    seed_entity=seed_name,
                    traversed_entity_ids=[seed_entity_id],
                    traversed_entities=[seed_name],
                    relation_chain=["seed_mention"],
                    supporting_chunk_ids=[chunk_id],
                    terminal_chunk_id=chunk_id,
                    score=0.55 + seed_score * 0.4,
                    metadata={"depth": 0},
                )
                if record.path_id not in path_ids_seen:
                    path_ids_seen.add(record.path_id)
                    path_records.append(record)

            frontier: list[tuple[str, list[str], list[str], list[str], int, float]] = [
                (seed_entity_id, [seed_entity_id], [seed_name], [], 0, seed_score)
            ]
            visited: set[tuple[str, int]] = {(seed_entity_id, 0)}
            while frontier:
                current_id, traversed_ids, traversed_names, relation_chain, depth, carry_score = frontier.pop(0)
                if depth >= self.settings.agent_max_graph_hops:
                    continue

                outgoing = [
                    (relation, relation["object_entity_id"], relation["relation_type"])
                    for relation in self.relations_by_subject.get(current_id, [])
                ]
                incoming = [
                    (relation, relation["subject_entity_id"], f"inv:{relation['relation_type']}")
                    for relation in self.relations_by_object.get(current_id, [])
                ]
                for relation, next_entity_id, relation_label in [*outgoing, *incoming]:
                    next_entity = self.entity_by_id.get(next_entity_id)
                    if not next_entity:
                        continue
                    source_chunk_id = relation.get("source_chunk_id", "")
                    supporting_chunk_ids = self._supporting_chunks_for_path(
                        signals,
                        source_chunk_id=source_chunk_id,
                        entity_id=next_entity_id,
                    )
                    if not supporting_chunk_ids:
                        continue
                    next_name = str(next_entity.get("canonical_name", next_entity_id))
                    next_match = self._entity_overlap_score(next_entity, signals)
                    path_score = (
                        max(carry_score, 0.35) * 0.45
                        + float(relation.get("confidence", 0.4)) * 0.35
                        + next_match * 0.35
                    ) / (depth + 1)
                    path_id = stable_hash(
                        f"{seed_entity_id}:{'|'.join(traversed_ids + [next_entity_id])}:{relation_label}:{source_chunk_id}",
                        prefix="path",
                    )
                    record = PathRecord(
                        path_id=path_id,
                        seed_entity_id=seed_entity_id,
                        seed_entity=seed_name,
                        traversed_entity_ids=traversed_ids + [next_entity_id],
                        traversed_entities=traversed_names + [next_name],
                        relation_chain=relation_chain + [relation_label],
                        supporting_chunk_ids=supporting_chunk_ids,
                        terminal_chunk_id=source_chunk_id,
                        score=path_score,
                        metadata={"depth": depth + 1},
                    )
                    if record.path_id not in path_ids_seen:
                        path_ids_seen.add(record.path_id)
                        path_records.append(record)
                    state = (next_entity_id, depth + 1)
                    if state not in visited:
                        visited.add(state)
                        frontier.append(
                            (
                                next_entity_id,
                                traversed_ids + [next_entity_id],
                                traversed_names + [next_name],
                                relation_chain + [relation_label],
                                depth + 1,
                                max(next_match, carry_score * 0.85),
                            )
                        )
        return sorted(path_records, key=lambda item: item.score, reverse=True)[:64]

    def _aggregate_path_scores(
        self,
        path_records: list[PathRecord],
        signals: QuerySignals,
    ) -> tuple[dict[str, float], dict[str, list[GraphPath]]]:
        raw_scores_by_chunk: dict[str, list[float]] = defaultdict(list)
        graph_paths: dict[str, list[GraphPath]] = defaultdict(list)
        for record in path_records:
            for index, chunk_id in enumerate(record.supporting_chunk_ids or [record.terminal_chunk_id]):
                if chunk_id not in self.chunk_by_id:
                    continue
                supporting_weight = 1.0 if index == 0 else 0.82
                score = (
                    record.score
                    * supporting_weight
                    * self._document_alignment(chunk_id, signals)
                    * self._article_alignment(chunk_id, signals)
                )
                raw_scores_by_chunk[chunk_id].append(score)
                graph_paths[chunk_id].append(
                    GraphPath(
                        seed_entity=record.seed_entity,
                        traversed_entities=record.traversed_entities,
                        relation_chain=record.relation_chain,
                        score=score,
                    )
                )
        score_by_chunk: dict[str, float] = {}
        for chunk_id, scores in raw_scores_by_chunk.items():
            top_scores = sorted(scores, reverse=True)[:4]
            content_alignment = self._chunk_content_alignment(chunk_id, signals)
            score_by_chunk[chunk_id] = (sum(top_scores) / max(len(top_scores), 1)) * (1.0 + content_alignment * 0.35)
        return score_by_chunk, dict(graph_paths)

    def _article_hit_alignment(self, hits: list[RetrievalHit], signals: QuerySignals) -> float:
        if not hits:
            return 0.0
        if signals.article_refs:
            matches = 0
            for hit in hits:
                if self._normalize_article_ref(hit.chunk.article_ref) in signals.article_refs:
                    matches += 1
            return matches / len(hits)
        if "which article" in signals.normalized_question or "provision" in signals.normalized_question:
            return self._article_density(hits)
        return 0.0

    @staticmethod
    def _article_density(hits: list[RetrievalHit]) -> float:
        if not hits:
            return 0.0
        article_hits = sum(1 for hit in hits if hit.chunk.article_ref)
        return article_hits / len(hits)

    def _document_hit_alignment(self, hits: list[RetrievalHit], signals: QuerySignals) -> float:
        if not hits or not signals.document_hints:
            return 0.0
        dominant_document = max(signals.document_hints, key=signals.document_hints.get)
        matching_hits = sum(1 for hit in hits if hit.document_name == dominant_document)
        return matching_hits / len(hits)

    @staticmethod
    def _graph_support_density(hits: list[RetrievalHit]) -> float:
        if not hits:
            return 0.0
        graph_paths = sum(len(hit.graph_paths) for hit in hits)
        return min(graph_paths / (len(hits) * 4), 1.0)

    def _build_hits(
        self,
        fused: dict[str, float],
        *,
        lexical_scores: dict[str, float],
        vector_scores: dict[str, float],
        graph_scores: dict[str, float],
        metadata_scores: dict[str, float],
        graph_paths: dict[str, list[GraphPath]],
        top_k: int,
    ) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for chunk_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            chunk = self.chunk_by_id[chunk_id]
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    document_name=self.document_label_by_id.get(
                        chunk.document_id,
                        self.document_name_by_id.get(chunk.document_id, chunk.document_id),
                    ),
                    text_score=lexical_scores.get(chunk_id, 0.0),
                    vector_score=vector_scores.get(chunk_id, 0.0),
                    graph_score=graph_scores.get(chunk_id, 0.0),
                    metadata_score=metadata_scores.get(chunk_id, 0.0),
                    fused_score=score,
                    graph_paths=sorted(graph_paths.get(chunk_id, []), key=lambda item: item.score, reverse=True)[:8],
                )
            )
        return hits

    def _analyze_question(self, question: str) -> QuerySignals:
        normalized_question = self._normalize_text(question)
        question_tokens = set(tokenize(question))
        content_tokens = self._content_tokens(question)
        article_refs = {
            self._normalize_article_ref(match.group(0))
            for match in ARTICLE_REF_RE.finditer(question)
            if self._normalize_article_ref(match.group(0))
        }
        document_hints = self._infer_document_hints(normalized_question)
        matched_entity_scores = self._match_entities(normalized_question, content_tokens, article_refs, document_hints)
        return QuerySignals(
            normalized_question=normalized_question,
            question_tokens=question_tokens,
            content_tokens=content_tokens,
            article_refs=article_refs,
            document_hints=document_hints,
            matched_entity_scores=matched_entity_scores,
        )

    def _match_entities(
        self,
        normalized_question: str,
        content_tokens: set[str],
        article_refs: set[str],
        document_hints: dict[str, float],
    ) -> dict[str, float]:
        matched: dict[str, float] = {}
        for entity in self.entities:
            score = self._entity_match_score(entity, normalized_question, content_tokens, article_refs, document_hints)
            if score >= 0.75:
                matched[entity["entity_id"]] = score
        return matched

    def _entity_match_score(
        self,
        entity: dict,
        normalized_question: str,
        content_tokens: set[str],
        article_refs: set[str],
        document_hints: dict[str, float],
    ) -> float:
        best_score = 0.0
        canonical_name = str(entity.get("canonical_name", ""))
        entity_type = str(entity.get("entity_type", ""))
        if entity_type == "article":
            normalized_article_ref = self._normalize_article_ref(canonical_name)
            if normalized_article_ref and normalized_article_ref in article_refs:
                best_score = max(best_score, 1.8)
        if entity_type == "regulation":
            best_score = max(best_score, document_hints.get(canonical_name, 0.0) * 1.3)

        candidate_names = [canonical_name, *entity.get("aliases", [])]
        for candidate in candidate_names:
            normalized_candidate = self._normalize_text(str(candidate))
            if not normalized_candidate:
                continue
            if normalized_candidate in normalized_question:
                token_count = len(normalized_candidate.split())
                best_score = max(best_score, 0.7 + min(token_count, 4) * 0.15)
            candidate_tokens = self._content_tokens(str(candidate))
            if not candidate_tokens:
                continue
            overlap = len(candidate_tokens & content_tokens)
            if overlap:
                best_score = max(best_score, 0.45 + 0.45 * (overlap / len(candidate_tokens)))
        return best_score

    def _entity_overlap_score(self, entity: dict, signals: QuerySignals) -> float:
        return self._entity_match_score(
            entity,
            signals.normalized_question,
            signals.content_tokens,
            signals.article_refs,
            signals.document_hints,
        )

    def _infer_document_hints(self, normalized_question: str) -> dict[str, float]:
        hints: dict[str, float] = {}
        for document_name, rules in DOCUMENT_HINT_RULES.items():
            score = 0.0
            for phrase, weight in rules:
                normalized_phrase = self._normalize_text(phrase)
                if normalized_phrase and normalized_phrase in normalized_question:
                    score += weight
            if score > 0:
                hints[document_name] = min(score, 1.5)
        return hints

    def _document_alignment(self, chunk_id: str, signals: QuerySignals) -> float:
        if not signals.document_hints:
            return 1.0
        chunk = self.chunk_by_id[chunk_id]
        document_name = self.document_label_by_id.get(chunk.document_id, "")
        if document_name in signals.document_hints:
            return 1.2 + signals.document_hints[document_name] * 0.1
        return 0.7

    def _article_alignment(self, chunk_id: str, signals: QuerySignals) -> float:
        if not signals.article_refs:
            return 1.0
        chunk = self.chunk_by_id[chunk_id]
        normalized_article_ref = self._normalize_article_ref(chunk.article_ref)
        if normalized_article_ref and normalized_article_ref in signals.article_refs:
            return 1.5
        if chunk.article_ref:
            return 0.42
        return 0.68

    def _chunk_content_alignment(self, chunk_id: str, signals: QuerySignals) -> float:
        if not signals.content_tokens:
            return 0.0
        overlap = len(self.chunk_tokens[chunk_id] & signals.content_tokens)
        return overlap / max(len(signals.content_tokens), 1)

    def _supporting_chunks_for_path(
        self,
        signals: QuerySignals,
        *,
        source_chunk_id: str,
        entity_id: str,
    ) -> list[str]:
        candidate_ids = []
        if source_chunk_id:
            candidate_ids.append(source_chunk_id)
        candidate_ids.extend(sorted(self.chunk_ids_by_entity_id.get(entity_id, set())))

        unique_ids: list[str] = []
        seen: set[str] = set()
        for chunk_id in candidate_ids:
            if chunk_id in seen or chunk_id not in self.chunk_by_id:
                continue
            seen.add(chunk_id)
            unique_ids.append(chunk_id)

        ranked = sorted(
            unique_ids,
            key=lambda chunk_id: self._path_support_score(chunk_id, signals),
            reverse=True,
        )
        filtered = [chunk_id for chunk_id in ranked if self._path_support_score(chunk_id, signals) >= 0.55]
        selected = filtered[:3] or ranked[:1]
        return selected

    def _path_support_score(self, chunk_id: str, signals: QuerySignals) -> float:
        return (
            self._document_alignment(chunk_id, signals)
            * self._article_alignment(chunk_id, signals)
            * (1.0 + self._chunk_content_alignment(chunk_id, signals) * 0.5)
        )

    @staticmethod
    def _is_multi_hop_question(normalized_question: str) -> bool:
        return any(hint in normalized_question for hint in MULTI_HOP_HINTS)

    @staticmethod
    def _is_cross_regulation_question(signals: QuerySignals) -> bool:
        return len(signals.document_hints) > 1

    def _chunk_search_text(self, chunk: ChunkRecord) -> str:
        section_title = str(chunk.metadata.get("section_title", ""))
        document_name = self.document_label_by_id.get(chunk.document_id, "")
        return " ".join(
            part
            for part in (
                document_name,
                chunk.article_ref or "",
                section_title,
                chunk.text,
            )
            if part
        )

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @classmethod
    def _normalize_article_ref(cls, value: str | None) -> str:
        if not value:
            return ""
        match = ARTICLE_REF_RE.search(value)
        if not match:
            return ""
        article = re.sub(r"\s+", " ", match.group(0)).strip()
        return article.title()

    @classmethod
    def _canonical_document_name(cls, value: str) -> str:
        normalized = cls._normalize_text(value)
        return DOCUMENT_NAME_MAP.get(normalized, value.replace("_", " ").strip())

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in tokenize(text)
            if token not in STOPWORDS and (token.isdigit() or len(token) > 2)
        }
