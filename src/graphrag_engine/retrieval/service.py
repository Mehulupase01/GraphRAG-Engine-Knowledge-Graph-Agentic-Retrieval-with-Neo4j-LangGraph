from __future__ import annotations

import math
from collections import defaultdict

from graphrag_engine.common.artifacts import read_json
from graphrag_engine.common.models import ChunkRecord, GraphPath, RetrievalHit
from graphrag_engine.common.providers import LLMProvider, tokenize
from graphrag_engine.common.settings import Settings

from .fusion import reciprocal_rank_fusion


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
        self.catalog = self._load_catalog()
        self.chunk_by_id = {
            chunk["chunk_id"]: ChunkRecord.model_validate(chunk)
            for chunk in self.catalog.get("chunks", [])
        }
        self.document_name_by_id = {
            doc["document_id"]: doc["name"] for doc in self.catalog.get("documents", [])
        }
        self.entities = self.catalog.get("entities", [])
        self.relations = self.catalog.get("relations", [])
        self.embeddings = {
            chunk_id: provider.embed_text(chunk.text)
            for chunk_id, chunk in self.chunk_by_id.items()
        }

    def retrieve(self, question: str, *, top_k: int | None = None, mode: str = "hybrid") -> list[RetrievalHit]:
        top_k = top_k or self.settings.default_retrieval_k
        query_embedding = self.provider.embed_text(question)
        vector_scores = {
            chunk_id: cosine_similarity(query_embedding, embedding)
            for chunk_id, embedding in self.embeddings.items()
        }
        lexical_scores = self._lexical_scores(question)
        graph_scores, graph_paths = self._graph_scores(question)

        vector_rank = [chunk_id for chunk_id, _ in sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)]
        lexical_rank = [chunk_id for chunk_id, _ in sorted(lexical_scores.items(), key=lambda item: item[1], reverse=True)]
        graph_rank = [chunk_id for chunk_id, _ in sorted(graph_scores.items(), key=lambda item: item[1], reverse=True)]

        if mode == "baseline":
            fused = {
                chunk_id: score for chunk_id, score in sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)
            }
        else:
            fused = reciprocal_rank_fusion([vector_rank, lexical_rank, graph_rank])

        hits: list[RetrievalHit] = []
        for chunk_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            chunk = self.chunk_by_id[chunk_id]
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    document_name=self.document_name_by_id.get(chunk.document_id, chunk.document_id),
                    text_score=lexical_scores.get(chunk_id, 0.0),
                    vector_score=vector_scores.get(chunk_id, 0.0),
                    graph_score=graph_scores.get(chunk_id, 0.0),
                    fused_score=score,
                    graph_paths=graph_paths.get(chunk_id, []),
                )
            )
        return hits

    def _load_catalog(self) -> dict:
        path = self.settings.processed_data_path / "graph" / "graph_catalog.json"
        if not path.exists():
            return {"documents": [], "chunks": [], "entities": [], "relations": [], "communities": {}}
        return read_json(path)

    def _lexical_scores(self, question: str) -> dict[str, float]:
        question_tokens = set(tokenize(question))
        scores: dict[str, float] = {}
        for chunk_id, chunk in self.chunk_by_id.items():
            chunk_tokens = set(tokenize(chunk.text))
            overlap = len(question_tokens & chunk_tokens)
            scores[chunk_id] = overlap / max(len(question_tokens), 1)
        return scores

    def _graph_scores(self, question: str) -> tuple[dict[str, float], dict[str, list[GraphPath]]]:
        matched_entities = [
            entity
            for entity in self.entities
            if any(token in entity.get("canonical_name", "").lower() for token in tokenize(question))
        ]
        score_by_chunk: dict[str, float] = defaultdict(float)
        paths: dict[str, list[GraphPath]] = defaultdict(list)
        relations_by_subject: dict[str, list[dict]] = defaultdict(list)
        entity_by_id = {entity["entity_id"]: entity for entity in self.entities}
        for relation in self.relations:
            relations_by_subject[relation["subject_entity_id"]].append(relation)

        for entity in matched_entities:
            traversed = [entity["canonical_name"]]
            for relation in relations_by_subject.get(entity["entity_id"], [])[: self.settings.agent_max_graph_hops + 1]:
                target = entity_by_id.get(relation["object_entity_id"])
                if not target:
                    continue
                path = GraphPath(
                    seed_entity=entity["canonical_name"],
                    traversed_entities=traversed + [target["canonical_name"]],
                    relation_chain=[relation["relation_type"]],
                    score=relation.get("confidence", 0.4),
                )
                score_by_chunk[relation["source_chunk_id"]] += relation.get("confidence", 0.4)
                paths[relation["source_chunk_id"]].append(path)
        return dict(score_by_chunk), dict(paths)

