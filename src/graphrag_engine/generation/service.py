from __future__ import annotations

from graphrag_engine.common.models import Citation, QueryResponse, RetrievalHit
from graphrag_engine.common.providers import LLMProvider


class AnswerGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, question: str, hits: list[RetrievalHit], trace: list[dict]) -> QueryResponse:
        evidence = self._compose_evidence(hits, trace)
        payload = self.provider.generate_grounded_answer(question, evidence)
        citations = [
            Citation(
                chunk_id=hit.chunk.chunk_id,
                document_name=hit.document_name,
                article_ref=hit.chunk.article_ref,
                snippet=hit.chunk.text[:280],
                page_start=hit.chunk.page_start,
                page_end=hit.chunk.page_end,
                score_breakdown={
                    "text": round(hit.text_score, 4),
                    "vector": round(hit.vector_score, 4),
                    "graph": round(hit.graph_score, 4),
                    "metadata": round(hit.metadata_score, 4),
                    "fused": round(hit.fused_score, 4),
                },
            )
            for hit in hits[:5]
        ]
        return QueryResponse(
            answer=payload["answer"],
            citations=citations,
            graph_paths=[path for hit in hits[:5] for path in hit.graph_paths],
            retrieved_chunks=[hit.chunk for hit in hits],
            retrieval_scores={
                "max_fused": round(max((hit.fused_score for hit in hits), default=0.0), 4),
                "avg_fused": round(sum(hit.fused_score for hit in hits) / max(len(hits), 1), 4),
            },
            confidence=float(payload.get("confidence", 0.0)),
            fallback_used=bool(payload.get("fallback_used", False)),
            trace=trace,
        )

    def _compose_evidence(self, hits: list[RetrievalHit], trace: list[dict]) -> list[str]:
        path_evidence = self._path_evidence_from_trace(trace)
        chunk_evidence = [self._chunk_evidence(hit) for hit in hits]
        return [*chunk_evidence, *path_evidence]

    @staticmethod
    def _chunk_evidence(hit: RetrievalHit) -> str:
        article_ref = hit.chunk.article_ref or "No article reference"
        page_span = f"{hit.chunk.page_start or '?'}-{hit.chunk.page_end or '?'}"
        return f"Source evidence ({hit.document_name}, {article_ref}, pages {page_span}): {hit.chunk.text}"

    def _path_evidence_from_trace(self, trace: list[dict]) -> list[str]:
        path_rows: list[str] = []
        for event in trace:
            if event.get("step") != "retrieve":
                continue
            retrieval_meta = event.get("retrieval_meta", {})
            if not isinstance(retrieval_meta, dict):
                continue
            for item in retrieval_meta.get("top_paths", [])[:6]:
                if not isinstance(item, dict):
                    continue
                traversed_entities = item.get("traversed_entities", [])
                relation_chain = item.get("relation_chain", [])
                supporting_chunk_ids = item.get("supporting_chunk_ids", [])
                chain = " -> ".join(str(entity) for entity in traversed_entities if entity)
                relations = " | ".join(str(relation) for relation in relation_chain if relation)
                support = ", ".join(str(chunk_id) for chunk_id in supporting_chunk_ids[:3])
                score = float(item.get("score", 0.0))
                if chain:
                    path_rows.append(
                        f"Path evidence: {chain}. Relations: {relations or 'none'}. Supporting chunks: {support or 'none'}. Path score: {score:.4f}."
                    )
        return path_rows
