from __future__ import annotations

from graphrag_engine.common.models import Citation, QueryResponse, RetrievalHit
from graphrag_engine.common.providers import LLMProvider


class AnswerGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, question: str, hits: list[RetrievalHit], trace: list[dict]) -> QueryResponse:
        evidence = [hit.chunk.text for hit in hits]
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

