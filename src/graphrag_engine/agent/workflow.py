from __future__ import annotations

import time

from graphrag_engine.common.models import QueryRequest, QueryResponse
from graphrag_engine.common.providers import LLMProvider
from graphrag_engine.common.settings import Settings
from graphrag_engine.generation.service import AnswerGenerator
from graphrag_engine.retrieval.service import HybridRetriever

try:
    from langgraph.graph import END, StateGraph  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    END = None
    StateGraph = None


class GraphRAGAgent:
    def __init__(self, settings: Settings, provider: LLMProvider, retriever: HybridRetriever) -> None:
        self.settings = settings
        self.provider = provider
        self.retriever = retriever
        self.generator = AnswerGenerator(provider)

    def run(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        trace: list[dict] = []
        question = request.question.strip()
        rewritten_question = question
        hits = []

        for rewrite_index in range(self.settings.agent_max_rewrites + 1):
            resolved_mode = request.retrieval_mode
            if request.retrieval_mode == "adaptive":
                routing = self.retriever.resolve_mode(rewritten_question, requested_mode=request.retrieval_mode)
                resolved_mode = str(routing.get("resolved_mode", "hybrid"))
                trace.append(
                    {
                        "step": "route",
                        "attempt": rewrite_index,
                        "question": rewritten_question,
                        **routing,
                    }
                )
            hits = self.retriever.retrieve(
                rewritten_question,
                top_k=request.top_k,
                mode=resolved_mode,
            )
            evidence = [hit.chunk.text for hit in hits]
            sufficient = self.provider.judge_evidence(question, evidence)
            trace.append(
                {
                    "step": "retrieve",
                    "attempt": rewrite_index,
                    "question": rewritten_question,
                    "hit_count": len(hits),
                    "top_chunk_ids": [hit.chunk.chunk_id for hit in hits[:3]],
                    "sufficient": sufficient,
                    "requested_mode": request.retrieval_mode,
                    "resolved_mode": resolved_mode,
                    "retrieval_meta": getattr(self.retriever, "last_retrieval_meta", {}),
                }
            )
            if sufficient or rewrite_index >= self.settings.agent_max_rewrites:
                break
            rewritten_question = self.provider.rewrite_query(question, evidence[:3])
            trace.append(
                {
                    "step": "rewrite_query",
                    "attempt": rewrite_index,
                    "rewritten_question": rewritten_question,
                }
            )

        response = self.generator.generate(question, hits, trace)
        total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response.retrieval_scores["total_query_ms"] = total_latency_ms
        trace.append(
            {
                "step": "generate",
                "answer_length": len(response.answer),
                "total_query_ms": total_latency_ms,
            }
        )
        if not response.citations:
            response.fallback_used = True
        return response

    def build_langgraph(self):  # pragma: no cover - optional dependency
        if StateGraph is None or END is None:
            raise RuntimeError("langgraph is not installed")

        class AgentState(dict):
            pass

        workflow = StateGraph(AgentState)

        def retrieve(state: AgentState) -> AgentState:
            request: QueryRequest = state["request"]
            resolved_mode = request.retrieval_mode
            if request.retrieval_mode == "adaptive":
                routing = self.retriever.resolve_mode(
                    state.get("question", request.question),
                    requested_mode=request.retrieval_mode,
                )
                resolved_mode = str(routing.get("resolved_mode", "hybrid"))
                state.setdefault("trace", []).append(
                    {
                        "step": "route",
                        "attempt": state.get("attempt", 0),
                        "question": state.get("question", request.question),
                        **routing,
                    }
                )
            hits = self.retriever.retrieve(
                state.get("question", request.question),
                top_k=request.top_k,
                mode=resolved_mode,
            )
            trace = state.setdefault("trace", [])
            evidence = [hit.chunk.text for hit in hits]
            sufficient = self.provider.judge_evidence(request.question, evidence)
            trace.append(
                {
                    "step": "retrieve",
                    "attempt": state.get("attempt", 0),
                    "question": state.get("question", request.question),
                    "hit_count": len(hits),
                    "sufficient": sufficient,
                    "requested_mode": request.retrieval_mode,
                    "resolved_mode": resolved_mode,
                    "retrieval_meta": getattr(self.retriever, "last_retrieval_meta", {}),
                }
            )
            state["hits"] = hits
            state["sufficient"] = sufficient
            return state

        def rewrite(state: AgentState) -> AgentState:
            hits = state.get("hits", [])
            evidence = [hit.chunk.text for hit in hits]
            state["attempt"] = state.get("attempt", 0) + 1
            state["question"] = self.provider.rewrite_query(state["request"].question, evidence[:3])
            state.setdefault("trace", []).append(
                {
                    "step": "rewrite_query",
                    "attempt": state["attempt"],
                    "rewritten_question": state["question"],
                }
            )
            return state

        def generate(state: AgentState) -> AgentState:
            state["response"] = self.generator.generate(
                state["request"].question,
                state.get("hits", []),
                state.get("trace", []),
            )
            return state

        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("generate", generate)
        workflow.set_entry_point("retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            lambda state: "generate"
            if state.get("sufficient") or state.get("attempt", 0) >= self.settings.agent_max_rewrites
            else "rewrite",
        )
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", END)
        return workflow.compile()
