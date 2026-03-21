from __future__ import annotations

import statistics
import time
from pathlib import Path

from graphrag_engine.agent.workflow import GraphRAGAgent
from graphrag_engine.common.artifacts import read_json, write_json
from graphrag_engine.common.hashing import stable_hash
from graphrag_engine.common.models import EvaluationCase, EvaluationResult, EvaluationSummary, QueryRequest
from graphrag_engine.common.settings import Settings

from .dataset import default_eval_cases, ensure_eval_fixture
from .metrics import answer_relevancy, context_precision, faithfulness, multi_hop_accuracy


class EvaluationService:
    def __init__(self, settings: Settings, agent: GraphRAGAgent) -> None:
        self.settings = settings
        self.agent = agent

    def run(self) -> EvaluationSummary:
        cases = self._load_cases()
        run_id = stable_hash(str(len(cases)), prefix="eval")
        results: list[EvaluationResult] = []
        scores_by_mode: dict[str, list[float]] = {mode: [] for mode in self._evaluation_modes()}
        latencies_by_mode: dict[str, list[float]] = {mode: [] for mode in self._evaluation_modes()}
        retrieval_latencies_by_mode: dict[str, list[float]] = {mode: [] for mode in self._evaluation_modes()}
        cache_hits_by_mode: dict[str, int] = {mode: 0 for mode in self._evaluation_modes()}
        path_cache_hits = 0
        path_cache_runs = 0

        for case in cases:
            for mode in self._evaluation_modes():
                should_warm = False
                if mode == "path_cache":
                    should_warm = True
                elif mode == "adaptive":
                    routing = self.agent.retriever.resolve_mode(case.question, requested_mode=mode)
                    should_warm = str(routing.get("resolved_mode")) == "path_cache"
                if should_warm:
                    # Warm the cache first so this path captures actual cache-aware behavior.
                    self.agent.run(
                        QueryRequest(question=case.question, retrieval_mode=mode, top_k=self.settings.default_retrieval_k)
                    )
                started = time.perf_counter()
                response = self.agent.run(
                    QueryRequest(question=case.question, retrieval_mode=mode, top_k=self.settings.default_retrieval_k)
                )
                total_latency_ms = float(response.retrieval_scores.get("total_query_ms", round((time.perf_counter() - started) * 1000, 2)))
                retrieval_latency_ms = self._retrieval_latency_ms(response)
                cache_hit = self._cache_hit(response)
                result = self._build_result(
                    case,
                    response,
                    approach=mode,
                    metadata={
                        "latency_ms": round(total_latency_ms, 2),
                        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                        "cache_hit": cache_hit,
                    },
                )
                results.append(result)
                scores_by_mode[mode].append(result.score)
                latencies_by_mode[mode].append(total_latency_ms)
                retrieval_latencies_by_mode[mode].append(retrieval_latency_ms)
                if cache_hit:
                    cache_hits_by_mode[mode] += 1
                if mode == "path_cache":
                    path_cache_runs += 1
                    if cache_hit:
                        path_cache_hits += 1

        baseline_avg = statistics.fmean(scores_by_mode.get("baseline", [])) if scores_by_mode.get("baseline") else 0.0
        summary = EvaluationSummary(
            run_id=run_id,
            total_cases=len(cases),
            aggregate_scores=self._aggregate_results(results, latencies_by_mode, retrieval_latencies_by_mode, cache_hits_by_mode),
            cases=results,
            regressions=self._regression_messages(scores_by_mode, baseline_avg),
            metadata={
                "evaluation_provider": self.agent.provider.provider_name,
                "retrieval_provider": self.agent.retriever.provider.provider_name,
                "baseline_average": round(baseline_avg, 4),
                **{
                    f"{mode}_average": round(statistics.fmean(values), 4)
                    for mode, values in scores_by_mode.items()
                    if values
                },
                **{
                    f"{mode}_latency_ms": round(statistics.fmean(values), 2)
                    for mode, values in latencies_by_mode.items()
                    if values
                },
                **{
                    f"{mode}_retrieval_latency_ms": round(statistics.fmean(values), 2)
                    for mode, values in retrieval_latencies_by_mode.items()
                    if values
                },
                "path_cache_hit_rate": round(path_cache_hits / path_cache_runs, 4) if path_cache_runs else 0.0,
            },
        )
        write_json(self.settings.processed_data_path / "evaluation" / f"{run_id}.json", summary.model_dump())
        return summary

    def _load_cases(self) -> list[EvaluationCase]:
        fixture_path = ensure_eval_fixture((Path.cwd() / self.settings.eval_cases_path).resolve())
        if fixture_path.exists():
            return [EvaluationCase.model_validate(item) for item in read_json(fixture_path)]
        return default_eval_cases()

    def _build_result(
        self,
        case: EvaluationCase,
        response,
        *,
        approach: str,
        metadata: dict[str, float | bool] | None = None,
    ) -> EvaluationResult:
        faith = faithfulness(case, response)
        precision = context_precision(case, response)
        relevancy = answer_relevancy(case, response)
        multi_hop = multi_hop_accuracy(case, response)
        score = round((faith + precision + relevancy + multi_hop) / 4, 4)
        return EvaluationResult(
            case_id=case.case_id,
            approach=approach,
            score=score,
            faithfulness=round(faith, 4),
            context_precision=round(precision, 4),
            answer_relevancy=round(relevancy, 4),
            multi_hop_accuracy=round(multi_hop, 4),
            notes=[f"difficulty={case.difficulty}"],
            metadata=metadata or {},
            response=response,
        )

    @staticmethod
    def _aggregate_results(
        results: list[EvaluationResult],
        latencies_by_mode: dict[str, list[float]],
        retrieval_latencies_by_mode: dict[str, list[float]],
        cache_hits_by_mode: dict[str, int],
    ) -> dict[str, dict[str, float]]:
        aggregate_scores: dict[str, dict[str, float]] = {}
        for approach in sorted({result.approach for result in results}):
            scoped = [result for result in results if result.approach == approach]
            aggregate_scores[approach] = {
                "average_score": round(statistics.fmean(result.score for result in scoped), 4),
                "faithfulness": round(statistics.fmean(result.faithfulness for result in scoped), 4),
                "context_precision": round(statistics.fmean(result.context_precision for result in scoped), 4),
                "answer_relevancy": round(statistics.fmean(result.answer_relevancy for result in scoped), 4),
                "multi_hop_accuracy": round(statistics.fmean(result.multi_hop_accuracy for result in scoped), 4),
                "avg_latency_ms": round(statistics.fmean(latencies_by_mode.get(approach, [0.0])), 2),
                "avg_retrieval_latency_ms": round(statistics.fmean(retrieval_latencies_by_mode.get(approach, [0.0])), 2),
                "cache_hit_rate": round(
                    cache_hits_by_mode.get(approach, 0) / max(len(scoped), 1),
                    4,
                ),
            }
        return aggregate_scores

    @staticmethod
    def _evaluation_modes() -> list[str]:
        return ["baseline", "hybrid", "path_hybrid", "path_cache", "adaptive"]

    @staticmethod
    def _regression_messages(scores_by_mode: dict[str, list[float]], baseline_avg: float) -> list[str]:
        regressions: list[str] = []
        for mode, values in scores_by_mode.items():
            if mode == "baseline" or not values:
                continue
            average = statistics.fmean(values)
            if average < baseline_avg:
                regressions.append(f"{mode} underperformed baseline.")
        return regressions

    @staticmethod
    def _retrieval_latency_ms(response) -> float:
        for event in response.trace:
            if event.get("step") != "retrieve":
                continue
            retrieval_meta = event.get("retrieval_meta", {})
            if isinstance(retrieval_meta, dict):
                return float(retrieval_meta.get("total_latency_ms", 0.0))
        return 0.0

    @staticmethod
    def _cache_hit(response) -> bool:
        for event in response.trace:
            if event.get("step") != "retrieve":
                continue
            retrieval_meta = event.get("retrieval_meta", {})
            if isinstance(retrieval_meta, dict):
                return bool(retrieval_meta.get("cache_hit"))
        return False
