from __future__ import annotations

from graphrag_engine.agent.workflow import GraphRAGAgent
from graphrag_engine.common.logging import configure_logging
from graphrag_engine.common.providers import build_provider
from graphrag_engine.common.settings import Settings
from graphrag_engine.evaluation.service import EvaluationService
from graphrag_engine.extraction.service import ExtractionService
from graphrag_engine.graph.loader import GraphLoader
from graphrag_engine.ingestion.service import IngestionService
from graphrag_engine.retrieval.service import HybridRetriever


class GraphRAGRuntime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        configure_logging(self.settings.log_level)
        self.provider = build_provider(self.settings)
        self.ingestion = IngestionService(self.settings)
        self.extraction = ExtractionService(self.settings, self.provider)
        self.graph = GraphLoader(self.settings)

    def build_retriever(self) -> HybridRetriever:
        return HybridRetriever(self.settings, self.provider)

    def build_agent(self) -> GraphRAGAgent:
        return GraphRAGAgent(self.settings, self.provider, self.build_retriever())

    def build_evaluator(self) -> EvaluationService:
        return EvaluationService(self.settings, self.build_agent())

