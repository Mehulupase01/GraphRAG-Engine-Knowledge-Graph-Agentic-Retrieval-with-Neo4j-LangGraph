from __future__ import annotations

import unittest

from graphrag_engine.common.providers import (
    AnthropicProvider,
    GeminiProvider,
    HeuristicLLMProvider,
    OpenAIProvider,
    build_provider,
)
from graphrag_engine.common.settings import Settings
from graphrag_engine.runtime import GraphRAGRuntime


class ProviderSelectionTests(unittest.TestCase):
    def test_explicit_heuristic_backend_uses_heuristic_provider(self) -> None:
        settings = Settings(model_backend="heuristic")
        provider = build_provider(settings)
        self.assertIsInstance(provider, HeuristicLLMProvider)
        self.assertEqual(provider.describe()["provider"], "heuristic")

    def test_openai_compatible_base_url_selects_openai_provider(self) -> None:
        settings = Settings(
            model_backend="openai",
            openai_base_url="http://localhost:11434/v1",
            openai_api_key="",
        )
        provider = build_provider(settings)
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.describe()["provider"], "openai")

    def test_anthropic_backend_selects_anthropic_provider(self) -> None:
        settings = Settings(
            model_backend="anthropic",
            anthropic_api_key="test-key",
        )
        provider = build_provider(settings)
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider.describe()["provider"], "anthropic")

    def test_gemini_backend_selects_gemini_provider(self) -> None:
        settings = Settings(
            model_backend="gemini",
            gemini_api_key="test-key",
        )
        provider = build_provider(settings)
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.describe()["provider"], "gemini")

    def test_local_runtime_uses_lightweight_evaluation_provider(self) -> None:
        runtime = GraphRAGRuntime(Settings(model_backend="local"))
        evaluator = runtime.build_evaluator()
        self.assertEqual(evaluator.agent.provider.provider_name, "heuristic")
        self.assertEqual(evaluator.agent.retriever.provider.provider_name, "local")


if __name__ == "__main__":
    unittest.main()
