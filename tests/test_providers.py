from __future__ import annotations

import unittest

from graphrag_engine.common.providers import HeuristicLLMProvider, OpenAIProvider, build_provider
from graphrag_engine.common.settings import Settings


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


if __name__ == "__main__":
    unittest.main()

