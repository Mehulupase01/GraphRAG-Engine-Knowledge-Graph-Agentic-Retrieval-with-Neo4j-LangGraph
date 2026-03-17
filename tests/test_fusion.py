from __future__ import annotations

import unittest

from graphrag_engine.retrieval.fusion import reciprocal_rank_fusion


class FusionTests(unittest.TestCase):
    def test_rrf_prefers_items_ranked_consistently(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"], ["b", "c", "a"]])
        self.assertGreater(fused["b"], fused["a"])


if __name__ == "__main__":
    unittest.main()

