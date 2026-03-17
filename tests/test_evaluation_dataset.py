from __future__ import annotations

import unittest

from graphrag_engine.evaluation.dataset import default_eval_cases


class EvaluationDatasetTests(unittest.TestCase):
    def test_default_fixture_has_at_least_fifty_cases(self) -> None:
        self.assertGreaterEqual(len(default_eval_cases()), 50)


if __name__ == "__main__":
    unittest.main()

