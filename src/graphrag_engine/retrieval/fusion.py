from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for index, item in enumerate(ranking, start=1):
            fused[item] += 1.0 / (k + index)
    return dict(fused)

