from __future__ import annotations

from graphrag_engine.common.models import EvaluationCase, QueryResponse
from graphrag_engine.common.providers import tokenize


def _keyword_coverage(case: EvaluationCase, response: QueryResponse) -> float:
    answer_tokens = set(tokenize(response.answer))
    keywords = set(token.lower() for token in case.expected_keywords)
    if not keywords:
        return 0.0
    matched = sum(1 for keyword in keywords if keyword.lower() in response.answer.lower() or keyword in answer_tokens)
    return matched / len(keywords)


def faithfulness(case: EvaluationCase, response: QueryResponse) -> float:
    if not response.citations:
        return 0.0
    citation_text = " ".join(citation.snippet.lower() for citation in response.citations)
    matched = sum(1 for keyword in case.expected_keywords if keyword.lower() in citation_text)
    return min(1.0, 0.4 + matched / max(len(case.expected_keywords), 1))


def context_precision(case: EvaluationCase, response: QueryResponse) -> float:
    if not response.citations:
        return 0.0
    relevant = 0
    for citation in response.citations:
        snippet = citation.snippet.lower()
        if any(keyword.lower() in snippet for keyword in case.expected_keywords):
            relevant += 1
    return relevant / len(response.citations)


def answer_relevancy(case: EvaluationCase, response: QueryResponse) -> float:
    return _keyword_coverage(case, response)


def multi_hop_accuracy(case: EvaluationCase, response: QueryResponse) -> float:
    article_hits = sum(
        1
        for article in case.expected_articles
        if article.lower() in response.answer.lower()
        or any(article.lower() == (citation.article_ref or "").lower() for citation in response.citations)
    )
    graph_bonus = 0.2 if len(response.graph_paths) >= 1 else 0.0
    return min(1.0, article_hits / max(len(case.expected_articles), 1) + graph_bonus)

