from __future__ import annotations

from pathlib import Path

from graphrag_engine.common.artifacts import write_json
from graphrag_engine.common.models import EvaluationCase


def default_eval_cases() -> list[EvaluationCase]:
    acts = [
        ("AI Act", ["high-risk", "provider", "conformity assessment"], ["Article 6"]),
        ("GDPR", ["data subject", "processing", "lawful basis"], ["Article 6"]),
        ("Digital Services Act", ["platform", "risk assessment", "transparency"], ["Article 34"]),
    ]
    prompts = [
        "Which obligations apply when {act} classifies a system as high-risk?",
        "How does {act} define transparency responsibilities for providers or platforms?",
        "Which article in {act} is relevant when enforcement depends on documented risk assessments?",
        "How do obligations in {act} interact with deployer responsibilities and record keeping?",
        "What evidence should be cited from {act} when explaining compliance duties?",
        "Which provision in {act} establishes the main legal basis for the responsibility being asked about?",
    ]

    cases: list[EvaluationCase] = []
    counter = 1
    for act, keywords, articles in acts:
        for template in prompts:
            for difficulty in ("medium", "hard", "multi-hop"):
                cases.append(
                    EvaluationCase(
                        case_id=f"case-{counter:03d}",
                        question=template.format(act=act),
                        expected_keywords=keywords,
                        expected_articles=articles,
                        difficulty=difficulty,
                        metadata={"act": act},
                    )
                )
                counter += 1
    return cases


def ensure_eval_fixture(path: Path) -> Path:
    if not path.exists():
        write_json(path, [case.model_dump() for case in default_eval_cases()])
    return path

