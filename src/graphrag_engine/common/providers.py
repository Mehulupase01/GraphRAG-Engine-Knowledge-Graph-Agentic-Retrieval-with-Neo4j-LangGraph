from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

from .hashing import stable_hash
from .models import ChunkRecord
from .settings import Settings

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from transformers import AutoTokenizer, pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    pipeline = None


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/]+")
ENTITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bAI Act\b", "regulation"),
    (r"\bGDPR\b", "regulation"),
    (r"\bDigital Services Act\b", "regulation"),
    (r"\bArticle\s+\d+[A-Za-z-]*\b", "article"),
    (r"\bprovider[s]?\b", "actor"),
    (r"\bdeployer[s]?\b", "actor"),
    (r"\bhigh-risk\b", "risk_class"),
    (r"\bminimal risk\b", "risk_class"),
    (r"\blimited risk\b", "risk_class"),
    (r"\bunacceptable risk\b", "risk_class"),
    (r"\bconformity assessment\b", "obligation"),
    (r"\btransparency obligation[s]?\b", "obligation"),
]
RELATION_KEYWORDS = {
    "requires": "requires",
    "shall": "requires",
    "must": "requires",
    "prohibits": "prohibits",
    "forbidden": "prohibits",
    "defines": "defines",
    "means": "defines",
    "applies to": "applies_to",
    "includes": "includes",
}


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _local_stack_available() -> bool:
    return AutoTokenizer is not None and pipeline is not None and SentenceTransformer is not None


class LLMProvider(ABC):
    provider_name = "abstract"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def extract_structured_knowledge(self, chunk: ChunkRecord) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def rewrite_query(self, question: str, context: list[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def judge_evidence(self, question: str, evidence: list[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_grounded_answer(self, question: str, evidence: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider_name}


class HeuristicLLMProvider(LLMProvider):
    provider_name = "heuristic"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        dimension = self.settings.vector_dimension
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * dimension
            for token in tokenize(text):
                slot = int(stable_hash(token, length=8), 16) % dimension
                vector[slot] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def extract_structured_knowledge(self, chunk: ChunkRecord) -> dict[str, Any]:
        text = chunk.text
        entities: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for pattern, entity_type in ENTITY_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                raw_name = match.group(0).strip()
                canonical = re.sub(r"\s+", " ", raw_name).strip().title()
                key = canonical.lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                entities.append(
                    {
                        "raw_name": raw_name,
                        "canonical_name": canonical,
                        "entity_type": entity_type,
                        "confidence": 0.65,
                        "evidence": [raw_name],
                    }
                )

        relations: list[dict[str, Any]] = []
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        for sentence in sentences:
            sentence_entities = [
                entity for entity in entities if entity["raw_name"].lower() in sentence.lower()
            ]
            if len(sentence_entities) < 2:
                continue
            relation_type = "references"
            for keyword, mapped in RELATION_KEYWORDS.items():
                if keyword in sentence.lower():
                    relation_type = mapped
                    break
            relations.append(
                {
                    "subject": sentence_entities[0]["canonical_name"],
                    "object": sentence_entities[1]["canonical_name"],
                    "relation_type": relation_type,
                    "confidence": 0.55,
                    "evidence": [sentence[:240]],
                }
            )
        return {"entities": entities, "relations": relations}

    def rewrite_query(self, question: str, context: list[str]) -> str:
        if not context:
            return question
        keywords = Counter(token for item in context for token in tokenize(item))
        top_terms = [term for term, _ in keywords.most_common(4)]
        suffix = " ".join(term for term in top_terms if term not in question.lower())
        return f"{question} {suffix}".strip()

    def judge_evidence(self, question: str, evidence: list[str]) -> bool:
        question_tokens = set(tokenize(question))
        evidence_tokens = set(token for item in evidence for token in tokenize(item))
        coverage = len(question_tokens & evidence_tokens) / max(len(question_tokens), 1)
        return len(evidence) >= 2 and coverage >= 0.25

    def generate_grounded_answer(self, question: str, evidence: list[str]) -> dict[str, Any]:
        if not evidence:
            return {
                "answer": "I could not find enough grounded evidence in the indexed corpus to answer this yet.",
                "confidence": 0.15,
                "fallback_used": True,
            }
        supporting_lines = []
        for snippet in evidence[:3]:
            cleaned = " ".join(snippet.split())
            supporting_lines.append(cleaned[:260])
        answer = (
            f"Question: {question}\n"
            "Grounded synthesis:\n"
            + "\n".join(f"- {line}" for line in supporting_lines)
        )
        return {"answer": answer, "confidence": min(0.35 + len(evidence) * 0.1, 0.85), "fallback_used": False}


class OpenAIProvider(HeuristicLLMProvider):
    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if OpenAI is None:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is not installed")
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key or "local-dev"}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**kwargs)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "chat_model": self.settings.chat_model,
            "embedding_model": self.settings.embedding_model,
            "base_url": self.settings.openai_base_url or "default",
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - external API
        if not (self.settings.openai_api_key or self.settings.openai_base_url):
            return super().embed_texts(texts)
        try:
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception:
            return super().embed_texts(texts)

    def extract_structured_knowledge(self, chunk: ChunkRecord) -> dict[str, Any]:  # pragma: no cover - external API
        if not (self.settings.openai_api_key or self.settings.openai_base_url):
            return super().extract_structured_knowledge(chunk)
        prompt = (
            "Extract entities and relations from this regulatory chunk. "
            "Return strict JSON with keys entities and relations.\n\n"
            f"Chunk:\n{chunk.text[:4000]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {"role": "system", "content": "You extract structured knowledge from legal documents."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            payload = response.choices[0].message.content or "{}"
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return super().extract_structured_knowledge(chunk)

    def rewrite_query(self, question: str, context: list[str]) -> str:  # pragma: no cover - external API
        if not (self.settings.openai_api_key or self.settings.openai_base_url):
            return super().rewrite_query(question, context)
        prompt = (
            "Rewrite this user question for better multi-hop legal retrieval. "
            "Keep it concise and preserve intent.\n\n"
            f"Question: {question}\nContext: {context[:3]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            return text.strip() or super().rewrite_query(question, context)
        except Exception:
            return super().rewrite_query(question, context)

    def judge_evidence(self, question: str, evidence: list[str]) -> bool:  # pragma: no cover - external API
        return super().judge_evidence(question, evidence)

    def generate_grounded_answer(self, question: str, evidence: list[str]) -> dict[str, Any]:  # pragma: no cover - external API
        if not (self.settings.openai_api_key or self.settings.openai_base_url) or not evidence:
            return super().generate_grounded_answer(question, evidence)
        prompt = (
            "Answer the question using only the evidence snippets. "
            "If evidence is insufficient, say so. Keep citations implicit; the system will attach them.\n\n"
            f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence[:6])
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            answer = response.choices[0].message.content or ""
            if answer.strip():
                return {"answer": answer.strip(), "confidence": 0.82, "fallback_used": False}
        except Exception:
            pass
        return super().generate_grounded_answer(question, evidence)


class LocalTransformersProvider(HeuristicLLMProvider):
    provider_name = "local"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not _local_stack_available():  # pragma: no cover - optional dependency
            raise RuntimeError("transformers and sentence-transformers are required for local model support")
        self._tokenizer = None
        self._generator = None
        self._embedder = None

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "chat_model": self.settings.local_chat_model,
            "embedding_model": self.settings.local_embedding_model,
            "device": self.settings.local_device,
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if SentenceTransformer is None:
            return super().embed_texts(texts)
        try:
            embedder = self._get_embedder()
            vectors = embedder.encode(texts, normalize_embeddings=True)
            return [list(map(float, vector)) for vector in vectors]
        except Exception:
            return super().embed_texts(texts)

    def extract_structured_knowledge(self, chunk: ChunkRecord) -> dict[str, Any]:
        prompt = (
            "Extract entities and relations from the following regulatory text. "
            "Return JSON with keys entities and relations. Each entity should contain raw_name, canonical_name, "
            "entity_type, confidence, and evidence. Each relation should contain subject, object, relation_type, "
            "confidence, and evidence.\n\n"
            f"{chunk.text[:3500]}"
        )
        try:
            generated = self._generate_text(
                system_prompt="You extract structured legal knowledge and only return valid JSON.",
                user_prompt=prompt,
            )
            payload = _first_json_object(generated)
            if payload:
                return payload
        except Exception:
            pass
        return super().extract_structured_knowledge(chunk)

    def rewrite_query(self, question: str, context: list[str]) -> str:
        prompt = (
            "Rewrite this question for better multi-hop legal retrieval. Preserve meaning, keep it short.\n\n"
            f"Question: {question}\nContext: {context[:3]}"
        )
        try:
            generated = self._generate_text(
                system_prompt="You optimize questions for retrieval over regulatory corpora.",
                user_prompt=prompt,
            ).strip()
            return generated or super().rewrite_query(question, context)
        except Exception:
            return super().rewrite_query(question, context)

    def judge_evidence(self, question: str, evidence: list[str]) -> bool:
        return super().judge_evidence(question, evidence)

    def generate_grounded_answer(self, question: str, evidence: list[str]) -> dict[str, Any]:
        if not evidence:
            return super().generate_grounded_answer(question, evidence)
        prompt = (
            "Answer the user question using only the evidence snippets. "
            "If the evidence is not enough, say that explicitly.\n\n"
            f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence[:6])
        )
        try:
            generated = self._generate_text(
                system_prompt="You are a careful regulatory analyst. Stay grounded in the supplied evidence.",
                user_prompt=prompt,
            ).strip()
            if generated:
                return {"answer": generated, "confidence": 0.72, "fallback_used": False}
        except Exception:
            pass
        return super().generate_grounded_answer(question, evidence)

    def _get_embedder(self):
        if self._embedder is None and SentenceTransformer is not None:
            device = self._resolve_embedding_device()
            self._embedder = SentenceTransformer(self.settings.local_embedding_model, device=device)
        return self._embedder

    def _get_generator(self):
        if self._generator is None:
            if AutoTokenizer is None or pipeline is None:
                raise RuntimeError("transformers is not installed")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.local_chat_model,
                trust_remote_code=self.settings.local_trust_remote_code,
            )
            pipe_kwargs: dict[str, Any] = {
                "task": "text-generation",
                "model": self.settings.local_chat_model,
                "tokenizer": self._tokenizer,
            }
            device = self._resolve_generation_device()
            if device != -1:
                pipe_kwargs["device"] = device
            self._generator = pipeline(**pipe_kwargs)
        return self._generator

    def _generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        generator = self._get_generator()
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("Local tokenizer is not initialized")
        prompt = user_prompt
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        response = generator(
            prompt,
            max_new_tokens=self.settings.local_max_new_tokens,
            temperature=self.settings.local_temperature,
            do_sample=self.settings.local_temperature > 0,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        if not response:
            return ""
        generated = response[0]
        text = generated.get("generated_text", "") if isinstance(generated, dict) else str(generated)
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()

    def _resolve_generation_device(self) -> int:
        desired = self.settings.local_device.lower()
        if desired in {"cuda", "gpu"} and torch is not None and torch.cuda.is_available():
            return 0
        if desired == "auto" and torch is not None and torch.cuda.is_available():
            return 0
        return -1

    def _resolve_embedding_device(self) -> str:
        desired = self.settings.local_device.lower()
        if desired in {"cuda", "gpu"} and torch is not None and torch.cuda.is_available():
            return "cuda"
        if desired == "auto" and torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"


def build_provider(settings: Settings) -> LLMProvider:
    backend = settings.model_backend.lower().strip()
    if backend == "heuristic":
        return HeuristicLLMProvider(settings)
    if backend == "openai":
        if OpenAI is not None and (settings.openai_api_key or settings.openai_base_url):
            return OpenAIProvider(settings)
        return HeuristicLLMProvider(settings)
    if backend == "local":
        if _local_stack_available():
            return LocalTransformersProvider(settings)
        return HeuristicLLMProvider(settings)
    if OpenAI is not None and (settings.openai_api_key or settings.openai_base_url):
        return OpenAIProvider(settings)
    if _local_stack_available():
        return LocalTransformersProvider(settings)
    return HeuristicLLMProvider(settings)

