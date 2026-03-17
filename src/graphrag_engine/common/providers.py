from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from .artifacts import ensure_dir
from .hashing import stable_hash
from .models import ChunkRecord
from .settings import Settings

try:
    from huggingface_hub import snapshot_download  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    snapshot_download = None

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
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
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


def _join_url(base_url: str, *parts: str) -> str:
    url = base_url.rstrip("/")
    for part in parts:
        url = f"{url}/{part.strip('/')}"
    return url


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
        self._tokenizer = None
        self._generator = None
        self._embedder = None
        local_stack_missing = not _local_stack_available()
        self._generator_disabled_reason: str | None = (
            "transformers and sentence-transformers are not installed" if local_stack_missing else None
        )
        self._embedder_disabled_reason: str | None = (
            "transformers and sentence-transformers are not installed" if local_stack_missing else None
        )
        self._hf_home = self._configure_cache_environment()
        self._local_model_root = ensure_dir(self._hf_home / "local_models")
        self._materialized_models: dict[str, Path] = {}

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "chat_model": self.settings.local_chat_model,
            "embedding_model": self.settings.local_embedding_model,
            "device": self.settings.local_device,
            "cache_dir": str(self._hf_home),
            "generator_available": self._generator_disabled_reason is None,
            "embedder_available": self._embedder_disabled_reason is None,
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if SentenceTransformer is None:
            return super().embed_texts(texts)
        try:
            embedder = self._get_embedder()
            vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
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
        if self._embedder_disabled_reason is not None:
            raise RuntimeError(self._embedder_disabled_reason)
        if self._embedder is None and SentenceTransformer is not None:
            device = self._resolve_embedding_device()
            try:
                self._embedder = SentenceTransformer(
                    self._materialize_model(self.settings.local_embedding_model, category="sentence_transformers"),
                    device=device,
                    cache_folder=str(self._hf_home / "sentence_transformers"),
                    local_files_only=True,
                )
            except Exception as exc:
                self._embedder_disabled_reason = f"local embedder unavailable: {exc}"
                raise
        return self._embedder

    def _get_generator(self):
        if self._generator_disabled_reason is not None:
            raise RuntimeError(self._generator_disabled_reason)
        if self._generator is None:
            if AutoTokenizer is None or AutoModelForCausalLM is None or pipeline is None:
                raise RuntimeError("transformers is not installed")
            try:
                model_path = self._materialize_model(self.settings.local_chat_model, category="transformers")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    cache_dir=str(self._hf_home / "transformers"),
                    local_files_only=True,
                    trust_remote_code=self.settings.local_trust_remote_code,
                )
                if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(
                    self._tokenizer, "eos_token_id", None
                ) is not None:
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

                model_kwargs: dict[str, Any] = {
                    "cache_dir": str(self._hf_home / "transformers"),
                    "trust_remote_code": self.settings.local_trust_remote_code,
                }
                if torch is not None:
                    if self._resolve_generation_device() != -1:
                        model_kwargs["dtype"] = getattr(torch, "float16", None) or getattr(torch, "float32")
                    else:
                        model_kwargs["dtype"] = getattr(torch, "float32")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    **model_kwargs,
                )

                pipe_kwargs: dict[str, Any] = {
                    "task": "text-generation",
                    "model": model,
                    "tokenizer": self._tokenizer,
                }
                device = self._resolve_generation_device()
                if device != -1:
                    pipe_kwargs["device"] = device
                self._generator = pipeline(**pipe_kwargs)
            except Exception as exc:
                self._generator_disabled_reason = f"local generator unavailable: {exc}"
                raise
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

    def _configure_cache_environment(self) -> Path:
        hf_home = ensure_dir(self.settings.cache_data_path / "huggingface")
        ensure_dir(hf_home / "transformers")
        ensure_dir(hf_home / "sentence_transformers")
        cache_values = {
            "HF_HOME": str(hf_home),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "SENTENCE_TRANSFORMERS_HOME": str(hf_home / "sentence_transformers"),
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for name, value in cache_values.items():
            os.environ[name] = value
        return hf_home

    def _materialize_model(self, model_ref: str, *, category: str) -> str:
        model_path = Path(model_ref)
        if model_path.exists():
            return str(model_path)
        cached = self._materialized_models.get(model_ref)
        if cached is not None:
            return str(cached)
        if snapshot_download is None:
            return model_ref

        target_dir = ensure_dir(self._local_model_root / category / self._slugify_model_ref(model_ref))
        if any(target_dir.iterdir()):
            self._materialized_models[model_ref] = target_dir
            return str(target_dir)
        snapshot_download(
            repo_id=model_ref,
            local_dir=str(target_dir),
            cache_dir=str(self._hf_home / "hub"),
        )
        self._materialized_models[model_ref] = target_dir
        return str(target_dir)

    @staticmethod
    def _slugify_model_ref(model_ref: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "--", model_ref).strip("-") or "model"


class AnthropicProvider(LocalTransformersProvider):
    provider_name = "anthropic"

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "provider": self.provider_name,
                "model": self.settings.anthropic_model,
                "base_url": self.settings.anthropic_base_url,
            }
        )
        return payload

    def extract_structured_knowledge(self, chunk: ChunkRecord) -> dict[str, Any]:
        prompt = (
            "Extract entities and relations from the following regulatory text. "
            "Return strict JSON with top-level keys entities and relations.\n\n"
            f"{chunk.text[:3500]}"
        )
        try:
            generated = self._call_messages(
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
            "Rewrite this user question for stronger legal retrieval over a regulatory knowledge graph. "
            "Keep it concise and preserve the meaning.\n\n"
            f"Question: {question}\nContext: {context[:3]}"
        )
        try:
            rewritten = self._call_messages(
                system_prompt="You optimize legal retrieval queries.",
                user_prompt=prompt,
            ).strip()
            return rewritten or super().rewrite_query(question, context)
        except Exception:
            return super().rewrite_query(question, context)

    def generate_grounded_answer(self, question: str, evidence: list[str]) -> dict[str, Any]:
        if not evidence:
            return super().generate_grounded_answer(question, evidence)
        prompt = (
            "Answer the question using only the supplied evidence snippets. "
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence[:6])
        )
        try:
            answer = self._call_messages(
                system_prompt="You are a careful regulatory analyst. Stay fully grounded in the evidence.",
                user_prompt=prompt,
            ).strip()
            if answer:
                return {"answer": answer, "confidence": 0.82, "fallback_used": False}
        except Exception:
            pass
        return super().generate_grounded_answer(question, evidence)

    def _call_messages(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.settings.anthropic_api_key:
            raise RuntimeError("Anthropic API key is not configured")
        base_url = self.settings.anthropic_base_url.rstrip("/")
        if base_url.endswith("/v1/messages"):
            url = base_url
        elif base_url.endswith("/v1"):
            url = _join_url(base_url, "messages")
        else:
            url = _join_url(base_url, "v1", "messages")
        response = httpx.post(
            url,
            headers={
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": self.settings.anthropic_version,
                "content-type": "application/json",
            },
            json={
                "model": self.settings.anthropic_model,
                "max_tokens": self.settings.local_max_new_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=90.0,
        )
        response.raise_for_status()
        payload = response.json()
        text_blocks = [
            block.get("text", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(item for item in text_blocks if item).strip()


class GeminiProvider(LocalTransformersProvider):
    provider_name = "gemini"

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "provider": self.provider_name,
                "model": self.settings.gemini_model,
                "embedding_model": self.settings.gemini_embedding_model,
                "base_url": self.settings.gemini_base_url,
            }
        )
        return payload

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.gemini_api_key:
            return super().embed_texts(texts)
        try:
            url = _join_url(
                self.settings.gemini_base_url,
                "models",
                f"{self.settings.gemini_embedding_model}:batchEmbedContents",
            )
            response = httpx.post(
                url,
                headers={
                    "x-goog-api-key": self.settings.gemini_api_key,
                    "content-type": "application/json",
                },
                json={
                    "requests": [
                        {
                            "model": f"models/{self.settings.gemini_embedding_model}",
                            "content": {"parts": [{"text": text}]},
                            "taskType": "SEMANTIC_SIMILARITY",
                        }
                        for text in texts
                    ]
                },
                timeout=90.0,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings", [])
            vectors = [self._extract_gemini_embedding(item) for item in embeddings]
            if vectors and all(vector for vector in vectors):
                return vectors
        except Exception:
            pass
        return super().embed_texts(texts)

    def extract_structured_knowledge(self, chunk: ChunkRecord) -> dict[str, Any]:
        prompt = (
            "Extract entities and relations from the following regulatory text. "
            "Return strict JSON with top-level keys entities and relations.\n\n"
            f"{chunk.text[:3500]}"
        )
        try:
            generated = self._call_generate_content(
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
            "Rewrite this user question for stronger legal retrieval over a regulatory knowledge graph. "
            "Keep it concise and preserve the meaning.\n\n"
            f"Question: {question}\nContext: {context[:3]}"
        )
        try:
            rewritten = self._call_generate_content(
                system_prompt="You optimize legal retrieval queries.",
                user_prompt=prompt,
            ).strip()
            return rewritten or super().rewrite_query(question, context)
        except Exception:
            return super().rewrite_query(question, context)

    def generate_grounded_answer(self, question: str, evidence: list[str]) -> dict[str, Any]:
        if not evidence:
            return super().generate_grounded_answer(question, evidence)
        prompt = (
            "Answer the question using only the supplied evidence snippets. "
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence[:6])
        )
        try:
            answer = self._call_generate_content(
                system_prompt="You are a careful regulatory analyst. Stay fully grounded in the evidence.",
                user_prompt=prompt,
            ).strip()
            if answer:
                return {"answer": answer, "confidence": 0.8, "fallback_used": False}
        except Exception:
            pass
        return super().generate_grounded_answer(question, evidence)

    def _call_generate_content(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.settings.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured")
        url = _join_url(
            self.settings.gemini_base_url,
            "models",
            f"{self.settings.gemini_model}:generateContent",
        )
        response = httpx.post(
            url,
            headers={
                "x-goog-api-key": self.settings.gemini_api_key,
                "content-type": "application/json",
            },
            json={
                "system_instruction": {
                    "parts": [{"text": system_prompt}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": self.settings.local_temperature,
                },
            },
            timeout=90.0,
        )
        response.raise_for_status()
        payload = response.json()
        parts = (
            payload.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        return "\n".join(item for item in text_parts if item).strip()

    @staticmethod
    def _extract_gemini_embedding(payload: dict[str, Any]) -> list[float]:
        if "values" in payload and isinstance(payload["values"], list):
            return [float(value) for value in payload["values"]]
        if "embedding" in payload and isinstance(payload["embedding"], dict):
            values = payload["embedding"].get("values", [])
            return [float(value) for value in values]
        return []


def build_provider(settings: Settings) -> LLMProvider:
    backend = settings.model_backend.lower().strip()
    if backend == "heuristic":
        return HeuristicLLMProvider(settings)
    if backend == "openai":
        if OpenAI is not None and (settings.openai_api_key or settings.openai_base_url):
            return OpenAIProvider(settings)
        return HeuristicLLMProvider(settings)
    if backend == "anthropic":
        if settings.anthropic_api_key:
            return AnthropicProvider(settings)
        return HeuristicLLMProvider(settings)
    if backend == "gemini":
        if settings.gemini_api_key:
            return GeminiProvider(settings)
        return HeuristicLLMProvider(settings)
    if backend == "local":
        if _local_stack_available():
            return LocalTransformersProvider(settings)
        return HeuristicLLMProvider(settings)
    if OpenAI is not None and (settings.openai_api_key or settings.openai_base_url):
        return OpenAIProvider(settings)
    if settings.anthropic_api_key:
        return AnthropicProvider(settings)
    if settings.gemini_api_key:
        return GeminiProvider(settings)
    if _local_stack_available():
        return LocalTransformersProvider(settings)
    return HeuristicLLMProvider(settings)
