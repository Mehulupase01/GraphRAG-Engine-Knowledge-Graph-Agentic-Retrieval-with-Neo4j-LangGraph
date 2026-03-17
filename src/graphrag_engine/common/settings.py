from __future__ import annotations

import os
from pathlib import Path

from .artifacts import ensure_dir
from .compat import BaseModel


class Settings(BaseModel):
    env: str = "development"
    data_dir: str = "./data"
    log_level: str = "INFO"
    model_backend: str = "local"
    openai_api_key: str = ""
    openai_base_url: str = ""
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"
    judge_model: str = "gpt-4.1-mini"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_version: str = "2023-06-01"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    local_chat_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_device: str = "auto"
    local_max_new_tokens: int = 384
    local_temperature: float = 0.1
    local_trust_remote_code: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me-now"
    max_chunk_tokens: int = 450
    chunk_overlap: int = 80
    vector_dimension: int = 64
    default_retrieval_k: int = 8
    agent_max_rewrites: int = 2
    agent_max_graph_hops: int = 2
    eval_cases_path: str = "./configs/eval_cases.json"

    @property
    def project_root(self) -> Path:
        return Path.cwd()

    @property
    def data_path(self) -> Path:
        return ensure_dir((self.project_root / self.data_dir).resolve())

    @property
    def raw_data_path(self) -> Path:
        return ensure_dir(self.data_path / "raw")

    @property
    def processed_data_path(self) -> Path:
        return ensure_dir(self.data_path / "processed")

    @property
    def cache_data_path(self) -> Path:
        return ensure_dir(self.data_path / "cache")

    @classmethod
    def load(cls) -> "Settings":
        env_values = {}
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env_values[key.strip()] = value.strip()

        def get(name: str, default: str) -> str:
            return os.getenv(name, env_values.get(name, default))

        def get_bool(name: str, default: bool) -> bool:
            raw = get(name, "true" if default else "false").strip().lower()
            return raw in {"1", "true", "yes", "on"}

        return cls(
            env=get("GRAPH_RAG_ENV", "development"),
            data_dir=get("GRAPH_RAG_DATA_DIR", "./data"),
            log_level=get("GRAPH_RAG_LOG_LEVEL", "INFO"),
            model_backend=get("GRAPH_RAG_MODEL_BACKEND", "local"),
            openai_api_key=get("GRAPH_RAG_OPENAI_API_KEY", ""),
            openai_base_url=get("GRAPH_RAG_OPENAI_BASE_URL", ""),
            chat_model=get("GRAPH_RAG_CHAT_MODEL", "gpt-4.1-mini"),
            embedding_model=get("GRAPH_RAG_EMBEDDING_MODEL", "text-embedding-3-large"),
            judge_model=get("GRAPH_RAG_JUDGE_MODEL", "gpt-4.1-mini"),
            anthropic_api_key=get("GRAPH_RAG_ANTHROPIC_API_KEY", ""),
            anthropic_base_url=get("GRAPH_RAG_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            anthropic_model=get("GRAPH_RAG_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            anthropic_version=get("GRAPH_RAG_ANTHROPIC_VERSION", "2023-06-01"),
            gemini_api_key=get("GRAPH_RAG_GEMINI_API_KEY", ""),
            gemini_base_url=get("GRAPH_RAG_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            gemini_model=get("GRAPH_RAG_GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_embedding_model=get("GRAPH_RAG_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            local_chat_model=get("GRAPH_RAG_LOCAL_CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            local_embedding_model=get(
                "GRAPH_RAG_LOCAL_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            local_device=get("GRAPH_RAG_LOCAL_DEVICE", "auto"),
            local_max_new_tokens=int(get("GRAPH_RAG_LOCAL_MAX_NEW_TOKENS", "384")),
            local_temperature=float(get("GRAPH_RAG_LOCAL_TEMPERATURE", "0.1")),
            local_trust_remote_code=get_bool("GRAPH_RAG_LOCAL_TRUST_REMOTE_CODE", True),
            neo4j_uri=get("GRAPH_RAG_NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=get("GRAPH_RAG_NEO4J_USER", "neo4j"),
            neo4j_password=get("GRAPH_RAG_NEO4J_PASSWORD", "change-me-now"),
            max_chunk_tokens=int(get("GRAPH_RAG_MAX_CHUNK_TOKENS", "450")),
            chunk_overlap=int(get("GRAPH_RAG_CHUNK_OVERLAP", "80")),
            vector_dimension=int(get("GRAPH_RAG_VECTOR_DIMENSION", "64")),
            default_retrieval_k=int(get("GRAPH_RAG_DEFAULT_RETRIEVAL_K", "8")),
            agent_max_rewrites=int(get("GRAPH_RAG_AGENT_MAX_REWRITES", "2")),
            agent_max_graph_hops=int(get("GRAPH_RAG_AGENT_MAX_GRAPH_HOPS", "2")),
            eval_cases_path=get("GRAPH_RAG_EVAL_CASES_PATH", "./configs/eval_cases.json"),
        )
