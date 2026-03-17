from __future__ import annotations

import hashlib


def stable_hash(value: str, *, prefix: str | None = None, length: int = 16) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}" if prefix else digest

