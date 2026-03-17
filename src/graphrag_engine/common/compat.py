from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field  # type: ignore
except ImportError:

    @dataclass
    class _FieldInfo:
        default: Any = None
        default_factory: Any = None
        description: str | None = None

    def Field(  # type: ignore[misc]
        default: Any = None,
        *,
        default_factory: Any = None,
        description: str | None = None,
    ) -> _FieldInfo:
        return _FieldInfo(default=default, default_factory=default_factory, description=description)

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    class BaseModel:
        model_config: dict[str, Any] = {}

        def __init__(self, **data: Any) -> None:
            annotations = self._collect_annotations()
            for name in annotations:
                if name in data:
                    value = data[name]
                elif hasattr(self.__class__, name):
                    raw_default = getattr(self.__class__, name)
                    if isinstance(raw_default, _FieldInfo):
                        if raw_default.default_factory is not None:
                            value = raw_default.default_factory()
                        else:
                            value = deepcopy(raw_default.default)
                    else:
                        value = deepcopy(raw_default)
                else:
                    raise TypeError(f"Missing required field: {name}")
                setattr(self, name, value)

            for name, value in data.items():
                if name not in annotations:
                    setattr(self, name, value)

        @classmethod
        def _collect_annotations(cls) -> dict[str, Any]:
            annotations: dict[str, Any] = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, "__annotations__", {}))
            annotations.pop("model_config", None)
            return annotations

        @classmethod
        def model_validate(cls, data: Any) -> "BaseModel":
            if isinstance(data, cls):
                return data
            if isinstance(data, dict):
                return cls(**data)
            raise TypeError(f"Cannot validate {type(data)!r} as {cls.__name__}")

        @classmethod
        def model_validate_json(cls, payload: str) -> "BaseModel":
            return cls.model_validate(json.loads(payload))

        def model_dump(self) -> dict[str, Any]:
            return {
                key: self._coerce_for_dump(getattr(self, key))
                for key in self._collect_annotations()
            }

        def model_dump_json(self, *, indent: int | None = None) -> str:
            return json.dumps(self.model_dump(), indent=indent)

        def model_copy(self, *, update: dict[str, Any] | None = None) -> "BaseModel":
            payload = self.model_dump()
            if update:
                payload.update(update)
            return self.__class__(**payload)

        @staticmethod
        def _coerce_for_dump(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump()
            if isinstance(value, list):
                return [BaseModel._coerce_for_dump(item) for item in value]
            if isinstance(value, dict):
                return {key: BaseModel._coerce_for_dump(item) for key, item in value.items()}
            return value
