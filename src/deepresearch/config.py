from math import isfinite
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, SecretStr, ValidationError, field_validator
from pydantic_core import InitErrorDetails
from pydantic_settings import BaseSettings, SettingsConfigDict

_SECRET_FIELDS = frozenset({"model_api_key", "tavily_api_key"})


def _safe_error_message(location: tuple[int | str, ...]) -> str:
    field = location[0] if location and isinstance(location[0], str) else "settings"
    if field == "model_base_url":
        return "model_base_url is invalid or contains credentials"
    if field in _SECRET_FIELDS:
        return f"{field} is invalid"
    if field == "model_id":
        return "model_id must not be empty"
    if field in {"connect_timeout_seconds", "read_timeout_seconds"}:
        return f"{field} must be finite and positive"
    return f"{field} is invalid"


def _sanitized_validation_error(error: ValidationError) -> ValidationError:
    safe_details: list[InitErrorDetails] = []
    for detail in error.errors(include_url=False):
        location = tuple(detail["loc"])
        message = _safe_error_message(location)
        safe_details.append(
            {
                "type": "value_error",
                "loc": location,
                "input": "[REDACTED]",
                "ctx": {"error": ValueError(message)},
            }
        )
    return ValidationError.from_exception_data("Settings", safe_details)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPRESEARCH_",
        env_file=".env",
        extra="ignore",
    )

    model_base_url: AnyHttpUrl
    model_id: str
    model_api_key: SecretStr
    tavily_api_key: SecretStr | None = None
    artifact_root: Path = Path("artifacts")
    cache_root: Path = Path(".cache/deepresearch")
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 45.0
    embedding_model_root: Path = Path("models/embedding")
    embedding_lock_path: Path = Path("models/embedding.lock.json")

    def __init__(self, **values: object) -> None:
        failure: ValidationError | None = None
        try:
            super().__init__(**cast("dict[str, Any]", values))
        except ValidationError as error:
            failure = _sanitized_validation_error(error)
        if failure is not None:
            raise failure

    @field_validator("model_base_url")
    @classmethod
    def validate_model_base_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        parsed = urlsplit(str(value))
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("model_base_url must not contain credentials")
        return value

    @field_validator("connect_timeout_seconds", "read_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if not isfinite(value) or value <= 0:
            raise ValueError("timeout must be finite and positive")
        return value

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_id must not be empty")
        return value


__all__ = ["Settings"]
