from math import isfinite
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @model_validator(mode="after")
    def validate_security_and_timeouts(self) -> Self:
        parsed = urlsplit(str(self.model_base_url))
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("model_base_url must not contain credentials")
        for name in ("connect_timeout_seconds", "read_timeout_seconds"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        return self


__all__ = ["Settings"]
