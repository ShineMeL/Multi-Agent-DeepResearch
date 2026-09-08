"""Server-owned service policy and secret loading for the ASGI factory."""

import os
import re
from decimal import Decimal
from ipaddress import ip_network
from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from deepresearch.domain import AccessProfile, ExecutionMode, RunBudget, RunPurpose
from deepresearch.runtime.deployment_policy import DeploymentPolicy
from deepresearch.runtime.runner_factory import FileProviderRouteCatalog, ProviderProfileDrift


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", hide_input_in_errors=True)

    database_url: str = "sqlite+aiosqlite:///./deepresearch.db"
    artifact_root: Path = Path("./artifacts")
    checkpoint_sqlite_path: Path = Path("./artifacts/checkpoints.sqlite")
    pricing_catalog_path: Path | None = None
    provider_profile_catalog_path: Path | None = None
    deployment_access_profile: AccessProfile = "showcase"
    allowed_execution_modes: tuple[ExecutionMode, ...] = ("replay",)
    allowed_provider_profile_ids: tuple[str, ...] = ("replay-default",)
    allowed_run_purposes: tuple[RunPurpose, ...] = ("demo", "test")
    allowed_budget_presets: tuple[Literal["low", "medium", "high"], ...] = ("low", "medium")
    daily_cost_limit_usd: Decimal = Decimal("5.00")
    session_signing_key: SecretStr
    cookie_secure: bool = False
    trusted_proxy_cidrs: tuple[str, ...] = ()
    langgraph_strict_msgpack: Literal[True]
    provider_credential_env_names: tuple[str, ...] = ("MODEL_API_KEY", "SEARCH_API_KEY")
    redaction_secret_env_names: tuple[str, ...] = (
        "MODEL_API_KEY",
        "SEARCH_API_KEY",
        "SESSION_SIGNING_KEY",
    )

    @field_validator("langgraph_strict_msgpack", mode="before")
    @classmethod
    def explicit_strict_msgpack(cls, value: object) -> object:
        # Literal[True] intentionally has no default; environment strings need
        # explicit conversion without accepting arbitrary truthy values.
        if isinstance(value, str) and value.lower() == "true":
            return True
        return value

    @field_validator("session_signing_key")
    @classmethod
    def strong_signing_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value()
        if not key.strip() or len(key.encode("utf-8")) < 32:
            raise ValueError("SESSION_SIGNING_KEY must be nonblank and contain at least 32 bytes")
        return value

    @field_validator("daily_cost_limit_usd")
    @classmethod
    def finite_daily_limit(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value < 0:
            raise ValueError("daily cost limit must be finite and non-negative")
        return value

    @field_validator("trusted_proxy_cidrs")
    @classmethod
    def valid_proxy_networks(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            ip_network(value)
        return values

    @model_validator(mode="after")
    def consistent_policy(self) -> Self:
        for values in (
            self.allowed_execution_modes,
            self.allowed_provider_profile_ids,
            self.allowed_run_purposes,
            self.allowed_budget_presets,
        ):
            if not values or any(not value.strip() or value != value.strip() for value in values):
                raise ValueError("policy allowlists must contain nonblank values")
        for name in (*self.provider_credential_env_names, *self.redaction_secret_env_names):
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None:
                raise ValueError("secret environment names must be logical uppercase names")
        if "SESSION_SIGNING_KEY" in self.provider_credential_env_names:
            raise ValueError("provider credentials cannot use SESSION_SIGNING_KEY")
        if not set(self.provider_credential_env_names) <= set(self.redaction_secret_env_names):
            raise ValueError("provider credential names must be included in redaction names")
        if self.deployment_access_profile == "public_live":
            if not self.cookie_secure:
                raise ValueError("public_live requires cookie_secure=true")
            if not set(self.allowed_budget_presets) <= {"low", "medium"}:
                raise ValueError("public_live permits only low and medium budgets")
            medium = RunBudget.preset("medium").model_dump()
            for preset in self.allowed_budget_presets:
                budget = RunBudget.preset(preset).model_dump()
                for name in (
                    "max_search_calls",
                    "max_pages",
                    "max_total_tokens",
                    "max_wall_time_seconds",
                    "max_cost_usd",
                    "max_retries",
                ):
                    if budget[name] is None or budget[name] > medium[name]:
                        raise ValueError("public budgets must not exceed Core medium limits")
        try:
            self.validate_route_catalog(
                FileProviderRouteCatalog.load(self.provider_profile_catalog_path)
            )
        except (OSError, TypeError, ValueError, ProviderProfileDrift):
            raise ValueError(
                "provider catalog must agree with policy and credential allowlists"
            ) from None
        return self

    def validate_route_catalog(self, catalog: FileProviderRouteCatalog) -> None:
        for profile_id in self.allowed_provider_profile_ids:
            profile = catalog.resolve(profile_id)
            if profile.execution_mode not in self.allowed_execution_modes:
                raise ValueError("provider profile execution mode is not allowed")
            if any(
                route.credential_ref is not None
                and route.credential_ref not in self.provider_credential_env_names
                for route in profile.routes
            ):
                raise ValueError("provider profile credential is not allowed")

    def deployment_policy(self) -> DeploymentPolicy:
        return DeploymentPolicy(
            forced_access_profile=self.deployment_access_profile,
            allowed_execution_modes=frozenset(self.allowed_execution_modes),
            allowed_provider_profile_ids=frozenset(self.allowed_provider_profile_ids),
            allowed_run_purposes=frozenset(self.allowed_run_purposes),
            allowed_budget_presets=frozenset(self.allowed_budget_presets),
            budget_presets={name: RunBudget.preset(name) for name in self.allowed_budget_presets},
        )

    def loaded_secret_values(self) -> tuple[str, ...]:
        values = (
            self.session_signing_key.get_secret_value(),
            *(os.environ.get(name, "") for name in self.redaction_secret_env_names),
        )
        return tuple(dict.fromkeys(value for value in values if value))
