"""Explicit, loopback-only demo configuration; credentials never enter catalogs."""

import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values
from fastapi import FastAPI
from pydantic import SecretStr

from deepresearch.security import redact

from .main import create_app
from .settings import ServiceSettings


@dataclass(frozen=True)
class PreparedDemo:
    settings: ServiceSettings
    environment: dict[str, str] = field(repr=False)


def demo_environment(repository: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """Load only the explicit local file, with process environment taking precedence."""
    values = {
        key: value
        for key, value in dotenv_values(
            repository / ".env.demo", encoding="utf-8-sig", interpolate=False
        ).items()
        if value is not None
    }
    values.update(environ)
    for target, aliases in {
        "MODEL_API_KEY": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "SEARCH_API_KEY": ("TAVILY_API_KEY",),
    }.items():
        if not values.get(target, "").strip():
            values[target] = next(
                (values[name] for name in aliases if values.get(name, "").strip()), ""
            )
    return values


def prepare_demo(repository: Path, state_dir: Path, *, environ: Mapping[str, str]) -> PreparedDemo:
    repository, state_dir = repository.resolve(), state_dir.resolve()
    environment = demo_environment(repository, environ)
    provider_id = environment.get("MODEL_PROVIDER", "kimi-instant")
    if provider_id not in {"kimi-instant", "openai-compatible"}:
        raise ValueError("MODEL_PROVIDER must be kimi-instant or openai-compatible")
    model_id = environment.get("MODEL_ID", "kimi-k2.6")
    if provider_id == "kimi-instant" and model_id not in {"kimi-k2.5", "kimi-k2.6"}:
        raise ValueError("kimi-instant supports MODEL_ID kimi-k2.5 or kimi-k2.6")
    profiles = cast(
        dict[str, Any],
        json.loads((repository / "deploy/replay/profiles.json").read_text(encoding="utf-8")),
    )
    for replay_route in profiles["profiles"]["replay-default"]["routes"]:
        if "bundle_path" in replay_route["parameters"]:
            replay_route["parameters"]["bundle_path"] = str(
                repository / replay_route["parameters"]["bundle_path"]
            )

    def route(operation: str, provider: str, **extra: object) -> dict[str, object]:
        return {
            "operation": operation,
            "provider_id": provider,
            "endpoint_type": operation,
            "model_id": None,
            "model_revision": None,
            "base_url": None,
            "credential_ref": None,
            "fallback_rank": 0,
            "parameters": {},
            **extra,
        }

    profiles["profiles"]["live-default"] = {
        "execution_mode": "live",
        "routes": [
            route(
                "model",
                provider_id,
                endpoint_type="chat.completions",
                model_id=model_id,
                model_revision="provider-managed-instant-v1"
                if provider_id == "kimi-instant"
                else "provider-managed",
                base_url=environment.get("MODEL_BASE_URL", "https://api.moonshot.cn/v1"),
                credential_ref="MODEL_API_KEY",
            ),
            route(
                "search",
                "tavily",
                base_url="https://api.tavily.com/search",
                credential_ref="SEARCH_API_KEY",
            ),
            route("fetch", "httpx-fetcher"),
            route("parse", "baseline-parser-router"),
            route(
                "embed",
                "lexical-hash",
                model_id="lexical-hash-v1",
                model_revision="1",
                parameters={"dimension": 384},
            ),
        ],
    }
    secret_values = tuple(environment.get(key, "") for key in ("MODEL_API_KEY", "SEARCH_API_KEY"))
    if redact(profiles, secrets=tuple(v for v in secret_values if v)) != profiles:
        raise ValueError("demo catalog must not contain credential values")
    state_dir.mkdir(parents=True, exist_ok=True)
    signing_path = state_dir / "session-signing.key"
    try:
        # Keep the browser owner stable across restarts. Never overwrite an
        # existing session key or print it in startup output.
        with signing_path.open("x", encoding="utf-8") as stream:
            stream.write(secrets.token_urlsafe(48))
        signing_path.chmod(0o600)
    except FileExistsError:
        pass
    profiles_path = state_dir / "profiles.json"
    profiles_path.write_text(json.dumps(profiles, ensure_ascii=False), encoding="utf-8")
    settings = ServiceSettings(
        database_url=f"sqlite+aiosqlite:///{(state_dir / 'runs.sqlite').as_posix()}",
        artifact_root=state_dir,
        checkpoint_sqlite_path=state_dir / "checkpoints.sqlite",
        provider_profile_catalog_path=profiles_path,
        pricing_catalog_path=repository / "deploy/replay/pricing.json",
        deployment_access_profile="local",
        local_unpriced_live=True,
        allowed_execution_modes=("replay", "live"),
        allowed_provider_profile_ids=("replay-default", "live-default"),
        allowed_run_purposes=("demo",),
        session_signing_key=SecretStr(signing_path.read_text(encoding="utf-8").strip()),
        langgraph_strict_msgpack=True,
    )
    return PreparedDemo(settings, environment)


def create_demo_app() -> FastAPI:
    repository = Path(__file__).resolve().parents[2]
    state = Path(os.environ.get("DEEPRESEARCH_DEMO_STATE", str(repository / "artifacts/demo")))
    prepared = prepare_demo(repository, state, environ=os.environ)
    # Only the service process receives loaded credentials. Neither catalogs nor
    # the Streamlit client receive them.
    for key in ("MODEL_API_KEY", "SEARCH_API_KEY"):
        os.environ[key] = prepared.environment.get(key, "")
    return create_app(prepared.settings)
