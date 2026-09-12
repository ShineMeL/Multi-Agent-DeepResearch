"""Server-selected demo modes. This is configuration discovery, not an online probe."""

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from deepresearch.providers import ProviderError
from deepresearch.providers.replay_schema import ReplayBundle
from deepresearch.runtime.runner_factory import FileProviderRouteCatalog, FrozenProviderRoutes

from .settings import ServiceSettings

router = APIRouter()

_SHIPPED_REPLAY_FILE_SHA256 = {
    "documents.jsonl": "25d7a516d29e1b5144c2c48e06a9e71f2b57120be2f019cfc276778c6c67ecd9",
    "embeddings.jsonl": "88dff3a9eeef98d21cd86c0576417ef58aed9c693047418a71305205551875c7",
    "model_responses.jsonl": "6e07f242ba0aa8f1f76ca8fc28d3597a2ded53a21fdc140a6b67ac8a089c8d2e",
    "search.jsonl": "84809bdb8d17b79c85a08eb895503af97e3b531240487f6cbf3706b0e6b095f0",
    "snapshot.json": "c3e40e217d7c6eaade4134388ca85c74554a898708b0da3ee9ebb5adbd9abbd6",
}


class DemoProfile(BaseModel):
    profile_id: str
    execution_mode: str
    available: bool
    reason: str | None
    workflow_id: Literal["baseline-v1", "research-v1"]
    planner_id: Literal["P1"] = "P1"
    ranker_id: Literal["R1"] = "R1"


class ReplayExample(BaseModel):
    provider_profile_id: str | None = None
    question: str = "Compare planner strategies"
    report_language: str = "en"
    source_languages: tuple[str, ...] = ("en",)
    budget_preset: str = "medium"
    seed: int = 0


class DemoCapabilities(BaseModel):
    profiles: list[DemoProfile]
    replay_example: ReplayExample | None
    budget_presets: tuple[str, ...]
    unpriced_live: bool


def _is_shipped_replay(routes: FrozenProviderRoutes) -> bool:
    bundle_paths = {
        path
        for route in routes.routes
        if route.operation != "parse"
        for path in (route.parameters.get("bundle_path"),)
        if isinstance(path, str) and path
    }
    if len(bundle_paths) != 1 or any(
        route.operation != "parse" and "bundle_path" not in route.parameters
        for route in routes.routes
    ):
        return False
    try:
        bundle = ReplayBundle.load(Path(next(iter(bundle_paths))))
    except (OSError, TypeError, ValueError, ProviderError):
        return False
    verification = bundle.verify()
    return verification.valid and verification.file_sha256 == _SHIPPED_REPLAY_FILE_SHA256


@router.get("/capabilities", response_model=DemoCapabilities)
async def capabilities(request: Request) -> DemoCapabilities:
    settings: ServiceSettings = request.app.state.settings
    # Match the manager's immutable startup selection; a changed file must not
    # advertise a route different from the one new runs will actually use.
    catalog: FileProviderRouteCatalog = request.app.state.manager.runner_factory.route_catalog
    profiles: list[DemoProfile] = []
    example: ReplayExample | None = None
    budgets = tuple(
        preset for preset in settings.allowed_budget_presets if preset in {"low", "medium"}
    )
    for profile_id in settings.allowed_provider_profile_ids:
        selected = catalog.resolve(profile_id)
        if selected.execution_mode not in {"replay", "live"}:
            continue
        complete = {r.operation for r in selected.routes} == {
            "model",
            "search",
            "fetch",
            "parse",
            "embed",
        }
        missing_credentials = any(
            r.credential_ref and not os.environ.get(r.credential_ref, "").strip()
            for r in selected.routes
        )
        replay = selected.execution_mode == "replay"
        demo_allowed = (
            "demo" in settings.allowed_run_purposes
            and bool(budgets)
            # A different budget changes the shipped recording's request hash.
            and (not replay or "medium" in budgets)
        )
        reason = (
            "PROVIDER_PROFILE_DRIFT"
            if not complete
            else "DEPLOYMENT_POLICY_VIOLATION"
            if not demo_allowed
            else "PROVIDER_NOT_CONFIGURED"
            if missing_credentials
            else None
        )
        profiles.append(
            DemoProfile(
                profile_id=profile_id,
                execution_mode=selected.execution_mode,
                available=reason is None,
                reason=reason,
                workflow_id="research-v1" if replay else "baseline-v1",
            )
        )
        # ReplayBundle.load verifies the bundle manifest before its recorded
        # run identity can select this one shipped, fixed request.
        if replay and reason is None and _is_shipped_replay(selected):
            example = ReplayExample(provider_profile_id=profile_id)
    return DemoCapabilities(
        profiles=profiles,
        replay_example=example,
        budget_presets=budgets,
        unpriced_live=settings.local_unpriced_live,
    )
