"""Server-selected demo modes. This is configuration discovery, not an online probe."""

import os
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from deepresearch.runtime.runner_factory import FileProviderRouteCatalog

from .settings import ServiceSettings

router = APIRouter()


class DemoProfile(BaseModel):
    profile_id: str
    execution_mode: str
    available: bool
    reason: str | None
    workflow_id: Literal["baseline-v1", "research-v1"]
    planner_id: Literal["P1"] = "P1"
    ranker_id: Literal["R1"] = "R1"


class ReplayExample(BaseModel):
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
        # Only advertise the shipped example, not a guessed question for a
        # custom recording. Execution still verifies the bundle's byte hashes.
        if (
            replay
            and reason is None
            and any(
                r.operation == "model" and r.model_id == "baseline-model-v1"
                for r in selected.routes
            )
        ):
            example = ReplayExample()
    return DemoCapabilities(
        profiles=profiles,
        replay_example=example,
        budget_presets=budgets,
        unpriced_live=settings.local_unpriced_live,
    )
