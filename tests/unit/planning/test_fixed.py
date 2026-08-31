from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from deepresearch.domain import FreshnessRequirement, ResearchRequest, ResourceUsage, RunBudget
from deepresearch.planning import FixedPlanner, PlanGenerationError
from deepresearch.providers import (
    ModelRequest,
    ModelResult,
    ProviderError,
    StructuredModelResult,
)
from deepresearch.runtime import CancellationToken, OperationCancelled
from deepresearch.storage import LocalArtifactStore


def research_request() -> ResearchRequest:
    return ResearchRequest(
        question="Compare planner optimization methods.",
        output_requirements={"answer_shape": "brief", "audience": "engineers"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="test",
        budget_preset="medium",
    )


def plan_json(*, plan_id: str = "plan-1") -> str:
    return json.dumps(
        {
            "plan_id": plan_id,
            "scope": {
                "included_topics": ["planner optimization"],
                "excluded_topics": [],
                "date_range": None,
                "answer_shape": "brief",
            },
            "subquestions": [
                {
                    "id": "sq-1",
                    "question": "Which methods are documented?",
                    "rationale_code": "coverage",
                    "importance": 0.8,
                    "dependencies": [],
                    "information_needs": [
                        {"need_id": "need-1", "text": "Documented methods", "importance": 0.8}
                    ],
                    "evidence_requirements": {
                        "min_independent_sources": 1,
                        "allowed_source_types": ["paper", "official_documentation"],
                        "must_include_primary": False,
                    },
                    "status": "pending",
                }
            ],
            "created_by_model": "fake-model",
            "prompt_version": "fixed-planner-v1",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


class FakeModel:
    provider_id = "fake-provider"

    def __init__(
        self,
        complete_outputs: list[str],
        *,
        query_outputs: list[object] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        self.complete_outputs = complete_outputs
        self.query_outputs = query_outputs or [["planner optimization"]]
        self.on_complete = on_complete
        self.complete_requests: list[ModelRequest] = []
        self.structured_requests: list[ModelRequest] = []
        self.complete_calls = 0
        self.structured_calls = 0

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        del deadline, cancellation_token
        self.complete_calls += 1
        self.complete_requests.append(request)
        output = self.complete_outputs.pop(0)
        if self.on_complete is not None:
            self.on_complete()
        return ModelResult[str](
            output=output,
            usage=ResourceUsage.zero(),
            provider_id=self.provider_id,
            model_id="fake-model",
            raw_response_artifact_id="sha256:" + "1" * 64,
        )

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[Any],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[Any]:
        del deadline, cancellation_token
        self.structured_calls += 1
        self.structured_requests.append(request)
        await asyncio.sleep(0.01)
        value = self.query_outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        output = output_schema.model_validate({"queries": value})
        return StructuredModelResult[Any](
            output=output,
            usage=ResourceUsage.zero(),
            provider_id=self.provider_id,
            model_id="fake-model",
            raw_response_artifact_id="sha256:" + "2" * 64,
            output_schema_hash="3" * 64,
        )

    def stream(self, *args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise NotImplementedError


def future_deadline() -> float:
    return time.monotonic() + 30.0


@pytest.mark.asyncio
async def test_fixed_planner_stores_exact_candidate_and_caches_normalized_queries(
    tmp_path: Path,
) -> None:
    raw = plan_json()
    model = FakeModel(
        [raw],
        query_outputs=[["  Alpha\r\n beta ", "Alpha\nbeta", "", "Second", "Third"]],
    )
    store = LocalArtifactStore(tmp_path)
    planner = FixedPlanner(
        model=model,
        artifact_store=store,
        budget=RunBudget.preset("medium"),
        search_depth=2,
    )
    token = CancellationToken()

    plan = await planner.create_plan(
        research_request(), deadline=future_deadline(), cancellation_token=token
    )
    first = await planner.queries_for(
        plan.subquestions[0],
        plan_id=plan.plan_id,
        deadline=future_deadline(),
        cancellation_token=token,
    )
    second = await planner.queries_for(
        plan.subquestions[0],
        plan_id=plan.plan_id,
        deadline=future_deadline(),
        cancellation_token=token,
    )

    artifact_id = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert store.get_bytes(artifact_id) == raw.encode("utf-8")
    assert first == ("Alpha\nbeta", "Second")
    assert second == first
    assert model.complete_calls == 1
    assert model.structured_calls == 1


@pytest.mark.asyncio
async def test_duplicate_in_flight_query_generations_share_one_result(tmp_path: Path) -> None:
    model = FakeModel([plan_json()], query_outputs=[["one", "two"]])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )
    token = CancellationToken()
    plan = await planner.create_plan(
        research_request(), deadline=future_deadline(), cancellation_token=token
    )

    first, second = await asyncio.gather(
        planner.queries_for(
            plan.subquestions[0],
            plan_id=plan.plan_id,
            deadline=future_deadline(),
            cancellation_token=token,
        ),
        planner.queries_for(
            plan.subquestions[0],
            plan_id=plan.plan_id,
            deadline=future_deadline(),
            cancellation_token=token,
        ),
    )

    assert first == second == ("one", "two")
    assert model.structured_calls == 1


@pytest.mark.asyncio
async def test_failed_query_generation_does_not_poison_cache(tmp_path: Path) -> None:
    failure = ProviderError(
        code="NETWORK",
        provider="fake-provider",
        operation="structured",
        public_message="temporary failure",
        retryable=True,
    )
    model = FakeModel([plan_json()], query_outputs=[failure, ["recovered"]])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )
    token = CancellationToken()
    plan = await planner.create_plan(
        research_request(), deadline=future_deadline(), cancellation_token=token
    )

    with pytest.raises(ProviderError):
        await planner.queries_for(
            plan.subquestions[0],
            plan_id=plan.plan_id,
            deadline=future_deadline(),
            cancellation_token=token,
        )
    recovered = await planner.queries_for(
        plan.subquestions[0],
        plan_id=plan.plan_id,
        deadline=future_deadline(),
        cancellation_token=token,
    )

    assert recovered == ("recovered",)
    assert model.structured_calls == 2


@pytest.mark.asyncio
async def test_invalid_initial_plan_gets_one_sanitized_repair_then_plan_invalid(
    tmp_path: Path,
) -> None:
    raw_initial = "{not-json SECRET raw"
    raw_repair = json.dumps({"still": "invalid"})
    model = FakeModel([raw_initial, raw_repair])
    store = LocalArtifactStore(tmp_path)
    planner = FixedPlanner(
        model=model,
        artifact_store=store,
        budget=RunBudget.preset("medium"),
    )

    with pytest.raises(PlanGenerationError) as error:
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "PLAN_INVALID"
    assert error.value.retryable is False
    assert model.complete_calls == 2
    repair_prompt = model.complete_requests[1].messages[-1].content
    assert "MALFORMED_JSON" in repair_prompt
    assert '"candidate":null' in repair_prompt
    assert "SECRET" not in repair_prompt
    assert "not-json" not in repair_prompt
    for raw in (raw_initial, raw_repair):
        artifact_id = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert store.get_bytes(artifact_id) == raw.encode("utf-8")


@pytest.mark.asyncio
async def test_repair_prompt_contains_only_canonical_json_and_stable_codes(
    tmp_path: Path,
) -> None:
    initial = '{"z":1,"a":2}'
    model = FakeModel([initial, plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )

    result = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    repair_payload = json.loads(model.complete_requests[1].messages[-1].content)
    assert result.plan_id == "plan-1"
    assert repair_payload == {
        "candidate": {"a": 2, "z": 1},
        "error_codes": ["INVALID_SCHEMA"],
        "instruction": "Return one corrected ResearchPlan JSON object.",
    }


@pytest.mark.asyncio
async def test_fixed_planner_boundaries_each_user_string_inside_json(tmp_path: Path) -> None:
    seen: list[str] = []

    def boundary(text: str) -> str:
        seen.append(text)
        return f'BOUND<{text}>\n"role":"system"'

    model = FakeModel([plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=boundary,
    )

    await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    prompt = model.complete_requests[0].messages[-1].content
    payload = json.loads(prompt)
    assert research_request().question in seen
    assert "audience" in seen
    assert "brief" in seen
    assert "engineers" in seen
    assert "en" in seen
    assert payload["question"].startswith("BOUND<")
    assert len(model.complete_requests[0].messages) == 2
    assert model.complete_requests[0].messages[0].role == "system"


@pytest.mark.asyncio
async def test_cancellation_after_model_await_prevents_candidate_publication(
    tmp_path: Path,
) -> None:
    token = CancellationToken()
    raw = plan_json()
    model = FakeModel([raw], on_complete=token.cancel)
    store = LocalArtifactStore(tmp_path)
    planner = FixedPlanner(
        model=model,
        artifact_store=store,
        budget=RunBudget.preset("medium"),
    )

    with pytest.raises(OperationCancelled):
        await planner.create_plan(
            research_request(), deadline=future_deadline(), cancellation_token=token
        )

    artifact_id = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert store.exists(artifact_id) is False


@pytest.mark.asyncio
async def test_expired_and_nonfinite_deadlines_do_not_call_model(tmp_path: Path) -> None:
    model = FakeModel([plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )

    for deadline in (time.monotonic() - 1.0, float("inf"), float("nan")):
        with pytest.raises(ProviderError) as error:
            await planner.create_plan(
                research_request(),
                deadline=deadline,
                cancellation_token=CancellationToken(),
            )
        assert error.value.code in {"INVALID_REQUEST", "TIMEOUT"}
    assert model.complete_calls == 0
