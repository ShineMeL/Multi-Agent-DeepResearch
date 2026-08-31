from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from deepresearch.domain import (
    FreshnessRequirement,
    ResearchPlan,
    ResearchRequest,
    ResourceUsage,
    RunBudget,
)
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


def model_usage(total_tokens: int, *, cached_tokens: int = 0) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=total_tokens,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=0,
    )


class FakeModel:
    provider_id = "fake-provider"

    def __init__(
        self,
        complete_outputs: Sequence[str | Exception],
        *,
        query_outputs: list[object] | None = None,
        on_complete: Callable[[], None] | None = None,
        on_structured: Callable[[], None] | None = None,
        complete_usages: list[ResourceUsage] | None = None,
        query_usages: list[ResourceUsage] | None = None,
    ) -> None:
        self.complete_outputs = list(complete_outputs)
        self.query_outputs = query_outputs or [["planner optimization"]]
        self.on_complete = on_complete
        self.on_structured = on_structured
        self.complete_usages = complete_usages or []
        self.query_usages = query_usages or []
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
        if isinstance(output, Exception):
            raise output
        if self.on_complete is not None:
            self.on_complete()
        return ModelResult[str](
            output=output,
            usage=(
                self.complete_usages.pop(0)
                if self.complete_usages
                else ResourceUsage.zero()
            ),
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
        if self.on_structured is not None:
            self.on_structured()
        output = output_schema.model_validate({"queries": value})
        return StructuredModelResult[Any](
            output=output,
            usage=(
                self.query_usages.pop(0)
                if self.query_usages
                else ResourceUsage.zero()
            ),
            provider_id=self.provider_id,
            model_id="fake-model",
            raw_response_artifact_id="sha256:" + "2" * 64,
            output_schema_hash="3" * 64,
        )

    def stream(self, *args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise NotImplementedError


class BlockingQueryModel(FakeModel):
    def __init__(self, complete_outputs: list[str]) -> None:
        super().__init__(complete_outputs)
        self.query_started = asyncio.Event()
        self.release_query = asyncio.Event()

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
        self.query_started.set()
        await self.release_query.wait()
        return StructuredModelResult[Any](
            output=output_schema.model_validate({"queries": ["shared"]}),
            usage=ResourceUsage.zero(),
            provider_id=self.provider_id,
            model_id="fake-model",
            raw_response_artifact_id="sha256:" + "2" * 64,
            output_schema_hash="3" * 64,
        )


class ConcurrentQueryModel(FakeModel):
    def __init__(self, *, usage: ResourceUsage | None = None) -> None:
        super().__init__([], query_outputs=[])
        self.started = asyncio.Event()
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()
        self.usage = usage or ResourceUsage.zero()

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
        self.started.set()
        if self.structured_calls == 2:
            self.all_started.set()
        await self.release.wait()
        return StructuredModelResult[Any](
            output=output_schema.model_validate({"queries": ["concurrent"]}),
            usage=self.usage,
            provider_id=self.provider_id,
            model_id="fake-model",
            raw_response_artifact_id="sha256:" + "4" * 64,
            output_schema_hash="5" * 64,
        )


class MixedConcurrentQueryModel(FakeModel):
    def __init__(self, *, success_usage: ResourceUsage) -> None:
        super().__init__([], query_outputs=[])
        self.success_usage = success_usage
        self.success_started = asyncio.Event()
        self.failure_started = asyncio.Event()
        self.release_success = asyncio.Event()
        self.release_failure = asyncio.Event()

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
        if self.structured_calls == 1:
            self.success_started.set()
            await self.release_success.wait()
            return StructuredModelResult[Any](
                output=output_schema.model_validate({"queries": ["overrun"]}),
                usage=self.success_usage,
                provider_id=self.provider_id,
                model_id="fake-model",
                raw_response_artifact_id="sha256:" + "6" * 64,
                output_schema_hash="7" * 64,
            )
        self.failure_started.set()
        await self.release_failure.wait()
        raise ProviderError(
            code="NETWORK",
            provider=self.provider_id,
            operation="structured",
            public_message="failure without usage",
            retryable=True,
        )


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
async def test_duplicate_json_names_keep_exact_artifact_and_repair_with_null(
    tmp_path: Path,
) -> None:
    raw = plan_json().replace(
        '"plan_id":"plan-1"',
        '"plan_id":"first","plan_id":"plan-1"',
        1,
    )
    model = FakeModel([raw, plan_json()])
    store = LocalArtifactStore(tmp_path)
    planner = FixedPlanner(
        model=model,
        artifact_store=store,
        budget=RunBudget.preset("medium"),
    )

    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    repair_payload = json.loads(model.complete_requests[1].messages[-1].content)
    artifact_id = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert plan.plan_id == "plan-1"
    assert model.complete_calls == 2
    assert store.get_bytes(artifact_id) == raw.encode("utf-8")
    assert repair_payload == {
        "candidate": None,
        "error_codes": ["MALFORMED_JSON"],
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


@pytest.mark.asyncio
async def test_fixed_planner_validates_with_its_configured_search_depth(
    tmp_path: Path,
) -> None:
    candidate = json.loads(plan_json())
    template = candidate["subquestions"][0]
    candidate["subquestions"] = []
    for index in range(5):
        item = json.loads(json.dumps(template))
        item["id"] = f"sq-{index}"
        item["information_needs"][0]["need_id"] = f"need-{index}"
        candidate["subquestions"].append(item)
    raw = json.dumps(candidate, separators=(",", ":"), sort_keys=True)
    shallow_model = FakeModel([raw])
    shallow = FixedPlanner(
        model=shallow_model,
        artifact_store=LocalArtifactStore(tmp_path / "shallow"),
        budget=RunBudget.preset("medium"),
        search_depth=1,
    )
    deep_model = FakeModel([raw, raw])
    deep = FixedPlanner(
        model=deep_model,
        artifact_store=LocalArtifactStore(tmp_path / "deep"),
        budget=RunBudget.preset("medium"),
        search_depth=2,
    )

    plan = await shallow.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    with pytest.raises(PlanGenerationError):
        await deep.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert len(plan.subquestions) == 5
    assert shallow_model.complete_calls == 1
    assert deep_model.complete_calls == 2


@pytest.mark.asyncio
async def test_same_key_waiters_observe_own_deadline_and_cancellation(
    tmp_path: Path,
) -> None:
    model = BlockingQueryModel([plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )
    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    owner = asyncio.create_task(
        planner.queries_for(
            plan.subquestions[0],
            plan_id=plan.plan_id,
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    )
    await model.query_started.wait()
    cancelled_token = CancellationToken()
    timeout_waiter = asyncio.create_task(
        planner.queries_for(
            plan.subquestions[0],
            plan_id=plan.plan_id,
            deadline=time.monotonic() + 0.04,
            cancellation_token=CancellationToken(),
        )
    )
    cancelled_waiter = asyncio.create_task(
        planner.queries_for(
            plan.subquestions[0],
            plan_id=plan.plan_id,
            deadline=future_deadline(),
            cancellation_token=cancelled_token,
        )
    )
    await asyncio.sleep(0.02)
    cancelled_token.cancel()

    try:
        with pytest.raises(ProviderError) as timeout_error:
            await asyncio.wait_for(timeout_waiter, timeout=0.3)
        with pytest.raises(OperationCancelled):
            await asyncio.wait_for(cancelled_waiter, timeout=0.3)
        assert timeout_error.value.code == "TIMEOUT"
        assert owner.done() is False
        assert model.structured_calls == 1
    finally:
        model.release_query.set()
        assert await owner == ("shared",)

    cached = await planner.queries_for(
        plan.subquestions[0],
        plan_id=plan.plan_id,
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert cached == ("shared",)
    assert model.structured_calls == 1


@pytest.mark.asyncio
async def test_repair_keys_and_query_plan_id_are_bounded_as_external_strings(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def boundary(text: str) -> str:
        seen.append(text)
        return f"BOUND<{text}>"

    initial = json.dumps({"ATTACK_KEY": {"nested": "PAYLOAD"}})
    model = FakeModel([initial, plan_json()], query_outputs=[["query"]])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=boundary,
    )

    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    repair_payload = json.loads(model.complete_requests[1].messages[-1].content)

    assert {"ATTACK_KEY", "nested", "PAYLOAD"} <= set(seen)
    assert repair_payload["candidate"] == {
        "BOUND<ATTACK_KEY>": {"BOUND<nested>": "BOUND<PAYLOAD>"}
    }
    seen.clear()

    await planner.queries_for(
        plan.subquestions[0],
        plan_id="UNTRUSTED_PLAN_ID",
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    query_payload = json.loads(model.structured_requests[0].messages[-1].content)

    assert "UNTRUSTED_PLAN_ID" in seen
    assert query_payload["plan_id"] == "BOUND<UNTRUSTED_PLAN_ID>"


@pytest.mark.asyncio
async def test_boundary_mapping_key_collisions_are_rejected_before_repair_prompt(
    tmp_path: Path,
) -> None:
    def boundary(text: str) -> str:
        if text in {"first-key", "second-key"}:
            return "COLLISION"
        return f"BOUND<{text}>"

    initial = json.dumps({"first-key": 1, "second-key": 2})
    model = FakeModel([initial, plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=boundary,
    )

    with pytest.raises(ValueError, match="duplicate mapping keys"):
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert model.complete_calls == 1


@pytest.mark.asyncio
async def test_deep_candidate_is_repaired_without_raw_parser_failure(tmp_path: Path) -> None:
    deep_candidate = "[" * 2_000 + "]" * 2_000
    model = FakeModel([deep_candidate, plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )

    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    repair_payload = json.loads(model.complete_requests[1].messages[-1].content)
    assert plan.plan_id == "plan-1"
    assert model.complete_calls == 2
    assert repair_payload["candidate"] is None
    assert repair_payload["error_codes"] == ["MALFORMED_JSON"]


@pytest.mark.asyncio
async def test_nonfinite_json_number_is_repaired_without_serialization_failure(
    tmp_path: Path,
) -> None:
    model = FakeModel(["1e10000", plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )

    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    repair_payload = json.loads(model.complete_requests[1].messages[-1].content)
    assert plan.plan_id == "plan-1"
    assert repair_payload["candidate"] is None
    assert repair_payload["error_codes"] == ["INVALID_SCHEMA"]


@pytest.mark.asyncio
async def test_oversized_rejected_candidate_is_not_republished_in_repair_prompt(
    tmp_path: Path,
) -> None:
    raw = json.dumps("x" * 900_000)
    model = FakeModel([raw, plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )

    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    repair_prompt = model.complete_requests[1].messages[-1].content
    repair_payload = json.loads(repair_prompt)
    assert plan.plan_id == "plan-1"
    assert repair_payload["candidate"] is None
    assert len(repair_prompt.encode("utf-8")) < 2_000


@pytest.mark.asyncio
async def test_boundary_expansion_cannot_publish_an_over_budget_initial_prompt(
    tmp_path: Path,
) -> None:
    model = FakeModel([plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=lambda text: text * 50_000,
    )

    with pytest.raises(ProviderError) as error:
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert error.value.retryable is False
    assert "Compare planner optimization methods" not in error.value.public_message
    assert model.complete_calls == 0


@pytest.mark.asyncio
async def test_boundary_expansion_cannot_publish_or_poison_an_over_budget_query(
    tmp_path: Path,
) -> None:
    state = {"expand": False}

    def boundary(text: str) -> str:
        if state["expand"] and text == "plan-1":
            return text * 50_000
        return text

    model = FakeModel([plan_json()], query_outputs=[["bounded query"]])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=boundary,
    )
    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    state["expand"] = True

    with pytest.raises(ProviderError) as error:
        await planner.queries_for(
            plan.subquestions[0],
            plan_id=plan.plan_id,
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert model.structured_calls == 0

    state["expand"] = False
    assert await planner.queries_for(
        plan.subquestions[0],
        plan_id=plan.plan_id,
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    ) == ("bounded query",)
    assert model.structured_calls == 1


@pytest.mark.asyncio
async def test_boundary_expansion_uses_null_for_an_over_budget_repair_candidate(
    tmp_path: Path,
) -> None:
    def boundary(text: str) -> str:
        if text in {"bad", "value"}:
            return text * 50_000
        return text

    model = FakeModel([json.dumps({"bad": "value"}), plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=boundary,
    )

    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    repair_payload = json.loads(model.complete_requests[1].messages[-1].content)
    assert plan.plan_id == "plan-1"
    assert repair_payload["candidate"] is None


@pytest.mark.asyncio
async def test_repair_candidate_fallback_uses_current_local_token_ledger(
    tmp_path: Path,
) -> None:
    rejected = json.dumps("y" * 1_500)
    model = FakeModel(
        [rejected, plan_json()],
        complete_usages=[model_usage(30_000), ResourceUsage.zero()],
    )
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=lambda text: (
            "x" * 30_000
            if text == "Compare planner optimization methods."
            else text
        ),
    )

    plan = await planner.create_plan(
        research_request(),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    repair_payload = json.loads(model.complete_requests[1].messages[-1].content)
    assert plan.plan_id == "plan-1"
    assert model.complete_calls == 2
    assert repair_payload["candidate"] is None
    assert repair_payload["error_codes"] == ["INVALID_SCHEMA"]


@pytest.mark.asyncio
async def test_provider_invalid_request_during_repair_does_not_trigger_null_retry(
    tmp_path: Path,
) -> None:
    provider_failure = ProviderError(
        code="INVALID_REQUEST",
        provider="fake-provider",
        operation="complete",
        public_message="provider rejected repair",
        retryable=False,
    )
    model = FakeModel([json.dumps({"invalid": "candidate"}), provider_failure, plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )

    with pytest.raises(ProviderError, match="provider rejected repair") as error:
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert model.complete_calls == 2


@pytest.mark.asyncio
async def test_non_utf8_boundary_output_is_rejected_before_prompt_publication(
    tmp_path: Path,
) -> None:
    model = FakeModel([plan_json()])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=lambda text: f"{text}\ud800",
    )

    with pytest.raises(ProviderError) as error:
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert error.value.retryable is False
    assert model.complete_calls == 0


@pytest.mark.asyncio
async def test_completed_initial_usage_prevents_over_budget_repair_call(
    tmp_path: Path,
) -> None:
    model = FakeModel(
        ["not json", plan_json()],
        complete_usages=[model_usage(31_500), ResourceUsage.zero()],
    )
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
        content_boundary=lambda text: (
            "x" * 30_000
            if text == "Compare planner optimization methods."
            else text
        ),
    )

    with pytest.raises(ProviderError) as error:
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert model.complete_calls == 1


@pytest.mark.asyncio
async def test_serial_query_usage_and_cache_are_accounted_locally(
    tmp_path: Path,
) -> None:
    model = FakeModel(
        [],
        query_outputs=[["first"], ["must not run"]],
        query_usages=[model_usage(3_500), ResourceUsage.zero()],
    )
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 6_000}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())

    first = await planner.queries_for(
        plan.subquestions[0],
        plan_id="plan-a",
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    cached = await planner.queries_for(
        plan.subquestions[0],
        plan_id="plan-a",
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    with pytest.raises(ProviderError) as error:
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-b",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert first == cached == ("first",)
    assert error.value.code == "INVALID_REQUEST"
    assert model.structured_calls == 1


@pytest.mark.asyncio
async def test_parallel_query_keys_reserve_budget_atomically(tmp_path: Path) -> None:
    model = ConcurrentQueryModel()
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 4_000}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())
    owner = asyncio.create_task(
        planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-a",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    )
    await model.started.wait()

    try:
        with pytest.raises(ProviderError) as error:
            await asyncio.wait_for(
                planner.queries_for(
                    plan.subquestions[0],
                    plan_id="plan-b",
                    deadline=future_deadline(),
                    cancellation_token=CancellationToken(),
                ),
                timeout=0.2,
            )
        assert error.value.code == "INVALID_REQUEST"
        assert owner.done() is False
        assert model.structured_calls == 1
    finally:
        model.release.set()
        assert await owner == ("concurrent",)


@pytest.mark.asyncio
async def test_usage_larger_than_reservation_is_fully_charged(tmp_path: Path) -> None:
    model = FakeModel(
        [],
        query_outputs=[["first"], ["must not run"]],
        query_usages=[model_usage(4_500), ResourceUsage.zero()],
    )
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 4_000}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())

    with pytest.raises(ProviderError) as overrun_error:
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-a",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    with pytest.raises(ProviderError):
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-b",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert overrun_error.value.code == "INVALID_REQUEST"
    assert model.structured_calls == 1
    assert vars(planner)["_model_tokens_used"] == 4_500
    assert vars(planner)["_query_cache"] == {}


@pytest.mark.asyncio
async def test_over_budget_valid_plan_is_audited_but_not_returned(
    tmp_path: Path,
) -> None:
    raw = plan_json()
    model = FakeModel([raw], complete_usages=[model_usage(21_000)])
    store = LocalArtifactStore(tmp_path)
    planner = FixedPlanner(
        model=model,
        artifact_store=store,
        budget=RunBudget.preset("low"),
    )

    with pytest.raises(ProviderError) as error:
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    artifact_id = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert error.value.code == "INVALID_REQUEST"
    assert model.complete_calls == 1
    assert store.get_bytes(artifact_id) == raw.encode("utf-8")
    assert vars(planner)["_model_tokens_used"] == 21_000


@pytest.mark.asyncio
async def test_concurrent_usage_overruns_are_charged_without_publishing_results(
    tmp_path: Path,
) -> None:
    model = ConcurrentQueryModel(usage=model_usage(4_000))
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 6_100}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())
    tasks = tuple(
        asyncio.create_task(
            planner.queries_for(
                plan.subquestions[0],
                plan_id=plan_id,
                deadline=future_deadline(),
                cancellation_token=CancellationToken(),
            )
        )
        for plan_id in ("plan-a", "plan-b")
    )
    await asyncio.wait_for(model.all_started.wait(), timeout=0.3)
    model.release.set()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(
        isinstance(result, ProviderError) and result.code == "INVALID_REQUEST"
        for result in results
    )
    assert vars(planner)["_model_tokens_used"] == 8_000
    assert vars(planner)["_model_tokens_reserved"] == 0
    assert vars(planner)["_query_cache"] == {}
    with pytest.raises(ProviderError) as later_error:
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-c",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    assert later_error.value.code == "INVALID_REQUEST"
    assert model.structured_calls == 2


@pytest.mark.asyncio
async def test_overrun_cannot_publish_while_an_inflight_reservation_later_fails(
    tmp_path: Path,
) -> None:
    model = MixedConcurrentQueryModel(success_usage=model_usage(4_000))
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 6_100}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())
    success = asyncio.create_task(
        planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-success",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    )
    await asyncio.wait_for(model.success_started.wait(), timeout=0.3)
    failure = asyncio.create_task(
        planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-failure",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    )
    await asyncio.wait_for(model.failure_started.wait(), timeout=0.3)
    model.release_success.set()
    success_result = (await asyncio.gather(success, return_exceptions=True))[0]
    model.release_failure.set()
    failure_result = (await asyncio.gather(failure, return_exceptions=True))[0]

    assert isinstance(success_result, ProviderError)
    assert success_result.code == "INVALID_REQUEST"
    assert isinstance(failure_result, ProviderError)
    assert failure_result.code == "NETWORK"
    assert failure_result.usage is None
    assert vars(planner)["_model_tokens_used"] == 4_000
    assert vars(planner)["_model_tokens_reserved"] == 0
    assert vars(planner)["_query_cache"] == {}
    with pytest.raises(ProviderError):
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-later",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    assert model.structured_calls == 2


@pytest.mark.asyncio
async def test_usage_bearing_provider_error_is_charged_before_later_calls(
    tmp_path: Path,
) -> None:
    failure = ProviderError(
        code="NETWORK",
        provider="fake-provider",
        operation="structured",
        public_message="sanitized failure",
        retryable=True,
        usage=model_usage(5_000),
    )
    model = FakeModel([], query_outputs=[failure, ["must not run"]])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 6_000}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())

    with pytest.raises(ProviderError, match="sanitized failure"):
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-a",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )
    with pytest.raises(ProviderError) as budget_error:
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-b",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert budget_error.value.code == "INVALID_REQUEST"
    assert model.structured_calls == 1


@pytest.mark.asyncio
async def test_completed_cancelled_query_usage_remains_charged(tmp_path: Path) -> None:
    token = CancellationToken()
    model = FakeModel(
        [],
        query_outputs=[["cancelled"], ["must not run"]],
        query_usages=[model_usage(5_000), ResourceUsage.zero()],
        on_structured=token.cancel,
    )
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 6_000}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())

    with pytest.raises(OperationCancelled):
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-a",
            deadline=future_deadline(),
            cancellation_token=token,
        )
    with pytest.raises(ProviderError) as error:
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="plan-b",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert model.structured_calls == 1


@pytest.mark.asyncio
async def test_cached_tokens_are_not_debited_twice(tmp_path: Path) -> None:
    model = FakeModel(
        [],
        query_outputs=[["first"], ["second"]],
        query_usages=[
            model_usage(3_000, cached_tokens=2_000),
            ResourceUsage.zero(),
        ],
    )
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium").model_copy(
            update={"max_total_tokens": 6_500}
        ),
    )
    plan = ResearchPlan.model_validate_json(plan_json())

    first = await planner.queries_for(
        plan.subquestions[0],
        plan_id="plan-a",
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )
    second = await planner.queries_for(
        plan.subquestions[0],
        plan_id="plan-b",
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert first == ("first",)
    assert second == ("second",)
    assert model.structured_calls == 2


@pytest.mark.asyncio
async def test_non_utf8_provider_output_is_sanitized_before_artifact_encoding(
    tmp_path: Path,
) -> None:
    model = FakeModel(["\ud800"])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )

    with pytest.raises(ProviderError) as error:
        await planner.create_plan(
            research_request(),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.retryable is False
    assert "codec" not in error.value.public_message.casefold()
    assert model.complete_calls == 1


@pytest.mark.asyncio
async def test_failed_unique_query_key_releases_its_gate(tmp_path: Path) -> None:
    failure = ProviderError(
        code="NETWORK",
        provider="fake-provider",
        operation="structured",
        public_message="temporary failure",
        retryable=True,
    )
    model = FakeModel([], query_outputs=[failure])
    planner = FixedPlanner(
        model=model,
        artifact_store=LocalArtifactStore(tmp_path),
        budget=RunBudget.preset("medium"),
    )
    plan = ResearchPlan.model_validate_json(plan_json())

    with pytest.raises(ProviderError):
        await planner.queries_for(
            plan.subquestions[0],
            plan_id="failed-key",
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert vars(planner)["_query_locks"] == {}
