from __future__ import annotations

import ast
import hashlib
import inspect
import json
from collections import UserDict
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from deepresearch.domain import (
    CoverageLedgerEntry,
    EvidenceRequirements,
    EvidenceSpan,
    FreshnessRequirement,
    HtmlLocator,
    InformationNeed,
    RerankScore,
    ResearchPlan,
    ResearchRequest,
    ResearchScope,
    ResourceUsage,
    RunBudget,
    SourceDocument,
    SubQuestion,
)
from deepresearch.reporting import identity_content_boundary
from deepresearch.runtime import BudgetAccountant, ResourceEstimate
from deepresearch.storage import ArtifactIntegrityError, LocalArtifactStore
from deepresearch.workflow import baseline_graph as baseline_graph_module
from deepresearch.workflow.baseline_graph import (
    BaselineDependencies,
    InvocationUsageObserver,
    WorkflowInvariantError,
    baseline_is_sufficient,
    build_baseline_graph,
    decide_baseline_stop,
    rank_baseline_coverage,
    route_after_decide,
)


def state():  # type: ignore[no-untyped-def]
    request = ResearchRequest(
        question="Which methods are documented?",
        output_requirements={"answer_shape": "brief"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="test",
        budget_preset="low",
    )
    return {
        "run_id": "run-1",
        "thread_id": "thread-1",
        "request": request,
        "config_sha256": "a" * 64,
        "plan_id": None,
        "plan_artifact_id": None,
        "pending_subquestion_ids": (),
        "active_subquestion_id": None,
        "query_ids": (),
        "source_ids": (),
        "evidence_ids": (),
        "selected_evidence_ids": (),
        "coverage_ledger": (),
        "high_priority_unresolved_conflict_ids": (),
        "blocked_needs": (),
        "recent_marginal_gains": (),
        "baseline_work_artifact_ids": (),
        "budget_snapshot": BudgetAccountant(RunBudget.preset("low")).snapshot(),
        "stop_reason": None,
        "is_partial": False,
        "draft_artifact_id": None,
        "report_artifact_id": None,
        "evidence_graph_artifact_id": None,
        "manifest_artifact_id": None,
        "next_event_seq": 1,
        "failed_node": None,
        "elapsed_wall_seconds": 0.0,
        "error_code": None,
    }


def test_runtime_shape_distinguishes_float_sign_and_key_runtime_type() -> None:
    class StringSubclass(str):
        pass

    same_shape = cast("Any", baseline_graph_module)._same_runtime_shape

    assert same_shape(-0.0, 0.0) is False
    assert same_shape({StringSubclass("key"): "value"}, {"key": "value"}) is False


def subquestion(
    identifier: str,
    *,
    importance: float,
    minimum_sources: int = 2,
) -> SubQuestion:
    return SubQuestion(
        id=identifier,
        question=f"Question {identifier}",
        rationale_code="coverage",
        importance=importance,
        dependencies=(),
        information_needs=(
            InformationNeed(
                need_id=f"need-{identifier}",
                text=f"Need {identifier}",
                importance=1.0,
            ),
        ),
        evidence_requirements=EvidenceRequirements(
            min_independent_sources=minimum_sources,
            allowed_source_types=frozenset({"paper"}),
            must_include_primary=False,
        ),
        status="pending",
    )


def plan(*, zero_importance: bool = False) -> ResearchPlan:
    return ResearchPlan(
        plan_id="plan-1",
        scope=ResearchScope(
            included_topics=("topic",),
            excluded_topics=(),
            date_range=None,
            answer_shape="brief",
        ),
        subquestions=(
            subquestion("sq-1", importance=0.0 if zero_importance else 0.6),
            subquestion("sq-2", importance=0.0 if zero_importance else 0.4),
        ),
        created_by_model="fake-model",
        prompt_version="fixed-planner-v1",
    )


def ledger_entry(
    identifier: str,
    *,
    coverage: float = 0.9,
    sources: int = 2,
    gain: float = 0.1,
) -> CoverageLedgerEntry:
    return CoverageLedgerEntry(
        subquestion_id=identifier,
        coverage_score=coverage,
        independent_source_count=sources,
        unresolved_conflict_ids=(),
        uncertainty_score=1.0 - coverage,
        last_marginal_gain=gain,
        evidence_ids=(f"E-{identifier}",),
        attempt_count=1,
        last_decision_code="RANKED",
    )


def rich_state() -> dict[str, object]:
    current = state()
    current.update(
        {
            "pending_subquestion_ids": ("sq-1",),
            "query_ids": ("query-1",),
            "source_ids": ("source-1",),
            "evidence_ids": ("evidence-1",),
            "selected_evidence_ids": ("evidence-1",),
            "coverage_ledger": (ledger_entry("sq-1"),),
            "high_priority_unresolved_conflict_ids": ("conflict-1",),
            "blocked_needs": (
                {
                    "need_id": "need-1",
                    "required_source_unavailable": True,
                    "alternative_strategies_exhausted": False,
                    "retry_count": 1,
                    "max_retries": 2,
                },
            ),
            "recent_marginal_gains": (0.25, 0.125),
            "baseline_work_artifact_ids": ("sha256:" + "a" * 64,),
            "elapsed_wall_seconds": 1.5,
        }
    )
    accountant = BudgetAccountant(RunBudget.preset("low"))
    reservation = accountant.reserve(
        ResourceEstimate(tokens=3, wall_seconds=0.5, cost_usd=Decimal("0.01")),
        node="Planner",
        idempotency_key="rich-state",
    )
    current["budget_snapshot"] = accountant.settle(
        reservation,
        actual=ResourceUsage(
            input_tokens=2,
            output_tokens=1,
            reasoning_tokens=0,
            cached_tokens=0,
            total_tokens=3,
            search_calls=0,
            pages=0,
            retries=0,
            wall_seconds=0.5,
            cost_usd=Decimal("0.01"),
        ),
    )
    return current


def _state_payload(value: object) -> dict[str, object]:
    return cast("Any", baseline_graph_module)._state_payload(value)


def _state_from_payload(value: object) -> dict[str, object]:
    return cast("Any", baseline_graph_module)._state_from_payload(value)


def test_state_receipt_codec_round_trips_rich_state_canonically() -> None:
    current = rich_state()

    payload = _state_payload(current)
    restored = _state_from_payload(payload)
    second_payload = _state_payload(restored)

    assert restored == current
    assert type(restored["request"]) is ResearchRequest
    assert type(cast("tuple[object, ...]", restored["coverage_ledger"])[0]) is CoverageLedgerEntry
    assert type(restored["budget_snapshot"]) is type(current["budget_snapshot"])
    assert type(restored["query_ids"]) is tuple
    assert type(restored["blocked_needs"]) is tuple
    canonical = cast("Any", baseline_graph_module)._canonical_bytes
    assert canonical(payload) == canonical(second_payload)


@pytest.mark.parametrize(
    ("location", "remove"),
    [
        (("unexpected",), False),
        (("query_ids",), True),
        (("blocked_needs", 0, "unexpected"), False),
        (("blocked_needs", 0, "max_retries"), True),
        (("request", "unexpected"), False),
        (("coverage_ledger", 0, "unexpected"), False),
        (("budget_snapshot", "unexpected"), False),
    ],
)
def test_state_receipt_codec_rejects_extra_or_missing_fields(
    location: tuple[str | int, ...],
    remove: bool,
) -> None:
    payload: object = deepcopy(_state_payload(rich_state()))
    parent = payload
    for part in location[:-1]:
        parent = cast("Any", parent)[part]
    key = location[-1]
    if remove:
        del cast("Any", parent)[key]
    else:
        cast("Any", parent)[key] = "collision"

    with pytest.raises(ArtifactIntegrityError):
        _state_from_payload(payload)


def test_state_receipt_codec_revives_json_arrays_and_domain_dictionaries() -> None:
    payload = _state_payload(rich_state())
    assert type(payload["query_ids"]) is list
    assert type(payload["request"]) is dict
    assert type(cast("list[object]", payload["coverage_ledger"])[0]) is dict
    assert type(payload["budget_snapshot"]) is dict

    restored = _state_from_payload(payload)

    assert type(restored["query_ids"]) is tuple
    assert type(restored["request"]) is ResearchRequest
    assert type(cast("tuple[object, ...]", restored["coverage_ledger"])[0]) is CoverageLedgerEntry
    assert type(restored["budget_snapshot"]) is type(rich_state()["budget_snapshot"])


@pytest.mark.parametrize(
    ("location", "bad_value"),
    [
        (("next_event_seq",), True),
        (("is_partial",), 1),
        (("elapsed_wall_seconds",), "1.5"),
        (("recent_marginal_gains", 0), "0.25"),
        (("blocked_needs", 0, "retry_count"), True),
        (("blocked_needs", 0, "max_retries"), "2"),
        (("budget_snapshot", "used_tokens"), "3"),
    ],
)
def test_state_receipt_codec_rejects_scalar_coercion_collisions(
    location: tuple[str | int, ...],
    bad_value: object,
) -> None:
    payload: object = deepcopy(_state_payload(rich_state()))
    parent = payload
    for part in location[:-1]:
        parent = cast("Any", parent)[part]
    cast("Any", parent)[location[-1]] = bad_value

    with pytest.raises(ArtifactIntegrityError):
        _state_from_payload(payload)


def test_state_receipt_codec_rejects_preconstructed_and_subclass_values() -> None:
    class ExtendedRequest(ResearchRequest):
        private_extension: str

    class StringSubclass(str):
        pass

    class IntSubclass(int):
        pass

    current = rich_state()
    extended = ExtendedRequest.model_validate(
        {
            **cast("ResearchRequest", current["request"]).model_dump(round_trip=True),
            "private_extension": "must-not-project",
        }
    )
    current["request"] = extended
    with pytest.raises(ArtifactIntegrityError):
        _state_payload(current)

    payload = _state_payload(rich_state())
    candidates: list[object] = []
    preconstructed = deepcopy(payload)
    cast("dict[str, object]", preconstructed)["request"] = rich_state()["request"]
    candidates.append(preconstructed)
    with_tuple = deepcopy(payload)
    cast("dict[str, object]", with_tuple)["query_ids"] = ("query-1",)
    candidates.append(with_tuple)
    with_string_subclass = deepcopy(payload)
    cast("dict[str, object]", with_string_subclass)["run_id"] = StringSubclass("run-1")
    candidates.append(with_string_subclass)
    with_int_subclass = deepcopy(payload)
    cast("dict[str, object]", with_int_subclass)["next_event_seq"] = IntSubclass(1)
    candidates.append(with_int_subclass)
    candidates.append(UserDict(payload))

    for candidate in candidates:
        with pytest.raises(ArtifactIntegrityError):
            _state_from_payload(candidate)


def test_state_receipt_encoder_rejects_subclass_and_constructed_projection() -> None:
    class StringSubclass(str):
        pass

    class TupleSubclass(tuple):
        pass

    string_state = rich_state()
    string_state["run_id"] = StringSubclass("run-1")
    tuple_state = rich_state()
    tuple_state["query_ids"] = TupleSubclass(("query-1",))

    budget_state = rich_state()
    snapshot = cast("Any", budget_state["budget_snapshot"])
    corrupt_usage = cast(
        "ResourceUsage",
        BaseModel.model_copy(
            snapshot.used_by_node["Tool"],
            update={"wall_seconds": 0},
        ),
    )
    mapping_type = type(snapshot.used_by_node)
    corrupt_mapping = mapping_type(snapshot.used_by_node)
    cast("Any", dict).__setitem__(corrupt_mapping, "Tool", corrupt_usage)
    budget_state["budget_snapshot"] = BaseModel.model_copy(
        snapshot,
        update={"used_by_node": corrupt_mapping},
    )

    for candidate in (string_state, tuple_state, budget_state):
        with pytest.raises(ArtifactIntegrityError):
            _state_payload(candidate)


def test_state_encoder_rejects_nested_models_before_projection() -> None:
    class StringSubclass(str):
        pass

    observations: list[str] = []

    class ObservedDict(dict[str, object]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            observations.append("iter")
            return super().__iter__()

        def __getitem__(self, key: str) -> object:
            observations.append("getitem")
            return super().__getitem__(key)

    observed_state = rich_state()
    observed_request = cast("ResearchRequest", observed_state["request"])
    observed_state["request"] = cast(
        "ResearchRequest",
        BaseModel.model_copy(
            observed_request,
            update={"output_requirements": ObservedDict({"shape": "brief"})},
        ),
    )
    with pytest.raises(ArtifactIntegrityError):
        _state_payload(observed_state)
    assert observations == []

    question_state = rich_state()
    question_request = cast("ResearchRequest", question_state["request"])
    question_state["request"] = cast(
        "ResearchRequest",
        BaseModel.model_copy(
            question_request,
            update={"question": StringSubclass("question")},
        ),
    )
    ledger_state = rich_state()
    entry = cast("tuple[CoverageLedgerEntry, ...]", ledger_state["coverage_ledger"])[0]
    ledger_state["coverage_ledger"] = (
        cast(
            "CoverageLedgerEntry",
            BaseModel.model_copy(entry, update={"evidence_ids": ["E-list"]}),
        ),
    )

    for candidate in (question_state, ledger_state):
        with pytest.raises(ArtifactIntegrityError):
            _state_payload(candidate)


def test_state_encoder_rejects_mapping_key_subclasses() -> None:
    class StringSubclass(str):
        pass

    top_level = rich_state()
    run_id = top_level.pop("run_id")
    dict.__setitem__(top_level, StringSubclass("run_id"), run_id)
    blocked = rich_state()
    need = dict(cast("tuple[dict[str, object], ...]", blocked["blocked_needs"])[0])
    need_id = need.pop("need_id")
    dict.__setitem__(need, StringSubclass("need_id"), need_id)
    blocked["blocked_needs"] = (need,)

    for candidate in (top_level, blocked):
        with pytest.raises(ArtifactIntegrityError):
            _state_payload(candidate)


def test_state_codec_rejects_inconsistent_budget_totals_in_both_directions() -> None:
    live = rich_state()
    snapshot = cast("Any", live["budget_snapshot"])
    live["budget_snapshot"] = BaseModel.model_copy(
        snapshot,
        update={"used_tokens": snapshot.used_tokens + 1},
    )
    with pytest.raises(ArtifactIntegrityError):
        _state_payload(live)

    payload = _state_payload(rich_state())
    cast("dict[str, object]", payload["budget_snapshot"])["used_tokens"] = 1
    with pytest.raises(ArtifactIntegrityError):
        _state_from_payload(payload)


def test_state_encoder_rejects_negative_nested_budget_cost_and_reservations() -> None:
    negative = rich_state()
    snapshot = cast("Any", negative["budget_snapshot"])
    invalid_usage = BaseModel.model_copy(
        snapshot.used_by_node["Tool"],
        update={"cost_usd": Decimal("-0.01")},
    )
    mapping_type = type(snapshot.used_by_node)
    rows = mapping_type(snapshot.used_by_node)
    dict.__setitem__(rows, "Tool", invalid_usage)
    negative["budget_snapshot"] = BaseModel.model_copy(
        snapshot,
        update={"used_by_node": rows},
    )
    reserved = rich_state()
    reserved_snapshot = cast("Any", reserved["budget_snapshot"])
    reserved["budget_snapshot"] = BaseModel.model_copy(
        reserved_snapshot,
        update={"reserved_tokens": 1},
    )

    for candidate in (negative, reserved):
        with pytest.raises(ArtifactIntegrityError):
            _state_payload(candidate)


def test_state_encoder_maps_missing_budget_node_to_integrity_error() -> None:
    current = rich_state()
    snapshot = cast("Any", current["budget_snapshot"])
    mapping_type = type(snapshot.used_by_node)
    rows = mapping_type(snapshot.used_by_node)
    dict.__delitem__(rows, "Tool")
    current["budget_snapshot"] = BaseModel.model_copy(
        snapshot,
        update={"used_by_node": rows},
    )

    with pytest.raises(ArtifactIntegrityError):
        _state_payload(current)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_state_receipt_codec_rejects_nonfinite_values_everywhere(
    bad_value: float,
) -> None:
    for location in (
        ("elapsed_wall_seconds",),
        ("recent_marginal_gains", 0),
        ("budget_snapshot", "used_wall_seconds"),
        ("request", "output_requirements", "unconstrained"),
    ):
        payload: object = deepcopy(_state_payload(rich_state()))
        parent = payload
        for part in location[:-1]:
            parent = cast("Any", parent)[part]
        cast("Any", parent)[location[-1]] = bad_value
        with pytest.raises(ArtifactIntegrityError):
            _state_from_payload(payload)


def _run_header_receipt() -> dict[str, object]:
    return {
        "schema_version": "baseline-audit-receipt-v1",
        "kind": "run-header",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "receipt_key": "header",
        "payload": {
            "code_commit": "a" * 40,
            "dependency_lock_sha256": "b" * 64,
            "graph_version": "baseline-v1",
            "pricing_snapshots": [],
            "pricing_status": "unknown",
            "provider_ids": ["offline"],
            "provider_profile_configuration_sha256": "c" * 64,
            "replay_parent": None,
            "seed_supported": True,
            "started_at": "2026-09-01T00:00:00Z",
            "workflow_id": "baseline-v1",
        },
    }


@pytest.mark.parametrize("variant", ["whitespace", "duplicate", "nonfinite"])
def test_audit_receipt_loader_rejects_noncanonical_or_collision_json(
    tmp_path: object,
    variant: str,
) -> None:
    store = LocalArtifactStore(cast("Any", tmp_path))
    receipt = _run_header_receipt()
    if variant == "whitespace":
        raw = json.dumps(receipt, sort_keys=False, indent=2).encode()
    elif variant == "duplicate":
        canonical = cast("Any", baseline_graph_module)._canonical_bytes(receipt)
        raw = canonical.replace(
            b'"run_id":"run-1"',
            b'"run_id":"ignored","run_id":"run-1"',
            1,
        )
    else:
        receipt["payload"] = {
            **cast("dict[str, object]", receipt["payload"]),
            "started_at": float("nan"),
        }
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ref = store.put_bytes(
        raw,
        media_type="application/vnd.deepresearch.baseline-audit+json",
    )

    with pytest.raises(ArtifactIntegrityError):
        cast("Any", baseline_graph_module)._load_audit_receipt(store, ref.artifact_id)


def test_audit_receipt_loader_ignores_another_canonical_artifact_schema(
    tmp_path: object,
) -> None:
    store = LocalArtifactStore(cast("Any", tmp_path))
    manifest_like = {
        "schema_version": "run-manifest-v1",
        "run_id": "run-1",
    }
    ref = store.put_bytes(
        json.dumps(
            manifest_like,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/json",
    )

    assert (
        cast("Any", baseline_graph_module)._load_audit_receipt(
            store,
            ref.artifact_id,
        )
        is None
    )


def test_audit_receipt_loader_rejects_duplicate_schema_discriminator_collision(
    tmp_path: object,
) -> None:
    store = LocalArtifactStore(cast("Any", tmp_path))
    ref = store.put_bytes(
        (
            b'{"schema_version":"baseline-audit-receipt-v1",'
            b'"schema_version":"run-manifest-v1"}'
        ),
        media_type="application/json",
    )

    with pytest.raises(ArtifactIntegrityError):
        cast("Any", baseline_graph_module)._load_audit_receipt(
            store,
            ref.artifact_id,
        )


def test_sufficiency_requires_exact_ledger_thresholds_sources_and_no_conflict() -> None:
    research_plan = plan()
    sufficient = (ledger_entry("sq-1"), ledger_entry("sq-2"))

    assert baseline_is_sufficient(research_plan, sufficient) is True
    assert (
        baseline_is_sufficient(
            research_plan,
            (ledger_entry("sq-1"), ledger_entry("sq-2", coverage=0.84)),
        )
        is False
    )
    assert (
        baseline_is_sufficient(
            research_plan,
            (ledger_entry("sq-1"), ledger_entry("sq-2", sources=1)),
        )
        is False
    )
    assert (
        baseline_is_sufficient(
            research_plan,
            sufficient,
            high_priority_unresolved_conflict_ids=("conflict-1",),
        )
        is False
    )
    assert baseline_is_sufficient(research_plan, sufficient[:1]) is False
    assert (
        baseline_is_sufficient(
            research_plan,
            (sufficient[0], sufficient[0], sufficient[1]),
        )
        is False
    )
    assert baseline_is_sufficient(plan(zero_importance=True), sufficient) is False


def test_decide_stop_uses_strict_precedence() -> None:
    research_plan = plan()
    current = state()
    current["coverage_ledger"] = (
        ledger_entry("sq-1"),
        ledger_entry("sq-2"),
    )
    current["recent_marginal_gains"] = (0.01, 0.02)
    current["blocked_needs"] = (
        {
            "need_id": "need-1",
            "required_source_unavailable": True,
            "alternative_strategies_exhausted": True,
            "retry_count": 2,
            "max_retries": 2,
        },
    )
    snapshot = current["budget_snapshot"]
    current["budget_snapshot"] = snapshot.model_copy(update={"exhausted": frozenset({"tokens"})})

    assert decide_baseline_stop(current, research_plan) == "SUFFICIENT"

    current["coverage_ledger"] = (
        ledger_entry("sq-1", coverage=0.7),
        ledger_entry("sq-2", coverage=0.7),
    )
    assert decide_baseline_stop(current, research_plan) == "BUDGET_EXHAUSTED"

    current["budget_snapshot"] = snapshot
    assert decide_baseline_stop(current, research_plan) == "PLATEAU"

    current["recent_marginal_gains"] = (0.1, 0.01)
    assert decide_baseline_stop(current, research_plan) == "BLOCKED"


def test_plateau_and_blocked_require_complete_typed_proof() -> None:
    research_plan = plan()
    current = state()
    current["pending_subquestion_ids"] = ()
    current["coverage_ledger"] = (
        ledger_entry("sq-1", coverage=0.5),
        ledger_entry("sq-2", coverage=0.5),
    )

    for gains in ((0.01,), (0.01, 0.05), (0.01, 0.1)):
        current["recent_marginal_gains"] = gains
        with pytest.raises(WorkflowInvariantError) as error:
            decide_baseline_stop(current, research_plan)
        assert error.value.code == "NO_LEGAL_CONTINUATION"

    current["recent_marginal_gains"] = (0.01, 0.02)
    assert decide_baseline_stop(current, research_plan) == "PLATEAU"

    current["recent_marginal_gains"] = (0.1, 0.1)
    current["blocked_needs"] = (
        {
            "need_id": "need-1",
            "required_source_unavailable": True,
            "alternative_strategies_exhausted": False,
            "retry_count": 2,
            "max_retries": 2,
        },
    )
    with pytest.raises(WorkflowInvariantError):
        decide_baseline_stop(current, research_plan)


def test_routes_search_until_queue_empty_and_then_draft() -> None:
    current = state()
    current["pending_subquestion_ids"] = ("sq-1",)
    assert route_after_decide(current) == "Search"

    current["pending_subquestion_ids"] = ()
    current["stop_reason"] = "SUFFICIENT"
    assert route_after_decide(current) == "DraftReport"

    current["stop_reason"] = None
    current["error_code"] = "PLAN_INVALID"
    assert route_after_decide(current) == "PersistResults"


def test_r1_coverage_uses_floor_top_three_and_distinct_source_families() -> None:
    research_plan = plan().model_copy(update={"subquestions": (plan().subquestions[0],)})
    sources = tuple(
        SourceDocument(
            source_id=f"S-{index}",
            canonical_url=f"https://example{family}.test/{index}",
            title=f"Source {index}",
            authors=(),
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
            content_hash=str(index) * 64,
            parsed_content_hash=str(index) * 64,
            source_type="paper",
            source_family_id=f"family-{family}",
            parser_version="parser-v1",
        )
        for index, family in ((1, 1), (2, 1), (3, 2), (4, 3))
    )
    evidence = tuple(
        EvidenceSpan(
            evidence_id=f"E-{index}",
            source_id=f"S-{index}",
            locator=HtmlLocator(
                paragraph_id=f"p-{index}",
                start_char=0,
                end_char=4,
            ),
            excerpt="text",
            excerpt_hash=hashlib.sha256(b"text").hexdigest(),
            language="en",
            information_need_ids=("need-sq-1",),
        )
        for index in range(1, 5)
    )
    scores = {
        "need-sq-1": tuple(
            RerankScore(
                evidence_id=f"E-{index}",
                total=score,
                feature_scores={"relevance": score},
                model_id="embed-v1",
            )
            for index, score in ((2, 0.95), (1, 0.95), (3, 0.85), (4, 0.54))
        )
    }

    selected, ledger = rank_baseline_coverage(
        research_plan,
        evidence,
        sources,
        scores,
        previous_ledger=(),
    )

    assert selected == ("E-1", "E-3")
    assert ledger[0].coverage_score == pytest.approx(0.95)
    assert ledger[0].independent_source_count == 2
    assert ledger[0].last_decision_code == "R1_DISTINCT_SOURCE_FAMILIES"


def test_invocation_usage_observer_is_runtime_checkable_and_destructive() -> None:
    values = [ResourceUsage.zero(cost_known=True), None]

    class Observer:
        def consume_invocation_usage(self) -> ResourceUsage | None:
            return values.pop(0)

    observer = Observer()
    assert isinstance(observer, InvocationUsageObserver)
    assert observer.consume_invocation_usage() == ResourceUsage.zero(cost_known=True)
    assert observer.consume_invocation_usage() is None


def test_baseline_dependencies_constructor_has_no_private_audit_composition() -> None:
    parameters = inspect.signature(BaselineDependencies).parameters

    assert "audit_composition" not in parameters
    assert not any(name.startswith("_") for name in parameters)


def test_invocation_audit_has_no_process_or_accountant_carrier() -> None:
    source_path = Path(baseline_graph_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    contextvar_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "contextvars" in ast.unparse(node)
    ]
    accountant_carriers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "_baseline_audit_buffer"
    ]
    audit_resolvers = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_audit_buffer"
    ]

    assert contextvar_imports == []
    assert accountant_carriers == []
    assert len(audit_resolvers) == 1
    resolver = audit_resolvers[0]
    assert len(resolver.body) == 1
    statement = resolver.body[0]
    assert isinstance(statement, ast.Return)
    assert isinstance(statement.value, ast.Attribute)
    assert statement.value.attr == "audit"
    assert isinstance(statement.value.value, ast.Name)
    assert statement.value.value.id == "context"


async def test_provider_agnostic_graph_has_exact_successful_node_order() -> None:
    calls: list[str] = []
    decision_count = 0

    def node(name: str):  # type: ignore[no-untyped-def]
        async def invoke(current: Mapping[str, object]) -> Mapping[str, object]:
            nonlocal decision_count
            del current
            calls.append(name)
            if name == "Plan":
                return {"plan_id": "plan-1", "pending_subquestion_ids": ("sq-1",)}
            if name == "DecideNext":
                decision_count += 1
                return (
                    {"active_subquestion_id": "sq-1"}
                    if decision_count == 1
                    else {"stop_reason": "SUFFICIENT", "pending_subquestion_ids": ()}
                )
            if name == "PersistResults":
                return {
                    "report_artifact_id": "sha256:" + "1" * 64,
                    "manifest_artifact_id": "sha256:" + "2" * 64,
                }
            return {}

        return invoke

    dependencies = BaselineDependencies(
        checkpointer=InMemorySaver(),
        validate_request=node("ValidateRequest"),
        plan=node("Plan"),
        decide_next=node("DecideNext"),
        search=node("Search"),
        fetch=node("Fetch"),
        parse_and_normalize=node("ParseAndNormalize"),
        store_evidence=node("StoreEvidence"),
        rank_evidence=node("RankEvidence"),
        content_boundary=identity_content_boundary,
        draft_report=node("DraftReport"),
        finalize_citations=node("FinalizeCitations"),
        persist_results=node("PersistResults"),
    )
    graph = build_baseline_graph(dependencies)

    output = await graph.ainvoke(
        cast("dict[str, object]", state()),
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert output["stop_reason"] == "SUFFICIENT"
    assert calls == [
        "ValidateRequest",
        "Plan",
        "DecideNext",
        "Search",
        "Fetch",
        "ParseAndNormalize",
        "StoreEvidence",
        "RankEvidence",
        "DecideNext",
        "DraftReport",
        "FinalizeCitations",
        "PersistResults",
    ]
