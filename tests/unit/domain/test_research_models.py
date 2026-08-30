from datetime import date
from hashlib import sha256
from typing import get_args

import pytest
from pydantic import ValidationError

from deepresearch.domain import (
    AccessProfile,
    DateRange,
    EvidenceRequirements,
    ExecutionMode,
    FreshnessRequirement,
    InformationNeed,
    ResearchPlan,
    ResearchRequest,
    ResearchScope,
    RunPurpose,
    SourceType,
    SubQuestion,
)


def information_need(need_id: str) -> InformationNeed:
    return InformationNeed(need_id=need_id, text=f"Need {need_id}", importance=0.5)


def subquestion(
    subquestion_id: str,
    *,
    dependencies: tuple[str, ...] = (),
    need_ids: tuple[str, ...] = ("need-1",),
) -> SubQuestion:
    return SubQuestion(
        id=subquestion_id,
        question=f"Question {subquestion_id}?",
        rationale_code="coverage",
        importance=0.8,
        dependencies=dependencies,
        information_needs=tuple(information_need(item) for item in need_ids),
        evidence_requirements=EvidenceRequirements(
            min_independent_sources=1,
            allowed_source_types=frozenset({"paper", "official_documentation"}),
            must_include_primary=False,
        ),
        status="pending",
    )


def research_plan(*subquestions: SubQuestion) -> ResearchPlan:
    return ResearchPlan(
        plan_id="plan-1",
        scope=ResearchScope(
            included_topics=("topic",),
            excluded_topics=(),
            date_range=None,
            answer_shape="brief",
        ),
        subquestions=subquestions,
        created_by_model="planner",
        prompt_version="v1",
    )


def test_public_literals_match_the_contract() -> None:
    assert set(get_args(ExecutionMode)) == {"live", "replay", "hybrid"}
    assert set(get_args(AccessProfile)) == {"showcase", "public_live", "local"}
    assert set(get_args(RunPurpose)) == {"demo", "benchmark", "test"}
    assert set(get_args(SourceType)) == {
        "paper",
        "official_documentation",
        "standard",
        "primary_data",
        "first_party_statement",
        "secondary_analysis",
        "news",
        "unknown",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "none", "published_after": "2026-01-01"},
        {"kind": "published_after"},
        {
            "kind": "retrieved_within_days",
            "published_after": "2026-01-01",
            "retrieved_within_days": 7,
        },
        {"kind": "retrieved_within_days", "retrieved_within_days": 0},
    ],
)
def test_freshness_payload_must_match_discriminator(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="freshness|greater than or equal"):
        FreshnessRequirement.model_validate(payload)


def test_date_range_rejects_reverse_order() -> None:
    with pytest.raises(ValidationError, match="start"):
        DateRange(start=date(2026, 2, 1), end=date(2026, 1, 1))


def test_research_request_has_exact_catalog_fields() -> None:
    request = ResearchRequest(
        question="What changed?",
        output_requirements={"sections": ["summary"], "citations": True},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="demo",
        budget_preset="medium",
    )

    assert set(ResearchRequest.model_fields) == {
        "question",
        "output_requirements",
        "report_language",
        "source_languages",
        "freshness_requirement",
        "execution_mode",
        "access_profile",
        "provider_profile_id",
        "run_purpose",
        "budget_preset",
    }
    with pytest.raises(ValidationError):
        ResearchRequest.model_validate({**request.model_dump(), "provider": "duplicate"})


@pytest.mark.parametrize(
    "subquestions",
    [
        (subquestion("sq-1"), subquestion("sq-1", need_ids=("need-2",))),
        (subquestion("sq-1"), subquestion("sq-2", need_ids=("need-1",))),
        (subquestion("sq-1", dependencies=("missing",)),),
        (
            subquestion("sq-1", dependencies=("sq-2",)),
            subquestion("sq-2", dependencies=("sq-1",), need_ids=("need-2",)),
        ),
    ],
)
def test_plan_rejects_duplicate_ids_unknown_dependencies_and_cycles(
    subquestions: tuple[SubQuestion, ...],
) -> None:
    with pytest.raises(ValidationError, match="duplicate|unknown|cycle"):
        research_plan(*subquestions)


def test_plan_accepts_an_acyclic_dependency_graph() -> None:
    plan = research_plan(
        subquestion("sq-1"),
        subquestion("sq-2", dependencies=("sq-1",), need_ids=("need-2",)),
    )

    assert plan.subquestions[1].dependencies == ("sq-1",)


def test_research_models_are_frozen_and_serialize_deterministically() -> None:
    plan = research_plan(subquestion("sq-1"))

    assert plan.model_dump_json() == plan.model_dump_json()
    assert ResearchPlan.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValidationError):
        plan.plan_id = "changed"


def test_canonical_json_hash_is_independent_of_unordered_input() -> None:
    evidence_requirements = EvidenceRequirements(
        min_independent_sources=1,
        allowed_source_types=frozenset({"unknown", "paper", "news", "standard"}),
        must_include_primary=False,
    )
    common: dict[str, object] = {
        "question": "What changed?",
        "report_language": "en",
        "source_languages": ("en",),
        "freshness_requirement": FreshnessRequirement(kind="none"),
        "execution_mode": "replay",
        "access_profile": "showcase",
        "provider_profile_id": "offline",
        "run_purpose": "demo",
        "budget_preset": "medium",
    }
    first = ResearchRequest.model_validate(
        {**common, "output_requirements": {"z": {"b": 2, "a": 1}, "a": True}}
    )
    second = ResearchRequest.model_validate(
        {**common, "output_requirements": {"a": True, "z": {"a": 1, "b": 2}}}
    )

    assert '"allowed_source_types":["news","paper","standard","unknown"]' in (
        evidence_requirements.model_dump_json()
    )
    assert sha256(first.model_dump_json().encode()).digest() == sha256(
        second.model_dump_json().encode()
    ).digest()
