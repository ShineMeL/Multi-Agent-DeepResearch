from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest

from deepresearch.domain import (
    DateRange,
    EvidenceRequirements,
    FreshnessRequirement,
    InformationNeed,
    ResearchPlan,
    ResearchRequest,
    ResearchScope,
    RunBudget,
    SubQuestion,
)
from deepresearch.planning import PlanValidator


def research_request() -> ResearchRequest:
    return ResearchRequest(
        question="Compare planner optimization methods.",
        output_requirements={
            "answer_shape": "brief",
            "date_range": {"end": "2026-12-31", "start": "2024-01-01"},
            "excluded_topics": ["credential harvesting"],
            "included_topics": ["planner optimization"],
        },
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="test",
        budget_preset="medium",
    )


def valid_candidate() -> dict[str, object]:
    return {
        "plan_id": "plan-1",
        "scope": {
            "included_topics": ["planner optimization"],
            "excluded_topics": ["credential harvesting"],
            "date_range": {"start": "2024-01-01", "end": "2026-12-31"},
            "answer_shape": "brief",
        },
        "subquestions": [
            {
                "id": "sq-1",
                "question": "Which planner optimization methods are documented?",
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
    }


def test_plan_validator_accepts_valid_candidate_without_mutating_input() -> None:
    candidate = valid_candidate()
    original = deepcopy(candidate)

    result = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
        candidate_artifact_id="sha256:" + "a" * 64,
    )

    assert result.valid is True
    assert isinstance(result.candidate, ResearchPlan)
    assert result.error_codes == ()
    assert result.candidate_artifact_id == "sha256:" + "a" * 64
    assert candidate == original


def test_plan_validator_rejects_duplicate_need_and_cycle() -> None:
    requirements = EvidenceRequirements(
        min_independent_sources=1,
        allowed_source_types=frozenset({"paper"}),
        must_include_primary=False,
    )
    first = SubQuestion(
        id="sq-1",
        question="First?",
        rationale_code="coverage",
        importance=0.8,
        dependencies=("sq-2",),
        information_needs=(InformationNeed(need_id="need", text="One", importance=0.5),),
        evidence_requirements=requirements,
        status="pending",
    )
    second = first.model_copy(
        update={"id": "sq-2", "question": "Second?", "dependencies": ("sq-1",)}
    )
    plan = ResearchPlan.model_construct(
        plan_id="plan-invalid",
        scope=ResearchScope(
            included_topics=("planner optimization",),
            excluded_topics=(),
            date_range=None,
            answer_shape="brief",
        ),
        subquestions=(first, second),
        created_by_model="fake-model",
        prompt_version="fixed-planner-v1",
    )

    result = PlanValidator().validate(plan)

    assert result.valid is False
    assert result.error_codes == ("DUPLICATE_NEED_ID", "DEPENDENCY_CYCLE")


def test_plan_validator_reports_all_graph_codes_in_stable_order() -> None:
    candidate = valid_candidate()
    subquestions = candidate["subquestions"]
    assert isinstance(subquestions, list)
    first = deepcopy(subquestions[0])
    second = deepcopy(subquestions[0])
    third = deepcopy(subquestions[0])
    first.update({"id": "sq-1", "dependencies": ["sq-2", "missing"]})
    second.update({"id": "sq-2", "dependencies": ["sq-1"]})
    third.update({"id": "sq-2", "dependencies": [], "question": "   "})
    candidate["subquestions"] = [first, second, third]

    result = PlanValidator().validate_candidate(candidate, request=None, budget=None)

    assert result.error_codes == (
        "DUPLICATE_SUBQUESTION_ID",
        "DUPLICATE_NEED_ID",
        "UNKNOWN_DEPENDENCY",
        "DEPENDENCY_CYCLE",
        "EMPTY_GOAL",
    )


def test_plan_validator_reports_budget_error_alongside_graph_errors() -> None:
    candidate = valid_candidate()
    template = candidate["subquestions"][0]  # type: ignore[index]
    subquestions: list[object] = []
    for index in range(9):
        item = deepcopy(template)
        item["id"] = f"sq-{index}"
        item["information_needs"][0]["need_id"] = f"need-{index}"
        subquestions.append(item)
    subquestions[0]["dependencies"] = ["sq-1"]  # type: ignore[index]
    subquestions[1]["dependencies"] = ["sq-0"]  # type: ignore[index]
    subquestions[8]["id"] = "sq-7"  # type: ignore[index]
    candidate["subquestions"] = subquestions

    result = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert result.error_codes == (
        "DUPLICATE_SUBQUESTION_ID",
        "DEPENDENCY_CYCLE",
        "BUDGET_INFEASIBLE",
    )


def test_plan_validator_emits_stable_public_codes_for_candidate_failures() -> None:
    validator = PlanValidator()
    request = research_request()
    budget = RunBudget.preset("medium")

    malformed = validator.validate_candidate("{not json", request=request, budget=budget)
    schema_candidate = valid_candidate()
    del schema_candidate["plan_id"]
    schema = validator.validate_candidate(schema_candidate, request=request, budget=budget)
    bool_candidate = valid_candidate()
    bool_candidate["subquestions"][0]["evidence_requirements"][  # type: ignore[index]
        "min_independent_sources"
    ] = True
    bool_as_int = validator.validate_candidate(bool_candidate, request=request, budget=budget)
    out_of_scope_candidate = valid_candidate()
    out_of_scope_candidate["scope"]["included_topics"] = [  # type: ignore[index]
        "credential harvesting"
    ]
    out_of_scope = validator.validate_candidate(
        out_of_scope_candidate, request=request, budget=budget
    )
    unexecutable_candidate = valid_candidate()
    unexecutable_candidate["subquestions"][0]["evidence_requirements"][  # type: ignore[index]
        "allowed_source_types"
    ] = ["unknown"]
    unexecutable = validator.validate_candidate(
        unexecutable_candidate, request=request, budget=budget
    )
    over_budget_candidate = valid_candidate()
    template = over_budget_candidate["subquestions"][0]  # type: ignore[index]
    over_budget_candidate["subquestions"] = []
    for index in range(9):
        item = deepcopy(template)
        item["id"] = f"sq-{index}"
        item["information_needs"][0]["need_id"] = f"need-{index}"
        over_budget_candidate["subquestions"].append(item)  # type: ignore[union-attr]
    over_budget = validator.validate_candidate(
        over_budget_candidate, request=request, budget=budget
    )

    assert malformed.error_codes == ("MALFORMED_JSON",)
    assert schema.error_codes == ("INVALID_SCHEMA",)
    assert bool_as_int.error_codes == ("INVALID_SCHEMA",)
    assert "OUT_OF_SCOPE_GOAL" in out_of_scope.error_codes
    assert "UNEXECUTABLE_GOAL" in unexecutable.error_codes
    assert "BUDGET_INFEASIBLE" in over_budget.error_codes
    assert all("validation" not in str(result.error_codes).lower() for result in (schema, bool_as_int))


def test_plan_validator_enforces_scope_shape_and_date_bounds() -> None:
    candidate = valid_candidate()
    candidate["scope"]["answer_shape"] = "book"  # type: ignore[index]
    candidate["scope"]["date_range"] = {  # type: ignore[index]
        "start": "2023-12-31",
        "end": "2027-01-01",
    }

    result = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert result.error_codes == ("OUT_OF_SCOPE_GOAL",)


def test_plan_validator_requires_requested_date_and_exclusion_bounds() -> None:
    candidate = valid_candidate()
    candidate["scope"]["date_range"] = None  # type: ignore[index]
    candidate["scope"]["excluded_topics"] = []  # type: ignore[index]

    result = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert result.error_codes == ("OUT_OF_SCOPE_GOAL",)


def test_plan_validator_rejects_whitespace_request_goal() -> None:
    request = research_request().model_copy(update={"question": "   "})

    result = PlanValidator().validate_candidate(
        valid_candidate(), request=request, budget=RunBudget.preset("medium")
    )

    assert result.error_codes == ("EMPTY_GOAL",)


def test_plan_validator_rejects_self_dependency_as_cycle() -> None:
    candidate = valid_candidate()
    candidate["subquestions"][0]["dependencies"] = ["sq-1"]  # type: ignore[index]

    result = PlanValidator().validate_candidate(candidate, request=None, budget=None)

    assert result.error_codes == ("DEPENDENCY_CYCLE",)


def test_validated_plan_keeps_date_values_and_frozen_models() -> None:
    result = PlanValidator().validate_candidate(
        valid_candidate(),
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert result.candidate is not None
    assert result.candidate.scope.date_range == DateRange(
        start=date(2024, 1, 1), end=date(2026, 12, 31)
    )


@pytest.mark.parametrize(
    "candidate",
    [
        "[" * 2_000 + "]" * 2_000,
        ("[" * 2_000 + "]" * 2_000).encode(),
        '"' + "x" * 1_100_000 + '"',
    ],
    ids=("deep-text", "deep-bytes", "oversize-text"),
)
def test_plan_validator_bounds_untrusted_json_without_leaking_runtime_errors(
    candidate: str | bytes,
) -> None:
    report = PlanValidator().validate_candidate(candidate, request=None, budget=None)

    assert report.valid is False
    assert report.error_codes == ("MALFORMED_JSON",)


def test_plan_validator_bounds_preparsed_mapping_depth() -> None:
    nested: object = "leaf"
    for _ in range(200):
        nested = {"nested": nested}
    candidate = valid_candidate()
    candidate["unexpected"] = nested

    report = PlanValidator().validate_candidate(candidate, request=None, budget=None)

    assert report.valid is False
    assert report.error_codes == ("INVALID_SCHEMA",)


@pytest.mark.parametrize(
    "path",
    [
        ("subquestions", 0, "importance"),
        ("subquestions", 0, "information_needs", 0, "importance"),
        ("subquestions", 0, "evidence_requirements", "min_independent_sources"),
        (
            "subquestions",
            0,
            "evidence_requirements",
            "freshness",
            "retrieved_within_days",
        ),
    ],
)
def test_plan_validator_rejects_bool_in_every_numeric_plan_field(
    path: tuple[str | int, ...],
) -> None:
    candidate = valid_candidate()
    candidate["subquestions"][0]["evidence_requirements"]["freshness"] = {  # type: ignore[index]
        "kind": "retrieved_within_days",
        "retrieved_within_days": 7,
    }
    target: object = candidate
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = True  # type: ignore[index]

    report = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert "INVALID_SCHEMA" in report.error_codes


def test_plan_validator_retains_real_boolean_fields() -> None:
    candidate = valid_candidate()
    requirements = candidate["subquestions"][0]["evidence_requirements"]  # type: ignore[index]
    requirements["must_include_primary"] = True

    report = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert report.valid is True
    assert report.candidate is not None
    assert report.candidate.subquestions[0].evidence_requirements.must_include_primary is True


def test_plan_validator_rejects_empty_included_topic_scope() -> None:
    candidate = valid_candidate()
    candidate["scope"]["included_topics"] = []  # type: ignore[index]

    report = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert report.error_codes == ("EMPTY_GOAL", "OUT_OF_SCOPE_GOAL")


@pytest.mark.parametrize(
    "date_range",
    [None, {"start": "2025-12-31", "end": "2026-12-31"}],
)
def test_plan_validator_rejects_omitted_or_insufficient_freshness_range(
    date_range: object,
) -> None:
    request = research_request().model_copy(
        update={
            "output_requirements": {
                "answer_shape": "brief",
                "excluded_topics": ["credential harvesting"],
                "included_topics": ["planner optimization"],
            },
            "freshness_requirement": FreshnessRequirement(
                kind="published_after", published_after=date(2026, 1, 1)
            )
        }
    )
    candidate = valid_candidate()
    candidate["scope"]["date_range"] = date_range  # type: ignore[index]

    report = PlanValidator().validate_candidate(
        candidate, request=request, budget=RunBudget.preset("medium")
    )

    assert "OUT_OF_SCOPE_GOAL" in report.error_codes


@pytest.mark.parametrize("reverse_duplicate_rows", [False, True])
def test_plan_validator_detects_cycles_across_every_duplicate_row_edge(
    reverse_duplicate_rows: bool,
) -> None:
    candidate = valid_candidate()
    template = candidate["subquestions"][0]  # type: ignore[index]
    first_a = deepcopy(template)
    first_a.update({"id": "A", "dependencies": []})
    first_a["information_needs"][0]["need_id"] = "need-a1"
    second_a = deepcopy(template)
    second_a.update({"id": "A", "dependencies": ["B"]})
    second_a["information_needs"][0]["need_id"] = "need-a2"
    b_row = deepcopy(template)
    b_row.update({"id": "B", "dependencies": ["A"]})
    b_row["information_needs"][0]["need_id"] = "need-b"
    duplicate_rows = [second_a, first_a] if reverse_duplicate_rows else [first_a, second_a]
    candidate["subquestions"] = [*duplicate_rows, b_row]

    report = PlanValidator().validate_candidate(candidate, request=None, budget=None)

    assert report.error_codes == (
        "DUPLICATE_SUBQUESTION_ID",
        "DEPENDENCY_CYCLE",
    )


def test_plan_validator_default_search_depth_accounts_for_p1_query_fanout() -> None:
    candidate = valid_candidate()
    template = candidate["subquestions"][0]  # type: ignore[index]
    candidate["subquestions"] = []
    for index in range(5):
        item = deepcopy(template)
        item["id"] = f"sq-{index}"
        item["information_needs"][0]["need_id"] = f"need-{index}"
        candidate["subquestions"].append(item)  # type: ignore[union-attr]

    default_report = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )
    shallow_report = PlanValidator(search_depth=1).validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert "BUDGET_INFEASIBLE" in default_report.error_codes
    assert shallow_report.valid is True


def test_plan_validator_accepts_acyclic_aliased_json_containers() -> None:
    shared: list[object] = []
    aliased = valid_candidate()
    aliased["scope"]["excluded_topics"] = shared  # type: ignore[index]
    aliased["subquestions"][0]["dependencies"] = shared  # type: ignore[index]
    separate = deepcopy(aliased)
    separate["subquestions"][0]["dependencies"] = []  # type: ignore[index]

    aliased_report = PlanValidator().validate_candidate(
        aliased, request=None, budget=None
    )
    separate_report = PlanValidator().validate_candidate(
        separate, request=None, budget=None
    )

    assert aliased_report.valid is True
    assert separate_report.valid is True
    assert aliased_report.candidate == separate_report.candidate


def test_plan_validator_rejects_real_container_cycles() -> None:
    candidate = valid_candidate()
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    candidate["unexpected"] = cycle

    report = PlanValidator().validate_candidate(candidate, request=None, budget=None)

    assert report.valid is False
    assert report.error_codes == ("INVALID_SCHEMA",)


def test_plan_validator_stops_wide_sequence_iteration_at_the_node_bound() -> None:
    class CountingList(list[object]):
        iterations = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            for item in super().__iter__():
                self.iterations += 1
                yield item

    wide = CountingList(["x"] * 100_000)
    candidate = valid_candidate()
    candidate["scope"]["included_topics"] = wide  # type: ignore[index]

    report = PlanValidator().validate_candidate(candidate, request=None, budget=None)

    assert report.valid is False
    assert report.error_codes == ("INVALID_SCHEMA",)
    assert wide.iterations < 25_000


def test_plan_validator_does_not_translate_memory_exhaustion_to_candidate_error() -> None:
    class MemoryFaultMapping(dict[str, object]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise MemoryError("synthetic allocation failure")

    with pytest.raises(MemoryError, match="synthetic allocation failure"):
        PlanValidator().validate_candidate(
            MemoryFaultMapping(valid_candidate()), request=None, budget=None
        )


def test_plan_validator_translates_invalid_mapping_iteration_to_stable_code() -> None:
    class MissingKeyMapping(dict[str, object]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            yield "missing-key"

    report = PlanValidator().validate_candidate(
        MissingKeyMapping(valid_candidate()), request=None, budget=None
    )

    assert report.valid is False
    assert report.error_codes == ("INVALID_SCHEMA",)


@pytest.mark.parametrize(
    "oversized_question",
    ["x " * 100_000, "界" * 20_000],
    ids=("ascii", "multibyte-utf8"),
)
def test_plan_validator_accounts_for_every_prompt_string_in_utf8_bytes(
    oversized_question: str,
) -> None:
    candidate = valid_candidate()
    candidate["subquestions"][0]["question"] = oversized_question  # type: ignore[index]

    report = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert report.valid is False
    assert report.error_codes == ("BUDGET_INFEASIBLE",)


def test_plan_validator_rejects_non_utf8_scalar_without_leaking_encoder_error() -> None:
    candidate = valid_candidate()
    candidate["subquestions"][0]["question"] = "invalid \ud800 scalar"  # type: ignore[index]

    report = PlanValidator().validate_candidate(
        candidate,
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )

    assert report.valid is False
    assert report.error_codes == ("INVALID_SCHEMA",)
