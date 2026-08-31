from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from deepresearch.domain import ResearchPlan, ResearchRequest, RunBudget

type PlanValidationCode = Literal[
    "MALFORMED_JSON",
    "INVALID_SCHEMA",
    "DUPLICATE_SUBQUESTION_ID",
    "DUPLICATE_NEED_ID",
    "UNKNOWN_DEPENDENCY",
    "DEPENDENCY_CYCLE",
    "EMPTY_GOAL",
    "OUT_OF_SCOPE_GOAL",
    "UNEXECUTABLE_GOAL",
    "BUDGET_INFEASIBLE",
]

_CODE_ORDER: tuple[PlanValidationCode, ...] = (
    "MALFORMED_JSON",
    "INVALID_SCHEMA",
    "DUPLICATE_SUBQUESTION_ID",
    "DUPLICATE_NEED_ID",
    "UNKNOWN_DEPENDENCY",
    "DEPENDENCY_CYCLE",
    "EMPTY_GOAL",
    "OUT_OF_SCOPE_GOAL",
    "UNEXECUTABLE_GOAL",
    "BUDGET_INFEASIBLE",
)
_PRIMARY_SOURCE_TYPES = frozenset(
    {
        "paper",
        "official_documentation",
        "standard",
        "primary_data",
        "first_party_statement",
    }
)


class PlanValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    candidate: ResearchPlan | None = None
    error_codes: tuple[PlanValidationCode, ...] = ()
    candidate_artifact_id: str | None = None


def _ordered(codes: set[PlanValidationCode]) -> tuple[PlanValidationCode, ...]:
    return tuple(code for code in _CODE_ORDER if code in codes)


def _strict_json(value: str) -> object:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(value, parse_constant=reject_constant)


def _mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast("Mapping[str, object]", value)


def _sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    return cast("Sequence[object]", value)


def _duplicates(values: Sequence[str]) -> bool:
    return len(values) != len(set(values))


def _graph_codes(raw: Mapping[str, object]) -> set[PlanValidationCode]:
    codes: set[PlanValidationCode] = set()
    raw_subquestions = _sequence(raw.get("subquestions"))
    if raw_subquestions is None:
        return codes

    subquestions = [item for raw_item in raw_subquestions if (item := _mapping(raw_item))]
    subquestion_ids = [item_id for item in subquestions if isinstance((item_id := item.get("id")), str)]
    if _duplicates(subquestion_ids):
        codes.add("DUPLICATE_SUBQUESTION_ID")

    need_ids: list[str] = []
    empty_goal = False
    dependency_rows: list[tuple[str, tuple[str, ...]]] = []
    for subquestion in subquestions:
        subquestion_id = subquestion.get("id")
        question = subquestion.get("question")
        if isinstance(question, str) and not question.strip():
            empty_goal = True
        raw_dependencies = _sequence(subquestion.get("dependencies"))
        dependencies = (
            tuple(item for item in raw_dependencies if isinstance(item, str))
            if raw_dependencies is not None
            else ()
        )
        if isinstance(subquestion_id, str):
            dependency_rows.append((subquestion_id, dependencies))
        raw_needs = _sequence(subquestion.get("information_needs"))
        if raw_needs is None:
            continue
        for raw_need in raw_needs:
            need = _mapping(raw_need)
            if need is None:
                continue
            need_id = need.get("need_id")
            if isinstance(need_id, str):
                need_ids.append(need_id)
            text = need.get("text")
            if isinstance(text, str) and not text.strip():
                empty_goal = True

    raw_scope = _mapping(raw.get("scope"))
    if raw_scope is not None:
        topics = _sequence(raw_scope.get("included_topics"))
        if topics is not None and any(isinstance(item, str) and not item.strip() for item in topics):
            empty_goal = True
    if not subquestions:
        empty_goal = True
    if empty_goal:
        codes.add("EMPTY_GOAL")
    if _duplicates(need_ids):
        codes.add("DUPLICATE_NEED_ID")

    known_ids = set(subquestion_ids)
    if any(dependency not in known_ids for _, dependencies in dependency_rows for dependency in dependencies):
        codes.add("UNKNOWN_DEPENDENCY")

    graph: dict[str, tuple[str, ...]] = {}
    for subquestion_id, dependencies in dependency_rows:
        graph.setdefault(
            subquestion_id,
            tuple(dependency for dependency in dependencies if dependency in known_ids),
        )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(dependency) for dependency in graph.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    if any(visit(node) for node in graph):
        codes.add("DEPENDENCY_CYCLE")
    return codes


def _has_bool_for_numeric_field(raw: Mapping[str, object]) -> bool:
    raw_subquestions = _sequence(raw.get("subquestions")) or ()
    for raw_subquestion in raw_subquestions:
        subquestion = _mapping(raw_subquestion)
        if subquestion is None:
            continue
        if isinstance(subquestion.get("importance"), bool):
            return True
        raw_requirements = _mapping(subquestion.get("evidence_requirements"))
        if raw_requirements is not None and isinstance(
            raw_requirements.get("min_independent_sources"), bool
        ):
            return True
        raw_needs = _sequence(subquestion.get("information_needs")) or ()
        for raw_need in raw_needs:
            need = _mapping(raw_need)
            if need is not None and isinstance(need.get("importance"), bool):
                return True
    return False


def _without_graph_conflicts(
    raw: Mapping[str, object], codes: set[PlanValidationCode]
) -> dict[str, object]:
    candidate = deepcopy(dict(raw))
    raw_subquestions = candidate.get("subquestions")
    if not isinstance(raw_subquestions, list):
        return candidate
    subquestions = cast("list[object]", raw_subquestions)
    seen_subquestions: set[str] = set()
    seen_needs: set[str] = set()
    for subquestion_index, raw_subquestion in enumerate(subquestions):
        if not isinstance(raw_subquestion, dict):
            continue
        subquestion = cast("dict[str, object]", raw_subquestion)
        subquestion_id = subquestion.get("id")
        if isinstance(subquestion_id, str):
            if subquestion_id in seen_subquestions:
                subquestion["id"] = f"{subquestion_id}__duplicate_{subquestion_index}"
            seen_subquestions.add(cast("str", subquestion["id"]))
        if "UNKNOWN_DEPENDENCY" in codes or "DEPENDENCY_CYCLE" in codes:
            subquestion["dependencies"] = []
        raw_needs = subquestion.get("information_needs")
        if not isinstance(raw_needs, list):
            continue
        needs = cast("list[object]", raw_needs)
        for need_index, raw_need in enumerate(needs):
            if not isinstance(raw_need, dict):
                continue
            need = cast("dict[str, object]", raw_need)
            need_id = need.get("need_id")
            if isinstance(need_id, str):
                if need_id in seen_needs:
                    need["need_id"] = (
                        f"{need_id}__duplicate_{subquestion_index}_{need_index}"
                    )
                seen_needs.add(cast("str", need["need_id"]))
    return candidate


def _normalized_strings(value: object) -> tuple[str, ...] | None:
    values = _sequence(value)
    if values is None or any(not isinstance(item, str) for item in values):
        return None
    return tuple(cast("str", item).strip().casefold() for item in values)


def _requested_date(value: object, key: str) -> date | None:
    mapping = _mapping(value)
    if mapping is None:
        return None
    raw = mapping.get(key)
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _scope_is_outside_request(plan: ResearchPlan, request: ResearchRequest) -> bool:
    requirements = request.output_requirements
    allowed = _normalized_strings(requirements.get("included_topics"))
    excluded = _normalized_strings(requirements.get("excluded_topics")) or ()
    plan_included = tuple(topic.strip().casefold() for topic in plan.scope.included_topics)
    plan_excluded = {topic.strip().casefold() for topic in plan.scope.excluded_topics}
    if allowed is not None and any(topic not in set(allowed) for topic in plan_included):
        return True
    if any(topic in set(excluded) or topic in plan_excluded for topic in plan_included):
        return True
    if not set(excluded).issubset(plan_excluded):
        return True
    answer_shape = requirements.get("answer_shape")
    if isinstance(answer_shape, str) and plan.scope.answer_shape != answer_shape:
        return True
    requested_range = requirements.get("date_range")
    requested_start = _requested_date(requested_range, "start")
    requested_end = _requested_date(requested_range, "end")
    plan_range = plan.scope.date_range
    if plan_range is None and (requested_start is not None or requested_end is not None):
        return True
    if plan_range is not None:
        if requested_start is not None and (
            plan_range.start is None or plan_range.start < requested_start
        ):
            return True
        if requested_end is not None and (
            plan_range.end is None or plan_range.end > requested_end
        ):
            return True
        freshness = request.freshness_requirement
        if (
            freshness.kind == "published_after"
            and freshness.published_after is not None
            and (plan_range.start is None or plan_range.start < freshness.published_after)
        ):
            return True
    return not request.report_language.strip()


def _is_unexecutable(plan: ResearchPlan, request: ResearchRequest) -> bool:
    if not request.source_languages or any(not language.strip() for language in request.source_languages):
        return True
    for subquestion in plan.subquestions:
        allowed = {str(source_type) for source_type in subquestion.evidence_requirements.allowed_source_types}
        executable = allowed - {"unknown"}
        if not executable:
            return True
        if (
            subquestion.evidence_requirements.must_include_primary
            and executable.isdisjoint(_PRIMARY_SOURCE_TYPES)
        ):
            return True
    return False


def _is_over_budget(plan: ResearchPlan, budget: RunBudget) -> bool:
    used_searches = sum(usage.search_calls for usage in budget.used_by_node.values())
    used_pages = sum(usage.pages for usage in budget.used_by_node.values())
    used_tokens = sum(usage.total_tokens for usage in budget.used_by_node.values())
    required_searches = sum(max(1, len(item.information_needs)) for item in plan.subquestions)
    required_pages = sum(
        max(1, len(item.information_needs))
        * item.evidence_requirements.min_independent_sources
        for item in plan.subquestions
    )
    required_tokens = 1_000 + 1_000 * required_searches + 1_000 * required_pages
    return (
        required_searches > budget.max_search_calls - used_searches
        or required_pages > budget.max_pages - used_pages
        or required_tokens > budget.max_total_tokens - used_tokens
    )


class PlanValidator:
    def validate(self, plan: ResearchPlan) -> PlanValidationReport:
        return self.validate_candidate(plan, request=None, budget=None)

    def validate_candidate(
        self,
        candidate: str | bytes | Mapping[str, JsonValue] | ResearchPlan,
        *,
        request: ResearchRequest | None,
        budget: RunBudget | None,
        candidate_artifact_id: str | None = None,
    ) -> PlanValidationReport:
        codes: set[PlanValidationCode] = set()
        if isinstance(candidate, ResearchPlan):
            raw: object = candidate.model_dump(mode="json")
        elif isinstance(candidate, bytes):
            try:
                raw = _strict_json(candidate.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                codes.add("MALFORMED_JSON")
                return PlanValidationReport(
                    valid=False,
                    error_codes=_ordered(codes),
                    candidate_artifact_id=candidate_artifact_id,
                )
        elif isinstance(candidate, str):
            try:
                raw = _strict_json(candidate)
            except (ValueError, json.JSONDecodeError):
                codes.add("MALFORMED_JSON")
                return PlanValidationReport(
                    valid=False,
                    error_codes=_ordered(codes),
                    candidate_artifact_id=candidate_artifact_id,
                )
        else:
            raw = deepcopy(dict(candidate))

        raw_mapping = _mapping(raw)
        if raw_mapping is None:
            codes.add("INVALID_SCHEMA")
            return PlanValidationReport(
                valid=False,
                error_codes=_ordered(codes),
                candidate_artifact_id=candidate_artifact_id,
            )

        graph_codes = _graph_codes(raw_mapping)
        codes.update(graph_codes)
        strict_numeric_failure = _has_bool_for_numeric_field(raw_mapping)
        validated: ResearchPlan | None = None
        try:
            validated = ResearchPlan.model_validate(raw_mapping)
        except ValidationError as error:
            has_field_error = any(item.get("loc") for item in error.errors(include_input=False))
            if strict_numeric_failure or has_field_error or not graph_codes:
                codes.add("INVALID_SCHEMA")
        except (TypeError, ValueError):
            codes.add("INVALID_SCHEMA")
        if strict_numeric_failure:
            codes.add("INVALID_SCHEMA")

        semantic_plan = validated
        if semantic_plan is None and graph_codes:
            try:
                semantic_plan = ResearchPlan.model_validate(
                    _without_graph_conflicts(raw_mapping, graph_codes)
                )
            except (TypeError, ValueError, ValidationError):
                semantic_plan = None

        if semantic_plan is not None:
            if request is not None:
                if not request.question.strip():
                    codes.add("EMPTY_GOAL")
                if _scope_is_outside_request(semantic_plan, request):
                    codes.add("OUT_OF_SCOPE_GOAL")
                if _is_unexecutable(semantic_plan, request):
                    codes.add("UNEXECUTABLE_GOAL")
            if budget is not None and _is_over_budget(semantic_plan, budget):
                codes.add("BUDGET_INFEASIBLE")

        ordered = _ordered(codes)
        return PlanValidationReport(
            valid=not ordered,
            candidate=validated,
            error_codes=ordered,
            candidate_artifact_id=candidate_artifact_id,
        )


__all__ = ["PlanValidationCode", "PlanValidationReport", "PlanValidator"]
