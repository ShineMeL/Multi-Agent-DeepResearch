from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
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
_MAX_CANDIDATE_BYTES = 1_000_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 20_000
_MAX_JSON_STRING_CHARS = 250_000
_P1_PROMPT_FIXED_BYTES = 512
_P1_PLAN_OUTPUT_ALLOWANCE = 8_000
_P1_QUERY_OUTPUT_ALLOWANCE = 2_000


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

    def reject_duplicate_names(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object name")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_names,
    )


def _attach_snapshot(
    destination: object,
    destination_key: object | None,
    snapshot_value: JsonValue,
) -> None:
    if isinstance(destination, list):
        cast("list[JsonValue]", destination).append(snapshot_value)
    else:
        cast("dict[str, JsonValue]", destination)[cast("str", destination_key)] = (
            snapshot_value
        )


def _bounded_json_snapshot(value: object) -> tuple[bool, JsonValue | None]:
    root: list[JsonValue] = []
    stack: list[tuple[str, object, int, object, object | None]] = [
        ("value", value, 0, root, None)
    ]
    active_containers: set[int] = set()
    node_count = 0
    string_chars = 0
    while stack:
        kind, payload, depth, destination, destination_key = stack.pop()
        if kind in {"mapping", "sequence"}:
            iterator, source, snapshot, container_id = cast(
                "tuple[Iterator[object], object, JsonValue, int]", payload
            )
            try:
                next_item = next(iterator)
            except StopIteration:
                active_containers.remove(container_id)
                continue
            except (MemoryError, OSError, AssertionError):
                raise
            except Exception:  # noqa: BLE001 - ordinary candidate iterator fault
                return False, None
            if kind == "mapping":
                key = next_item
                if not isinstance(key, str):
                    return False, None
                plain_key = str.__str__(key)
                string_chars += len(plain_key)
                if string_chars > _MAX_JSON_STRING_CHARS:
                    return False, None
                try:
                    plain_key.encode("utf-8")
                except UnicodeEncodeError:
                    return False, None
                snapshot_mapping = cast("dict[str, JsonValue]", snapshot)
                if plain_key in snapshot_mapping:
                    return False, None
                try:
                    child = cast("Mapping[object, object]", source)[key]
                except (MemoryError, OSError, AssertionError):
                    raise
                except Exception:  # noqa: BLE001 - ordinary candidate lookup fault
                    return False, None
                child_destination: object = snapshot_mapping
                child_key: object | None = plain_key
            else:
                child = next_item
                child_destination = snapshot
                child_key = None
            stack.append((kind, payload, depth, destination, destination_key))
            stack.append(
                ("value", child, depth + 1, child_destination, child_key)
            )
            continue

        item = payload
        node_count += 1
        if node_count > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            return False, None

        if isinstance(item, str):
            plain_item = str.__str__(item)
            string_chars += len(plain_item)
            if string_chars > _MAX_JSON_STRING_CHARS:
                return False, None
            try:
                plain_item.encode("utf-8")
            except UnicodeEncodeError:
                return False, None
            _attach_snapshot(destination, destination_key, plain_item)
            continue
        if item is None or isinstance(item, bool):
            _attach_snapshot(destination, destination_key, item)
            continue
        if isinstance(item, int):
            _attach_snapshot(destination, destination_key, int.__int__(item))
            continue
        if isinstance(item, float):
            plain_float = float.__float__(item)
            if not math.isfinite(plain_float):
                return False, None
            _attach_snapshot(destination, destination_key, plain_float)
            continue
        if isinstance(item, Mapping):
            mapping = cast("Mapping[object, object]", item)
            identity = id(mapping)
            if identity in active_containers:
                return False, None
            active_containers.add(identity)
            snapshot_mapping: dict[str, JsonValue] = {}
            _attach_snapshot(destination, destination_key, snapshot_mapping)
            try:
                iterator = iter(mapping)
            except (MemoryError, OSError, AssertionError):
                raise
            except Exception:  # noqa: BLE001 - ordinary candidate iterator fault
                return False, None
            stack.append(
                (
                    "mapping",
                    (iterator, mapping, snapshot_mapping, identity),
                    depth,
                    destination,
                    destination_key,
                )
            )
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            sequence = cast("Sequence[object]", item)
            identity = id(sequence)
            if identity in active_containers:
                return False, None
            active_containers.add(identity)
            snapshot_sequence: list[JsonValue] = []
            _attach_snapshot(destination, destination_key, snapshot_sequence)
            try:
                iterator = iter(sequence)
            except (MemoryError, OSError, AssertionError):
                raise
            except Exception:  # noqa: BLE001 - ordinary candidate iterator fault
                return False, None
            stack.append(
                (
                    "sequence",
                    (iterator, sequence, snapshot_sequence, identity),
                    depth,
                    destination,
                    destination_key,
                )
            )
            continue
        return False, None
    return (len(root) == 1), (root[0] if len(root) == 1 else None)


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
        if topics is not None and (
            not topics
            or any(isinstance(item, str) and not item.strip() for item in topics)
        ):
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

    graph: dict[str, set[str]] = {subquestion_id: set() for subquestion_id in known_ids}
    for subquestion_id, dependencies in dependency_rows:
        graph[subquestion_id].update(
            dependency for dependency in dependencies if dependency in known_ids
        )
    remaining = set(graph)
    while remaining:
        ready = {
            subquestion_id
            for subquestion_id in remaining
            if graph[subquestion_id].isdisjoint(remaining)
        }
        if not ready:
            codes.add("DEPENDENCY_CYCLE")
            break
        remaining.difference_update(ready)
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
        if raw_requirements is not None:
            if isinstance(raw_requirements.get("min_independent_sources"), bool):
                return True
            freshness = _mapping(raw_requirements.get("freshness"))
            if freshness is not None and isinstance(
                freshness.get("retrieved_within_days"), bool
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
    if not plan_included:
        return True
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
        and (
            plan_range is None
            or plan_range.start is None
            or plan_range.start < freshness.published_after
        )
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


def _json_utf8_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _p1_prompt_token_upper_bound(
    plan: ResearchPlan,
    request: ResearchRequest | None,
    *,
    search_depth: int,
) -> int:
    total = 0
    if request is not None:
        total += (
            _P1_PROMPT_FIXED_BYTES
            + _json_utf8_size(request.model_dump(mode="json"))
            + _P1_PLAN_OUTPUT_ALLOWANCE
        )
    for subquestion in plan.subquestions:
        query_payload = {
            "plan_id": plan.plan_id,
            "search_depth": search_depth,
            "subquestion": subquestion.model_dump(mode="json"),
        }
        total += (
            _P1_PROMPT_FIXED_BYTES
            + _json_utf8_size(query_payload)
            + _P1_QUERY_OUTPUT_ALLOWANCE
        )
    return total


def _is_over_budget(
    plan: ResearchPlan,
    budget: RunBudget,
    *,
    request: ResearchRequest | None,
    search_depth: int,
) -> bool:
    used_searches = sum(usage.search_calls for usage in budget.used_by_node.values())
    used_pages = sum(usage.pages for usage in budget.used_by_node.values())
    used_tokens = sum(usage.total_tokens for usage in budget.used_by_node.values())
    required_searches = search_depth * len(plan.subquestions)
    required_pages = sum(
        max(
            search_depth,
            max(1, len(item.information_needs))
            * item.evidence_requirements.min_independent_sources,
        )
        for item in plan.subquestions
    )
    required_tokens = (
        1_000
        + 1_000 * required_searches
        + 1_000 * required_pages
        + _p1_prompt_token_upper_bound(
            plan,
            request,
            search_depth=search_depth,
        )
    )
    return (
        required_searches > budget.max_search_calls - used_searches
        or required_pages > budget.max_pages - used_pages
        or required_tokens > budget.max_total_tokens - used_tokens
    )


class PlanValidator:
    def __init__(self, *, search_depth: int = 2) -> None:
        if (
            isinstance(search_depth, bool)
            or not isinstance(search_depth, int)  # pyright: ignore[reportUnnecessaryIsInstance]
            or search_depth <= 0
        ):
            raise ValueError("search_depth must be a positive integer")
        self.search_depth = search_depth

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
            if len(candidate) > _MAX_CANDIDATE_BYTES:
                codes.add("MALFORMED_JSON")
                return PlanValidationReport(
                    valid=False,
                    error_codes=_ordered(codes),
                    candidate_artifact_id=candidate_artifact_id,
                )
            try:
                raw = _strict_json(candidate.decode("utf-8"))
            except (RecursionError, UnicodeDecodeError, ValueError):
                codes.add("MALFORMED_JSON")
                return PlanValidationReport(
                    valid=False,
                    error_codes=_ordered(codes),
                    candidate_artifact_id=candidate_artifact_id,
                )
        elif isinstance(candidate, str):
            try:
                encoded_candidate = candidate.encode("utf-8")
            except UnicodeEncodeError:
                codes.add("MALFORMED_JSON")
                return PlanValidationReport(
                    valid=False,
                    error_codes=_ordered(codes),
                    candidate_artifact_id=candidate_artifact_id,
                )
            if len(encoded_candidate) > _MAX_CANDIDATE_BYTES:
                codes.add("MALFORMED_JSON")
                return PlanValidationReport(
                    valid=False,
                    error_codes=_ordered(codes),
                    candidate_artifact_id=candidate_artifact_id,
                )
            try:
                raw = _strict_json(candidate)
            except (RecursionError, ValueError):
                codes.add("MALFORMED_JSON")
                return PlanValidationReport(
                    valid=False,
                    error_codes=_ordered(codes),
                    candidate_artifact_id=candidate_artifact_id,
                )
        else:
            raw = candidate

        snapshot: JsonValue | None = None
        try:
            structure_is_valid, snapshot = _bounded_json_snapshot(raw)
        except (LookupError, OverflowError, RecursionError, TypeError, ValueError):
            structure_is_valid = False
        if not structure_is_valid:
            codes.add("INVALID_SCHEMA")
            return PlanValidationReport(
                valid=False,
                error_codes=_ordered(codes),
                candidate_artifact_id=candidate_artifact_id,
            )

        raw_mapping = _mapping(snapshot)
        if raw_mapping is None:
            codes.add("INVALID_SCHEMA")
            return PlanValidationReport(
                valid=False,
                error_codes=_ordered(codes),
                candidate_artifact_id=candidate_artifact_id,
            )

        try:
            graph_codes = _graph_codes(raw_mapping)
        except (OverflowError, RecursionError, TypeError, ValueError):
            codes.add("INVALID_SCHEMA")
            return PlanValidationReport(
                valid=False,
                error_codes=_ordered(codes),
                candidate_artifact_id=candidate_artifact_id,
            )
        codes.update(graph_codes)
        strict_numeric_failure = _has_bool_for_numeric_field(raw_mapping)
        validated: ResearchPlan | None = None
        try:
            validated = ResearchPlan.model_validate(raw_mapping)
        except ValidationError as error:
            has_field_error = any(item.get("loc") for item in error.errors(include_input=False))
            if strict_numeric_failure or has_field_error or not graph_codes:
                codes.add("INVALID_SCHEMA")
        except (OverflowError, RecursionError, TypeError, ValueError):
            codes.add("INVALID_SCHEMA")
        if strict_numeric_failure:
            codes.add("INVALID_SCHEMA")

        semantic_plan = validated
        if semantic_plan is None and graph_codes:
            try:
                semantic_plan = ResearchPlan.model_validate(
                    _without_graph_conflicts(raw_mapping, graph_codes)
                )
            except (OverflowError, RecursionError, TypeError, ValueError, ValidationError):
                semantic_plan = None

        if semantic_plan is not None:
            if request is not None:
                if not request.question.strip():
                    codes.add("EMPTY_GOAL")
                if _scope_is_outside_request(semantic_plan, request):
                    codes.add("OUT_OF_SCOPE_GOAL")
                if _is_unexecutable(semantic_plan, request):
                    codes.add("UNEXECUTABLE_GOAL")
            if budget is not None and _is_over_budget(
                semantic_plan,
                budget,
                request=request,
                search_depth=self.search_depth,
            ):
                codes.add("BUDGET_INFEASIBLE")

        ordered = _ordered(codes)
        return PlanValidationReport(
            valid=not ordered,
            candidate=validated,
            error_codes=ordered,
            candidate_artifact_id=candidate_artifact_id,
        )


__all__ = ["PlanValidationCode", "PlanValidationReport", "PlanValidator"]
