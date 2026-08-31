from decimal import Decimal
from hashlib import sha256

import pytest
from pydantic import ValidationError

from deepresearch.domain import (
    FreshnessRequirement,
    ResearchRequest,
    ResourceUsage,
    RunBudget,
    RunConfig,
)

NODES = {"Planner", "Ranker", "Writer", "Judge", "Tool"}


def request(
    *, access_profile: str = "local", run_purpose: str = "test"
) -> ResearchRequest:
    return ResearchRequest(
        question="Question?",
        output_requirements={},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile=access_profile,
        provider_profile_id="offline",
        run_purpose=run_purpose,
        budget_preset="medium",
    )


def run_config(
    *,
    access_profile: str = "local",
    run_purpose: str = "test",
    budget: RunBudget | None = None,
) -> RunConfig:
    return RunConfig(
        request=request(access_profile=access_profile, run_purpose=run_purpose),
        workflow_id="baseline-v1",
        planner_id="P0",
        ranker_id="R0",
        budget=budget or RunBudget.preset("medium"),
        prompt_versions={"planner": "v1"},
    )


def test_resource_usage_zero_has_explicit_cost_semantics() -> None:
    unknown_cost = ResourceUsage.zero()
    known_cost = ResourceUsage.zero(cost_known=True)

    assert unknown_cost.cost_usd is None
    assert known_cost.cost_usd == Decimal(0)
    assert unknown_cost.total_tokens == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"input_tokens": 2, "cached_tokens": 3},
        {"input_tokens": 2, "output_tokens": 3, "reasoning_tokens": 1, "total_tokens": 5},
    ],
)
def test_resource_usage_enforces_cached_and_total_token_invariants(
    updates: dict[str, int],
) -> None:
    payload: dict[str, object] = ResourceUsage.zero().model_dump()
    payload.update(updates)

    with pytest.raises(ValidationError, match="cached_tokens|total_tokens"):
        ResourceUsage.model_validate(payload)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("low", (4, 8, 20_000, 180, "0.25")),
        ("medium", (8, 12, 40_000, 300, "0.50")),
        ("high", (12, 20, 70_000, 480, "1.00")),
    ],
)
def test_budget_presets_match_approved_spec(
    name: str, expected: tuple[int, int, int, int, str]
) -> None:
    budget = RunBudget.preset(name)

    assert (
        budget.max_search_calls,
        budget.max_pages,
        budget.max_total_tokens,
        budget.max_wall_time_seconds,
        str(budget.max_cost_usd),
    ) == expected
    assert budget.max_retries == 2
    assert set(budget.used_by_node) == NODES
    assert all(value == ResourceUsage.zero() for value in budget.used_by_node.values())


def test_local_allows_token_only_budget_but_public_and_benchmark_require_cost() -> None:
    token_only = RunBudget.preset("medium").model_copy(update={"max_cost_usd": None})

    assert run_config(budget=token_only).budget.max_cost_usd is None
    with pytest.raises(ValidationError, match="max_cost_usd"):
        run_config(access_profile="public_live", budget=token_only)
    with pytest.raises(ValidationError, match="max_cost_usd"):
        run_config(run_purpose="benchmark", budget=token_only)


def test_run_config_does_not_duplicate_provider_profile_and_is_frozen() -> None:
    config = run_config()

    assert "provider_profile_id" not in RunConfig.model_fields
    assert config.request.provider_profile_id == "offline"
    with pytest.raises(ValidationError):
        config.seed = 7


def test_run_config_serialization_is_stable_and_round_trips() -> None:
    config = run_config()
    first = config.model_dump_json()

    assert config.model_dump_json() == first
    assert RunConfig.model_validate_json(first) == config


def test_budget_and_prompt_mapping_serialization_is_canonical() -> None:
    first_budget = RunBudget.preset("medium")
    reversed_usage = dict(reversed(tuple(first_budget.used_by_node.items())))
    second_budget = first_budget.model_copy(update={"used_by_node": reversed_usage})
    first = RunConfig(
        request=request(),
        workflow_id="baseline-v1",
        planner_id="P0",
        ranker_id="R0",
        budget=first_budget,
        prompt_versions={"writer": "v1", "planner": "v2"},
    )
    second = RunConfig(
        request=request(),
        workflow_id="baseline-v1",
        planner_id="P0",
        ranker_id="R0",
        budget=second_budget,
        prompt_versions={"planner": "v2", "writer": "v1"},
    )

    assert first.model_dump_json() == second.model_dump_json()


def test_used_by_node_is_detached_and_rejects_invalid_key_injection() -> None:
    preset = RunBudget.preset("low")
    caller_owned = dict(preset.used_by_node)
    budget = RunBudget.model_validate({**preset.model_dump(), "used_by_node": caller_owned})
    digest = sha256(budget.model_dump_json().encode()).digest()
    caller_owned["Bad"] = ResourceUsage.zero()

    assert "Bad" not in budget.used_by_node
    with pytest.raises(TypeError, match="immutable"):
        budget.used_by_node["Bad"] = ResourceUsage.zero()
    with pytest.raises(TypeError, match="immutable"):
        budget.used_by_node.update({"Bad": ResourceUsage.zero()})
    assert sha256(budget.model_dump_json().encode()).digest() == digest


def test_prompt_versions_are_detached_and_immutable() -> None:
    caller_owned = {"planner": "v1"}
    payload = run_config().model_dump()
    payload["prompt_versions"] = caller_owned
    config = RunConfig.model_validate(payload)
    digest = sha256(config.model_dump_json().encode()).digest()
    caller_owned["planner"] = "v2"

    assert config.prompt_versions == {"planner": "v1"}
    with pytest.raises(TypeError, match="immutable"):
        config.prompt_versions["planner"] = "v3"
    with pytest.raises(TypeError, match="immutable"):
        config.prompt_versions.clear()
    assert sha256(config.model_dump_json().encode()).digest() == digest


@pytest.mark.parametrize("deep", [False, True])
def test_budget_update_copy_rejects_invalid_nodes_and_freezes_valid_mapping(
    deep: bool,
) -> None:
    budget = RunBudget.preset("low")
    caller_owned = dict(budget.used_by_node)

    copied = budget.model_copy(update={"used_by_node": caller_owned}, deep=deep)
    caller_owned["Bad"] = ResourceUsage.zero()

    assert "Bad" not in copied.used_by_node
    with pytest.raises(TypeError, match="immutable"):
        copied.used_by_node["Bad"] = ResourceUsage.zero()
    with pytest.raises(ValidationError):
        budget.model_copy(
            update={"used_by_node": {"Bad": ResourceUsage.zero()}}, deep=deep
        )
    assert budget.model_copy(update={"max_cost_usd": None}, deep=deep).max_cost_usd is None


@pytest.mark.parametrize("deep", [False, True])
def test_prompt_update_copy_detaches_and_revalidates_run_config(deep: bool) -> None:
    config = run_config()
    caller_owned = {"planner": "v2"}

    copied = config.model_copy(update={"prompt_versions": caller_owned}, deep=deep)
    caller_owned["planner"] = "changed"

    assert copied.prompt_versions == {"planner": "v2"}
    with pytest.raises(TypeError, match="immutable"):
        copied.prompt_versions.clear()
    with pytest.raises(ValidationError):
        config.model_copy(
            update={"prompt_versions": {"planner": object()}}, deep=deep
        )

    token_only = RunBudget.preset("medium").model_copy(update={"max_cost_usd": None})
    with pytest.raises(ValidationError, match="max_cost_usd"):
        config.model_copy(
            update={
                "request": request(access_profile="public_live"),
                "budget": token_only,
            },
            deep=deep,
        )


@pytest.mark.parametrize("deep", [False, True])
def test_usage_update_copy_revalidates_token_totals(deep: bool) -> None:
    usage = ResourceUsage.zero()

    with pytest.raises(ValidationError, match="total_tokens"):
        usage.model_copy(update={"output_tokens": 1}, deep=deep)
