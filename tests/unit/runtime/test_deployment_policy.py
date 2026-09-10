import pytest

from deepresearch.domain import FreshnessRequirement, ResearchRequest, RunBudget, RunConfig
from deepresearch.runtime.deployment_policy import DeploymentPolicy, PolicyViolation


@pytest.fixture
def replay_request() -> ResearchRequest:
    return ResearchRequest(
        question="What changed?",
        output_requirements={},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="demo",
        budget_preset="medium",
    )


@pytest.fixture
def public_policy() -> DeploymentPolicy:
    return DeploymentPolicy(
        forced_access_profile="public_live",
        allowed_execution_modes=frozenset({"replay"}),
        allowed_provider_profile_ids=frozenset({"offline"}),
        allowed_run_purposes=frozenset({"demo"}),
        allowed_budget_presets=frozenset({"medium"}),
        budget_presets={"medium": RunBudget.preset("medium")},
    )


def test_public_policy_overrides_client_profile_and_rejects_unapproved_route_or_budget(
    public_policy: DeploymentPolicy, replay_request: ResearchRequest
) -> None:
    normalized = public_policy.normalize_request(
        replay_request.model_copy(update={"access_profile": "local"}),
    )
    assert normalized.access_profile == "public_live"
    with pytest.raises(PolicyViolation):
        public_policy.normalize_request(
            replay_request.model_copy(update={"provider_profile_id": "other"}),
        )
    with pytest.raises(PolicyViolation):
        public_policy.normalize_request(
            replay_request.model_copy(update={"budget_preset": "high"}),
        )


def test_policy_rejects_client_budget_that_is_not_canonical(
    public_policy: DeploymentPolicy, replay_request: ResearchRequest
) -> None:
    config = RunConfig(
        request=public_policy.normalize_request(replay_request),
        workflow_id="research-v1",
        planner_id="P0",
        ranker_id="R0",
        budget=RunBudget.preset("low"),
        prompt_versions={},
    )

    with pytest.raises(PolicyViolation, match="budget"):
        public_policy.validate_config(config)
