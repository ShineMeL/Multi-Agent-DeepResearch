from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from deepresearch.domain import (
    AccessProfile,
    ExecutionMode,
    ResearchRequest,
    RunBudget,
    RunConfig,
    RunPurpose,
)


class PolicyViolation(ValueError):
    """Raised when a request or config falls outside deployment policy."""


@dataclass(frozen=True)
class DeploymentPolicy:
    forced_access_profile: AccessProfile
    allowed_execution_modes: frozenset[ExecutionMode]
    allowed_provider_profile_ids: frozenset[str]
    allowed_run_purposes: frozenset[RunPurpose]
    allowed_budget_presets: frozenset[Literal["low", "medium", "high"]]
    budget_presets: Mapping[str, RunBudget]
    local_unpriced_live: bool = False

    def __post_init__(self) -> None:
        if self.local_unpriced_live and (
            self.forced_access_profile != "local" or "benchmark" in self.allowed_run_purposes
        ):
            raise ValueError("unpriced live runs are local demos only")
        if not self.allowed_execution_modes:
            raise ValueError("allowed_execution_modes must not be empty")
        if not self.allowed_provider_profile_ids:
            raise ValueError("allowed_provider_profile_ids must not be empty")
        if not self.allowed_run_purposes:
            raise ValueError("allowed_run_purposes must not be empty")
        if not self.allowed_budget_presets:
            raise ValueError("allowed_budget_presets must not be empty")
        if set(self.budget_presets) != set(self.allowed_budget_presets):
            raise ValueError("budget_presets must exactly cover allowed_budget_presets")

    def normalize_request(self, request: ResearchRequest) -> ResearchRequest:
        self._validate_request(request)
        return request.model_copy(update={"access_profile": self.forced_access_profile})

    def validate_config(self, config: RunConfig) -> None:
        self._validate_request(config.request)
        if config.request.access_profile != self.forced_access_profile:
            raise PolicyViolation("access profile has not been normalized")
        preset = self.budget_for_request(config.request)
        if config.budget.model_dump(mode="json") != preset.model_dump(mode="json"):
            raise PolicyViolation("budget does not match canonical preset")

    def budget_for_request(self, request: ResearchRequest) -> RunBudget:
        self._validate_request(request)
        preset = self.budget_presets[request.budget_preset].model_copy(deep=True)
        if self.local_unpriced_live and request.execution_mode == "live":
            # Missing prices are unknown, never free. Keep every non-monetary
            # limit; replay's frozen request/budget identity is unchanged.
            return preset.model_copy(update={"max_cost_usd": None})
        return preset

    def _validate_request(self, request: ResearchRequest) -> None:
        if request.execution_mode not in self.allowed_execution_modes:
            raise PolicyViolation("execution mode is not allowed")
        if request.provider_profile_id not in self.allowed_provider_profile_ids:
            raise PolicyViolation("provider profile is not allowed")
        if request.run_purpose not in self.allowed_run_purposes:
            raise PolicyViolation("run purpose is not allowed")
        if request.budget_preset not in self.allowed_budget_presets:
            raise PolicyViolation("budget preset is not allowed")


__all__ = ["DeploymentPolicy", "PolicyViolation"]
