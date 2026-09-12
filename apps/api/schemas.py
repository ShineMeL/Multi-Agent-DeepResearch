"""Public payloads and the server-owned request-to-config boundary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from deepresearch.domain import ResearchRequest, ResourceUsage, RunConfig, RunStatus, StopReason
from deepresearch.runtime.deployment_policy import DeploymentPolicy

from .error_handlers import APIError

ArtifactKind = Literal["report", "evidence", "manifest"]


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    request: ResearchRequest
    workflow_id: Literal["baseline-v1", "research-v1"] = "research-v1"
    planner_id: Literal["P0", "P1", "P2"] = "P2"
    ranker_id: Literal["R0", "R1", "R2"] = "R2"
    seed: int | None = None

    def to_run_config(self, policy: DeploymentPolicy) -> RunConfig:
        request = policy.normalize_request(self.request)
        if self.workflow_id == "baseline-v1" and (self.planner_id, self.ranker_id) != ("P1", "R1"):
            raise APIError("INVALID_REQUEST")
        planner_version = {
            "P0": "react-planner-v1",
            "P1": "fixed-planner-v1",
            "P2": "adaptive-planner-v1",
        }[self.planner_id]
        versions = {"planner": planner_version, "writer": "baseline-writer-v1"}
        if self.planner_id == "P1":
            versions["planner_queries"] = "fixed-planner-v1-queries"
        if self.ranker_id == "R2":
            versions["ranker"] = "r2-utility-v1"
        return RunConfig(
            request=request,
            workflow_id=self.workflow_id,
            planner_id=self.planner_id,
            ranker_id=self.ranker_id,
            budget=policy.budget_for_request(request),
            prompt_versions=versions,
            ranker_weights_version="r2-v1" if self.ranker_id == "R2" else None,
            seed=self.seed,
        )


class RunAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    thread_id: str
    status: RunStatus
    events_url: str


class RunViewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    run_id: str
    thread_id: str
    status: RunStatus
    stop_reason: StopReason | None
    is_partial: bool
    report_artifact_id: str | None
    evidence_graph_artifact_id: str | None
    manifest_artifact_id: str | None
    final_usage: ResourceUsage | None
    error_code: str | None
