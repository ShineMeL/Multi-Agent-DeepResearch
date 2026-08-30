from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .research import ResearchRequest, _freeze_mapping  # pyright: ignore[reportPrivateUsage]

_NodeName = Literal["Planner", "Ranker", "Writer", "Judge", "Tool"]


class ResourceUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)]
    cached_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    search_calls: Annotated[int, Field(ge=0)]
    pages: Annotated[int, Field(ge=0)]
    retries: Annotated[int, Field(ge=0)]
    wall_seconds: Annotated[float, Field(ge=0.0)]
    cost_usd: Decimal | None = None

    @classmethod
    def zero(cls, *, cost_known: bool = False) -> Self:
        return cls(
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cached_tokens=0,
            total_tokens=0,
            search_calls=0,
            pages=0,
            retries=0,
            wall_seconds=0.0,
            cost_usd=Decimal(0) if cost_known else None,
        )

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached_tokens must not exceed input_tokens")
        expected_total = self.input_tokens + self.output_tokens + self.reasoning_tokens
        if self.total_tokens != expected_total:
            raise ValueError("total_tokens invariant violated")
        return self


class RunBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_search_calls: Annotated[int, Field(ge=0)]
    max_pages: Annotated[int, Field(ge=0)]
    max_total_tokens: Annotated[int, Field(ge=0)]
    max_wall_time_seconds: Annotated[int, Field(gt=0)]
    max_cost_usd: Annotated[Decimal | None, Field(ge=0)] = None
    max_retries: Annotated[int, Field(ge=0)]
    used_by_node: dict[_NodeName, ResourceUsage]

    @field_validator("used_by_node")
    @classmethod
    def freeze_used_by_node(
        cls, value: dict[_NodeName, ResourceUsage]
    ) -> dict[_NodeName, ResourceUsage]:
        return _freeze_mapping(value)

    @field_serializer("used_by_node", when_used="json")
    def serialize_used_by_node(
        self, value: dict[_NodeName, ResourceUsage]
    ) -> dict[_NodeName, ResourceUsage]:
        return {key: value[key] for key in sorted(value)}

    @classmethod
    def preset(cls, name: Literal["low", "medium", "high"]) -> Self:
        limits: dict[str, tuple[int, int, int, int, Decimal]] = {
            "low": (4, 8, 20_000, 180, Decimal("0.25")),
            "medium": (8, 12, 40_000, 300, Decimal("0.50")),
            "high": (12, 20, 70_000, 480, Decimal("1.00")),
        }
        searches, pages, tokens, seconds, cost = limits[name]
        return cls(
            max_search_calls=searches,
            max_pages=pages,
            max_total_tokens=tokens,
            max_wall_time_seconds=seconds,
            max_cost_usd=cost,
            max_retries=2,
            used_by_node={node: ResourceUsage.zero() for node in _node_names()},
        )


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: ResearchRequest
    workflow_id: Literal["baseline-v1", "research-v1"]
    planner_id: Literal["P0", "P1", "P2"]
    ranker_id: Literal["R0", "R1", "R2"]
    budget: RunBudget
    prompt_versions: dict[str, str]
    ranker_weights_version: str | None = None
    seed: int | None = None

    @field_validator("prompt_versions")
    @classmethod
    def freeze_prompt_versions(cls, value: dict[str, str]) -> dict[str, str]:
        return _freeze_mapping(value)

    @field_serializer("prompt_versions", when_used="json")
    def serialize_prompt_versions(self, value: dict[str, str]) -> dict[str, str]:
        return {key: value[key] for key in sorted(value)}

    @model_validator(mode="after")
    def require_cost_budget_for_profile(self) -> Self:
        cost_required = (
            self.request.access_profile == "public_live"
            or self.request.run_purpose == "benchmark"
        )
        if cost_required and self.budget.max_cost_usd is None:
            raise ValueError("max_cost_usd is required for public_live and benchmark runs")
        return self


def _node_names() -> tuple[_NodeName, ...]:
    return ("Planner", "Ranker", "Writer", "Judge", "Tool")


__all__ = ["ResourceUsage", "RunBudget", "RunConfig"]
