from datetime import datetime

from pydantic import BaseModel, ConfigDict, JsonValue, field_serializer, field_validator

from .enums import RunStatus, StopReason
from .research import _freeze_json_mapping  # pyright: ignore[reportPrivateUsage]
from .usage import ResourceUsage


def _canonical_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seq: int
    run_id: str
    timestamp: datetime
    node: str
    kind: str
    status: RunStatus
    public_payload: dict[str, JsonValue]
    usage_delta: ResourceUsage
    artifact_ids: tuple[str, ...]
    error_code: str | None = None

    @field_validator("public_payload")
    @classmethod
    def freeze_public_payload(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)

    @field_serializer("public_payload", when_used="json")
    def serialize_public_payload(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {key: _canonical_json(value[key]) for key in sorted(value)}

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    thread_id: str
    status: RunStatus
    stop_reason: StopReason | None = None
    is_partial: bool
    report_artifact_id: str | None = None
    evidence_graph_artifact_id: str | None = None
    manifest_artifact_id: str | None = None
    final_usage: ResourceUsage
    error_code: str | None = None


__all__ = ["RunEvent", "RunResult"]
