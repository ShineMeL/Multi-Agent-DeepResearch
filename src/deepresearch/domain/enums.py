from typing import Literal

RunStatus = Literal[
    "queued", "running", "interrupted", "completed", "failed", "cancelled"
]
StopReason = Literal["SUFFICIENT", "PLATEAU", "BUDGET_EXHAUSTED", "BLOCKED"]
ExecutionMode = Literal["live", "replay", "hybrid"]
AccessProfile = Literal["showcase", "public_live", "local"]
RunPurpose = Literal["demo", "benchmark", "test"]
SourceType = Literal[
    "paper",
    "official_documentation",
    "standard",
    "primary_data",
    "first_party_statement",
    "secondary_analysis",
    "news",
    "unknown",
]
ClaimType = Literal["fact", "numeric", "comparison", "trend", "causal", "limitation"]
VerificationStatus = Literal["supported", "contradicted", "uncertain", "unsupported"]

__all__ = [
    "AccessProfile",
    "ClaimType",
    "ExecutionMode",
    "RunPurpose",
    "RunStatus",
    "SourceType",
    "StopReason",
    "VerificationStatus",
]
