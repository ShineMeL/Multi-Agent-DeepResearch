from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Sequence
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deepresearch.domain import Claim, ClaimEvidenceLink, EvidenceSpan
from deepresearch.providers import Deadline, ModelMessage, ModelProvider, ModelRequest
from deepresearch.runtime import BudgetAccountant, CancellationToken, ResourceEstimate


class _ClaimsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: tuple[Claim, ...] = Field(default_factory=tuple)


class _LinksOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    links: tuple[ClaimEvidenceLink, ...] = Field(default_factory=tuple)


def _hash_payload(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request(
    provider: ModelProvider,
    *,
    payload: object,
    prompt_version: str,
    output_schema: type[BaseModel],
) -> ModelRequest:
    system = "Return only public atomic claims or claim-evidence links; do not include hidden reasoning."
    return ModelRequest(
        model_id=getattr(provider, "model_id", provider.provider_id),
        messages=(
            ModelMessage(role="system", content=system),
            ModelMessage(
                role="user",
                content=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ),
        ),
        temperature=Decimal(0),
        seed=0,
        max_output_tokens=512,
        prompt_version=prompt_version,
        system_prompt_hash=_hash_payload(system),
        tool_schema_hash=_hash_payload([]),
        output_schema_hash=_hash_payload(output_schema.model_json_schema()),
    )


async def _settle_model_call(
    provider: ModelProvider,
    request: ModelRequest,
    output_schema: type[BaseModel],
    *,
    deadline: Deadline,
    cancellation_token: CancellationToken,
    budget: BudgetAccountant | None,
    idempotency_key: str,
) -> BaseModel:
    reservation = None
    if budget is not None:
        reservation = budget.reserve(
            ResourceEstimate(tokens=512, wall_seconds=5.0, cost_usd=Decimal(0)),
            node="Ranker",
            idempotency_key=idempotency_key,
        )
    try:
        result = await provider.structured(
            request,
            output_schema,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        if budget is not None and reservation is not None:
            actual = result.usage
            if budget.snapshot().used_cost_usd is not None and actual.cost_usd is None:
                actual = actual.model_copy(update={"cost_usd": Decimal(0)})
            budget.settle(reservation, actual=actual)
        return result.output
    except BaseException:
        if budget is not None and reservation is not None:
            try:
                budget.release(reservation)
            except (RuntimeError, ValueError):
                pass
        raise


class ClaimExtractor:
    def __init__(
        self,
        *,
        model_provider: ModelProvider | None = None,
        budget: BudgetAccountant | None = None,
        model_id: str = "claim-extractor-v1",
        prompt_version: str = "claim-extractor-v1",
    ) -> None:
        self.model_provider = model_provider
        self.budget = budget
        self.model_id = model_id
        self.prompt_version = prompt_version

    async def extract(
        self,
        draft_markdown: str,
        *,
        evidence_ids: Collection[str],
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[Claim]:
        cancellation_token.raise_if_cancelled()
        if self.model_provider is not None:
            output = await _settle_model_call(
                self.model_provider,
                _request(
                    self.model_provider,
                    payload={
                        "draft_markdown": draft_markdown,
                        "evidence_ids": sorted(evidence_ids),
                    },
                    prompt_version=self.prompt_version,
                    output_schema=_ClaimsOutput,
                ),
                _ClaimsOutput,
                deadline=deadline,
                cancellation_token=cancellation_token,
                budget=self.budget,
                idempotency_key="claims:" + hashlib.sha256(draft_markdown.encode()).hexdigest(),
            )
            assert isinstance(output, _ClaimsOutput)
            return list(output.claims)

        allowed_ids = set(evidence_ids)
        claims: list[Claim] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", draft_markdown):
            text = re.sub(r"\s*\[[^\]]+\]", "", sentence).strip()
            citations = set(re.findall(r"\[([^\]]+)\]", sentence))
            if not text or (citations and not citations & allowed_ids):
                continue
            claim_id = "claim-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=text,
                    claim_type="fact",
                    entities=(),
                    numbers=tuple(re.findall(r"\b\d+(?:\.\d+)?\b", text)),
                    qualifiers=(),
                    report_section="findings",
                    verification_status="uncertain",
                )
            )
        return claims


class EvidenceJudge:
    def __init__(
        self,
        *,
        model_provider: ModelProvider | None = None,
        budget: BudgetAccountant | None = None,
        model_id: str = "evidence-judge-v1",
        prompt_version: str = "evidence-judge-v1",
    ) -> None:
        self.model_provider = model_provider
        self.budget = budget
        self.model_id = model_id
        self.prompt_version = prompt_version

    async def judge(
        self,
        claim: Claim,
        evidence: Sequence[EvidenceSpan],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[ClaimEvidenceLink]:
        cancellation_token.raise_if_cancelled()
        if self.model_provider is not None:
            output = await _settle_model_call(
                self.model_provider,
                _request(
                    self.model_provider,
                    payload={
                        "claim": claim.model_dump(mode="json"),
                        "evidence": [item.model_dump(mode="json") for item in evidence],
                    },
                    prompt_version=self.prompt_version,
                    output_schema=_LinksOutput,
                ),
                _LinksOutput,
                deadline=deadline,
                cancellation_token=cancellation_token,
                budget=self.budget,
                idempotency_key="judge:" + hashlib.sha256(claim.claim_id.encode()).hexdigest(),
            )
            assert isinstance(output, _LinksOutput)
            return list(output.links)

        links: list[ClaimEvidenceLink] = []
        claim_terms = set(re.findall(r"\w+", claim.text.casefold()))
        for item in evidence:
            evidence_terms = set(re.findall(r"\w+", item.excerpt.casefold()))
            overlap = len(claim_terms & evidence_terms)
            relation: Literal["support", "contradict", "context", "insufficient"] = (
                "support" if overlap else "insufficient"
            )
            score = 1.0 if overlap else 0.0
            links.append(
                ClaimEvidenceLink(
                    claim_id=claim.claim_id,
                    evidence_id=item.evidence_id,
                    relation=relation,
                    entailment_score=score,
                    relevance_score=score,
                    judge_model=self.model_id,
                    prompt_version=self.prompt_version,
                    decision_code=f"EVIDENCE_{relation.upper()}",
                )
            )
        return links


__all__ = ["ClaimExtractor", "EvidenceJudge"]
