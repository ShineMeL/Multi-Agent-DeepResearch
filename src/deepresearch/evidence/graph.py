from __future__ import annotations

from dataclasses import dataclass

from deepresearch.domain import Claim, ClaimEvidenceLink, EvidenceSpan


@dataclass(frozen=True)
class GraphValidationResult:
    valid: bool
    error_codes: tuple[str, ...]


class ClaimEvidenceGraph:
    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}
        self._evidence: dict[str, EvidenceSpan] = {}
        self._links: list[ClaimEvidenceLink] = []

    def add_claim(self, claim: Claim) -> None:
        if type(claim) is not Claim:
            raise TypeError("claim must be a Claim")
        if claim.claim_id in self._claims:
            raise ValueError(f"duplicate claim: {claim.claim_id}")
        self._claims[claim.claim_id] = claim

    def add_evidence(self, evidence: EvidenceSpan) -> None:
        if type(evidence) is not EvidenceSpan:
            raise TypeError("evidence must be an EvidenceSpan")
        if evidence.evidence_id in self._evidence:
            raise ValueError(f"duplicate evidence: {evidence.evidence_id}")
        self._evidence[evidence.evidence_id] = evidence

    def add_link(self, link: ClaimEvidenceLink) -> None:
        if type(link) is not ClaimEvidenceLink:
            raise TypeError("link must be a ClaimEvidenceLink")
        if link.claim_id not in self._claims:
            raise ValueError("unknown claim")
        if link.evidence_id not in self._evidence:
            raise ValueError("unknown evidence")
        if link in self._links:
            raise ValueError("duplicate link")
        self._links.append(link)

    def links_for_claim(self, claim_id: str) -> tuple[ClaimEvidenceLink, ...]:
        return tuple(
            sorted(
                (link for link in self._links if link.claim_id == claim_id),
                key=lambda link: (link.evidence_id, link.relation),
            )
        )

    def validate(self) -> GraphValidationResult:
        errors: set[str] = set()
        seen: set[ClaimEvidenceLink] = set()
        for link in self._links:
            if link.claim_id not in self._claims:
                errors.add("UNKNOWN_CLAIM")
            if link.evidence_id not in self._evidence:
                errors.add("UNKNOWN_EVIDENCE")
            if link in seen:
                errors.add("DUPLICATE_LINK")
            seen.add(link)
        ordered = tuple(sorted(errors))
        return GraphValidationResult(valid=not ordered, error_codes=ordered)

    def to_json(self) -> dict[str, object]:
        return {
            "claims": [
                self._claims[claim_id].model_dump(mode="json")
                for claim_id in sorted(self._claims)
            ],
            "evidence": [
                self._evidence[evidence_id].model_dump(mode="json")
                for evidence_id in sorted(self._evidence)
            ],
            "links": [
                link.model_dump(mode="json")
                for link in sorted(
                    self._links,
                    key=lambda item: (item.claim_id, item.evidence_id, item.relation),
                )
            ],
        }


__all__ = ["ClaimEvidenceGraph", "GraphValidationResult"]
