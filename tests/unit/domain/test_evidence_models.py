from datetime import UTC, datetime
from math import inf, nan

import pytest
from pydantic import ValidationError

from deepresearch.domain import (
    Claim,
    ClaimEvidenceLink,
    ClaimType,
    CoverageLedgerEntry,
    EvidenceSpan,
    HtmlLocator,
    RerankScore,
    SourceDocument,
    VerificationStatus,
)

SHA256 = "a" * 64


def source_document(**updates: object) -> SourceDocument:
    payload: dict[str, object] = {
        "source_id": "src-1",
        "canonical_url": "https://example.com/source",
        "title": "Source",
        "authors": ("Author",),
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "retrieved_at": datetime(2026, 1, 2, tzinfo=UTC),
        "content_hash": SHA256,
        "parsed_content_hash": "b" * 64,
        "source_type": "paper",
        "source_family_id": "family-1",
        "parser_version": "v1",
    }
    payload.update(updates)
    return SourceDocument.model_validate(payload)


def evidence_span(**updates: object) -> EvidenceSpan:
    payload: dict[str, object] = {
        "evidence_id": "ev-1",
        "source_id": "src-1",
        "locator": HtmlLocator(paragraph_id="p-1", start_char=0, end_char=5),
        "excerpt": "proof",
        "excerpt_hash": SHA256,
        "language": "en",
        "information_need_ids": ("need-1",),
    }
    payload.update(updates)
    return EvidenceSpan.model_validate(payload)


@pytest.mark.parametrize("field", ["content_hash", "parsed_content_hash"])
@pytest.mark.parametrize("bad_hash", ["bad", "A" * 64, "g" * 64, "a" * 63])
def test_source_document_requires_lowercase_sha256(field: str, bad_hash: str) -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        source_document(**{field: bad_hash})


@pytest.mark.parametrize("field", ["published_at", "retrieved_at"])
def test_source_document_datetimes_require_timezone(field: str) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        source_document(**{field: datetime(2026, 1, 1)})  # noqa: DTZ001 - deliberately naive


def test_evidence_requires_lowercase_sha256_and_valid_locator_range() -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        evidence_span(excerpt_hash="bad")
    with pytest.raises(ValidationError):
        evidence_span(
            locator={"kind": "html", "paragraph_id": "p", "start_char": 5, "end_char": 5}
        )


def test_evidence_locator_round_trips_through_discriminator() -> None:
    evidence = evidence_span(
        locator={
            "kind": "pdf",
            "page_index": 0,
            "block_index": 2,
            "start_char": 1,
            "end_char": 3,
        }
    )

    assert evidence.model_dump(mode="json")["locator"]["kind"] == "pdf"
    assert EvidenceSpan.model_validate_json(evidence.model_dump_json()) == evidence


def test_claim_and_link_preserve_canonical_fields() -> None:
    claim = Claim(
        claim_id="claim-1",
        text="The value increased.",
        claim_type="trend",
        entities=("value",),
        numbers=(),
        qualifiers=(),
        report_section="findings",
        verification_status="supported",
    )
    link = ClaimEvidenceLink(
        claim_id=claim.claim_id,
        evidence_id="ev-1",
        relation="support",
        entailment_score=0.9,
        relevance_score=0.8,
        judge_model="judge",
        prompt_version="v1",
        decision_code="entailed",
    )

    assert set(Claim.model_fields) == {
        "claim_id",
        "text",
        "claim_type",
        "entities",
        "numbers",
        "qualifiers",
        "report_section",
        "verification_status",
    }
    assert set(ClaimEvidenceLink.model_fields) == {
        "claim_id",
        "evidence_id",
        "relation",
        "entailment_score",
        "relevance_score",
        "judge_model",
        "prompt_version",
        "decision_code",
    }
    assert link.relation == "support"


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_all_evidence_score_models_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        ClaimEvidenceLink(
            claim_id="claim-1",
            evidence_id="ev-1",
            relation="support",
            entailment_score=value,
            relevance_score=0.5,
            judge_model="judge",
            prompt_version="v1",
            decision_code="decision",
        )
    with pytest.raises(ValidationError):
        CoverageLedgerEntry(
            subquestion_id="sq-1",
            coverage_score=value,
            independent_source_count=1,
            unresolved_conflict_ids=(),
            uncertainty_score=0.2,
            last_marginal_gain=0.1,
            evidence_ids=("ev-1",),
            attempt_count=1,
            last_decision_code="continue",
        )
    with pytest.raises(ValidationError):
        RerankScore(
            evidence_id="ev-1",
            total=0.5,
            feature_scores={"semantic": value},
        )


def test_claim_literals_are_public_contract_types() -> None:
    claim_type: ClaimType = "numeric"
    verification_status: VerificationStatus = "uncertain"

    assert claim_type == "numeric"
    assert verification_status == "uncertain"


def test_rerank_feature_serialization_is_canonical() -> None:
    first = RerankScore(
        evidence_id="ev-1",
        total=0.5,
        feature_scores={"z_score": 0.2, "a_score": 0.8},
    )
    second = RerankScore(
        evidence_id="ev-1",
        total=0.5,
        feature_scores={"a_score": 0.8, "z_score": 0.2},
    )

    assert first.model_dump_json() == second.model_dump_json()
