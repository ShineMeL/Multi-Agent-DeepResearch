from __future__ import annotations

import base64
import hashlib
from datetime import UTC, date, datetime
from typing import Any, cast

import pytest
from pydantic import ValidationError

from benchmarks.datasets.models import (
    AnnotatedQuestion,
    DatasetManifest,
    FrozenEvidenceRecord,
    GoldEvidenceSpan,
    PrivateDatasetManifest,
    RuntimeTask,
    TaskCategory,
)
from deepresearch.domain import Claim, FreshnessRequirement, HtmlLocator, ResearchRequest


def _request() -> ResearchRequest:
    return ResearchRequest(
        question="How do planner strategies compare?",
        output_requirements={"answer_shape": "markdown"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="fixture-profile",
        run_purpose="benchmark",
        budget_preset="low",
    )


def _claim(claim_id: str = "claim-1") -> Claim:
    return Claim(
        claim_id=claim_id,
        text="Planner strategy claim.",
        claim_type="fact",
        entities=(),
        numbers=(),
        qualifiers=(),
        report_section="findings",
        verification_status="supported",
    )


def _question(**updates: object) -> dict[str, object]:
    evidence = {
        "evidence_id": "ev-1",
        "source_id": "src-1",
        "locator": {
            "kind": "html",
            "paragraph_id": "p-1",
            "start_char": 0,
            "end_char": 5,
        },
        "relevance_grade": 3,
        "excerpt_hash": hashlib.sha256(b"hello").hexdigest(),
    }
    payload: dict[str, object] = {
        "task_id": "dev-ts-01",
        "split": "dev",
        "category": "technical_survey",
        "request": _request(),
        "evaluation_cutoff": date(2026, 8, 29),
        "information_needs": [
            {
                "need_id": "need-1",
                "text": "Main strategy",
                "importance": 1.0,
                "acceptable_claim_ids": ["claim-1"],
            }
        ],
        "acceptable_claims": [_claim().model_dump(mode="json")],
        "candidate_source_ids": ["src-1"],
        "gold_source_family_ids": ["family-1"],
        "snapshot_id": "snapshot-1",
        "corpus_version": "corpus-v1",
        "index_version": "index-v1",
        "gold_evidence_spans": [evidence],
        "gold_claim_links": [
            {
                "claim_id": "claim-1",
                "evidence_links": [
                    {"evidence_id": "ev-1", "relation": "support"}
                ],
            }
        ],
        "rubric": {
            "coverage": {
                "rubric_id": "coverage",
                "description": "Coverage",
                "weight": 0.5,
                "levels": {0: "none", 1: "low", 2: "mid", 3: "high"},
            },
            "evidence": {
                "rubric_id": "evidence",
                "description": "Evidence",
                "weight": 0.5,
                "levels": {0: "none", 1: "low", 2: "mid", 3: "high"},
            },
        },
        "expected_stop_reason": "SUFFICIENT",
        "expected_is_partial": False,
        "created_at": date(2026, 8, 29),
        "annotation_version": "v1",
    }
    payload.update(updates)
    return payload


def test_annotated_question_rejects_cross_task_gold_link() -> None:
    question = _question(
        gold_claim_links=[
            {
                "claim_id": "claim-other",
                "evidence_links": [{"evidence_id": "ev-1", "relation": "support"}],
            }
        ]
    )
    with pytest.raises(ValidationError, match="unknown claim_id"):
        AnnotatedQuestion.model_validate(question)


def test_gold_relevance_is_graded_zero_to_three() -> None:
    with pytest.raises(ValidationError):
        GoldEvidenceSpan(
            evidence_id="ev-1",
            source_id="src-1",
            locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=5),
            relevance_grade=cast("Any", 4),
            excerpt_hash="a" * 64,
        )


def test_category_values_are_frozen() -> None:
    assert {item.value for item in TaskCategory} == {
        "technical_survey",
        "method_comparison",
        "multi_hop_history",
        "freshness",
        "bilingual",
        "source_conflict",
    }


def test_question_rejects_duplicate_ids_empty_links_and_bad_weights() -> None:
    duplicate = _question(
        information_needs=[
            {
                "need_id": "need-1",
                "text": "Main strategy",
                "importance": 1.0,
                "acceptable_claim_ids": ["claim-1"],
            },
            {
                "need_id": "need-1",
                "text": "Duplicate",
                "importance": 0.1,
                "acceptable_claim_ids": ["claim-1"],
            },
        ]
    )
    with pytest.raises(ValidationError, match="duplicate information need"):
        AnnotatedQuestion.model_validate(duplicate)

    with pytest.raises(ValidationError, match="evidence_links"):
        AnnotatedQuestion.model_validate(
            _question(gold_claim_links=cast("Any", [{"claim_id": "claim-1", "evidence_links": []}]))
        )

    with pytest.raises(ValidationError, match="rubric weights"):
        AnnotatedQuestion.model_validate(
            _question(
                rubric={
                    "coverage": {
                        "rubric_id": "coverage",
                        "description": "Coverage",
                        "weight": 0.25,
                        "levels": {0: "none", 1: "low", 2: "mid", 3: "high"},
                    }
                }
            )
        )


def test_question_preserves_support_and_contradict_links_but_rejects_duplicates() -> None:
    payload = _question(
        gold_claim_links=[
            {
                "claim_id": "claim-1",
                "evidence_links": [
                    {"evidence_id": "ev-1", "relation": "support"},
                    {"evidence_id": "ev-1", "relation": "contradict"},
                ],
            }
        ]
    )
    question = AnnotatedQuestion.model_validate(payload)
    assert {link.relation for link in question.gold_claim_links[0].evidence_links} == {
        "support",
        "contradict",
    }

    with pytest.raises(ValidationError, match="duplicate evidence link"):
        AnnotatedQuestion.model_validate(
            _question(
                gold_claim_links=[
                    {
                        "claim_id": "claim-1",
                        "evidence_links": [
                            {"evidence_id": "ev-1", "relation": "support"},
                            {"evidence_id": "ev-1", "relation": "support"},
                        ],
                    }
                ]
            )
        )


def test_frozen_evidence_record_verifies_raw_and_parsed_hashes_and_locator() -> None:
    text = "hello"
    raw = text.encode("utf-8")
    record = FrozenEvidenceRecord(
        task_id="dev-ts-01",
        evidence_id="ev-1",
        source_id="src-1",
        source_family_id="family-1",
        canonical_url=cast("Any", "https://example.test/doc"),
        title="Fixture",
        authors=(),
        media_type="text/html",
        raw_body_b64=base64.b64encode(raw).decode("ascii"),
        content_hash=hashlib.sha256(raw).hexdigest(),
        normalized_text=text,
        parsed_content_hash=hashlib.sha256(raw).hexdigest(),
        locator_text=text,
        locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=5),
        excerpt=text,
        excerpt_hash=hashlib.sha256(raw).hexdigest(),
        published_at=datetime(2026, 8, 29, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
        language="en",
        source_type="paper",
    )
    assert record.evidence_id == "ev-1"

    with pytest.raises(ValidationError, match="content_hash"):
        FrozenEvidenceRecord.model_validate(
            {**record.model_dump(mode="json"), "content_hash": "a" * 64}
        )
    with pytest.raises(ValidationError, match="locator"):
        FrozenEvidenceRecord.model_validate(
            {
                **record.model_dump(mode="json"),
                "locator": {
                    "kind": "html",
                    "paragraph_id": "p-1",
                    "start_char": 1,
                    "end_char": 5,
                },
            }
        )


def test_frozen_evidence_requires_exactly_one_publication_explanation() -> None:
    base = FrozenEvidenceRecord.model_validate(
        {
            "task_id": "dev-ts-01",
            "evidence_id": "ev-1",
            "source_id": "src-1",
            "source_family_id": "family-1",
            "canonical_url": "https://example.test/doc",
            "title": "Fixture",
            "authors": [],
            "media_type": "text/html",
            "raw_body_b64": base64.b64encode(b"hello").decode("ascii"),
            "content_hash": hashlib.sha256(b"hello").hexdigest(),
            "normalized_text": "hello",
            "parsed_content_hash": hashlib.sha256(b"hello").hexdigest(),
            "locator_text": "hello",
            "locator": {"kind": "html", "paragraph_id": "p-1", "start_char": 0, "end_char": 5},
            "excerpt": "hello",
            "excerpt_hash": hashlib.sha256(b"hello").hexdigest(),
            "published_at": None,
            "unknown_published_at_reason": "not supplied by source",
            "retrieved_at": "2026-08-29T00:00:00Z",
            "language": "en",
            "source_type": "paper",
        }
    )
    assert base.unknown_published_at_reason is not None
    with pytest.raises(ValidationError, match="exactly one"):
        FrozenEvidenceRecord.model_validate(
            {**base.model_dump(mode="json"), "published_at": "2026-08-29T00:00:00Z"}
        )


def test_runtime_view_has_no_private_annotation_fields() -> None:
    runtime = RuntimeTask(
        task_id="dev-ts-01",
        category=TaskCategory.TECHNICAL_SURVEY,
        request=_request(),
        evaluation_cutoff=date(2026, 8, 29),
        snapshot_id="snapshot-1",
        corpus_version="corpus-v1",
        index_version="index-v1",
    )
    payload = runtime.model_dump(mode="json")
    assert set(payload) == {
        "task_id",
        "category",
        "request",
        "evaluation_cutoff",
        "snapshot_id",
        "corpus_version",
        "index_version",
    }
    assert "gold" not in repr(payload).casefold()
    assert "rubric" not in repr(payload).casefold()


def test_public_and_private_manifests_validate_hashes_and_sealed_ids() -> None:
    public = DatasetManifest(
        dataset_id="frozen-ai-cs-60",
        version="v1",
        record_count=1,
        split_counts={"dev": 1, "test": 0},
        category_counts={TaskCategory.TECHNICAL_SURVEY: 1},
        public_runtime_files=["runtime/dev/technical_survey.jsonl"],
        private_manifest_sha256="a" * 64,
        snapshot_collection_sha256="b" * 64,
        cost_subset_sha256="c" * 64,
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    assert public.record_count == 1
    private = PrivateDatasetManifest(
        dataset_id=public.dataset_id,
        version=public.version,
        record_count=1,
        split_counts={"dev": 1, "test": 0},
        category_counts={TaskCategory.TECHNICAL_SURVEY: 1},
        batch_sha256={TaskCategory.TECHNICAL_SURVEY: "d" * 64},
        snapshot_manifest_sha256={"task-1": "e" * 64},
        public_runtime_files=list(public.public_runtime_files),
        private_test_runtime_files=[],
        main_test_task_ids=("task-1",),
        stability_task_ids=(),
        cost_subset_task_ids=(),
        p0_task_ids=(),
        oracle_task_ids=(),
        subset_seed=7,
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    assert private.main_test_task_ids == ("task-1",)
