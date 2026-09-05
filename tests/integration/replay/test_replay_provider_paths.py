from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from deepresearch.domain import (
    Claim,
    ClaimEvidenceLink,
    EvidenceSpan,
    HtmlLocator,
    RunBudget,
    SourceDocument,
)
from deepresearch.evidence import SimilarityRanker
from deepresearch.evidence.citation_guard import CitationGuard, CitationMaterialResolver
from deepresearch.evidence.features import DefaultEvidenceFeatureCalculator
from deepresearch.evidence.graph import ClaimEvidenceGraph
from deepresearch.evidence.rankers import R1SimilarityOnly, R2EvidenceUtility
from deepresearch.planning.query_scheduler import QueryScheduler
from deepresearch.providers import ProviderError
from deepresearch.providers.replay import (
    ReplayBundle,
    ReplayFetcher,
    ReplaySearchProvider,
    ReplayTextEmbedder,
)
from deepresearch.runtime import BudgetAccountant, CancellationToken

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "replay" / "planner_ranker"
BASELINE_EVIDENCE = Path(__file__).parents[2] / "fixtures" / "replay" / "baseline" / "expected-evidence.json"


def _deadline() -> float:
    return time.monotonic() + 30.0


def _evidence_by_id(*evidence_ids: str) -> tuple[EvidenceSpan, ...]:
    payload = json.loads(BASELINE_EVIDENCE.read_text(encoding="utf-8"))
    evidence = {
        item.evidence_id: item
        for item in (
            EvidenceSpan.model_validate_json(
                json.dumps(raw, ensure_ascii=False).encode("utf-8"), strict=True
            )
            for raw in cast("list[object]", payload["evidence"])
        )
    }
    return tuple(evidence[evidence_id] for evidence_id in evidence_ids)


@pytest.mark.asyncio
async def test_replay_unknown_query_returns_replay_miss_and_known_query_is_strict() -> None:
    bundle = ReplayBundle.load(FIXTURE_ROOT)
    verification = bundle.verify()
    assert verification.valid is True

    provider = ReplaySearchProvider(bundle)
    hits = await provider.search(
        "planner comparison gamma",
        10,
        None,
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert hits
    assert hits[0].title == "Alpha planner study"
    assert hits[0].provider_metadata["fixture"] == "baseline"

    with pytest.raises(ProviderError) as error:
        await provider.search(
            "unknown query",
            10,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    assert error.value.code == "REPLAY_MISS"
    assert provider.live_calls == 0

    document = await ReplayFetcher(bundle).fetch(
        "https://alpha.example/strategy",
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert document.content_type == "text/html"


@pytest.mark.asyncio
async def test_replay_embedder_is_shared_by_every_semantic_consumer() -> None:
    bundle = ReplayBundle.load(FIXTURE_ROOT)
    embedder = ReplayTextEmbedder(bundle)
    scheduler = QueryScheduler(
        embedder=embedder,
        budget=BudgetAccountant(RunBudget.preset("low")),
    )
    similarity = SimilarityRanker(embedder)
    r1 = R1SimilarityOnly(delegate=similarity)
    feature_calculator = DefaultEvidenceFeatureCalculator(
        embedder=embedder,
        embedding_model_id=embedder.model_id,
        materials=cast("Any", object()),
        support_judge=cast("Any", object()),
    )
    r2 = R2EvidenceUtility(feature_calculator=feature_calculator)

    assert scheduler._embedder is embedder  # pyright: ignore[reportPrivateUsage]
    assert cast("Any", r1.delegate).embedder is embedder
    assert cast("Any", r2.feature_calculator).embedder is embedder

    scores = await similarity.score(
        "Main planner strategies",
        _evidence_by_id(
            "E-0573288ab889d131de9536e66e444935f957e458ecdbe061b3a208189d734f01",
            "E-515ca32979a4d1af47d51e3b5b26f5baaadacc21065285c49f3ed0b6be4f2127",
            "E-54af2bb90230bb8e406c1004f19976e403eb8087076a3377df0ca9ce4307fca5",
        ),
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert len(scores) == 3
    vectors = await embedder.embed(
        (
            "Main planner strategies",
            "Gamma planner evidence provides an independent comparison.",
            "Beta planner evidence offers a second strategy perspective.",
            "Alpha planner evidence describes a primary strategy route.",
        ),
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert len(vectors) == 4
    assert all(len(vector) == 384 for vector in vectors)


class _CitationMaterials(CitationMaterialResolver):
    def __init__(self, *, source_id: str, text: str) -> None:
        self.source_id = source_id
        self.text = text
        self.raw = text.encode("utf-8")

    def raw_bytes_for_source(self, source_id: str) -> bytes:
        if source_id != self.source_id:
            raise LookupError(source_id)
        return self.raw

    def normalized_document_text(self, source_id: str) -> str:
        if source_id != self.source_id:
            raise LookupError(source_id)
        return self.text

    def html_paragraph_text(self, source_id: str, paragraph_id: str) -> str:
        if source_id != self.source_id or paragraph_id != "main-0":
            raise LookupError(paragraph_id)
        return self.text

    def pdf_block_text(self, source_id: str, page_index: int, block_index: int) -> str:
        raise LookupError((source_id, page_index, block_index))


def test_replay_path_can_emit_a_valid_claim_graph_and_citation_guard_result() -> None:
    evidence_id = "E-" + "a" * 64
    source_id = "S-replay"
    excerpt = "Replay evidence supports a planner claim."
    source = SourceDocument(
        source_id=source_id,
        canonical_url=cast("Any", "https://replay.example/evidence"),
        title="Replay evidence",
        authors=(),
        retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
        content_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        parsed_content_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        source_type="paper",
        source_family_id="replay.example",
        parser_version="replay-parser-v1",
    )
    span = EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        locator=HtmlLocator(
            paragraph_id="main-0",
            start_char=0,
            end_char=len(excerpt),
        ),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        language="en",
        information_need_ids=("need-1",),
    )
    claim = Claim(
        claim_id="claim-replay",
        text=excerpt,
        claim_type="fact",
        entities=(),
        numbers=(),
        qualifiers=(),
        report_section="findings",
        verification_status="supported",
    )
    link = ClaimEvidenceLink(
        claim_id=claim.claim_id,
        evidence_id=evidence_id,
        relation="support",
        entailment_score=1.0,
        relevance_score=1.0,
        judge_model="replay-judge-v1",
        prompt_version="replay-judge-v1",
        decision_code="EVIDENCE_SUPPORT",
    )
    graph = ClaimEvidenceGraph()
    graph.add_claim(claim)
    graph.add_evidence(span)
    graph.add_link(link)

    result = CitationGuard().verify(
        f"{excerpt} [{evidence_id}]",
        graph,
        {evidence_id: span},
        {source_id: source},
        materials=_CitationMaterials(source_id=source_id, text=excerpt),
    )
    assert graph.validate().valid is True
    assert result.valid is True
    assert result.checked_citation_ids == (evidence_id,)
