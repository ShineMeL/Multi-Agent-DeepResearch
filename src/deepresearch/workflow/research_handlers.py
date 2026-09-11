from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

from langgraph.runtime import get_runtime  # pyright: ignore[reportMissingTypeStubs]

from deepresearch.domain import Claim, EvidenceSpan, RunConfig, RunEvent
from deepresearch.evidence.claims import ClaimExtractor, EvidenceJudge
from deepresearch.evidence.graph import ClaimEvidenceGraph
from deepresearch.storage import ArtifactIntegrityError, LocalArtifactStore, LocalEvidenceStore

from .baseline_graph import (
    BaselineNode,
    BaselineNodeHandlers,
    BaselineRuntimeContext,
    ReceiptStateBuilder,
    ReceiptStateRecovery,
    StateUpdate,
    WorkflowInvariantError,
    _AuditComposition,  # pyright: ignore[reportPrivateUsage]
    _effective_deadline,  # pyright: ignore[reportPrivateUsage]
    _hash_json,  # pyright: ignore[reportPrivateUsage]
    _load_audit_receipt,  # pyright: ignore[reportPrivateUsage]
    _safe_node,  # pyright: ignore[reportPrivateUsage]
    _state_payload,  # pyright: ignore[reportPrivateUsage]
)
from .research_graph import (
    ClaimResolutionRecord,
    InitialPlanNode,
    NodeHandler,
    ResearchGraphDependencies,
)
from .state import BaselineState, ResearchState, validate_research_state

_BASELINE_FIELDS = frozenset(BaselineState.__annotations__)
_RESEARCH_FIELDS = frozenset(ResearchState.__annotations__) - _BASELINE_FIELDS
_RESEARCH_TUPLE_FIELDS = frozenset({"claim_ids", "unsupported_claim_ids"})
_CLAIM_GRAPH_MEDIA_TYPE = "application/vnd.deepresearch.claim-graph+json"
_CLAIM_RESOLUTION_MEDIA_TYPE = "application/vnd.deepresearch.claim-resolution+json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _research_extension(state: ResearchState | Mapping[str, object]) -> dict[str, object]:
    """Encode only research channels in their strict JSON representation."""

    restored = validate_research_state(cast("Mapping[str, object]", state))
    restored_mapping = cast("Mapping[str, object]", restored)
    extension: dict[str, object] = {}
    for name in sorted(_RESEARCH_FIELDS):
        if name not in state:
            continue
        value = restored_mapping[name]
        if name in _RESEARCH_TUPLE_FIELDS:
            if type(value) is not tuple:
                raise ArtifactIntegrityError("research state receipt is corrupt")
            extension[name] = list(cast("tuple[str, ...]", value))
        else:
            extension[name] = value
    # The dump/load round trip below rejects values that are not JSON-safe and
    # gives callers a deterministic object shape for receipt payloads.
    encoded = _canonical_json(extension)
    decoded = json.loads(encoded)
    if type(decoded) is not dict:
        raise ArtifactIntegrityError("research state receipt is corrupt")
    return cast("dict[str, object]", decoded)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON constant")


def _decode_research_extension(
    value: object,
    *,
    baseline: BaselineState,
) -> tuple[dict[str, object], dict[str, object]]:
    """Decode and validate a canonical input/output research receipt pair."""

    if type(value) is not dict:
        raise ArtifactIntegrityError("research state receipt is missing or corrupt")
    pair = cast("dict[str, object]", value)
    if set(pair) != {"input", "output"}:
        raise ArtifactIntegrityError("research state receipt is missing or corrupt")

    def decode_one(raw: object) -> dict[str, object]:
        if type(raw) is not dict:
            raise ArtifactIntegrityError("research state receipt is corrupt")
        candidate = cast("dict[str, object]", raw)
        if any(type(key) is not str or key not in _RESEARCH_FIELDS for key in candidate):
            raise ArtifactIntegrityError("research state receipt is corrupt")
        converted: dict[str, object] = {}
        for name, item in candidate.items():
            if name in _RESEARCH_TUPLE_FIELDS:
                if type(item) is not list:
                    raise ArtifactIntegrityError("research state receipt is corrupt")
                members = cast("list[object]", item)
                if any(type(member) is not str for member in members):
                    raise ArtifactIntegrityError("research state receipt is corrupt")
                converted[name] = tuple(cast("list[str]", members))
            else:
                converted[name] = item
        try:
            validated = validate_research_state({**baseline, **converted})
        except (TypeError, ValueError):
            raise ArtifactIntegrityError("research state receipt is corrupt") from None
        # Compare the parsed object to our canonical encoder so a receipt can
        # never smuggle alternate tuple/list or numeric representations.
        if _research_extension(validated) != candidate:
            raise ArtifactIntegrityError("research state receipt is not canonical")
        return converted

    return decode_one(pair["input"]), decode_one(pair["output"])


class _ResearchAuditComposition:
    """Research-aware facade over Core's baseline audit composition."""

    def __init__(self, baseline: _AuditComposition) -> None:
        # A distinct graph version makes the research execution auditable while
        # preserving every provider/pricing/provenance field from Core.
        self._baseline = replace(baseline, graph_version="research-graph-v1")

    def __getattr__(self, name: str) -> object:
        return getattr(self._baseline, name)

    def create_run_header(
        self,
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig,
        started_at: datetime,
    ) -> str:
        return self._baseline.create_run_header(
            run_id=run_id,
            thread_id=thread_id,
            config=config,
            started_at=started_at,
        )

    def load_run_header(
        self,
        artifact_ids: Sequence[str],
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig | None = None,
    ) -> dict[str, object]:
        return self._baseline.load_run_header(
            artifact_ids,
            run_id=run_id,
            thread_id=thread_id,
            config=config,
        )

    def validate_durable_event_state(
        self,
        state: ResearchState,
        event: RunEvent,
    ) -> None:
        restored = validate_research_state(cast("Mapping[str, object]", state))
        baseline = cast(
            "BaselineState",
            {field: restored[field] for field in _BASELINE_FIELDS},
        )
        self._baseline.validate_durable_event_state(baseline, event)
        input_hash = _hash_json(_state_payload(baseline))
        matches: list[Any] = []
        for artifact_id in event.artifact_ids:
            receipt = _load_audit_receipt(self._baseline.artifact_store, artifact_id)
            if (
                receipt is not None
                and receipt.kind == "node-execution"
                and receipt.run_id == restored["run_id"]
                and receipt.thread_id == restored["thread_id"]
                and receipt.payload.get("input_state_sha256") == input_hash
            ):
                matches.append(receipt)
        if len(matches) != 1:
            raise ArtifactIntegrityError("research state receipt is missing or ambiguous")
        input_extension, _ = _decode_research_extension(
            matches[0].payload.get("research_state"),
            baseline=baseline,
        )
        if input_extension != _research_extension(restored):
            raise ArtifactIntegrityError("research state receipt conflicts with checkpoint")


def _claim_match_text(value: str) -> str:
    """Normalize punctuation/citations for citation-aware claim matching."""

    return " ".join(re.findall(r"\w+", re.sub(r"\[[^\]]+\]", " ", value.casefold())))


def _dedupe_claims(claims: Sequence[Claim]) -> tuple[Claim, ...]:
    by_id: dict[str, Claim] = {}
    for claim in claims:
        previous = by_id.get(claim.claim_id)
        if previous is not None and previous != claim:
            raise ArtifactIntegrityError("claim IDs resolve to conflicting claims")
        by_id[claim.claim_id] = claim
    return tuple(by_id[claim_id] for claim_id in sorted(by_id))


def _new_graph(
    *,
    claims: Sequence[Claim],
    evidence: Sequence[EvidenceSpan],
) -> ClaimEvidenceGraph:
    graph = ClaimEvidenceGraph()
    for claim in _dedupe_claims(claims):
        graph.add_claim(claim)
    seen_evidence: set[str] = set()
    for item in sorted(evidence, key=lambda value: value.evidence_id):
        if item.evidence_id in seen_evidence:
            continue
        graph.add_evidence(item)
        seen_evidence.add(item.evidence_id)
    return graph


def _graph_payload(graph: ClaimEvidenceGraph) -> bytes:
    validation = graph.validate()
    if not validation.valid:
        raise ArtifactIntegrityError(
            "claim-evidence graph is invalid: " + ",".join(validation.error_codes)
        )
    return _canonical_json(graph.to_json())


def _load_graph(artifact_store: LocalArtifactStore, artifact_id: str) -> ClaimEvidenceGraph:
    try:
        ref, raw = artifact_store._read(artifact_id)  # pyright: ignore[reportPrivateUsage]
        if ref.media_type != _CLAIM_GRAPH_MEDIA_TYPE:
            raise ArtifactIntegrityError("claim-evidence graph artifact has an invalid media type")
        value: object = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        if _canonical_json(value) != raw:
            raise ArtifactIntegrityError("claim-evidence graph artifact is not canonical")
    except ArtifactIntegrityError:
        raise
    except (FileNotFoundError, TypeError, ValueError):
        raise ArtifactIntegrityError("claim-evidence graph artifact is corrupt") from None
    return _load_graph_from_value(value)


def _load_graph_from_value(value: object) -> ClaimEvidenceGraph:
    if type(value) is not dict:
        raise ArtifactIntegrityError("claim-evidence graph artifact has an invalid shape")
    payload = cast("dict[str, object]", value)
    if set(payload) != {"claims", "evidence", "links"}:
        raise ArtifactIntegrityError("claim-evidence graph artifact has an invalid shape")
    claims = payload["claims"]
    evidence = payload["evidence"]
    links = payload["links"]
    if type(claims) is not list or type(evidence) is not list or type(links) is not list:
        raise ArtifactIntegrityError("claim-evidence graph members must be arrays")
    graph = ClaimEvidenceGraph()
    try:
        for raw_claim in cast("list[object]", claims):
            graph.add_claim(Claim.model_validate(raw_claim))
        for raw_evidence in cast("list[object]", evidence):
            graph.add_evidence(EvidenceSpan.model_validate(raw_evidence))
        # Importing the link type at module level would not change behavior,
        # but keeping it local makes the artifact parser's domain boundary
        # explicit and keeps the facade's public imports small.
        from deepresearch.domain import ClaimEvidenceLink

        for raw_link in cast("list[object]", links):
            graph.add_link(ClaimEvidenceLink.model_validate(raw_link))
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("claim-evidence graph artifact is corrupt") from None
    validation = graph.validate()
    if not validation.valid:
        raise ArtifactIntegrityError(
            "claim-evidence graph artifact is invalid: " + ",".join(validation.error_codes)
        )
    return graph


def _split_sentences(paragraph: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", paragraph) if item.strip())


def _remove_unsupported_claims(
    draft: str,
    unsupported: Sequence[Claim],
) -> tuple[str, tuple[ClaimResolutionRecord, ...]]:
    unsupported_by_id = {item.claim_id: _claim_match_text(item.text) for item in unsupported}
    paragraphs: list[str] = []
    removed_ids: set[str] = set()
    for paragraph in draft.strip().split("\n\n"):
        lines = paragraph.splitlines()
        heading_lines = [line for line in lines if line.lstrip().startswith("#")]
        body = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
        sentences = _split_sentences(body)
        kept: list[str] = []
        pending_citations: list[str] = []
        previous_removed = False
        for sentence in sentences:
            prose = re.sub(r"\[[^\]]+\]", " ", sentence).strip()
            if not prose:
                # A citation emitted after a period belongs to the preceding
                # sentence.  Retain it unless that sentence was the one we
                # just removed; otherwise resolution would delete support for
                # an unrelated claim.
                if previous_removed:
                    continue
                if kept:
                    kept[-1] = f"{kept[-1]} {sentence}".strip()
                else:
                    pending_citations.append(sentence)
                continue
            normalised = _claim_match_text(prose)
            matched_ids = {
                claim_id
                for claim_id, claim_text in unsupported_by_id.items()
                if claim_text and claim_text in normalised
            }
            if matched_ids:
                removed_ids.update(matched_ids)
                previous_removed = True
                pending_citations.clear()
                continue
            if pending_citations:
                sentence = " ".join((*pending_citations, sentence))
                pending_citations.clear()
            kept.append(sentence.strip())
            previous_removed = False
        if pending_citations:
            if kept:
                kept[-1] = f"{kept[-1]} {' '.join(pending_citations)}".strip()
            else:
                kept.extend(pending_citations)
        remaining = "\n".join((*heading_lines, *kept)).strip()
        if remaining:
            paragraphs.append(remaining)
    missing_ids = set(unsupported_by_id) - removed_ids
    if missing_ids:
        raise ArtifactIntegrityError("unsupported claim is absent from the draft")
    records = tuple(
        ClaimResolutionRecord(
            claim_id=item.claim_id,
            action="DELETE",
            reason_code="UNSUPPORTED_FACT",
            replacement_text=None,
        )
        for item in sorted(unsupported, key=lambda value: value.claim_id)
        if item.claim_id in removed_ids
    )
    if removed_ids and not paragraphs:
        # A heading is intentionally used instead of an uncited prose note:
        # MarkdownReportWriter correctly rejects visible claims without
        # evidence citations.
        paragraphs.append("## Limitations")
    return "\n\n".join(paragraphs).strip(), records


class ResearchNodeHandlers:
    """Research-v1 facade over one Core baseline handler composition.

    The first production composition is deliberately deterministic. Shared
    retrieval/report nodes remain the exact baseline handlers; only claim
    extraction, evidence judging and conservative unsupported-claim deletion
    are added here. The node wrapper delegates durable event/audit handling to
    Core's existing ``_safe_node`` envelope.
    """

    def __init__(
        self,
        baseline_handlers: BaselineNodeHandlers,
        *,
        claim_extractor: ClaimExtractor | None = None,
        evidence_judge: EvidenceJudge | None = None,
    ) -> None:
        self.baseline = baseline_handlers
        self._plan_node = cast("InitialPlanNode", baseline_handlers.plan)
        self.artifact_store: LocalArtifactStore = baseline_handlers.artifact_store
        self.evidence_store: LocalEvidenceStore = baseline_handlers.evidence_store
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.evidence_judge = evidence_judge or EvidenceJudge()
        if (
            self.claim_extractor.model_provider is not None
            or self.evidence_judge.model_provider is not None
        ):
            raise ValueError("production research handlers require deterministic claim stages")
        baseline_audit = cast(
            "_AuditComposition | None",
            getattr(baseline_handlers, "_audit_composition", None),
        )
        self._audit_composition = (
            _ResearchAuditComposition(baseline_audit) if baseline_audit is not None else None
        )
        self._persist_baseline = copy(baseline_handlers)
        self._persist_baseline._audit_composition = self._audit_composition  # type: ignore[attr-defined]

    @staticmethod
    def _runtime() -> BaselineRuntimeContext:
        context = cast("BaselineRuntimeContext | None", get_runtime(BaselineRuntimeContext).context)
        if context is None:
            raise WorkflowInvariantError(code="INVALID_RUNTIME_CONTEXT")
        return context

    @staticmethod
    def _baseline_view(state: ResearchState) -> BaselineState:
        """Strip research-only channels before calling a Core baseline node.

        Core's baseline state validator intentionally requires an exact field
        set.  LangGraph keeps the research channels in the shared state, so a
        direct cast is not sufficient once the first research decision has
        been recorded.  Keeping this projection at the facade boundary makes
        every delegated baseline call obey the original Core contract while
        research handlers still receive the full state below.
        """

        restored = validate_research_state(cast("Mapping[str, object]", state))
        return cast(
            "BaselineState",
            {field: restored[field] for field in _BASELINE_FIELDS},
        )

    def node_wrapper(self, node: str, handler: NodeHandler) -> NodeHandler:
        """Wrap a research node in Core's baseline audit/event envelope.

        ``_safe_node`` deliberately validates only the baseline state shape.
        This adapter supplies that shadow state and merges research-only fields
        after the envelope returns, so no second persistence or event protocol
        is introduced.
        """

        async def invoke(state: ResearchState) -> StateUpdate:
            restored = validate_research_state(cast("Mapping[str, object]", state))
            shadow = cast(
                "BaselineState",
                {field: restored[field] for field in _BASELINE_FIELDS},
            )
            captured: dict[str, object] = {}

            async def baseline_handler(base_state: BaselineState) -> StateUpdate:
                full_state = cast("ResearchState", {**restored, **base_state})
                raw = dict(await handler(full_state))
                baseline_update = {
                    key: value for key, value in raw.items() if key in _BASELINE_FIELDS
                }
                captured.update(
                    {key: value for key, value in raw.items() if key not in _BASELINE_FIELDS}
                )
                return baseline_update

            def receipt_state_builder(
                merged: BaselineState,
                _update: Mapping[str, object],
            ) -> Mapping[str, object]:
                output = dict(restored)
                output.update(merged)
                if merged["error_code"] is None:
                    output.update(captured)
                output_research = validate_research_state(output)
                return {
                    "input": _research_extension(restored),
                    "output": _research_extension(output_research),
                }

            def receipt_state_recovery(
                recovered: BaselineState,
                payload: object,
            ) -> Mapping[str, object]:
                input_extension, output_extension = _decode_research_extension(
                    payload,
                    baseline=recovered,
                )
                if input_extension != _research_extension(restored):
                    raise ArtifactIntegrityError("research state receipt conflicts with checkpoint")
                validate_research_state({**restored, **recovered, **output_extension})
                return output_extension

            safe = _safe_node(  # pyright: ignore[reportPrivateUsage]
                node,
                cast("BaselineNode", baseline_handler),
                audit_composition=cast("Any", self._audit_composition),
                receipt_state_builder=cast("ReceiptStateBuilder", receipt_state_builder),
                receipt_state_recovery=cast("ReceiptStateRecovery", receipt_state_recovery),
            )
            update = dict(await safe(shadow))
            if update.get("error_code") is None:
                update.update(captured)
            return update

        return invoke

    def _record_result(self, context: BaselineRuntimeContext, artifact_id: str) -> None:
        self.baseline._audit_result_artifact(  # pyright: ignore[reportPrivateUsage]
            context,
            artifact_id,
        )

    async def validate_request(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.validate_request(self._baseline_view(state))

    async def decide_next(self, state: ResearchState) -> StateUpdate:
        update = dict(await self.baseline.decide_next(self._baseline_view(state)))
        # The baseline graph derives its edge from stop_reason directly.  The
        # research graph keeps that decision as an explicit, validated field so
        # the checkpoint/replay route is observable and deterministic.
        update["decision_route"] = "STOP" if update.get("stop_reason") is not None else "SEARCH"
        return update

    async def search(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.search(self._baseline_view(state))

    async def fetch(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.fetch(self._baseline_view(state))

    async def parse_and_normalize(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.parse_and_normalize(self._baseline_view(state))

    async def store_evidence(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.store_evidence(self._baseline_view(state))

    async def rank_evidence(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.rank_evidence(self._baseline_view(state))

    async def draft_report(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.draft_report(self._baseline_view(state))

    async def extract_claims(self, state: ResearchState) -> StateUpdate:
        context = self._runtime()
        draft_id = state["draft_artifact_id"]
        if draft_id is None:
            raise WorkflowInvariantError(code="REPORT_MISSING")
        try:
            draft = self.artifact_store.get_bytes(draft_id).decode("utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            raise ArtifactIntegrityError("draft artifact is corrupt") from None
        claims = await self.claim_extractor.extract(
            draft,
            evidence_ids=state["selected_evidence_ids"],
            deadline=_effective_deadline(context),  # pyright: ignore[reportPrivateUsage]
            cancellation_token=context.cancellation_token,
        )
        evidence = tuple(
            self.evidence_store.get_evidence(item) for item in state["selected_evidence_ids"]
        )
        graph = _new_graph(claims=claims, evidence=evidence)
        ref = self.artifact_store.put_bytes(
            _graph_payload(graph),
            media_type=_CLAIM_GRAPH_MEDIA_TYPE,
        )
        self._record_result(context, ref.artifact_id)
        return {
            "rank_artifact_id": ref.artifact_id,
            "claim_ids": tuple(claim.claim_id for claim in _dedupe_claims(claims)),
            "unsupported_claim_ids": (),
        }

    async def verify_claims(self, state: ResearchState) -> StateUpdate:
        context = self._runtime()
        rank_artifact_id = state.get("rank_artifact_id")
        if rank_artifact_id is None:
            raise WorkflowInvariantError(code="CLAIMS_MISSING")
        graph = _load_graph(self.artifact_store, rank_artifact_id)
        claims_payload = json.loads(_graph_payload(graph))
        # Rebuild from the canonical artifact so this node never judges an
        # in-memory object that was not content-addressed by ExtractClaims.
        graph = _load_graph_from_value(claims_payload)
        evidence = tuple(
            self.evidence_store.get_evidence(item) for item in state["selected_evidence_ids"]
        )
        claims = _claims_from_graph(graph)
        for claim in claims:
            links = await self.evidence_judge.judge(
                claim,
                evidence,
                deadline=_effective_deadline(context),  # pyright: ignore[reportPrivateUsage]
                cancellation_token=context.cancellation_token,
            )
            for link in links:
                graph.add_link(link)
        linked_claim_ids = {
            link.claim_id
            for claim in claims
            for link in graph.links_for_claim(claim.claim_id)
            if link.relation == "support"
        }
        updated = _new_graph(
            claims=tuple(
                claim.model_copy(
                    update={
                        "verification_status": (
                            "supported" if claim.claim_id in linked_claim_ids else "unsupported"
                        )
                    }
                )
                for claim in claims
            ),
            evidence=evidence,
        )
        for link in cast("list[object]", graph.to_json()["links"]):
            from deepresearch.domain import ClaimEvidenceLink

            updated.add_link(ClaimEvidenceLink.model_validate(link))
        ref = self.artifact_store.put_bytes(
            _graph_payload(updated),
            media_type=_CLAIM_GRAPH_MEDIA_TYPE,
        )
        self._record_result(context, ref.artifact_id)
        unsupported = tuple(
            sorted(claim.claim_id for claim in claims if claim.claim_id not in linked_claim_ids)
        )
        return {
            "rank_artifact_id": ref.artifact_id,
            "claim_ids": tuple(claim.claim_id for claim in claims),
            "unsupported_claim_ids": unsupported,
            "verification_route": "RESOLVE_UNSUPPORTED" if unsupported else "FINALIZE",
        }

    async def targeted_research(self, state: ResearchState) -> StateUpdate:
        # The deterministic showcase does not invent new queries.  If a
        # caller routes here explicitly, record the bounded round and let the
        # normal search loop decide whether existing evidence is sufficient.
        rounds = state.get("directional_research_rounds", 0)
        if rounds >= 1:
            return {"verification_route": "RESOLVE_UNSUPPORTED"}
        return {"directional_research_rounds": 1}

    async def resolve_unsupported_claims(self, state: ResearchState) -> StateUpdate:
        context = self._runtime()
        rank_artifact_id = state.get("rank_artifact_id")
        draft_id = state["draft_artifact_id"]
        if rank_artifact_id is None or draft_id is None:
            raise WorkflowInvariantError(code="CLAIMS_MISSING")
        graph = _load_graph(self.artifact_store, rank_artifact_id)
        claims = _claims_from_graph(graph)
        unsupported_ids = set(state.get("unsupported_claim_ids", ()))
        known_ids = {claim.claim_id for claim in claims}
        if not unsupported_ids.issubset(known_ids):
            raise ArtifactIntegrityError("unsupported claim ID is absent from the claim graph")
        unsupported = tuple(claim for claim in claims if claim.claim_id in unsupported_ids)
        try:
            draft = self.artifact_store.get_bytes(draft_id).decode("utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            raise ArtifactIntegrityError("draft artifact is corrupt") from None
        resolved_draft, records = _remove_unsupported_claims(draft, unsupported)
        draft_ref = self.artifact_store.put_bytes(
            resolved_draft.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
        )
        resolution_payload = {
            "actions": {record.claim_id: record.action for record in records},
            "records": [
                {
                    "action": record.action,
                    "claim_id": record.claim_id,
                    "reason_code": record.reason_code,
                    "replacement_text": record.replacement_text,
                }
                for record in records
            ],
        }
        resolution_ref = self.artifact_store.put_bytes(
            _canonical_json(resolution_payload),
            media_type=_CLAIM_RESOLUTION_MEDIA_TYPE,
        )
        self._record_result(context, draft_ref.artifact_id)
        self._record_result(context, resolution_ref.artifact_id)
        return {
            "draft_artifact_id": draft_ref.artifact_id,
            "claim_resolution_artifact_id": resolution_ref.artifact_id,
            "unsupported_claim_ids": (),
            "verification_route": "FINALIZE",
        }

    async def finalize_citations(self, state: ResearchState) -> StateUpdate:
        return await self.baseline.finalize_citations(self._baseline_view(state))

    async def persist_results(self, state: ResearchState) -> StateUpdate:
        return await self._persist_baseline.persist_results(self._baseline_view(state))

    def as_dependencies(self, checkpointer: Any) -> ResearchGraphDependencies:
        return ResearchGraphDependencies(
            validate_request=self.validate_request,
            initial_plan_generator=self.baseline.initial_plan_generator,
            plan=self._plan_node,
            decide_next=self.decide_next,
            search=self.search,
            fetch=self.fetch,
            parse_and_normalize=self.parse_and_normalize,
            store_evidence=self.store_evidence,
            rank_evidence=self.rank_evidence,
            draft_report=self.draft_report,
            extract_claims=self.extract_claims,
            verify_claims=self.verify_claims,
            targeted_research=self.targeted_research,
            resolve_unsupported_claims=self.resolve_unsupported_claims,
            finalize_citations=self.finalize_citations,
            persist_results=self.persist_results,
            checkpointer=checkpointer,
            node_wrapper=self.node_wrapper,
            audit_composition=self._audit_composition,
        )


def _claims_from_graph(graph: ClaimEvidenceGraph) -> tuple[Claim, ...]:
    payload = graph.to_json()["claims"]
    return tuple(Claim.model_validate(item) for item in cast("list[object]", payload))


__all__ = ["ResearchNodeHandlers"]
