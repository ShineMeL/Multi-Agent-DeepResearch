from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol, cast

from deepresearch.domain import EvidenceSpan, SourceDocument
from deepresearch.retrieval import canonicalize_url
from deepresearch.storage import LocalEvidenceStore

from .boundary import ContentBoundary, identity_content_boundary

_CITATION = re.compile(r"\[(E-[A-Za-z0-9_-]+)\]")
_CITATION_START = re.compile(r"\[E")


class UnknownEvidenceCitation(ValueError):
    pass


class MalformedEvidenceCitation(ValueError):
    pass


class EvidenceBackedClaimRequired(ValueError):
    pass


class _PromptRecorder(Protocol):
    last_prompt: str


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _citation_ids(report_markdown: str) -> tuple[str, ...]:
    matches = tuple(_CITATION.finditer(report_markdown))
    valid_starts = {match.start() for match in matches}
    if any(match.start() not in valid_starts for match in _CITATION_START.finditer(report_markdown)):
        raise MalformedEvidenceCitation("report contains a malformed evidence citation")
    return _dedupe(tuple(match.group(1) for match in matches))


def validate_citations(
    report_markdown: str,
    evidence_store: LocalEvidenceStore,
    *,
    selected_evidence_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    citation_ids = _citation_ids(report_markdown)
    selected = None if selected_evidence_ids is None else set(selected_evidence_ids)
    missing = [
        evidence_id
        for evidence_id in citation_ids
        if not evidence_store.has_evidence(evidence_id)
        or (selected is not None and evidence_id not in selected)
    ]
    if missing:
        raise UnknownEvidenceCitation(", ".join(missing))
    return citation_ids


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class MarkdownReportWriter:
    def __init__(
        self,
        evidence_store: LocalEvidenceStore,
        *,
        model: object | None = None,
        content_boundary: ContentBoundary = identity_content_boundary,
    ) -> None:
        self.evidence_store = evidence_store
        self.model = model
        self.content_boundary = content_boundary

    def _bound(self, text: str) -> str:
        result = self.content_boundary(text)
        if not isinstance(result, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("content boundary must return text")
        return result

    def _material(
        self, evidence_id: str
    ) -> tuple[EvidenceSpan, SourceDocument, dict[str, object]]:
        if not self.evidence_store.has_evidence(evidence_id):
            raise UnknownEvidenceCitation(evidence_id)
        evidence = self.evidence_store.get_evidence(evidence_id)
        source = self.evidence_store.get_source(evidence.source_id)
        safe_url = canonicalize_url(str(source.canonical_url))
        locator = _canonical_json(evidence.locator.model_dump(mode="json"))
        material: dict[str, object] = {
            "evidence_id": evidence.evidence_id,
            "excerpt": self._bound(evidence.excerpt),
            "language": self._bound(evidence.language),
            "locator": self._bound(locator),
            "source": {
                "authors": [self._bound(author) for author in source.authors],
                "canonical_url": self._bound(safe_url),
                "parser_version": self._bound(source.parser_version),
                "published_at": (
                    None
                    if source.published_at is None
                    else self._bound(source.published_at.isoformat())
                ),
                "retrieved_at": self._bound(source.retrieved_at.isoformat()),
                "source_family_id": self._bound(source.source_family_id),
                "source_id": source.source_id,
                "source_type": self._bound(str(source.source_type)),
                "title": self._bound(source.title),
            },
        }
        return evidence, source, material

    def validate_citations(
        self,
        report_markdown: str,
        *,
        selected_evidence_ids: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        return validate_citations(
            report_markdown,
            self.evidence_store,
            selected_evidence_ids=selected_evidence_ids,
        )

    def render_prompt(
        self,
        *,
        selected_evidence_ids: Sequence[str],
        user_strings: Sequence[str] = (),
    ) -> str:
        selected = tuple(sorted(set(selected_evidence_ids)))
        materials = [self._material(evidence_id)[2] for evidence_id in selected]
        prompt = _canonical_json(
            {
                "evidence": materials,
                "instruction": (
                    "Write Markdown using only the selected evidence. Cite factual claims inline "
                    "with exact [E-...] IDs."
                ),
                "user_strings": [self._bound(value) for value in user_strings],
            }
        )
        if self.model is not None and hasattr(self.model, "last_prompt"):
            recorder = cast("_PromptRecorder", self.model)
            recorder.last_prompt = prompt
        return prompt

    def _validate_claim_support(self, report_markdown: str) -> None:
        for paragraph in re.split(r"\n\s*\n", report_markdown):
            claim = "\n".join(
                line for line in paragraph.splitlines() if not line.lstrip().startswith("#")
            ).strip()
            if claim and _CITATION.search(claim) is None:
                raise EvidenceBackedClaimRequired(
                    "every report claim paragraph requires an inline evidence citation"
                )

    def finalize_report(
        self,
        draft_markdown: str,
        *,
        selected_evidence_ids: Sequence[str],
        is_partial: bool = False,
        stop_reason: str | None = None,
        uncovered_information_needs: Sequence[str] = (),
    ) -> str:
        draft = draft_markdown.strip()
        citations = self.validate_citations(
            draft,
            selected_evidence_ids=selected_evidence_ids,
        )
        self._validate_claim_support(draft)
        if is_partial and not stop_reason:
            raise ValueError("partial reports require a stop reason")
        if not is_partial and (stop_reason is not None or uncovered_information_needs):
            raise ValueError("stop reason and uncovered needs require a partial report")

        sections = [draft]
        if is_partial:
            needs = sorted(set(uncovered_information_needs))
            partial = ["## Partial report", "", f"Stop reason: `{stop_reason}`"]
            if needs:
                partial.extend(("", "Uncovered information needs:", ""))
                partial.extend(f"- {need}" for need in needs)
            sections.append("\n".join(partial))

        references = ["## References", ""]
        for evidence_id in sorted(citations):
            evidence = self.evidence_store.get_evidence(evidence_id)
            source = self.evidence_store.get_source(evidence.source_id)
            safe_url = canonicalize_url(str(source.canonical_url))
            references.append(f"- [{evidence_id}] {source.title} — {safe_url}")
        sections.append("\n".join(references))
        return "\n\n".join(sections)


__all__ = [
    "EvidenceBackedClaimRequired",
    "MalformedEvidenceCitation",
    "MarkdownReportWriter",
    "UnknownEvidenceCitation",
    "validate_citations",
]
