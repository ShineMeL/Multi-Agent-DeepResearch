from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol, cast

from deepresearch.domain import EvidenceSpan, SourceDocument, StopReason
from deepresearch.retrieval import canonicalize_url, normalize_text
from deepresearch.storage import LocalEvidenceStore

from .boundary import ContentBoundary, identity_content_boundary

_CITATION = re.compile(r"(?<!\\)\[(E-[A-Za-z0-9_-]+)\]")
_CITATION_START = re.compile(r"(?<!\\)\[E")
_FENCE_OPEN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_REFERENCE_DEFINITION = re.compile(r"^[ \t]{0,3}\[[^]\n]+\]:")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]+|$)")
_INLINE_LINK = re.compile(r"!?\[[^]\n]*\]\([^\n)]*\)")
_REFERENCE_LINK = re.compile(r"!?\[[^]\n]*\]\[[^]\n]*\]")
_HTML_CODE = re.compile(r"<(?:code|pre)\b[^>]*>.*?</(?:code|pre)\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>\n]+>")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{\[\]}<>#!|~])")
_STOP_REASONS = frozenset({"SUFFICIENT", "PLATEAU", "BUDGET_EXHAUSTED", "BLOCKED"})


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


def _blank(chars: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        if chars[index] not in {"\n", "\r"}:
            chars[index] = " "


def _blank_pattern(chars: list[str], pattern: re.Pattern[str]) -> None:
    text = "".join(chars)
    for match in pattern.finditer(text):
        _blank(chars, match.start(), match.end())


def _blank_inline_code(chars: list[str]) -> None:
    text = "".join(chars)
    cursor = 0
    while cursor < len(text):
        opening = text.find("`", cursor)
        if opening < 0:
            return
        opening_end = opening
        while opening_end < len(text) and text[opening_end] == "`":
            opening_end += 1
        marker = text[opening:opening_end]
        search_from = opening_end
        closing = -1
        while True:
            candidate = text.find(marker, search_from)
            if candidate < 0:
                break
            before_is_tick = candidate > 0 and text[candidate - 1] == "`"
            after = candidate + len(marker)
            after_is_tick = after < len(text) and text[after] == "`"
            if not before_is_tick and not after_is_tick:
                closing = candidate
                break
            search_from = candidate + 1
        if closing < 0:
            cursor = opening_end
            continue
        end = closing + len(marker)
        _blank(chars, opening, end)
        cursor = end


def _rendered_claim_prose(markdown: str) -> str:
    chars = list(markdown)
    cursor = 0
    while True:
        start = markdown.find("<!--", cursor)
        if start < 0:
            break
        closing = markdown.find("-->", start + 4)
        end = len(markdown) if closing < 0 else closing + 3
        _blank(chars, start, end)
        cursor = end

    _blank_inline_code(chars)

    offset = 0
    fence: tuple[str, int] | None = None
    for line in "".join(chars).splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if fence is None:
            opening = _FENCE_OPEN.match(content)
            if opening is not None:
                marker = opening.group(1)
                fence = (marker[0], len(marker))
                _blank(chars, offset, offset + len(line))
            elif (
                content.startswith(("    ", "\t"))
                or _REFERENCE_DEFINITION.match(content)
                or _HEADING.match(content)
            ):
                _blank(chars, offset, offset + len(line))
        else:
            marker, minimum = fence
            stripped = content.lstrip(" \t")
            marker_length = len(stripped) - len(stripped.lstrip(marker))
            _blank(chars, offset, offset + len(line))
            if marker_length >= minimum and not stripped[marker_length:].strip():
                fence = None
        offset += len(line)

    _blank_pattern(chars, _INLINE_LINK)
    _blank_pattern(chars, _REFERENCE_LINK)
    _blank_pattern(chars, _HTML_CODE)
    _blank_pattern(chars, _HTML_TAG)
    return "".join(chars)


def _citation_ids(report_markdown: str) -> tuple[str, ...]:
    prose = _rendered_claim_prose(report_markdown)
    matches = tuple(_CITATION.finditer(prose))
    valid_starts = {match.start() for match in matches}
    if any(match.start() not in valid_starts for match in _CITATION_START.finditer(prose)):
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


def _markdown_safe_single_line(text: str) -> str:
    normalized = normalize_text(text)
    without_controls = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    single_line = " ".join(without_controls.split())
    if not single_line:
        return "(empty)"
    return _MARKDOWN_SPECIAL.sub(r"\\\1", single_line)


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
        prose = _rendered_claim_prose(report_markdown)
        for paragraph in re.split(r"\n\s*\n", prose):
            claim = paragraph.strip()
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
        stop_reason: StopReason | None = None,
        uncovered_information_needs: Sequence[str] = (),
    ) -> str:
        draft = draft_markdown.strip()
        citations = self.validate_citations(
            draft,
            selected_evidence_ids=selected_evidence_ids,
        )
        self._validate_claim_support(draft)
        if stop_reason is not None and (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                stop_reason, str
            )
            or stop_reason not in _STOP_REASONS
        ):
            raise ValueError("stop reason must be a public StopReason literal")
        if is_partial and stop_reason is None:
            raise ValueError("partial reports require a stop reason")
        if not is_partial and (stop_reason is not None or uncovered_information_needs):
            raise ValueError("stop reason and uncovered needs require a partial report")

        sections = [draft]
        if is_partial:
            needs = sorted(
                {_markdown_safe_single_line(need) for need in uncovered_information_needs}
            )
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
            safe_title = _markdown_safe_single_line(source.title)
            references.append(f"- [{evidence_id}] {safe_title} — {safe_url}")
        sections.append("\n".join(references))
        return "\n\n".join(sections)


__all__ = [
    "EvidenceBackedClaimRequired",
    "MalformedEvidenceCitation",
    "MarkdownReportWriter",
    "UnknownEvidenceCitation",
    "validate_citations",
]
