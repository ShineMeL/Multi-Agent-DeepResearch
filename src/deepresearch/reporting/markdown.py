from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol, cast

from markdown_it import MarkdownIt
from markdown_it.token import Token

from deepresearch.domain import EvidenceSpan, SourceDocument, StopReason
from deepresearch.retrieval import canonicalize_url, normalize_text
from deepresearch.storage import LocalEvidenceStore

from .boundary import ContentBoundary, identity_content_boundary

_CITATION = re.compile(r"\[(E-[A-Za-z0-9_-]+)\]")
_CITATION_START = re.compile(r"\[E")
_AMBIGUOUS_HTML_CITATION = re.compile(
    r"<\s*/?\s*[A-Za-z][^>\r\n]*\[E",
    re.IGNORECASE,
)
_MARKDOWN_SPECIAL = re.compile(r"([&\\`*_{\[\]}<>#!|~])")
_STOP_REASONS = frozenset({"SUFFICIENT", "PLATEAU", "BUDGET_EXHAUSTED", "BLOCKED"})
_HIDDEN_HTML_TAGS = frozenset({"code", "pre", "script", "style"})
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_MARKDOWN = MarkdownIt("commonmark")


class UnknownEvidenceCitation(ValueError):
    pass


class MalformedEvidenceCitation(ValueError):
    pass


class EvidenceBackedClaimRequired(ValueError):
    pass


class _PromptRecorder(Protocol):
    last_prompt: str


@dataclass(frozen=True)
class _RenderedMarkdown:
    visible_units: tuple[str, ...]
    claim_units: tuple[tuple[str, str], ...]


class _HtmlVisibilityParser(HTMLParser):
    def __init__(self, *, collect_text: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.collect_text = collect_text
        self.hidden_tags: list[str] = []
        self.visible_parts: list[str] = []
        self.ambiguous = False

    @property
    def hides_text(self) -> bool:
        return bool(self.hidden_tags)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in _HIDDEN_HTML_TAGS:
            self.hidden_tags.append(normalized)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del tag, attrs

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized not in _HIDDEN_HTML_TAGS:
            return
        if not self.hidden_tags or self.hidden_tags[-1] != normalized:
            self.ambiguous = True
            return
        self.hidden_tags.pop()

    def handle_data(self, data: str) -> None:
        if self.collect_text and not self.hides_text:
            self.visible_parts.append(data)

    def handle_comment(self, data: str) -> None:
        del data


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _feed_html(parser: _HtmlVisibilityParser, content: str) -> None:
    try:
        parser.feed(content)
    except MemoryError:
        raise
    except (
        AssertionError,
        IndexError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise MalformedEvidenceCitation(
            "report contains an ambiguous evidence citation"
        ) from None


def _inline_views(children: Sequence[Token]) -> tuple[str, str]:
    visible: list[str] = []
    eligible: list[str] = []
    link_depth = 0
    html = _HtmlVisibilityParser(collect_text=False)
    raw_html_participated = False
    for child in children:
        if child.type == "html_inline":
            raw_html_participated = True
            _feed_html(html, child.content)
            eligible.append(" ")
            continue
        if child.type == "link_open":
            link_depth += 1
            eligible.append(" ")
            continue
        if child.type == "link_close":
            if link_depth == 0:
                html.ambiguous = True
            else:
                link_depth -= 1
            eligible.append(" ")
            continue
        if child.type == "image":
            eligible.append(" ")
            continue
        if child.type in {"softbreak", "hardbreak"}:
            if not html.hides_text:
                visible.append("\n")
                eligible.append("\n" if link_depth == 0 else " ")
            continue
        if child.type == "text":
            if _AMBIGUOUS_HTML_CITATION.search(child.content):
                html.ambiguous = True
            if not html.hides_text:
                visible.append(child.content)
                eligible.append(child.content if link_depth == 0 else " ")
            continue
        if child.type == "code_inline":
            eligible.append(" ")
            continue
        if "[E" in child.content:
            html.ambiguous = True
        eligible.append(" ")
    if link_depth or html.hides_text or html.ambiguous:
        raise MalformedEvidenceCitation(
            "report contains an ambiguous evidence citation"
        )
    return "".join(visible), ("" if raw_html_participated else "".join(eligible))


def _html_block_visible(content: str) -> str:
    parser = _HtmlVisibilityParser(collect_text=True)
    _feed_html(parser, content)
    try:
        parser.close()
    except MemoryError:
        raise
    except (
        AssertionError,
        IndexError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise MalformedEvidenceCitation(
            "report contains an ambiguous evidence citation"
        ) from None
    if parser.hides_text or parser.ambiguous:
        raise MalformedEvidenceCitation(
            "report contains an ambiguous evidence citation"
        )
    return "".join(parser.visible_parts)


def _rendered_markdown(report_markdown: str) -> _RenderedMarkdown:
    try:
        tokens = _MARKDOWN.parse(report_markdown)
    except MemoryError:
        raise
    except (
        AssertionError,
        IndexError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise MalformedEvidenceCitation(
            "report contains an ambiguous evidence citation"
        ) from None

    visible_units: list[str] = []
    claim_units: list[tuple[str, str]] = []
    paragraph_depth = 0
    for token in tokens:
        if token.type == "paragraph_open":
            paragraph_depth += 1
            continue
        if token.type == "paragraph_close":
            paragraph_depth = max(0, paragraph_depth - 1)
            continue
        if token.type == "inline":
            visible, eligible = _inline_views(token.children or ())
            if visible:
                visible_units.append(visible)
                if paragraph_depth:
                    claim_units.append((visible, eligible))
            continue
        if token.type == "html_block":
            visible = _html_block_visible(token.content)
            if visible:
                visible_units.append(visible)
                claim_units.append((visible, ""))
    return _RenderedMarkdown(
        visible_units=tuple(visible_units),
        claim_units=tuple(claim_units),
    )


def _citation_ids_from_rendered(rendered: _RenderedMarkdown) -> tuple[str, ...]:
    citation_ids: list[str] = []
    for visible in rendered.visible_units:
        matches = tuple(_CITATION.finditer(visible))
        valid_starts = {match.start() for match in matches}
        if any(
            match.start() not in valid_starts
            for match in _CITATION_START.finditer(visible)
        ):
            raise MalformedEvidenceCitation(
                "report contains a malformed evidence citation"
            )
        citation_ids.extend(match.group(1) for match in matches)
    return _dedupe(citation_ids)


def _validate_visible_citations(
    rendered: _RenderedMarkdown,
    evidence_store: LocalEvidenceStore,
    *,
    selected_evidence_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    citation_ids = _citation_ids_from_rendered(rendered)
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


def validate_citations(
    report_markdown: str,
    evidence_store: LocalEvidenceStore,
    *,
    selected_evidence_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    rendered = _rendered_markdown(report_markdown)
    return _validate_visible_citations(
        rendered,
        evidence_store,
        selected_evidence_ids=selected_evidence_ids,
    )


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
        " "
        if unicodedata.category(character) == "Cc" or character in _BIDI_CONTROLS
        else character
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

    def _validate_claim_support(self, rendered: _RenderedMarkdown) -> None:
        for visible, eligible in rendered.claim_units:
            if visible.strip() and _CITATION.search(eligible) is None:
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
        rendered = _rendered_markdown(draft)
        citations = _validate_visible_citations(
            rendered,
            self.evidence_store,
            selected_evidence_ids=selected_evidence_ids,
        )
        self._validate_claim_support(rendered)
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
