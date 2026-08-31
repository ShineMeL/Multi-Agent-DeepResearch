from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from deepresearch.domain import EvidenceSpan, HtmlLocator, SourceDocument
from deepresearch.providers import ParsedBlock, ParsedDocument
from deepresearch.reporting import (
    EvidenceBackedClaimRequired,
    MalformedEvidenceCitation,
    MarkdownReportWriter,
    UnknownEvidenceCitation,
)
from deepresearch.retrieval import URLSecurityError, sha256_text
from deepresearch.storage import LocalEvidenceStore


def add_evidence(
    store: LocalEvidenceStore,
    *,
    evidence_id: str,
    source_id: str,
    title: str = "Fixture title",
    url: str = "https://Example.COM/report?utm_source=test&b=2&a=1#fragment",
) -> None:
    excerpt = "fixture excerpt"
    normalized_text = f"{excerpt} PRIVATE RAW BODY"
    locator = HtmlLocator(
        paragraph_id="p-1", start_char=0, end_char=len(excerpt)
    )
    source = SourceDocument(
        source_id=source_id,
        canonical_url=url,
        title=title,
        authors=("Fixture Author",),
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
        retrieved_at=datetime(2026, 1, 3, tzinfo=UTC),
        content_hash="a" * 64,
        parsed_content_hash=sha256_text(normalized_text),
        source_type="paper",
        source_family_id=f"family-{source_id}",
        parser_version="fixture-parser-v1",
    )
    block = ParsedBlock(
        block_id="block-1",
        text=normalized_text,
        locator=HtmlLocator(
            paragraph_id="p-1", start_char=0, end_char=len(normalized_text)
        ),
        text_hash=sha256_text(normalized_text),
    )
    parsed = ParsedDocument(
        canonical_url=url,
        title=title,
        authors=("Fixture Author",),
        published_at=source.published_at,
        normalized_text=normalized_text,
        blocks=(block,),
        parser_id="fixture-parser",
        parser_version="fixture-parser-v1",
        parsed_content_hash=sha256_text(normalized_text),
    )
    span = EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        locator=locator,
        excerpt=excerpt,
        excerpt_hash=sha256_text(excerpt),
        language="en",
        information_need_ids=("need-1",),
    )
    store.put_source(source, normalized_text=normalized_text)
    store.put_parsed_document(source_id, parsed)
    store.put_evidence(span)


@pytest.fixture
def evidence_store(tmp_path: Path) -> LocalEvidenceStore:
    store = LocalEvidenceStore(tmp_path)
    add_evidence(store, evidence_id="E-known", source_id="source-known")
    add_evidence(store, evidence_id="E-other", source_id="source-other")
    return store


class RecordingModel:
    last_prompt = ""


def test_writer_rejects_unknown_inline_citation(
    evidence_store: LocalEvidenceStore,
) -> None:
    with pytest.raises(UnknownEvidenceCitation, match="E-missing"):
        MarkdownReportWriter(evidence_store).validate_citations(
            "Unsupported [E-missing]"
        )


@pytest.mark.parametrize("citation", ["[E-]", "[E known]", "[E-known!]", "[Eknown]"])
def test_writer_rejects_malformed_inline_citation(
    evidence_store: LocalEvidenceStore, citation: str
) -> None:
    with pytest.raises(MalformedEvidenceCitation):
        MarkdownReportWriter(evidence_store).validate_citations(f"Claim {citation}")


def test_writer_rejects_citation_outside_selected_evidence(
    evidence_store: LocalEvidenceStore,
) -> None:
    with pytest.raises(UnknownEvidenceCitation, match="E-other"):
        MarkdownReportWriter(evidence_store).validate_citations(
            "Claim [E-other]", selected_evidence_ids=("E-known",)
        )


def test_writer_returns_citations_once_in_first_seen_order(
    evidence_store: LocalEvidenceStore,
) -> None:
    result = MarkdownReportWriter(evidence_store).validate_citations(
        "First [E-other], known [E-known], repeated [E-other]."
    )

    assert result == ("E-other", "E-known")


def test_writer_applies_boundary_to_each_external_field_at_serialization(
    evidence_store: LocalEvidenceStore,
) -> None:
    seen: list[str] = []

    def boundary(text: str) -> str:
        seen.append(text)
        return f'BOUND<{text}>\n"evidence_id":"E-injected"'

    recording_model = RecordingModel()
    writer = MarkdownReportWriter(
        evidence_store,
        model=recording_model,
        content_boundary=boundary,
    )

    prompt = writer.render_prompt(
        selected_evidence_ids=("E-known",),
        user_strings=("Write a concise answer",),
    )
    payload = json.loads(prompt)

    assert "fixture excerpt" in seen
    assert "Fixture title" in seen
    assert "Fixture Author" in seen
    assert "Write a concise answer" in seen
    assert any('"kind":"html"' in value for value in seen)
    assert payload["evidence"][0]["evidence_id"] == "E-known"
    assert len(payload["evidence"]) == 1
    assert recording_model.last_prompt == prompt
    assert "PRIVATE RAW BODY" not in prompt

    boundary_call_count = len(seen)
    writer.finalize_report(
        "Supported claim [E-known].",
        selected_evidence_ids=("E-known",),
    )
    assert len(seen) == boundary_call_count


def test_writer_requires_citations_for_claim_paragraphs(
    evidence_store: LocalEvidenceStore,
) -> None:
    writer = MarkdownReportWriter(evidence_store)

    with pytest.raises(EvidenceBackedClaimRequired):
        writer.finalize_report(
            "# Findings\n\nPlanner optimization improved performance.",
            selected_evidence_ids=("E-known",),
        )


def test_heading_does_not_hide_an_uncited_claim_on_the_next_line(
    evidence_store: LocalEvidenceStore,
) -> None:
    with pytest.raises(EvidenceBackedClaimRequired):
        MarkdownReportWriter(evidence_store).finalize_report(
            "# Findings\nPlanner optimization improved performance.",
            selected_evidence_ids=("E-known",),
        )


def test_writer_builds_deterministic_canonical_references(
    evidence_store: LocalEvidenceStore,
) -> None:
    report = MarkdownReportWriter(evidence_store).finalize_report(
        "# Findings\n\nSupported claim [E-known].",
        selected_evidence_ids=("E-known",),
    )

    assert report.endswith(
        "## References\n\n"
        "- [E-known] Fixture title — https://example.com/report?a=1&b=2"
    )
    assert "utm_source" not in report
    assert "PRIVATE RAW BODY" not in report


def test_writer_labels_partial_stop_and_uncovered_needs(
    evidence_store: LocalEvidenceStore,
) -> None:
    report = MarkdownReportWriter(evidence_store).finalize_report(
        "Supported claim [E-known].",
        selected_evidence_ids=("E-known",),
        is_partial=True,
        stop_reason="BUDGET_EXHAUSTED",
        uncovered_information_needs=("Need beta", "Need alpha"),
    )

    assert "## Partial report" in report
    assert "Stop reason: `BUDGET_EXHAUSTED`" in report
    assert "- Need alpha" in report
    assert "- Need beta" in report


def test_writer_rejects_credential_bearing_source_urls(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)
    add_evidence(
        store,
        evidence_id="E-secret",
        source_id="source-secret",
        url="https://user:password@example.com/report",
    )

    with pytest.raises(URLSecurityError, match="credentials"):
        MarkdownReportWriter(store).finalize_report(
            "Supported claim [E-secret].",
            selected_evidence_ids=("E-secret",),
        )


@pytest.mark.parametrize(
    "hidden_citation",
    [
        "<!-- [E-known] -->",
        "`[E-known]`",
        "\n\n```text\n[E-known]\n```",
        "\n\n    [E-known]",
        "<code>[E-known]</code>",
        "[ordinary link](https://example.com/[E-known])",
        "[ordinary link][ref]\n\n[ref]: https://example.com/[E-known]",
    ],
)
def test_non_prose_citations_cannot_support_a_visible_claim(
    evidence_store: LocalEvidenceStore,
    hidden_citation: str,
) -> None:
    with pytest.raises(EvidenceBackedClaimRequired):
        MarkdownReportWriter(evidence_store).finalize_report(
            f"Unsupported visible claim. {hidden_citation}",
            selected_evidence_ids=("E-known",),
        )


def test_hidden_valid_citation_cannot_mask_malformed_visible_citation(
    evidence_store: LocalEvidenceStore,
) -> None:
    with pytest.raises(MalformedEvidenceCitation):
        MarkdownReportWriter(evidence_store).finalize_report(
            "Malformed visible [E-bad!] <!-- [E-known] -->",
            selected_evidence_ids=("E-known",),
        )


def test_citation_inside_inline_code_is_not_added_to_references(
    evidence_store: LocalEvidenceStore,
) -> None:
    report = MarkdownReportWriter(evidence_store).finalize_report(
        "Visible support [E-known]; example code `[E-other]`.",
        selected_evidence_ids=("E-known", "E-other"),
    )

    assert "- [E-known] Fixture title" in report
    assert "- [E-other]" not in report


def test_writer_serializes_hostile_final_metadata_as_markdown_safe_single_lines(
    tmp_path: Path,
) -> None:
    store = LocalEvidenceStore(tmp_path)
    add_evidence(
        store,
        evidence_id="E-known",
        source_id="source-known",
        title="Safe title\n## Injected section\n- [E-missing] `forged`",
    )

    report = MarkdownReportWriter(store).finalize_report(
        "Supported claim [E-known].",
        selected_evidence_ids=("E-known",),
        is_partial=True,
        stop_reason="BUDGET_EXHAUSTED",
        uncovered_information_needs=(
            "Need one\n## Injected need",
            "Need `two` with [E-missing]",
        ),
    )

    assert "\n## Injected" not in report
    assert "\n- [E-missing]" not in report
    assert "\\[E-missing\\]" in report
    assert "Need one \\#\\# Injected need" in report
    assert "Need \\`two\\` with \\[E-missing\\]" in report
    assert [line for line in report.splitlines() if line.startswith("## ")] == [
        "## Partial report",
        "## References",
    ]


@pytest.mark.parametrize(
    "stop_reason",
    ["BUDGET_EXHAUSTED`\n## Injected", "UNKNOWN", ""],
)
def test_writer_restricts_partial_stop_reason_to_public_literals(
    evidence_store: LocalEvidenceStore,
    stop_reason: str,
) -> None:
    with pytest.raises(ValueError, match="stop reason"):
        MarkdownReportWriter(evidence_store).finalize_report(
            "Supported claim [E-known].",
            selected_evidence_ids=("E-known",),
            is_partial=True,
            stop_reason=stop_reason,
        )


@pytest.mark.parametrize(
    "hidden_citation",
    [
        "> ~~~text\n> [E-known]\n> ~~~",
        "<script>[E-known]</script>",
        '<span\n data-note="[E-known]">visible text</span>',
        r"[ordinary](https://example.com/foo\)[E-known])",
        "<style>p::after { content: '[E-known]' }</style>",
        "<pre>example [E-known]</pre>",
    ],
)
def test_commonmark_hidden_citations_cannot_support_visible_claims(
    evidence_store: LocalEvidenceStore,
    hidden_citation: str,
) -> None:
    with pytest.raises(EvidenceBackedClaimRequired):
        MarkdownReportWriter(evidence_store).finalize_report(
            f"Unsupported visible claim. {hidden_citation}",
            selected_evidence_ids=("E-known",),
        )


@pytest.mark.parametrize(
    "draft",
    [
        "# Heading [E-missing]\n\nSupported [E-known].",
        "Supported [E-known]; unknown &#91;E-missing].",
        r"Supported [E-known]; unknown \\[E-missing].",
        "# Malformed [E-bad!]\n\nSupported [E-known].",
    ],
)
def test_all_rendered_visible_citations_are_validated(
    evidence_store: LocalEvidenceStore,
    draft: str,
) -> None:
    error_type = (
        MalformedEvidenceCitation if "bad!" in draft else UnknownEvidenceCitation
    )
    with pytest.raises(error_type):
        MarkdownReportWriter(evidence_store).finalize_report(
            draft, selected_evidence_ids=("E-known",)
        )


@pytest.mark.parametrize(
    "draft",
    [
        "Supported claim &#91;E-known]",
        r"Supported claim with literal tick \` [E-known] `",
        r"Supported claim with visible slash \\[E-known].",
        "Supported claim with literal ```text\n[E-known]\n``` markers.",
        "Supported claim with unresolved [ordinary][E-known].",
    ],
)
def test_commonmark_rendered_citations_support_claim_prose(
    evidence_store: LocalEvidenceStore,
    draft: str,
) -> None:
    report = MarkdownReportWriter(evidence_store).finalize_report(
        draft, selected_evidence_ids=("E-known",)
    )

    assert "- [E-known] Fixture title" in report


def test_html_attributes_and_hidden_bodies_are_ignored_but_visible_html_text_is_validated(
    evidence_store: LocalEvidenceStore,
) -> None:
    writer = MarkdownReportWriter(evidence_store)

    with pytest.raises(UnknownEvidenceCitation, match="E-missing"):
        writer.finalize_report(
            '<span data-note="[E-other]">Visible [E-missing]</span> [E-known].',
            selected_evidence_ids=("E-known",),
        )

    with pytest.raises(EvidenceBackedClaimRequired):
        writer.finalize_report(
            '<span data-note="[E-other]">Visible claim</span> [E-known]. '
            '<script>[E-other]</script>',
            selected_evidence_ids=("E-known",),
        )


def test_link_destinations_and_images_are_hidden_but_visible_link_labels_are_validated(
    evidence_store: LocalEvidenceStore,
) -> None:
    writer = MarkdownReportWriter(evidence_store)
    report = writer.finalize_report(
        "Supported [E-known] with [ordinary](https://example.com/[E-other]) "
        "and ![image [E-other]](https://example.com/image.png).",
        selected_evidence_ids=("E-known",),
    )

    assert "- [E-other]" not in report

    with pytest.raises(UnknownEvidenceCitation, match="E-other"):
        writer.finalize_report(
            "Supported [E-known] and [visible [E-other]](https://example.com).",
            selected_evidence_ids=("E-known",),
        )


def test_ambiguous_raw_html_citation_fails_closed_without_leaking_draft(
    evidence_store: LocalEvidenceStore,
) -> None:
    draft = "Supported [E-known]. <script data-x='[E-other]'"

    with pytest.raises(MalformedEvidenceCitation) as error:
        MarkdownReportWriter(evidence_store).finalize_report(
            draft, selected_evidence_ids=("E-known",)
        )

    assert draft not in str(error.value)
    assert "E-other" not in str(error.value)


def test_raw_html_block_visibility_is_validated_without_supporting_claims(
    evidence_store: LocalEvidenceStore,
) -> None:
    writer = MarkdownReportWriter(evidence_store)

    with pytest.raises(UnknownEvidenceCitation, match="E-missing"):
        writer.finalize_report(
            "<div>\nVisible [E-missing]\n</div>\n\nSupported [E-known].",
            selected_evidence_ids=("E-known",),
        )

    with pytest.raises(EvidenceBackedClaimRequired):
        writer.finalize_report(
            "<div>\nVisible raw claim [E-known]\n</div>",
            selected_evidence_ids=("E-known",),
        )


def test_citation_split_by_inline_markup_cannot_support_claim_prose(
    evidence_store: LocalEvidenceStore,
) -> None:
    with pytest.raises(EvidenceBackedClaimRequired):
        MarkdownReportWriter(evidence_store).finalize_report(
            "Rendered but structurally ambiguous [E-**known**].",
            selected_evidence_ids=("E-known",),
        )


def test_markdown_metadata_preserves_joiners_and_neutralizes_bidi_controls(
    tmp_path: Path,
) -> None:
    store = LocalEvidenceStore(tmp_path)
    title = "研究 👩\u200d💻 Persian می\u200cخواهم \u202eunsafe"
    need = "Need 👨\u200d👩\u200d👧 and می\u200cخواهم \u2066unsafe"
    add_evidence(
        store,
        evidence_id="E-known",
        source_id="source-known",
        title=title,
    )

    report = MarkdownReportWriter(store).finalize_report(
        "Supported [E-known].",
        selected_evidence_ids=("E-known",),
        is_partial=True,
        stop_reason="BUDGET_EXHAUSTED",
        uncovered_information_needs=(need,),
    )

    assert "👩\u200d💻" in report
    assert "👨\u200d👩\u200d👧" in report
    assert "می\u200cخواهم" in report
    assert "\u202e" not in report
    assert "\u2066" not in report


@pytest.mark.parametrize(
    "raw_html",
    [
        "<span hidden>[E-known]</span>",
        '<span style="display:none">[E-known]</span>',
        "<textarea>[E-known]</textarea>",
        "<template>[E-known]</template>",
        "<script/>[E-known]",
        "<span hidden />[E-known]",
    ],
)
def test_any_inline_raw_html_makes_claim_support_fail_closed(
    evidence_store: LocalEvidenceStore,
    raw_html: str,
) -> None:
    with pytest.raises(EvidenceBackedClaimRequired):
        MarkdownReportWriter(evidence_store).finalize_report(
            f"Unsupported visible claim. {raw_html}",
            selected_evidence_ids=("E-known",),
        )


def test_metadata_entities_remain_literal_after_commonmark_rendering(
    tmp_path: Path,
) -> None:
    store = LocalEvidenceStore(tmp_path)
    sequence = "👩\u200d💻\ufe0f"
    add_evidence(
        store,
        evidence_id="E-known",
        source_id="source-known",
        title=f"Safe &lbrack;E-missing&rbrack; R&D {sequence}",
    )

    report = MarkdownReportWriter(store).finalize_report(
        "Supported [E-known].",
        selected_evidence_ids=("E-known",),
        is_partial=True,
        stop_reason="BUDGET_EXHAUSTED",
        uncovered_information_needs=(
            "Need &#91;E-missing&#93; &amp; details",
        ),
    )
    rendered = MarkdownIt("commonmark").render(report)

    assert "[E-missing]" not in rendered
    assert "&amp;lbrack;E-missing&amp;rbrack;" in rendered
    assert "&amp;#91;E-missing&amp;#93;" in rendered
    assert sequence in rendered
    assert "https://example.com/report?a=1&amp;b=2" in rendered
