"""Render hash-verified public benchmark summaries.

The renderer is deliberately a small publication boundary.  It accepts only
aggregate JSON objects and the public artifact manifest produced by
``experiments.summarize``.  It never walks an experiment's ``raw`` tree and it
does not load the repository's private dataset, prompts, or provider output.

The module also accepts an in-memory mapping.  That form is useful for the
contract tests and for downstream tooling that has already performed the
artifact verification step.  A directory input performs the verification in
this module before any values are rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast


class ResultValidationError(ValueError):
    """Raised when a public result or its artifact seal is not trustworthy."""


_PROTOCOLS: Final[tuple[str, ...]] = (
    "ranker_component",
    "planner_policy",
    "end_to_end",
)
_PUBLIC_ARTIFACTS: Final[frozenset[str]] = frozenset(
    {
        "summary.json",
        "task_metrics.jsonl",
        "confidence_intervals.json",
        "pareto.json",
        "failures.jsonl",
        "replication-bindings.jsonl",
        "oracle-reference.jsonl",
        "evaluator-reference-manifest.json",
    }
)
_FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "private",
    "gold",
    "prompt",
    "provider",
    "credential",
    "secret",
    "response",
    "raw",
    "rubric",
    "acceptable_claim",
)
_CI_KEYS: Final[tuple[str, ...]] = (
    "confidence_interval",
    "ci",
    "primary_ci",
    "ranker_primary_ci",
)


PublicMap = dict[str, object]
SectionMap = dict[str, PublicMap]
Source = Mapping[str, object] | Path | str


def _mapping(value: object, *, label: str) -> PublicMap:
    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        return {str(key): item for key, item in raw.items()}
    raise ResultValidationError(f"{label} must be a JSON object")


def _safe_path(path: Path, *, label: str) -> Path:
    """Reject links/reparse points before opening a publication input/output."""

    absolute = path.absolute()
    current = absolute
    while True:
        try:
            if current.is_symlink():
                raise ResultValidationError(f"{label} contains a symlink or reparse point")
        except OSError as error:
            raise ResultValidationError(f"{label} is unavailable") from error
        if current == Path(current.anchor):
            break
        current = current.parent
    return absolute


def _read_json(source: Source, *, label: str) -> PublicMap:
    if isinstance(source, Mapping):
        return _mapping(source, label=label)
    path = _safe_path(Path(source), label=label)
    if not path.is_file():
        raise ResultValidationError(f"{label} is missing")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, TypeError, ValueError) as error:
        raise ResultValidationError(f"{label} is invalid") from error
    return _mapping(payload, label=label)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        and value != "0" * 64
    )


def _verify_manifest(root: Path, manifest: Mapping[str, object]) -> dict[str, bytes]:
    """Verify and return only the public artifacts named by a manifest."""

    raw = _mapping(manifest, label="public manifest")
    if raw.get("schema_version") != "experiment-artifact-manifest-v1":
        raise ResultValidationError("public manifest schema is invalid")
    files = raw.get("files")
    if not isinstance(files, Mapping):
        # A small, explicit alternative is useful to callers publishing one
        # summary file, while still requiring a hash over the bytes on disk.
        summary_hash = raw.get("summary_sha256")
        if not _valid_hash(summary_hash):
            raise ResultValidationError("public manifest files are invalid")
        files = {"summary.json": summary_hash}
    raw_files = cast(Mapping[object, object], files)
    listed: dict[str, object] = {str(name): digest for name, digest in raw_files.items()}
    if "summary.json" not in listed:
        raise ResultValidationError("public manifest must hash summary.json")
    if not listed:
        raise ResultValidationError("public manifest is empty")

    result: dict[str, bytes] = {}
    for name, expected in sorted(listed.items()):
        if (
            name not in _PUBLIC_ARTIFACTS
            or Path(name).name != name
            or any(token in name.casefold() for token in _FORBIDDEN_TOKENS)
            or not _valid_hash(expected)
        ):
            raise ResultValidationError("public manifest contains an unsafe entry")
        path = _safe_path(root / name, label=f"public artifact {name}")
        if not path.is_file():
            raise ResultValidationError(f"public artifact {name} is missing")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ResultValidationError(f"public artifact {name} is unavailable") from error
        if _sha256(payload) != expected:
            raise ResultValidationError(f"public artifact hash mismatch: {name}")
        result[name] = payload
    return result


def _load_jsonl(payload: bytes, *, label: str) -> list[PublicMap]:
    rows: list[PublicMap] = []
    for line in payload.splitlines():
        try:
            rows.append(_mapping(json.loads(line), label=label))
        except (TypeError, ValueError, ResultValidationError) as error:
            raise ResultValidationError(f"{label} is invalid") from error
    return rows


def _normalise_sections(payload: Mapping[str, object]) -> tuple[PublicMap, SectionMap]:
    """Return the public summary plus a canonical three-protocol section map."""

    source = _mapping(payload, label="summary")
    nested = source.get("sections")
    section_source = _mapping(nested, label="summary sections") if nested is not None else source
    aliases: dict[str, tuple[str, ...]] = {
        "ranker_component": ("ranker_component", "ranker", "ranker_result"),
        "planner_policy": ("planner_policy", "planner", "planner_result"),
        "end_to_end": ("end_to_end", "policy", "abcd", "end_to_end_result"),
    }
    sections: SectionMap = {}
    for protocol, names in aliases.items():
        found = next((section_source[name] for name in names if name in section_source), None)
        if found is None:
            raise ResultValidationError(f"summary is missing {protocol}")
        sections[protocol] = _mapping(found, label=f"{protocol} section")
    reference = section_source.get("reference", source.get("reference"))
    sections["reference"] = (
        _mapping(reference, label="reference section")
        if reference is not None
        else {"status": "not_present", "reason": "reference summary was not supplied"}
    )
    if source.get("schema_version") not in (None, "experiment-summary-v1"):
        raise ResultValidationError("summary schema is invalid")
    return source, sections


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _fmt(value: object, *, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        if value is None:
            return "not reported"
        return _safe_text(value)
    if number == 0:
        return "0"
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _safe_text(value: object, *, max_length: int = 160) -> str:
    if not isinstance(value, str):
        return str(value)
    text = " ".join(value.split())
    if any(token in text.casefold() for token in _FORBIDDEN_TOKENS):
        return "[redacted]"
    return text[:max_length]


def _first(mapping: Mapping[str, object], names: Sequence[str]) -> object | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _ci(value: object) -> tuple[float | None, float | None, float | None]:
    """Read either a CI object or the compact ``(lower, upper)`` form."""

    if isinstance(value, Mapping):
        item = cast(Mapping[str, object], value)
        estimate = _as_float(_first(item, ("estimate", "mean", "value")))
        lower = _as_float(_first(item, ("lower", "low", "min")))
        upper = _as_float(_first(item, ("upper", "high", "max")))
        if lower is not None and upper is not None:
            return estimate, lower, upper
        return None, None, None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_values = cast(Sequence[object], value)
        values = tuple(_as_float(item) for item in raw_values)
        if len(values) == 2 and all(item is not None for item in values):
            lower, upper = cast(tuple[float, float], values)
            return (lower + upper) / 2.0, lower, upper
        if len(values) >= 3 and all(item is not None for item in values[:3]):
            estimate, lower, upper = cast(tuple[float, float, float], values[:3])
            return estimate, lower, upper
    return None, None, None


def _find_ci(
    source: Mapping[str, object], payload: Mapping[str, object]
) -> tuple[float | None, float | None, float | None]:
    for mapping in (source, payload):
        for key in _CI_KEYS:
            if key in mapping:
                result = _ci(mapping[key])
                if result[1] is not None and result[2] is not None:
                    return result
    nested = source.get("results")
    if isinstance(nested, Mapping):
        return _find_ci(cast(Mapping[str, object], nested), payload)
    return None, None, None


def _metrics(section: Mapping[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key in (
        "metrics",
        "quality_metrics",
        "evidence_metrics",
        "efficiency_metrics",
        "failure_metrics",
    ):
        candidate = section.get(key)
        if isinstance(candidate, Mapping):
            raw_candidate = cast(Mapping[object, object], candidate)
            for name, value in raw_candidate.items():
                name_text = str(name)
                if any(token in name_text.casefold() for token in _FORBIDDEN_TOKENS):
                    continue
                if isinstance(value, Mapping):
                    scalar = _first(
                        cast(Mapping[str, object], value), ("mean", "value", "estimate", "count")
                    )
                    values.setdefault(name_text, scalar)
                elif _as_float(value) is not None or isinstance(value, str):
                    values.setdefault(name_text, value)
    return dict(sorted(values.items()))


def _variant_rows(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    candidate = _first(payload, ("abcd", "variant_metrics", "variants"))
    rows: dict[str, dict[str, object]] = {}
    if isinstance(candidate, Mapping):
        raw_candidate = cast(Mapping[object, object], candidate)
        for variant in ("A", "B", "C", "D"):
            item = raw_candidate.get(variant)
            if isinstance(item, Mapping):
                rows[variant] = _mapping(
                    cast(Mapping[object, object], item), label=f"variant {variant}"
                )
    elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        raw_candidate = cast(Sequence[object], candidate)
        for item in raw_candidate:
            if isinstance(item, Mapping):
                row = _mapping(cast(Mapping[object, object], item), label="variant row")
                variant = row.get("variant")
                if isinstance(variant, str) and variant in {"A", "B", "C", "D"}:
                    rows[variant] = row
    return rows


def _status_text(value: object) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "failed"
    if value is None:
        return "not reported"
    return _safe_text(value)


def _planner_noninferior(
    section: Mapping[str, object], payload: Mapping[str, object]
) -> bool | None:
    direct = _first(section, ("non_inferior", "noninferior", "planner_noninferiority"))
    if direct is None:
        direct = payload.get("planner_noninferiority")
    if isinstance(direct, bool):
        return direct
    comparisons = section.get("comparisons")
    if isinstance(comparisons, Mapping):
        raw_comparisons = cast(Mapping[object, object], comparisons)
        values = [
            cast(Mapping[str, object], item).get("non_inferior")
            for item in raw_comparisons.values()
            if isinstance(item, Mapping)
        ]
        bools = [item for item in values if isinstance(item, bool)]
        if bools:
            return all(bools)
    return None


def _failure_reasons(
    payload: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
    *,
    ranker_ci: tuple[float | None, float | None, float | None],
    planner_noninferior: bool | None,
) -> list[str]:
    reasons: list[str] = []
    _, lower, upper = ranker_ci
    if upper is not None and upper < 0:
        reasons.append("ranker confidence interval is below zero")
    elif lower is not None and lower < 0:
        reasons.append("ranker interval crosses the null")
    if planner_noninferior is False:
        reasons.append("planner non-inferiority criterion was not met")
    failures = payload.get("failures")
    if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes, bytearray)):
        counts: dict[str, int] = {}
        raw_failures = cast(Sequence[object], failures)
        for item in raw_failures:
            if isinstance(item, Mapping):
                raw_item = cast(Mapping[str, object], item)
                code = raw_item.get("error_code") or raw_item.get("status")
                if isinstance(code, str) and not any(
                    token in code.casefold() for token in _FORBIDDEN_TOKENS
                ):
                    counts[code] = counts.get(code, 0) + 1
        reasons.extend(f"{code}: {count}" for code, count in sorted(counts.items()))
    if not reasons:
        statuses = sections.get("end_to_end", {}).get("statuses")
        if isinstance(statuses, Mapping):
            raw_statuses = cast(Mapping[object, object], statuses)
            failed = sum(
                int(_as_float(value) or 0)
                for key, value in raw_statuses.items()
                if str(key).casefold() not in {"completed", "success"}
            )
            if failed:
                reasons.append(f"non-completed end-to-end runs: {failed}")
    return reasons


def _metadata_lines(payload: Mapping[str, object]) -> list[str]:
    metadata = payload.get("metadata")
    source = (
        _mapping(cast(Mapping[object, object], metadata), label="metadata")
        if isinstance(metadata, Mapping)
        else dict(payload)
    )
    allowed = (
        "dataset_version",
        "dataset_id",
        "config_version",
        "config_sha256",
        "code_version",
        "code_tree_sha256",
        "model_version",
        "model_id",
        "evaluator_version",
        "evaluation_date",
        "evaluation_timestamp",
        "group_id",
    )
    lines: list[str] = []
    for key in allowed:
        value = source.get(key)
        if value is None or any(token in key.casefold() for token in _FORBIDDEN_TOKENS):
            continue
        lines.append(f"- `{key}`: {_safe_text(value)}")
    return lines


def _confidence_lines(payload: Mapping[str, object]) -> list[str]:
    confidence = payload.get("confidence_intervals")
    if not isinstance(confidence, Mapping):
        return []
    raw_confidence = cast(Mapping[str, object], confidence)
    sections = raw_confidence.get("sections", raw_confidence)
    if not isinstance(sections, Mapping):
        return []
    raw_sections = cast(Mapping[object, object], sections)
    lines: list[str] = []
    for protocol, value in sorted(raw_sections.items(), key=lambda item: str(item[0])):
        if isinstance(value, Mapping):
            ci = _first(cast(Mapping[str, object], value), _CI_KEYS)
            if ci is not None:
                estimate, lower, upper = _ci(ci)
                lines.append(
                    f"- `{protocol}`: estimate {_fmt(estimate)}, 95% CI [{_fmt(lower)}, {_fmt(upper)}]"
                )
    return lines


def _pareto_lines(payload: Mapping[str, object]) -> list[str]:
    pareto = payload.get("pareto")
    if not isinstance(pareto, Mapping):
        return []
    raw_pareto = cast(Mapping[str, object], pareto)
    sections = raw_pareto.get("sections", raw_pareto)
    if not isinstance(sections, Mapping):
        return []
    raw_sections = cast(Mapping[object, object], sections)
    lines: list[str] = []
    for protocol, value in sorted(raw_sections.items(), key=lambda item: str(item[0])):
        values: list[Mapping[str, object]] = []
        if isinstance(value, Mapping):
            values = [cast(Mapping[str, object], value)]
            raw_value = cast(Mapping[object, object], value)
            for child in raw_value.values():
                if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                    raw_child = cast(Sequence[object], child)
                    values.extend(
                        cast(Mapping[str, object], item)
                        for item in raw_child
                        if isinstance(item, Mapping)
                    )
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            raw_value = cast(Sequence[object], value)
            values = [
                cast(Mapping[str, object], item) for item in raw_value if isinstance(item, Mapping)
            ]
        for item in values:
            dominance = item.get("bootstrap_dominance_proportion")
            if dominance is None:
                dominance = item.get("dominance_proportion")
            if dominance is not None:
                lines.append(f"- `{protocol}` bootstrap dominance: {_fmt(dominance)}")
    return lines


def _render_human(value: Mapping[str, object] | None) -> list[str]:
    if value is None:
        return ["Human rating summary: not sealed/present."]
    summary = value.get("human_summary")
    source = (
        _mapping(cast(Mapping[object, object], summary), label="human summary")
        if isinstance(summary, Mapping)
        else dict(value)
    )
    lines = ["Human rating summary (aggregate only):"]
    for key in (
        "rating_count",
        "valid_tasks",
        "tie_rate",
        "mean_dimension_scores",
        "majority_preference",
        "krippendorff_alpha",
        "auto_metric_spearman",
    ):
        item = source.get(key)
        if item is None:
            continue
        if isinstance(item, Mapping):
            raw_item = cast(Mapping[object, object], item)
            rendered = ", ".join(
                f"{_safe_text(name)}={_fmt(metric)}"
                for name, metric in sorted(raw_item.items(), key=lambda pair: str(pair[0]))
            )
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            rendered = str(len(cast(Sequence[object], item)))
        else:
            rendered = _fmt(item)
        lines.append(f"- `{key}`: {rendered}")
    return lines


def _render_external(value: Mapping[str, object] | None) -> list[str]:
    if value is None:
        return ["External 10/20/10 benchmark results: not sealed/present."]
    source = _mapping(value, label="external summary")
    counts = source.get("benchmark_counts")
    if not isinstance(counts, Mapping):
        counts = source.get("counts")
    lines = ["External 10/20/10 benchmark results (separate from primary CIs):"]
    if isinstance(counts, Mapping):
        raw_counts = cast(Mapping[object, object], counts)
        for name, count in sorted(raw_counts.items(), key=lambda pair: str(pair[0])):
            lines.append(f"- `{_safe_text(name)}`: {_fmt(count)} tasks")
    else:
        lines.append("- aggregate metrics: not reported")
    return lines


def _render_markdown(
    payload: Mapping[str, object],
    sections: Mapping[str, Mapping[str, object]],
    *,
    human: Mapping[str, object] | None,
    external: Mapping[str, object] | None,
    verified: bool,
) -> str:
    ranker = sections["ranker_component"]
    planner = sections["planner_policy"]
    ranker_ci = _find_ci(ranker, payload)
    planner_noninferior = _planner_noninferior(planner, payload)
    _, lower, upper = ranker_ci
    negative = (upper is not None and upper < 0) or planner_noninferior is False
    reasons = _failure_reasons(
        payload,
        sections,
        ranker_ci=ranker_ci,
        planner_noninferior=planner_noninferior,
    )
    outcome = (
        "primary hypothesis not established" if negative else "primary result is not yet sealed"
    )

    lines = [
        "# Benchmark results",
        "",
        f"> Factual outcome: {outcome}.",
        "",
        "This page is generated from public aggregate values; it is not a claim of a completed formal run.",
        "",
        "## Provenance",
        "",
        f"- Public artifact seal verified: {'yes' if verified else 'no (caller supplied an in-memory summary)'}.",
    ]
    metadata = _metadata_lines(payload)
    if metadata:
        lines.extend(metadata)
    lines.extend(
        [
            "",
            "## A/B/C/D summary",
            "",
            "| Variant | Quality | Evidence | Efficiency | Failure |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    rows = _variant_rows(payload)
    for variant in ("A", "B", "C", "D"):
        row = rows.get(variant, {})
        values = [
            _metrics(row).get(name) for name in ("quality", "evidence", "efficiency", "failure")
        ]
        lines.append(f"| {variant} | {' | '.join(_fmt(value) for value in values)} |")
    lines.extend(
        [
            "",
            "## Ranker primary result",
            "",
            f"- Comparison: {_safe_text(ranker.get('candidate', 'R2'))} vs {_safe_text(ranker.get('baseline', 'R1'))}.",
            f"- Estimate: {_fmt(ranker_ci[0])}; 95% CI: [{_fmt(lower)}, {_fmt(upper)}].",
            "",
            "## Planner policy",
            "",
            f"- Non-inferiority: {_status_text(planner_noninferior)}.",
            "- Secondary efficiency results are reported only when present in the public summary.",
        ]
    )
    planner_metrics = _metrics(planner)
    for name, value in planner_metrics.items():
        lines.append(f"- `{name}` mean: {_fmt(value)}.")
    comparisons = planner.get("comparisons")
    if isinstance(comparisons, Mapping):
        raw_comparisons = cast(Mapping[object, object], comparisons)
        for name, comparison in sorted(raw_comparisons.items(), key=lambda item: str(item[0])):
            if not isinstance(comparison, Mapping):
                continue
            raw_comparison = cast(Mapping[str, object], comparison)
            completeness = raw_comparison.get("completeness")
            if completeness is not None:
                estimate, ci_lower, ci_upper = _ci(completeness)
                lines.append(
                    f"- `{_safe_text(name)}` completeness: estimate {_fmt(estimate)}, "
                    f"95% CI [{_fmt(ci_lower)}, {_fmt(ci_upper)}]."
                )
            for metric_name in ("search_calls_reduction", "query_redundancy_reduction"):
                reduction = raw_comparison.get(metric_name)
                if reduction is not None:
                    estimate, ci_lower, ci_upper = _ci(reduction)
                    lines.append(
                        f"- `{_safe_text(name)}.{metric_name}`: estimate {_fmt(estimate)}, "
                        f"95% CI [{_fmt(ci_lower)}, {_fmt(ci_upper)}]."
                    )
    lines.extend(
        [
            "",
            "## Pareto planes",
            "",
            "Both quality-versus-cost and quality-versus-search planes are retained.",
        ]
    )
    pareto = _pareto_lines(payload)
    lines.extend(pareto or ["- Bootstrap dominance proportions: not reported."])
    lines.extend(
        [
            "",
            "## Efficiency and failures",
            "",
            "p50/p95 latency, tokens, estimated USD cost, and failure breakdown are shown only when present in the public aggregate.",
        ]
    )
    for key in (
        "p50_latency",
        "p95_latency",
        "p50_wall_seconds",
        "p95_wall_seconds",
        "p50_tokens",
        "p95_tokens",
        "p50_cost_usd",
        "p95_cost_usd",
    ):
        if key in payload:
            lines.append(f"- `{key}`: {_fmt(payload[key])}")
    for protocol in _PROTOCOLS:
        values = _metrics(sections[protocol])
        for name, value in values.items():
            lines.append(f"- `{protocol}.{name}` mean: {_fmt(value)}")
    end_to_end_ci = _find_ci(sections["end_to_end"], payload)
    if end_to_end_ci[1] is not None and end_to_end_ci[2] is not None:
        lines.append(
            f"- `end_to_end` estimate: {_fmt(end_to_end_ci[0])}; 95% CI: "
            f"[{_fmt(end_to_end_ci[1])}, {_fmt(end_to_end_ci[2])}]"
        )
    failure_rows = payload.get("failures")
    if isinstance(failure_rows, Sequence) and not isinstance(failure_rows, (str, bytes, bytearray)):
        failure_counts: dict[str, int] = {}
        for row in cast(Sequence[object], failure_rows):
            if isinstance(row, Mapping):
                raw_row = cast(Mapping[str, object], row)
                code = raw_row.get("error_code") or raw_row.get("status")
                if isinstance(code, str) and not any(
                    token in code.casefold() for token in _FORBIDDEN_TOKENS
                ):
                    failure_counts[code] = failure_counts.get(code, 0) + 1
        for code, count in sorted(failure_counts.items()):
            lines.append(f"- failure `{_safe_text(code)}`: {count}")
    confidence = _confidence_lines(payload)
    if confidence:
        lines.extend(["", "Confidence intervals from the verified aggregate:", *confidence])
    lines.extend(
        [
            "",
            "## P0 reference and evaluator ceiling",
            "",
            "P0 agent reference and the ORACLE ceiling remain evaluator-side, hash-only metadata; they are not mixed into agent cost or confidence intervals.",
            f"- Reference status: {_safe_text(sections['reference'].get('status', 'not reported'))}.",
            "",
            "## Human evaluation",
            "",
            *_render_human(human),
            "",
            "## External evaluation",
            "",
            *_render_external(external),
            "",
            "## Failure analysis",
            "",
        ]
    )
    if reasons:
        lines.append("主假设未成立；失败分析：" + "; ".join(reasons) + ".")
    else:
        lines.append("主假设未判定；失败分析：公开摘要没有足够的失败标量可供归因。")
    lines.extend(
        [
            "",
            "## Limitations and disclosure",
            "",
            "- Missing human/external artifacts are preserved as missing; they are not imputed.",
            "- Re-runs must use the same sealed configuration, model/environment locks, and replication policy.",
            "- Figures in this checkout are deterministic layout placeholders until formal public aggregates are sealed.",
            "- This publication boundary does not read sealed prompts, private gold, raw provider responses, or credentials.",
            "",
        ]
    )
    return "\n".join(lines)


def _svg(*, chart_id: str, title: str, x_label: str, y_label: str) -> str:
    """Return a fixed, accessible placeholder chart with no random values."""

    title_id = f"{chart_id}-title"
    description_id = f"{chart_id}-description"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420" role="img" aria-labelledby="{title_id} {description_id}">
  <title id="{title_id}">{html.escape(title)}</title>
  <desc id="{description_id}">Illustrative deterministic placeholder; no formal result is sealed. Horizontal axis: {html.escape(x_label)}. Vertical axis: {html.escape(y_label)}.</desc>
  <rect width="720" height="420" fill="#ffffff"/>
  <text x="360" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#111827">{html.escape(title)}</text>
  <line x1="80" y1="350" x2="670" y2="350" stroke="#374151" stroke-width="2"/>
  <line x1="80" y1="70" x2="80" y2="350" stroke="#374151" stroke-width="2"/>
  <line x1="80" y1="280" x2="670" y2="280" stroke="#d1d5db"/>
  <line x1="80" y1="210" x2="670" y2="210" stroke="#d1d5db"/>
  <line x1="80" y1="140" x2="670" y2="140" stroke="#d1d5db"/>
  <rect x="150" y="265" width="100" height="85" fill="#2563eb"/>
  <rect x="315" y="210" width="100" height="140" fill="#f97316"/>
  <rect x="480" y="295" width="100" height="55" fill="#059669"/>
  <text x="200" y="370" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">illustrative A</text>
  <text x="365" y="370" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">illustrative B</text>
  <text x="530" y="370" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">illustrative C</text>
  <text x="375" y="405" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#111827">{html.escape(x_label)}</text>
  <text x="22" y="210" text-anchor="middle" transform="rotate(-90 22 210)" font-family="sans-serif" font-size="14" fill="#111827">{html.escape(y_label)}</text>
  <text x="360" y="58" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#991b1b">Illustrative placeholder — no formal result is sealed</text>
</svg>
"""


def _evaluation_doc() -> str:
    return """# Evaluation protocol

This page records the reproducible benchmark procedure.  It is intentionally
usable before a formal result is sealed: absent artifacts remain marked as
missing, and this repository does not manufacture human, external, or CI data.

## Commands

```powershell
uv run deepresearch experiment config id --config benchmarks/configs/formal.yaml
uv run deepresearch experiment run --config benchmarks/configs/formal.yaml --variants A,B,C,D
uv run deepresearch experiment summarize --experiment-dir experiments/<group-id>
uv run python -m benchmarks.scripts.render_results --experiment-dir experiments/<group-id> --docs-dir docs
```

For the optional Portfolio extension, pass a separately sealed external
directory with `--external-experiment-dir`.  Primary confidence intervals are
never pooled with external metrics.

## Isolation and provenance

Agents receive RuntimeTask projections and hash-addressed snapshots.  The
evaluator owns private gold and rubric data in a separate process boundary.
Publication consumes only `summary.json`, the public `manifest.sha256`, and
other manifest-listed aggregate artifacts.  Raw run records, prompts, provider
responses, credentials, and private gold are outside this boundary.

## Metrics and statistics

Quality, evidence, efficiency, latency, token, cost, and failure metrics are
reported with their metric versions and missingness.  Seeds are aggregated at
the task level before paired comparisons.  Formal paired intervals use the
sealed budget and replication policy with 10,000 stratified bootstrap draws;
the planner non-inferiority margin is the pre-registered completeness margin.

## Budget and subset policy

The formal configuration fixes the primary budget, sensitivity presets, task
subsets, seeds/repeats, model/environment locks, pricing snapshot, and
evaluation timestamp.  A rerun must use the same sealed values.  Cost is
labelled estimated when it comes from the normalized pricing schedule.

## Reproducibility limitations

The checked-in SVGs are deterministic, accessible placeholders until a public
formal summary is available.  Human ratings and external 10/20/10 results are
optional aggregate inputs; no score is imputed when those inputs are absent.
"""


def _write_output(path: Path, content: str) -> None:
    _safe_path(path.parent, label="documentation output parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ResultValidationError(f"documentation output {path.name} is a symlink")
    path.write_text(content, encoding="utf-8", newline="\n")


def _load_external_directory(root: Path) -> PublicMap:
    root = _safe_path(root, label="external experiment directory")
    if not root.is_dir():
        raise ResultValidationError("external experiment directory is missing")
    metrics_path = root / "metrics.json"
    metrics_path = _safe_path(metrics_path, label="external metrics")
    if not metrics_path.is_file():
        raise ResultValidationError("external metrics are missing")
    outer_manifest = root / "manifest.sha256"
    if outer_manifest.is_file():
        manifest = _read_json(outer_manifest, label="external artifact manifest")
        files = manifest.get("files")
        if not isinstance(files, Mapping):
            raise ResultValidationError("external artifact manifest is invalid")
        raw_files = cast(Mapping[object, object], files)
        expected = raw_files.get("metrics.json")
        if not _valid_hash(expected):
            raise ResultValidationError("external metrics hash is invalid")
        try:
            metrics_bytes = metrics_path.read_bytes()
        except OSError as error:
            raise ResultValidationError("external metrics are unavailable") from error
        if _sha256(metrics_bytes) != expected:
            raise ResultValidationError("external metrics hash mismatch")
    try:
        payload = json.loads(metrics_path.read_bytes())
    except (OSError, TypeError, ValueError) as error:
        raise ResultValidationError("external metrics are invalid") from error
    result = _mapping(payload, label="external metrics")
    if result.get("schema_version") not in (None, "external-metrics-v1"):
        raise ResultValidationError("external metrics schema is invalid")
    for key in ("formal_config_sha256", "external_lock_sha256"):
        if key in result and not _valid_hash(result[key]):
            raise ResultValidationError(f"external {key} is invalid")
    counts = result.get("benchmark_counts")
    if counts is not None:
        if not isinstance(counts, Mapping):
            raise ResultValidationError("external benchmark counts are invalid")
        raw_counts = cast(Mapping[object, object], counts)
        for name, count in raw_counts.items():
            numeric_count = _as_float(count)
            if not isinstance(name, str) or numeric_count is None or numeric_count < 0:
                raise ResultValidationError("external benchmark counts are invalid")
    return result


def _artifact_overlay(artifacts: Mapping[str, bytes]) -> PublicMap:
    overlay: PublicMap = {}
    for name in ("confidence_intervals.json", "pareto.json"):
        payload = artifacts.get(name)
        if payload is not None:
            try:
                overlay[name.removesuffix(".json")] = json.loads(payload)
            except (TypeError, ValueError) as error:
                raise ResultValidationError(f"public artifact {name} is invalid") from error
    failures = artifacts.get("failures.jsonl")
    if failures is not None:
        overlay["failures"] = _load_jsonl(failures, label="public failures")
    return overlay


def render_results(
    summary: Source | None = None,
    *,
    manifest: Source | None = None,
    human_summary: Source | None = None,
    external_summary: Source | None = None,
    experiment_dir: Path | str | None = None,
    external_experiment_dir: Path | str | None = None,
    docs_dir: Path | str | None = None,
) -> str:
    """Validate and render public benchmark results deterministically.

    ``render_results(mapping)`` returns the Markdown page without writing it.
    Supplying ``experiment_dir`` (or a directory as the first argument) reads
    only manifest-listed public artifacts and writes documentation when
    ``docs_dir`` is provided.
    """

    root: Path | None = Path(experiment_dir) if experiment_dir is not None else None
    summary_payload: PublicMap
    artifacts: dict[str, bytes] = {}
    verified = False

    if root is None and isinstance(summary, (Path, str)):
        possible = Path(summary)
        if possible.is_dir():
            root = possible
            summary = None
    if root is None and isinstance(manifest, (Path, str)):
        manifest_path = _safe_path(Path(manifest), label="public manifest")
        root = manifest_path.parent
        manifest_payload = _read_json(manifest_path, label="public manifest")
        artifacts = _verify_manifest(root, manifest_payload)
        verified = True
        if summary is None:
            summary_payload = _read_json(root / "summary.json", label="public summary")
        else:
            summary_payload = _read_json(summary, label="public summary")
        summary_payload.update(_artifact_overlay(artifacts))
    if root is not None:
        root = _safe_path(root, label="experiment directory")
        if not root.is_dir():
            raise ResultValidationError("experiment directory is missing")
        summary_path = root / "summary.json"
        manifest_path = root / "manifest.sha256"
        summary_payload = _read_json(summary_path, label="public summary")
        manifest_payload = _read_json(manifest_path, label="public manifest")
        artifacts = _verify_manifest(root, manifest_payload)
        verified = True
        summary_payload.update(_artifact_overlay(artifacts))
    elif summary is not None:
        summary_payload = _read_json(summary, label="public summary")
        if manifest is not None and isinstance(manifest, Mapping):
            manifest_map = _mapping(manifest, label="public manifest")
            if manifest_map.get("verified") is False:
                raise ResultValidationError("public manifest verification failed")
            if manifest_map.get("schema_version") not in (None, "experiment-artifact-manifest-v1"):
                raise ResultValidationError("public manifest schema is invalid")
    else:
        raise ResultValidationError("public summary is required")

    source_payload, sections = _normalise_sections(summary_payload)
    embedded_manifest = source_payload.get("manifest")
    if isinstance(embedded_manifest, Mapping):
        embedded_map = _mapping(
            cast(Mapping[object, object], embedded_manifest), label="public manifest"
        )
        if embedded_map.get("verified") is False:
            raise ResultValidationError("public manifest verification failed")
    # A supplied top-level manifest is metadata only for in-memory callers;
    # actual byte verification is available through ``experiment_dir`` above.
    if manifest is not None and isinstance(manifest, Mapping):
        source_payload["manifest"] = {"verified": True}

    human: PublicMap | None = None
    if human_summary is not None:
        human = _read_json(human_summary, label="human aggregate summary")
    external: PublicMap | None = None
    if external_summary is not None:
        external = _read_json(external_summary, label="external aggregate summary")
    elif external_experiment_dir is not None:
        external = _load_external_directory(Path(external_experiment_dir))

    page = _render_markdown(
        source_payload,
        sections,
        human=human,
        external=external,
        verified=verified,
    )
    if docs_dir is not None:
        output_root = _safe_path(Path(docs_dir), label="documentation output root")
        assets = output_root / "assets" / "results"
        _write_output(output_root / "results.md", page)
        _write_output(output_root / "evaluation.md", _evaluation_doc())
        _write_output(
            assets / "citation-support-vs-usd.svg",
            _svg(
                chart_id="citation-support-vs-usd",
                title="Citation support vs estimated USD",
                x_label="estimated USD",
                y_label="citation support",
            ),
        )
        _write_output(
            assets / "completeness-vs-search.svg",
            _svg(
                chart_id="completeness-vs-search",
                title="Completeness vs search calls",
                x_label="search calls",
                y_label="information completeness",
            ),
        )
        _write_output(
            assets / "abcd-metrics.svg",
            _svg(
                chart_id="abcd-metrics",
                title="A/B/C/D metric overview",
                x_label="variant",
                y_label="metric value",
            ),
        )
    return page


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--external-experiment-dir", type=Path)
    parser.add_argument("--human-summary", type=Path)
    parser.add_argument("--external-summary", type=Path)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        render_results(
            experiment_dir=args.experiment_dir,
            external_experiment_dir=args.external_experiment_dir,
            human_summary=args.human_summary,
            external_summary=args.external_summary,
            docs_dir=args.docs_dir,
        )
    except ResultValidationError as error:
        print(f"render_results: {error}")
        return 2
    print(f"rendered public results to {args.docs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ResultValidationError", "main", "render_results"]
