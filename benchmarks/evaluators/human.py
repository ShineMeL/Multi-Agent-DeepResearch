"""Blinded human evaluation contracts for the Portfolio benchmark.

The public part of the human-evaluation workflow deliberately contains only
blind packets, ratings and aggregate statistics.  The mapping from a packet's
``X``/``Y`` labels back to the two experimental variants is a private
artifact, so it is not represented by any model in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

HUMAN_DIMENSIONS = (
    "factual_correctness",
    "evidence_sufficiency",
    "information_coverage",
    "analysis_depth",
    "readability",
    "citation_verifiability",
)
HumanDimension = Literal[
    "factual_correctness",
    "evidence_sufficiency",
    "information_coverage",
    "analysis_depth",
    "readability",
    "citation_verifiability",
]
Preference = Literal["X", "Y", "TIE"]
_PREFERENCES = frozenset({"X", "Y", "TIE"})


def _nonblank(value: str, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _finite_nonnegative(value: object, *, field: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return value


class ReportPair(BaseModel):
    """The evaluator-side input containing a paired A/D report.

    This model is intentionally richer than :class:`BlindPacket`: its fields
    are used only while constructing a packet and never cross the rater
    boundary.  ``left`` is conventionally A and ``right`` conventionally D,
    but the convention is not exposed in the resulting packet.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    left_run_id: str
    right_run_id: str
    left_variant: str
    right_variant: str
    left_report: str
    right_report: str
    left_model: str | None = None
    right_model: str | None = None
    left_cost_usd: float | None = None
    right_cost_usd: float | None = None
    left_tokens: int | None = None
    right_tokens: int | None = None
    left_latency_ms: float | None = None
    right_latency_ms: float | None = None
    left_automatic_metrics: dict[str, float] = Field(default_factory=dict)
    right_automatic_metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator(
        "task_id",
        "left_run_id",
        "right_run_id",
        "left_variant",
        "right_variant",
        "left_report",
        "right_report",
    )
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _nonblank(value, field=str(getattr(info, "field_name", "value")))

    @field_validator("left_model", "right_model")
    @classmethod
    def validate_optional_model(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, field="model")

    @field_validator("left_cost_usd", "right_cost_usd", "left_latency_ms", "right_latency_ms")
    @classmethod
    def validate_nonnegative_float(cls, value: float | None, info: object) -> float | None:
        result = _finite_nonnegative(value, field=str(getattr(info, "field_name", "value")))
        return cast(float | None, result)

    @field_validator("left_tokens", "right_tokens")
    @classmethod
    def validate_tokens(cls, value: int | None, info: object) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(f"{getattr(info, 'field_name', 'tokens')} must be a nonnegative integer")
        return value

    @field_validator("left_automatic_metrics", "right_automatic_metrics")
    @classmethod
    def validate_automatic_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        for name, metric in value.items():
            _nonblank(name, field="automatic metric name")
            if not math.isfinite(float(metric)):
                raise ValueError("automatic metric values must be finite")
        return value


class BlindPacket(BaseModel):
    """A rater-facing packet with no experimental identity metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    task_id: str
    report_x: str
    report_y: str

    @field_validator("packet_id", "task_id", "report_x", "report_y")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _nonblank(value, field=str(getattr(info, "field_name", "value")))


def _lookup(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return cast(object, value[name])
    else:
        for name in names:
            if hasattr(value, name):
                return cast(object, getattr(value, name))
    return default


@dataclass(frozen=True)
class _PairValues:
    task_id: str
    left_run_id: str
    right_run_id: str
    left_variant: str
    right_variant: str
    left_report: str
    right_report: str
    left_model: str | None
    right_model: str | None
    left_cost_usd: float | int | None
    right_cost_usd: float | int | None
    left_tokens: int | None
    right_tokens: int | None
    left_latency_ms: float | int | None
    right_latency_ms: float | int | None


def _pair_values(pair: object) -> _PairValues:
    """Read a ReportPair or a compatible evaluator-side mapping.

    The small compatibility reader makes migration from existing experiment
    result records painless while keeping the packet model strict.  It never
    copies arbitrary metadata into the public packet.
    """

    if isinstance(pair, ReportPair):
        return _PairValues(
            task_id=pair.task_id,
            left_run_id=pair.left_run_id,
            right_run_id=pair.right_run_id,
            left_variant=pair.left_variant,
            right_variant=pair.right_variant,
            left_report=pair.left_report,
            right_report=pair.right_report,
            left_model=pair.left_model,
            right_model=pair.right_model,
            left_cost_usd=pair.left_cost_usd,
            right_cost_usd=pair.right_cost_usd,
            left_tokens=pair.left_tokens,
            right_tokens=pair.right_tokens,
            left_latency_ms=pair.left_latency_ms,
            right_latency_ms=pair.right_latency_ms,
        )

    left = _lookup(pair, "left", "a", "variant_a", default=pair)
    right = _lookup(pair, "right", "d", "variant_d", default=pair)
    values = {
        "task_id": _lookup(pair, "task_id", "id"),
        "left_run_id": _lookup(left, "run_id", "left_run_id", "run_id_a", default=_lookup(pair, "left_run_id", "run_a_id")),
        "right_run_id": _lookup(right, "run_id", "right_run_id", "run_id_d", default=_lookup(pair, "right_run_id", "run_d_id")),
        "left_variant": _lookup(left, "variant", "left_variant", default=_lookup(pair, "left_variant", "variant_a")),
        "right_variant": _lookup(right, "variant", "right_variant", default=_lookup(pair, "right_variant", "variant_d")),
        "left_report": _lookup(left, "report", "text", "left_report", default=_lookup(pair, "left_report", "report_a")),
        "right_report": _lookup(right, "report", "text", "right_report", default=_lookup(pair, "right_report", "report_d")),
        "left_model": _lookup(left, "model", "model_id", "left_model", default=_lookup(pair, "left_model", "left_model_id")),
        "right_model": _lookup(right, "model", "model_id", "right_model", default=_lookup(pair, "right_model", "right_model_id")),
        "left_cost_usd": _lookup(left, "cost_usd", "cost", "left_cost_usd", default=_lookup(pair, "left_cost_usd")),
        "right_cost_usd": _lookup(right, "cost_usd", "cost", "right_cost_usd", default=_lookup(pair, "right_cost_usd")),
        "left_tokens": _lookup(left, "tokens", "token_count", "left_tokens", default=_lookup(pair, "left_tokens")),
        "right_tokens": _lookup(right, "tokens", "token_count", "right_tokens", default=_lookup(pair, "right_tokens")),
        "left_latency_ms": _lookup(left, "latency_ms", "latency", "left_latency_ms", default=_lookup(pair, "left_latency_ms")),
        "right_latency_ms": _lookup(right, "latency_ms", "latency", "right_latency_ms", default=_lookup(pair, "right_latency_ms")),
    }
    try:
        return _PairValues(
            task_id=_nonblank(cast(str, values["task_id"]), field="task_id"),
            left_run_id=_nonblank(cast(str, values["left_run_id"]), field="left_run_id"),
            right_run_id=_nonblank(cast(str, values["right_run_id"]), field="right_run_id"),
            left_variant=_nonblank(cast(str, values["left_variant"]), field="left_variant"),
            right_variant=_nonblank(cast(str, values["right_variant"]), field="right_variant"),
            left_report=_nonblank(cast(str, values["left_report"]), field="left_report"),
            right_report=_nonblank(cast(str, values["right_report"]), field="right_report"),
            left_model=None if values["left_model"] is None else _nonblank(cast(str, values["left_model"]), field="left_model"),
            right_model=None if values["right_model"] is None else _nonblank(cast(str, values["right_model"]), field="right_model"),
            left_cost_usd=cast(float | int | None, values["left_cost_usd"]),
            right_cost_usd=cast(float | int | None, values["right_cost_usd"]),
            left_tokens=cast(int | None, values["left_tokens"]),
            right_tokens=cast(int | None, values["right_tokens"]),
            left_latency_ms=cast(float | int | None, values["left_latency_ms"]),
            right_latency_ms=cast(float | int | None, values["right_latency_ms"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("pair does not contain the required report identities and texts") from error


def _redact(text: str, values: Sequence[object]) -> str:
    result = text
    # Long values first prevents a short identifier from partially masking a
    # longer one.  Numeric metadata is included so cost/token/latency values
    # accidentally copied into a report are not a side channel.
    candidates = sorted(
        {
            str(value)
            for value in values
            if value is not None and str(value).strip()
        },
        key=len,
        reverse=True,
    )
    for value in candidates:
        result = re.sub(re.escape(value), "[REDACTED]", result, flags=re.IGNORECASE)
    return result


def blind_pair(pair: object, *, seed: int) -> BlindPacket:
    """Create a deterministic X/Y packet without exposing the A/D mapping."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    values = _pair_values(pair)
    left_report = _redact(
        values.left_report,
        (
            values.left_run_id,
            values.right_run_id,
            values.left_variant,
            values.right_variant,
            values.left_model,
            values.right_model,
            values.left_cost_usd,
            values.right_cost_usd,
            values.left_tokens,
            values.right_tokens,
            values.left_latency_ms,
            values.right_latency_ms,
        ),
    )
    right_report = _redact(
        values.right_report,
        (
            values.left_run_id,
            values.right_run_id,
            values.left_variant,
            values.right_variant,
            values.left_model,
            values.right_model,
            values.left_cost_usd,
            values.right_cost_usd,
            values.left_tokens,
            values.right_tokens,
            values.left_latency_ms,
            values.right_latency_ms,
        ),
    )
    # A local PRNG gives stable labels without using process-global state.  A
    # stable packet ID is safe to publish because it is derived only from the
    # already-redacted payload and task ID; the private mapping is not encoded.
    swap = __import__("random").Random(seed).getrandbits(1) == 1
    report_x, report_y = (
        (right_report, left_report) if swap else (left_report, right_report)
    )
    packet_material = json.dumps(
        {
            "task_id": values.task_id,
            "report_x": report_x,
            "report_y": report_y,
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    packet_id = "blind-" + hashlib.sha256(packet_material).hexdigest()[:32]
    return BlindPacket(
        packet_id=packet_id,
        task_id=values.task_id,
        report_x=report_x,
        report_y=report_y,
    )


class HumanRating(BaseModel):
    """One pseudonymous rater's public scores for one blind packet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str
    task_id: str
    rater_id: str
    report_label: Literal["X", "Y"]
    scores: dict[str, Annotated[int, Field(strict=True, ge=1, le=5)]]
    overall_preference: Preference
    rationale: str

    @field_validator("packet_id", "task_id", "rater_id", "rationale")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return _nonblank(value, field=str(getattr(info, "field_name", "value")))

    @field_validator("scores")
    @classmethod
    def validate_scores(
        cls, value: dict[str, Annotated[int, Field(strict=True, ge=1, le=5)]]
    ) -> dict[str, Annotated[int, Field(strict=True, ge=1, le=5)]]:
        if not value:
            raise ValueError("scores must contain at least one observed dimension")
        unknown = set(value) - set(HUMAN_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown human dimensions: {sorted(unknown)}")
        if any(type(score) is not int or not 1 <= score <= 5 for score in value.values()):
            raise ValueError("human scores must be integers from 1 to 5")
        return value


class HumanRatingsValidation(BaseModel):
    """Machine-readable validation result for a proposed rating batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    raters_per_task: dict[str, int]
    errors: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()


# Short alias for callers that prefer the noun used in the plan.
ValidationReport = HumanRatingsValidation


def _coerce_rating(value: object) -> HumanRating:
    if isinstance(value, HumanRating):
        return value
    return HumanRating.model_validate(value, strict=True)


def _expected_task_ids(expected_tasks: int | Sequence[str] | None) -> tuple[str, ...] | None:
    if expected_tasks is None:
        return None
    if isinstance(expected_tasks, int):
        if expected_tasks < 1:
            raise ValueError("expected_tasks must be positive")
        return None
    if isinstance(expected_tasks, (str, bytes, bytearray)):
        raise TypeError("expected_tasks must be a sequence of task IDs")
    ids: tuple[str, ...] = tuple(expected_tasks)
    if not ids or any(type(task_id) is not str or not task_id.strip() for task_id in ids):
        raise ValueError("expected_tasks must contain non-empty task IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("expected_tasks must be unique")
    return ids


def validate_human_ratings(
    ratings: Sequence[HumanRating] | Sequence[Mapping[str, object]],
    *,
    expected_tasks: int | Sequence[str] | None = None,
    raters_per_task: int = 3,
) -> HumanRatingsValidation:
    """Validate task coverage and distinct-rater requirements without raising.

    Invalid rows are reported in ``errors`` so a batch can be audited.  The
    formal summarizer calls this function and raises when the report is not
    valid; callers that need a preflight check can inspect ``.valid`` instead.
    """

    if type(raters_per_task) is not int or raters_per_task < 1:
        raise ValueError("raters_per_task must be a positive integer")
    expected_ids = _expected_task_ids(expected_tasks)
    rows: list[HumanRating] = []
    errors: list[str] = []
    for index, raw in enumerate(ratings):
        try:
            rows.append(_coerce_rating(raw))
        except (TypeError, ValueError, ValidationError) as error:
            errors.append(f"rating {index} is invalid: {error}")
    by_task: dict[str, list[HumanRating]] = defaultdict(list)
    for row in rows:
        by_task[row.task_id].append(row)
    counts = {task_id: len(items) for task_id, items in sorted(by_task.items())}
    if expected_ids is not None:
        missing = sorted(set(expected_ids) - set(by_task))
        unexpected = sorted(set(by_task) - set(expected_ids))
        if missing:
            errors.append(f"missing expected tasks: {', '.join(missing)}")
        if unexpected:
            errors.append(f"unexpected tasks: {', '.join(unexpected)}")
    elif isinstance(expected_tasks, int) and len(by_task) != expected_tasks:
        errors.append(f"expected {expected_tasks} tasks, observed {len(by_task)}")
    for task_id, task_rows in sorted(by_task.items()):
        if len(task_rows) != raters_per_task:
            errors.append(
                f"task {task_id} requires {raters_per_task} distinct raters; observed {len(task_rows)}"
            )
        rater_ids = [row.rater_id for row in task_rows]
        if len(set(rater_ids)) != len(rater_ids):
            errors.append(f"task {task_id} requires distinct rater IDs")
        packet_ids = {row.packet_id for row in task_rows}
        if len(packet_ids) != 1:
            errors.append(f"task {task_id} must use one blind packet")
    valid = not errors and bool(by_task)
    return HumanRatingsValidation(
        valid=valid,
        raters_per_task=counts,
        errors=tuple(errors),
        task_ids=tuple(sorted(by_task)),
    )


def _normalise_automatic_metrics(
    automatic_metrics: object,
) -> dict[str, dict[str, float]]:
    if automatic_metrics is None:
        return {}
    result: dict[str, dict[str, float]] = defaultdict(dict)
    if isinstance(automatic_metrics, Mapping):
        rows = cast(Mapping[object, object], automatic_metrics).items()
        for task_id, metrics in rows:
            if not isinstance(task_id, str):
                raise TypeError("automatic metric task IDs must be strings")
            _nonblank(task_id, field="automatic metric task_id")
            if not isinstance(metrics, Mapping):
                raise TypeError("automatic metrics must map task IDs to metric mappings")
            metric_items = cast(Mapping[str, object], metrics).items()
            for name, value in metric_items:
                _nonblank(name, field="automatic metric name")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError("automatic metric values must be finite numbers")
                result[task_id][name] = float(value)
        return {task: dict(metrics) for task, metrics in result.items()}
    if not isinstance(automatic_metrics, Sequence) or isinstance(automatic_metrics, (str, bytes, bytearray)):
        raise TypeError("automatic metrics must be a mapping or a sequence of rows")
    for raw_row in cast(Sequence[object], automatic_metrics):
        if not isinstance(raw_row, Mapping):
            raise TypeError("automatic metric rows must be mappings")
        row = cast(Mapping[str, object], raw_row)
        task_id = row.get("task_id")
        metrics = row.get("metrics")
        if not isinstance(task_id, str) or not task_id.strip() or not isinstance(metrics, Mapping):
            raise ValueError("automatic metric rows require task_id and metrics")
        metric_items = cast(Mapping[str, object], metrics).items()
        for name, value in metric_items:
            if not name.strip() or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("automatic metric rows contain an invalid metric")
            if not math.isfinite(float(value)):
                raise ValueError("automatic metric values must be finite numbers")
            result[task_id][name] = float(value)
    return {task: dict(metrics) for task, metrics in result.items()}


def _ordinal_alpha(groups: Mapping[str, Sequence[int]]) -> float | None:
    values = [score for scores in groups.values() for score in scores]
    if len(values) < 2:
        return None
    observed_numerator = 0.0
    observed_denominator = 0
    for scores in groups.values():
        if len(scores) < 2:
            continue
        observed_denominator += len(scores) * (len(scores) - 1) // 2
        for index, first in enumerate(scores):
            for second in scores[index + 1 :]:
                observed_numerator += float((first - second) ** 2)
    if observed_denominator == 0:
        return None
    observed = observed_numerator / observed_denominator
    counts = Counter(values)
    denominator = len(values) * (len(values) - 1) // 2
    expected_numerator = 0.0
    categories = sorted(counts)
    for index, first in enumerate(categories):
        for second in categories[index + 1 :]:
            expected_numerator += counts[first] * counts[second] * float((first - second) ** 2)
    expected = expected_numerator / denominator if denominator else 0.0
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return float(1.0 - observed / expected)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_ranks = _average_ranks(x)
    y_ranks = _average_ranks(y)
    x_mean = math.fsum(x_ranks) / len(x_ranks)
    y_mean = math.fsum(y_ranks) / len(y_ranks)
    numerator = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x_ranks, y_ranks, strict=True))
    x_variance = math.fsum((a - x_mean) ** 2 for a in x_ranks)
    y_variance = math.fsum((b - y_mean) ** 2 for b in y_ranks)
    if x_variance == 0.0 or y_variance == 0.0:
        return None
    return float(numerator / math.sqrt(x_variance * y_variance))


class HumanSummary(BaseModel):
    """Aggregate public statistics; missing observations stay missing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_dimension_scores: dict[str, float | None]
    majority_preference: dict[Preference, int]
    majority_by_task: dict[str, Preference]
    tie_rate: float
    krippendorff_alpha: dict[str, float | None]
    auto_metric_spearman: dict[str, dict[str, float | None]]
    valid_tasks: tuple[str, ...]
    rating_count: int

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if set(self.mean_dimension_scores) != set(HUMAN_DIMENSIONS):
            raise ValueError("summary must contain every human dimension")
        if set(self.krippendorff_alpha) != set(HUMAN_DIMENSIONS):
            raise ValueError("summary must contain alpha for every human dimension")
        if set(self.auto_metric_spearman) != set(HUMAN_DIMENSIONS):
            raise ValueError("summary must contain Spearman results for every dimension")
        if set(self.majority_preference) != set(_PREFERENCES):
            raise ValueError("summary preference counts are incomplete")
        if not 0.0 <= self.tie_rate <= 1.0 or not math.isfinite(self.tie_rate):
            raise ValueError("tie_rate must be a finite proportion")
        return self

    @property
    def mean_scores(self) -> dict[str, float | None]:
        return self.mean_dimension_scores

    @property
    def spearman(self) -> dict[str, dict[str, float | None]]:
        return self.auto_metric_spearman

    @property
    def preference_counts(self) -> dict[Preference, int]:
        return self.majority_preference


def summarize_human_ratings(
    ratings: Sequence[HumanRating] | Sequence[Mapping[str, object]],
    *,
    automatic_metrics: Mapping[str, Mapping[str, float]]
    | Sequence[Mapping[str, object]]
    | None = None,
    auto_metrics: Mapping[str, Mapping[str, float]]
    | Sequence[Mapping[str, object]]
    | None = None,
    expected_tasks: int | Sequence[str] | None = None,
    raters_per_task: int = 3,
) -> HumanSummary:
    """Compute the preregistered human summary from validated ratings.

    Automatic metrics are supplied separately from the rater rows.  This keeps
    machine scores out of blind packets and permits missing metric values to
    be omitted rather than imputed.
    """

    if automatic_metrics is not None and auto_metrics is not None:
        raise ValueError("provide only one of automatic_metrics or auto_metrics")
    metric_input = automatic_metrics if automatic_metrics is not None else auto_metrics
    validation = validate_human_ratings(
        ratings,
        expected_tasks=expected_tasks,
        raters_per_task=raters_per_task,
    )
    if not validation.valid:
        detail = "; ".join(validation.errors) or "rating batch is empty"
        raise ValueError(f"human ratings require three distinct raters per task: {detail}")
    rows = [_coerce_rating(raw) for raw in ratings]
    by_task: dict[str, list[HumanRating]] = defaultdict(list)
    for row in rows:
        by_task[row.task_id].append(row)
    means: dict[str, float | None] = {}
    alpha: dict[str, float | None] = {}
    for dimension in HUMAN_DIMENSIONS:
        observed = [row.scores[dimension] for row in rows if dimension in row.scores]
        means[dimension] = None if not observed else float(math.fsum(observed) / len(observed))
        alpha[dimension] = _ordinal_alpha(
            {
                task_id: [row.scores[dimension] for row in task_rows if dimension in row.scores]
                for task_id, task_rows in by_task.items()
            }
        )

    majority_by_task: dict[str, Preference] = {}
    preference_counts: Counter[str] = Counter()
    for task_id, task_rows in sorted(by_task.items()):
        counts = Counter(row.overall_preference for row in task_rows)
        highest = max(counts.get(preference, 0) for preference in _PREFERENCES)
        winners = [preference for preference in ("X", "Y") if counts.get(preference, 0) == highest]
        if len(winners) == 1 and highest > len(task_rows) / 2:
            majority = cast(Preference, winners[0])
        else:
            majority = "TIE"
        majority_by_task[task_id] = majority
        preference_counts[majority] += 1
    task_count = len(majority_by_task)

    auto = _normalise_automatic_metrics(metric_input)
    spearman: dict[str, dict[str, float | None]] = {}
    for dimension in HUMAN_DIMENSIONS:
        dimension_result: dict[str, float | None] = {}
        for metric_name in sorted({name for metrics in auto.values() for name in metrics}):
            human_values: list[float] = []
            metric_values: list[float] = []
            for task_id, task_rows in sorted(by_task.items()):
                metric_value = auto.get(task_id, {}).get(metric_name)
                dimension_values = [row.scores[dimension] for row in task_rows if dimension in row.scores]
                if metric_value is None or not dimension_values:
                    continue
                human_values.append(float(math.fsum(dimension_values) / len(dimension_values)))
                metric_values.append(metric_value)
            dimension_result[metric_name] = _spearman(human_values, metric_values)
        spearman[dimension] = dimension_result

    return HumanSummary(
        mean_dimension_scores=means,
        majority_preference={
            "X": preference_counts.get("X", 0),
            "Y": preference_counts.get("Y", 0),
            "TIE": preference_counts.get("TIE", 0),
        },
        majority_by_task=majority_by_task,
        tie_rate=(preference_counts.get("TIE", 0) / task_count if task_count else 0.0),
        krippendorff_alpha=alpha,
        auto_metric_spearman=spearman,
        valid_tasks=tuple(sorted(by_task)),
        rating_count=len(rows),
    )


__all__ = [
    "HUMAN_DIMENSIONS",
    "BlindPacket",
    "HumanDimension",
    "HumanRating",
    "HumanRatingsValidation",
    "HumanSummary",
    "Preference",
    "ReportPair",
    "ValidationReport",
    "blind_pair",
    "summarize_human_ratings",
    "validate_human_ratings",
]
