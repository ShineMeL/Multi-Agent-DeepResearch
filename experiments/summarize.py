"""Safe aggregate outputs for formal experiment groups.

The summarizer consumes only public ``ExperimentTaskRun`` records and
hash-only evaluator reference metadata.  It never opens private gold, prompts,
model responses or credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from benchmarks.datasets.models import TaskCategory
from benchmarks.evaluators.metrics import MetricValue
from benchmarks.evaluators.pareto import analyze_pareto
from benchmarks.evaluators.statistics import (
    ConfidenceInterval,
    SeedRunRecord,
    aggregate_seeds,
    compare_planners,
    compare_rankers,
    paired_stratified_bootstrap,
)
from deepresearch.domain import ResourceUsage
from experiments.models import (
    EvaluatorReferenceManifest,
    ExperimentTaskRun,
    OracleReferenceResult,
    canonical_sha256,
)

_OUTPUTS = (
    "summary.json",
    "task_metrics.jsonl",
    "confidence_intervals.json",
    "pareto.json",
    "failures.jsonl",
    "oracle-reference.jsonl",
    "evaluator-reference-manifest.json",
)
_PROTOCOLS = ("ranker_component", "planner_policy", "end_to_end", "reference")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
        + b"\n"
    )


def _write_reusable(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise FileExistsError(path)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging")
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
        staging.rename(path)
    finally:
        staging.unlink(missing_ok=True)


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    return b"".join(_canonical(row) for row in rows)


def _load_runs(experiment_dir: Path, *, group_id: str) -> tuple[ExperimentTaskRun, ...]:
    # Keep the filename contract in one place: the runner owns the canonical
    # idempotency key construction, and the summarizer verifies that a copied
    # or renamed raw record cannot be smuggled into aggregate statistics.
    from experiments.runner import ExperimentRunner

    raw = experiment_dir / "raw"
    runs: list[ExperimentTaskRun] = []
    seen: set[str] = set()
    if not raw.is_dir():
        return ()
    for path in sorted(raw.glob("*.json")):
        key = path.stem
        if key in seen:
            raise ValueError("duplicate idempotency key")
        seen.add(key)
        run = ExperimentTaskRun.model_validate_json(path.read_bytes(), strict=True)
        expected = ExperimentRunner.idempotency_key(
            group_id,
            run.protocol,
            run.variant.value,
            run.task_id,
            run.seed,
            run.repeat_id,
            run.budget_preset,
        )
        if key != expected:
            raise ValueError("raw record filename is not the sealed idempotency key")
        runs.append(run)
    return tuple(runs)


def _safe_run_row(run: ExperimentTaskRun) -> dict[str, object]:
    return {
        "task_id": run.task_id,
        "protocol": run.protocol,
        "variant": run.variant.value,
        "planner_id": run.planner_id,
        "ranker_id": run.ranker_id,
        "budget_preset": run.budget_preset,
        "seed": run.seed,
        "repeat_id": run.repeat_id,
        "status": run.status,
        "validity": run.validity,
        "error_code": run.error_code,
        "category": None if run.category is None else run.category.value,
        "quality_metrics": {
            name: {
                "name": metric.name,
                "value": metric.value,
                "numerator": metric.numerator,
                "denominator": metric.denominator,
                "version": metric.version,
            }
            for name, metric in sorted(run.metrics.items())
        },
        "metrics": {
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
            "total_tokens": run.usage.total_tokens,
            "search_calls": run.usage.search_calls,
            "pages": run.usage.pages,
            "wall_seconds": run.usage.wall_seconds,
            "cost_usd": None if run.usage.cost_usd is None else str(run.usage.cost_usd),
        },
    }


def _resource_metric(name: str, value: float) -> MetricValue:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"resource metric {name} is invalid")
    return MetricValue(
        name=name,
        value=value,
        numerator=value,
        denominator=1.0,
        version="resource-metrics-v1",
    )


def _setup_cost_section(experiment_dir: Path) -> dict[str, object]:
    setup_root = experiment_dir / "candidate-pool-setup"
    usages: list[ResourceUsage] = []
    pricing_snapshot_ids: set[str] = set()
    if not setup_root.exists():
        return {"count": 0, "usage": {}, "pricing_snapshot_ids": []}
    if setup_root.is_symlink() or not setup_root.is_dir():
        raise ValueError("candidate pool setup root is unsafe")
    for path in sorted(setup_root.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("candidate pool setup artifact is unsafe")
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict):
            raise TypeError("candidate pool setup artifact is invalid")
        typed = cast(dict[str, object], payload)
        if set(typed) != {
            "schema_version",
            "group_id",
            "task_id",
            "pool_key",
            "candidate_pool_sha256",
            "evidence_ids_sha256",
            "manifest_sha256",
            "usage",
            "pricing_snapshot_ids",
            "pricing_status",
        }:
            raise ValueError("candidate pool setup artifact schema is invalid")
        usage = ResourceUsage.model_validate_json(_canonical(typed["usage"]), strict=True)
        if usage.cost_usd is None or typed["pricing_status"] != "estimated":
            raise ValueError("candidate pool setup cost is not verified")
        raw_pricing_ids = typed["pricing_snapshot_ids"]
        raw_pricing_items = (
            cast(list[object], raw_pricing_ids)
            if isinstance(raw_pricing_ids, list)
            else None
        )
        if (
            raw_pricing_items is None
            or any(not isinstance(item, str) or not item for item in raw_pricing_items)
        ):
            raise ValueError("candidate pool setup pricing identity is invalid")
        usages.append(usage)
        pricing_snapshot_ids.update(cast(list[str], raw_pricing_ids))
    total_fields: dict[str, object] = {
        name: sum(int(getattr(usage, name)) for usage in usages)
        for name in (
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "search_calls",
            "pages",
            "retries",
        )
    }
    total_fields["wall_seconds"] = sum(
        float(usage.wall_seconds) for usage in usages
    )
    total_fields["cost_usd"] = str(
        sum(
            (usage.cost_usd for usage in usages if usage.cost_usd is not None),
            Decimal(0),
        )
    )
    return {
        "count": len(usages),
        "usage": total_fields,
        "pricing_snapshot_ids": sorted(pricing_snapshot_ids),
    }


def _task_categories(group: Mapping[str, object]) -> dict[str, TaskCategory]:
    raw = _string_object_map(group.get("task_categories"), field="task_categories")
    categories: dict[str, TaskCategory] = {}
    for task_id, value in raw.items():
        if not isinstance(value, str):
            raise TypeError("task category must be a string")
        try:
            categories[task_id] = TaskCategory(value)
        except ValueError as error:
            raise ValueError("task category is invalid") from error
    return categories


def _seed_records(
    runs: Sequence[ExperimentTaskRun], *, categories: Mapping[str, TaskCategory]
) -> tuple[SeedRunRecord, ...]:
    records: list[SeedRunRecord] = []
    required_quality: dict[str, tuple[str, ...]] = {
        "ranker_component": (
            "citation_support_precision",
            "information_completeness",
        ),
        "planner_policy": (
            "citation_support_precision",
            "information_completeness",
            "query_redundancy",
        ),
        "end_to_end": (
            "citation_support_precision",
            "information_completeness",
        ),
        "reference": ("information_completeness",),
    }
    for run in runs:
        if run.status != "completed" or run.validity != "valid":
            raise ValueError("quality summary requires valid completed raw records")
        category = run.category or categories.get(run.task_id)
        if category is None:
            raise ValueError("task category is required for quality summary")
        missing = [name for name in required_quality[run.protocol] if name not in run.metrics]
        if missing:
            raise ValueError("raw record is missing a quality metric")
        metrics = dict(run.metrics)
        if run.usage.cost_usd is None:
            raise ValueError("raw record is missing verified cost for quality summary")
        metrics["search_calls"] = _resource_metric(
            "search_calls", float(run.usage.search_calls)
        )
        metrics["cost_usd"] = _resource_metric("cost_usd", float(run.usage.cost_usd))
        records.append(
            SeedRunRecord(
                task_id=run.task_id,
                category=category,
                variant=(
                    "P0"
                    if run.protocol == "reference"
                    else run.variant.value
                ),
                budget_preset=run.budget_preset,
                seed=run.seed,
                repeat_id=run.repeat_id,
                metrics=metrics,
            )
        )
    return tuple(records)


def _ci_payload(interval: ConfidenceInterval) -> dict[str, object]:
    return cast(dict[str, object], asdict(interval))


def _paired_ci(
    records: Sequence[SeedRunRecord],
    *,
    left_variant: str,
    right_variant: str,
    metric_name: str,
    budget: str,
    n_resamples: int,
) -> ConfidenceInterval:
    selected = [
        record
        for record in records
        if record.variant in {left_variant, right_variant}
        and record.budget_preset == budget
    ]
    aggregates = aggregate_seeds(selected, metric_name=metric_name)
    left = [row for row in aggregates if row.variant == left_variant]
    right = [row for row in aggregates if row.variant == right_variant]
    return paired_stratified_bootstrap(
        left,
        right,
        categories={row.task_id: row.category for row in aggregates},
        n_resamples=n_resamples,
        budget_preset=cast("Literal['low', 'medium', 'high']", budget),
    )


def _manifest_payload(experiment_dir: Path, files: tuple[str, ...]) -> bytes:
    hashes = {
        name: hashlib.sha256((experiment_dir / name).read_bytes()).hexdigest()
        for name in files
        if (experiment_dir / name).is_file()
    }
    return _canonical({"schema_version": "experiment-artifact-manifest-v1", "files": hashes})


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    value = math.fsum(values) / len(values)
    if not math.isfinite(value):
        raise ValueError("summary metric is not finite")
    return value


def _string_list(value: object, *, field: str, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list of strings")
    raw_values = cast(list[object], value)
    if any(type(item) is not str for item in raw_values):
        raise TypeError(f"{field} must be a list of strings")
    values = cast(list[str], raw_values)
    if nonempty and not values:
        raise ValueError(f"{field} must not be empty")
    if any(not item for item in values):
        raise ValueError(f"{field} contains an empty value")
    return values


def _string_object_map(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    raw_values = cast(dict[object, object], value)
    if any(type(key) is not str for key in raw_values):
        raise TypeError(f"{field} must be an object")
    return cast(dict[str, object], raw_values)


def _validate_coverage(group: Mapping[str, object], runs: Sequence[ExperimentTaskRun]) -> None:
    protocols = _string_list(group.get("protocols"), field="protocols")
    if tuple(protocols) != _PROTOCOLS:
        raise ValueError("experiment group protocol coverage is incomplete")
    task_map = _string_object_map(group.get("protocol_task_ids"), field="protocol_task_ids")
    variant_map = _string_object_map(group.get("expected_variants"), field="expected_variants")
    required_budgets = _string_object_map(
        group.get("required_budgets"), field="required_budgets"
    )
    if set(task_map) != set(_PROTOCOLS) or set(variant_map) != set(_PROTOCOLS):
        raise ValueError("experiment group coverage metadata is incomplete")
    if set(required_budgets) != set(_PROTOCOLS):
        raise ValueError("experiment group budget coverage is incomplete")
    all_budgets = _string_list(group.get("budgets"), field="budgets")
    if len(set(all_budgets)) != len(all_budgets):
        raise ValueError("experiment group budgets must be unique")
    if any(item not in {"low", "medium", "high"} for item in all_budgets):
        raise ValueError("experiment group contains an unknown budget")
    expected_variants: dict[str, tuple[str, ...]] = {}
    expected_tasks: dict[str, tuple[str, ...]] = {}
    expected_budgets: dict[str, tuple[str, ...]] = {}
    for protocol in _PROTOCOLS:
        tasks = _string_list(task_map.get(protocol), field=f"{protocol} tasks")
        variants = _string_list(variant_map.get(protocol), field=f"{protocol} variants")
        budgets = _string_list(
            required_budgets.get(protocol), field=f"{protocol} budgets"
        )
        if len(set(tasks)) != len(tasks) or len(set(variants)) != len(variants):
            raise ValueError("protocol task/variant coverage is not unique")
        if any(item not in all_budgets for item in budgets):
            raise ValueError("protocol budget is not part of sealed budgets")
        expected_tasks[protocol] = tuple(tasks)
        expected_variants[protocol] = tuple(variants)
        expected_budgets[protocol] = tuple(budgets)
    cost_subset_task_ids = _string_list(
        group.get("cost_subset_task_ids"), field="cost_subset_task_ids"
    )
    if len(set(cost_subset_task_ids)) != len(cost_subset_task_ids):
        raise ValueError("cost subset task coverage is not unique")
    if any(task_id not in expected_tasks["end_to_end"] for task_id in cost_subset_task_ids):
        raise ValueError("cost subset task is outside end-to-end coverage")
    expected_variant_sets = {
        "ranker_component": ("R0", "R1", "R2"),
        "planner_policy": ("A", "B", "C", "D"),
        "end_to_end": ("A", "B", "C", "D"),
        "reference": ("P0",),
    }
    if any(
        expected_variants[protocol] != expected_variant_sets[protocol]
        for protocol in _PROTOCOLS
    ):
        raise ValueError("protocol variant coverage is invalid")
    replication = _string_object_map(group.get("replication"), field="replication")
    mode = replication.get("mode")
    if mode == "seeds":
        if replication.get("seed_supported") is not True:
            raise ValueError("seed replication support metadata is invalid")
        seeds_value = replication.get("seed_values")
        if not isinstance(seeds_value, list):
            raise TypeError("experiment group seed coverage is invalid")
        raw_seeds = cast(list[object], seeds_value)
        if any(type(seed) is not int for seed in raw_seeds):
            raise TypeError("experiment group seed coverage is invalid")
        seeds = cast(list[int], raw_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("experiment group seed coverage is invalid")
        replication_values: set[tuple[str, int]] = {("seed", seed) for seed in seeds}
    elif mode == "independent_repeats":
        if replication.get("seed_supported") is not False:
            raise ValueError("repeat replication support metadata is invalid")
        repeats_value = replication.get("repeat_ids")
        if not isinstance(repeats_value, list):
            raise TypeError("experiment group repeat coverage is invalid")
        raw_repeats = cast(list[object], repeats_value)
        if any(type(item) is not int for item in raw_repeats):
            raise TypeError("experiment group repeat coverage is invalid")
        repeats = cast(list[int], raw_repeats)
        if not repeats or len(set(repeats)) != len(repeats) or any(item < 1 for item in repeats):
            raise ValueError("experiment group repeat coverage is invalid")
        replication_values = {("repeat", item) for item in repeats}
    else:
        raise ValueError("experiment group replication metadata is invalid")
    group_id = group.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("experiment group ID is invalid")
    identities: set[tuple[str, str, str, str, tuple[str, int]]] = set()
    for run in runs:
        if run.protocol not in _PROTOCOLS:
            raise ValueError("run protocol is outside the sealed coverage")
        if run.task_id not in expected_tasks[run.protocol]:
            raise ValueError("run task is outside the sealed coverage")
        if run.variant.value not in expected_variants[run.protocol]:
            raise ValueError("run variant is outside the sealed coverage")
        if run.budget_preset not in all_budgets:
            raise ValueError("run budget is outside the sealed coverage")
        is_budget_sensitivity = (
            run.protocol == "end_to_end"
            and run.variant.value == "D"
            and run.task_id in cost_subset_task_ids
        )
        if run.budget_preset not in expected_budgets[run.protocol] and not is_budget_sensitivity:
            raise ValueError("run budget is outside the sealed coverage")
        replication_value: tuple[str, int]
        if run.seed is not None:
            replication_value = ("seed", run.seed)
        elif run.repeat_id is not None:
            replication_value = ("repeat", run.repeat_id)
        else:  # guarded by the ExperimentTaskRun model, retained for clarity
            raise ValueError("run replication is incomplete")
        if replication_value not in replication_values:
            raise ValueError("run replication is outside the sealed coverage")
        identity = (
            run.protocol,
            run.variant.value,
            run.task_id,
            run.budget_preset,
            replication_value,
        )
        if identity in identities:
            raise ValueError("duplicate task/variant/budget/replication identity")
        identities.add(identity)
    for protocol in _PROTOCOLS:
        for task_id in expected_tasks[protocol]:
            for variant in expected_variants[protocol]:
                for budget in expected_budgets[protocol]:
                    for replication_value in replication_values:
                        expected = (protocol, variant, task_id, budget, replication_value)
                        if expected not in identities:
                            raise ValueError("experiment task/variant/replication coverage is incomplete")
    for task_id in cost_subset_task_ids:
        for budget in all_budgets:
            if budget in expected_budgets["end_to_end"]:
                continue
            for replication_value in replication_values:
                expected = ("end_to_end", "D", task_id, budget, replication_value)
                if expected not in identities:
                    raise ValueError("budget sensitivity coverage is incomplete")


def _sections(runs: Sequence[ExperimentTaskRun]) -> dict[str, dict[str, object]]:
    grouped: defaultdict[str, list[ExperimentTaskRun]] = defaultdict(list)
    for run in runs:
        grouped[run.protocol].append(run)
    result: dict[str, dict[str, object]] = {}
    for protocol in _PROTOCOLS:
        items = grouped.get(protocol, [])
        metric_values = {
            name: [float(getattr(run.usage, name)) for run in items]
            for name in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "search_calls",
                "pages",
                "wall_seconds",
            )
        }
        costs = [float(run.usage.cost_usd) for run in items if run.usage.cost_usd is not None]
        if costs:
            metric_values["cost_usd"] = costs
        for name in sorted({metric_name for item in items for metric_name in item.metrics}):
            metric_values[name] = [item.metrics[name].value for item in items]
        result[protocol] = {
            "run_count": len(items),
            "task_count": len({item.task_id for item in items}),
            "task_ids": sorted({item.task_id for item in items}),
            "variants": dict(sorted(Counter(item.variant.value for item in items).items())),
            "budgets": dict(sorted(Counter(item.budget_preset for item in items).items())),
            "statuses": dict(sorted(Counter(item.status for item in items).items())),
            "metrics": {name: {"mean": _mean(values), "n": len(values)} for name, values in metric_values.items()},
        }
    return result


def _confidence_sections(
    records_by_protocol: Mapping[str, Sequence[SeedRunRecord]],
    *,
    n_resamples: int,
    budget: str,
) -> dict[str, object]:
    ranker = list(records_by_protocol["ranker_component"])
    ranker_ci = compare_rankers(
        ranker,
        budget_preset=cast("Literal['low', 'medium', 'high']", budget),
        n_resamples=n_resamples,
    )

    planner_sections: dict[str, object] = {}
    for ranker_id, left_variant, right_variant in (
        ("R1", "A", "C"),
        ("R2", "B", "D"),
    ):
        selected = [
            record
            for record in records_by_protocol["planner_policy"]
            if record.variant in {left_variant, right_variant}
        ]
        # Re-label the two fixed-ranker arms to the Task 13 planner protocol.
        relabeled = tuple(
            record.model_copy(
                update={"variant": "P1" if record.variant == left_variant else "P2"}
            )
            for record in selected
        )
        comparison = compare_planners(
            relabeled,
            budget_preset=cast("Literal['low', 'medium', 'high']", budget),
            n_resamples=n_resamples,
        )
        planner_sections[ranker_id] = {
            "baseline": left_variant,
            "candidate": right_variant,
            "completeness": _ci_payload(comparison.completeness),
            "non_inferior": comparison.non_inferior,
            "search_calls_reduction": (
                None
                if comparison.search_calls_reduction is None
                else _ci_payload(comparison.search_calls_reduction)
            ),
            "query_redundancy_reduction": (
                None
                if comparison.query_redundancy_reduction is None
                else _ci_payload(comparison.query_redundancy_reduction)
            ),
        }

    end_to_end = [
        record
        for record in records_by_protocol["end_to_end"]
        if record.variant in {"A", "D"}
    ]
    end_ci = _paired_ci(
        end_to_end,
        left_variant="A",
        right_variant="D",
        metric_name="information_completeness",
        budget=budget,
        n_resamples=n_resamples,
    )
    return {
        "ranker_component": {
            "baseline": "R1",
            "candidate": "R2",
            "metric": "citation_support_precision",
            "confidence_interval": _ci_payload(ranker_ci),
        },
        "planner_policy": {
            "comparisons": planner_sections,
        },
        "end_to_end": {
            "baseline": "A",
            "candidate": "D",
            "metric": "information_completeness",
            "confidence_interval": _ci_payload(end_ci),
        },
        "reference": {
            "status": "not_applicable",
            "reason": "ORACLE is evaluator-only",
        },
    }


def _pareto_sections(
    records_by_protocol: Mapping[str, Sequence[SeedRunRecord]],
    *,
    n_resamples: int,
    budget: str,
) -> dict[str, object]:
    selected_budget = cast("Literal['low', 'medium', 'high']", budget)
    ranker = list(records_by_protocol["ranker_component"])
    ranker_results = analyze_pareto(
        ranker,
        baseline="R1",
        candidate="R2",
        budget_preset=selected_budget,
        n_resamples=n_resamples,
    )
    planner_results: dict[str, object] = {}
    for ranker_id, left_variant, right_variant in (("R1", "A", "C"), ("R2", "B", "D")):
        planner_records = [
            record
            for record in records_by_protocol["planner_policy"]
            if record.variant in {left_variant, right_variant}
        ]
        planner_results[ranker_id] = [
            asdict(result)
            for result in analyze_pareto(
                planner_records,
                baseline=left_variant,
                candidate=right_variant,
                budget_preset=selected_budget,
                n_resamples=n_resamples,
            )
        ]
    end_to_end = [
        record
        for record in records_by_protocol["end_to_end"]
        if record.variant in {"A", "D"}
    ]
    end_results = analyze_pareto(
        end_to_end,
        baseline="A",
        candidate="D",
        budget_preset=selected_budget,
        n_resamples=n_resamples,
    )
    return {
        "ranker_component": [asdict(result) for result in ranker_results],
        "planner_policy": planner_results,
        "end_to_end": [asdict(result) for result in end_results],
        "reference": {"status": "not_applicable", "reason": "ORACLE is evaluator-only"},
    }


def _reference_metadata(
    experiment_dir: Path,
    *,
    group_id: str,
    expected_task_ids: Sequence[str],
    expected_private_manifest_sha256: str,
    expected_evaluator_version: str,
) -> dict[str, object]:
    oracle_path = experiment_dir / "oracle-reference.jsonl"
    manifest_path = experiment_dir / "evaluator-reference-manifest.json"
    if oracle_path.exists() != manifest_path.exists():
        raise ValueError("reference artifacts are incomplete")
    if not oracle_path.exists():
        return {"status": "not_present"}
    results: list[OracleReferenceResult] = []
    for line in oracle_path.read_bytes().splitlines():
        results.append(OracleReferenceResult.model_validate_json(line, strict=True))
    if not results or len({item.task_id for item in results}) != len(results):
        raise ValueError("oracle reference results must be non-empty and task-unique")
    if {item.task_id for item in results} != set(expected_task_ids):
        raise ValueError("oracle reference task coverage is incomplete")
    manifest = EvaluatorReferenceManifest.model_validate_json(manifest_path.read_bytes(), strict=True)
    ordered = sorted(results, key=lambda item: item.task_id)
    if (
        manifest.group_id != group_id
        or manifest.private_manifest_sha256 != expected_private_manifest_sha256
        or manifest.evaluator_version != expected_evaluator_version
    ):
        raise ValueError("oracle reference group identity mismatch")
    if manifest.oracle_results_sha256 != canonical_sha256(
        [item.model_dump(mode="json") for item in ordered]
    ):
        raise ValueError("oracle reference hash mismatch")
    if manifest.task_ids_sha256 != canonical_sha256(tuple(item.task_id for item in ordered)):
        raise ValueError("oracle reference task identity hash mismatch")
    for result in ordered:
        if (
            result.private_manifest_sha256 != manifest.private_manifest_sha256
            or result.evaluator_version != manifest.evaluator_version
            or result.created_at != manifest.created_at
        ):
            raise ValueError("oracle reference evaluator identity mismatch")
    return {
        "status": "present",
        "count": len(results),
        "oracle_results_sha256": manifest.oracle_results_sha256,
        "task_ids_sha256": manifest.task_ids_sha256,
        "evaluator_version": manifest.evaluator_version,
        "incomparable_to_agent_cost_trace": True,
    }


def _verify_manifest(experiment_dir: Path) -> None:
    manifest_path = experiment_dir / "manifest.sha256"
    if not manifest_path.is_file():
        raise ValueError("experiment manifest is missing")
    payload = json.loads(manifest_path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError("experiment manifest is invalid")
    raw_manifest = cast(dict[str, object], payload)
    if raw_manifest.get("schema_version") != "experiment-artifact-manifest-v1":
        raise ValueError("experiment manifest is invalid")
    files = raw_manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("experiment manifest file list is invalid")
    typed_files = cast(dict[object, object], files)
    if set(typed_files) != set(_OUTPUTS):
        raise ValueError("experiment manifest is incomplete")
    for name, expected in typed_files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, str)
            or len(expected) != 64
            or expected != expected.lower()
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("experiment manifest contains unsafe entries")
        path = experiment_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("experiment artifact hash mismatch")


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is invalid") from error
    return _string_object_map(payload, field=label)


def _load_jsonl_objects(path: Path, *, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise ValueError(f"{label} is invalid") from error
    for line in lines:
        try:
            rows.append(_string_object_map(json.loads(line), field=label))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} is invalid") from error
    return rows


def _verify_output_sections(
    experiment_dir: Path,
    *,
    group_id: str,
    runs: Sequence[ExperimentTaskRun],
    reference: Mapping[str, object],
    setup_cost: Mapping[str, object],
) -> None:
    summary = _load_json_object(experiment_dir / "summary.json", label="summary")
    if (
        summary.get("schema_version") != "experiment-summary-v1"
        or summary.get("group_id") != group_id
        or summary.get("run_count") != len(runs)
    ):
        raise ValueError("summary coverage is invalid")
    sections = _string_object_map(summary.get("sections"), field="summary sections")
    if set(sections) != set(_PROTOCOLS):
        raise ValueError("summary protocol sections are incomplete")
    for protocol in _PROTOCOLS:
        section = _string_object_map(sections.get(protocol), field=f"{protocol} summary")
        run_count = section.get("run_count")
        if not isinstance(run_count, int) or run_count <= 0:
            raise ValueError("summary protocol section is empty")
        metrics = _string_object_map(section.get("metrics"), field=f"{protocol} metrics")
        if not metrics:
            raise ValueError("summary metrics section is empty")
    reference_section = _string_object_map(sections["reference"], field="reference summary")
    if reference_section.get("reference_artifacts") != dict(reference):
        raise ValueError("summary reference metadata is inconsistent")
    ranker_section = _string_object_map(
        sections["ranker_component"], field="ranker summary"
    )
    if ranker_section.get("protocol_setup_cost") != dict(setup_cost):
        raise ValueError("ranker setup cost summary is inconsistent")

    task_rows = _load_jsonl_objects(
        experiment_dir / "task_metrics.jsonl", label="task metrics"
    )
    expected_rows = [_safe_run_row(run) for run in runs]
    if task_rows != expected_rows:
        raise ValueError("task metrics do not match raw records")

    confidence = _load_json_object(
        experiment_dir / "confidence_intervals.json", label="confidence intervals"
    )
    if confidence.get("schema_version") != "confidence-intervals-v1":
        raise ValueError("confidence interval schema is invalid")
    confidence_sections = _string_object_map(
        confidence.get("sections"), field="confidence sections"
    )
    if set(confidence_sections) != set(_PROTOCOLS):
        raise ValueError("confidence interval sections are incomplete")
    for protocol in _PROTOCOLS:
        section = _string_object_map(
            confidence_sections.get(protocol), field="confidence section"
        )
        if not section:
            raise ValueError("confidence interval section is empty")
        if protocol == "ranker_component":
            interval = _string_object_map(
                section.get("confidence_interval"), field="ranker confidence interval"
            )
            if section.get("metric") != "citation_support_precision" or not {
                "estimate",
                "lower",
                "upper",
                "n_tasks",
                "n_resamples",
            }.issubset(interval):
                raise ValueError("ranker confidence interval is invalid")
        elif protocol == "planner_policy":
            comparisons = _string_object_map(
                section.get("comparisons"), field="planner confidence comparisons"
            )
            if set(comparisons) != {"R1", "R2"}:
                raise ValueError("planner confidence comparisons are incomplete")
        elif protocol == "end_to_end":
            interval = _string_object_map(
                section.get("confidence_interval"), field="end-to-end confidence interval"
            )
            if section.get("metric") != "information_completeness" or not {
                "estimate",
                "lower",
                "upper",
                "n_tasks",
                "n_resamples",
            }.issubset(interval):
                raise ValueError("end-to-end confidence interval is invalid")

    pareto = _load_json_object(experiment_dir / "pareto.json", label="pareto")
    if pareto.get("schema_version") != "pareto-v1":
        raise ValueError("pareto schema is invalid")
    pareto_sections = _string_object_map(pareto.get("sections"), field="pareto sections")
    if set(pareto_sections) != set(_PROTOCOLS):
        raise ValueError("pareto sections are incomplete")
    for protocol in _PROTOCOLS:
        value = pareto_sections.get(protocol)
        if protocol in {"ranker_component", "end_to_end"}:
            if not isinstance(value, list) or not value:
                raise ValueError("pareto section is empty")
        elif protocol == "planner_policy":
            planner = _string_object_map(value, field="planner pareto section")
            if set(planner) != {"R1", "R2"}:
                raise ValueError("planner pareto section is incomplete")
        elif not _string_object_map(value, field="reference pareto section"):
            raise ValueError("pareto section is empty")

    failure_rows = _load_jsonl_objects(
        experiment_dir / "failures.jsonl", label="failure records"
    )
    expected_failures = [
        {
            "task_id": run.task_id,
            "protocol": run.protocol,
            "variant": run.variant.value,
            "error_code": run.error_code,
            "status": run.status,
        }
        for run in runs
        if run.status != "completed" or run.validity != "valid"
    ]
    if failure_rows != expected_failures:
        raise ValueError("failure records do not match raw records")


def summarize_experiment(
    experiment_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    verify_only: bool = False,
) -> dict[str, object]:
    """Build or verify the safe aggregate files for one group."""
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    root = Path(experiment_dir).resolve()
    group_path = root / "group.json"
    if not group_path.is_file():
        raise ValueError("experiment group metadata is missing")
    group = json.loads(group_path.read_bytes())
    if not isinstance(group, dict):
        raise TypeError("experiment group metadata is invalid")
    raw_group = cast(dict[str, object], group)
    if not isinstance(raw_group.get("group_id"), str):
        raise TypeError("experiment group metadata is invalid")
    group_id = cast(str, group["group_id"])
    private_manifest_sha256 = raw_group.get("private_manifest_sha256")
    evaluator_version = raw_group.get("evaluator_version")
    if not isinstance(private_manifest_sha256, str) or not isinstance(evaluator_version, str):
        raise TypeError("experiment group evaluator identity is incomplete")
    oracle_task_ids = _string_list(raw_group.get("oracle_task_ids"), field="oracle_task_ids")
    if len(set(oracle_task_ids)) != len(oracle_task_ids):
        raise ValueError("oracle task coverage is not unique")
    end_to_end_tasks = _string_object_map(
        raw_group.get("protocol_task_ids"), field="protocol_task_ids"
    ).get("end_to_end")
    if any(
        task_id not in _string_list(end_to_end_tasks, field="end_to_end tasks")
        for task_id in oracle_task_ids
    ):
        raise ValueError("oracle task is outside end-to-end coverage")
    runs = _load_runs(root, group_id=group_id)
    _validate_coverage(raw_group, runs)
    categories = _task_categories(raw_group)
    records_by_protocol = {
        protocol: _seed_records(
            [run for run in runs if run.protocol == protocol], categories=categories
        )
        for protocol in _PROTOCOLS
    }
    required_budgets = _string_object_map(
        raw_group.get("required_budgets"), field="required_budgets"
    )
    primary_budget_values = _string_list(
        required_budgets.get("end_to_end"), field="end_to_end budgets"
    )
    if len(primary_budget_values) != 1:
        raise ValueError("exactly one primary budget is required for summary")
    primary_budget = primary_budget_values[0]
    setup_cost = _setup_cost_section(root)
    if verify_only:
        _verify_manifest(root)
        reference = _reference_metadata(
            root,
            group_id=group_id,
            expected_task_ids=oracle_task_ids,
            expected_private_manifest_sha256=private_manifest_sha256,
            expected_evaluator_version=evaluator_version,
        )
        if reference.get("status") != "present":
            raise ValueError("oracle reference artifacts are missing")
        _verify_output_sections(
            root,
            group_id=group_id,
            runs=runs,
            reference=reference,
            setup_cost=setup_cost,
        )
        summary = _load_json_object(root / "summary.json", label="summary")
        return {**summary, "verified": True}

    sections = _sections(runs)
    reference = _reference_metadata(
        root,
        group_id=group_id,
        expected_task_ids=oracle_task_ids,
        expected_private_manifest_sha256=private_manifest_sha256,
        expected_evaluator_version=evaluator_version,
    )
    if reference.get("status") != "present":
        raise ValueError("oracle reference artifacts are missing")
    sections["reference"]["reference_artifacts"] = reference
    sections["ranker_component"]["protocol_setup_cost"] = setup_cost
    confidence_sections = _confidence_sections(
        records_by_protocol,
        n_resamples=bootstrap_resamples,
        budget=primary_budget,
    )
    pareto_sections = _pareto_sections(
        records_by_protocol,
        n_resamples=bootstrap_resamples,
        budget=primary_budget,
    )
    task_rows = [_safe_run_row(run) for run in runs]
    failures = [
        {
            "task_id": run.task_id,
            "protocol": run.protocol,
            "variant": run.variant.value,
            "error_code": run.error_code,
            "status": run.status,
        }
        for run in runs
        if run.status != "completed" or run.validity != "valid"
    ]
    summary: dict[str, object] = {
        "schema_version": "experiment-summary-v1",
        "group_id": group_id,
        "bootstrap_resamples": bootstrap_resamples,
        "sections": sections,
        "run_count": len(runs),
        "reference_note": "ORACLE is evaluator-only and incomparable to agent cost/latency traces.",
        "coverage": {
            "protocols": raw_group["protocols"],
            "replication": raw_group["replication"],
        },
    }
    outputs = {
        "summary.json": _canonical(summary),
        "task_metrics.jsonl": _jsonl(task_rows),
        "confidence_intervals.json": _canonical(
            {
                "schema_version": "confidence-intervals-v1",
                "sections": confidence_sections,
            }
        ),
        "pareto.json": _canonical(
            {"schema_version": "pareto-v1", "sections": pareto_sections}
        ),
        "failures.jsonl": _jsonl(failures),
    }
    for name, payload in outputs.items():
        _write_reusable(root / name, payload)
    manifest_payload = _manifest_payload(root, _OUTPUTS)
    _write_reusable(root / "manifest.sha256", manifest_payload)
    return summary


__all__ = ["summarize_experiment"]
