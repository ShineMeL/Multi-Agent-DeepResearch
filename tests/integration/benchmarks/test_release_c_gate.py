from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from experiments.summarize import summarize_experiment
from scripts.release_c_gate import run_c_gate
from scripts.release_preflight import GateReport
from tests.unit.experiments.test_summarize import (  # pyright: ignore[reportPrivateUsage]
    _write_full_group,
)

_PUBLICATION_PATHS = (
    "results.md",
    "assets/results/abcd-metrics.svg",
    "assets/results/citation-support-vs-usd.svg",
    "assets/results/completeness-vs-search.svg",
)


def _write_self_hashed_primary_experiment(root: Path) -> Path:
    experiment = root / "group"
    experiment.mkdir()
    summary = {
        "schema_version": "experiment-summary-v1",
        "group_id": "fixture-group",
        "dataset_version": "frozen-ai-cs-60-v1",
        "evaluation_date": "2026-09-07",
        "ranker_component": {
            "baseline": "R1",
            "candidate": "R2",
            "metric": "citation_support_precision",
            "confidence_interval": {"estimate": 0.02, "lower": 0.01, "upper": 0.03},
            "metrics": {"citation_support_precision": {"mean": 0.82, "n": 4}},
        },
        "planner_policy": {
            "comparisons": {
                "R1": {
                    "baseline": "A",
                    "candidate": "C",
                    "non_inferior": True,
                    "completeness": {"estimate": 0.01, "lower": -0.01, "upper": 0.03},
                },
                "R2": {
                    "baseline": "B",
                    "candidate": "D",
                    "non_inferior": True,
                    "completeness": {"estimate": 0.01, "lower": -0.01, "upper": 0.03},
                },
            },
            "metrics": {"search_calls": {"mean": 3.0, "n": 4}},
        },
        "end_to_end": {
            "baseline": "A",
            "candidate": "D",
            "metric": "information_completeness",
            "confidence_interval": {"estimate": 0.01, "lower": -0.01, "upper": 0.03},
            "metrics": {"information_completeness": {"mean": 0.8, "n": 4}},
        },
        "reference": {"status": "not_present", "reason": "ORACLE is evaluator-only"},
    }
    summary_path = experiment / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    (experiment / "manifest.sha256").write_text(
        json.dumps(
            {
                "schema_version": "experiment-artifact-manifest-v1",
                "files": {"summary.json": hashlib.sha256(summary_path.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    return experiment


def _write_valid_primary_experiment(root: Path) -> Path:
    experiment = _write_full_group(root / "group")
    summarize_experiment(experiment, bootstrap_resamples=10_000)
    return experiment


def _write_placeholder_publication(root: Path) -> dict[str, bytes]:
    previous: dict[str, bytes] = {}
    for relative in _PUBLICATION_PATHS:
        path = root / "docs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            b"Factual outcome: primary result is not yet sealed.\n"
            if relative == "results.md"
            else f"old:{relative}\n".encode()
        )
        path.write_bytes(content)
        previous[relative] = content
    return previous


def _make_windows_junction(link: Path, target: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows junction regression")
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("Windows junction creation is unavailable")
    assert link.is_junction()


def test_c1_does_not_create_partial_formal_config(tmp_path: Path) -> None:
    output = tmp_path / "benchmarks" / "configs" / "formal.yaml"

    result = run_c_gate(tmp_path, stage="c1")

    assert result.status == "blocked"
    assert result.reason == "FORMAL_INPUT_MISSING"
    assert not output.exists()


def test_c2_never_launches_provider_when_formal_config_is_absent(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, ...]] = []

    def recording_runner(*args: Any, **kwargs: Any) -> None:
        calls.append(tuple(args))
        raise AssertionError("formal gate must fail before command delegation")

    result = run_c_gate(
        tmp_path,
        stage="c2",
        experiment_dir=tmp_path / "group",
        runner=recording_runner,
    )

    assert result.status == "blocked"
    assert result.reason == "FORMAL_INPUT_MISSING"
    assert calls == []


def test_c3_requires_separate_external_and_human_sidecars(tmp_path: Path) -> None:
    result = run_c_gate(tmp_path, stage="c3")

    assert result.status == "blocked"
    assert result.reason in {"PORTFOLIO_INPUT_MISSING", "HUMAN_AGGREGATE_INCOMPLETE"}


def test_c4_keeps_unsealed_results_unchanged(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "results.md"
    results.parent.mkdir(parents=True)
    results.write_text(
        "Factual outcome: primary result is not yet sealed.\n", encoding="utf-8"
    )
    before = results.read_bytes()

    result = run_c_gate(tmp_path, stage="c4")

    assert result.status == "blocked"
    assert result.reason == "PUBLICATION_UNSEALED"
    assert results.read_bytes() == before


def test_c4_promotes_verified_results_over_the_expected_placeholder(tmp_path: Path) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    _write_placeholder_publication(tmp_path)

    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "ready"
    published = (tmp_path / "docs" / "results.md").read_text(encoding="utf-8")
    assert "primary result is not yet sealed" not in published.casefold()
    assert "Public artifact seal verified: yes" in published


def test_c4_rejects_a_summary_that_only_hashes_itself(tmp_path: Path) -> None:
    experiment = _write_self_hashed_primary_experiment(tmp_path)
    previous = _write_placeholder_publication(tmp_path)

    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "blocked"
    assert result.reason == "PUBLICATION_UNSEALED"
    assert {
        relative: (tmp_path / "docs" / relative).read_bytes()
        for relative in _PUBLICATION_PATHS
    } == previous


def test_c4_rejects_a_formal_summary_without_10_000_bootstrap_resamples(
    tmp_path: Path,
) -> None:
    experiment = _write_full_group(tmp_path / "group")
    summarize_experiment(experiment, bootstrap_resamples=4)
    previous = _write_placeholder_publication(tmp_path)

    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "blocked"
    assert result.reason == "PUBLICATION_UNSEALED"
    assert {
        relative: (tmp_path / "docs" / relative).read_bytes()
        for relative in _PUBLICATION_PATHS
    } == previous


def test_c4_rejects_a_hash_sealed_human_summary_without_20x3_ratings(tmp_path: Path) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    previous = _write_placeholder_publication(tmp_path)
    human = tmp_path / "human-summary.json"
    human.write_text(json.dumps({"rating_count": 999}), encoding="utf-8")
    (tmp_path / "manifest.sha256").write_text(
        json.dumps(
            {
                "schema_version": "benchmark-aggregate-manifest-v1",
                "files": {human.name: hashlib.sha256(human.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )

    result = run_c_gate(
        tmp_path,
        stage="c4",
        experiment_dir=experiment,
        human_summary=human,
    )

    assert result.status == "blocked"
    assert result.reason == "PUBLICATION_UNSEALED"
    assert {
        relative: (tmp_path / "docs" / relative).read_bytes()
        for relative in _PUBLICATION_PATHS
    } == previous


def test_c4_restores_every_existing_publication_file_when_promotion_fails(
    tmp_path: Path,
) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    previous = _write_placeholder_publication(tmp_path)
    newly_created = tmp_path / "docs" / _PUBLICATION_PATHS[2]
    newly_created.unlink()
    previous.pop(_PUBLICATION_PATHS[2])
    from scripts import release_c_gate

    real_replace = release_c_gate.os.replace
    calls = 0

    def fail_second_replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected publication failure")
        real_replace(source, target)

    with patch("scripts.release_c_gate.os.replace", fail_second_replace):
        result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "failed"
    assert result.reason == "PUBLICATION_VALIDATION_FAILED"
    assert {
        relative: (tmp_path / "docs" / relative).read_bytes() for relative in previous
    } == previous
    assert not newly_created.exists()


def test_c4_continues_rollback_and_retains_backups_when_rollback_fails(
    tmp_path: Path,
) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    previous = _write_placeholder_publication(tmp_path)
    from scripts import release_c_gate

    real_replace = release_c_gate.os.replace
    calls = 0

    def fail_promotion_then_one_rollback(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls in {4, 5}:
            raise OSError("injected replace failure")
        real_replace(source, target)

    with patch("scripts.release_c_gate.os.replace", fail_promotion_then_one_rollback):
        result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "failed"
    assert result.reason == "PUBLICATION_VALIDATION_FAILED"
    assert (tmp_path / "docs" / _PUBLICATION_PATHS[0]).read_bytes() == previous[
        _PUBLICATION_PATHS[0]
    ]
    assert (tmp_path / "docs" / _PUBLICATION_PATHS[1]).read_bytes() == previous[
        _PUBLICATION_PATHS[1]
    ]
    assert (tmp_path / "docs" / _PUBLICATION_PATHS[3]).read_bytes() == previous[
        _PUBLICATION_PATHS[3]
    ]
    recovery_directories = list(tmp_path.glob(".deepresearch-publication-*"))
    assert len(recovery_directories) == 1
    assert (
        recovery_directories[0] / "rollback" / _PUBLICATION_PATHS[2]
    ).read_bytes() == previous[_PUBLICATION_PATHS[2]]


def test_c4_never_promotes_complete_deterministic_output_with_an_unsealed_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    previous = _write_placeholder_publication(tmp_path)

    def unsealed_renderer(*args: Any, **kwargs: Any) -> str:
        docs = Path(kwargs["docs_dir"])
        for relative in _PUBLICATION_PATHS:
            path = docs / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "Factual outcome: primary result is not yet sealed.\n",
                encoding="utf-8",
            )
        return "primary result is not yet sealed"

    monkeypatch.setattr("scripts.release_c_gate.render_results", unsealed_renderer)

    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "blocked"
    assert result.reason == "PUBLICATION_UNSEALED"
    assert {
        relative: (tmp_path / "docs" / relative).read_bytes()
        for relative in _PUBLICATION_PATHS
    } == previous


@pytest.mark.parametrize(
    "results_page",
    [
        "",
        (
            "> Factual outcome: publication verification pending.\n\n"
            "- Public artifact seal verified: yes.\n"
        ),
    ],
)
def test_c4_rejects_complete_staging_without_an_allowlisted_sealed_outcome(
    tmp_path: Path,
    monkeypatch,
    results_page: str,
) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    previous = _write_placeholder_publication(tmp_path)

    def pending_renderer(*args: Any, **kwargs: Any) -> str:
        docs = Path(kwargs["docs_dir"])
        for relative in _PUBLICATION_PATHS:
            path = docs / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(results_page if relative == "results.md" else "<svg/>\n")
        return results_page

    monkeypatch.setattr("scripts.release_c_gate.render_results", pending_renderer)

    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "blocked"
    assert result.reason == "PUBLICATION_UNSEALED"
    assert {
        relative: (tmp_path / "docs" / relative).read_bytes()
        for relative in _PUBLICATION_PATHS
    } == previous


def test_c4_prepares_replacements_on_the_publication_filesystem(tmp_path: Path) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    _write_placeholder_publication(tmp_path)
    from scripts import release_c_gate

    real_replace = release_c_gate.os.replace

    def reject_cross_filesystem_source(source: Path, target: Path) -> None:
        if not Path(source).is_relative_to(tmp_path) or not Path(target).is_relative_to(tmp_path):
            raise OSError("simulated cross-filesystem replace")
        real_replace(source, target)

    with patch("scripts.release_c_gate.os.replace", reject_cross_filesystem_source):
        result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "ready"
    assert "primary result is sealed" in (tmp_path / "docs" / "results.md").read_text(
        encoding="utf-8"
    )


def test_c4_rejects_a_publication_parent_symlink_before_replacing_any_target(
    tmp_path: Path,
) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    results = docs / "results.md"
    old_results = b"Factual outcome: primary result is not yet sealed.\n"
    results.write_bytes(old_results)
    outside_results = tmp_path / "outside-assets" / "results"
    outside_results.mkdir(parents=True)
    outside_before: dict[str, bytes] = {}
    for relative in _PUBLICATION_PATHS[1:]:
        path = outside_results / Path(relative).name
        content = f"old:{relative}\n".encode()
        path.write_bytes(content)
        outside_before[path.name] = content
    try:
        (docs / "assets").symlink_to(outside_results.parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "failed"
    assert result.reason == "PUBLICATION_VALIDATION_FAILED"
    assert results.read_bytes() == old_results
    assert {
        path.name: path.read_bytes() for path in outside_results.iterdir() if path.is_file()
    } == outside_before


def test_c4_rejects_a_publication_parent_junction_before_replacing_any_target(
    tmp_path: Path,
) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    results = docs / "results.md"
    old_results = b"Factual outcome: primary result is not yet sealed.\n"
    results.write_bytes(old_results)
    outside_assets = tmp_path / "outside-assets"
    outside_results = outside_assets / "results"
    outside_results.mkdir(parents=True)
    outside_before: dict[str, bytes] = {}
    for relative in _PUBLICATION_PATHS[1:]:
        path = outside_results / Path(relative).name
        content = f"old:{relative}\n".encode()
        path.write_bytes(content)
        outside_before[path.name] = content
    junction = docs / "assets"
    _make_windows_junction(junction, outside_assets)
    try:
        result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

        assert result.status == "failed"
        assert result.reason == "PUBLICATION_VALIDATION_FAILED"
        assert results.read_bytes() == old_results
        assert {
            path.name: path.read_bytes() for path in outside_results.iterdir() if path.is_file()
        } == outside_before
    finally:
        os.rmdir(junction)


def test_c_result_details_do_not_include_input_payloads(tmp_path: Path) -> None:
    marker = tmp_path / "docs" / "results.md"
    marker.parent.mkdir(parents=True)
    marker.write_text("primary result is not sealed", encoding="utf-8")
    secret = "operator-private-rating-rationale"
    marker.write_text(f"{secret}: primary result is not sealed", encoding="utf-8")

    result = run_c_gate(tmp_path, stage="c4")

    assert secret not in json.dumps(result, default=lambda value: value.__dict__)


def test_c2_delegates_fixed_formal_protocol_in_order(tmp_path: Path, monkeypatch) -> None:
    formal = tmp_path / "benchmarks" / "configs"
    formal.mkdir(parents=True)
    (formal / "formal.yaml").write_text("sealed: true\n", encoding="utf-8")
    experiment = tmp_path / "group"
    experiment.mkdir()
    monkeypatch.setattr(
        "scripts.release_c_gate.assess_gate",
        lambda *args, **kwargs: GateReport("c2", "ready", None, ()),
    )
    commands: list[list[str]] = []

    def fake_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(list(args[0]))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    result = run_c_gate(tmp_path, stage="c2", experiment_dir=experiment, runner=fake_runner)

    assert result.status == "ready"
    assert [step.name for step in result.steps] == [
        "formal_validate",
        "ranker_execution",
        "planner_execution",
        "abcd_execution",
        "stability_execution",
        "cost_subset_execution",
        "reference_execution",
        "formal_summary",
        "formal_summary_verify",
    ]
    assert "10000" in commands[-1]
    assert "A,B,C,D" in commands[3]
    assert "P0,ORACLE" in commands[6]


def test_c4_promotes_only_byte_stable_renderer_output(tmp_path: Path, monkeypatch) -> None:
    experiment = _write_valid_primary_experiment(tmp_path)
    _write_placeholder_publication(tmp_path)
    publication = (
        "> Factual outcome: primary result is sealed.\n\n"
        "- Public artifact seal verified: yes.\n"
    )

    def fake_renderer(*args: Any, **kwargs: Any) -> str:
        docs = Path(kwargs["docs_dir"])
        for relative in (
            "results.md",
            "evaluation.md",
            "assets/results/abcd-metrics.svg",
            "assets/results/citation-support-vs-usd.svg",
            "assets/results/completeness-vs-search.svg",
        ):
            path = docs / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(publication, encoding="utf-8")
        return publication

    monkeypatch.setattr("scripts.release_c_gate.render_results", fake_renderer)
    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "ready"
    assert (tmp_path / "docs" / "results.md").read_text(encoding="utf-8") == publication
