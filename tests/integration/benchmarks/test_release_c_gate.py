from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.release_c_gate import run_c_gate
from scripts.release_preflight import GateReport


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
    experiment = tmp_path / "group"
    experiment.mkdir()
    (experiment / "summary.json").write_text("{}", encoding="utf-8")
    (experiment / "manifest.sha256").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.release_c_gate.assess_gate",
        lambda *args, **kwargs: GateReport("c4", "ready", None, ()),
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
            path.write_text("verified publication\n", encoding="utf-8")
        return "verified publication\n"

    monkeypatch.setattr("scripts.release_c_gate.render_results", fake_renderer)
    result = run_c_gate(tmp_path, stage="c4", experiment_dir=experiment)

    assert result.status == "ready"
    assert (tmp_path / "docs" / "results.md").read_text(encoding="utf-8") == "verified publication\n"
