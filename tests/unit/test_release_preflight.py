from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.release_preflight import GateReport, assess_gate, main


def test_b1_is_blocked_without_docker(tmp_path: Path) -> None:
    report = assess_gate(
        tmp_path,
        "b1",
        environ={"DATABASE_URL": "postgresql+asyncpg://user:p%40ss@db/research"},
        command_exists=lambda _name: False,
    )
    assert report.status == "blocked"
    assert report.reason == "DEPLOYMENT_PREREQUISITE_MISSING"
    assert all("p%40ss" not in check.detail for check in report.checks)


def test_b2_reports_presence_without_secret_values(tmp_path: Path) -> None:
    env = {
        "MODEL_API_KEY": "new-secret-value",
        "SEARCH_API_KEY": "another-secret-value",
        "SESSION_SIGNING_KEY": "a" * 32,
        "PROVIDER_PROFILE_CATALOG_PATH": str(tmp_path / "providers.json"),
        "PRICING_CATALOG_PATH": str(tmp_path / "pricing.json"),
    }
    (tmp_path / "providers.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pricing.json").write_text("{}", encoding="utf-8")
    report = assess_gate(tmp_path, "b2", environ=env, command_exists=lambda _name: True)
    encoded = json.dumps(asdict(report), sort_keys=True)
    assert "new-secret-value" not in encoded
    assert "another-secret-value" not in encoded
    assert "present" in encoded
    assert report.reason == "ONLINE_SMOKE_INCOMPLETE"


def test_c1_requires_formal_inputs_and_lf_fixture_bytes(tmp_path: Path) -> None:
    report = assess_gate(tmp_path, "c1", environ={}, command_exists=lambda _name: True)
    assert report.status == "blocked"
    assert report.reason == "FORMAL_INPUT_MISSING"


def test_c4_is_not_ready_when_results_are_unsealed(tmp_path: Path) -> None:
    results = tmp_path / "docs" / "results.md"
    results.parent.mkdir(parents=True)
    results.write_text(
        "Factual outcome: primary result is not yet sealed.\n", encoding="utf-8"
    )
    report = assess_gate(tmp_path, "c4", environ={}, command_exists=lambda _name: True)
    assert report.status == "blocked"
    assert report.reason == "PUBLICATION_UNSEALED"


def test_cli_json_is_stable_and_secret_free(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--repository",
            str(tmp_path),
            "--gate",
            "b2",
            "--format",
            "json",
        ]
    )
    assert exit_code == 1
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["gate"] == "b2"
    assert payload["status"] == "blocked"
    assert "secret" not in output.out.casefold()
    assert "secret" not in output.err.casefold()


def test_gate_report_is_frozen() -> None:
    report = GateReport("b1", "ready", None, ())
    try:
        report.status = "blocked"  # type: ignore[misc]
    except Exception as error:  # noqa: BLE001
        assert type(error).__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("GateReport must be immutable")
