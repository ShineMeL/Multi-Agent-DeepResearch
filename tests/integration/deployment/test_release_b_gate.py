from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.release_b_gate import BRunResult, run_b_gate
from scripts.release_preflight import GateReport


def test_b1_stops_before_docker_when_preflight_is_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "scripts.release_b_gate.assess_gate",
        lambda *args, **kwargs: GateReport(
            "b1", "blocked", "DEPLOYMENT_PREREQUISITE_MISSING", ()
        ),
    )

    result = run_b_gate(
        tmp_path,
        profile="replay",
        runner=lambda *args, **kwargs: calls.append(args),
    )

    assert result.status == "blocked"
    assert result.reason == "DEPLOYMENT_PREREQUISITE_MISSING"
    assert calls == []


def test_b1_runs_config_build_up_health_and_cleanup_in_order(
    tmp_path: Path, monkeypatch
) -> None:
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.release_b_gate.assess_gate",
        lambda *args, **kwargs: GateReport("b1", "ready", None, ()),
    )
    calls: list[tuple[object, ...]] = []

    def fake_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def fake_health(url: str, *, timeout: float) -> None:
        assert url in {"http://127.0.0.1:8000/health/live", "http://127.0.0.1:8000/health/ready"}
        assert timeout > 0

    result = run_b_gate(
        tmp_path,
        profile="replay",
        runner=fake_runner,
        health_getter=fake_health,
    )

    assert result.status == "ready"
    assert [step.name for step in result.steps] == [
        "compose_config",
        "image_build",
        "stack_up",
        "health_live",
        "health_ready",
        "stack_down",
    ]
    assert calls[0][0][0:3] == ["docker", "compose", "config"]
    assert calls[1][0][0:2] == ["docker", "build"]


def test_b2_requires_complete_catalog_and_does_not_echo_keys(
    tmp_path: Path, monkeypatch
) -> None:
    secret_one = "secret-that-must-not-print"
    secret_two = "another-secret-that-must-not-print"
    monkeypatch.setenv("MODEL_API_KEY", secret_one)
    monkeypatch.setenv("SEARCH_API_KEY", secret_two)
    monkeypatch.setenv("SESSION_SIGNING_KEY", "a" * 32)
    providers = tmp_path / "providers.json"
    pricing = tmp_path / "pricing.json"
    providers.write_text("{}", encoding="utf-8")
    pricing.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROVIDER_PROFILE_CATALOG_PATH", str(providers))
    monkeypatch.setenv("PRICING_CATALOG_PATH", str(pricing))
    calls: list[tuple[object, ...]] = []

    result = run_b_gate(
        tmp_path,
        profile="online-smoke",
        runner=lambda *args, **kwargs: calls.append(args),
    )

    payload = json.dumps(result, default=lambda value: value.__dict__)
    assert secret_one not in payload
    assert secret_two not in payload
    assert calls == []
    assert result.status == "blocked"
    assert result.reason == "ONLINE_SMOKE_INCOMPLETE"


def test_result_is_immutable() -> None:
    result = BRunResult("ready", None, ())
    try:
        result.status = "failed"  # type: ignore[misc]
    except Exception as error:  # noqa: BLE001
        assert type(error).__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("BRunResult must be immutable")


def test_b1_attempts_cleanup_after_partial_stack_start(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.release_b_gate.assess_gate",
        lambda *args, **kwargs: GateReport("b1", "ready", None, ()),
    )
    commands: list[list[str]] = []

    def fake_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command = list(args[0])
        commands.append(command)
        failed = command[:3] == ["docker", "compose", "up"]
        return subprocess.CompletedProcess(
            args=args, returncode=1 if failed else 0, stdout="", stderr=""
        )

    result = run_b_gate(tmp_path, profile="replay", runner=fake_runner)

    assert result.status == "failed"
    assert commands[-1] == ["docker", "compose", "down"]
    assert result.steps[-1].name == "stack_down"
