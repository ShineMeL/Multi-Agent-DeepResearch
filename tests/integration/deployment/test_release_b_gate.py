from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import release_b_gate
from scripts.release_b_gate import BRunResult, run_b_gate
from scripts.release_preflight import GateReport


@pytest.fixture
def ready_gate(monkeypatch):
    monkeypatch.setattr(
        release_b_gate, "assess_gate", lambda *args, **kwargs: GateReport("b1", "ready", None, ())
    )
    monkeypatch.setattr(release_b_gate, "_safe_revision", lambda *args, **kwargs: "a" * 40)


def test_b1_stops_before_docker_when_preflight_is_blocked(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "scripts.release_b_gate.assess_gate",
        lambda *args, **kwargs: GateReport("b1", "blocked", "DEPLOYMENT_PREREQUISITE_MISSING", ()),
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
    tmp_path: Path, monkeypatch, ready_gate
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

    def fake_health(url: str, *, timeout: float) -> object:
        assert url in {"http://127.0.0.1:8000/health/live", "http://127.0.0.1:8000/health/ready"}
        assert timeout > 0
        return SimpleNamespace(status=200)

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
        "replay_smoke",
        "stack_down",
    ]
    assert calls[0][0][0:3] == ["docker", "compose", "config"]
    assert calls[1][0] == [
        "docker",
        "compose",
        "build",
        "--build-arg",
        f"DEEPRESEARCH_CODE_COMMIT={'a' * 40}",
    ]
    assert "--no-build" in calls[2][0]
    assert "--wait" in calls[2][0]


def test_b2_requires_complete_catalog_and_does_not_echo_keys(tmp_path: Path, monkeypatch) -> None:
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
    tmp_path: Path, monkeypatch, ready_gate
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


def test_b1_forwards_validated_environment_and_checks_configured_ports(tmp_path, ready_gate):
    commands = []
    health_urls = []
    environment = {
        "API_HOST_PORT": "18003",
        "UI_HOST_PORT": "18503",
        "SESSION_SIGNING_KEY": "test-secret",
    }

    def runner(command, **kwargs):
        assert kwargs["env"] == environment
        assert 0 < kwargs["timeout"] <= 1200
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def health(url, **kwargs):
        health_urls.append(url)
        return SimpleNamespace(status=200)

    result = run_b_gate(
        tmp_path, profile="replay", runner=runner, health_getter=health, environ=environment
    )

    assert result.status == "ready"
    assert health_urls == [
        "http://127.0.0.1:18003/health/live",
        "http://127.0.0.1:18003/health/ready",
    ]
    smoke_command = commands[-2]
    assert smoke_command[1:3] == ["-m", "scripts.smoke_replay"]
    assert smoke_command[3:] == [
        "--api-url",
        "http://127.0.0.1:18003",
        "--ui-url",
        "http://127.0.0.1:18503",
        "--timeout",
        "90",
    ]
    assert "test-secret" not in repr(result)


@pytest.mark.parametrize("never_ready", [False, True])
def test_b1_waits_for_health_with_a_bounded_deadline(
    tmp_path, monkeypatch, ready_gate, never_ready
):
    now = [0.0]
    commands = []
    attempts = []

    def sleep(delay):
        assert 0 < delay <= 1
        now[0] += delay

    monkeypatch.setattr(
        release_b_gate,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], sleep=sleep),
        raising=False,
    )

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    def health(url, *, timeout):
        assert 0 < timeout <= 10
        attempts.append(url)
        if never_ready or len(attempts) == 1:
            raise TimeoutError("sensitive transport details")
        return SimpleNamespace(status=200)

    result = run_b_gate(tmp_path, profile="replay", runner=runner, health_getter=health)

    assert len(attempts) > 1
    assert 0 < now[0] <= 30
    assert commands[-1] == ["docker", "compose", "down"]
    assert "sensitive transport details" not in repr(result)
    if never_ready:
        assert result.reason == "DEPLOYMENT_HEALTH_FAILED"
        assert not any("scripts.smoke_replay" in command for command in commands)
    else:
        assert result.status == "ready"
        assert any("scripts.smoke_replay" in command for command in commands)


def test_b1_report_failure_cannot_pass_on_healthy_probes(tmp_path, ready_gate):
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1 if "scripts.smoke_replay" in command else 0)

    result = run_b_gate(
        tmp_path,
        profile="replay",
        runner=runner,
        health_getter=lambda *args, **kwargs: SimpleNamespace(status=200),
    )

    assert result.status == "failed"
    assert result.reason == "DEPLOYMENT_REPLAY_SMOKE_FAILED"
    assert commands[-1] == ["docker", "compose", "down"]
    assert any(step.name == "replay_smoke" and step.status == "failed" for step in result.steps)


@pytest.mark.parametrize("port", ["0", "65536", "abc", "12/path", "-1"])
def test_b1_invalid_port_stops_before_external_commands(tmp_path, ready_gate, port):
    commands = []
    result = run_b_gate(
        tmp_path,
        profile="replay",
        environ={"API_HOST_PORT": port},
        runner=lambda *args, **kwargs: commands.append(args),
    )
    assert result.status == "blocked"
    assert result.reason == "DEPLOYMENT_PORT_INVALID"
    assert commands == []


def test_b1_unavailable_revision_stops_before_build(tmp_path, monkeypatch, ready_gate):
    monkeypatch.setattr(release_b_gate, "_safe_revision", lambda *args, **kwargs: "0" * 40)
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    result = run_b_gate(tmp_path, profile="replay", runner=runner)

    assert result.status == "failed"
    assert result.reason == "DEPLOYMENT_SOURCE_REVISION_MISSING"
    assert commands == [["docker", "compose", "config", "--quiet"]]
