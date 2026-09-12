"""Fail-closed Release B deployment and online-smoke gate.

The gate is an orchestration boundary only.  It does not implement service
behavior and it never places credential values in command arguments or result
details.  A capability preflight always runs before Docker, pytest, or HTTP
work is attempted.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.error import URLError
from urllib.request import urlopen

# ``python scripts/release_b_gate.py`` is a documented invocation.  Python
# otherwise puts only ``scripts/`` on ``sys.path`` for a direct file launch.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_preflight import GateReport, assess_gate

Profile = Literal["replay", "online-smoke"]
BStepStatus = Literal["ready", "blocked", "failed"]
BRunStatus = Literal["ready", "blocked", "failed"]


@dataclass(frozen=True)
class BStep:
    """One redacted operation in the deployment gate."""

    name: str
    status: BStepStatus
    detail: str


@dataclass(frozen=True)
class BRunResult:
    """Immutable, secret-free result of a Release B run."""

    status: BRunStatus
    reason: str | None
    steps: tuple[BStep, ...]


Runner = Callable[..., subprocess.CompletedProcess[str] | None]
HealthGetter = Callable[..., object]

_ZERO_REVISION = "0" * 40
_ONLINE_JSON_VARIABLES = {
    "PRICING_CATALOG_JSON": "PRICING_CATALOG_PATH",
    "PROVIDER_PROFILE_CATALOG_JSON": "PROVIDER_PROFILE_CATALOG_PATH",
}


def _blocked(report: GateReport) -> BRunResult:
    steps = tuple(
        BStep(check.name, "ready" if check.present else "blocked", check.detail)
        for check in report.checks
    )
    return BRunResult("blocked", report.reason, steps)


def _safe_revision(repository: Path, *, environ: Mapping[str, str]) -> str:
    """Return a validated 40-character revision, or the explicit unavailable value.

    Source archives used by unit tests do not have Git metadata.  Packaged
    builds still fail their own provenance check when the unavailable value is
    used; a real checkout is always required to provide a validated revision.
    """

    candidate = environ.get("GITHUB_SHA", "").strip()
    if (
        len(candidate) == 40
        and candidate != _ZERO_REVISION
        and all(character in "0123456789abcdefABCDEF" for character in candidate)
    ):
        return candidate.lower()
    git_marker = repository / ".git"
    if not git_marker.exists():
        return _ZERO_REVISION
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return _ZERO_REVISION
    revision = result.stdout.strip()
    if (
        result.returncode == 0
        and len(revision) == 40
        and revision != _ZERO_REVISION
        and all(character in "0123456789abcdefABCDEF" for character in revision)
    ):
        return revision.lower()
    return _ZERO_REVISION


def _invoke(
    runner: Runner,
    command: Sequence[str],
    repository: Path,
    *,
    environ: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str] | None:
    kwargs: dict[str, object] = {
        "cwd": str(repository),
        "check": False,
        "capture_output": True,
        "text": True,
    }
    if environ is not None:
        kwargs["env"] = dict(environ)
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return runner(list(command), **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None


def _command_detail(result: subprocess.CompletedProcess[str] | None) -> str:
    if result is None:
        return "unavailable"
    if result.returncode == 0:
        return "completed"
    return f"exit_{result.returncode}"


def _host_port(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name) or str(default)
    if not value.isascii() or not value.isdecimal() or len(value) > 5:
        raise ValueError("invalid host port")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("invalid host port")
    return port


def _wait_for_health(health_getter: HealthGetter, url: str) -> bool:
    deadline = time.monotonic() + 30.0
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            response = health_getter(url, timeout=min(10.0, remaining))
            status = getattr(response, "status", None)
            if isinstance(status, int) and 200 <= status < 300:
                return True
        except (OSError, URLError, TimeoutError, ValueError):
            pass
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.5, remaining))
    return False


def _run_b1(
    repository: Path,
    *,
    keep_up: bool,
    runner: Runner,
    health_getter: HealthGetter,
    environ: Mapping[str, str],
    dry_run: bool,
) -> BRunResult:
    try:
        api_port = _host_port(environ, "API_HOST_PORT", 8000)
        ui_port = _host_port(environ, "UI_HOST_PORT", 8501)
    except ValueError:
        return BRunResult("blocked", "DEPLOYMENT_PORT_INVALID", ())
    # Keep probes and Compose on the same explicit ports even if a local .env
    # file contains different defaults. Never mutate the caller's environment.
    prepared = dict(environ, API_HOST_PORT=str(api_port), UI_HOST_PORT=str(ui_port))
    api_url = f"http://127.0.0.1:{api_port}"
    ui_url = f"http://127.0.0.1:{ui_port}"
    if dry_run:
        commands = (
            "docker compose config --quiet",
            "docker compose build --build-arg DEEPRESEARCH_CODE_COMMIT=<validated>",
            "docker compose up -d --no-build --wait --wait-timeout 180",
            "GET /health/live",
            "GET /health/ready",
            "python -m scripts.smoke_replay --api-url <local-api> --ui-url <local-ui> --timeout 90",
            "docker compose down",
        )
        if keep_up:
            commands = commands[:-1]
        return BRunResult(
            "blocked",
            "DRY_RUN_NOT_EXECUTED",
            tuple(
                BStep(f"plan_{index}", "blocked", command)
                for index, command in enumerate(commands, 1)
            ),
        )

    steps: list[BStep] = []
    stack_started = False
    final_status: BRunStatus = "ready"
    reason: str | None = None

    def run_command(
        name: str,
        command: Sequence[str],
        *,
        timeout: float = 60.0,
        failure_reason: str = "DEPLOYMENT_COMMAND_FAILED",
    ) -> bool:
        nonlocal final_status, reason
        result = _invoke(runner, command, repository, environ=prepared, timeout=timeout)
        if result is not None and result.returncode == 0:
            steps.append(BStep(name, "ready", "completed"))
            return True
        steps.append(BStep(name, "failed", _command_detail(result)))
        final_status = "failed"
        reason = failure_reason
        return False

    try:
        if run_command("compose_config", ("docker", "compose", "config", "--quiet")):
            revision = _safe_revision(repository, environ=prepared)
            if revision == _ZERO_REVISION:
                steps.append(BStep("source_revision", "failed", "unavailable"))
                final_status = "failed"
                reason = "DEPLOYMENT_SOURCE_REVISION_MISSING"
            elif run_command(
                "image_build",
                (
                    "docker",
                    "compose",
                    "build",
                    "--build-arg",
                    f"DEEPRESEARCH_CODE_COMMIT={revision}",
                ),
                timeout=1200.0,
            ):
                # Compose can partially create services before returning a
                # non-zero status, so cleanup is attempted after every up
                # attempt unless the operator explicitly keeps the stack up.
                stack_started = True
                if run_command(
                    "stack_up",
                    (
                        "docker",
                        "compose",
                        "up",
                        "-d",
                        "--no-build",
                        "--wait",
                        "--wait-timeout",
                        "180",
                    ),
                    timeout=240.0,
                ):
                    for name, path in (
                        ("health_live", "/health/live"),
                        ("health_ready", "/health/ready"),
                    ):
                        if _wait_for_health(health_getter, f"{api_url}{path}"):
                            steps.append(BStep(name, "ready", "http_2xx"))
                            continue
                        steps.append(BStep(name, "failed", "readiness_deadline_exceeded"))
                        final_status = "failed"
                        reason = "DEPLOYMENT_HEALTH_FAILED"
                        break
                    else:
                        run_command(
                            "replay_smoke",
                            (
                                sys.executable,
                                "-m",
                                "scripts.smoke_replay",
                                "--api-url",
                                api_url,
                                "--ui-url",
                                ui_url,
                                "--timeout",
                                "90",
                            ),
                            timeout=100.0,
                            failure_reason="DEPLOYMENT_REPLAY_SMOKE_FAILED",
                        )
    finally:
        if stack_started and not keep_up:
            result = _invoke(
                runner, ("docker", "compose", "down"), repository, environ=prepared, timeout=60.0
            )
            if result is not None and result.returncode == 0:
                steps.append(BStep("stack_down", "ready", "completed"))
            else:
                steps.append(BStep("stack_down", "failed", _command_detail(result)))
                final_status = "failed"
                reason = "DEPLOYMENT_COMMAND_FAILED"
    return BRunResult(final_status, reason, tuple(steps))


def _health_get(url: str, *, timeout: float) -> object:
    with urlopen(url, timeout=timeout) as response:
        response.read(1)
        return response


def _write_catalog_json(directory: Path, variable: str, payload: str) -> Path:
    token = secrets.token_hex(8)
    path = directory / f"{variable.casefold()}-{token}.json"
    path.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    return path


def _run_b2(
    repository: Path,
    *,
    runner: Runner,
    environ: Mapping[str, str],
    dry_run: bool,
) -> BRunResult:
    with tempfile.TemporaryDirectory(prefix="deepresearch-online-") as temporary:
        prepared = dict(environ)
        temporary_root = Path(temporary)
        for json_variable, path_variable in _ONLINE_JSON_VARIABLES.items():
            payload = prepared.get(json_variable, "")
            if payload.strip():
                path = _write_catalog_json(temporary_root, path_variable, payload)
                prepared[path_variable] = str(path)
        report = assess_gate(repository, "b2", environ=prepared)
        if report.status != "ready":
            return _blocked(report)
        if dry_run:
            return BRunResult(
                "blocked",
                "DRY_RUN_NOT_EXECUTED",
                (
                    BStep(
                        "online_smoke_plan",
                        "blocked",
                        "uv run pytest -q tests/integration/deployment/test_smoke.py -m online",
                    ),
                ),
            )
        result = _invoke(
            runner,
            (
                "uv",
                "run",
                "pytest",
                "-q",
                "tests/integration/deployment/test_smoke.py",
                "-m",
                "online",
            ),
            repository,
            environ=prepared,
        )
        if result is not None and result.returncode == 0:
            return BRunResult("ready", None, (BStep("online_smoke", "ready", "completed"),))
        return BRunResult(
            "failed",
            "ONLINE_SMOKE_FAILED",
            (BStep("online_smoke", "failed", _command_detail(result)),),
        )


def run_b_gate(
    repository: Path,
    *,
    profile: Profile,
    keep_up: bool = False,
    runner: Runner = subprocess.run,
    health_getter: HealthGetter = _health_get,
    environ: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> BRunResult:
    """Run one Release B profile after a secret-safe capability preflight."""

    root = repository.resolve()
    current_env = dict(environ if environ is not None else os.environ)
    if profile == "replay":
        report = assess_gate(root, "b1", environ=current_env)
        if report.status != "ready":
            return _blocked(report)
        return _run_b1(
            root,
            keep_up=keep_up,
            runner=runner,
            health_getter=health_getter,
            environ=current_env,
            dry_run=dry_run,
        )
    return _run_b2(root, runner=runner, environ=current_env, dry_run=dry_run)


def _payload(result: BRunResult) -> dict[str, object]:
    return cast(dict[str, object], asdict(result))


def _parse_profile(value: str) -> Profile:
    if value not in {"replay", "online-smoke"}:
        raise argparse.ArgumentTypeError("profile must be replay or online-smoke")
    return cast(Profile, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--profile", type=_parse_profile, default="replay")
    parser.add_argument("--keep-up", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    arguments = parser.parse_args(argv)
    result = run_b_gate(
        arguments.repository,
        profile=arguments.profile,
        keep_up=arguments.keep_up,
        dry_run=arguments.dry_run,
    )
    if arguments.format == "json":
        print(
            json.dumps(_payload(result), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    else:
        print(f"release_b status={result.status} reason={result.reason or 'none'}")
        for step in result.steps:
            print(f"  step={step.name} status={step.status} detail={step.detail}")
    return 0 if result.status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BRunResult", "BStep", "Profile", "main", "run_b_gate"]
