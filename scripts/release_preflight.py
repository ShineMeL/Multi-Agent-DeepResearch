"""Fail-closed capability checks for Release B and Release C.

The preflight command is deliberately small and dependency-free.  It reports
only capability presence and stable relative labels; it never prints secret
values, command output, or provider responses.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

GateName = Literal["b1", "b2", "c1", "c2", "c3", "c4"]
GateStatus = Literal["ready", "blocked", "skipped"]
_GateArgument = Literal["b1", "b2", "c1", "c2", "c3", "c4", "all"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One secret-safe capability observation."""

    name: str
    present: bool
    detail: str


@dataclass(frozen=True, slots=True)
class GateReport:
    """The deterministic result for one release gate."""

    gate: GateName
    status: GateStatus
    reason: str | None
    checks: tuple[CheckResult, ...]


_HASHED_ROOTS = (
    "benchmarks/snapshots/frozen_ai_cs_60",
    "tests/fixtures/frozen_corpus",
    "tests/fixtures/replay",
)
_C1_REQUIRED_FILES = (
    "benchmarks/configs/qwen3-8b.lock.json",
    "benchmarks/configs/inference-environment.lock.json",
    "models/embedding.lock.json",
    "benchmarks/private/frozen_ai_cs_60/private_manifest.json",
    "benchmarks/datasets/frozen_ai_cs_60/public_manifest.json",
    "benchmarks/configs/formal.template.yaml",
)


def _default_command_exists(name: str) -> bool:
    if name == "docker compose":
        if shutil.which("docker") is None:
            return False
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0
    return shutil.which(name) is not None


def _file_check(repository: Path, relative: str) -> CheckResult:
    path = repository / relative
    try:
        present = path.is_file() and not path.is_symlink()
    except OSError:
        present = False
    return CheckResult(relative, present, "present" if present else "missing")


def _directory_check(repository: Path, relative: str) -> CheckResult:
    path = repository / relative
    try:
        present = path.is_dir() and not path.is_symlink()
    except OSError:
        present = False
    return CheckResult(relative, present, "present" if present else "missing")


def _command_check(name: str, command_exists: Callable[[str], object]) -> CheckResult:
    present = bool(command_exists(name))
    return CheckResult(name, present, "available" if present else "missing")


def _secret_presence(environ: Mapping[str, str], name: str) -> CheckResult:
    value = environ.get(name, "")
    present = bool(value.strip())
    if name == "SESSION_SIGNING_KEY" and present:
        present = len(value.encode("utf-8")) >= 32
    return CheckResult(name, present, "present" if present else "missing")


def _catalog_check(repository: Path, environ: Mapping[str, str], variable: str) -> CheckResult:
    configured = environ.get(variable, "").strip()
    if not configured:
        return CheckResult(variable, False, "missing")
    path = Path(configured)
    if not path.is_absolute():
        path = repository / path
    try:
        present = path.is_file() and not path.is_symlink()
    except OSError:
        present = False
    if not present:
        return CheckResult(variable, False, "missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError):
        return CheckResult(variable, False, "invalid_json")
    valid = isinstance(payload, dict) and isinstance(payload.get("profiles"), dict)
    return CheckResult(variable, valid, "valid" if valid else "profile_missing")


def _catalog_profile_check(
    repository: Path, environ: Mapping[str, str], variable: str, profile: str
) -> CheckResult:
    base = _catalog_check(repository, environ, variable)
    if not base.present:
        return base
    configured = environ[variable].strip()
    path = Path(configured)
    if not path.is_absolute():
        path = repository / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles = payload.get("profiles")
        present = isinstance(profiles, dict) and profile in profiles
    except (OSError, UnicodeError, TypeError, ValueError):
        present = False
    return CheckResult(variable, present, "profile_present" if present else "profile_missing")


def _database_check(environ: Mapping[str, str]) -> CheckResult:
    value = environ.get("DATABASE_URL", "").strip().casefold()
    present = value.startswith(("postgresql+asyncpg://", "postgresql://"))
    return CheckResult("DATABASE_URL", present, "configured" if present else "missing_or_invalid")


def _git_clean_check(repository: Path) -> CheckResult:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return CheckResult("result_affecting_git_tree", False, "unavailable")
    clean = result.returncode == 0 and not result.stdout.strip()
    return CheckResult("result_affecting_git_tree", clean, "clean" if clean else "dirty")


def _hashed_fixture_files(repository: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for relative_root in _HASHED_ROOTS:
        root = repository / relative_root
        if not root.is_dir():
            continue
        try:
            files.extend(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
        except OSError:
            continue
    return tuple(files)


def _lf_check(repository: Path) -> CheckResult:
    try:
        offending = any(b"\r\n" in path.read_bytes() for path in _hashed_fixture_files(repository))
    except OSError:
        return CheckResult("hash_addressed_fixture_lf", False, "unreadable")
    return CheckResult(
        "hash_addressed_fixture_lf",
        not offending,
        "clean" if not offending else "crlf_detected",
    )


def _result_marker(repository: Path) -> bool:
    path = repository / "docs" / "results.md"
    try:
        text = path.read_text(encoding="utf-8").casefold()
    except (OSError, UnicodeError):
        return False
    return "not yet sealed" in text or "not sealed" in text


def _report(
    gate: GateName,
    checks: Sequence[CheckResult],
    *,
    blocked_reason: str,
    skipped_reason: str | None = None,
) -> GateReport:
    frozen_checks = tuple(checks)
    if all(check.present for check in frozen_checks):
        return GateReport(gate, "ready", None, frozen_checks)
    if skipped_reason is not None and not any(check.present for check in frozen_checks):
        return GateReport(gate, "skipped", skipped_reason, frozen_checks)
    return GateReport(gate, "blocked", blocked_reason, frozen_checks)


def _assess_b1(
    repository: Path,
    environ: Mapping[str, str],
    command_exists: Callable[[str], object],
) -> GateReport:
    checks = (
        _command_check("docker", command_exists),
        _command_check("docker compose", command_exists),
        _file_check(repository, "docker-compose.yml"),
        _database_check(environ),
    )
    return _report("b1", checks, blocked_reason="DEPLOYMENT_PREREQUISITE_MISSING")


def _assess_b2(repository: Path, environ: Mapping[str, str]) -> GateReport:
    secret_checks = tuple(_secret_presence(environ, name) for name in (
        "MODEL_API_KEY",
        "SEARCH_API_KEY",
        "SESSION_SIGNING_KEY",
    ))
    if not all(check.present for check in secret_checks):
        return GateReport(
            "b2",
            "blocked",
            "ONLINE_SMOKE_NOT_AUTHORIZED",
            secret_checks,
        )
    catalog_checks = (
        _catalog_profile_check(repository, environ, "PROVIDER_PROFILE_CATALOG_PATH", "online-smoke"),
        _catalog_profile_check(repository, environ, "PRICING_CATALOG_PATH", "online-smoke"),
    )
    return _report(
        "b2",
        (*secret_checks, *catalog_checks),
        blocked_reason="ONLINE_SMOKE_INCOMPLETE",
    )


def _assess_c1(
    repository: Path,
    command_exists: Callable[[str], object],
) -> GateReport:
    checks = tuple(_file_check(repository, relative) for relative in _C1_REQUIRED_FILES)
    if not all(check.present for check in checks):
        return GateReport("c1", "blocked", "FORMAL_INPUT_MISSING", checks)
    tree_check = _git_clean_check(repository)
    lf_check = _lf_check(repository)
    tool_check = _command_check("python", command_exists)
    all_checks = (*checks, tree_check, lf_check, tool_check)
    if not tree_check.present:
        return GateReport("c1", "blocked", "FORMAL_TREE_DIRTY", all_checks)
    return _report("c1", all_checks, blocked_reason="FORMAL_INPUT_MISSING")


def _assess_c2(repository: Path, experiment_dir: Path | None) -> GateReport:
    checks = [
        _file_check(repository, "benchmarks/configs/formal.yaml"),
        _directory_check(repository, "experiments"),
    ]
    if experiment_dir is not None:
        try:
            present = experiment_dir.is_dir() and not experiment_dir.is_symlink()
        except OSError:
            present = False
        checks.append(CheckResult("experiment_dir", present, "present" if present else "missing"))
    return _report("c2", checks, blocked_reason="FORMAL_INPUT_MISSING")


def _assess_c3(
    repository: Path,
    external_experiment_dir: Path | None,
    human_summary: Path | None,
) -> GateReport:
    checks = [
        _file_check(repository, "benchmarks/configs/external.yaml"),
        _file_check(repository, "benchmarks/external/external.lock.json"),
        _file_check(repository, "benchmarks/configs/formal-portfolio.yaml"),
    ]
    if external_experiment_dir is None:
        checks.extend(
            CheckResult(name, False, "missing")
            for name in ("external_metrics", "external_manifest")
        )
    else:
        checks.extend(
            _file_check(external_experiment_dir, relative)
            for relative in ("metrics.json", "manifest.sha256")
        )
    if human_summary is None:
        checks.extend(
            CheckResult(name, False, "missing")
            for name in ("human_summary", "human_summary.manifest.sha256")
        )
    else:
        checks.append(CheckResult("human_summary", human_summary.is_file(), "present" if human_summary.is_file() else "missing"))
        sidecar = human_summary.with_name(human_summary.name + ".sha256")
        checks.append(CheckResult("human_summary.manifest.sha256", sidecar.is_file(), "present" if sidecar.is_file() else "missing"))
    return _report("c3", checks, blocked_reason="PORTFOLIO_INPUT_MISSING")


def _assess_c4(
    repository: Path,
    experiment_dir: Path | None,
    external_experiment_dir: Path | None,
    human_summary: Path | None,
) -> GateReport:
    if _result_marker(repository):
        return GateReport(
            "c4",
            "blocked",
            "PUBLICATION_UNSEALED",
            (CheckResult("docs/results.md", True, "unsealed"),),
        )
    checks = [
        _file_check(repository, "docs/results.md"),
        _file_check(repository, "docs/assets/results/abcd-metrics.svg"),
    ]
    if experiment_dir is None:
        checks.extend(CheckResult(name, False, "missing") for name in ("summary.json", "manifest.sha256"))
    else:
        checks.extend(_file_check(experiment_dir, relative) for relative in ("summary.json", "manifest.sha256"))
    if external_experiment_dir is not None:
        checks.extend(_file_check(external_experiment_dir, relative) for relative in ("metrics.json", "manifest.sha256"))
    if human_summary is not None:
        checks.append(CheckResult("human_summary", human_summary.is_file(), "present" if human_summary.is_file() else "missing"))
    return _report("c4", checks, blocked_reason="PUBLICATION_UNSEALED")


def assess_gate(
    repository: Path,
    gate: GateName,
    *,
    experiment_dir: Path | None = None,
    external_experiment_dir: Path | None = None,
    human_summary: Path | None = None,
    environ: Mapping[str, str] | None = None,
    command_exists: Callable[[str], object] = _default_command_exists,
) -> GateReport:
    """Assess one gate without invoking Docker, providers, or benchmark runs."""

    root = repository.resolve()
    current_env = environ if environ is not None else dict(__import__("os").environ)
    if gate == "b1":
        return _assess_b1(root, current_env, command_exists)
    if gate == "b2":
        return _assess_b2(root, current_env)
    if gate == "c1":
        return _assess_c1(root, command_exists)
    if gate == "c2":
        return _assess_c2(root, experiment_dir)
    if gate == "c3":
        return _assess_c3(root, external_experiment_dir, human_summary)
    return _assess_c4(root, experiment_dir, external_experiment_dir, human_summary)


def _report_payload(report: GateReport) -> dict[str, object]:
    return cast(dict[str, object], asdict(report))


def _parse_gate(value: str) -> _GateArgument:
    allowed = {"b1", "b2", "c1", "c2", "c3", "c4", "all"}
    if value not in allowed:
        raise argparse.ArgumentTypeError("gate must be b1, b2, c1, c2, c3, c4, or all")
    return cast(_GateArgument, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gate", type=_parse_gate, default="all")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--external-experiment-dir", type=Path)
    parser.add_argument("--human-summary", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    arguments = parser.parse_args(argv)
    selected: tuple[GateName, ...] = (
        ("b1", "b2", "c1", "c2", "c3", "c4")
        if arguments.gate == "all"
        else (cast(GateName, arguments.gate),)
    )
    reports = tuple(
        assess_gate(
            arguments.repository,
            gate,
            experiment_dir=arguments.experiment_dir,
            external_experiment_dir=arguments.external_experiment_dir,
            human_summary=arguments.human_summary,
        )
        for gate in selected
    )
    if arguments.format == "json":
        payload: object = (
            _report_payload(reports[0]) if len(reports) == 1 else {"reports": [_report_payload(item) for item in reports]}
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    else:
        for report in reports:
            print(f"release_gate gate={report.gate} status={report.status} reason={report.reason or 'none'}")
            for check in report.checks:
                print(f"  check={check.name} present={str(check.present).lower()} detail={check.detail}")
    return 0 if all(report.status == "ready" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CheckResult",
    "GateName",
    "GateReport",
    "GateStatus",
    "assess_gate",
    "main",
]
