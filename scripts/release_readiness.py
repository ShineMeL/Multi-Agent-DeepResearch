"""Deterministic, secret-free Release A package readiness assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from deepresearch.providers import ProviderError
from deepresearch.providers.replay_schema import REPLAY_FILES, ReplayBundle
from deepresearch.runtime.runner_factory import FilePricingCatalog, FileProviderRouteCatalog

RELEASE_VERSION = "0.1.0"
ZERO_DIGEST = "0" * 64


class ReadinessFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ReadinessSummary:
    profile_count: int
    route_count: int
    pricing_count: int
    bundle_sha256: str


def assess(repository: Path) -> ReadinessSummary:
    profile_path = repository / "deploy" / "replay" / "profiles.json"
    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    bundle_root = repository / "tests" / "fixtures" / "replay" / "baseline"
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    pricing_payload = json.loads(pricing_path.read_text(encoding="utf-8"))
    if not isinstance(profile_payload, dict) or set(
        profile_payload.get("profiles", {})
    ) != {"replay-default"}:
        raise ReadinessFailure("profile_shape")
    if not isinstance(pricing_payload, dict) or set(
        pricing_payload.get("profiles", {})
    ) != {"replay-default"}:
        raise ReadinessFailure("pricing_profile_shape")
    routes = FileProviderRouteCatalog.load(profile_path).resolve("replay-default")
    prices = FilePricingCatalog.load(pricing_path).resolve("replay-default")
    try:
        verification = ReplayBundle.load(bundle_root).verify()
    except ProviderError as error:
        raise ReadinessFailure("bundle_verification") from error
    if not verification.valid:
        raise ReadinessFailure("bundle_verification")
    if routes.execution_mode != "replay" or len(routes.routes) != 5:
        raise ReadinessFailure("route_shape")
    if len(prices) != 6:
        raise ReadinessFailure("pricing_shape")
    for route in routes.routes:
        if route.operation == "parse":
            continue
        if route.parameters.get("bundle_path") != "tests/fixtures/replay/baseline":
            raise ReadinessFailure("bundle_path")
    required = {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured")
            if route.operation == "model"
            else (route.operation,)
        )
    }
    available = {(item.provider_id, item.endpoint_type, item.model_id) for item in prices}
    if not required <= available:
        raise ReadinessFailure("pricing_coverage")
    digest_input = json.dumps(
        {name: verification.file_sha256[name] for name in REPLAY_FILES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ReadinessSummary(
        profile_count=1,
        route_count=len(routes.routes),
        pricing_count=len(prices),
        bundle_sha256=hashlib.sha256(digest_input).hexdigest(),
    )


def _line(*, status: str, summary: ReadinessSummary) -> str:
    return (
        f"release_a version={RELEASE_VERSION} status={status} "
        f"profile_count={summary.profile_count} route_count={summary.route_count} "
        f"pricing_count={summary.pricing_count} bundle_sha256={summary.bundle_sha256}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args(argv)
    try:
        summary = assess(arguments.repository)
    except ReadinessFailure as error:
        failure_code = error.code
    except (OSError, UnicodeError, TypeError, ValueError, ValidationError):
        failure_code = "package_validation"
    else:
        print(_line(status="pass", summary=summary))
        return 0

    empty = ReadinessSummary(0, 0, 0, ZERO_DIGEST)
    print(_line(status="fail", summary=empty))
    print(f"readiness_error={failure_code}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
