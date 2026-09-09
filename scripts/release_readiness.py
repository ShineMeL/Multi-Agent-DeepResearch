"""Deterministic, secret-free Release A package readiness assessment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Never, cast

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


class _SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ReadinessFailure("cli_arguments")


@dataclass(frozen=True)
class ReadinessSummary:
    profile_count: int
    route_count: int
    pricing_count: int
    bundle_sha256: str


def _has_release_profile(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    profiles = cast("dict[str, object]", payload).get("profiles")
    if not isinstance(profiles, dict):
        return False
    return set(cast("dict[str, object]", profiles)) == {"replay-default"}


def assess(repository: Path) -> ReadinessSummary:
    profile_path = repository / "deploy" / "replay" / "profiles.json"
    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    bundle_root = repository / "tests" / "fixtures" / "replay" / "baseline"
    profile_payload: object = json.loads(profile_path.read_text(encoding="utf-8"))
    pricing_payload: object = json.loads(pricing_path.read_text(encoding="utf-8"))
    if not _has_release_profile(profile_payload):
        raise ReadinessFailure("profile_shape")
    if not _has_release_profile(pricing_payload):
        raise ReadinessFailure("pricing_profile_shape")
    routes = FileProviderRouteCatalog.load(profile_path).resolve("replay-default")
    prices = FilePricingCatalog.load(pricing_path).resolve("replay-default")
    try:
        bundle = ReplayBundle.load(bundle_root)
        verification = bundle.verify()
    except ProviderError as error:
        raise ReadinessFailure("bundle_verification") from error
    if not verification.valid:
        raise ReadinessFailure("bundle_verification")
    if routes.execution_mode != "replay" or len(routes.routes) != 5:
        raise ReadinessFailure("route_shape")
    route_by_operation = {route.operation: route for route in routes.routes}
    if set(route_by_operation) != {"model", "search", "fetch", "parse", "embed"}:
        raise ReadinessFailure("route_topology")
    endpoint_by_operation = {
        "model": "chat.completions",
        "search": "search",
        "fetch": "fetch",
        "parse": "parse",
        "embed": "embed",
    }
    for operation, endpoint_type in endpoint_by_operation.items():
        route = route_by_operation[operation]
        if route.endpoint_type != endpoint_type or route.fallback_rank != 0:
            raise ReadinessFailure("route_topology")
        if operation == "parse":
            if route.parameters:
                raise ReadinessFailure("route_topology")
            expected_identity = ("baseline-parser-router", None, None)
        else:
            if route.parameters.get("bundle_path") != "tests/fixtures/replay/baseline":
                raise ReadinessFailure("bundle_path")
            if set(route.parameters) != {"bundle_path"}:
                raise ReadinessFailure("route_topology")
            snapshot = bundle.snapshot.providers.get(operation)
            if snapshot is None:
                raise ReadinessFailure("route_identity")
            expected_identity = (
                snapshot.provider_id,
                snapshot.model_id,
                snapshot.model_revision,
            )
        if (route.provider_id, route.model_id, route.model_revision) != expected_identity:
            raise ReadinessFailure("route_identity")
    if len(prices) != 6:
        raise ReadinessFailure("pricing_shape")
    required = {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured") if route.operation == "model" else (route.operation,)
        )
    }
    available = {(item.provider_id, item.endpoint_type, item.model_id) for item in prices}
    if required != available:
        raise ReadinessFailure("pricing_coverage")
    # Release A is entirely zero-cost. This also enforces the runtime's
    # complete/structured equality and zero-cost non-model pricing contracts.
    if any(
        rate != 0
        for item in prices
        for rate in (
            item.input_tokens_per_million_usd,
            item.output_tokens_per_million_usd,
            item.cached_tokens_per_million_usd,
            item.reasoning_tokens_per_million_usd,
        )
    ):
        raise ReadinessFailure("pricing_rates")
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
    parser = _SanitizedArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    try:
        arguments = parser.parse_args(argv)
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
