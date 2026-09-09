"""Contract for the packaged Replay catalog and public baseline fixture."""

import json
from pathlib import Path

from deepresearch.providers.replay import ReplayBundle
from deepresearch.runtime.runner_factory import FilePricingCatalog, FileProviderRouteCatalog


def _lf_copy(source: Path, destination: Path) -> Path:
    destination.mkdir()
    for item in source.iterdir():
        if item.is_file():
            destination.joinpath(item.name).write_bytes(
                item.read_bytes().replace(b"\r\n", b"\n")
            )
    return destination


def _catalog_with_bundle(source: Path, bundle: Path, tmp_path: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for route in payload["profiles"]["replay-default"]["routes"]:
        if route["operation"] != "parse":
            route["parameters"]["bundle_path"] = str(bundle)
    destination = tmp_path / "profiles.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_packaged_replay_catalog_is_complete(tmp_path: Path) -> None:
    bundle = _lf_copy(Path("tests/fixtures/replay/baseline"), tmp_path / "bundle")
    profiles = _catalog_with_bundle(Path("deploy/replay/profiles.json"), bundle, tmp_path)
    pricing = FilePricingCatalog.load(Path("deploy/replay/pricing.json"))
    routes = FileProviderRouteCatalog.load(profiles).resolve("replay-default")
    snapshots = pricing.resolve("replay-default")

    assert routes.execution_mode == "replay"
    assert {route.operation for route in routes.routes} == {
        "model",
        "search",
        "fetch",
        "parse",
        "embed",
    }
    assert ReplayBundle.load(bundle).verify().valid
    required = {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured")
            if route.operation == "model"
            else (route.operation,)
        )
    }
    available = {
        (item.provider_id, item.endpoint_type, item.model_id) for item in snapshots
    }
    assert required <= available
    assert len(snapshots) == 6

    checked_in = json.loads(
        Path("deploy/replay/profiles.json").read_text(encoding="utf-8")
    )
    non_parse_routes = (
        route
        for route in checked_in["profiles"]["replay-default"]["routes"]
        if route["operation"] != "parse"
    )
    assert all(
        route["parameters"]["bundle_path"] == "tests/fixtures/replay/baseline"
        for route in non_parse_routes
    )
