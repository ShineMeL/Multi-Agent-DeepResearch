"""HTTP/SSE + durable SQLite + real Core in strict replay execution mode.

Record deterministic test providers, then replay model/search/fetch/embed through
Core replay adapters with the real HTML parser. Strict replay is covered against
the shipped baseline bundle in a separate service-level test below.
"""

import asyncio
import json
import time
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from apps.api import create_app
from apps.api.main import create_app as create_hosted_app
from apps.api.settings import ServiceSettings
from deepresearch.domain import RunConfig
from deepresearch.providers import ProviderUsageResult, RawDocument
from deepresearch.providers.httpx_fetcher import no_op_host_slot
from deepresearch.providers.parsers import HtmlParser
from deepresearch.providers.recording import (
    RecordingFetcher,
    RecordingModelProvider,
    RecordingSearchProvider,
    RecordingTextEmbedder,
    ReplayBundleWriter,
)
from deepresearch.providers.replay import ReplayBundle
from deepresearch.runtime.checkpointers import open_service_checkpointer
from deepresearch.runtime.manifest import RunManifest
from deepresearch.runtime.runner_factory import (
    FilePricingCatalog,
    FileProviderRouteCatalog,
    ProviderConstructor,
)
from deepresearch.storage.migrations.runner import upgrade_service_schema
from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore
from deepresearch.workflow.runner import BaselineRuntimeHooks
from tests.integration.replay.test_baseline_graph import (
    ControlledSegmentClock,
    CountingOfflineFetcher,
)
from tests.integration.replay.test_manager_replay import (
    setup,  # pyright: ignore[reportUnknownVariableType] - existing untyped fixture
)
from tests.unit.runtime.test_runner_factory_execution import freeze, pricing, route


class HtmlOfflineFetcher(CountingOfflineFetcher):
    """Supply actual HTML so the production parser is used in both phases."""

    async def fetch_with_usage(
        self, url: str, **kwargs: object
    ) -> ProviderUsageResult[RawDocument]:
        result = await super().fetch_with_usage(url, **kwargs)
        html = (
            "<html><head><title>Offline evidence</title></head><body><article><p>"
            "Offline research evidence supports a reproducible baseline workflow. "
            "The fixed planner selects sources, extracts their supporting evidence, "
            "and writes a report with citations. This frozen article supplies enough "
            "main text to exercise the production HTML parser while keeping the example "
            f"independent of external websites. Source: {url}</p></article></body></html>"
        )
        return ProviderUsageResult(
            value=result.value.model_copy(update={"body_bytes": html.encode()}),
            usage=result.usage,
        )


@pytest.mark.parametrize(
    "strict_replay",
    [
        pytest.param(False, id="offline-recording"),
    ],
)
async def test_full_service_has_artifacts_terminal_event_and_owned_reconnect(
    tmp_path: Path,
    strict_replay: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    writer = ReplayBundleWriter.create(tmp_path / "bundle", run_id="service-recording")
    for replaying in (False, True) if strict_replay else (False,):
        root = tmp_path / str(replaying)
        manager, conf, calls, artifacts, _ = setup(root)
        factory: Any = manager.runner_factory
        builder = factory.builder
        routes = factory.route_catalog.resolve("offline")
        parser = HtmlParser()
        fetcher = HtmlOfflineFetcher(calls)
        rows = [item.model_dump(mode="json") for item in routes.routes if item.operation != "parse"]
        rows.append(route("parse", parser))
        parser_constructor: ProviderConstructor = lambda route, secret, slot, provider=parser: (
            provider
        )
        fetch_constructor: ProviderConstructor = lambda route, secret, slot, provider=fetcher: (
            provider
        )
        builder.provider_constructors[parser.parser_id] = parser_constructor
        builder.provider_constructors[fetcher.provider_id] = fetch_constructor
        snapshots = tuple(
            pricing(
                row["provider_id"],
                endpoint,
                row["model_id"] or row["operation"],
                "1" if row["operation"] == "model" else "0",
            )
            for row in rows
            for endpoint in (
                ("complete", "structured") if row["operation"] == "model" else (row["operation"],)
            )
        )
        manager.pricing_catalog = FilePricingCatalog({"offline": snapshots})
        if replaying:
            conf = conf.model_copy(
                update={"request": conf.request.model_copy(update={"execution_mode": "replay"})}
            )
            manager.deployment_policy = replace(
                manager.deployment_policy, allowed_execution_modes=frozenset({"replay"})
            )
            for row in rows:
                if row["operation"] != "parse":
                    row["parameters"] = {"bundle_path": str(tmp_path / "bundle")}
        else:
            wrappers: dict[str, Any] = {
                "model": RecordingModelProvider,
                "search": RecordingSearchProvider,
                "fetch": RecordingFetcher,
                "embed": RecordingTextEmbedder,
            }
            for frozen_route in routes.routes:
                if frozen_route.operation == "parse":
                    continue
                delegate = builder.provider_constructors[frozen_route.provider_id](
                    frozen_route, None, no_op_host_slot
                )
                if frozen_route.operation == "model":
                    writer.register_provider(
                        "model",
                        provider_id=frozen_route.provider_id,
                        model_id=frozen_route.model_id,
                        model_revision=frozen_route.model_revision,
                    )
                recorded = wrappers[frozen_route.operation](delegate, writer)
                constructor: ProviderConstructor = lambda route, secret, slot, provider=recorded: (
                    provider
                )
                builder.provider_constructors[frozen_route.provider_id] = constructor
        factory.route_catalog = FileProviderRouteCatalog({"offline": freeze(conf, rows)})
        store = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{root / 'runs.sqlite'}", root)
        await upgrade_service_schema(store.engine)
        manager.store = store
        try:
            async with AsyncExitStack() as stack:
                if replaying:
                    # Core compares monotonic active duration to a UTC envelope.
                    # Keep its existing deterministic clock fixture; all service
                    # composition, route selection and provider types stay real.
                    def runtime_clock() -> BaselineRuntimeHooks:
                        clock = ControlledSegmentClock(
                            monotonic_start=time.monotonic(),
                            utc_offset_seconds=0,
                        )
                        return BaselineRuntimeHooks(
                            monotonic=clock.monotonic, utc_now=clock.utc_now
                        )

                    monkeypatch.setattr(
                        "deepresearch.workflow.runner.BaselineRuntimeHooks", runtime_clock
                    )
                    # Catalog files are the supported runtime input. Lifespan
                    # constructs the default registry, factory, saver and manager.
                    profiles_path = root / "profiles.json"
                    profiles_path.write_text(
                        json.dumps(
                            {
                                "profiles": {
                                    "offline": {
                                        "execution_mode": "replay",
                                        "routes": rows,
                                    }
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    pricing_path = root / "pricing.json"
                    pricing_path.write_text(
                        json.dumps(
                            {
                                "profiles": {
                                    "offline": [
                                        snapshot.model_dump(mode="json") for snapshot in snapshots
                                    ]
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    app = create_hosted_app(
                        ServiceSettings(
                            database_url=f"sqlite+aiosqlite:///{root / 'runs.sqlite'}",
                            artifact_root=root,
                            checkpoint_sqlite_path=root / "checkpoints.sqlite",
                            provider_profile_catalog_path=profiles_path,
                            pricing_catalog_path=pricing_path,
                            allowed_provider_profile_ids=("offline",),
                            session_signing_key=SecretStr("s" * 32),
                            langgraph_strict_msgpack=True,
                        )
                    )
                    await stack.enter_async_context(app.router.lifespan_context(app))
                    manager = app.state.manager
                    saver = manager.checkpointer
                    store = app.state.store
                else:
                    saver = await stack.enter_async_context(
                        open_service_checkpointer(
                            database_url=f"sqlite+aiosqlite:///{root / 'runs.sqlite'}",
                            sqlite_path=root / "checkpoints.sqlite",
                        )
                    )
                    manager.checkpointer = saver
                    app = create_app(
                        manager=manager,
                        deployment_policy=manager.deployment_policy,
                        artifact_store=artifacts,
                        session_secret=b"s" * 32,
                    )
                body: dict[str, Any] = {
                    "request": conf.request.model_dump(mode="json"),
                    "workflow_id": "baseline-v1",
                    "planner_id": "P1",
                    "ranker_id": "R1",
                    "seed": 0,
                }
                if replaying:
                    assert body["request"]["execution_mode"] == "replay"
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                ) as client:
                    accepted = await client.post(
                        "/runs", json=body, headers={"Idempotency-Key": "e2e-1"}
                    )
                    assert accepted.status_code == 202, accepted.text
                    run_id = accepted.json()["run_id"]
                    await asyncio.wait_for(manager.wait(run_id), 30)
                    repeated = await client.post(
                        "/runs", json=body, headers={"Idempotency-Key": "e2e-1"}
                    )
                    assert repeated.json()["run_id"] == run_id
                    final = await client.get(f"/runs/{run_id}")
                    assert final.json()["status"] == "completed", final.text
                    for kind in ("report", "evidence", "manifest"):
                        artifact = await client.get(f"/runs/{run_id}/artifacts/{kind}")
                        assert artifact.status_code == 200 and artifact.content
                        if kind == "manifest":
                            manifest = RunManifest.model_validate_json(artifact.content)
                            assert manifest.pricing_snapshots
                            assert manifest.pricing_status == "estimated"
                            if replaying:
                                assert manifest.provider_profiles[0].execution_mode == "replay"
                            record = await store.get_run(run_id)
                            assert record is not None
                            assert RunConfig.model_validate(
                                record.config_json
                            ).request.execution_mode == ("replay" if replaying else "live")
                            assert manifest.pricing_snapshots == record.pricing_snapshots
                            assert (
                                manifest.provider_profiles[0].configuration_sha256
                                == record.provider_profile_sha256
                            )
                    events_response = await client.get(f"/runs/{run_id}/events")
                    assert events_response.status_code == 200
                    events = [
                        json.loads(line[6:])
                        for line in events_response.text.splitlines()
                        if line.startswith("data: ")
                    ]
                    assert events[-1]["status"] == "completed"
                    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
                    cursor = events[len(events) // 2]["seq"]
                    resumed = await client.get(
                        f"/runs/{run_id}/events", headers={"Last-Event-ID": str(cursor)}
                    )
                    after = [
                        json.loads(line[6:])
                        for line in resumed.text.splitlines()
                        if line.startswith("data: ")
                    ]
                    assert after == [event for event in events if event["seq"] > cursor]
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="https://testserver"
                    ) as stranger:
                        for suffix in (
                            "",
                            "/events",
                            "/artifacts/report",
                            "/artifacts/evidence",
                            "/artifacts/manifest",
                        ):
                            assert (
                                await stranger.get(f"/runs/{run_id}{suffix}")
                            ).status_code == 404
                        for action in ("resume", "cancel"):
                            assert (
                                await stranger.post(f"/runs/{run_id}/{action}")
                            ).status_code == 404
                await manager.shutdown(0)
                checkpoint = await saver.aget_tuple(
                    {"configurable": {"thread_id": accepted.json()["thread_id"]}}
                )
                assert checkpoint is not None
            reopened = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{root / 'runs.sqlite'}", root)
            try:
                persisted = await reopened.get_run(run_id)
                assert persisted is not None and persisted.status == "completed"
                assert len(await reopened.list_events_after(run_id, 0)) == len(events)
            finally:
                await reopened.engine.dispose()
        finally:
            await manager.shutdown(0)
            await store.engine.dispose()
        if not replaying:
            await writer.finalize()
            assert ReplayBundle.load(tmp_path / "bundle").verify().valid
        else:
            assert not calls


async def test_full_service_strict_replay_completes_against_verified_baseline_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = Path("tests/fixtures/replay/baseline").resolve()
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    for item in source.iterdir():
        if item.is_file():
            (bundle_root / item.name).write_bytes(item.read_bytes().replace(b"\r\n", b"\n"))
    bundle = ReplayBundle.load(bundle_root)
    verification = bundle.verify()
    assert verification.valid
    snapshot = json.loads((bundle_root / "snapshot.json").read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for operation in ("model", "search", "fetch", "embed"):
        provider = snapshot["providers"][operation]
        rows.append(
            {
                "operation": operation,
                "provider_id": provider["provider_id"],
                "endpoint_type": "chat.completions" if operation == "model" else operation,
                "model_id": provider["model_id"],
                "model_revision": provider["model_revision"],
                "base_url": None,
                "credential_ref": None,
                "fallback_rank": 0,
                "parameters": {"bundle_path": str(bundle_root)},
            }
        )
    rows.append(
        {
            "operation": "parse",
            "provider_id": "baseline-parser-router",
            "endpoint_type": "parse",
            "model_id": None,
            "model_revision": None,
            "base_url": None,
            "credential_ref": None,
            "fallback_rank": 0,
            "parameters": {},
        }
    )
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps({"profiles": {"offline": {"execution_mode": "replay", "routes": rows}}}),
        encoding="utf-8",
    )
    pricing_rows = [
        pricing(
            row["provider_id"],
            endpoint,
            row["model_id"] or row["operation"],
            "0",
        ).model_dump(mode="json")
        for row in rows
        for endpoint in (
            ("complete", "structured") if row["operation"] == "model" else (row["operation"],)
        )
    ]
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps({"profiles": {"offline": pricing_rows}}),
        encoding="utf-8",
    )

    def runtime_clock() -> BaselineRuntimeHooks:
        clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
        return BaselineRuntimeHooks(monotonic=clock.monotonic, utc_now=clock.utc_now)

    monkeypatch.setattr("deepresearch.runtime.runner_factory.paired_runtime_hooks", runtime_clock)

    settings = ServiceSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
        artifact_root=tmp_path,
        checkpoint_sqlite_path=tmp_path / "checkpoints.sqlite",
        provider_profile_catalog_path=profiles_path,
        pricing_catalog_path=pricing_path,
        deployment_access_profile="local",
        allowed_execution_modes=("replay",),
        allowed_provider_profile_ids=("offline",),
        allowed_run_purposes=("demo",),
        allowed_budget_presets=("medium",),
        session_signing_key=SecretStr("s" * 32),
        langgraph_strict_msgpack=True,
    )
    app = create_hosted_app(settings)
    async with app.router.lifespan_context(app):
        body = {
            "request": {
                "question": "Compare planner strategies",
                "output_requirements": {"answer_shape": "markdown"},
                "report_language": "en",
                "source_languages": ["en"],
                "freshness_requirement": {"kind": "none"},
                "execution_mode": "replay",
                "access_profile": "showcase",
                "provider_profile_id": "offline",
                "run_purpose": "demo",
                "budget_preset": "medium",
            },
            "workflow_id": "baseline-v1",
            "planner_id": "P1",
            "ranker_id": "R1",
            "seed": 0,
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://testserver"
        ) as client:
            accepted = await client.post(
                "/runs", json=body, headers={"Idempotency-Key": "strict-replay-1"}
            )
            assert accepted.status_code == 202, accepted.text
            run_id = accepted.json()["run_id"]
            await asyncio.wait_for(app.state.manager.wait(run_id), 30)
            final = await client.get(f"/runs/{run_id}")
            assert final.status_code == 200
            assert final.json()["status"] == "completed", final.text
            manifest_response = await client.get(f"/runs/{run_id}/artifacts/manifest")
            assert manifest_response.status_code == 200
            manifest = RunManifest.model_validate_json(manifest_response.content)
            assert manifest.replay_parent == snapshot["run_id"]
            assert manifest.provider_profiles[0].execution_mode == "replay"
            assert "baseline-parser-router" in manifest.provider_profiles[0].provider_ids
            events_response = await client.get(f"/runs/{run_id}/events")
            assert events_response.status_code == 200
            assert "run_completed" in events_response.text
