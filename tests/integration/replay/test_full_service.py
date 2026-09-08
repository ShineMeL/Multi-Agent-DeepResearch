"""HTTP/SSE + durable SQLite + real Core, recording/replaying an offline model.

Search/fetch/parse/embed use deterministic test providers. This hybrid fixture
is not a shipped replay-default bundle or a production research-v1 graph.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from apps.api import create_app
from deepresearch.providers.httpx_fetcher import no_op_host_slot
from deepresearch.providers.recording import RecordingModelProvider, ReplayBundleWriter
from deepresearch.runtime.checkpointers import open_service_checkpointer
from deepresearch.runtime.manifest import RunManifest
from deepresearch.runtime.runner_factory import (
    FileProviderRouteCatalog,
    ProviderConstructor,
    default_provider_constructors,
)
from deepresearch.storage.migrations.runner import upgrade_service_schema
from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore
from tests.integration.replay.test_manager_replay import (
    setup,  # pyright: ignore[reportUnknownVariableType] - existing untyped fixture
)
from tests.unit.runtime.test_runner_factory_execution import freeze


async def test_full_service_model_replay_has_artifacts_terminal_event_and_owned_reconnect(
    tmp_path: Path,
):
    writer = ReplayBundleWriter.create(tmp_path / "bundle", run_id="service-recording")
    for replaying in (False, True):
        root = tmp_path / str(replaying)
        manager, conf, calls, artifacts, _ = setup(root)
        factory: Any = manager.runner_factory
        builder = factory.builder
        routes = factory.route_catalog.resolve("offline")
        model_route = next(item for item in routes.routes if item.operation == "model")
        if replaying:
            rows = [item.model_dump(mode="json") for item in routes.routes]
            for row in rows:
                if row["operation"] == "model":
                    row["parameters"] = {"bundle_path": str(tmp_path / "bundle")}
            factory.route_catalog = FileProviderRouteCatalog({"offline": freeze(conf, rows)})
            builder.provider_constructors[model_route.provider_id] = (
                default_provider_constructors()["replay"]
            )
        else:
            delegate = builder.provider_constructors[model_route.provider_id](
                model_route, None, no_op_host_slot
            )
            writer.register_provider(
                "model",
                provider_id=model_route.provider_id,
                model_id=model_route.model_id,
                model_revision=model_route.model_revision,
            )
            recorded = RecordingModelProvider(delegate, writer)
            constructor: ProviderConstructor = lambda route, secret, slot, provider=recorded: (
                provider
            )
            builder.provider_constructors[model_route.provider_id] = constructor
        store = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{root / 'runs.sqlite'}", root)
        await upgrade_service_schema(store.engine)
        manager.store = store
        try:
            async with open_service_checkpointer(
                database_url=f"sqlite+aiosqlite:///{root / 'runs.sqlite'}",
                sqlite_path=root / "checkpoints.sqlite",
            ) as saver:
                manager.checkpointer = saver
                app = create_app(
                    manager=manager,
                    deployment_policy=manager.deployment_policy,
                    artifact_store=artifacts,
                    session_secret=b"s" * 32,
                )
                body = {
                    "request": conf.request.model_dump(mode="json"),
                    "workflow_id": "baseline-v1",
                    "planner_id": "P1",
                    "ranker_id": "R1",
                    "seed": 0,
                }
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
                            record = await store.get_run(run_id)
                            assert record is not None
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
        else:
            assert not any(key.startswith("model:") for key in calls)
