"""Security boundaries against real serialization, storage and Core execution."""

from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from starlette.requests import Request

from apps.api.error_handlers import APIError, error_response, handle_error
from apps.api.sse import encode_sse, event_stream
from deepresearch.runtime.manager import RunManager
from deepresearch.runtime.runner_factory import FilePricingCatalog
from deepresearch.security import redact, wrap_untrusted_content
from deepresearch.security.logging import RedactingFilter
from deepresearch.storage.migrations.runner import upgrade_service_schema
from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore
from tests.contracts.api.test_sse import finalize, make_event, sse_frames
from tests.fakes.service_store import make_record
from tests.integration.replay.test_baseline_graph import config
from tests.unit.runtime.test_manager import Factory, policy

SECRET = "sk-security-fixture-915d"
FIXTURES = Path(__file__).parents[2] / "fixtures" / "security"


def test_recursive_redaction_removes_keys_headers_and_overlapping_secrets():
    payload = {
        SECRET: {"Authorization": "Bearer unknown-token", "x-api-key": "unknown-key"},
        "values": [f"Bearer {SECRET}", (f"api_key=unknown-other {SECRET}-suffix", 7, None)],
    }
    safe = redact(payload, secrets=[SECRET, SECRET + "-suffix", ""])
    encoded = json.dumps(safe)
    for value in (SECRET, "unknown-token", "unknown-key", "unknown-other", "-suffix"):
        assert value not in encoded
    assert safe["values"][1][1:] == [7, None]
    assert payload[SECRET]["Authorization"] == "Bearer unknown-token"
    assert redact(safe, secrets=[SECRET]) == safe


def test_prompt_guard_escapes_closing_tag_injection():
    attack = (FIXTURES / "prompt_injection.html").read_text(encoding="utf-8")
    wrapped = wrap_untrusted_content(attack)
    assert wrapped.startswith("<untrusted_web_content>\n")
    assert wrapped.endswith("\n</untrusted_web_content>")
    assert wrapped.lower().count("</untrusted_web_content") == 1
    assert "ignore all rules" in wrapped
    assert "<system>" not in wrapped


@pytest.mark.parametrize("mapping", [False, True])
def test_logging_filter_copies_record_and_redacts_exception_chain_and_cached_text(mapping):
    try:
        try:
            raise ValueError(SECRET)
        except ValueError as cause:
            raise RuntimeError("Authorization: Bearer hidden-token") from cause
    except RuntimeError:
        record = logging.LogRecord(
            "security",
            logging.ERROR,
            __file__,
            1,
            "provider=%(provider)s" if mapping else "provider=%s",
            ({"provider": SECRET},) if mapping else (SECRET,),
            sys.exc_info(),
        )
    record.exc_text = f"cached exception {SECRET}"
    record.stack_info = f"stack {SECRET}"
    safe = RedactingFilter(secrets=[SECRET]).filter(record)
    assert safe is not record and record.exc_info is not None
    assert safe.exc_info is None
    formatted = logging.Formatter().format(safe)
    assert SECRET not in formatted and "hidden-token" not in formatted
    assert "RuntimeError" in formatted
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.addFilter(RedactingFilter(secrets=[SECRET]))
    handler.handle(record)
    assert SECRET not in output.getvalue()


@pytest.mark.parametrize("mapping", [False, True])
def test_logging_redacts_structured_arguments_without_loaded_secrets(mapping):
    headers = {"x-api-key": "review-api-credential", "Authorization": "Basic review-basic"}
    record = logging.LogRecord(
        "security",
        logging.INFO,
        __file__,
        1,
        "headers=%(headers)s count=%(count)d" if mapping else "headers=%s count=%d",
        ({"headers": headers, "count": 2},) if mapping else (headers, 2),
        None,
    )
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.addFilter(RedactingFilter(secrets=()))
    handler.handle(record)
    assert "review-api-credential" not in output.getvalue()
    assert "review-basic" not in output.getvalue()
    assert "count=2" in output.getvalue()
    assert headers["x-api-key"] == "review-api-credential"


@pytest.mark.parametrize(
    "text",
    [
        "headers={'x-api-key': 'review-api-credential'}",
        '{"api_key": "review-api-credential"}',
        "{'Authorization': 'Basic review-api-credential'}",
        "Authorization: Basic review-api-credential",
    ],
)
def test_quoted_credentials_are_redacted_in_message_and_exception_text(text):
    try:
        raise RuntimeError(text)
    except RuntimeError:
        record = logging.LogRecord("security", logging.ERROR, __file__, 1, text, (), sys.exc_info())
    safe = RedactingFilter(secrets=()).filter(record)
    assert "review-api-credential" not in logging.Formatter().format(safe)


def test_credential_assignment_format_placeholder_remains_valid():
    record = logging.LogRecord(
        "security",
        logging.INFO,
        __file__,
        1,
        "api_key=%s count=%d",
        ("review-api-credential", 3),
        None,
    )
    safe = RedactingFilter(secrets=()).filter(record)
    assert safe.getMessage() == "api_key=[REDACTED] count=3"


def test_url_userinfo_and_fragment_credentials_are_redacted():
    url = "https://user:review-password@public.example/file#access_token=review-token&q=visible"
    safe = redact(url, secrets=())
    assert "review-password" not in safe and "review-token" not in safe
    assert "public.example/file" in safe and "q=visible" in safe
    assert redact(safe, secrets=()) == safe


@pytest.mark.parametrize(
    "parameter",
    [
        "X-Amz-Signature",
        "x-amz-signature",
        "X-Amz-Credential",
        "X-Amz-Security-Token",
        "X-Goog-Signature",
        "X-Goog-Credential",
        "AWSAccessKeyId",
        "Signature",
        "sig",
        "access_token",
        "api_key",
        "%58-Amz-Signature",
    ],
)
def test_signed_url_credentials_redacted_without_changing_core_url_policy(parameter):
    from deepresearch.retrieval import canonicalize_url

    url = f"https://public.example/file?q=visible&{parameter}=review-signature&z=last"
    canonical = canonicalize_url(url)
    assert "review-signature" in canonical
    safe = redact({"url": url}, secrets=())["url"]
    assert "review-signature" not in safe
    assert "q=visible" in safe and "z=last" in safe
    assert redact(safe, secrets=()) == safe
    event = make_event(1).model_copy(update={"public_payload": {"url": url}})
    assert "review-signature" not in encode_sse(event)
    assert event.public_payload["url"] == url


def test_httpx_request_log_redacts_signed_url_without_network():
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.addFilter(RedactingFilter(secrets=()))
    logger = logging.getLogger("httpx")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    url = "https://public.example/file?X-Amz-Signature=review-signature"
    try:
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        ) as client:
            response = client.get(url)
        assert str(response.request.url) == url
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    assert "HTTP Request: GET" in output.getvalue()
    assert "review-signature" not in output.getvalue()


@pytest.fixture
async def secured(tmp_path):
    conf = config()
    store = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}", tmp_path)
    await upgrade_service_schema(store.engine)
    secrets = [SECRET]
    manager = RunManager(
        runner_factory=Factory(conf),
        store=store,
        checkpointer=InMemorySaver(),
        pricing_catalog=FilePricingCatalog({}),
        deployment_policy=policy(conf),
        secrets=secrets,
    )
    secrets.clear()
    await store.create_run(make_record("r1", status="running"))
    yield manager, store
    await manager.shutdown(0)
    await store.engine.dispose()


async def test_emit_redacts_before_sqlite_and_sse_preserving_original_and_sequence(secured):
    manager, store = secured
    assert manager.secrets == (SECRET,)
    with pytest.raises(AttributeError):
        manager.secrets = ()
    event = make_event(1).model_copy(update={"public_payload": {"nested": [SECRET]}})
    await manager.emit(event)
    assert event.model_dump(mode="json")["public_payload"] == {"nested": [SECRET]}
    durable = await store.list_events_after("r1", 0)
    assert SECRET not in durable[0].model_dump_json()
    frame = sse_frames(encode_sse(event, secrets=manager.secrets))[0]
    assert json.loads(frame["data"]) == durable[0].model_dump(mode="json")
    assert durable[0].seq == 1
    # Simulate legacy unredacted durable data to exercise SSE independently.
    await store.append_event(make_event(2).model_copy(update={"public_payload": {"x": SECRET}}))
    await finalize(store)
    record = await store.get_run("r1")
    frames = [
        frame
        async for frame in event_stream(
            "r1",
            last_event_id=0,
            owner_scope_sha256=record.owner_scope_sha256,
            store=store,
            manager=manager,
        )
    ]
    assert [frame["id"] for frame in frames] == ["1", "2", "3"]
    assert SECRET not in json.dumps(frames)


async def test_signed_url_redacted_before_durable_store_without_loaded_signature(secured):
    manager, store = secured
    url = "https://public.example/file?X-Amz-Signature=review-signature&q=visible"
    event = make_event(1).model_copy(update={"public_payload": {"url": url}})
    await manager.emit(event)
    durable = await store.list_events_after("r1", 0)
    assert "review-signature" not in durable[0].model_dump_json()
    assert "q=visible" in durable[0].public_payload["url"]
    assert event.public_payload["url"] == url


async def test_public_errors_do_not_echo_secret_exception_or_optional_run_id(secured):
    manager, _ = secured
    app = FastAPI()
    app.state.manager = manager
    request = Request({"type": "http", "app": app})
    for error in (ValueError(SECRET), APIError("INTERNAL_ERROR", run_id=SECRET)):
        response = await handle_error(request, error)
        assert SECRET.encode() not in response.body
        assert json.loads(bytes(response.body))["code"] == "INTERNAL_ERROR"
    assert (
        SECRET.encode()
        not in error_response(APIError("INTERNAL_ERROR", run_id=SECRET), secrets=[SECRET]).body
    )


async def test_manager_rejects_credentials_before_storing_config_or_profile(secured):
    from deepresearch.runtime.runner_factory import ProviderProfileDrift

    manager, store = secured
    conf = config()
    conf = conf.model_copy(
        update={
            "request": conf.request.model_copy(update={"question": SECRET}),
        }
    )
    with pytest.raises(ProviderProfileDrift):
        await manager.create(conf, client_ip="local", session_id="local")
    assert manager.runner_factory.creates == []
    assert await store.list_events_after("r1", 0) == []


def test_old_manager_construction_defaults_to_no_secrets(tmp_path):
    conf = config()
    manager = RunManager(
        runner_factory=Factory(conf),
        store=object(),
        checkpointer=InMemorySaver(),
        pricing_catalog=FilePricingCatalog({}),
        deployment_policy=policy(conf),
    )
    assert manager.secrets == ()


@pytest.mark.parametrize("surface", ["config", "profile", "pricing"])
def test_builder_rejects_loaded_credentials_in_serialized_metadata_before_provider_calls(
    tmp_path,
    surface,
):
    from deepresearch.runtime.manifest import CostCalculator
    from deepresearch.runtime.runner_factory import DefaultCoreRunnerBuilder, ProviderProfileDrift
    from tests.unit.runtime.test_runner_factory_execution import composition, freeze

    original, conf, routes, snapshots, calls, _ = composition(tmp_path)
    builder = DefaultCoreRunnerBuilder(
        provider_constructors=original.provider_constructors,
        credential_resolver=original.credential_resolver,
        artifact_store=original.artifact_store,
        evidence_store=original.evidence_store,
        secrets=[SECRET],
    )
    if surface == "config":
        conf = conf.model_copy(update={"prompt_versions": {"planner": SECRET}})
    elif surface == "profile":
        rows = [item.model_dump(mode="json") for item in routes.routes]
        rows[0]["model_revision"] = SECRET
        routes = freeze(conf, rows)
    else:
        snapshots = (snapshots[0].model_copy(update={"snapshot_id": SECRET}), *snapshots[1:])
    with pytest.raises(ProviderProfileDrift) as error:
        builder.build(
            config=conf,
            provider_routes=routes,
            pricing_snapshots=snapshots,
            checkpointer=InMemorySaver(),
            cost_calculator=CostCalculator,
        )
    assert SECRET not in str(error.value)
    assert not calls


async def test_real_service_keeps_credentials_out_of_manifest_and_checkpoint_and_bounds_web(
    tmp_path,
    monkeypatch,
):
    import time

    from deepresearch.providers.httpx_fetcher import no_op_host_slot
    from deepresearch.runtime.checkpoints import open_sqlite_checkpointer
    from deepresearch.runtime.manifest import RunManifest
    from deepresearch.runtime.runner_factory import (
        DefaultCoreRunnerBuilder,
        EnvCredentialResolver,
        FileProviderRouteCatalog,
        LangGraphServiceRunnerFactory,
    )
    from deepresearch.workflow.runner import BaselineRuntimeHooks
    from tests.integration.replay.test_baseline_graph import ControlledSegmentClock
    from tests.unit.runtime.test_runner_factory_execution import composition, freeze

    original, conf, routes, snapshots, calls, artifacts = composition(tmp_path / "artifacts")
    rows = [item.model_dump(mode="json") for item in routes.routes]
    model_row = next(row for row in rows if row["operation"] == "model")
    model_row["credential_ref"] = "SECURITY_MODEL_KEY"
    monkeypatch.setenv("SECURITY_MODEL_KEY", SECRET)
    routes = freeze(conf, rows)
    model_route = next(row for row in routes.routes if row.operation == "model")
    model = original.provider_constructors[model_route.provider_id](
        model_route, None, no_op_host_slot
    )
    credentials_received = []

    def model_constructor(route, secret, slot):
        credentials_received.append(secret)
        return model

    original.provider_constructors[model_route.provider_id] = model_constructor
    attack = (FIXTURES / "prompt_injection.html").read_text(encoding="utf-8")
    fetch_route = next(row for row in routes.routes if row.operation == "fetch")
    fetcher = original.provider_constructors[fetch_route.provider_id](
        fetch_route, None, no_op_host_slot
    )
    original_fetch = fetcher.fetch_with_usage

    async def fetch_with_attack(*args, **kwargs):
        result = await original_fetch(*args, **kwargs)
        return replace(
            result, value=result.value.model_copy(update={"body_bytes": attack.encode()})
        )

    fetcher.fetch_with_usage = fetch_with_attack
    # The offline parser normally receives single-line plain text. Match its
    # real normalization contract for this multiline hostile fixture.
    from deepresearch.retrieval import normalize_text

    parser_route = next(row for row in routes.routes if row.operation == "parse")
    parser = original.provider_constructors[parser_route.provider_id](
        parser_route,
        None,
        no_op_host_slot,
    )
    original_parse = parser.parse

    async def parse_attack(raw, **kwargs):
        return await original_parse(
            raw.model_copy(update={"body_bytes": normalize_text(raw.body_bytes.decode()).encode()}),
            **kwargs,
        )

    parser.parse = parse_attack
    builder = DefaultCoreRunnerBuilder(
        provider_constructors=original.provider_constructors,
        credential_resolver=EnvCredentialResolver(frozenset({"SECURITY_MODEL_KEY"})),
        artifact_store=artifacts,
        evidence_store=original.evidence_store,
        secrets=[SECRET],
    )
    assert builder.content_boundary is wrap_untrusted_content
    factory = LangGraphServiceRunnerFactory(builder, FileProviderRouteCatalog({"offline": routes}))
    original_create = factory.create

    def create(**kwargs):
        runner = original_create(**kwargs)
        clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
        runner._runtime_hooks = BaselineRuntimeHooks(
            monotonic=clock.monotonic, utc_now=clock.utc_now
        )
        return runner

    factory.create = create
    store = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}", tmp_path)
    await upgrade_service_schema(store.engine)
    async with open_sqlite_checkpointer(tmp_path / "checkpoints.db") as saver:
        manager = RunManager(
            runner_factory=factory,
            store=store,
            checkpointer=saver,
            pricing_catalog=FilePricingCatalog({"offline": snapshots}),
            deployment_policy=policy(conf),
            secrets=[SECRET],
        )
        try:
            view = await manager.create(conf, client_ip="local", session_id="local")
            result = await manager.wait(view.run_id)
            assert result.status == "completed", (
                result.error_code,
                [
                    (event.node, event.error_code)
                    for event in await store.list_events_after(view.run_id, 0)
                    if event.error_code
                ],
            )
            assert credentials_received == [SECRET] and calls
            manifest_bytes = artifacts.get_bytes(result.manifest_artifact_id)
            manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
            assert SECRET.encode() not in manifest_bytes
            assert manifest.canonical_sha256() == manifest.manifest_sha256
            checkpoints = [
                checkpoint
                async for checkpoint in saver.alist(
                    {"configurable": {"thread_id": view.thread_id}},
                )
            ]
            assert checkpoints
            for checkpoint in checkpoints:
                assert SECRET not in json.dumps(checkpoint.checkpoint, default=str)
            events = await store.list_events_after(view.run_id, 0)
            assert events and SECRET not in json.dumps(
                [event.model_dump(mode="json") for event in events]
            )
            record = await store.get_run(view.run_id)
            assert SECRET not in json.dumps(record.provider_profile_json)
            prompts = [
                message.content for request in model.requests for message in request.messages
            ]
            guarded = [prompt for prompt in prompts if "ignore all rules" in prompt]
            assert guarded
            for prompt in guarded:
                assert "<untrusted_web_content>" in prompt
                assert "&lt;/untrusted_web_content&gt;" in prompt
                assert "&lt;system&gt;" in prompt
            # The full raw body remains a content-addressed artifact, and
            # evidence verification still succeeds with the original excerpt.
            from deepresearch.providers import RawDocument

            raw_bodies = []
            for event in events:
                if event.node != "Fetch":
                    continue
                for artifact_id in event.artifact_ids:
                    payload = json.loads(artifacts.get_bytes(artifact_id))
                    if payload.get("kind") == "raw":
                        raw_bodies.extend(
                            RawDocument.model_validate(document).body_bytes
                            for document in payload["documents"]
                        )
            assert raw_bodies and all(body == attack.encode() for body in raw_bodies)
            assert manifest.evidence_hashes
            for evidence in manifest.evidence_hashes:
                stored = original.evidence_store.get_evidence(evidence.evidence_id)
                assert stored.excerpt == normalize_text(attack)
        finally:
            await manager.shutdown(0)
    await store.engine.dispose()
