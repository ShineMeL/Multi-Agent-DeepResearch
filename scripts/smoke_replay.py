"""Bounded HTTP acceptance for the hosted, replay-only demonstration path."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Literal, NoReturn, TypedDict, cast
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from apps.ui.api_client import DemoCapabilities, RunAccepted, RunView
from deepresearch.domain import RunEvent
from deepresearch.runtime.manifest import RunManifest

_JSON_LIMIT = 1 * 1024 * 1024
_EVENT_LIMIT = 4 * 1024 * 1024
_ARTIFACT_LIMIT = 16 * 1024 * 1024
_MAX_EVENTS = 10_000
_TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
_RUN_ID = re.compile(r"[^\x00-\x20]{1,256}")


class SmokeSuccess(TypedDict):
    ok: Literal[True]
    mode: Literal["replay"]
    run_id: str
    status: Literal["completed"]
    stop_reason: Literal["SUFFICIENT"]
    is_partial: Literal[False]
    event_count: int
    artifact_sha256: dict[str, str]
    cost_usd: Literal["0"]


class SmokeError(RuntimeError):
    """A public, static failure code with only bounded diagnostic metadata."""

    def __init__(
        self,
        code: str,
        *,
        run_id: str | None = None,
        status: str | None = None,
        cancel_attempted: bool = False,
        cancel_status: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.run_id = run_id
        self.status = status
        self.cancel_attempted = cancel_attempted
        self.cancel_status = cancel_status

    def payload(self) -> dict[str, object]:
        result: dict[str, object] = {"ok": False, "code": self.code}
        if self.run_id is not None:
            result["run_id"] = self.run_id
        if self.status is not None:
            result["status"] = self.status
        if self.cancel_attempted:
            result["cancel_attempted"] = True
            if self.cancel_status is not None:
                result["cancel_status"] = self.cancel_status
        return result


def _base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise SmokeError("INVALID_BASE_URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is None and ":" in parsed.netloc.removeprefix("[").removesuffix("]")
    ):
        raise SmokeError("INVALID_BASE_URL")
    return value.rstrip("/")


def _run_path(run_id: str) -> str:
    if _RUN_ID.fullmatch(run_id) is None or run_id in {".", ".."}:
        raise SmokeError("MALFORMED_RESPONSE")
    return f"/runs/{quote(run_id, safe='')}"


async def _response_bytes(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    expected: Iterable[int],
    limit: int,
    json_body: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[httpx.Headers, bytes]:
    try:
        async with client.stream(
            method, path, json=json_body, headers=headers
        ) as response:
            if response.status_code not in expected:
                raise SmokeError("HTTP_ERROR")
            length = response.headers.get("content-length")
            if length is not None:
                try:
                    if int(length) < 0 or int(length) > limit:
                        raise SmokeError("RESPONSE_TOO_LARGE")
                except ValueError:
                    raise SmokeError("MALFORMED_RESPONSE") from None
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > limit:
                    raise SmokeError("RESPONSE_TOO_LARGE")
            return response.headers, bytes(body)
    except SmokeError:
        raise
    except httpx.HTTPError:
        raise SmokeError("NETWORK_ERROR") from None


def _json_object(data: bytes) -> dict[str, object]:
    try:
        value: object = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SmokeError("MALFORMED_RESPONSE") from None
    if not isinstance(value, dict):
        raise SmokeError("MALFORMED_RESPONSE")
    return cast("dict[str, object]", value)


async def _json_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    expected: Iterable[int] = (200,),
    json_body: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    _, body = await _response_bytes(
        client,
        method,
        path,
        expected=expected,
        limit=_JSON_LIMIT,
        json_body=json_body,
        headers=headers,
    )
    return _json_object(body)


def _model[ModelT: BaseModel](model: type[ModelT], data: object) -> ModelT:
    try:
        return model.model_validate(data)
    except ValidationError:
        raise SmokeError("MALFORMED_RESPONSE") from None


def _event_data(lines: Sequence[str]) -> list[str]:
    frames: list[str] = []
    fields: list[str] = []
    for line in lines:
        if line == "":
            if fields:
                frames.append("\n".join(fields))
            fields = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            fields.append(value.removeprefix(" ") if separator else "")
    return frames


def _events(data: bytes, run_id: str) -> tuple[RunEvent, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SmokeError("MALFORMED_RESPONSE") from None
    parsed: list[RunEvent] = []
    for item in _event_data(text.splitlines()):
        if len(parsed) >= _MAX_EVENTS:
            raise SmokeError("RESPONSE_TOO_LARGE")
        try:
            event = RunEvent.model_validate_json(item)
        except (ValidationError, ValueError):
            raise SmokeError("MALFORMED_RESPONSE") from None
        expected_seq = len(parsed) + 1
        if event.run_id != run_id or event.seq != expected_seq:
            raise SmokeError("EVENT_LOG_INVALID")
        parsed.append(event)
    if not parsed or parsed[-1].kind != "run_completed" or parsed[-1].status != "completed":
        raise SmokeError("TERMINAL_EVENT_MISSING")
    return tuple(parsed)


def _artifact_hash(data: bytes, expected_id: str | None) -> str:
    if not data or expected_id is None:
        raise SmokeError("ARTIFACT_MISSING")
    digest = hashlib.sha256(data).hexdigest()
    if expected_id != f"sha256:{digest}":
        raise SmokeError("ARTIFACT_HASH_MISMATCH")
    return digest


async def _cancel(
    client: httpx.AsyncClient, run_id: str, *, timeout: float
) -> tuple[bool, str | None]:
    try:
        async with asyncio.timeout(timeout):
            data = await _json_request(client, "POST", f"{_run_path(run_id)}/cancel")
            view = _model(RunView, data)
            if view.run_id != run_id:
                return True, None
            return True, view.status
    except (SmokeError, TimeoutError):
        return True, None


async def _accept_replay(
    client: httpx.AsyncClient,
    *,
    state: dict[str, str | None],
) -> SmokeSuccess:
    ready = await _json_request(client, "GET", "/health/ready")
    checks = ready.get("checks")
    check_values = cast("dict[object, object]", checks) if isinstance(checks, dict) else {}
    if (
        ready.get("status") != "ok"
        or not isinstance(checks, dict)
        or not checks
        or any(value != "ok" for value in check_values.values())
    ):
        raise SmokeError("API_NOT_READY")

    capabilities = _model(
        DemoCapabilities, await _json_request(client, "GET", "/capabilities")
    )
    example = capabilities.replay_example
    profile = next(
        (
            item
            for item in capabilities.profiles
            if item.execution_mode == "replay"
            and item.available
            and item.workflow_id == "research-v1"
            and item.planner_id == "P1"
            and item.ranker_id == "R1"
        ),
        None,
    )
    if example is None or profile is None or example.budget_preset not in capabilities.budget_presets:
        raise SmokeError("NO_REPLAY_CAPABILITY")

    payload: dict[str, object] = {
        "request": {
            "question": example.question,
            "output_requirements": {"answer_shape": "markdown"},
            "report_language": example.report_language,
            "source_languages": list(example.source_languages),
            "freshness_requirement": {"kind": "none"},
            "execution_mode": "replay",
            "access_profile": "showcase",
            "provider_profile_id": profile.profile_id,
            "run_purpose": "demo",
            "budget_preset": example.budget_preset,
        },
        "workflow_id": profile.workflow_id,
        "planner_id": profile.planner_id,
        "ranker_id": profile.ranker_id,
        "seed": example.seed,
    }
    accepted = _model(
        RunAccepted,
        await _json_request(
            client,
            "POST",
            "/runs",
            expected=(202,),
            json_body=payload,
            headers={"Idempotency-Key": f"replay-smoke-{uuid4()}"},
        ),
    )
    run_id = accepted.run_id
    run_path = _run_path(run_id)
    state["run_id"] = run_id
    state["status"] = accepted.status
    if accepted.status not in {"queued", "running", "completed"}:
        raise SmokeError("RUN_NOT_ACCEPTED", run_id=run_id, status=accepted.status)

    event_headers, event_bytes = await _response_bytes(
        client,
        "GET",
        f"{run_path}/events",
        expected=(200,),
        limit=_EVENT_LIMIT,
        headers={"Accept": "text/event-stream", "Last-Event-ID": "0"},
    )
    if not event_headers.get("content-type", "").lower().startswith("text/event-stream"):
        raise SmokeError("MALFORMED_RESPONSE")
    events = _events(event_bytes, run_id)

    final = _model(RunView, await _json_request(client, "GET", run_path))
    if final.run_id != run_id:
        raise SmokeError("MALFORMED_RESPONSE")
    state["status"] = final.status
    if (
        final.status != "completed"
        or final.stop_reason != "SUFFICIENT"
        or final.is_partial
        or final.error_code is not None
        or final.final_usage is None
        or final.final_usage.cost_usd != Decimal(0)
    ):
        raise SmokeError("RUN_UNSUCCESSFUL", run_id=run_id, status=final.status)

    artifact_bytes: dict[str, bytes] = {}
    artifact_ids = {
        "report": final.report_artifact_id,
        "evidence": final.evidence_graph_artifact_id,
        "manifest": final.manifest_artifact_id,
    }
    artifact_hashes: dict[str, str] = {}
    for kind, artifact_id in artifact_ids.items():
        _, data = await _response_bytes(
            client,
            "GET",
            f"{run_path}/artifacts/{kind}",
            expected=(200,),
            limit=_ARTIFACT_LIMIT,
        )
        artifact_bytes[kind] = data
        artifact_hashes[kind] = _artifact_hash(data, artifact_id)

    try:
        manifest = RunManifest.model_validate_json(artifact_bytes["manifest"], strict=True)
    except (ValidationError, ValueError):
        raise SmokeError("MANIFEST_INVALID") from None
    required_ids = {artifact_ids["report"], artifact_ids["evidence"]}
    if (
        manifest.run_id != run_id
        or manifest.thread_id != final.thread_id
        or manifest.workflow_id != profile.workflow_id
        or manifest.planner_id != profile.planner_id
        or manifest.ranker_id != profile.ranker_id
        or manifest.seed != example.seed
        or manifest.stop_reason != "SUFFICIENT"
        or manifest.is_partial
        or not manifest.replay_parent
        or manifest.replay_parent == run_id
        or manifest.usage.cost_usd != Decimal(0)
        or manifest.pricing_status != "estimated"
        or any(item.execution_mode != "replay" for item in manifest.provider_profiles)
        or {item.profile_id for item in manifest.provider_profiles} != {profile.profile_id}
        or not required_ids.issubset(set(manifest.artifact_ids))
        or any(
            call.estimated_cost_usd not in {None, Decimal(0)}
            for call in manifest.provider_calls
        )
    ):
        raise SmokeError("MANIFEST_MISMATCH", run_id=run_id, status=final.status)

    return {
        "ok": True,
        "mode": "replay",
        "run_id": run_id,
        "status": final.status,
        "stop_reason": final.stop_reason,
        "is_partial": final.is_partial,
        "event_count": len(events),
        "artifact_sha256": artifact_hashes,
        "cost_usd": "0",
    }


async def replay_smoke(
    api_url: str,
    *,
    ui_url: str | None = None,
    timeout: float = 60.0,
    api_transport: httpx.AsyncBaseTransport | None = None,
    ui_transport: httpx.AsyncBaseTransport | None = None,
) -> SmokeSuccess:
    """Run one replay-only acceptance and return bounded, non-sensitive metadata."""

    api_url = _base_url(api_url)
    ui_url = _base_url(ui_url) if ui_url is not None else None
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 3600:
        raise SmokeError("INVALID_TIMEOUT")
    state: dict[str, str | None] = {"run_id": None, "status": None}
    request_timeout = httpx.Timeout(connect=min(timeout, 5), read=None, write=5, pool=5)
    async with httpx.AsyncClient(
        base_url=api_url,
        transport=api_transport,
        timeout=request_timeout,
        follow_redirects=False,
    ) as client:
        try:
            async with asyncio.timeout(timeout):
                if ui_url is not None:
                    async with httpx.AsyncClient(
                        base_url=ui_url,
                        transport=ui_transport,
                        timeout=request_timeout,
                        follow_redirects=False,
                    ) as ui:
                        await _response_bytes(
                            ui,
                            "GET",
                            "/_stcore/health",
                            expected=(200,),
                            limit=64 * 1024,
                        )
                return await _accept_replay(client, state=state)
        except TimeoutError:
            error = SmokeError("TIMEOUT", run_id=state["run_id"], status=state["status"])
        except SmokeError as caught:
            error = caught
            if error.run_id is None:
                error.run_id = state["run_id"]
            if error.status is None:
                error.status = state["status"]

        if state["run_id"] is not None and state["status"] not in _TERMINAL:
            attempted, cancel_status = await _cancel(
                client, state["run_id"], timeout=min(2.0, max(0.1, timeout))
            )
            error.cancel_attempted = attempted
            error.cancel_status = cancel_status
        raise error


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise SmokeError("INVALID_ARGUMENT")


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _Parser(description="Replay-only hosted deployment acceptance")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--ui-url")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _arguments(argv)
        payload: SmokeSuccess | dict[str, object] = asyncio.run(
            replay_smoke(
                arguments.api_url,
                ui_url=arguments.ui_url,
                timeout=arguments.timeout,
            )
        )
        exit_code = 0
    except SmokeError as error:
        payload = error.payload()
        exit_code = 124 if error.code == "TIMEOUT" else 2 if error.code.startswith("INVALID_") else 1
    print(json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
