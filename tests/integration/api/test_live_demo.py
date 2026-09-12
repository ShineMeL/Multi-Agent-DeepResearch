"""Production API/Core/provider adapters with only external HTTP replaced.

These are contract tests, not evidence that paid credentials were exercised.
"""

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx

from apps.api.demo import prepare_demo
from apps.api.main import create_app
from apps.ui.replay import replay_payload
from deepresearch.runtime.manifest import RunManifest
from deepresearch.workflow.runner import BaselineRuntimeHooks
from tests.integration.replay.test_baseline_graph import ControlledSegmentClock
from tests.unit.providers.test_httpx_fetcher import ChunkStream
from tests.unit.providers.test_openai_compatible import _response

ARTICLE = (
    "Planner evidence ranking strategies improve research. "
    "A fixed planner decomposes the question into information needs before searching. "
    "The planner selects sources and evidence ranking identifies relevant passages. "
    "A research report cites those passages so readers can inspect the original sources. "
    "This technical guide describes planner strategies, evidence selection and search "
    "budgets for comparing agents using reproducible evaluation."
)


def plan_text(aligned_evidence):
    records = [
        json.loads(line)
        for line in Path("tests/fixtures/replay/baseline/model_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    record = next(r for r in records if r["key"]["prompt_version"] == "fixed-planner-v1")
    plan = json.loads(record["outcome"]["response"]["output"])
    plan["created_by_model"] = "kimi-k2.6"
    plan["subquestions"] = plan["subquestions"][:1]
    subquestion = plan["subquestions"][0]
    subquestion["importance"] = 1.0
    subquestion["evidence_requirements"]["min_independent_sources"] = 1
    subquestion["evidence_requirements"]["allowed_source_types"] = ["unknown", "paper"]
    if aligned_evidence:
        subquestion["information_needs"][0]["text"] = ARTICLE
    return json.dumps(plan)


@pytest.mark.parametrize("aligned_evidence", [True, False])
async def test_live_demo_runs_real_adapters_and_preserves_effective_parameters(
    tmp_path, monkeypatch, aligned_evidence
):
    monkeypatch.setenv("MODEL_API_KEY", "fixture-live-model-key")
    monkeypatch.setenv("SEARCH_API_KEY", "fixture-live-search-key")
    prepared = prepare_demo(Path.cwd(), tmp_path, environ={})
    app = create_app(prepared.settings)
    clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
    monkeypatch.setattr(
        "deepresearch.runtime.runner_factory.paired_runtime_hooks",
        lambda: BaselineRuntimeHooks(monotonic=clock.monotonic, utc_now=clock.utc_now),
    )
    bodies = []

    def model_response(request):
        payload = json.loads(request.content)
        bodies.append(payload)
        assert payload["thinking"] == {"type": "disabled"}
        assert "seed" not in payload
        prompt = payload["messages"][-1]["content"]
        if '"evidence":' in prompt:
            evidence_id = re.search(r"E-[a-f0-9]{64}", prompt).group()
            text = f"规划策略应根据证据选择搜索方向 [{evidence_id}]。"
        elif '"subquestion":' in prompt:
            text = json.dumps({"queries": ["planner evidence ranking strategies"]})
        else:
            text = plan_text(aligned_evidence)
        response = _response(text)
        response["usage"]["cached_tokens"] = 2
        return httpx.Response(200, json=response)

    fetched = []

    @asynccontextmanager
    async def fetch_stream(self, url, *, pinned_ip, deadline, cancellation_token):
        assert pinned_ip == "93.184.216.34"
        cancellation_token.raise_if_cancelled()
        fetched.append(url)
        html = (
            "<html><head><title>Planner evidence ranking strategies</title></head>"
            f"<body><article><p>{ARTICLE}</p></article></body></html>"
        ).encode()
        response = httpx.Response(
            200,
            headers={"content-type": "text/html"},
            stream=ChunkStream((html,)),
            request=httpx.Request("GET", url),
            extensions={"peer_ip": pinned_ip},
        )
        try:
            yield response
        finally:
            await response.aclose()

    monkeypatch.setattr(
        "deepresearch.providers.httpx_transport.PinnedPeerTransport.stream", fetch_stream
    )
    with respx.mock(assert_all_called=True) as router:
        router.post("https://api.moonshot.cn/v1/chat/completions").mock(side_effect=model_response)
        router.post("https://api.tavily.com/search").respond(
            200,
            json={
                "results": [
                    {
                        "url": "https://93.184.216.34/guide",
                        "title": "Planner evidence ranking strategies",
                        "content": "Planner evidence ranking strategies and source selection.",
                        "score": 0.9,
                    }
                ]
            },
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client,
        ):
            body = replay_payload(
                "比较智能体的规划策略", report_language="zh", provider_profile_id="live-default"
            )
            body["request"]["execution_mode"] = "live"
            body["seed"] = None
            accepted = await client.post(
                "/runs", json=body, headers={"Idempotency-Key": "live-demo"}
            )
            assert accepted.status_code == 202, accepted.text
            run_id = accepted.json()["run_id"]
            await asyncio.wait_for(app.state.manager.wait(run_id), 30)
            final = await client.get(f"/runs/{run_id}")
            assert final.json()["status"] == "completed", final.text
            assert final.json()["stop_reason"] == (
                "SUFFICIENT" if aligned_evidence else "BLOCKED"
            ), final.text
            assert final.json()["is_partial"] is not aligned_evidence
            assert final.json()["final_usage"]["cost_usd"] is None
            report = await client.get(f"/runs/{run_id}/artifacts/report")
            assert "规划策略" in report.text and "https://93.184.216.34/guide" in report.text
            response = await client.get(f"/runs/{run_id}/artifacts/manifest")
            manifest = RunManifest.model_validate_json(response.content)
            assert manifest.provider_profiles[0].execution_mode == "live"
            model_calls = [c for c in manifest.provider_calls if c.operation == "model"]
            assert len(model_calls) >= 3
            assert all(c.temperature == Decimal("0.6") for c in model_calls)
            assert all(c.seed is None for c in model_calls)
            assert all(c.usage.cached_tokens == 2 for c in model_calls)
            assert {c.operation for c in manifest.provider_calls} >= {
                "model",
                "search",
                "fetch",
                "parse",
                "embed",
            }
            assert fetched
            assert "fixture-live-model-key" not in response.text
            assert "fixture-live-search-key" not in response.text
            assert any("untrusted" in json.dumps(b).lower() for b in bodies)
            assert "Report language: zh" in bodies[-1]["messages"][0]["content"]
