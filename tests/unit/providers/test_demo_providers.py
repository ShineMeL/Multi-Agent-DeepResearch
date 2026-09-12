import json
import math
import time

import httpx
import pytest
from pydantic import SecretStr

from deepresearch.runtime import CancellationToken, OperationCancelled
from tests.unit.providers.test_openai_compatible import ResearchPlan, _request, _response


async def test_kimi_instant_uses_supported_wire_contract_and_reports_cached_tokens():
    from deepresearch.providers.kimi import KimiInstantModelProvider

    received = []

    def respond(request):
        received.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer fixture-key"
        payload = _response('{"objective":"Compare planner strategies"}')
        payload["usage"]["cached_tokens"] = 4
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = KimiInstantModelProvider(
            base_url="https://api.moonshot.cn/v1", api_key=SecretStr("fixture-key"), client=http
        )
        request = _request().model_copy(update={"model_id": "kimi-k2.6"})
        result = await provider.structured(
            request,
            ResearchPlan,
            deadline=time.monotonic() + 10,
            cancellation_token=CancellationToken(),
        )
        assert isinstance(result.output, ResearchPlan)
        assert result.output.objective == "Compare planner strategies"
        assert result.usage.cached_tokens == 4
        body = received[0]
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_completion_tokens"] == 100
        assert "temperature" not in body
        assert "seed" not in body
        assert "max_tokens" not in body
        assert body["response_format"]["type"] == "json_schema"
        assert "fixture-key" not in json.dumps(body)


@pytest.mark.parametrize(
    "query,related,unrelated",
    [
        (
            "planner evidence ranking",
            "planner evidence ranking and search strategies",
            "banana bread recipe",
        ),
        ("多模态智能体规划", "多模态智能体规划与证据筛选方法", "足球世界杯比赛结果"),
    ],
)
async def test_lexical_demo_embedder_scores_shared_terms_above_unrelated_text(
    query, related, unrelated
):
    from deepresearch.providers.lexical import LexicalHashTextEmbedder

    embedder = LexicalHashTextEmbedder()
    vectors = await embedder.embed(
        (query, related, unrelated),
        deadline=time.monotonic() + 10,
        cancellation_token=CancellationToken(),
    )
    positive = sum(a * b for a, b in zip(vectors[0], vectors[1], strict=True))
    negative = sum(a * b for a, b in zip(vectors[0], vectors[2], strict=True))
    assert positive > negative + 0.3
    assert all(math.isclose(math.hypot(*v), 1.0) for v in vectors)
    assert embedder.model_id == "lexical-hash-v1"
    assert embedder.network_calls == 0
    assert vectors == await embedder.embed(
        (query, related, unrelated),
        deadline=time.monotonic() + 10,
        cancellation_token=CancellationToken(),
    )


async def test_lexical_demo_embedder_respects_cancellation():
    from deepresearch.providers.lexical import LexicalHashTextEmbedder

    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        await LexicalHashTextEmbedder().embed(
            ("planner",),
            deadline=time.monotonic() + 10,
            cancellation_token=token,
        )
