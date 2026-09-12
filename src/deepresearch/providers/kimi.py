"""Kimi K2 instant wire profile (vendor-fixed sampling, no seed support).

This explicit provider identity differs from generic OpenAI-compatible sampling.
K2.5/K2.6 instant mode uses the vendor's fixed temperature (0.6); requests are
not advertised as seed-reproducible. See https://platform.kimi.com/docs/api/chat.
"""

from decimal import Decimal
from typing import override

from pydantic import JsonValue

from deepresearch.providers import ModelRequest, ProviderError

from .openai_compatible import OpenAICompatibleModelProvider


class KimiInstantModelProvider(OpenAICompatibleModelProvider):
    def normalize_request(self, request: ModelRequest) -> ModelRequest:
        # The service rejects user-configured seeds at admission. FixedPlanner
        # has a legacy internal seed=0; normalize it before cache/audit hashing.
        return request.model_copy(update={"temperature": Decimal("0.6"), "seed": None})

    @staticmethod
    @override
    def _request_payload(
        request: ModelRequest,
        *,
        stream: bool,
        response_format: dict[str, JsonValue] | None = None,
    ) -> dict[str, object]:
        if request.model_id not in {"kimi-k2.5", "kimi-k2.6"} or request.seed is not None:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider="kimi-instant",
                operation="model",
                public_message="Kimi instant requires a supported K2 model and no seed",
                retryable=False,
            )
        payload = OpenAICompatibleModelProvider._request_payload(
            request, stream=stream, response_format=response_format
        )
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        payload.pop("temperature")
        payload["thinking"] = {"type": "disabled"}
        return payload
