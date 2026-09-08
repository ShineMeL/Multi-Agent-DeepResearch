"""The service constructor must continue to use Core's pinned, validated fetcher."""

import inspect
import ipaddress
import json
import time
from pathlib import Path

import pytest

from deepresearch.providers import ProviderError
from deepresearch.providers.httpx_fetcher import HttpxFetcher, no_op_host_slot
from deepresearch.providers.httpx_transport import PinnedPeerTransport
from deepresearch.retrieval import URLSecurityError, canonicalize_url, validate_public_http_url
from deepresearch.runtime import CancellationToken
from deepresearch.runtime.runner_factory import FrozenProviderRoute, default_provider_constructors
from tests.unit.providers.test_httpx_fetcher import ResponseSpec, SequenceTransport, _resolver


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "10.0.0.1", "::1"])
def test_core_policy_still_rejects_private_resolution(address):
    with pytest.raises(URLSecurityError):
        validate_public_http_url(
            "https://attacker.example/path",
            resolved_ips=[ipaddress.ip_address(address)],
        )


def test_service_keeps_core_url_policy_contract():
    assert tuple(inspect.signature(validate_public_http_url).parameters) == ("url", "resolved_ips")
    assert canonicalize_url("HTTPS://Example.COM:443/a#x") == "https://example.com/a"


async def test_service_fetcher_revalidates_redirect_dns_before_connecting():
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/security/malicious_redirect.json").read_text()
    )
    route = FrozenProviderRoute(
        operation="fetch",
        provider_id="httpx-fetcher",
        endpoint_type="fetch",
        model_id=None,
        model_revision=None,
        base_url=None,
        credential_ref=None,
        fallback_rank=0,
        parameters={},
    )
    fetcher = default_provider_constructors()["httpx-fetcher"](route, None, no_op_host_slot)
    assert isinstance(fetcher, HttpxFetcher)
    assert isinstance(fetcher._transport, PinnedPeerTransport)
    transport = SequenceTransport(
        ResponseSpec(
            302,
            {"location": fixture["location"]},
            (),
            fixture["initial_ips"][0],
        )
    )
    fetcher._transport = transport
    fetcher._resolver = _resolver(
        {
            "public.example": tuple(fixture["initial_ips"]),
            "redirect.example": tuple(fixture["redirect_ips"]),
        }
    )
    with pytest.raises(ProviderError) as error:
        await fetcher.fetch(
            fixture["initial_url"],
            deadline=time.monotonic() + 10,
            cancellation_token=CancellationToken(),
        )
    assert error.value.code == "INVALID_REQUEST"
    assert transport.urls == [fixture["initial_url"]]
    assert transport.streams[0].closed
