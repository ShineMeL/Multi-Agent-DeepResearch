import ipaddress

import pytest

from deepresearch.retrieval import (
    URLSecurityError,
    canonicalize_url,
    validate_public_http_url,
)


def test_canonical_url_removes_fragment_tracking_and_default_port() -> None:
    assert (
        canonicalize_url("HTTPS://Example.COM:443/a?utm_source=x&b=2&a=1#top")
        == "https://example.com/a?a=1&b=2"
    )


def test_canonical_url_idna_normalizes_host_and_preserves_repeated_query_values() -> None:
    canonical = canonicalize_url("http://BÜCHER.example:80/p?tag=z&tag=a&gclid=ignored&q=a+b")

    assert canonical == "http://xn--bcher-kva.example/p?q=a+b&tag=a&tag=z"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "https:///missing-host",
        "https://user:pass@example.com/path",
        "https://example.com:bad/path",
        "https://example.com:0/path",
        "https://exa mple.com/path",
    ],
)
def test_canonical_url_rejects_unsupported_or_ambiguous_syntax(url: str) -> None:
    with pytest.raises(URLSecurityError) as raised:
        canonicalize_url(url)

    assert raised.value.code == "URL_NOT_PUBLIC"
    assert raised.value.public_message


def test_validate_public_url_requires_at_least_one_resolved_address() -> None:
    with pytest.raises(URLSecurityError, match="resolved"):
        validate_public_http_url("https://example.com", resolved_ips=())


@pytest.mark.parametrize(
    "host,address",
    [
        ("localhost", "93.184.216.34"),
        ("example.com", "127.0.0.1"),
        ("example.com", "10.0.0.1"),
        ("example.com", "169.254.1.1"),
        ("example.com", "224.0.0.1"),
        ("example.com", "240.0.0.1"),
        ("example.com", "0.0.0.0"),
        ("example.com", "::1"),
        ("example.com", "::ffff:127.0.0.1"),
    ],
)
def test_validate_public_url_rejects_non_public_hosts_or_addresses(
    host: str, address: str
) -> None:
    with pytest.raises(URLSecurityError, match="public"):
        validate_public_http_url(
            f"https://{host}/path",
            resolved_ips=(ipaddress.ip_address(address),),
        )


def test_validate_public_url_rejects_entire_resolution_if_any_address_is_unsafe() -> None:
    with pytest.raises(URLSecurityError, match="public"):
        validate_public_http_url(
            "https://example.com/path",
            resolved_ips=(
                ipaddress.ip_address("93.184.216.34"),
                ipaddress.ip_address("192.168.1.5"),
            ),
        )


def test_validate_public_url_returns_canonical_boundary_value() -> None:
    result = validate_public_http_url(
        "HTTPS://Example.COM:443/path?utm_medium=x&b=2&a=1#fragment",
        resolved_ips=(ipaddress.ip_address("93.184.216.34"),),
    )

    assert result == "https://example.com/path?a=1&b=2"
