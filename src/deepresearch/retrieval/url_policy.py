import ipaddress
import re
from collections.abc import Sequence
from typing import Literal, NewType
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_QUERY_KEYS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
    }
)

CanonicalURL = NewType("CanonicalURL", str)
_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_VALID_PERCENT_ESCAPE = re.compile(r"%(?:[0-9A-Fa-f]{2})")


class URLSecurityError(ValueError):
    code: Literal["URL_NOT_PUBLIC"] = "URL_NOT_PUBLIC"

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


def _reject_invalid_percent_escapes(value: str) -> None:
    without_escapes = _VALID_PERCENT_ESCAPE.sub("", value)
    if "%" in without_escapes:
        raise URLSecurityError("URL contains an invalid percent escape")


def _canonical_host(host: str) -> str:
    if not host or any(character.isspace() for character in host):
        raise URLSecurityError("URL must contain a valid host")
    if "\\" in host or "%" in host:
        raise URLSecurityError("URL must contain a valid host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            result = host.encode("idna").decode("ascii").rstrip(".").lower()
        except UnicodeError as error:
            raise URLSecurityError("URL must contain a valid host") from error
        labels = result.split(".")
        if (
            not result
            or len(result) > 253
            or any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                for label in labels
            )
        ):
            raise URLSecurityError("URL must contain a valid host")
        return result
    return address.compressed.lower()


def canonicalize_url(url: str) -> str:
    """Return a deterministic HTTP(S) URL without tracking-only components."""
    if not url or url != url.strip():
        raise URLSecurityError("URL must be a non-empty HTTP(S) URL")
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise URLSecurityError("URL contains invalid control characters")
    _reject_invalid_percent_escapes(url)
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            raise URLSecurityError("URL scheme must be HTTP or HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise URLSecurityError("URL credentials are not allowed")
        if parsed.hostname is None:
            raise URLSecurityError("URL must contain a host")
        host = _canonical_host(parsed.hostname)
        port = parsed.port
        if port == 0:
            raise URLSecurityError("URL port must be between 1 and 65535")
    except URLSecurityError:
        raise
    except (UnicodeError, ValueError) as error:
        raise URLSecurityError("URL syntax is invalid") from error

    default_port = 80 if scheme == "http" else 443
    display_host = f"[{host}]" if ":" in host else host
    netloc = display_host if port is None or port == default_port else f"{display_host}:{port}"
    try:
        pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False)
    except ValueError as error:
        raise URLSecurityError("URL query syntax is invalid") from error
    filtered_pairs = sorted(
        (key, value) for key, value in pairs if key.casefold() not in TRACKING_QUERY_KEYS
    )
    query = urlencode(filtered_pairs, doseq=True)
    return urlunsplit((scheme, netloc, parsed.path, query, ""))


def _is_public_address(address: _IPAddress) -> bool:
    mapped = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
    checked: _IPAddress = mapped if mapped is not None else address
    return not (
        checked.is_loopback
        or checked.is_private
        or checked.is_link_local
        or checked.is_multicast
        or checked.is_reserved
        or checked.is_unspecified
    ) and checked.is_global


def validate_public_http_url(
    url: str,
    *,
    resolved_ips: Sequence[_IPAddress],
) -> CanonicalURL:
    """Validate one already-resolved URL at a network access boundary.

    Fetchers remain responsible for resolving and revalidating every redirect.
    """
    canonical = canonicalize_url(url)
    parsed = urlsplit(canonical)
    host = parsed.hostname
    if host is None:
        raise URLSecurityError("URL must contain a host")
    normalized_host = host.rstrip(".").casefold()
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise URLSecurityError("URL host must be public")
    try:
        literal_address = ipaddress.ip_address(host)
    except ValueError:
        literal_address = None
    if literal_address is not None and not _is_public_address(literal_address):
        raise URLSecurityError("URL host must be public")
    if not resolved_ips:
        raise URLSecurityError("URL must have at least one resolved IP address")
    for address in resolved_ips:
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            address, (ipaddress.IPv4Address, ipaddress.IPv6Address)
        ):
            raise URLSecurityError("URL resolved addresses are invalid")
        if not _is_public_address(address):
            raise URLSecurityError("URL and all resolved addresses must be public")
    return CanonicalURL(canonical)


__all__ = [
    "TRACKING_QUERY_KEYS",
    "CanonicalURL",
    "URLSecurityError",
    "canonicalize_url",
    "validate_public_http_url",
]
