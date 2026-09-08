"""Server-signed session identity and explicitly trusted proxy address resolution."""

import base64
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network, ip_address

from pydantic import SecretStr
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from deepresearch.runtime.manager import owner_scope_sha256

from .error_handlers import APIError, error_response, handle_error

_COOKIE = "dr_session"
_COOKIE_PATTERN = re.compile(r"([A-Za-z0-9_-]{43})\.([0-9a-f]{64})")


@dataclass(frozen=True)
class OwnerIdentity:
    client_ip: str
    session_id: str
    owner_scope_sha256: str


class TrustedClientIpResolver:
    def __init__(self, trusted_proxy_cidrs: tuple[IPv4Network | IPv6Network, ...] = ()) -> None:
        self.trusted_proxy_cidrs = trusted_proxy_cidrs

    def _trusted(self, address: IPv4Address | IPv6Address) -> bool:
        return any(address in network for network in self.trusted_proxy_cidrs)

    @staticmethod
    def _literal(value: str) -> IPv4Address | IPv6Address:
        # Zone identifiers are interface-local, not client IP identities.
        if "%" in value:
            raise ValueError("scoped address")
        return ip_address(value)

    def _forwarded(self, entries: list[str]) -> list[IPv4Address | IPv6Address]:
        addresses: list[IPv4Address | IPv6Address] = []
        for entry in entries:
            values: list[str] = []
            for parameter in entry.split(";"):
                name, separator, value = parameter.strip().partition("=")
                if not separator or not name or not value:
                    raise ValueError("malformed forwarded parameter")
                if name.lower() == "for":
                    values.append(value)
            if len(values) != 1:
                raise ValueError("missing or repeated forwarded address")
            value = values[0]
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            addresses.append(self._literal(value))
        return addresses

    def resolve(self, request: Request) -> str:
        try:
            if request.client is None:
                raise ValueError("missing peer")
            peer = self._literal(request.client.host)
        except ValueError:
            raise APIError("INVALID_CLIENT_IP") from None
        if not self._trusted(peer):
            return str(peer)
        chains: list[list[IPv4Address | IPv6Address]] = []
        try:
            for header in ("x-forwarded-for", "forwarded"):
                values = request.headers.getlist(header)
                if not values:
                    continue
                entries = [part.strip() for value in values for part in value.split(",")]
                if not entries or len(entries) > 16:
                    raise ValueError("invalid hop count")
                chains.append(
                    [self._literal(value) for value in entries]
                    if header == "x-forwarded-for"
                    else self._forwarded(entries)
                )
            if len(chains) == 2 and chains[0] != chains[1]:
                raise ValueError("conflicting forwarded address chains")
        except ValueError:
            raise APIError("INVALID_FORWARDED_HEADER") from None
        if not chains:
            return str(peer)
        chain = chains[0]
        for address in reversed(chain):
            if not self._trusted(address):
                return str(address)
        return str(chain[0])


class OwnerSessionMiddleware:
    """Validate/issue the signed dr_session cookie and set request.state.owner.

    Pure ASGI middleware preserves streaming and disconnect behavior for SSE.
    The secret and trust list must come exclusively from server configuration.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        session_secret: bytes | SecretStr,
        ip_resolver: TrustedClientIpResolver,
        secure: bool = False,
    ) -> None:
        if isinstance(session_secret, SecretStr):
            session_secret = session_secret.get_secret_value().encode("utf-8")
        if len(session_secret) < 32:
            raise ValueError("session secret must contain at least 32 bytes")
        self.app = app
        self.session_secret = session_secret
        self.ip_resolver = ip_resolver
        self.secure = secure

    def _signature(self, session: str) -> str:
        return hmac.new(self.session_secret, session.encode("ascii"), hashlib.sha256).hexdigest()

    def _session(self, cookie: str | None) -> tuple[str, bool]:
        match = _COOKIE_PATTERN.fullmatch(cookie or "")
        if match is not None:
            session, signature = match.groups()
            raw = base64.urlsafe_b64decode(session + "=")
            canonical = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
            if canonical == session and hmac.compare_digest(signature, self._signature(session)):
                return session, False
        return secrets.token_urlsafe(32), True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        try:
            client_ip = self.ip_resolver.resolve(request)
        except APIError as error:
            await error_response(error)(scope, receive, send)
            return
        session, issue = self._session(request.cookies.get(_COOKIE))
        request.state.owner = OwnerIdentity(
            client_ip=client_ip,
            session_id=session,
            owner_scope_sha256=owner_scope_sha256(client_ip=client_ip, session_id=session),
        )
        started = False

        async def send_with_cookie(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = list(message.get("headers", []))
                if issue:
                    response = Response()
                    response.set_cookie(
                        _COOKIE,
                        f"{session}.{self._signature(session)}",
                        httponly=True,
                        secure=self.secure,
                        samesite="lax",
                        path="/",
                    )
                    headers.extend(
                        (key, value) for key, value in response.raw_headers if key == b"set-cookie"
                    )
                headers = [(key, value) for key, value in headers if key != b"cache-control"]
                headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_cookie)
        except Exception as error:
            if started:
                # A streaming response cannot be replaced once headers are sent.
                raise
            response = await handle_error(request, error)
            await response(scope, receive, send_with_cookie)
