"""Injectable HTTP composition; the hosting lifespan owns manager/store shutdown."""

from ipaddress import IPv4Network, IPv6Network

from fastapi import FastAPI

from deepresearch.runtime.deployment_policy import DeploymentPolicy
from deepresearch.runtime.manager import RunManager
from deepresearch.storage import LocalArtifactStore

from .error_handlers import install_error_handlers
from .identity import OwnerSessionMiddleware, TrustedClientIpResolver
from .routes_events import router as events_router
from .routes_runs import router


def create_app(
    *,
    manager: RunManager,
    deployment_policy: DeploymentPolicy,
    artifact_store: LocalArtifactStore,
    session_secret: bytes,
    trusted_proxy_cidrs: tuple[IPv4Network | IPv6Network, ...] = (),
) -> FastAPI:
    if len(session_secret) < 32:
        raise ValueError("session secret must contain at least 32 bytes")
    app = FastAPI()
    app.state.manager = manager
    app.state.deployment_policy = deployment_policy
    app.state.artifact_store = artifact_store
    install_error_handlers(app)
    app.add_middleware(
        OwnerSessionMiddleware,
        session_secret=session_secret,
        ip_resolver=TrustedClientIpResolver(trusted_proxy_cidrs),
        secure=deployment_policy.forced_access_profile == "public_live",
    )
    app.include_router(router)
    app.include_router(events_router)
    return app


__all__ = ["create_app"]
