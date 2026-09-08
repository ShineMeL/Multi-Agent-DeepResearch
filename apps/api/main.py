"""Uvicorn factory and the service's resource-owning lifespan."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from copy import copy
from datetime import UTC, datetime
from ipaddress import ip_network

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn.logging import ColourizedFormatter

from deepresearch.runtime.checkpointers import open_service_checkpointer
from deepresearch.runtime.limits import LimitManager
from deepresearch.runtime.manager import RunManager
from deepresearch.runtime.runner_factory import (
    DefaultCoreRunnerBuilder,
    EnvCredentialResolver,
    FilePricingCatalog,
    FileProviderRouteCatalog,
    LangGraphServiceRunnerFactory,
    default_provider_constructors,
)
from deepresearch.security import redact, wrap_untrusted_content
from deepresearch.security.logging import RedactingFilter
from deepresearch.storage import LocalArtifactStore, LocalEvidenceStore
from deepresearch.storage.migrations.runner import upgrade_service_schema
from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

from .error_handlers import APIError, error_response, install_error_handlers
from .health import router as health_router
from .identity import OwnerSessionMiddleware, TrustedClientIpResolver
from .routes_events import router as events_router
from .routes_runs import router as runs_router
from .settings import ServiceSettings


class AdmissionGate:
    """Pure ASGI admission gate; pass SSE bodies and disconnects through intact."""

    def __init__(self, app: ASGIApp, *, service: FastAPI) -> None:
        self.app = app
        self.service = service

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "").rstrip("/")
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and (path == "/runs" or (path.startswith("/runs/") and path.endswith("/resume")))
            and not self.service.state.accepting_runs
        ):
            await error_response(APIError("SERVICE_SHUTDOWN"))(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _RedactingOutputFormatter(logging.Formatter):
    """Let Uvicorn consume its structured arguments, then sanitize the output."""

    def __init__(self, delegate: logging.Formatter, secrets: tuple[str, ...]) -> None:
        super().__init__()
        self.delegate = delegate
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        # Formatting can cache message/exception text; do not mutate the record
        # that other handlers will receive. Only the sanitized string is emitted.
        rendered = self.delegate.format(copy(record))
        return str(redact(rendered, secrets=self.secrets))


def _install_redaction(secrets: tuple[str, ...]) -> None:
    # Handler filters also protect child loggers and propagated records. Logger
    # filters alone do not run on records emitted by their descendants.
    names = {"deepresearch", "uvicorn", "uvicorn.error", "uvicorn.access"}
    names.update(
        name
        for name in logging.root.manager.loggerDict
        if name.startswith(("deepresearch.", "uvicorn."))
    )
    handlers: set[logging.Handler] = set()
    for name in names:
        logger: logging.Logger | None = logging.getLogger(name)
        while logger is not None:
            handlers.update(logger.handlers)
            logger = logger.parent if logger.propagate else None
    if logging.lastResort is not None:
        handlers.add(logging.lastResort)
    filter_ = RedactingFilter(secrets=secrets)
    for handler in handlers:
        formatter = handler.formatter
        if isinstance(formatter, _RedactingOutputFormatter):
            formatter.secrets = tuple(dict.fromkeys((*formatter.secrets, *secrets)))
        elif isinstance(formatter, ColourizedFormatter):
            # AccessFormatter needs five args; DefaultFormatter can interpolate
            # color_message with those original args. A flattening record filter
            # breaks both contracts, so redact after the formatter instead.
            handler.setFormatter(_RedactingOutputFormatter(formatter, secrets))
        else:
            handler.addFilter(filter_)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: ServiceSettings = app.state.settings
    app.state.accepting_runs = False
    app.state.checkpointer_ready = False
    secrets = settings.loaded_secret_values()
    _install_redaction(secrets)
    try:
        artifact_root = settings.artifact_root.resolve()
        checkpoint_path = settings.checkpoint_sqlite_path
        if not checkpoint_path.resolve().is_relative_to(artifact_root):
            raise ValueError("checkpoint_sqlite_path must be inside artifact_root")
        artifact_root.mkdir(parents=True, exist_ok=True)
        app.state.artifact_root = artifact_root
        app.state.artifact_store = LocalArtifactStore(artifact_root)
        app.state.evidence_store = LocalEvidenceStore(artifact_root)
        async with open_service_checkpointer(
            database_url=settings.database_url,
            sqlite_path=checkpoint_path,
        ) as checkpointer:
            app.state.checkpointer_ready = True
            await app.state.schema_upgrader(app.state.store.engine)
            await app.state.store.reconcile_startup(datetime.now(UTC))
            limits = LimitManager(app.state.store, daily_limit=settings.daily_cost_limit_usd)
            route_catalog = FileProviderRouteCatalog.load(settings.provider_profile_catalog_path)
            # All selected profiles must exist and agree with the server policy;
            # credentials resolve only through the dedicated provider allowlist.
            settings.validate_route_catalog(route_catalog)
            builder = DefaultCoreRunnerBuilder(
                provider_constructors=default_provider_constructors(),
                credential_resolver=EnvCredentialResolver(
                    frozenset(settings.provider_credential_env_names),
                ),
                artifact_store=app.state.artifact_store,
                evidence_store=app.state.evidence_store,
                content_boundary=wrap_untrusted_content,
                search_slot=limits.search_slot,
                host_slot=limits.fetch_slot,
                secrets=secrets,
            )
            app.state.manager = RunManager(
                runner_factory=LangGraphServiceRunnerFactory(builder, route_catalog),
                store=app.state.store,
                checkpointer=checkpointer,
                pricing_catalog=FilePricingCatalog.load(settings.pricing_catalog_path),
                deployment_policy=app.state.deployment_policy,
                admission=limits,
                secrets=secrets,
            )
            app.state.accepting_runs = True
            try:
                yield
            finally:
                app.state.accepting_runs = False
                await app.state.manager.shutdown(grace_seconds=20.0)
    finally:
        app.state.accepting_runs = False
        app.state.checkpointer_ready = False
        await app.state.store.engine.dispose()


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    # BaseSettings loads the required fields from the environment at runtime.
    settings = ServiceSettings() if settings is None else settings  # pyright: ignore[reportCallIssue]
    _install_redaction(settings.loaded_secret_values())
    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.accepting_runs = False
    app.state.checkpointer_ready = False
    app.state.artifact_root = settings.artifact_root.resolve()
    app.state.store = SqlAlchemyRunStore(settings.database_url, app.state.artifact_root)
    app.state.schema_upgrader = upgrade_service_schema
    app.state.deployment_policy = settings.deployment_policy()
    install_error_handlers(app)

    app.add_middleware(AdmissionGate, service=app)
    app.add_middleware(
        OwnerSessionMiddleware,
        session_secret=settings.session_signing_key,
        ip_resolver=TrustedClientIpResolver(
            tuple(ip_network(cidr) for cidr in settings.trusted_proxy_cidrs)
        ),
        secure=settings.cookie_secure,
    )
    app.include_router(runs_router)
    app.include_router(events_router)
    app.include_router(health_router)
    return app
