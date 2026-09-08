"""Service configuration boundaries and composition of the existing Core runner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Never, Protocol, TypeAlias, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, JsonValue, SecretStr, model_validator

from deepresearch.domain import ExecutionMode, ResourceUsage, RunConfig
from deepresearch.evidence.similarity import SimilarityRanker
from deepresearch.planning import FixedPlanner
from deepresearch.providers import (
    Fetcher,
    ModelProvider,
    Parser,
    SearchProvider,
    TextEmbedder,
    UsageReportingSearchProvider,
)
from deepresearch.providers.httpx_fetcher import HostSlot, no_op_host_slot
from deepresearch.reporting import ContentBoundary, MarkdownReportWriter, identity_content_boundary
from deepresearch.runtime.manifest import CostCalculator, PricingSnapshot
from deepresearch.runtime.ports import ResearchRunner
from deepresearch.storage import FileCache, LocalArtifactStore, LocalEvidenceStore
from deepresearch.workflow.baseline_graph import BaselineNodeHandlers, build_baseline_graph
from deepresearch.workflow.runner import LangGraphResearchRunner

_SECRET = re.compile(r"(?i)(authorization|api.?key|password|secret|bearer|cookie|access.?token)")
_PARAMETERS = frozenset(
    {
        "dimension",
        "bundle_path",
        "snapshot_id",
    }
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


class ProviderProfileDrift(RuntimeError):
    code: Literal["PROVIDER_PROFILE_DRIFT"] = "PROVIDER_PROFILE_DRIFT"

    def __init__(self) -> None:
        super().__init__("provider profile is unavailable or no longer permitted")


class ResearchGraphUnavailable(RuntimeError):
    """Core defines research wiring but has no production research node handlers."""

    code: Literal["RESEARCH_GRAPH_UNAVAILABLE"] = "RESEARCH_GRAPH_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__("research-v1 production node handlers are not available")


class _FrozenParameters(dict[str, JsonValue]):
    def _deny(self: object, *args: Any, **kwargs: Any) -> Never:
        raise TypeError("frozen provider parameters cannot be changed")

    __setitem__ = _deny
    __delitem__ = _deny
    __ior__ = _deny
    clear = _deny
    pop = _deny
    popitem = _deny
    setdefault = _deny
    update = _deny


class FrozenProviderRoute(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True
    )

    operation: Literal["model", "search", "fetch", "parse", "embed"]
    provider_id: str = Field(min_length=1)
    endpoint_type: str = Field(min_length=1)
    model_id: str | None
    model_revision: str | None
    base_url: AnyHttpUrl | None
    credential_ref: str | None
    fallback_rank: int = Field(ge=0)
    parameters: dict[str, JsonValue]

    @model_validator(mode="after")
    def nonsecret_configuration(self) -> FrozenProviderRoute:
        if (
            self.credential_ref is not None
            and re.fullmatch(r"[A-Z][A-Z0-9_]*", self.credential_ref) is None
        ):
            raise ValueError("credential_ref must be a logical environment name")
        if self.base_url is not None and (
            self.base_url.username
            or self.base_url.password
            or self.base_url.query
            or self.base_url.fragment
        ):
            raise ValueError("base URL must not contain credentials, query or fragment")
        if set(self.parameters) - _PARAMETERS:
            raise ValueError("route parameters contain unsupported or secret fields")
        allowed: set[str] = {"snapshot_id"} if self.operation in {"search", "fetch"} else set()
        if self.operation != "parse":
            allowed.add("bundle_path")
        if self.operation == "embed" and self.provider_id == "deterministic-hash":
            allowed.add("dimension")
        if set(self.parameters) - allowed:
            raise ValueError("route parameters are not supported by this operation")
        if "dimension" in self.parameters:
            dimension = self.parameters["dimension"]
            if type(dimension) is not int or not 1 <= dimension <= 65536:
                raise ValueError("embedding dimension must be an integer from 1 through 65536")
        for key in ("bundle_path", "snapshot_id"):
            if key in self.parameters and (
                not isinstance(self.parameters[key], str) or not self.parameters[key]
            ):
                raise ValueError("route path and snapshot parameters must be nonempty strings")
        for value in self.parameters.values():
            if type(value) not in {str, int, float, bool, type(None)}:
                raise ValueError("route parameters must be nonsecret scalar values")
            if isinstance(value, str) and (_SECRET.search(value) or value.startswith("sk-")):
                raise ValueError("route parameters must not contain secret values")
        object.__setattr__(self, "parameters", _FrozenParameters(self.parameters))
        return self


class FrozenProviderRoutes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    profile_id: str = Field(min_length=1)
    execution_mode: ExecutionMode
    routes: tuple[FrozenProviderRoute, ...]
    configuration_sha256: str

    @model_validator(mode="after")
    def validate_digest(self) -> FrozenProviderRoutes:
        expected = hashlib.sha256(
            _canonical(self.model_dump(mode="json", exclude={"configuration_sha256"}))
        ).hexdigest()
        if self.configuration_sha256 != expected:
            raise ValueError("provider route configuration digest does not match")
        if self.execution_mode == "replay" and any(
            route.credential_ref is not None or route.base_url is not None for route in self.routes
        ):
            raise ValueError("strict replay must not contain credentials or live endpoints")
        identities = [(route.operation, route.fallback_rank) for route in self.routes]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate operation fallback rank")
        return self


def validate_provider_route_binding(
    config: RunConfig, provider_routes: FrozenProviderRoutes
) -> None:
    try:
        FrozenProviderRoutes.model_validate(provider_routes.model_dump(mode="json"))
    except ValueError:
        raise ProviderProfileDrift() from None
    if (
        config.request.provider_profile_id != provider_routes.profile_id
        or config.request.execution_mode != provider_routes.execution_mode
    ):
        raise ProviderProfileDrift()


class PricingCatalog(Protocol):
    def resolve(self, provider_profile_id: str) -> tuple[PricingSnapshot, ...]: ...


def _profiles(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(cast(dict[str, Any], payload)) != {"profiles"}:
        raise ValueError("catalog must contain only profiles")
    profiles = cast(dict[str, Any], payload)["profiles"]
    if not isinstance(profiles, dict):
        raise TypeError("profiles must be an object")
    return cast(dict[str, Any], profiles)


def _pricing_key(snapshot: PricingSnapshot) -> tuple[str, str, str]:
    return snapshot.provider_id, snapshot.endpoint_type, snapshot.model_id


class FilePricingCatalog:
    def __init__(self, profiles: Mapping[str, tuple[PricingSnapshot, ...]]) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def load(cls, path: Path | None) -> FilePricingCatalog:
        profiles: dict[str, tuple[PricingSnapshot, ...]] = {}
        for profile, raw in ({} if path is None else _profiles(path)).items():
            if not isinstance(raw, list):
                raise TypeError("pricing profile must be an array")
            snapshots = tuple(PricingSnapshot.model_validate(item) for item in cast(list[Any], raw))
            if len({_pricing_key(item) for item in snapshots}) != len(snapshots):
                raise ValueError("duplicate pricing key")
            profiles[profile] = tuple(sorted(snapshots, key=_pricing_key))
        return cls(profiles)

    def resolve(self, provider_profile_id: str) -> tuple[PricingSnapshot, ...]:
        return self._profiles.get(provider_profile_id, ())


class ProviderRouteCatalog(Protocol):
    def resolve(self, provider_profile_id: str) -> FrozenProviderRoutes: ...


class FileProviderRouteCatalog:
    def __init__(self, profiles: Mapping[str, FrozenProviderRoutes]) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def load(cls, path: Path | None) -> FileProviderRouteCatalog:
        raw_profiles: dict[str, Any] = (
            {"replay": {"execution_mode": "replay", "routes": []}}
            if path is None
            else _profiles(path)
        )
        profiles: dict[str, FrozenProviderRoutes] = {}
        for profile_id, raw in raw_profiles.items():
            if not isinstance(raw, dict) or set(cast(dict[str, Any], raw)) != {
                "execution_mode",
                "routes",
            }:
                raise ValueError("route profile must contain execution_mode and routes")
            raw = cast(dict[str, Any], raw)
            if not isinstance(raw["routes"], list):
                raise TypeError("routes must be an array")
            routes: list[FrozenProviderRoute] = []
            for item in cast(list[Any], raw["routes"]):
                if not isinstance(item, dict):
                    raise TypeError("route must be an object")
                item = cast(dict[str, Any], item)
                parameters = item.get("parameters", {})
                if not isinstance(parameters, dict):
                    raise TypeError("parameters must be an object")
                clean = {
                    name: value
                    for name, value in item.items()
                    if name in FrozenProviderRoute.model_fields
                }
                clean["parameters"] = {
                    name: value
                    for name, value in cast(dict[str, Any], parameters).items()
                    if not (_SECRET.search(name) or name.lower() in {"token", "key", "headers"})
                }
                routes.append(FrozenProviderRoute.model_validate(clean))
            ordered = sorted(
                routes,
                key=lambda route: (
                    route.operation,
                    route.fallback_rank,
                    route.provider_id,
                    route.endpoint_type,
                    route.model_id or "",
                    route.model_revision or "",
                ),
            )
            payload = {
                "profile_id": profile_id,
                "execution_mode": raw["execution_mode"],
                "routes": [route.model_dump(mode="json") for route in ordered],
            }
            payload["configuration_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
            profiles[profile_id] = FrozenProviderRoutes.model_validate(payload)
        return cls(profiles)

    def resolve(self, provider_profile_id: str) -> FrozenProviderRoutes:
        try:
            return self._profiles[provider_profile_id]
        except KeyError:
            raise ProviderProfileDrift() from None


ProviderAdapter: TypeAlias = ModelProvider | SearchProvider | Fetcher | Parser | TextEmbedder  # noqa: UP040 - public contract
ProviderConstructor: TypeAlias = Callable[  # noqa: UP040 - public contract
    [FrozenProviderRoute, str | None, HostSlot], ProviderAdapter
]
SearchSlot: TypeAlias = Callable[[], AbstractAsyncContextManager[None]]  # noqa: UP040 - public contract


@asynccontextmanager
async def no_op_search_slot() -> AsyncGenerator[None]:
    yield


class EnvCredentialResolver:
    def __init__(self, allowed_env_names: frozenset[str]) -> None:
        self.allowed_env_names = allowed_env_names

    def resolve(self, credential_ref: str | None) -> str | None:
        if credential_ref is None:
            return None
        if credential_ref not in self.allowed_env_names or not os.environ.get(credential_ref):
            raise ProviderProfileDrift()
        return os.environ[credential_ref]


def default_provider_constructors() -> dict[str, ProviderConstructor]:
    from deepresearch.providers.embeddings import DeterministicHashTextEmbedder
    from deepresearch.providers.httpx_fetcher import HttpxFetcher
    from deepresearch.providers.httpx_transport import PinnedPeerTransport
    from deepresearch.providers.openai_compatible import OpenAICompatibleModelProvider
    from deepresearch.providers.parsers import HtmlParser, PdfParser
    from deepresearch.providers.replay import (
        ReplayFetcher,
        ReplayModelProvider,
        ReplaySearchProvider,
        ReplayTextEmbedder,
    )
    from deepresearch.providers.replay_schema import ReplayBundle
    from deepresearch.providers.tavily import TavilySearchProvider

    def replay(route: FrozenProviderRoute, secret: str | None, slot: HostSlot) -> ProviderAdapter:
        del slot
        if secret is not None or route.credential_ref is not None:
            raise ProviderProfileDrift()
        bundle_path = route.parameters.get("bundle_path")
        if not isinstance(bundle_path, str):
            raise ProviderProfileDrift()
        bundle = ReplayBundle.load(Path(bundle_path))
        if not bundle.verify().valid:
            raise ProviderProfileDrift()
        constructors: dict[str, Any] = {
            "model": ReplayModelProvider,
            "search": ReplaySearchProvider,
            "fetch": ReplayFetcher,
            "embed": ReplayTextEmbedder,
        }
        if route.operation not in constructors:
            raise ProviderProfileDrift()
        snapshot = bundle.provider_snapshot(route.operation)
        if (
            snapshot.provider_id != route.provider_id
            or snapshot.model_id != route.model_id
            or snapshot.model_revision != route.model_revision
        ):
            raise ProviderProfileDrift()
        return cast(ProviderAdapter, constructors[route.operation](bundle))

    def model(route: FrozenProviderRoute, secret: str | None, slot: HostSlot) -> ProviderAdapter:
        del slot
        if route.operation != "model" or route.base_url is None or secret is None:
            raise ProviderProfileDrift()
        return OpenAICompatibleModelProvider(
            base_url=str(route.base_url),
            api_key=SecretStr(secret),
            provider_id=route.provider_id,
            model_revision=route.model_revision or "provider-managed",
        )

    def search(route: FrozenProviderRoute, secret: str | None, slot: HostSlot) -> ProviderAdapter:
        del slot
        if route.operation != "search" or secret is None or route.base_url is None:
            raise ProviderProfileDrift()
        return TavilySearchProvider(api_key=SecretStr(secret), endpoint=str(route.base_url))

    return {
        "replay": replay,
        "openai-compatible": model,
        "tavily": search,
        "html": lambda route, secret, slot: HtmlParser(),
        "trafilatura-html": lambda route, secret, slot: HtmlParser(),
        "pdf": lambda route, secret, slot: PdfParser(),
        "pymupdf-pdf": lambda route, secret, slot: PdfParser(),
        "httpx-fetcher": lambda route, secret, slot: HttpxFetcher(
            transport=PinnedPeerTransport(), host_slot=slot
        ),
        "deterministic-hash": lambda route, secret, slot: DeterministicHashTextEmbedder(
            dimension=cast(int, route.parameters.get("dimension", 384))
        ),
    }


class CoreRunnerBuilder(Protocol):
    def required_pricing_keys(
        self, config: RunConfig, provider_routes: FrozenProviderRoutes
    ) -> set[tuple[str, str, str]]: ...

    def build(
        self,
        *,
        config: RunConfig,
        provider_routes: FrozenProviderRoutes,
        pricing_snapshots: tuple[PricingSnapshot, ...],
        checkpointer: BaseCheckpointSaver[Any],
        cost_calculator: type[CostCalculator],
    ) -> ResearchRunner: ...


class _BoundModel:
    """Supply the frozen identity while keeping Core request/result types intact."""

    def __init__(self, delegate: ModelProvider, route: FrozenProviderRoute) -> None:
        if route.model_id is None or route.model_revision is None:
            raise ProviderProfileDrift()
        self.delegate = delegate
        self.provider_id = delegate.provider_id
        self.model_id = route.model_id
        self.model_revision = route.model_revision

    async def complete(self, request: Any, **kwargs: Any) -> Any:
        return await self.delegate.complete(
            request.model_copy(update={"model_id": self.model_id}), **kwargs
        )

    async def structured(self, request: Any, output_schema: type[Any], **kwargs: Any) -> Any:
        return await self.delegate.structured(
            request.model_copy(update={"model_id": self.model_id}), output_schema, **kwargs
        )

    def stream(self, request: Any, **kwargs: Any) -> Any:
        return self.delegate.stream(
            request.model_copy(update={"model_id": self.model_id}), **kwargs
        )


class _SlottedSearch:
    def __init__(self, delegate: SearchProvider, slot: SearchSlot) -> None:
        self.delegate = delegate
        self.slot = slot
        self.provider_id = delegate.provider_id

    @property
    def last_usage(self) -> ResourceUsage | None:
        return cast(ResourceUsage | None, getattr(self.delegate, "last_usage", None))

    async def search(self, query: str, limit: int, filters: Any, **kwargs: Any) -> Any:
        async with self.slot():
            return await self.delegate.search(query, limit, filters, **kwargs)


class _SlottedUsageSearch(_SlottedSearch):
    async def search_with_usage(self, query: str, limit: int, filters: Any, **kwargs: Any) -> Any:
        async with self.slot():
            return await cast(UsageReportingSearchProvider, self.delegate).search_with_usage(
                query, limit, filters, **kwargs
            )


class _SnapshotCostResolver:
    def __init__(
        self,
        routes: FrozenProviderRoutes,
        snapshots: tuple[PricingSnapshot, ...],
        calculator: type[CostCalculator],
    ) -> None:
        self.routes = routes
        self.snapshots = {_pricing_key(item): item for item in snapshots}
        self.calculator = calculator

    def resolve_cost(
        self,
        *,
        operation: str,
        provider_id: str,
        model_id: str | None,
        outcome: str,
        usage: ResourceUsage,
    ) -> Decimal | None:
        del outcome
        matches = [
            route
            for route in self.routes.routes
            if route.provider_id == provider_id
            and (
                route.model_id == model_id
                or (model_id is None and route.operation in {"search", "fetch", "parse"})
            )
            and (
                route.operation == operation
                or (route.operation == "model" and operation in {"complete", "structured", "model"})
            )
        ]
        if len(matches) != 1:
            return None
        route = matches[0]
        snapshot = self.snapshots.get(
            (
                route.provider_id,
                "complete" if route.operation == "model" else route.operation,
                route.model_id or route.operation,
            )
        )
        return None if snapshot is None else self.calculator.estimate(usage, snapshot).total_usd


def _required_pricing_keys(routes: FrozenProviderRoutes) -> set[tuple[str, str, str]]:
    # Core's method endpoint, not the transport API path, is the audited identity.
    return {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured") if route.operation == "model" else (route.operation,)
        )
    }


def _rates(snapshot: PricingSnapshot) -> tuple[Decimal, ...]:
    return (
        snapshot.input_tokens_per_million_usd,
        snapshot.output_tokens_per_million_usd,
        snapshot.cached_tokens_per_million_usd,
        snapshot.reasoning_tokens_per_million_usd,
    )


def _validate_pricing(routes: FrozenProviderRoutes, snapshots: tuple[PricingSnapshot, ...]) -> None:
    indexed = {_pricing_key(snapshot): snapshot for snapshot in snapshots}
    if len(indexed) != len(snapshots) or not _required_pricing_keys(routes) <= indexed.keys():
        raise ProviderProfileDrift()
    for route in routes.routes:
        if route.operation == "model":
            complete = indexed[(route.provider_id, "complete", route.model_id or "model")]
            structured = indexed[(route.provider_id, "structured", route.model_id or "model")]
            # UsageCostResolver receives operation='model' without a method endpoint.
            # Different rates cannot be resolved honestly with that Core interface.
            if _rates(complete) != _rates(structured):
                raise ProviderProfileDrift()
        elif any(
            _rates(indexed[(route.provider_id, route.operation, route.model_id or route.operation)])
        ):
            # Core provider-call auditing only permits zero/unknown non-model costs.
            raise ProviderProfileDrift()


def _validate_adapter_identity(route: FrozenProviderRoute, adapter: ProviderAdapter) -> None:
    actual = getattr(adapter, "parser_id" if route.operation == "parse" else "provider_id", None)
    if actual != route.provider_id:
        raise ProviderProfileDrift()
    if route.operation in {"model", "embed"}:
        if not route.model_id or not route.model_revision:
            raise ProviderProfileDrift()
        if (
            getattr(adapter, "model_id", route.model_id) != route.model_id
            or getattr(adapter, "model_revision", None) != route.model_revision
        ):
            raise ProviderProfileDrift()


class DefaultCoreRunnerBuilder:
    def __init__(
        self,
        *,
        provider_constructors: Mapping[str, ProviderConstructor],
        credential_resolver: EnvCredentialResolver,
        artifact_store: LocalArtifactStore,
        evidence_store: LocalEvidenceStore,
        content_boundary: ContentBoundary = identity_content_boundary,
        search_slot: SearchSlot = no_op_search_slot,
        host_slot: HostSlot = no_op_host_slot,
    ) -> None:
        self.provider_constructors = dict(provider_constructors)
        self.credential_resolver = credential_resolver
        self.artifact_store = artifact_store
        self.evidence_store = evidence_store
        self.content_boundary = content_boundary
        self.search_slot = search_slot
        self.host_slot = host_slot

    def required_pricing_keys(
        self, config: RunConfig, provider_routes: FrozenProviderRoutes
    ) -> set[tuple[str, str, str]]:
        del config
        return _required_pricing_keys(provider_routes)

    def build(
        self,
        *,
        config: RunConfig,
        provider_routes: FrozenProviderRoutes,
        pricing_snapshots: tuple[PricingSnapshot, ...],
        checkpointer: BaseCheckpointSaver[Any],
        cost_calculator: type[CostCalculator],
    ) -> ResearchRunner:
        validate_provider_route_binding(config, provider_routes)
        if config.workflow_id == "research-v1":
            raise ResearchGraphUnavailable()
        routes = {route.operation: route for route in provider_routes.routes}
        if set(routes) != {"model", "search", "fetch", "parse", "embed"}:
            raise ProviderProfileDrift()
        # Only single routes are currently supported by the Core composition.
        # Reject an unsupported deployment policy before any adapter construction.
        if any(
            route.fallback_rank != 0 or route.provider_id not in self.provider_constructors
            for route in provider_routes.routes
        ):
            raise ProviderProfileDrift()
        if (
            pricing_snapshots
            or config.budget.max_cost_usd is not None
            or config.request.access_profile == "public_live"
            or config.request.run_purpose == "benchmark"
        ):
            _validate_pricing(provider_routes, pricing_snapshots)
        secrets = {
            route.operation: self.credential_resolver.resolve(route.credential_ref)
            for route in provider_routes.routes
        }
        adapters = {
            operation: self.provider_constructors[route.provider_id](
                route, secrets[operation], self.host_slot
            )
            for operation, route in routes.items()
        }
        for operation, adapter in adapters.items():
            frozen = routes[operation]
            _validate_adapter_identity(frozen, adapter)
            if "bundle_path" in frozen.parameters:
                from deepresearch.providers.replay import (
                    ReplayFetcher,
                    ReplayModelProvider,
                    ReplaySearchProvider,
                    ReplayTextEmbedder,
                )

                if not isinstance(
                    adapter,
                    (ReplayFetcher, ReplayModelProvider, ReplaySearchProvider, ReplayTextEmbedder),
                ):
                    raise ProviderProfileDrift()
            if (
                "dimension" in frozen.parameters
                and getattr(adapter, "dimension", None) != frozen.parameters["dimension"]
            ):
                raise ProviderProfileDrift()
            if provider_routes.execution_mode == "replay":
                from deepresearch.providers.embeddings import DeterministicHashTextEmbedder
                from deepresearch.providers.parsers import HtmlParser, PdfParser
                from deepresearch.providers.replay import (
                    ReplayFetcher,
                    ReplayModelProvider,
                    ReplaySearchProvider,
                    ReplayTextEmbedder,
                )

                if not isinstance(
                    adapter,
                    (
                        ReplayFetcher,
                        ReplayModelProvider,
                        ReplaySearchProvider,
                        ReplayTextEmbedder,
                        DeterministicHashTextEmbedder,
                        HtmlParser,
                        PdfParser,
                    ),
                ):
                    raise ProviderProfileDrift()
        model = _BoundModel(cast(ModelProvider, adapters["model"]), routes["model"])
        planner = FixedPlanner(
            model=model,
            artifact_store=self.artifact_store,
            budget=config.budget,
            content_boundary=self.content_boundary,
            prompt_version=config.prompt_versions.get("planner", "fixed-planner-v1"),
        )
        writer = MarkdownReportWriter(
            self.evidence_store, model=model, content_boundary=self.content_boundary
        )
        search = cast(SearchProvider, adapters["search"])
        slotted_search = (
            _SlottedUsageSearch(search, self.search_slot)
            if isinstance(search, UsageReportingSearchProvider)
            else _SlottedSearch(search, self.search_slot)
        )
        # The audited Core handlers require actual source and lock identities.
        repository = Path(__file__).resolve().parents[3]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        lock_digest = hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()
        handlers = BaselineNodeHandlers(
            initial_plan_generator=planner,
            ranker=cast(Any, SimilarityRanker(cast(TextEmbedder, adapters["embed"]))),
            writer=writer,
            search_provider=slotted_search,
            fetcher=cast(Fetcher, adapters["fetch"]),
            parser=cast(Parser, adapters["parse"]),
            artifact_store=self.artifact_store,
            evidence_store=self.evidence_store,
            cache=FileCache(self.artifact_store._root),  # pyright: ignore[reportPrivateUsage] - Core exposes no root accessor
            usage_cost_resolver=_SnapshotCostResolver(
                provider_routes, pricing_snapshots, cost_calculator
            ),
            search_snapshot_id=str(
                routes["search"].parameters.get("snapshot_id", provider_routes.configuration_sha256)
            ),
            fetch_snapshot_id=str(
                routes["fetch"].parameters.get("snapshot_id", provider_routes.configuration_sha256)
            ),
            code_commit=commit,
            dependency_lock_sha256=lock_digest,
            provider_profile_configuration_sha256=provider_routes.configuration_sha256,
            seed_supported=config.seed is not None,
            pricing_status="estimated" if pricing_snapshots else "unknown",
            pricing_snapshots=pricing_snapshots,
            replay_parent=None,
            writer_prompt_version=config.prompt_versions.get("writer", "baseline-writer-v1"),
        )
        baseline = build_baseline_graph(handlers.as_dependencies(checkpointer))
        return LangGraphResearchRunner(baseline_graph=baseline)


class ServiceRunnerFactory(Protocol):
    def resolve_provider_routes(self, provider_profile_id: str) -> FrozenProviderRoutes: ...

    def required_pricing_keys(
        self, config: RunConfig, provider_routes: FrozenProviderRoutes
    ) -> set[tuple[str, str, str]]: ...

    def create(
        self,
        *,
        config: RunConfig,
        provider_routes: FrozenProviderRoutes,
        pricing_snapshots: tuple[PricingSnapshot, ...],
        checkpointer: BaseCheckpointSaver[Any],
    ) -> ResearchRunner: ...


class LangGraphServiceRunnerFactory:
    def __init__(self, builder: CoreRunnerBuilder, route_catalog: ProviderRouteCatalog) -> None:
        self.builder = builder
        self.route_catalog = route_catalog

    def resolve_provider_routes(self, provider_profile_id: str) -> FrozenProviderRoutes:
        return self.route_catalog.resolve(provider_profile_id)

    def required_pricing_keys(
        self, config: RunConfig, provider_routes: FrozenProviderRoutes
    ) -> set[tuple[str, str, str]]:
        validate_provider_route_binding(config, provider_routes)
        return self.builder.required_pricing_keys(config, provider_routes)

    def create(
        self,
        *,
        config: RunConfig,
        provider_routes: FrozenProviderRoutes,
        pricing_snapshots: tuple[PricingSnapshot, ...],
        checkpointer: BaseCheckpointSaver[Any],
    ) -> ResearchRunner:
        validate_provider_route_binding(config, provider_routes)
        return self.builder.build(
            config=config,
            provider_routes=provider_routes,
            pricing_snapshots=pricing_snapshots,
            checkpointer=checkpointer,
            cost_calculator=CostCalculator,
        )
