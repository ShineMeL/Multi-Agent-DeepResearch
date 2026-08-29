# Core Foundation 与 Replay Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 从空仓库交付 Python 3.12、typed、可重放的最小 Deep Research 基线，使一个 P1 Fixed Plan + R1 Similarity 流程可通过 LangGraph 在 strict Replay 和 Live Provider 下生成带证据引用的 Markdown 报告。

**Architecture:** deepresearch.domain 是唯一共享领域模型入口；providers/protocols.py 隔离所有外部 SDK；runtime 统一预算、取消、manifest 和 ResearchRunner；正文与大对象进入 content-addressed stores，Graph State 只保存 ID 和小型状态。BaselineGraph 使用固定计划、固定深度和 R1 相似度跑通 Plan→Search→Fetch→Parse→Rank→Write→Manifest。

**Tech Stack:** Python 3.12、uv、Pydantic v2、pydantic-settings、LangGraph、HTTPX、Trafilatura、PyMuPDF、sentence-transformers、Typer、Rich、pytest、pytest-asyncio、respx、Ruff、Pyright。

**Spec:** [Multi-Agent Deep Research 设计文档](../specs/2026-08-29-multi-agent-deep-research-design.md)

## Global Constraints

- Python version is exactly 3.12 in pyproject.toml, CI and Docker; do not widen it to 3.11 or an unspecified 3.x.
- deepresearch.domain is the only public import surface for domain models: ResearchRequest, ResearchPlan, SubQuestion, InformationNeed, CoverageLedgerEntry, RunBudget, SourceDocument, EvidenceSpan, Claim, ClaimEvidenceLink, RerankScore, RunConfig, RunEvent and RunResult.
- Provider request/result types live in deepresearch.providers.types; ProviderError lives in providers.errors; async protocols live in providers.protocols. No downstream plan may duplicate them.
- CancellationToken, BudgetAccountant, BudgetSnapshot, ResourceEstimate, CheckpointRef and ResearchRunner are re-exported from deepresearch.runtime; RunManifest and PricingSnapshot remain in deepresearch.runtime.manifest. These contracts are consumed unchanged by Planner, benchmark and service plans.
- This plan is the initial owner of .gitignore, pyproject.toml, uv.lock, apps/cli/main.py, README.md, retrieval/url_policy.py, providers/replay.py and workflow/runner.py. Later plans may make only the explicitly listed additive modifications and must preserve every existing public signature and dependency.
- Provider methods are async, accept Deadline and CancellationToken, return typed results, and never leak concrete SDK objects.
- The Graph State stores IDs, small typed records, counters and decisions only. Raw bytes, normalized text, full model responses and reports live in ArtifactStore/EvidenceStore.
- Every external call uses a deterministic cache/idempotency key. A cached successful call restores recorded usage but does not charge it twice.
- strict Replay returns REPLAY_MISS for any unknown request, schema version or hash mismatch and never calls Live.
- HTML offsets are normalized Unicode code-point half-open ranges; PDF locators are page plus block and character range. Every excerpt and parsed body has SHA-256.
- No hidden chain of thought is persisted. Structured public decisions contain only codes, scores, short reasons and selected IDs.
- Credentials are read from environment by Settings, never accepted in ResearchRequest, RunConfig, Graph State, logs, manifests or artifacts.
- Normal CI and every task in this plan run entirely offline with Fake or Replay providers.
- Every code task is test-first: red test, observed expected failure, minimum implementation, green test, Ruff/Pyright where relevant, then one focused commit.
- Modify only files listed for the active task; preserve unrelated user changes.

## Canonical Shared Contract Catalog

The following names and fields are authoritative for all four plans. Task 2 may split them across private modules, but deepresearch.domain must re-export them without aliases or duplicate replacements.

~~~python
RunStatus = Literal[
    "queued", "running", "interrupted", "completed", "failed", "cancelled"
]
StopReason = Literal["SUFFICIENT", "PLATEAU", "BUDGET_EXHAUSTED", "BLOCKED"]
ExecutionMode = Literal["live", "replay", "hybrid"]
AccessProfile = Literal["showcase", "public_live", "local"]
RunPurpose = Literal["demo", "benchmark", "test"]
SourceType = Literal[
    "paper",
    "official_documentation",
    "standard",
    "primary_data",
    "first_party_statement",
    "secondary_analysis",
    "news",
    "unknown",
]
ClaimType = Literal["fact", "numeric", "comparison", "trend", "causal", "limitation"]
VerificationStatus = Literal["supported", "contradicted", "uncertain", "unsupported"]


class HtmlLocator(BaseModel):
    kind: Literal["html"] = "html"
    paragraph_id: Annotated[str, Field(min_length=1)]
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]


class PdfLocator(BaseModel):
    kind: Literal["pdf"] = "pdf"
    page_index: Annotated[int, Field(ge=0)]
    block_index: Annotated[int, Field(ge=0)]
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]


Locator: TypeAlias = Annotated[HtmlLocator | PdfLocator, Field(discriminator="kind")]


class FreshnessRequirement(BaseModel):
    kind: Literal["none", "published_after", "retrieved_within_days"]
    published_after: date | None = None
    retrieved_within_days: Annotated[int | None, Field(ge=1)] = None


class ResearchRequest(BaseModel):
    question: str
    output_requirements: dict[str, JsonValue]
    report_language: str
    source_languages: tuple[str, ...]
    freshness_requirement: FreshnessRequirement
    execution_mode: ExecutionMode
    access_profile: AccessProfile
    provider_profile_id: str
    run_purpose: RunPurpose
    budget_preset: Literal["low", "medium", "high"]


class DateRange(BaseModel):
    start: date | None = None
    end: date | None = None


class ResearchScope(BaseModel):
    included_topics: tuple[str, ...]
    excluded_topics: tuple[str, ...]
    date_range: DateRange | None = None
    answer_shape: str


class InformationNeed(BaseModel):
    need_id: str
    text: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]


class EvidenceRequirements(BaseModel):
    min_independent_sources: Annotated[int, Field(ge=1)]
    allowed_source_types: frozenset[SourceType]
    must_include_primary: bool
    freshness: FreshnessRequirement | None = None


class SubQuestion(BaseModel):
    id: str
    question: str
    rationale_code: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]
    dependencies: tuple[str, ...]
    information_needs: tuple[InformationNeed, ...]
    evidence_requirements: EvidenceRequirements
    status: Literal["pending", "active", "covered", "blocked"]


class ResearchPlan(BaseModel):
    plan_id: str
    scope: ResearchScope
    subquestions: tuple[SubQuestion, ...]
    created_by_model: str
    prompt_version: str


class CoverageLedgerEntry(BaseModel):
    subquestion_id: str
    coverage_score: Annotated[float, Field(ge=0.0, le=1.0)]
    independent_source_count: Annotated[int, Field(ge=0)]
    unresolved_conflict_ids: tuple[str, ...]
    uncertainty_score: Annotated[float, Field(ge=0.0, le=1.0)]
    last_marginal_gain: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: tuple[str, ...]
    attempt_count: Annotated[int, Field(ge=0)]
    last_decision_code: str


class SourceDocument(BaseModel):
    source_id: str
    canonical_url: AnyHttpUrl
    title: str
    authors: tuple[str, ...]
    published_at: datetime | None = None
    retrieved_at: datetime
    content_hash: str
    parsed_content_hash: str
    source_type: SourceType
    source_family_id: str
    parser_version: str


class EvidenceSpan(BaseModel):
    evidence_id: str
    source_id: str
    locator: Locator
    excerpt: str
    excerpt_hash: str
    language: str
    information_need_ids: tuple[str, ...]


class Claim(BaseModel):
    claim_id: str
    text: str
    claim_type: ClaimType
    entities: tuple[str, ...]
    numbers: tuple[str, ...]
    qualifiers: tuple[str, ...]
    report_section: str
    verification_status: VerificationStatus


class ClaimEvidenceLink(BaseModel):
    claim_id: str
    evidence_id: str
    relation: Literal["support", "contradict", "context", "insufficient"]
    entailment_score: Annotated[float, Field(ge=0.0, le=1.0)]
    relevance_score: Annotated[float, Field(ge=0.0, le=1.0)]
    judge_model: str
    prompt_version: str
    decision_code: str


class RerankScore(BaseModel):
    evidence_id: str
    total: Annotated[float, Field(ge=0.0, le=1.0)]
    feature_scores: dict[str, float]
    model_id: str | None = None
    prompt_version: str | None = None


class ResourceUsage(BaseModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)]
    cached_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    search_calls: Annotated[int, Field(ge=0)]
    pages: Annotated[int, Field(ge=0)]
    retries: Annotated[int, Field(ge=0)]
    wall_seconds: Annotated[float, Field(ge=0.0)]
    cost_usd: Decimal | None = None

    @classmethod
    def zero(cls, *, cost_known: bool = False) -> "ResourceUsage":
        return cls(
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cached_tokens=0,
            total_tokens=0,
            search_calls=0,
            pages=0,
            retries=0,
            wall_seconds=0.0,
            cost_usd=Decimal("0") if cost_known else None,
        )


class RunBudget(BaseModel):
    max_search_calls: Annotated[int, Field(ge=0)]
    max_pages: Annotated[int, Field(ge=0)]
    max_total_tokens: Annotated[int, Field(ge=0)]
    max_wall_time_seconds: Annotated[int, Field(gt=0)]
    max_cost_usd: Annotated[Decimal | None, Field(ge=0)] = None
    max_retries: Annotated[int, Field(ge=0)]
    used_by_node: dict[
        Literal["Planner", "Ranker", "Writer", "Judge", "Tool"],
        ResourceUsage,
    ]


class RunConfig(BaseModel):
    request: ResearchRequest
    workflow_id: Literal["baseline-v1", "research-v1"]
    planner_id: Literal["P0", "P1", "P2"]
    ranker_id: Literal["R0", "R1", "R2"]
    budget: RunBudget
    prompt_versions: dict[str, str]
    ranker_weights_version: str | None = None
    seed: int | None = None


class RunEvent(BaseModel):
    seq: int
    run_id: str
    timestamp: datetime
    node: str
    kind: str
    status: RunStatus
    public_payload: dict[str, JsonValue]
    usage_delta: ResourceUsage
    artifact_ids: tuple[str, ...]
    error_code: str | None = None


class RunResult(BaseModel):
    run_id: str
    thread_id: str
    status: RunStatus
    stop_reason: StopReason | None = None
    is_partial: bool
    report_artifact_id: str | None = None
    evidence_graph_artifact_id: str | None = None
    manifest_artifact_id: str | None = None
    final_usage: ResourceUsage
    error_code: str | None = None
~~~

All these models use ConfigDict(extra="forbid", frozen=True). Date/time validators require RFC 3339 timezone-aware values. Score validators reject NaN and infinity before clipping where the spec permits clipping. Hash validators accept lowercase 64-character SHA-256 only.

## Exact File Map

Create:

    pyproject.toml
    uv.lock
    .env.example
    src/deepresearch/__init__.py
    src/deepresearch/config.py
    src/deepresearch/domain/__init__.py
    src/deepresearch/domain/enums.py
    src/deepresearch/domain/locators.py
    src/deepresearch/domain/research.py
    src/deepresearch/domain/evidence.py
    src/deepresearch/domain/usage.py
    src/deepresearch/domain/events.py
    src/deepresearch/providers/__init__.py
    src/deepresearch/providers/types.py
    src/deepresearch/providers/errors.py
    src/deepresearch/providers/protocols.py
    src/deepresearch/providers/resilience.py
    src/deepresearch/providers/recording.py
    src/deepresearch/providers/embeddings.py
    src/deepresearch/providers/openai_compatible.py
    src/deepresearch/providers/tavily.py
    src/deepresearch/providers/httpx_fetcher.py
    src/deepresearch/providers/httpx_transport.py
    src/deepresearch/providers/parsers/__init__.py
    src/deepresearch/providers/parsers/html.py
    src/deepresearch/providers/parsers/pdf.py
    src/deepresearch/providers/replay_schema.py
    src/deepresearch/providers/replay.py
    src/deepresearch/retrieval/__init__.py
    src/deepresearch/retrieval/normalize.py
    src/deepresearch/retrieval/url_policy.py
    src/deepresearch/retrieval/chunking.py
    src/deepresearch/retrieval/dedupe.py
    src/deepresearch/storage/__init__.py
    src/deepresearch/storage/artifacts.py
    src/deepresearch/storage/cache.py
    src/deepresearch/storage/evidence_store.py
    src/deepresearch/runtime/__init__.py
    src/deepresearch/runtime/budget.py
    src/deepresearch/runtime/cancellation.py
    src/deepresearch/runtime/manifest.py
    src/deepresearch/runtime/checkpoints.py
    src/deepresearch/runtime/ports.py
    src/deepresearch/planning/__init__.py
    src/deepresearch/planning/validation.py
    src/deepresearch/planning/fixed.py
    src/deepresearch/evidence/__init__.py
    src/deepresearch/evidence/similarity.py
    src/deepresearch/workflow/__init__.py
    src/deepresearch/workflow/state.py
    src/deepresearch/workflow/baseline_graph.py
    src/deepresearch/workflow/runner.py
    src/deepresearch/reporting/__init__.py
    src/deepresearch/reporting/boundary.py
    src/deepresearch/reporting/markdown.py
    apps/__init__.py
    apps/cli/__init__.py
    apps/cli/main.py
    tests/conftest.py
    tests/unit/test_package.py
    tests/unit/test_config.py
    tests/unit/test_architecture_boundaries.py
    tests/unit/test_secret_redaction_boundary.py
    tests/unit/domain/test_research_models.py
    tests/unit/domain/test_evidence_models.py
    tests/unit/domain/test_locators.py
    tests/unit/domain/test_usage.py
    tests/unit/domain/test_events.py
    tests/unit/retrieval/test_normalize.py
    tests/unit/retrieval/test_url_policy.py
    tests/unit/retrieval/test_chunking.py
    tests/unit/retrieval/test_dedupe.py
    tests/unit/runtime/test_cancellation.py
    tests/unit/runtime/test_budget.py
    tests/unit/runtime/test_manifest.py
    tests/unit/runtime/test_checkpoints.py
    tests/unit/storage/test_artifacts.py
    tests/unit/storage/test_cache.py
    tests/unit/storage/test_evidence_store.py
    tests/unit/planning/test_validation.py
    tests/unit/planning/test_fixed.py
    tests/unit/evidence/test_similarity.py
    tests/unit/reporting/test_markdown.py
    tests/unit/reporting/test_boundary.py
    tests/unit/providers/test_types.py
    tests/unit/providers/test_replay.py
    tests/unit/providers/test_recording.py
    tests/unit/providers/test_resilience.py
    tests/unit/providers/test_embeddings.py
    tests/unit/providers/test_openai_compatible.py
    tests/unit/providers/test_tavily.py
    tests/unit/providers/test_httpx_fetcher.py
    tests/unit/providers/test_html_parser.py
    tests/unit/providers/test_pdf_parser.py
    tests/unit/workflow/test_state.py
    tests/unit/workflow/test_baseline_routes.py
    tests/contracts/test_model_provider_contract.py
    tests/contracts/test_search_provider_contract.py
    tests/contracts/test_fetcher_contract.py
    tests/contracts/test_parser_contract.py
    tests/integration/replay/test_baseline_graph.py
    tests/cli/test_research_command.py
    tests/fixtures/providers/article.html
    tests/fixtures/providers/paper.pdf
    tests/fixtures/models/embedding.lock.json
    tests/fixtures/replay/provider_contract/
    tests/fixtures/replay/baseline/

Modify:

    .gitignore
    README.md

Generated, not hand-written:

    .venv/
    artifacts/
    .cache/deepresearch/
    models/embedding.lock.json

---

### Task 1: 初始化 Python 3.12 工程与质量门

**Files:**

- Create: pyproject.toml
- Generate: uv.lock
- Create: .env.example
- Create: src/deepresearch/__init__.py
- Create: apps/__init__.py
- Create: apps/cli/__init__.py
- Create: apps/cli/main.py
- Create: tests/conftest.py
- Create: tests/unit/test_package.py
- Modify: .gitignore
- Modify: README.md

**Interfaces:** Consumes an empty Python workspace. Produces the `deepresearch` and `apps` packages, `deepresearch.__version__`, Typer `app`, locked Python 3.12 dependencies, typed test configuration, non-secret Settings example and repository ignore policy consumed by every later task.

- [ ] **Step 1: 写包与 CLI 红测**

~~~python
from typer.testing import CliRunner

import deepresearch
from apps.cli.main import app


def test_package_exposes_version():
    assert deepresearch.__version__ == "0.1.0"


def test_cli_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
~~~

- [ ] **Step 2: 在没有工程配置时确认失败**

Run: uv run pytest tests/unit/test_package.py -q

Expected: FAIL because pyproject.toml and package modules do not exist.

- [ ] **Step 3: 创建 pyproject.toml**

Use:

~~~toml
[project]
name = "multi-agent-deepresearch"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
  "httpx>=0.28,<0.29",
  "langgraph>=1.2.11,<2",
  "langgraph-checkpoint-sqlite>=3.1.1,<4",
  "pydantic>=2.11,<3",
  "pydantic-settings>=2.10,<3",
  "pymupdf>=1.26,<2",
  "rich>=14,<15",
  "sentence-transformers>=6,<7",
  "trafilatura>=2.2,<3",
  "typer>=0.27,<1",
]

[project.optional-dependencies]
dev = [
  "pyright>=1.1.400,<2",
  "pytest>=8.4,<9",
  "pytest-asyncio>=1.1,<2",
  "respx>=0.22,<0.23",
  "ruff>=0.12,<1",
]

[project.scripts]
deepresearch = "apps.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/deepresearch", "apps"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
include = ["src", "apps", "benchmarks", "experiments", "tests"]
~~~

The major-version ranges are based on the 2026-08-29 releases of [LangGraph](https://pypi.org/project/langgraph/), [LangGraph SQLite Checkpoint](https://pypi.org/project/langgraph-checkpoint-sqlite/), [Sentence Transformers](https://pypi.org/project/sentence-transformers/), [Trafilatura](https://pypi.org/project/trafilatura/) and [Typer](https://pypi.org/project/typer/). uv.lock, not a floating install, is the executable dependency record.

Run uv lock and uv sync --all-extras. Commit uv.lock; never edit it by hand.

- [ ] **Step 4: 添加最小包、CLI、环境示例与 ignore**

~~~python
# src/deepresearch/__init__.py
__version__ = "0.1.0"

# apps/cli/main.py
import typer
from deepresearch import __version__

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo(__version__)
~~~

.env.example contains empty DEEPRESEARCH_MODEL_API_KEY and DEEPRESEARCH_TAVILY_API_KEY plus non-secret defaults for DEEPRESEARCH_MODEL_BASE_URL, DEEPRESEARCH_MODEL_ID, DEEPRESEARCH_ARTIFACT_ROOT and DEEPRESEARCH_CACHE_ROOT. Append .venv/, .env, artifacts/, .cache/deepresearch/, models/embedding/ and __pycache__/ to .gitignore; retain the existing .superpowers/ rule. Do not ignore `models/embedding.lock.json`.

- [ ] **Step 5: 运行绿测和质量门**

Run:

    uv run pytest tests/unit/test_package.py -q
    uv run ruff check src apps tests/unit/test_package.py
    uv run pyright src apps

Expected: PASS and zero type errors.

- [ ] **Step 6: 提交**

    git add pyproject.toml uv.lock .env.example .gitignore README.md src/deepresearch/__init__.py apps tests/conftest.py tests/unit/test_package.py
    git commit -m "build: initialize Python 3.12 deepresearch project"

### Task 2: 定义唯一领域模型与稳定序列化

**Files:**

- Create: src/deepresearch/domain/__init__.py
- Create: src/deepresearch/domain/enums.py
- Create: src/deepresearch/domain/locators.py
- Create: src/deepresearch/domain/research.py
- Create: src/deepresearch/domain/evidence.py
- Create: src/deepresearch/domain/usage.py
- Create: src/deepresearch/domain/events.py
- Test: tests/unit/domain/test_research_models.py
- Test: tests/unit/domain/test_evidence_models.py
- Test: tests/unit/domain/test_locators.py
- Test: tests/unit/domain/test_usage.py
- Test: tests/unit/domain/test_events.py

**Interfaces:** Consumes Pydantic v2 only. Produces the complete canonical models re-exported from `deepresearch.domain`; `SubQuestion.id`, RunConfig cost-profile validation, stable canonical JSON/hash behavior and all enum/status literals are frozen for every later plan.

- [ ] **Step 1: 写约束红测**

~~~python
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from deepresearch.domain import (
    EvidenceSpan,
    HtmlLocator,
    ResearchPlan,
    RunBudget,
    RunEvent,
    SubQuestion,
)


def test_plan_rejects_duplicate_ids_and_dependency_cycle(plan_factory):
    with pytest.raises(ValidationError, match="duplicate|cycle"):
        plan_factory(
            subquestion_ids=("sq-1", "sq-1"),
            dependencies={"sq-1": ("sq-1",)},
        )


def test_html_locator_uses_half_open_code_point_offsets():
    locator = HtmlLocator(paragraph_id="p-1", start_char=1, end_char=3)
    assert "多模态"[locator.start_char:locator.end_char] == "模态"


def test_evidence_requires_sha256_and_existing_range():
    with pytest.raises(ValidationError):
        EvidenceSpan(
            evidence_id="ev-1",
            source_id="src-1",
            locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=5),
            excerpt_hash="bad",
            excerpt="proof",
            language="en",
            information_need_ids=("need-1",),
        )


def test_medium_budget_matches_approved_spec():
    budget = RunBudget.preset("medium")
    assert (
        budget.max_search_calls,
        budget.max_pages,
        budget.max_total_tokens,
        budget.max_wall_time_seconds,
        str(budget.max_cost_usd),
    ) == (8, 12, 40_000, 300, "0.50")


def test_local_allows_token_only_budget_but_public_and_benchmark_require_cost(
    run_config_factory,
):
    token_only = RunBudget.preset("medium").model_copy(
        update={"max_cost_usd": None}
    )
    assert run_config_factory(access_profile="local", budget=token_only).budget.max_cost_usd is None
    with pytest.raises(ValidationError, match="max_cost_usd"):
        run_config_factory(access_profile="public_live", budget=token_only)
    with pytest.raises(ValidationError, match="max_cost_usd"):
        run_config_factory(run_purpose="benchmark", budget=token_only)


def test_event_timestamp_requires_timezone():
    with pytest.raises(ValidationError):
        RunEvent(
            seq=1,
            run_id="r1",
            timestamp=datetime(2026, 8, 29),
            node="Plan",
            kind="node_started",
            status="running",
            public_payload={},
            usage_delta=ResourceUsage.zero(),
            artifact_ids=(),
            error_code=None,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "none", "published_after": "2026-01-01"},
        {"kind": "published_after"},
        {
            "kind": "retrieved_within_days",
            "published_after": "2026-01-01",
            "retrieved_within_days": 7,
        },
    ],
)
def test_freshness_payload_matches_discriminator(payload):
    with pytest.raises(ValidationError):
        FreshnessRequirement.model_validate(payload)


def test_date_range_and_usage_invariants():
    with pytest.raises(ValidationError):
        DateRange(start=date(2026, 2, 1), end=date(2026, 1, 1))
    with pytest.raises(ValidationError):
        usage(input_tokens=2, cached_tokens=3, output_tokens=0, reasoning_tokens=0, total_tokens=2)
    with pytest.raises(ValidationError):
        usage(input_tokens=2, output_tokens=3, reasoning_tokens=1, total_tokens=5)
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/domain -q

Expected: FAIL because deepresearch.domain exports do not exist.

- [ ] **Step 3: 实现研究与预算模型**

~~~python
@model_validator(mode="after")
def validate_plan_graph(self: ResearchPlan) -> ResearchPlan:
    subquestion_ids = [item.id for item in self.subquestions]
    need_ids = [
        need.need_id
        for subquestion in self.subquestions
        for need in subquestion.information_needs
    ]
    require_unique(subquestion_ids, label="subquestion")
    require_unique(need_ids, label="information need")
    assert_known_dependencies(self.subquestions)
    assert_acyclic(self.subquestions)
    return self


@model_validator(mode="after")
def validate_freshness_payload(self: FreshnessRequirement) -> FreshnessRequirement:
    expected = {
        "none": (False, False),
        "published_after": (True, False),
        "retrieved_within_days": (False, True),
    }[self.kind]
    actual = (self.published_after is not None, self.retrieved_within_days is not None)
    if actual != expected:
        raise ValueError("freshness payload does not match kind")
    return self


@classmethod
def preset(
    cls,
    name: Literal["low", "medium", "high"],
) -> RunBudget:
    limits = {
        "low": (4, 8, 20_000, 180, Decimal("0.25")),
        "medium": (8, 12, 40_000, 300, Decimal("0.50")),
        "high": (12, 20, 70_000, 480, Decimal("1.00")),
    }
    searches, pages, tokens, seconds, cost = limits[name]
    return cls(
        max_search_calls=searches,
        max_pages=pages,
        max_total_tokens=tokens,
        max_wall_time_seconds=seconds,
        max_cost_usd=cost,
        max_retries=2,
        used_by_node=zero_usage_by_node(),
    )
~~~

Implement every field exactly as listed in Canonical Shared Contract Catalog. Add a `RunConfig` model validator that requires `budget.max_cost_usd` when `request.access_profile="public_live"` or `request.run_purpose="benchmark"`; local/showcase replay may use null for token-only accounting. The preset method uses max_retries=2 and zero-valued used_by_node entries:

    low:    4 searches, 8 pages, 20k tokens, 180 seconds, USD 0.25
    medium: 8 searches, 12 pages, 40k tokens, 300 seconds, USD 0.50
    high:   12 searches, 20 pages, 70k tokens, 480 seconds, USD 1.00

ResearchRequest contains exactly question, output_requirements, report_language, source_languages, freshness_requirement, execution_mode, access_profile, provider_profile_id, run_purpose and budget_preset. RunConfig deliberately does not duplicate provider_profile_id; provider selection always reads `config.request.provider_profile_id`. ResearchPlan validator enforces unique subquestion/need IDs, valid dependencies and an acyclic graph. SubQuestion uses the approved `id` field and additionally contains rationale_code, evidence_requirements and status. EvidenceRequirements carries min_independent_sources, allowed_source_types, must_include_primary and optional freshness.

- [ ] **Step 4: 实现 Evidence、locator、usage 和 run 模型**

~~~python
@model_validator(mode="after")
def validate_html_range(self: HtmlLocator) -> HtmlLocator:
    if self.end_char <= self.start_char:
        raise ValueError("end_char must be greater than start_char")
    return self


@model_validator(mode="after")
def validate_pdf_range(self: PdfLocator) -> PdfLocator:
    if self.end_char <= self.start_char:
        raise ValueError("end_char must be greater than start_char")
    return self


@model_validator(mode="after")
def validate_date_range(self: DateRange) -> DateRange:
    if self.start is not None and self.end is not None and self.start > self.end:
        raise ValueError("date range start must not exceed end")
    return self


@model_validator(mode="after")
def validate_usage_totals(self: ResourceUsage) -> ResourceUsage:
    if self.cached_tokens > self.input_tokens:
        raise ValueError("cached_tokens must not exceed input_tokens")
    if self.total_tokens != self.input_tokens + self.output_tokens + self.reasoning_tokens:
        raise ValueError("total_tokens invariant violated")
    return self


@field_validator("timestamp")
@classmethod
def require_timezone(cls, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value
~~~

Implement `HtmlLocator`, `PdfLocator`, the discriminated `Locator` alias, SourceDocument, EvidenceSpan, Claim, ClaimEvidenceLink, RerankScore, CoverageLedgerEntry, ResourceUsage, RunConfig, RunEvent and RunResult from the catalog exactly once, and re-export Locator from `deepresearch.domain`. The locator union is discriminated by the required/defaulted `kind` literal; HTML offsets use a non-empty paragraph ID and Unicode code-point half-open `[start_char,end_char)` range, while PDF uses zero-based page/block indexes and the same nonnegative half-open character range. RunResult separates terminal status, stop_reason and is_partial. Validate locator ranges, timezone-aware timestamps, SHA-256 strings, finite scores and total_tokens accounting.

- [ ] **Step 5: 从 deepresearch.domain 统一 re-export**

domain/__init__.py explicitly imports and lists every public class in __all__. Downstream code imports from deepresearch.domain instead of private module paths.

- [ ] **Step 6: 运行测试与提交**

Run:

    uv run pytest tests/unit/domain -q
    uv run ruff check src/deepresearch/domain tests/unit/domain
    uv run pyright src/deepresearch/domain

Expected: all pass.

    git add src/deepresearch/domain tests/unit/domain
    git commit -m "feat: define typed research evidence and run domain"

### Task 3: Provider 类型、异步 Protocol、取消和预算结算

**Files:**

- Create: src/deepresearch/providers/__init__.py
- Create: src/deepresearch/providers/types.py
- Create: src/deepresearch/providers/errors.py
- Create: src/deepresearch/providers/protocols.py
- Create: src/deepresearch/providers/resilience.py
- Create: src/deepresearch/runtime/__init__.py
- Create: src/deepresearch/runtime/cancellation.py
- Create: src/deepresearch/runtime/budget.py
- Test: tests/unit/providers/test_types.py
- Test: tests/unit/runtime/test_cancellation.py
- Test: tests/unit/runtime/test_budget.py
- Test: tests/unit/providers/test_resilience.py
- Test: tests/contracts/test_model_provider_contract.py
- Test: tests/contracts/test_search_provider_contract.py
- Test: tests/contracts/test_fetcher_contract.py
- Test: tests/contracts/test_parser_contract.py

**Interfaces:** Consumes `deepresearch.domain`. Produces `providers.types`, `ProviderError`, async Model/Search/Fetch/Parse/Embed/Rerank protocols, `ProviderCallExecutor`, `OperationCancelled`, `BudgetExceeded`, CancellationToken, ResourceEstimate, BudgetReservation, BudgetSnapshot and idempotent BudgetAccountant; downstream code imports these names unchanged.

- [ ] **Step 1: 写取消、超限和 contract 红测**

~~~python
@pytest.mark.asyncio
async def test_cancelled_search_fails_before_provider_call(
    fake_search_provider, cancelled_token
):
    with pytest.raises(OperationCancelled):
        await call_search(
            fake_search_provider,
            query="agent planning",
            limit=5,
            filters=None,
            deadline=100.0,
            cancellation_token=cancelled_token,
        )
    assert fake_search_provider.calls == 0


def test_budget_reserve_rejects_hard_limit():
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    with pytest.raises(BudgetExceeded):
        accountant.reserve(
            ResourceEstimate(
                search_calls=9,
                pages=0,
                tokens=0,
                wall_seconds=0,
                cost_usd=Decimal("0"),
            ),
            node="Tool",
            idempotency_key="search-overrun",
        )


def test_failed_call_with_usage_is_charged_once():
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    reservation = accountant.reserve(small_estimate(), node="Planner", idempotency_key="m1")
    first = accountant.settle(reservation, actual=usage(tokens=100, cost="0.01"))
    second = accountant.settle(reservation, actual=usage(tokens=100, cost="0.01"))
    assert first == second
    assert second.used_tokens == 100


def test_cached_usage_is_observable_but_not_charged_again():
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    reservation = accountant.reserve(small_estimate(), node="Tool", idempotency_key="cached")
    snapshot = accountant.settle(
        reservation,
        actual=usage(tokens=100, cost="0.01"),
        charge=False,
    )
    assert snapshot.used_tokens == 0
    assert snapshot.last_observed_usage.total_tokens == 100


@pytest.mark.asyncio
async def test_retry_policy_uses_injected_jitter_and_never_retries_auth(
    fake_clock, fake_sleeper
):
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=fake_clock,
        sleeper=fake_sleeper,
        random=Random(7),
    )
    result = await executor.call("search", upstream_503_then_ok(), remaining_deadline=100.0)
    assert result == "ok"
    assert len(fake_sleeper.delays) == 1
    with pytest.raises(ProviderError) as error:
        await executor.call("model", always_auth_error(), remaining_deadline=100.0)
    assert error.value.code == "AUTHENTICATION"
    assert always_auth_error.calls == 1
~~~

- [ ] **Step 2: 运行红测**

Run:

    uv run pytest tests/unit/providers/test_types.py tests/unit/providers/test_resilience.py tests/unit/runtime/test_cancellation.py tests/unit/runtime/test_budget.py tests/contracts -q

Expected: FAIL because provider and runtime contracts are missing.

- [ ] **Step 3: 实现公共 Provider 结果和错误**

~~~python
Deadline: TypeAlias = float


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    url: AnyHttpUrl
    title: str
    snippet: str
    rank: Annotated[int, Field(ge=1)]
    published_at: datetime | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RawDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requested_url: AnyHttpUrl
    final_url: AnyHttpUrl
    status: Annotated[int, Field(ge=100, le=599)]
    headers: dict[str, str]
    content_type: str
    body_bytes: bytes
    retrieved_at: datetime


class ParsedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    block_id: str
    text: str
    locator: Locator
    text_hash: str


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_url: AnyHttpUrl
    title: str
    authors: tuple[str, ...]
    published_at: datetime | None = None
    normalized_text: str
    blocks: tuple[ParsedBlock, ...]
    parser_id: str
    parser_version: str
    parsed_content_hash: str


class ModelMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[dict[str, JsonValue], ...] = ()
    temperature: Decimal
    seed: int | None = None
    max_output_tokens: Annotated[int, Field(gt=0)]
    prompt_version: str
    system_prompt_hash: str
    tool_schema_hash: str
    output_schema_hash: str


class ToolCall(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, JsonValue]


class ModelResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output: T
    usage: ResourceUsage
    provider_id: str
    model_id: str
    tool_calls: tuple[ToolCall, ...] = ()
    raw_response_artifact_id: str


class StructuredModelResult(ModelResult[T], Generic[T]):
    output_schema_hash: str


class ModelStreamChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    index: Annotated[int, Field(ge=0)]
    text_delta: str = ""
    tool_call_delta: dict[str, JsonValue] | None = None
    finish_reason: str | None = None
    final_usage: ResourceUsage | None = None


class ProviderError(RuntimeError):
    def __init__(
        self,
        *,
        code: Literal[
            "TIMEOUT", "RATE_LIMITED", "INVALID_REQUEST", "INVALID_RESPONSE",
            "AUTHENTICATION", "NETWORK", "REPLAY_MISS", "INVALID_SNAPSHOT",
            "PARSE_UNSUPPORTED", "UPSTREAM_5XX",
        ],
        provider: str,
        operation: str,
        public_message: str,
        retryable: bool,
        retry_after: float | None = None,
        usage: ResourceUsage | None = None,
    ) -> None: ...
~~~

SearchHit never carries per-hit usage; one search operation is charged exactly once by `ProviderCallExecutor` and its call trace, regardless of hit count. ParsedDocument.blocks is the only parsed structural collection; downstream EvidenceSpan objects are created later by chunking/storage and are never exposed as ParsedDocument.spans. Validators enforce aware timestamps, lowercase SHA-256 fields, ParsedBlock locator bounds/text hash, parsed_content_hash over normalized_text, non-empty ModelRequest messages, sorted stream indexes, and exactly one final stream chunk carrying final_usage. The Fetch node writes RawDocument.body_bytes to ArtifactStore immediately and stores only the resulting artifact ID in Graph State. A parser whose supports(content_type) is false raises non-retryable PARSE_UNSUPPORTED; malformed supported content raises INVALID_RESPONSE.

- [ ] **Step 4: 实现唯一 Protocol**

~~~python
class ModelProvider(Protocol):
    provider_id: str

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]: ...

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[T]: ...

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]: ...


class SearchProvider(Protocol):
    provider_id: str

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]: ...


class Parser(Protocol):
    parser_id: str
    parser_version: str

    def supports(self, content_type: str) -> bool: ...

    async def parse(
        self,
        raw_document: RawDocument,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> ParsedDocument: ...


class Fetcher(Protocol):
    provider_id: str

    async def fetch(
        self,
        url: str,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> RawDocument: ...


class TextEmbedder(Protocol):
    provider_id: str
    model_id: str
    model_revision: str
    snapshot_sha256: str

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]: ...


class Reranker(Protocol):
    reranker_id: str

    async def score(
        self,
        information_need: str,
        evidence_spans: Sequence[EvidenceSpan],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[RerankScore]: ...
~~~

ModelProvider.stream returns an async iterator whose final chunk contains settled usage; cancellation closes the upstream response. TextEmbedder rejects a result count/dimension mismatch and non-finite vectors. Contract tests are reusable pytest mixins that assert success, streaming success/cancellation, timeout, invalid response, usage and stable typed serialization.

- [ ] **Step 5: 实现 CancellationToken 与 BudgetAccountant**

~~~python
BudgetNode = Literal["Planner", "Ranker", "Writer", "Judge", "Tool"]
BudgetDimension = Literal[
    "search_calls", "pages", "tokens", "wall_seconds", "cost_usd", "retries"
]


class ResourceEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    search_calls: Annotated[int, Field(ge=0)] = 0
    pages: Annotated[int, Field(ge=0)] = 0
    tokens: Annotated[int, Field(ge=0)] = 0
    wall_seconds: Annotated[float, Field(ge=0.0)] = 0.0
    cost_usd: Annotated[Decimal | None, Field(ge=0)] = None
    retries: Annotated[int, Field(ge=0)] = 0


class BudgetReservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reservation_id: str
    idempotency_key: str
    node: BudgetNode
    estimate: ResourceEstimate


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    used_search_calls: Annotated[int, Field(ge=0)]
    used_pages: Annotated[int, Field(ge=0)]
    used_tokens: Annotated[int, Field(ge=0)]
    used_wall_seconds: Annotated[float, Field(ge=0.0)]
    used_cost_usd: Annotated[Decimal | None, Field(ge=0)]
    used_retries: Annotated[int, Field(ge=0)]
    reserved_search_calls: Annotated[int, Field(ge=0)]
    reserved_pages: Annotated[int, Field(ge=0)]
    reserved_tokens: Annotated[int, Field(ge=0)]
    reserved_wall_seconds: Annotated[float, Field(ge=0.0)]
    reserved_cost_usd: Annotated[Decimal | None, Field(ge=0)]
    reserved_retries: Annotated[int, Field(ge=0)]
    exhausted: frozenset[BudgetDimension]
    last_observed_usage: ResourceUsage
    used_by_node: dict[BudgetNode, ResourceUsage]


class OperationCancelled(RuntimeError):
    code: Literal["CANCELLED"] = "CANCELLED"


class BudgetExceeded(RuntimeError):
    code: Literal["BUDGET_EXCEEDED"] = "BUDGET_EXCEEDED"

    def __init__(
        self,
        dimensions: frozenset[BudgetDimension],
        snapshot: BudgetSnapshot,
    ) -> None:
        super().__init__(",".join(sorted(dimensions)))
        self.dimensions = dimensions
        self.snapshot = snapshot


class CancellationToken:
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...
    def raise_if_cancelled(self) -> None: ...


class BudgetAccountant:
    def __init__(self, budget: RunBudget, *, run_scope: str = "local") -> None: ...

    def snapshot(self) -> BudgetSnapshot: ...

    def reserve(
        self,
        estimate: ResourceEstimate,
        *,
        node: str,
        idempotency_key: str,
    ) -> BudgetReservation: ...

    def settle(
        self,
        reservation: BudgetReservation,
        *,
        actual: ResourceUsage,
        charge: bool = True,
    ) -> BudgetSnapshot: ...

    def release(self, reservation: BudgetReservation) -> BudgetSnapshot: ...
~~~

`CancellationToken` is thread-safe and monotonic: the first `cancel()` sets an internal event, later calls are no-ops, and `raise_if_cancelled()` raises only `OperationCancelled` with the stable public code. Every provider/graph boundary checks it before work and after awaited I/O.

Reservation IDs are SHA-256 of the accountant run scope plus idempotency_key. `reserve` is idempotent for an identical key/node/estimate and rejects reuse with different inputs. Reservations count against available capacity immediately; an estimate that makes any `used + reserved` dimension exceed its hard maximum raises `BudgetExceeded` with all offending dimensions and the unchanged snapshot. `settle` atomically removes the estimate, records actual usage once, and returns the already stored snapshot on repeats; `release` atomically removes an unsettled reservation and is also idempotent. Settling/releasing a reservation from another accountant or an unknown ID raises ValueError. `exhausted` contains every dimension whose charged-plus-reserved value has reached its limit, so Planner may use its truth value directly.

Cached hits restore the original ResourceUsage in traces with `settle(..., charge=False)`, so `last_observed_usage` remains auditable while charged totals do not increment twice. Failed calls carrying usage settle normally. ResourceUsage.search_calls/pages/retries map directly; tokens use total_tokens; wall uses wall_seconds. When max_cost_usd is null, both used/reserved cost remain null and the accountant enforces every non-cost hard limit; RunConfig prevents that mode for public_live/benchmark. Workflow node names map to the five canonical budget buckets: Plan→Planner, RankEvidence→Ranker, DraftReport→Writer, evidence judging→Judge, and Search/Fetch/Parse→Tool; never write arbitrary graph node names into `used_by_node`.

runtime/__init__.py re-exports `OperationCancelled`, `BudgetExceeded`, CancellationToken, BudgetAccountant, BudgetReservation, BudgetSnapshot and ResourceEstimate at this task. Task 8 later adds CheckpointRef and ResearchRunner to the same explicit __all__ list.

- [ ] **Step 6: 实现统一 deadline、retry、jitter 与显式 fallback**

~~~python
@dataclass(frozen=True)
class ProviderCallPolicy:
    default_timeout_seconds: Mapping[
        Literal["model", "search", "fetch", "parse", "embed"], float
    ]
    max_retries: int = 2
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 4.0
    jitter_ratio: float = 0.20

    @classmethod
    def defaults(cls) -> "ProviderCallPolicy":
        return cls(
            default_timeout_seconds={
                "model": 120.0,
                "search": 30.0,
                "fetch": 30.0,
                "parse": 30.0,
                "embed": 30.0,
            }
        )


class ProviderCallExecutor:
    async def call(
        self,
        operation: Literal["model", "search", "fetch", "parse", "embed"],
        invoke: Callable[[float], Awaitable[T]],
        *,
        remaining_deadline: float,
        fallback_invocations: Sequence[Callable[[float], Awaitable[T]]] = (),
    ) -> T: ...
~~~

Each attempt receives `min(remaining_deadline, now + operation_default)` as its absolute deadline. Retry at most `RunBudget.max_retries` times for TIMEOUT, RATE_LIMITED, NETWORK and UPSTREAM_5XX only; honor `retry_after`, otherwise use capped exponential backoff plus injected seeded jitter. The unit test covers both a 503→success path and a non-retryable authentication failure. Never retry or fall back on AUTHENTICATION, INVALID_REQUEST, INVALID_RESPONSE, PARSE_UNSUPPORTED, REPLAY_MISS or INVALID_SNAPSHOT. Fallback is opt-in through an ordered provider profile, keeps the same idempotency key and remaining hard budget, records every attempt, and is disabled in strict Replay. Tests inject clock/sleeper/RNG and perform no real sleep.

- [ ] **Step 7: 运行 contract 与提交**

Run:

    uv run pytest tests/unit/providers/test_types.py tests/unit/providers/test_resilience.py tests/unit/runtime tests/contracts -q
    uv run ruff check src/deepresearch/providers src/deepresearch/runtime tests
    uv run pyright src/deepresearch/providers src/deepresearch/runtime

Expected: all pass.

    git add src/deepresearch/providers src/deepresearch/runtime tests/unit/providers tests/unit/runtime tests/contracts
    git commit -m "feat: add provider contracts cancellation and budgets"

### Task 4: 内容寻址 Storage、规范化、分块与去重

**Files:**

- Create: src/deepresearch/retrieval/__init__.py
- Create: src/deepresearch/retrieval/normalize.py
- Create: src/deepresearch/retrieval/url_policy.py
- Create: src/deepresearch/retrieval/chunking.py
- Create: src/deepresearch/retrieval/dedupe.py
- Create: src/deepresearch/storage/__init__.py
- Create: src/deepresearch/storage/artifacts.py
- Create: src/deepresearch/storage/cache.py
- Create: src/deepresearch/storage/evidence_store.py
- Test: tests/unit/retrieval/test_normalize.py
- Test: tests/unit/retrieval/test_url_policy.py
- Test: tests/unit/retrieval/test_chunking.py
- Test: tests/unit/retrieval/test_dedupe.py
- Test: tests/unit/storage/test_artifacts.py
- Test: tests/unit/storage/test_cache.py
- Test: tests/unit/storage/test_evidence_store.py

**Interfaces:** Consumes canonical domain/provider types. Produces `CanonicalURL`, `URLSecurityError`, `normalize_text`, `sha256_text`, `canonicalize_url`, `validate_public_http_url`, operation-specific CacheKey models, LocalArtifactStore, FileCache and LocalEvidenceStore with content-addressed atomic persistence; the URL names are re-exported from `deepresearch.retrieval` for Service security tests.

- [ ] **Step 1: 写 hash、原子写入和 locator 红测**

~~~python
def test_text_normalization_and_hash_are_unicode_stable():
    left = normalize_text("  A\t  B  \r\n多模态\u00a0 Agent  ")
    right = normalize_text("A B\n多模态 Agent")
    assert left == right
    assert sha256_text(left) == sha256_text(right)


def test_canonical_url_removes_fragment_tracking_and_default_port():
    assert canonicalize_url(
        "HTTPS://Example.COM:443/a?utm_source=x&b=2&a=1#top"
    ) == "https://example.com/a?a=1&b=2"


def test_artifact_put_is_content_addressed_and_atomic(tmp_path):
    store = LocalArtifactStore(tmp_path)
    first = store.put_bytes(b"same", media_type="text/plain")
    second = store.put_bytes(b"same", media_type="text/plain")
    assert first.artifact_id == second.artifact_id
    assert store.get_bytes(first.artifact_id) == b"same"
    assert not list(tmp_path.rglob("*.tmp"))


def test_evidence_store_rejects_locator_hash_mismatch(store, source):
    store.put_source(source, normalized_text="short text")
    with pytest.raises(EvidenceIntegrityError, match="excerpt_hash"):
        store.put_evidence(
            evidence_span(
                source_id=source.source_id,
                start_char=0,
                end_char=5,
                excerpt_hash="0" * 64,
            )
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("snapshot_id", "snapshot-2"),
        ("locale", "zh-CN"),
        ("time_policy", "published-after-2025"),
    ],
)
def test_search_cache_key_hash_includes_every_policy_field(
    search_cache_key, field, replacement
):
    changed = search_cache_key.model_copy(update={field: replacement})
    assert cache_key_sha256(changed) != cache_key_sha256(search_cache_key)
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/retrieval tests/unit/storage -q

Expected: FAIL because retrieval and storage modules are missing.

- [ ] **Step 3: 实现规范化与安全 URL policy**

~~~python
TRACKING_QUERY_KEYS = frozenset(
    {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
)

CanonicalURL = NewType("CanonicalURL", str)


class URLSecurityError(ValueError):
    code: Literal["URL_NOT_PUBLIC"] = "URL_NOT_PUBLIC"

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_url(url: str) -> str: ...


def validate_public_http_url(
    url: str,
    *,
    resolved_ips: Sequence[ipaddress.IPv4Address | ipaddress.IPv6Address],
) -> CanonicalURL: ...
~~~

`canonicalize_url` accepts only HTTP(S), lowercases host, IDNA-normalizes it, removes fragments/default ports/tracking parameters, sorts remaining query pairs and preserves meaningful repeated parameters; invalid syntax/scheme raises `URLSecurityError` rather than leaking parser details. `validate_public_http_url` canonicalizes first, requires at least one resolved address, rejects missing hosts, credentials in URL, localhost, loopback, private, link-local, multicast, reserved and unspecified IPs, then returns `CanonicalURL(canonical)`. Redirects must be revalidated by the Fetcher. `CanonicalURL` is a validated-at-boundary NewType, not a second URL parser or Pydantic domain model.

- [ ] **Step 4: 实现 chunk/dedupe 和 stores**

Chunk normalized text at paragraph boundaries with target 900 and maximum 1,200 Unicode code points, overlap 120, and locators mapped to source offsets. Exact duplicate key is parsed_content_hash; near-duplicate helper returns SimHash and normalized-title similarity but does not merge conflicting evidence automatically.

~~~python
class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_id: str
    sha256: str
    media_type: str
    size_bytes: Annotated[int, Field(ge=0)]


class CacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key_sha256: str
    value_artifact_id: str
    producer_version: str
    usage: ResourceUsage
    created_at: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SearchCacheKey(BaseModel):
    operation: Literal["search"]
    snapshot_id: str
    normalized_query: str
    provider_id: str
    endpoint_type: str
    locale: str | None = None
    complete_parameters: dict[str, JsonValue]
    time_policy: str


class FetchCacheKey(BaseModel):
    operation: Literal["fetch"]
    snapshot_id: str
    canonical_url: AnyHttpUrl
    fetch_policy: str
    accepted_content_types: tuple[str, ...]


class ParseCacheKey(BaseModel):
    operation: Literal["parse"]
    snapshot_id: str
    raw_content_hash: str
    parser_id: str
    parser_version: str
    normalization_version: str


class ModelCacheKey(BaseModel):
    operation: Literal["model"]
    provider_id: str
    endpoint_type: str
    model_id: str
    prompt_version: str
    system_prompt_hash: str
    tool_schema_hash: str
    output_schema_hash: str
    temperature: Decimal
    seed: int | None = None
    canonical_request_hash: str


class EmbedCacheKey(BaseModel):
    operation: Literal["embed"]
    model_id: str
    model_revision: str
    snapshot_sha256: str
    normalize_embeddings: bool
    canonical_texts_hash: str


CacheKey = Annotated[
    SearchCacheKey | FetchCacheKey | ParseCacheKey | ModelCacheKey | EmbedCacheKey,
    Field(discriminator="operation"),
]


class LocalArtifactStore:
    def put_bytes(self, data: bytes, *, media_type: str) -> ArtifactRef: ...
    def get_bytes(self, artifact_id: str) -> bytes: ...
    def exists(self, artifact_id: str) -> bool: ...


class FileCache:
    def get(self, key: CacheKey) -> CacheEntry | None: ...
    def put_if_absent(self, key: CacheKey, value: CacheEntry) -> CacheEntry: ...


class LocalEvidenceStore:
    def put_source(
        self,
        source: SourceDocument,
        *,
        normalized_text: str,
    ) -> SourceDocument: ...

    def put_evidence(self, evidence: EvidenceSpan) -> EvidenceSpan: ...
    def get_source(self, source_id: str) -> SourceDocument: ...
    def get_evidence(self, evidence_id: str) -> EvidenceSpan: ...
    def has_evidence(self, evidence_id: str) -> bool: ...
~~~

Use per-key lock files and write-to-temp plus os.replace. ArtifactRef verifies artifact_id equals `sha256:<sha256>` and size/hash against bytes on every read. CacheEntry validates its key/value IDs, aware timestamp and secret-free metadata. Each key model uses `extra="forbid", frozen=True`; `cache_key_sha256` hashes its sorted canonical JSON. Search, Fetch, Parse, Model and Embed keys contain every operation-specific field shown above, including empty tool/output-schema hashes when genuinely absent. Never put secrets in the key or value metadata.

- [ ] **Step 5: 运行绿测**

Run:

    uv run pytest tests/unit/retrieval tests/unit/storage -q
    uv run ruff check src/deepresearch/retrieval src/deepresearch/storage tests/unit/retrieval tests/unit/storage
    uv run pyright src/deepresearch/retrieval src/deepresearch/storage

Expected: all pass.

- [ ] **Step 6: 提交**

    git add src/deepresearch/retrieval src/deepresearch/storage tests/unit/retrieval tests/unit/storage
    git commit -m "feat: add normalized content-addressed research storage"

### Task 5: Strict Replay schema、Provider 与 Run Manifest

**Files:**

- Create: src/deepresearch/providers/replay_schema.py
- Create: src/deepresearch/providers/replay.py
- Create: src/deepresearch/providers/recording.py
- Create: src/deepresearch/runtime/manifest.py
- Test: tests/unit/providers/test_replay.py
- Test: tests/unit/providers/test_recording.py
- Test: tests/unit/runtime/test_manifest.py
- Create: tests/fixtures/replay/provider_contract/search.jsonl
- Create: tests/fixtures/replay/provider_contract/model_responses.jsonl
- Create: tests/fixtures/replay/provider_contract/embeddings.jsonl
- Create: tests/fixtures/replay/provider_contract/documents.jsonl
- Create: tests/fixtures/replay/provider_contract/snapshot.json
- Create: tests/fixtures/replay/provider_contract/manifest.sha256

**Interfaces:** Consumes the provider protocols, CacheKey models, stores and budget usage. Produces strict Replay providers plus ReplayTextEmbedder, recording wrappers/ReplayBundleWriter, `PricingSnapshot`, `CostCalculator` and immutable `RunManifest`; later benchmark/service code consumes these exact runtime manifest types.

- [ ] **Step 1: 写 exact-match、miss 和 manifest 红测**

~~~python
@pytest.mark.asyncio
async def test_replay_search_exact_request_returns_recorded_hit(bundle, token):
    provider = ReplaySearchProvider(bundle)
    hits = await provider.search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=100.0,
        cancellation_token=token,
    )
    assert [hit.provider_metadata["source_id"] for hit in hits] == ["src-1"]
    assert provider.live_calls == 0


@pytest.mark.asyncio
async def test_replay_miss_never_falls_back(bundle, token):
    provider = ReplaySearchProvider(bundle)
    with pytest.raises(ProviderError) as error:
        await provider.search(
            "unknown query",
            5,
            None,
            deadline=100.0,
            cancellation_token=token,
        )
    assert error.value.code == "REPLAY_MISS"
    assert provider.live_calls == 0


def test_manifest_hash_changes_when_prompt_version_changes(manifest_factory):
    left = manifest_factory(prompt_versions={"planner": "v1"})
    right = manifest_factory(prompt_versions={"planner": "v2"})
    assert left.canonical_sha256() != right.canonical_sha256()


def test_estimated_cost_requires_a_complete_pricing_snapshot(manifest_factory):
    with pytest.raises(ValidationError):
        manifest_factory(pricing_status="estimated", pricing_snapshots=())


def test_cost_calculator_uses_decimal_unit_rates(pricing_snapshot):
    usage = ResourceUsage.zero().model_copy(
        update={
            "input_tokens": 1_000_000,
            "cached_tokens": 200_000,
            "output_tokens": 100_000,
            "reasoning_tokens": 50_000,
            "total_tokens": 1_150_000,
        }
    )
    breakdown = CostCalculator.estimate(usage, pricing_snapshot)
    assert breakdown.total_usd == Decimal("0.114")


def test_manifest_preserves_cache_hit_usage_without_double_charging(manifest_factory):
    base = manifest_factory()
    model_index = next(
        index
        for index, call in enumerate(base.provider_calls)
        if call.operation == "model"
    )
    cached_call = base.provider_calls[model_index].model_copy(
        update={"cache_hit": True, "estimated_cost_usd": Decimal("0")}
    )
    calls = list(base.provider_calls)
    calls[model_index] = cached_call
    charged_cost = sum(
        (
            call.estimated_cost_usd
            for call in calls
            if call.operation == "model" and not call.cache_hit
        ),
        start=Decimal("0"),
    )
    manifest = RunManifest.model_validate(
        {
            **base.model_dump(),
            "provider_calls": calls,
            "cache_hit_count": sum(call.cache_hit for call in calls),
            "usage": base.usage.model_copy(update={"cost_usd": charged_cost}),
        }
    )
    assert cached_call.usage.total_tokens > 0
    assert cached_call.estimated_cost_usd == Decimal("0")
    assert manifest.cache_hit_count == sum(call.cache_hit for call in calls)
    assert manifest.usage.cost_usd == charged_cost

    calls[model_index] = cached_call.model_copy(
        update={"estimated_cost_usd": Decimal("0.000000001")}
    )
    with pytest.raises(ValidationError, match="cache hit.*zero"):
        RunManifest.model_validate(
            {
                **base.model_dump(),
                "provider_calls": calls,
                "cache_hit_count": sum(call.cache_hit for call in calls),
                "usage": base.usage.model_copy(update={"cost_usd": charged_cost}),
            }
        )


def test_manifest_hash_covers_lock_and_call_schema(manifest_factory):
    left = manifest_factory(dependency_lock_sha256="1" * 64)
    right = manifest_factory(dependency_lock_sha256="2" * 64)
    assert left.canonical_sha256() != right.canonical_sha256()
    changed_temperature = left.model_copy(
        update={
            "provider_calls": (
                left.provider_calls[0].model_copy(update={"temperature": Decimal("0.1")}),
                *left.provider_calls[1:],
            )
        }
    )
    assert left.canonical_sha256() != changed_temperature.canonical_sha256()


@pytest.mark.asyncio
async def test_recorded_model_search_fetch_and_embed_replay_without_delegates(
    recording_harness, tmp_path
):
    live_result = await recording_harness.record_once(tmp_path / "bundle")
    replay_result = await recording_harness.strict_replay(tmp_path / "bundle")
    assert replay_result.canonical_output == live_result.canonical_output
    assert replay_result.live_calls == 0
    assert (tmp_path / "bundle" / "manifest.sha256").is_file()
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/providers/test_replay.py tests/unit/providers/test_recording.py tests/unit/runtime/test_manifest.py -q

Expected: FAIL because Replay and manifest modules are missing.

- [ ] **Step 3: 实现版本化 Replay records**

~~~python
class ReplayRequestKey(BaseModel):
    operation: Literal[
        "model.complete", "model.structured", "model.stream", "search", "fetch", "embed"
    ]
    provider_id: str
    request_sha256: str
    prompt_version: str | None = None
    schema_version: str


class ReplaySuccess(BaseModel):
    kind: Literal["success"]
    response: JsonValue


class ReplayFailure(BaseModel):
    kind: Literal["failure"]
    code: str
    public_message: str
    retryable: bool
    retry_after: float | None = None


class ReplayRecord(BaseModel):
    key: ReplayRequestKey
    outcome: Annotated[
        ReplaySuccess | ReplayFailure,
        Field(discriminator="kind"),
    ]
    usage: ResourceUsage
    latency_ms: int
    outcome_sha256: str


class ReplayVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    valid: bool
    record_count_by_operation: dict[str, int]
    file_sha256: dict[str, str]
    errors: tuple[str, ...] = ()


class ReplayBundle:
    @classmethod
    def load(cls, root: Path) -> "ReplayBundle": ...
    def lookup(self, key: ReplayRequestKey) -> ReplayRecord: ...
    def verify(self) -> ReplayVerification: ...
~~~

Canonical request JSON sorts keys, uses UTF-8 and normalizes URLs/query text before hashing. `ReplayRecord.outcome` is a discriminated success/failure union: Replay re-raises a recorded failure with the same public ProviderError fields and usage, never fabricates a success. ReplayVerification requires `valid == not errors`, counts every operation, and verifies its file hashes exactly against manifest.sha256. Duplicate keys or mismatched record/file hashes make bundle loading fail. ReplayModelProvider, ReplaySearchProvider, ReplayFetcher and ReplayTextEmbedder implement the shared protocols only; they have no Live delegate argument. Streaming records preserve ordered chunks plus final usage and replay them as an async iterator.

- [ ] **Step 4: 实现 Live 录制 wrapper 与原子 BundleWriter**

~~~python
class ReplayBundleWriter:
    @classmethod
    def create(cls, final_root: Path, *, run_id: str) -> "ReplayBundleWriter": ...

    async def append(
        self,
        *,
        key: ReplayRequestKey,
        outcome: ReplaySuccess | ReplayFailure,
        usage: ResourceUsage,
        latency_ms: int,
    ) -> None: ...

    async def finalize(self) -> Path: ...
    async def abort(self) -> None: ...


class RecordingModelProvider:
    def __init__(self, delegate: ModelProvider, writer: ReplayBundleWriter) -> None: ...


class RecordingSearchProvider:
    def __init__(self, delegate: SearchProvider, writer: ReplayBundleWriter) -> None: ...


class RecordingFetcher:
    def __init__(self, delegate: Fetcher, writer: ReplayBundleWriter) -> None: ...


class RecordingTextEmbedder:
    def __init__(self, delegate: TextEmbedder, writer: ReplayBundleWriter) -> None: ...
~~~

Each wrapper delegates exactly once through `ProviderCallExecutor`, records the canonical request, typed response/chunks, actual usage, latency, provider/model/revision/prompt/schema versions and a redacted typed failure when applicable, then returns the untouched typed result. Fetch bodies are base64-encoded only inside the private replay bundle; Authorization headers, cookies, API keys and unrelated response headers are never recorded. `ReplayBundleWriter` takes an exclusive file lock, writes sorted JSONL and content hashes into a sibling staging directory, fsyncs, verifies by loading `ReplayBundle`, and atomically renames to a previously nonexistent final root. Abort removes only its verified staging child. A formal first pass uses these wrappers; strict Replay constructs only the four replay implementations and fails on any miss.

- [ ] **Step 5: 实现完整 manifest**

~~~python
class PricingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_id: str
    provider_id: str
    endpoint_type: str
    model_id: str
    effective_at: datetime
    currency: Literal["USD"]
    input_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]
    output_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]
    cached_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]
    reasoning_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]


class CostBreakdown(BaseModel):
    pricing_snapshot_id: str
    input_usd: Decimal
    cached_input_usd: Decimal
    output_usd: Decimal
    reasoning_usd: Decimal
    total_usd: Decimal


class CostCalculator:
    QUANTUM = Decimal("0.000000001")

    @staticmethod
    def estimate(usage: ResourceUsage, pricing: PricingSnapshot) -> CostBreakdown:
        unit = Decimal(1_000_000)
        billable_input = max(usage.input_tokens - usage.cached_tokens, 0)
        input_usd = Decimal(billable_input) * pricing.input_tokens_per_million_usd / unit
        cached_usd = Decimal(usage.cached_tokens) * pricing.cached_tokens_per_million_usd / unit
        output_usd = Decimal(usage.output_tokens) * pricing.output_tokens_per_million_usd / unit
        reasoning_usd = Decimal(usage.reasoning_tokens) * pricing.reasoning_tokens_per_million_usd / unit
        quantize = lambda value: value.quantize(
            CostCalculator.QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )
        return CostBreakdown(
            pricing_snapshot_id=pricing.snapshot_id,
            input_usd=quantize(input_usd),
            cached_input_usd=quantize(cached_usd),
            output_usd=quantize(output_usd),
            reasoning_usd=quantize(reasoning_usd),
            total_usd=quantize(input_usd + cached_usd + output_usd + reasoning_usd),
        )


class ProviderProfileRecord(BaseModel):
    profile_id: str
    execution_mode: ExecutionMode
    provider_ids: tuple[str, ...]
    configuration_sha256: str


class ProviderCallRecord(BaseModel):
    operation: Literal["model", "search", "fetch", "parse", "embed"]
    node: str
    provider_id: str
    endpoint_type: str
    model_id: str | None = None
    model_revision: str | None = None
    request_sha256: str
    snapshot_id: str | None = None
    normalized_query: str | None = None
    locale: str | None = None
    complete_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    time_policy: str | None = None
    prompt_version: str | None = None
    system_prompt_hash: str | None = None
    tool_schema_hash: str | None = None
    output_schema_hash: str | None = None
    temperature: Decimal | None = None
    seed: int | None = None
    started_at: datetime
    finished_at: datetime
    latency_ms: Annotated[int, Field(ge=0)]
    attempt: Annotated[int, Field(ge=1)]
    cache_hit: bool
    outcome_code: str
    usage: ResourceUsage
    pricing_snapshot_id: str | None = None
    estimated_cost_usd: Decimal | None = None


class NodeExecutionRecord(BaseModel):
    node: str
    attempt: Annotated[int, Field(ge=1)]
    started_at: datetime
    finished_at: datetime
    latency_ms: Annotated[int, Field(ge=0)]
    status: RunStatus
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    usage: ResourceUsage
    error_code: str | None = None


class ParsedArtifactRecord(BaseModel):
    source_id: str
    raw_content_hash: str
    parsed_content_hash: str
    parser_id: str
    parser_version: str
    normalization_version: str
    artifact_id: str


class EvidenceHashRecord(BaseModel):
    evidence_id: str
    source_id: str
    locator_sha256: str
    excerpt_hash: str
    artifact_id: str


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["run-manifest-v1"]
    run_id: str
    thread_id: str
    code_commit: str
    dependency_lock_sha256: str
    request_sha256: str
    config_sha256: str
    workflow_id: str
    graph_version: str
    planner_id: str
    provider_profiles: tuple[ProviderProfileRecord, ...]
    model_ids: tuple[str, ...]
    prompt_versions: dict[str, str]
    parser_versions: dict[str, str]
    ranker_id: str
    ranker_weights_version: str | None
    budget: RunBudget
    usage: ResourceUsage
    usage_by_node: dict[str, ResourceUsage]
    pricing_status: Literal["estimated", "unknown"]
    pricing_snapshots: tuple[PricingSnapshot, ...]
    provider_calls: tuple[ProviderCallRecord, ...]
    node_executions: tuple[NodeExecutionRecord, ...]
    parsed_artifacts: tuple[ParsedArtifactRecord, ...]
    evidence_hashes: tuple[EvidenceHashRecord, ...]
    source_snapshot_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    run_event_count: Annotated[int, Field(ge=0)]
    run_events_sha256: str
    seed: int | None = None
    seed_supported: bool
    cache_hit_count: Annotated[int, Field(ge=0)]
    stop_reason: StopReason | None = None
    is_partial: bool
    failure_codes: tuple[str, ...]
    replay_parent: str | None = None
    started_at: datetime
    finished_at: datetime
    manifest_sha256: str
~~~

Add operation-specific validators to ProviderCallRecord: search requires snapshot/query/locale/complete parameters/time policy; model requires model ID/revision, prompt/system/tool/output-schema hashes, temperature and explicit seed support; fetch/parse/embed require the corresponding Task 4 cache-key fields in `complete_parameters`. `finished_at >= started_at` and latency must match the injected monotonic clock record.

Add a RunManifest validator: `pricing_status="estimated"` requires one matching PricingSnapshot for every distinct `(provider_id, endpoint_type, model_id)` model call. For each non-cache model call, recalculate cost with `CostCalculator`, reject a mismatched `estimated_cost_usd`, and include that value in the charged Decimal sum. A `cache_hit=True` model call preserves its recorded non-cost `ResourceUsage` for audit, but must have `estimated_cost_usd=Decimal("0")`; exclude it from the charged sum so replay/cache reads cannot bill twice. Require `cache_hit_count` to equal the number of cache-hit provider-call records and `usage.cost_usd` to equal only that charged sum. `pricing_status="unknown"` requires an empty pricing_snapshots tuple and null estimated costs. Public Live and formal benchmark callers must use estimated; local exploratory runs may use unknown but still enforce token/search/page limits. Verify all dependency/request/config/schema/parser/artifact/event/manifest hashes as lowercase SHA-256, require workflow/planner/ranker IDs to match RunConfig, require model_ids to equal call-record model IDs, and require node executions/events to be contiguous. `manifest_sha256` equals canonical_sha256 over the complete model dump with only `manifest_sha256` removed; the hash includes every result-affecting version, call parameter, unit rate, latency and content hash.

- [ ] **Step 6: 运行测试并提交**

Run:

    uv run pytest tests/unit/providers/test_replay.py tests/unit/providers/test_recording.py tests/unit/runtime/test_manifest.py -q
    uv run ruff check src/deepresearch/providers/replay.py src/deepresearch/providers/replay_schema.py src/deepresearch/providers/recording.py src/deepresearch/runtime/manifest.py
    uv run pyright src/deepresearch/providers/replay.py src/deepresearch/providers/recording.py src/deepresearch/runtime/manifest.py

Expected: PASS.

    git add src/deepresearch/providers/replay.py src/deepresearch/providers/replay_schema.py src/deepresearch/providers/recording.py src/deepresearch/runtime/manifest.py tests/unit/providers/test_replay.py tests/unit/providers/test_recording.py tests/unit/runtime/test_manifest.py tests/fixtures/replay/provider_contract
    git commit -m "feat: add strict replay and immutable run manifests"

### Task 6: Live Adapter、HTML/PDF Parser 与 typed Settings

**Files:**

- Create: src/deepresearch/config.py
- Create: src/deepresearch/providers/openai_compatible.py
- Create: src/deepresearch/providers/tavily.py
- Create: src/deepresearch/providers/httpx_fetcher.py
- Create: src/deepresearch/providers/httpx_transport.py
- Create: src/deepresearch/providers/embeddings.py
- Create: src/deepresearch/providers/parsers/__init__.py
- Create: src/deepresearch/providers/parsers/html.py
- Create: src/deepresearch/providers/parsers/pdf.py
- Test: tests/unit/providers/test_openai_compatible.py
- Test: tests/unit/providers/test_tavily.py
- Test: tests/unit/providers/test_httpx_fetcher.py
- Test: tests/unit/providers/test_html_parser.py
- Test: tests/unit/providers/test_pdf_parser.py
- Test: tests/unit/providers/test_embeddings.py
- Test: tests/unit/test_config.py
- Create: tests/fixtures/providers/article.html
- Create: tests/fixtures/providers/paper.pdf
- Create: tests/fixtures/models/embedding.lock.json
- Generate: models/embedding.lock.json

**Interfaces:** Consumes provider protocols, resilience executor, URL policy and Replay contracts. Produces OpenAI-compatible/Tavily/HTTPX/HTML/PDF adapters, typed Settings, locked SentenceTransformerTextEmbedder and offline DeterministicHashTextEmbedder; all adapters pass the shared contracts.

- [ ] **Step 1: 写 HTTP 边界和解析红测**

~~~python
@pytest.mark.asyncio
async def test_openai_structured_rejects_invalid_schema(respx_mock, provider, token):
    respx_mock.post("https://model.test/v1/chat/completions").respond(
        200,
        json={
            "choices": [{"message": {"content": '{"unexpected": true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        },
    )
    with pytest.raises(ProviderError) as error:
        await provider.structured(
            model_request("make plan"),
            ResearchPlan,
            deadline=100.0,
            cancellation_token=token,
        )
    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.usage.total_tokens == 13


@pytest.mark.asyncio
async def test_fetcher_revalidates_redirect_target(respx_mock, fetcher, token):
    respx_mock.get("https://public.test/start").respond(
        302, headers={"Location": "http://127.0.0.1/private"}
    )
    with pytest.raises(ProviderError, match="private"):
        await fetcher.fetch(
            "https://public.test/start",
            deadline=100.0,
            cancellation_token=token,
        )


@pytest.mark.asyncio
async def test_fetcher_rejects_dns_rebinding_and_ignores_proxy_env(
    monkeypatch, rebinding_transport, token
):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    fetcher = HttpxFetcher(transport=rebinding_transport)
    with pytest.raises(ProviderError, match="peer IP"):
        await fetcher.fetch(
            "https://public.test/article",
            deadline=100.0,
            cancellation_token=token,
        )
    assert rebinding_transport.proxy_connections == 0


@pytest.mark.asyncio
async def test_redirect_reacquires_host_slot_for_each_hostname(
    redirect_transport, recording_host_slot, token
):
    fetcher = HttpxFetcher(
        transport=redirect_transport,
        host_slot=recording_host_slot,
    )
    await fetcher.fetch(
        "https://a.public.test/start",
        deadline=100.0,
        cancellation_token=token,
    )
    assert recording_host_slot.hosts == ["a.public.test", "b.public.test"]


@pytest.mark.asyncio
async def test_html_and_pdf_parsers_emit_stable_locators(
    html_parser, pdf_parser, token
):
    html = await html_parser.parse(
        html_fixture(), deadline=100.0, cancellation_token=token
    )
    pdf = await pdf_parser.parse(
        pdf_fixture(), deadline=100.0, cancellation_token=token
    )
    assert html.blocks[0].locator.kind == "html"
    assert pdf.blocks[0].locator.kind == "pdf"
    assert html.parsed_content_hash == sha256_text(html.normalized_text)


def test_sentence_embedder_refuses_unlocked_or_missing_snapshot(tmp_path):
    lock = embedding_lock(revision="e62509716f15c5fd03a6fd3156a4bc5e43f83f26")
    with pytest.raises(ModelSnapshotUnavailable):
        SentenceTransformerTextEmbedder.from_lock(lock, model_root=tmp_path)


@pytest.mark.asyncio
async def test_fake_embedder_is_byte_stable_without_network(fake_embedder, token):
    first = await fake_embedder.embed(["Agent", "多模态"], deadline=100.0, cancellation_token=token)
    second = await fake_embedder.embed(["Agent", "多模态"], deadline=100.0, cancellation_token=token)
    assert first == second
    assert fake_embedder.network_calls == 0
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/providers tests/unit/test_config.py -q

Expected: FAIL on missing adapters, parsers and Settings.

- [ ] **Step 3: 实现 Settings 与 OpenAI-compatible ModelProvider**

~~~python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPRESEARCH_",
        env_file=".env",
        extra="ignore",
    )
    model_base_url: AnyHttpUrl
    model_id: str
    model_api_key: SecretStr
    tavily_api_key: SecretStr | None = None
    artifact_root: Path = Path("artifacts")
    cache_root: Path = Path(".cache/deepresearch")
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 45.0
    embedding_model_root: Path = Path("models/embedding")
    embedding_lock_path: Path = Path("models/embedding.lock.json")
~~~

OpenAICompatibleModelProvider implements complete, structured and stream against `/chat/completions`, validates status/body/usage, parses structured JSON with the requested Pydantic schema, maps timeout/429/auth/invalid responses to ProviderError, and records no Authorization header or raw secret. All calls go through Task 3 `ProviderCallExecutor`; adapter code contains no second retry loop.

`providers/embeddings.py` defines `EmbeddingModelLock`, `SentenceTransformerTextEmbedder`, and test-only `DeterministicHashTextEmbedder`. The committed production lock pins model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e62509716f15c5fd03a6fd3156a4bc5e43f83f26`, vector dimension 384, normalize_embeddings=true, every downloaded file SHA-256 and the canonical root manifest SHA-256. Implement an explicit setup-only module command:

    uv run python -m deepresearch.providers.embeddings fetch-and-lock --model-id sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --revision e62509716f15c5fd03a6fd3156a4bc5e43f83f26 --model-root models/embedding --lock models/embedding.lock.json

It downloads only that revision into a staging child, verifies the resolved commit equals the requested revision, hashes every file, atomically replaces only the staging destination, and refuses to overwrite a different existing lock without `--replace`. Runtime always uses `local_files_only=True`, `trust_remote_code=False`, verifies the lock before construction, and never downloads. CI uses `DeterministicHashTextEmbedder`; Replay uses Task 5 `ReplayTextEmbedder`. Neither path imports or constructs SentenceTransformer. The pinned upstream model and revision are documented from the [official model repository](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

- [ ] **Step 4: 实现 Tavily、HTTPX Fetcher 和 parsers**

~~~python
HostSlot: TypeAlias = Callable[[str], AbstractAsyncContextManager[None]]


@asynccontextmanager
async def no_op_host_slot(hostname: str) -> AsyncIterator[None]:
    del hostname
    yield


class HttpxFetcher:
    def __init__(
        self,
        *,
        transport: PinnedPeerTransport,
        host_slot: HostSlot = no_op_host_slot,
        max_redirects: int = 5,
        max_body_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self._transport = transport
        self._host_slot = host_slot
        self._max_redirects = max_redirects
        self._max_body_bytes = max_body_bytes
~~~

TavilySearchProvider maps provider results to SearchHit and applies stable rank. `PinnedPeerTransport` constructs HTTPX with `trust_env=False`. For every initial request and redirect, HttpxFetcher canonicalizes the URL, resolves all addresses, rejects every non-public candidate, selects/pins one allowed IP for the connection while preserving the original hostname for Host and TLS SNI/certificate verification, acquires `host_slot(hostname)`, sends, then reads the transport's actual peer address and rejects it unless it equals the pinned public IP. Release that host slot before resolving/acquiring the redirect destination; never hold two domain slots. Redirects are manual and capped at 5. Permit text/html and application/pdf only, stream at most 10 MiB, and use the Task 3 30-second fetch deadline split into bounded connect/read phases. Tests simulate validation against a public IP followed by a private actual peer (DNS rebinding), proxy environment variables, and A→B redirect slot reacquisition without real network.

HtmlParser uses Trafilatura, normalizes the extracted main text and creates `ParsedBlock` entries with HtmlLocator offsets against that exact normalized string. PdfParser uses PyMuPDF, preserves zero-indexed page_index plus block_index, emits `ParsedDocument.blocks`, normalizes block text and records document/page hashes. Their async parse methods check cancellation and run blocking parser work in a worker thread. Unsupported media and a syntactically readable but password-protected/empty/textless PDF return non-retryable PARSE_UNSUPPORTED so the graph can seek HTML/abstract alternatives; malformed bytes that claim to be a supported PDF return INVALID_RESPONSE.

- [ ] **Step 5: 运行 contract、Ruff 和 Pyright**

Run:

    uv run pytest tests/unit/providers tests/unit/test_config.py tests/contracts -q
    uv run ruff check src/deepresearch/config.py src/deepresearch/providers tests/unit/providers
    uv run pyright src/deepresearch/config.py src/deepresearch/providers

Expected: all pass with respx; no real network.

- [ ] **Step 6: 提交**

    git add src/deepresearch/config.py src/deepresearch/providers models/embedding.lock.json tests/unit/providers tests/unit/test_config.py tests/fixtures/providers tests/fixtures/models
    git commit -m "feat: add live provider adapters and document parsers"

### Task 7: P1 Fixed Planner、R1 Similarity 与 evidence-first Writer

**Files:**

- Create: src/deepresearch/planning/__init__.py
- Create: src/deepresearch/planning/validation.py
- Create: src/deepresearch/planning/fixed.py
- Create: src/deepresearch/evidence/__init__.py
- Create: src/deepresearch/evidence/similarity.py
- Create: src/deepresearch/reporting/__init__.py
- Create: src/deepresearch/reporting/boundary.py
- Create: src/deepresearch/reporting/markdown.py
- Test: tests/unit/planning/test_validation.py
- Test: tests/unit/planning/test_fixed.py
- Test: tests/unit/evidence/test_similarity.py
- Test: tests/unit/reporting/test_markdown.py
- Test: tests/unit/reporting/test_boundary.py

**Interfaces:** Consumes ModelProvider, TextEmbedder, domain plan/evidence models and LocalEvidenceStore. Produces PlanValidator, `FixedPlanner.create_plan/queries_for`, non-retryable PLAN_INVALID, R1 `SimilarityRanker.score`, `ContentBoundary`/`identity_content_boundary` and `MarkdownReportWriter.validate_citations` for the baseline graph, planner optimization and service composition.

- [ ] **Step 1: 写 fixed-plan、R1 和 citation 红测**

~~~python
def test_plan_validator_rejects_duplicate_need_and_cycle(plan_factory):
    result = PlanValidator().validate(plan_factory(duplicate_need=True, cycle=True))
    assert result.valid is False
    assert set(result.error_codes) == {"DUPLICATE_NEED_ID", "DEPENDENCY_CYCLE"}


@pytest.mark.parametrize(
    "case,code",
    [
        ("malformed_json", "MALFORMED_JSON"),
        ("schema", "INVALID_SCHEMA"),
        ("duplicate_subquestion", "DUPLICATE_SUBQUESTION_ID"),
        ("empty_goal", "EMPTY_GOAL"),
        ("out_of_scope", "OUT_OF_SCOPE_GOAL"),
        ("unexecutable", "UNEXECUTABLE_GOAL"),
        ("over_budget", "BUDGET_INFEASIBLE"),
    ],
)
def test_plan_validator_emits_stable_public_code(case, code, plan_candidate_factory):
    result = PlanValidator().validate_candidate(
        plan_candidate_factory(case),
        request=research_request(),
        budget=RunBudget.preset("medium"),
    )
    assert code in result.error_codes


@pytest.mark.asyncio
async def test_fixed_planner_generates_once_and_caches_queries(
    model, token, artifact_store
):
    planner = FixedPlanner(
        model=model,
        artifact_store=artifact_store,
        budget=RunBudget.preset("medium"),
        search_depth=2,
    )
    plan = await planner.create_plan(
        research_request(),
        deadline=100.0,
        cancellation_token=token,
    )
    first = await planner.queries_for(
        plan.subquestions[0],
        plan_id=plan.plan_id,
        deadline=100.0,
        cancellation_token=token,
    )
    second = await planner.queries_for(
        plan.subquestions[0],
        plan_id=plan.plan_id,
        deadline=100.0,
        cancellation_token=token,
    )
    assert second == first
    assert len(first) <= 2
    assert model.complete_calls == 1
    assert model.structured_calls == 1  # cached query generation


@pytest.mark.asyncio
async def test_invalid_initial_plan_gets_one_repair_then_plan_invalid(
    always_invalid_plan_model, token, artifact_store
):
    planner = FixedPlanner(
        model=always_invalid_plan_model,
        artifact_store=artifact_store,
        budget=RunBudget.preset("medium"),
    )
    with pytest.raises(PlanGenerationError) as error:
        await planner.create_plan(
            research_request(), deadline=100.0, cancellation_token=token
        )
    assert error.value.code == "PLAN_INVALID"
    assert always_invalid_plan_model.complete_calls == 2


@pytest.mark.asyncio
async def test_similarity_ranker_orders_by_cosine_then_id(embedder, evidence):
    scores = await SimilarityRanker(embedder).score(
        "planner optimization",
        evidence,
        deadline=100.0,
        cancellation_token=token(),
    )
    assert [score.evidence_id for score in scores] == ["ev-2", "ev-1"]
    assert all(set(score.feature_scores) == {"relevance"} for score in scores)


def test_writer_rejects_unknown_inline_citation(evidence_store):
    with pytest.raises(UnknownEvidenceCitation):
        MarkdownReportWriter(evidence_store).validate_citations(
            "Unsupported [E-missing]"
        )


def test_writer_applies_content_boundary_before_prompting(
    evidence_store, recording_model
):
    writer = MarkdownReportWriter(
        evidence_store,
        model=recording_model,
        content_boundary=lambda text: f"<external>{text}</external>",
    )
    writer.render_prompt(selected_evidence_ids=("E-known",))
    assert "<external>fixture excerpt</external>" in recording_model.last_prompt
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/planning tests/unit/evidence tests/unit/reporting -q

Expected: FAIL because the planning, evidence and reporting modules are missing.

- [ ] **Step 3: 实现 P1 和结构化 query**

~~~python
class PlanGenerationError(RuntimeError):
    code: Literal["PLAN_INVALID"] = "PLAN_INVALID"


class PlanValidationReport(BaseModel):
    valid: bool
    candidate: ResearchPlan | None = None
    error_codes: tuple[
        Literal[
            "MALFORMED_JSON",
            "INVALID_SCHEMA",
            "DUPLICATE_SUBQUESTION_ID",
            "DUPLICATE_NEED_ID",
            "UNKNOWN_DEPENDENCY",
            "DEPENDENCY_CYCLE",
            "EMPTY_GOAL",
            "OUT_OF_SCOPE_GOAL",
            "UNEXECUTABLE_GOAL",
            "BUDGET_INFEASIBLE",
        ],
        ...,
    ] = ()
    candidate_artifact_id: str | None = None


class PlanValidator:
    def validate(self, plan: ResearchPlan) -> PlanValidationReport:
        return self.validate_candidate(
            plan,
            request=None,
            budget=None,
        )

    def validate_candidate(
        self,
        candidate: str | bytes | Mapping[str, JsonValue] | ResearchPlan,
        *,
        request: ResearchRequest | None,
        budget: RunBudget | None,
        candidate_artifact_id: str | None = None,
    ) -> PlanValidationReport: ...


class FixedPlanner:
    variant = "P1"

    def __init__(
        self,
        *,
        model: ModelProvider,
        artifact_store: LocalArtifactStore,
        budget: RunBudget,
        search_depth: int = 2,
        prompt_version: str = "fixed-planner-v1",
    ) -> None: ...

    async def create_plan(
        self,
        request: ResearchRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> ResearchPlan: ...

    async def queries_for(
        self,
        subquestion: SubQuestion,
        *,
        plan_id: str,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> tuple[str, ...]: ...
~~~

`PlanValidator.validate` performs model/graph invariants that need no request or budget. `validate_candidate` first parses strings/bytes as JSON, applies ResearchPlan Pydantic validation, then runs subquestion/need dedupe, known acyclic dependencies, non-empty goals; when request/budget are supplied it additionally checks request scope/date/language bounds, supported/executable evidence requirements and a conservative plan-level search/page/token estimate. It always returns `PlanValidationReport`, including stable codes and the caller-supplied candidate_artifact_id, and never throws for candidate invalidity.

`create_plan` uses `ModelProvider.complete`, immediately stores the exact returned UTF-8 bytes in the injected LocalArtifactStore with media type `application/vnd.deepresearch.plan-candidate+json`, passes the resulting artifact ID into `validate_candidate`, then performs only local parsing/validation. This path makes malformed JSON and schema-invalid output repairable; ModelProvider.structured is not used for the initial plan. On failure it makes exactly one repair call containing only a successfully canonicalized rejected candidate (or null for malformed JSON) plus stable public codes; it never includes hidden reasoning or arbitrary raw text. The repair response is stored as a second private artifact before validation. A second failure raises non-retryable PlanGenerationError(code="PLAN_INVALID"). No generic provider retry/fallback is applied to semantic validation. `queries_for` uses structured output, returns at most search_depth normalized queries per subquestion, caches by `(plan_id, subquestion.id, prompt_version)`, and never replans after evidence arrives.

- [ ] **Step 4: 实现 R1 与 Writer**

SimilarityRanker uses the Task 6 locked TextEmbedder and maps cosine from [-1,1] to [0,1] with `(cosine + 1) / 2`, clamping only floating-point overflow at the final boundary. It uses evidence_id as the deterministic tie-break and writes only relevance in feature_scores. CI/replay never instantiate or download the SentenceTransformer model.

Every prompt assembler that can include fetched, parsed or user-supplied external text accepts the same additive boundary hook:

~~~python
ContentBoundary: TypeAlias = Callable[[str], str]


def identity_content_boundary(text: str) -> str:
    return text
~~~

`ContentBoundary` and `identity_content_boundary` live in `reporting/boundary.py` and are re-exported from `deepresearch.reporting`. The Core/local default is identity so this addition does not change baseline behavior. `MarkdownReportWriter(..., content_boundary=identity_content_boundary)` applies the callable separately to every excerpt/title/metadata string at the final moment before prompt serialization. Later Planner prompt assemblers receive the same callable through graph dependencies and must apply it to every search result or evidence excerpt before model input. The Service runner factory injects its security wrapper here; it must not monkey-patch provider payloads or duplicate prompt rendering.

MarkdownReportWriter gives the model only selected evidence IDs, boundary-wrapped excerpts, locators and source metadata. It requires inline [E-...] IDs, validates every ID against EvidenceStore, emits a References section with canonical URLs, and marks a partial report with the stop reason and uncovered information needs.

~~~python
class SimilarityRanker:
    ranker_id = "R1"

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder

    async def score(
        self,
        information_need: str,
        evidence_spans: Sequence[EvidenceSpan],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[RerankScore]:
        texts = [information_need, *(span.excerpt for span in evidence_spans)]
        vectors = await self.embedder.embed(
            texts,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        query_vector, evidence_vectors = vectors[0], vectors[1:]
        scores = []
        for span, vector in zip(evidence_spans, evidence_vectors, strict=True):
            relevance = clip01((cosine(query_vector, vector) + 1.0) / 2.0)
            scores.append(
                RerankScore(
                    evidence_id=span.evidence_id,
                    total=relevance,
                    feature_scores={"relevance": relevance},
                    model_id=self.embedder.model_id,
                    prompt_version=None,
                )
            )
        return sorted(scores, key=lambda item: (-item.total, item.evidence_id))


_CITATION = re.compile(r"\[(E-[A-Za-z0-9_-]+)\]")


def validate_citations(
    report_markdown: str,
    evidence_store: LocalEvidenceStore,
) -> tuple[str, ...]:
    citation_ids = tuple(dict.fromkeys(_CITATION.findall(report_markdown)))
    missing = [
        evidence_id
        for evidence_id in citation_ids
        if not evidence_store.has_evidence(evidence_id)
    ]
    if missing:
        raise UnknownEvidenceCitation(", ".join(missing))
    return citation_ids
~~~

- [ ] **Step 5: 运行测试与提交**

Run:

    uv run pytest tests/unit/planning tests/unit/evidence tests/unit/reporting -q
    uv run ruff check src/deepresearch/planning src/deepresearch/evidence src/deepresearch/reporting tests/unit
    uv run pyright src/deepresearch/planning src/deepresearch/evidence src/deepresearch/reporting

Expected: all pass.

    git add src/deepresearch/planning src/deepresearch/evidence src/deepresearch/reporting tests/unit/planning tests/unit/evidence tests/unit/reporting
    git commit -m "feat: add fixed planner similarity ranker and cited writer"

### Task 8: 构建受控 LangGraph Baseline 和 ResearchRunner

**Files:**

- Create: src/deepresearch/runtime/ports.py
- Create: src/deepresearch/runtime/checkpoints.py
- Modify: src/deepresearch/runtime/__init__.py
- Create: src/deepresearch/workflow/__init__.py
- Create: src/deepresearch/workflow/state.py
- Create: src/deepresearch/workflow/baseline_graph.py
- Create: src/deepresearch/workflow/runner.py
- Test: tests/unit/workflow/test_state.py
- Test: tests/unit/workflow/test_baseline_routes.py
- Test: tests/unit/runtime/test_checkpoints.py
- Test: tests/integration/replay/test_baseline_graph.py

**Interfaces:** Consumes P1/R1/Writer, `ContentBoundary`, Replay providers, stores and BudgetAccountant. Produces `checkpoint_serializer()`, `open_sqlite_checkpointer(path)`, `CheckpointRef`, `ResearchRunner.run`, checkpoint-safe `BaselineState`/`BaselineBlockedNeed`, `decide_baseline_stop`, `build_baseline_graph` and `LangGraphResearchRunner`; service construction injects a concrete saver while resume callers pass only CheckpointRef.

- [ ] **Step 1: 写状态边界、路由和 runner 红测**

~~~python
def test_graph_state_contains_ids_not_raw_documents():
    annotations = get_type_hints(BaselineState)
    forbidden = {RawDocument, ParsedDocument, bytes}
    assert forbidden.isdisjoint(set(annotations.values()))
    assert {
        "run_id",
        "thread_id",
        "request",
        "plan_id",
        "pending_subquestion_ids",
        "selected_evidence_ids",
        "coverage_ledger",
        "high_priority_unresolved_conflict_ids",
        "blocked_needs",
        "recent_marginal_gains",
        "budget_snapshot",
        "stop_reason",
        "is_partial",
        "report_artifact_id",
    } <= set(annotations)


def test_baseline_route_searches_until_fixed_queue_empty():
    state = state_with(pending_subquestion_ids=("sq-1",))
    assert route_after_decide(state) == "Search"
    assert route_after_decide(
        state_with(pending_subquestion_ids=(), stop_reason="SUFFICIENT")
    ) == "DraftReport"


def test_sufficient_requires_thresholds_sources_and_no_priority_conflict(plan):
    ledger = ledger_for(
        plan,
        coverage={"sq-1": 0.90, "sq-2": 0.84},
        independent_sources={"sq-1": 2, "sq-2": 2},
    )
    assert baseline_is_sufficient(plan, ledger) is False
    ledger = ledger_for(
        plan,
        coverage={"sq-1": 0.90, "sq-2": 0.86},
        independent_sources={"sq-1": 2, "sq-2": 2},
        high_priority_conflicts=("conflict-1",),
    )
    assert baseline_is_sufficient(plan, ledger) is False


def test_baseline_never_fabricates_blocked_when_queue_is_only_empty(plan):
    state = state_with(
        pending_subquestion_ids=(),
        blocked_needs=(),
        recent_marginal_gains=(0.20, 0.10),
    )
    with pytest.raises(WorkflowInvariantError) as error:
        decide_baseline_stop(state, plan)
    assert error.value.code == "NO_LEGAL_CONTINUATION"


def test_baseline_blocked_requires_typed_exhaustion(plan):
    state = state_with(
        pending_subquestion_ids=(),
        blocked_needs=(
            {
                "need_id": "need-1",
                "required_source_unavailable": True,
                "alternative_strategies_exhausted": True,
                "retry_count": 2,
                "max_retries": 2,
            },
        ),
    )
    assert decide_baseline_stop(state, plan) == "BLOCKED"


@pytest.mark.asyncio
async def test_replay_baseline_emits_events_and_artifacts(harness):
    result = await harness.run_fixture("baseline")
    assert result.status == "completed"
    assert result.stop_reason in {"SUFFICIENT", "BUDGET_EXHAUSTED"}
    assert result.report_artifact_id
    assert result.evidence_graph_artifact_id
    assert result.manifest_artifact_id
    assert [event.seq for event in harness.events] == list(
        range(1, len(harness.events) + 1)
    )


@pytest.mark.asyncio
async def test_sqlite_checkpointer_survives_reopen_and_resume(
    tmp_path, interruptible_graph
):
    path = (tmp_path / "checkpoints.sqlite3").resolve()
    async with open_sqlite_checkpointer(path) as saver:
        interrupted = await interruptible_graph(saver).run_until_interrupt("thread-1")
    async with open_sqlite_checkpointer(path) as saver:
        resumed = await interruptible_graph(saver).resume(interrupted)
    assert resumed.output == "complete"
    assert resumed.side_effect_calls == 1


@pytest.mark.asyncio
async def test_sqlite_checkpointer_rejects_unapproved_serialized_type(tmp_path):
    async with open_sqlite_checkpointer((tmp_path / "strict.sqlite3").resolve()) as saver:
        restored = await round_trip_checkpoint(
            saver,
            {
                "request": research_request(),
                "coverage_ledger": coverage_ledger_entries(),
                "legacy_bytes": b"checkpoint-bytes",
            },
        )
        assert restored["request"] == research_request()
        assert restored["legacy_bytes"] == b"checkpoint-bytes"
        with pytest.raises(CheckpointSerializationError):
            await put_unapproved_object(saver, object())
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/workflow tests/unit/runtime/test_checkpoints.py tests/integration/replay/test_baseline_graph.py -q

Expected: FAIL because workflow state, concrete SQLite checkpointer, graph and runner are missing.

- [ ] **Step 3: 定义共享 Runner port 与 checkpoint-safe state**

~~~python
class ResearchRunner(Protocol):
    async def run(
        self,
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig,
        checkpoint: CheckpointRef | None,
        emit: Callable[[RunEvent], Awaitable[None]],
        cancellation_token: CancellationToken,
    ) -> RunResult: ...


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    thread_id: str
    created_at: datetime


class BaselineState(TypedDict):
    run_id: str
    thread_id: str
    request: ResearchRequest
    plan_id: str | None
    pending_subquestion_ids: tuple[str, ...]
    active_subquestion_id: str | None
    query_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    selected_evidence_ids: tuple[str, ...]
    coverage_ledger: tuple[CoverageLedgerEntry, ...]
    high_priority_unresolved_conflict_ids: tuple[str, ...]
    blocked_needs: tuple["BaselineBlockedNeed", ...]
    recent_marginal_gains: tuple[float, ...]
    budget_snapshot: BudgetSnapshot
    stop_reason: StopReason | None
    is_partial: bool
    report_artifact_id: str | None
    evidence_graph_artifact_id: str | None
    manifest_artifact_id: str | None
    error_code: str | None


class BaselineBlockedNeed(TypedDict):
    need_id: str
    required_source_unavailable: bool
    alternative_strategies_exhausted: bool
    retry_count: int
    max_retries: int
~~~

CheckpointRef is defined in runtime/ports.py and re-exported with ResearchRunner, CancellationToken, BudgetAccountant and BudgetSnapshot from runtime/__init__.py. It is not LangGraph's concrete saver object.

- [ ] **Step 4: 实现 concrete SQLite checkpointer factory**

~~~python
_ALLOWED_CHECKPOINT_TYPES = (
    ResearchRequest,
    ResearchPlan,
    SubQuestion,
    InformationNeed,
    CoverageLedgerEntry,
    RunBudget,
    ResourceUsage,
    BudgetSnapshot,
)


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_json_modules=tuple(
            (item.__module__, item.__name__) for item in _ALLOWED_CHECKPOINT_TYPES
        ),
        allowed_msgpack_modules=_ALLOWED_CHECKPOINT_TYPES,
        pickle_fallback=False,
    )


@asynccontextmanager
async def open_sqlite_checkpointer(
    path: Path,
) -> AsyncIterator[BaseCheckpointSaver]:
    if not path.is_absolute():
        raise ValueError("checkpoint path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    serde = checkpoint_serializer()
    async with aiosqlite.connect(path) as connection:
        saver = AsyncSqliteSaver(connection, serde=serde)
        await saver.setup()
        yield saver
~~~

`checkpoint_serializer()` is the sole public serializer factory and is reused unchanged by the Service Postgres saver; downstream plans must not duplicate the allow-list. JSON revival receives exact `(module, class-name)` tuples, msgpack revival receives concrete types, and built-in bytes remain round-trippable without adding a broad custom module. `open_sqlite_checkpointer` is the sole local concrete saver factory. It owns the connection for exactly one async context, creates schema idempotently, rejects relative/broad paths, and disables pickle fallback. The Runner converts LangGraph config `{thread_id, checkpoint_ns, checkpoint_id}` to/from `CheckpointRef`; resume always reopens the same database and requests the exact checkpoint ID. Tests round-trip a real state containing ResearchRequest and CoverageLedgerEntry tuples plus legacy bytes, reject an unapproved object, interrupt after a persisted side effect, close the first connection, reopen, resume, and prove the effect is not repeated. This follows the official [LangGraph SQLite Checkpoint](https://pypi.org/project/langgraph-checkpoint-sqlite/) async saver and [JsonPlusSerializer allow-list](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py) guidance.

- [ ] **Step 5: 构建显式 StateGraph**

~~~python
StateUpdate: TypeAlias = Mapping[str, object]
BaselineNode: TypeAlias = Callable[[BaselineState], Awaitable[StateUpdate]]


@dataclass(frozen=True)
class BaselineDependencies:
    checkpointer: BaseCheckpointSaver
    validate_request: BaselineNode
    plan: BaselineNode
    decide_next: BaselineNode
    search: BaselineNode
    fetch: BaselineNode
    parse_and_normalize: BaselineNode
    store_evidence: BaselineNode
    rank_evidence: BaselineNode
    content_boundary: ContentBoundary
    draft_report: BaselineNode
    finalize_citations: BaselineNode
    persist_results: BaselineNode


def build_baseline_graph(dependencies: BaselineDependencies) -> CompiledStateGraph:
    graph = StateGraph(BaselineState)
    graph.add_node("ValidateRequest", dependencies.validate_request)
    graph.add_node("Plan", dependencies.plan)
    graph.add_node("DecideNext", dependencies.decide_next)
    graph.add_node("Search", dependencies.search)
    graph.add_node("Fetch", dependencies.fetch)
    graph.add_node("ParseAndNormalize", dependencies.parse_and_normalize)
    graph.add_node("StoreEvidence", dependencies.store_evidence)
    graph.add_node("RankEvidence", dependencies.rank_evidence)
    graph.add_node("DraftReport", dependencies.draft_report)
    graph.add_node("FinalizeCitations", dependencies.finalize_citations)
    graph.add_node("PersistResults", dependencies.persist_results)
    graph.set_entry_point("ValidateRequest")
    graph.add_edge("ValidateRequest", "Plan")
    graph.add_edge("Plan", "DecideNext")
    graph.add_conditional_edges("DecideNext", route_after_decide)
    graph.add_edge("Search", "Fetch")
    graph.add_edge("Fetch", "ParseAndNormalize")
    graph.add_edge("ParseAndNormalize", "StoreEvidence")
    graph.add_edge("StoreEvidence", "RankEvidence")
    graph.add_edge("RankEvidence", "DecideNext")
    graph.add_edge("DraftReport", "FinalizeCitations")
    graph.add_edge("FinalizeCitations", "PersistResults")
    graph.add_edge("PersistResults", END)
    return graph.compile(checkpointer=dependencies.checkpointer)


def decide_baseline_stop(
    state: BaselineState,
    plan: ResearchPlan,
) -> StopReason | None: ...
~~~

Each node reserves budget before an external call, checks cancellation, persists its result before returning IDs, then emits one public RunEvent. Use idempotency key SHA-256(run_id, node, logical_input_hash); checkpoint resume sees a completed key and reads its artifact instead of repeating the call.

`BaselineDependencies.content_boundary` is passed into every prompt assembler. Baseline Planner/query prompts do not currently contain fetched text, but the dependency is still threaded explicitly so `research-v1` can use the identical hook. `DraftReport` must call it for every external field before constructing `ModelRequest`; tests fail if a raw excerpt reaches ModelProvider. Directly persisted evidence remains unwrapped so locators and hashes stay verifiable.

Baseline sufficiency uses the same public predicate as later planners: every plan subquestion must have `coverage_score >= 0.85`; its `independent_source_count` must meet `EvidenceRequirements.min_independent_sources` (default fixtures use two), except a need explicitly marked as describing one primary source may be satisfied by that single primary source; importance-weighted coverage must be at least 0.80; and `high_priority_unresolved_conflict_ids` must be empty. The predicate returns only bool plus stable reason codes, never hidden reasoning. True produces SUFFICIENT. A hard budget reached produces BUDGET_EXHAUSTED and a partial report. Two consecutive completed evidence rounds with measured marginal gain below 0.05 produce PLATEAU. BLOCKED is legal only when an internal typed `BaselineBlockedNeed` record proves a required source unavailable, all declared alternative strategies exhausted and retry_count >= max_retries; merely finishing the fixed query list or lacking a target is not BLOCKED. The later research-v1 graph maps its richer public `BlockedNeed` into the same predicate. If sufficiency is false and none of those typed terminal predicates holds, `DecideNext` raises `WorkflowInvariantError(code="NO_LEGAL_CONTINUATION")`, the runner returns status failed with stop_reason null, and PersistResults records the public-safe code. Do not use the weaker “one span per need” shortcut or fabricate terminal evidence.

- [ ] **Step 6: 实现 LangGraphResearchRunner**

LangGraphResearchRunner implements ResearchRunner.run, requires workflow_id=baseline-v1 with planner_id=P1 and ranker_id=R1 in this plan, supplies thread_id to LangGraph checkpoint config, converts node failures to typed RunResult, and always attempts PersistResults if at least one evidence span exists. It never exposes raw LangGraph events. The Planner/Evidence plan later adds research-v1 selection without altering baseline-v1.

- [ ] **Step 7: 运行测试和提交**

Run:

    uv run pytest tests/unit/workflow tests/unit/runtime/test_checkpoints.py tests/integration/replay/test_baseline_graph.py -q
    uv run ruff check src/deepresearch/workflow src/deepresearch/runtime/ports.py src/deepresearch/runtime/checkpoints.py tests/unit/workflow tests/unit/runtime/test_checkpoints.py tests/integration/replay
    uv run pyright src/deepresearch/workflow src/deepresearch/runtime/ports.py src/deepresearch/runtime/checkpoints.py

Expected: PASS.

    git add src/deepresearch/runtime/ports.py src/deepresearch/runtime/checkpoints.py src/deepresearch/runtime/__init__.py src/deepresearch/workflow tests/unit/runtime/test_checkpoints.py tests/unit/workflow tests/integration/replay/test_baseline_graph.py
    git commit -m "feat: run fixed deep research baseline in LangGraph"

### Task 9: 完成 Baseline Replay fixture 与 CLI

**Files:**

- Modify: apps/cli/main.py
- Create: tests/cli/test_research_command.py
- Create: tests/fixtures/replay/baseline/snapshot.json
- Create: tests/fixtures/replay/baseline/search.jsonl
- Create: tests/fixtures/replay/baseline/documents.jsonl
- Create: tests/fixtures/replay/baseline/model_responses.jsonl
- Create: tests/fixtures/replay/baseline/embeddings.jsonl
- Create: tests/fixtures/replay/baseline/expected-report.md
- Create: tests/fixtures/replay/baseline/expected-evidence.json
- Create: tests/fixtures/replay/baseline/manifest.sha256

**Interfaces:** Consumes LangGraphResearchRunner, the SQLite checkpointer factory and Live/Replay/recording provider factories. Produces the `deepresearch research` command, a fully offline baseline replay bundle (including embeddings) and the stable public files report.md/evidence.json/run-manifest.json.

- [ ] **Step 1: 写 CLI 红测**

~~~python
def test_research_replay_writes_all_public_outputs(cli_runner, tmp_path):
    result = cli_runner.invoke(
        app,
        [
            "research",
            "--question",
            "Compare planner strategies",
            "--mode",
            "replay",
            "--replay-root",
            "tests/fixtures/replay/baseline",
            "--budget",
            "medium",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "evidence.json").exists()
    assert (tmp_path / "run-manifest.json").exists()
    assert "SUFFICIENT" in result.stdout


def test_replay_cli_rejects_live_credentials(cli_runner):
    result = cli_runner.invoke(
        app,
        [
            "research",
            "--mode",
            "replay",
            "--provider-api-key",
            "secret",
        ],
    )
    assert result.exit_code != 0
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/cli/test_research_command.py -q

Expected: FAIL because research command and fixture are missing.

- [ ] **Step 3: 构建完整可重放 fixture**

The fixture question has two subquestions and four deterministic queries. Include two HTML documents and one PDF document, at least four EvidenceSpan records, one near-duplicate source, recorded plan/query/write structured model calls, recorded 384-dimensional embedding responses and exact SHA-256 values. expected-report.md cites only stored E- IDs; expected-evidence.json contains sources, locators and rank scores but no full copyrighted pages. Replay constructs `ReplayTextEmbedder` from embeddings.jsonl and never imports SentenceTransformer or touches the network.

- [ ] **Step 4: 实现 research 命令**

~~~python
@app.command()
def research(
    question: Annotated[str, typer.Option()],
    mode: Annotated[Literal["live", "replay", "hybrid"], typer.Option()],
    replay_root: Annotated[Path | None, typer.Option()] = None,
    record_replay_root: Annotated[Path | None, typer.Option()] = None,
    checkpoint_db: Annotated[Path, typer.Option()] = Path("artifacts/checkpoints.sqlite3"),
    resume_checkpoint: Annotated[str | None, typer.Option()] = None,
    budget: Annotated[Literal["low", "medium", "high"], typer.Option()] = "medium",
    output: Annotated[Path, typer.Option()] = Path("artifacts/latest"),
) -> None: ...
~~~

Replay requires replay_root and never constructs Live providers or SentenceTransformer. Live requires Settings credentials and may use OpenAICompatibleModelProvider plus TavilySearchProvider plus the verified local locked embedder. Hybrid requires frozen Search/Fetch plus a Live model profile. `--record-replay-root` is valid only for live/hybrid, requires a nonexistent destination, and wraps model/search/fetch/embed providers with Task 5 recorders; finalize occurs only after manifest persistence. The command resolves checkpoint_db to an absolute path, opens Task 8 SQLite saver for the run, and maps resume_checkpoint plus thread_id to an exact CheckpointRef. It creates a new run directory atomically, streams compact RunEvent rows to Rich, then writes report.md, evidence.json and run-manifest.json. It exits non-zero on REPLAY_MISS, PLAN_INVALID, a replay/record option conflict or missing report.

- [ ] **Step 5: 运行 CLI 两次确认确定性**

Run:

    uv run pytest tests/cli/test_research_command.py tests/integration/replay/test_baseline_graph.py -q
    uv run deepresearch research --question "Compare planner strategies" --mode replay --replay-root tests/fixtures/replay/baseline --budget medium --output artifacts/replay-check

Expected: PASS; CLI prints a completed run and three public output paths. Delete no output during verification.

Run the same fixture into artifacts/replay-check-2 and compare SHA-256 of report.md and evidence.json. They must match; manifest differs only in explicitly allowed run/time identity fields and links replay_parent correctly.

- [ ] **Step 6: 提交**

    git add apps/cli/main.py tests/cli/test_research_command.py tests/fixtures/replay/baseline
    git commit -m "feat: expose replayable deep research baseline CLI"

### Task 10: 锁定架构边界、全量离线验证与 README

**Files:**

- Create: tests/unit/test_architecture_boundaries.py
- Create: tests/unit/test_secret_redaction_boundary.py
- Modify: README.md

**Interfaces:** Consumes the completed Core package only. Produces permanent architecture/import/redaction tests and Core quickstart documentation; it must remain green after later Planner, benchmark and service packages are added.

- [ ] **Step 1: 写边界红测**

~~~python
def test_external_sdks_are_imported_only_in_provider_modules():
    violations = imports_matching(
        roots=["src/deepresearch", "apps"],
        prefixes=["openai", "tavily", "fitz", "trafilatura"],
        allowed_roots=["src/deepresearch/providers"],
    )
    assert violations == []


def test_core_baseline_does_not_depend_on_later_service_packages():
    violations = imports_matching(
        roots=["src/deepresearch"],
        prefixes=["apps.api", "apps.ui"],
        allowed_roots=[],
    )
    assert violations == []


def test_graph_state_contains_no_raw_provider_payloads():
    assert "raw_document" not in BaselineState.__annotations__
    assert "model_response" not in BaselineState.__annotations__


def test_secret_fields_never_serialize(settings, run_config, manifest):
    serialized = json.dumps(
        [run_config.model_dump(mode="json"), manifest.model_dump(mode="json")]
    )
    assert settings.model_api_key.get_secret_value() not in serialized
~~~

- [ ] **Step 2: 运行边界测试**

Run: uv run pytest tests/unit/test_architecture_boundaries.py tests/unit/test_secret_redaction_boundary.py -q

Expected: FAIL on any accidental cross-layer import or secret-bearing model field; otherwise the new tests should pass after fixtures are wired.

- [ ] **Step 3: 修正实际边界问题，不放宽断言**

Move SDK imports into adapters, keep the Core dependency direction free of `apps.api`/`apps.ui`, replace raw bodies in Graph State with artifact IDs, and keep SecretStr fields only in Settings. The service plan may later create those packages; this boundary test deliberately checks imports rather than filesystem absence so it remains valid in the final full repository. Do not add allow-list entries merely to make the test green.

- [ ] **Step 4: 写 Core README quickstart**

README must include:

- project outcome and non-goals;
- Python 3.12 and uv setup;
- copy .env.example to .env for Live, without real secrets;
- one strict Replay command that works offline;
- one Live command with budget warning;
- exact output files and citation format;
- architecture boundaries and Provider protocol;
- link to design plus all four implementation plans;
- note that this plan is P1+R1 baseline and later plans add P2/R2, evaluation and service.

- [ ] **Step 5: 运行完整质量门**

Run:

    uv lock --check
    uv run ruff check .
    uv run pyright src apps
    uv run pytest tests/unit tests/contracts tests/integration/replay tests/cli -q
    uv run python -m compileall -q src apps benchmarks experiments
    git diff --check

Expected: all commands exit 0. Normal test output must show no network access and no deselected failing tests.

- [ ] **Step 6: 提交**

    git add README.md tests/unit/test_architecture_boundaries.py tests/unit/test_secret_redaction_boundary.py
    git commit -m "docs: document and verify replay baseline boundaries"

## Final Acceptance

- [ ] A fresh Python 3.12 environment installs from uv.lock and passes Ruff, Pyright and all offline tests.
- [ ] deepresearch.domain, providers.protocols and runtime.ports are the only shared public contract surfaces used by later plans.
- [ ] P1 + R1 executes through a compiled LangGraph StateGraph, not a hand-written orchestration loop.
- [ ] Strict Replay produces the expected report, evidence JSON and manifest without constructing Live providers.
- [ ] Live OpenAI-compatible, Tavily, HTTPX, HTML and PDF adapters pass the common mocked contracts.
- [ ] Budget, usage, cache, checkpoint and idempotency behavior are observable and tested.
- [ ] Every citation points to a stored EvidenceSpan with a valid locator and content hash.
- [ ] Graph State contains IDs and small typed state only; artifacts and secrets stay outside it.

## Execution Handoff

Plan complete. Choose one execution mode before implementation:

1. Subagent-Driven (recommended): use superpowers:subagent-driven-development in this session, one fresh worker and two-stage review per task.
2. Inline Execution: start a separate implementation session with superpowers:executing-plans and run tasks sequentially at the documented checkpoints.

Do not start the Planner/Evidence plan until every Final Acceptance item above is verified.
