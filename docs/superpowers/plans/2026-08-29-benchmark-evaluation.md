# Benchmark 与实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 建立可冻结、可隔离、可重放的 AI/CS 研究评测系统，用内部 60 题、三类互补实验协议和统计检验证明 Planner 与 Evidence Ranker 的真实收益。

**Architecture:** 评测运行时只读取脱敏后的 RuntimeTask、任务级 FrozenCorpus 和锁定配置；独立 evaluator 进程读取私有 gold 并产出分项指标。ExperimentRunner 通过与产品运行相同的 ResearchRunner 执行 P0/P1/P2 与 R0/R1/R2，所有正式调用写入 strict Replay 和不可变 manifest。

**Tech Stack:** Python 3.12、Pydantic v2、Typer、PyYAML、rank-bm25、NumPy、SciPy、pandas、pytest、Ruff、Pyright。

**Spec:** [Multi-Agent Deep Research 设计文档](../specs/2026-08-29-multi-agent-deep-research-design.md)

## Global Constraints

- 本计划依赖 [核心底座与 Replay 基线计划](./2026-08-29-core-foundation-replay-baseline.md) 和 [Planner / Evidence 优化计划](./2026-08-29-planner-evidence-optimization.md) 已完成。
- 严格复用 deepresearch.domain、deepresearch.providers.protocols、deepresearch.runtime.manifest、deepresearch.planning 和 deepresearch.evidence 的公共类型；不得在 benchmarks 或 experiments 中复制一套相似模型。
- Core owns .gitignore, pyproject.toml, uv.lock, apps/cli/main.py and README.md. This plan only appends benchmark ignore rules/dependencies, registers the experiment Typer sub-app, and adds evaluation/result sections; it may not replace existing content or public commands.
- Agent 进程只接收一题 `RuntimeTask`、该题 `snapshot_dir`、位于 group run-root 内且哈希验证的 sealed-config 副本，以及 ranker-component 专用的同 run-root 哈希验证 candidate-pool artifact；不存在第二种 task schema。私有 gold 根目录只注入 evaluator 进程；evaluator 必须先把脱敏后的单题 `RuntimeTask` 原子写入独立 agent-input root，不得把 private-root 下的原始路径传给 agent。Gold、private-root 路径、仓库内原始 sealed-config 路径及 `DEEPRESEARCH_BENCHMARK_GOLD_ROOT` 不得进入 AgentRunRequest、Graph State、prompt、Provider metadata、RunManifest 或公开 artifact。
- 普通 CI 只使用小型 fixture、FakeModelProvider 和 ReplayModelProvider；不得调用付费模型、公开搜索或下载外部数据。
- FrozenCorpusSearchProvider 对任意合法 query 都必须确定性返回；未知或损坏 snapshot 返回 REPLAY_MISS 或 INVALID_SNAPSHOT，绝不回退到在线搜索。
- 正式实验配置任一冻结字段变化都必须生成新的 experiment_group_id，禁止覆盖旧结果。Seed 列表、不支持 seed 时的 repeat 数和成本敏感性预算组都是 sealed config 字段，runner/CLI 不得在 config 之外追加任意 seed 或预算。
- 正式实验必须携带 Core `PricingSnapshot` 的 `(provider_id, endpoint_type, model_id)` 身份、日期与四类 token 单价；缺失价格、负单价、非 USD、调用身份不匹配或 `pricing_status!=estimated` 时在任何模型调用前拒绝启动。正式结果中的 `cost_usd` 只由 Core `CostCalculator` 计算并标为基于该快照的估算值。
- `formal.template.yaml` 是可评审输入，不是可运行配置。Task 16 在所有 **primary automatic evaluation** 结果代码提交后生成并提交 `formal.yaml`；Task 17 为后续 external extension 生成独立 `formal-portfolio.yaml`。正式 runner 只接受相应 sealed 文件、干净的结果影响源树、匹配的 code/model/environment hashes 和当前 40-character commit。
- 先在 task/config 内对 sealed `seed_values` 求均值，再以 task ID 为重采样单位做分层 paired bootstrap；不得把 seed 当成独立样本。
- 不制造单一综合分数。质量、证据、检索效率、延迟和费用分别报告。
- 每个代码任务遵循红—绿—重构；只提交本任务文件；每次提交前运行目标测试和 uv run ruff check。
- Tasks 5–10 are data-curation tasks, so their red/green loop is validator-driven: missing or incomplete batch must fail first, then the fully annotated batch/snapshots must pass the same immutable validator before export. They do not invent production code solely to mimic source-code TDD.
- 所有 JSON/JSONL 输出采用 UTF-8、稳定 key 顺序和末尾换行；时间统一为 UTC ISO 8601。
- 所有随机操作显式接收 seed；同一输入、版本和 seed 必须产生逐字节一致的 evaluator JSON。

## Exact File Map

Create:

    benchmarks/__init__.py
    benchmarks/datasets/__init__.py
    benchmarks/datasets/models.py
    benchmarks/datasets/isolation.py
    benchmarks/processes/__init__.py
    benchmarks/processes/agent.py
    benchmarks/processes/evaluator.py
    benchmarks/datasets/validator.py
    benchmarks/datasets/builder.py
    benchmarks/datasets/README.md
    benchmarks/datasets/templates/question.example.json
    benchmarks/datasets/frozen_ai_cs_60/public_manifest.json
    benchmarks/datasets/frozen_ai_cs_60/runtime/dev/technical_survey.jsonl
    benchmarks/datasets/frozen_ai_cs_60/runtime/dev/method_comparison.jsonl
    benchmarks/datasets/frozen_ai_cs_60/runtime/dev/multi_hop_history.jsonl
    benchmarks/datasets/frozen_ai_cs_60/runtime/dev/freshness.jsonl
    benchmarks/datasets/frozen_ai_cs_60/runtime/dev/bilingual.jsonl
    benchmarks/datasets/frozen_ai_cs_60/runtime/dev/source_conflict.jsonl
    benchmarks/snapshots/README.md
    benchmarks/evaluators/__init__.py
    benchmarks/evaluators/metrics.py
    benchmarks/evaluators/statistics.py
    benchmarks/evaluators/pareto.py
    benchmarks/evaluators/oracle.py
    benchmarks/evaluators/human.py
    benchmarks/configs/formal.template.yaml
    benchmarks/configs/external.yaml
    benchmarks/external/__init__.py
    benchmarks/external/base.py
    benchmarks/external/livedrbench.py
    benchmarks/external/frames.py
    benchmarks/external/deepresearchbench.py
    benchmarks/external/external.lock.json
    benchmarks/scripts/fetch_external.py
    benchmarks/scripts/build_snapshot.py
    benchmarks/scripts/lock_model.py
    benchmarks/scripts/capture_inference_environment.py
    benchmarks/scripts/render_results.py
    src/deepresearch/providers/frozen_index.py
    src/deepresearch/providers/frozen_search.py
    experiments/__init__.py
    experiments/models.py
    experiments/config.py
    experiments/factories.py
    experiments/runner.py
    experiments/external_runner.py
    experiments/summarize.py
    apps/cli/experiment.py
    tests/unit/benchmarks/test_models.py
    tests/unit/benchmarks/test_isolation.py
    tests/unit/benchmarks/test_validator.py
    tests/unit/benchmarks/test_builder.py
    tests/unit/benchmarks/test_metrics.py
    tests/unit/benchmarks/test_statistics.py
    tests/unit/benchmarks/test_pareto.py
    tests/unit/benchmarks/test_oracle.py
    tests/unit/benchmarks/test_human.py
    tests/unit/benchmarks/test_external_adapters.py
    tests/unit/providers/test_frozen_index.py
    tests/unit/providers/test_frozen_search.py
    tests/unit/experiments/test_config.py
    tests/unit/experiments/test_factories.py
    tests/contracts/test_frozen_search_contract.py
    tests/integration/experiments/test_abcd_runner.py
    tests/integration/experiments/test_strict_replay.py
    tests/integration/experiments/test_external_runner.py
    tests/integration/benchmarks/test_formal_protocols.py
    tests/integration/benchmarks/test_process_isolation.py
    tests/cli/test_experiment_commands.py
    tests/fixtures/benchmark/minimal_private/
    tests/fixtures/benchmark/minimal_runtime/
    tests/fixtures/frozen_corpus/task-fixture/
    tests/fixtures/experiments/
    docs/evaluation.md
    docs/results.md

Modify:

    .gitignore
    pyproject.toml
    apps/cli/main.py
    README.md
    src/deepresearch/providers/replay.py
    src/deepresearch/workflow/research_graph.py
    experiments/runner.py

Generated but not hand-edited:

    uv.lock
    benchmarks/configs/formal.yaml
    benchmarks/configs/formal-portfolio.yaml
    benchmarks/configs/qwen3-8b.lock.json
    benchmarks/configs/inference-environment.lock.json
    benchmarks/private/frozen_ai_cs_60/batches/*.jsonl
    benchmarks/private/frozen_ai_cs_60/gold/*.jsonl
    benchmarks/private/frozen_ai_cs_60/runtime/test/*.jsonl
    benchmarks/private/external/raw/**
    benchmarks/private/external/staging/**
    benchmarks/private/human/blinded_packets/*
    benchmarks/private/human/ratings.jsonl
    benchmarks/snapshots/frozen_ai_cs_60/ (one child directory per task_id)
    benchmarks/snapshots/external/ (one ignored child directory per external task_id)
    experiments/ (one child directory per experiment_group_id)
    docs/assets/results/*.svg

## Stable Benchmark Interfaces

The following boundaries apply to every task:

- Private input: one UTF-8 AnnotatedQuestion JSON object per line plus one FrozenEvidenceRecord JSONL per task.
- Agent-visible output: RuntimeTask JSONL containing only task_id, category, request, evaluation_cutoff, snapshot_id, corpus_version and index_version.
- Snapshot output: documents.jsonl, index.json, snapshot.json and manifest.sha256; build is staging-then-rename and never overwrites an existing snapshot.
- Validator output: one stable BatchValidationReport or DatasetValidationReport JSON object; exit 0 means valid, exit 2 means schema/data invalid, exit 3 means isolation violation, and exit 4 means snapshot/hash failure.
- Experiment raw output: immutable task/config/replication/budget records keyed by SHA-256(group, protocol, variant, task_id, seed-or-repeat, budget_preset), with the exact pricing snapshot ID and estimated USD usage copied from each RunManifest.
- Evaluator output: task_metrics.jsonl plus summary/confidence/Pareto/failure JSON; evaluator artifacts never contain private gold records.
- Public result output: deterministic Markdown/SVG and public hashes only.

Library code raises typed DatasetValidationError, GoldAccessViolation, ProviderError or ResultValidationError; CLI adapters map only those errors to the documented exit codes.

---

### Task 1: 建立 Benchmark Schema 与运行时视图

**Files:**

- Create: benchmarks/__init__.py
- Create: benchmarks/datasets/__init__.py
- Create: benchmarks/datasets/models.py
- Create: benchmarks/datasets/templates/question.example.json
- Test: tests/unit/benchmarks/test_models.py
- Modify: pyproject.toml

**Interfaces:** Consumes the canonical Core domain models. Produces `TaskCategory`, private `AnnotatedQuestion`/gold records, public `RuntimeTask`, `FrozenEvidenceRecord`, public `DatasetManifest` and evaluator-only `PrivateDatasetManifest`; all later dataset, snapshot and evaluator tasks import these exact models.

- [ ] **Step 1: 写 schema 红测**

~~~python
from datetime import date

import pytest
from pydantic import ValidationError

from benchmarks.datasets.models import (
    AnnotatedQuestion,
    GoldEvidenceSpan,
    TaskCategory,
)


def test_annotated_question_rejects_cross_task_gold_link(question_factory):
    question = question_factory(
        task_id="dev-ts-01",
        gold_claim_links=[
            {
                "claim_id": "claim-other",
                "evidence_links": [{"evidence_id": "ev-1", "relation": "support"}],
            }
        ],
    )
    with pytest.raises(ValidationError, match="unknown claim_id"):
        AnnotatedQuestion.model_validate(question)


def test_gold_relevance_is_graded_zero_to_three(html_locator):
    with pytest.raises(ValidationError):
        GoldEvidenceSpan(
            evidence_id="ev-1",
            source_id="src-1",
            locator=html_locator,
            relevance_grade=4,
            excerpt_hash="a" * 64,
        )


def test_category_values_are_frozen():
    assert {item.value for item in TaskCategory} == {
        "technical_survey",
        "method_comparison",
        "multi_hop_history",
        "freshness",
        "bilingual",
        "source_conflict",
    }
~~~

- [ ] **Step 2: 运行红测并确认失败原因**

Run: uv run pytest tests/unit/benchmarks/test_models.py -q

Expected: FAIL with ModuleNotFoundError: No module named 'benchmarks.datasets.models'.

- [ ] **Step 3: 实现唯一的私有标注模型和公开运行模型**

~~~python
class TaskCategory(StrEnum):
    TECHNICAL_SURVEY = "technical_survey"
    METHOD_COMPARISON = "method_comparison"
    MULTI_HOP_HISTORY = "multi_hop_history"
    FRESHNESS = "freshness"
    BILINGUAL = "bilingual"
    SOURCE_CONFLICT = "source_conflict"


class GoldInformationNeed(BaseModel):
    need_id: str
    text: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]
    acceptable_claim_ids: list[str]


class GoldEvidenceSpan(BaseModel):
    evidence_id: str
    source_id: str
    locator: Locator
    relevance_grade: Literal[0, 1, 2, 3]
    excerpt_hash: str


class GoldClaimEvidenceLink(BaseModel):
    evidence_id: str
    relation: Literal["support", "contradict", "context"]


class GoldClaimLink(BaseModel):
    claim_id: str
    evidence_links: list[GoldClaimEvidenceLink]


class RubricDimension(BaseModel):
    rubric_id: str
    description: str
    weight: Annotated[float, Field(gt=0.0, le=1.0)]
    levels: dict[Literal[0, 1, 2, 3], str]


class FrozenEvidenceRecord(BaseModel):
    task_id: str
    evidence_id: str
    source_id: str
    source_family_id: str
    canonical_url: AnyHttpUrl
    title: str
    authors: tuple[str, ...]
    media_type: str
    raw_body_b64: str
    content_hash: str
    normalized_text: str
    parsed_content_hash: str
    locator_text: str
    locator: Locator
    excerpt: str
    excerpt_hash: str
    published_at: datetime | None = None
    unknown_published_at_reason: str | None = None
    retrieved_at: datetime
    language: str
    source_type: SourceType


class AnnotatedQuestion(BaseModel):
    task_id: str
    split: Literal["dev", "test"]
    category: TaskCategory
    request: ResearchRequest
    evaluation_cutoff: date
    information_needs: list[GoldInformationNeed]
    acceptable_claims: list[Claim]
    candidate_source_ids: list[str]
    gold_source_family_ids: list[str]
    snapshot_id: str
    corpus_version: str
    index_version: str
    gold_evidence_spans: list[GoldEvidenceSpan]
    gold_claim_links: list[GoldClaimLink]
    rubric: dict[str, RubricDimension]
    expected_stop_reason: StopReason
    expected_is_partial: bool
    created_at: date
    annotation_version: str


class RuntimeTask(BaseModel):
    task_id: str
    category: TaskCategory
    request: ResearchRequest
    evaluation_cutoff: date
    snapshot_id: str
    corpus_version: str
    index_version: str


class DatasetManifest(BaseModel):
    dataset_id: str
    version: str
    record_count: int
    split_counts: dict[Literal["dev", "test"], int]
    category_counts: dict[TaskCategory, int]
    public_runtime_files: list[str]
    private_manifest_sha256: str
    snapshot_collection_sha256: str
    cost_subset_sha256: str
    created_at: datetime


class PrivateDatasetManifest(BaseModel):
    dataset_id: str
    version: str
    record_count: int
    split_counts: dict[Literal["dev", "test"], int]
    category_counts: dict[TaskCategory, int]
    batch_sha256: dict[TaskCategory, str]
    snapshot_manifest_sha256: dict[str, str]
    public_runtime_files: list[str]
    private_test_runtime_files: list[str]
    main_test_task_ids: tuple[str, ...]
    stability_task_ids: tuple[str, ...]
    cost_subset_task_ids: tuple[str, ...]
    p0_task_ids: tuple[str, ...]
    oracle_task_ids: tuple[str, ...]
    subset_seed: int
    created_at: datetime
~~~

Add model validators that reject duplicate IDs, dangling claim/evidence links, an empty `evidence_links` list, missing information needs, non-64-character lowercase SHA-256 values, source IDs absent from candidate_source_ids, rubric weights whose sum is not 1.0, and inconsistent stop/partial labels (`SUFFICIENT` cannot be partial; BUDGET_EXHAUSTED must be partial). A claim may intentionally have both support and contradict links; validators preserve those distinct relations and reject duplicate `(evidence_id, relation)` pairs. `FrozenEvidenceRecord` is the only snapshot-builder input schema: decode raw_body_b64 and verify content_hash; hash full normalized_text for parsed_content_hash; verify the locator range against locator_text and its exact slice against excerpt/excerpt_hash. Require timezone-aware `retrieved_at`; exactly one of timezone-aware `published_at` or a non-empty `unknown_published_at_reason` must be present. `PrivateDatasetManifest` is the sole schema carrying sealed task IDs/subset membership and private file hashes; `DatasetManifest` exposes only counts, public paths, aggregate hashes and the private manifest hash. `source_type` is the shared domain SourceType. Use extra="forbid" on every benchmark model. `RuntimeTask` intentionally omits raw/gold/rubric/expected-stop fields.

Append benchmark dependencies without replacing Core dependencies:

    uv add rank-bm25 numpy scipy pandas pyyaml
    uv lock

- [ ] **Step 4: 添加可执行的完整 example**

question.example.json must contain one valid technical_survey record with a ResearchRequest, two information needs, two atomic claims, two source families, HTML/PDF locator examples, rubric dimensions, and valid synthetic hashes. It is documentation data only and must use fixture URLs under https://example.test/.

- [ ] **Step 5: 运行测试和静态检查**

Run: uv run pytest tests/unit/benchmarks/test_models.py -q

Expected: PASS.

Run: uv run ruff check benchmarks/datasets tests/unit/benchmarks/test_models.py

Expected: PASS.

- [ ] **Step 6: 提交**

    git add pyproject.toml uv.lock benchmarks/__init__.py benchmarks/datasets tests/unit/benchmarks/test_models.py
    git commit -m "feat: define frozen benchmark schemas"

### Task 2: 强制 Gold 与 Agent Runtime 隔离

**Files:**

- Create: benchmarks/datasets/isolation.py
- Create: benchmarks/datasets/README.md
- Create: benchmarks/processes/__init__.py
- Create: benchmarks/processes/agent.py
- Create: benchmarks/processes/evaluator.py
- Test: tests/unit/benchmarks/test_isolation.py
- Test: tests/integration/benchmarks/test_process_isolation.py
- Modify: .gitignore

**Interfaces:** Consumes `AnnotatedQuestion` and `RuntimeTask`. Produces `GoldIsolationGuard`, `GoldAccessViolation`, the real agent/evaluator process entrypoints and the environment/path contract used by every formal run.

- [ ] **Step 1: 写隔离红测**

~~~python
import json
from pathlib import Path

import pytest

from benchmarks.datasets.isolation import GoldAccessViolation, GoldIsolationGuard
from benchmarks.processes.evaluator import materialize_agent_runtime_task


def test_runtime_view_serializes_no_gold_fields(annotated_question):
    runtime = GoldIsolationGuard.runtime_view(annotated_question)
    payload = runtime.model_dump(mode="json")
    assert set(payload) == {
        "task_id",
        "category",
        "request",
        "evaluation_cutoff",
        "snapshot_id",
        "corpus_version",
        "index_version",
    }
    assert "gold" not in str(payload).casefold()
    assert "rubric" not in str(payload).casefold()


def test_agent_process_cannot_resolve_private_gold(tmp_path: Path):
    guard = GoldIsolationGuard(
        runtime_root=tmp_path / "runtime",
        snapshot_root=tmp_path / "snapshots",
        private_root=tmp_path / "private",
    )
    with pytest.raises(GoldAccessViolation, match="private benchmark path"):
        guard.assert_agent_readable(tmp_path / "private" / "gold" / "test.jsonl")


def test_runtime_manifest_rejects_private_absolute_path(tmp_path: Path):
    guard = GoldIsolationGuard(
        runtime_root=tmp_path / "runtime",
        snapshot_root=tmp_path / "snapshots",
        private_root=tmp_path / "private",
    )
    with pytest.raises(GoldAccessViolation):
        guard.validate_run_payload(
            {"snapshot_dir": str(tmp_path / "private" / "gold")}
        )


def test_agent_subprocess_refuses_gold_environment(tmp_path, subprocess_runner):
    result = subprocess_runner.agent_probe(
        env={
            "DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "DEEPRESEARCH_BENCHMARK_GOLD_ROOT": str(tmp_path / "private"),
        }
    )
    assert result.returncode == 3
    assert "GOLD_ROOT_FORBIDDEN" in result.stderr


def test_evaluator_materializes_one_public_task_outside_private_root(
    tmp_path, annotated_question
):
    private_root = tmp_path / "private"
    agent_input_root = tmp_path / "agent-inputs"
    runtime = GoldIsolationGuard.runtime_view(annotated_question)
    staged = materialize_agent_runtime_task(
        runtime,
        agent_input_root=agent_input_root,
        request_id="request-1",
        forbidden_private_root=private_root,
    )
    assert private_root.resolve() not in staged.resolve().parents
    assert set(json.loads(staged.read_text(encoding="utf-8"))) == {
        "task_id", "category", "request", "evaluation_cutoff",
        "snapshot_id", "corpus_version", "index_version",
    }
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/benchmarks/test_isolation.py tests/integration/benchmarks/test_process_isolation.py -q

Expected: FAIL because isolation and process entrypoints do not exist.

- [ ] **Step 3: 实现路径边界与最小公开视图**

~~~python
class GoldAccessViolation(RuntimeError):
    def __init__(self, message: str, *, code: str = "GOLD_ACCESS_VIOLATION") -> None:
        super().__init__(message)
        self.code = code


class GoldIsolationGuard:
    def __init__(
        self,
        runtime_root: Path,
        snapshot_root: Path,
        private_root: Path,
    ) -> None:
        self.runtime_root = runtime_root.resolve()
        self.snapshot_root = snapshot_root.resolve()
        self.private_root = private_root.resolve()

    @staticmethod
    def runtime_view(question: AnnotatedQuestion) -> RuntimeTask:
        return RuntimeTask(
            task_id=question.task_id,
            category=question.category,
            request=question.request,
            evaluation_cutoff=question.evaluation_cutoff,
            snapshot_id=question.snapshot_id,
            corpus_version=question.corpus_version,
            index_version=question.index_version,
        )

    def assert_agent_readable(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved == self.private_root or self.private_root in resolved.parents:
            raise GoldAccessViolation("private benchmark path is evaluator-only")
        allowed_roots = (self.runtime_root, self.snapshot_root)
        if not any(
            resolved == root or root in resolved.parents for root in allowed_roots
        ):
            raise GoldAccessViolation("path is outside runtime benchmark root")
        return resolved


class AgentRuntimeGuard:
    """Positive allow-list used inside the agent; it never receives private_root."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        snapshot_root: Path,
        run_root: Path,
    ) -> None:
        self.runtime_root = runtime_root.resolve()
        self.snapshot_root = snapshot_root.resolve()
        self.run_root = run_root.resolve()

    def resolve_runtime_task(self, path: Path) -> Path: ...
    def resolve_snapshot(self, path: Path) -> Path: ...
    def resolve_output(self, path: Path) -> Path: ...
    def validate_payload(self, payload: JsonValue) -> None: ...


def materialize_agent_runtime_task(
    task: RuntimeTask,
    *,
    agent_input_root: Path,
    request_id: str,
    forbidden_private_root: Path,
) -> Path: ...
~~~

`GoldIsolationGuard` and `AgentRuntimeGuard` live in `benchmarks/datasets/isolation.py`; `materialize_agent_runtime_task` lives in `benchmarks/processes/evaluator.py` and is evaluator-side only. `materialize_agent_runtime_task` first proves that `agent_input_root.resolve()` is neither the private root nor its descendant, serializes exactly one already-validated `RuntimeTask` to `<agent_input_root>/<request_id>.json` through a sibling staging file, fsyncs, atomically renames, reopens and byte-validates it. `AgentRuntimeGuard` is the only path guard instantiated inside the agent; it uses positive runtime/snapshot/output roots and is intentionally never told the private root path. Both guards recursively reject the names acceptable_claims, gold_evidence_spans, gold_claim_links and rubric; `assert_agent_environment` rejects the environment key DEEPRESEARCH_BENCHMARK_GOLD_ROOT before optional imports.

Implement two real module entrypoints without depending on future experiment modules. `python -m benchmarks.processes.agent --probe-runtime-task <json> --snapshot-dir <dir> --output <json>` calls `assert_agent_environment(os.environ)` before any optional import, exits 3 with code GOLD_ROOT_FORBIDDEN if the gold variable is present, resolves runtime/snapshot/output inputs through `AgentRuntimeGuard`, validates one RuntimeTask, verifies the snapshot manifest, and atomically emits only `{task_id, snapshot_id, probe_status:"ok"}`. `python -m benchmarks.processes.evaluator probe-agent ...` is the only entrypoint that may read DEEPRESEARCH_BENCHMARK_GOLD_ROOT; it reads a private test RuntimeTask itself, calls `materialize_agent_runtime_task` into a distinct agent-input root, then launches the probe with an explicit environment allow-list (that agent-input root, snapshot root and run root), deliberately removes GOLD_ROOT and verifies the output hash. Neither the request nor child environment contains the original private runtime path. The agent container/process receives no private mount. The integration test launches an actual Python subprocess twice: injected GOLD_ROOT must fail before optional imports; a sanitized staged fixture must complete while a sentinel under private_root remains unread/unmentioned. Task 15 extends these same entrypoints with typed run requests/receipts; Task 2's probe remains a permanent startup/isolation check.

- [ ] **Step 4: 忽略私有数据和大体积 snapshot**

Append exact entries:

    benchmarks/private/
    benchmarks/snapshots/**/documents.jsonl
    benchmarks/snapshots/**/search.jsonl
    benchmarks/snapshots/**/index.json
    benchmarks/snapshots/frozen_ai_cs_60/test-*/
    experiments/*/

The `experiments/*/` rule ignores every generated group child (requests, agent inputs, raw records, `group.json`, summaries and external outputs) while leaving the source modules `experiments/*.py` tracked. This is required so the second formal command and resume preflight still see a clean worktree. Do not ignore snapshot.json, manifest.sha256, fixtures, `docs/results.md` or `docs/assets/results`; the renderer is the only path that promotes hash-verified aggregate results into tracked public artifacts.

- [ ] **Step 5: 写数据访问说明**

benchmarks/datasets/README.md must document two separate process invocations:

    Agent (public dev): DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT=benchmarks/datasets/frozen_ai_cs_60/runtime/dev
    Agent (sealed test): DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT=experiments/<group>/agent-inputs
    Agent snapshots: DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT=benchmarks/snapshots/frozen_ai_cs_60
    Evaluator: DEEPRESEARCH_BENCHMARK_GOLD_ROOT=benchmarks/private/frozen_ai_cs_60

It must state that the evaluator alone reads `benchmarks/private/.../runtime/test`, stages one redacted task outside private root for each child launch, the API/UI/agent container never mounts the private root, and a sealed test question becomes publishable only after the first formal result is frozen.

- [ ] **Step 6: 验证并提交**

Run: uv run pytest tests/unit/benchmarks/test_isolation.py tests/integration/benchmarks/test_process_isolation.py -q

Expected: PASS.

Run:

    git check-ignore benchmarks/private/example.json
    git check-ignore experiments/example-group/group.json

Expected: prints both paths; `git check-ignore experiments/__init__.py` remains non-zero because source modules are tracked.

    git add .gitignore benchmarks/datasets/isolation.py benchmarks/datasets/README.md benchmarks/processes tests/unit/benchmarks/test_isolation.py tests/integration/benchmarks/test_process_isolation.py
    git commit -m "feat: isolate benchmark gold from agent runtime"

### Task 3: 实现确定性 FrozenCorpus 索引、SearchProvider 与 Fetch/Materialize 适配

**Files:**

- Create: src/deepresearch/providers/frozen_index.py
- Create: src/deepresearch/providers/frozen_search.py
- Create: benchmarks/snapshots/README.md
- Create: benchmarks/scripts/build_snapshot.py
- Create: tests/fixtures/frozen_corpus/task-fixture/snapshot.json
- Create: tests/fixtures/frozen_corpus/task-fixture/documents.jsonl
- Create: tests/fixtures/frozen_corpus/task-fixture/index.json
- Create: tests/fixtures/frozen_corpus/task-fixture/manifest.sha256
- Test: tests/unit/providers/test_frozen_index.py
- Test: tests/unit/providers/test_frozen_search.py
- Test: tests/contracts/test_frozen_search_contract.py
- Modify: pyproject.toml

**Interfaces:** Consumes `FrozenEvidenceRecord` JSONL and the shared `SearchProvider`/`Fetcher` plus canonical `RawDocument`/`ParsedDocument`/`SourceDocument`/`EvidenceSpan` types. Produces `FrozenCorpusManifest`, deterministic `FrozenBm25Index`, `FrozenCorpusSnapshot`, `FrozenCorpusSearchProvider`, `FrozenCorpusFetcher`, `FrozenCorpusMaterializer` and the `build_snapshot one|batch` CLI used by Tasks 5–15. Formal research-v1 composition must use all three frozen adapters; it may not pair frozen search with a Live fetcher.

- [ ] **Step 1: 写 tokenizer、排序和禁止 fallback 红测**

~~~python
from hashlib import sha256

import pytest

from deepresearch.providers.errors import ProviderError
from deepresearch.providers.frozen_index import deterministic_tokens
from deepresearch.providers.frozen_search import (
    FrozenCorpusFetcher,
    FrozenCorpusMaterializer,
    FrozenCorpusSearchProvider,
)


def test_tokenizer_handles_english_and_cjk_deterministically():
    assert deterministic_tokens("Multi-Agent 多模态 Agent") == (
        "multi",
        "agent",
        "多",
        "模",
        "态",
        "agent",
    )


@pytest.mark.asyncio
async def test_ties_use_evidence_then_source_id(provider, cancellation_token):
    hits = await provider.search(
        "equal score",
        limit=10,
        filters=None,
        deadline=100.0,
        cancellation_token=cancellation_token,
    )
    assert [
        (
            hit.provider_metadata["evidence_id"],
            hit.provider_metadata["source_id"],
        )
        for hit in hits
    ] == [("ev-1", "src-2"), ("ev-2", "src-1")]


def test_unknown_snapshot_never_calls_live(tmp_path):
    with pytest.raises(ProviderError) as error:
        FrozenCorpusSearchProvider.from_snapshot(
            tmp_path / "missing",
            task_id="test-ts-01",
        )
    assert error.value.code == "REPLAY_MISS"


@pytest.mark.asyncio
async def test_frozen_fetcher_returns_locked_raw_body(
    snapshot, cancellation_token
):
    raw = await FrozenCorpusFetcher(snapshot).fetch(
        "https://example.test/source-1",
        deadline=100.0,
        cancellation_token=cancellation_token,
    )
    assert sha256(raw.body_bytes).hexdigest() == snapshot.record("ev-1").content_hash


def test_materializer_preserves_snapshot_evidence_id_and_verifies_parse(
    snapshot, parsed_document
):
    result = FrozenCorpusMaterializer(snapshot).materialize(
        selected_evidence_ids=("ev-1",),
        parsed_documents={"src-1": parsed_document},
        information_need_ids=("need-runtime-1",),
    )
    assert result.evidence_spans[0].evidence_id == "ev-1"
    assert result.evidence_spans[0].information_need_ids == ("need-runtime-1",)
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/providers/test_frozen_index.py tests/unit/providers/test_frozen_search.py tests/contracts/test_frozen_search_contract.py -q

Expected: FAIL because frozen_index and frozen_search are missing.

- [ ] **Step 3: 实现 manifest、Unicode tokenizer 和 BM25 索引**

~~~python
class FrozenCorpusManifest(BaseModel):
    snapshot_id: str
    task_id: str
    corpus_version: str
    index_version: str
    document_count: int
    documents_sha256: str
    index_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenHit:
    score: float
    record: FrozenEvidenceRecord


def deterministic_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    pieces = re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", normalized)
    return tuple(pieces)


class FrozenBm25Index:
    @classmethod
    def build(
        cls,
        records: Sequence[FrozenEvidenceRecord],
        *,
        index_version: str,
    ) -> "FrozenBm25Index": ...

    def search(self, query: str, limit: int) -> tuple[FrozenHit, ...]: ...


class FrozenCorpusSnapshot:
    @classmethod
    def load(cls, snapshot_dir: Path, *, task_id: str) -> "FrozenCorpusSnapshot": ...

    @property
    def manifest(self) -> FrozenCorpusManifest: ...

    @property
    def index(self) -> FrozenBm25Index: ...

    def record(self, evidence_id: str) -> FrozenEvidenceRecord: ...

    def records_for_url(self, url: str) -> tuple[FrozenEvidenceRecord, ...]: ...
~~~

Use rank_bm25.BM25Okapi. Sort with key (-score, evidence_id, source_id). The serialized index contains only normalized records and version metadata; load must recompute and verify SHA-256 before accepting it. `FrozenCorpusSnapshot.load` is the sole loader used by search, fetch and materialization. It verifies manifest.sha256 plus documents/index hashes, unique evidence IDs, and that all records sharing a source ID or canonical URL agree on raw bytes/hash, media type, retrieval metadata and parsed-content hash.

- [ ] **Step 4: 实现 SearchProvider contract**

~~~python
class FrozenCorpusSearchProvider:
    provider_id = "frozen-corpus"

    def __init__(self, snapshot: FrozenCorpusSnapshot) -> None: ...

    @classmethod
    def from_snapshot(
        cls,
        snapshot_dir: Path,
        *,
        task_id: str,
    ) -> "FrozenCorpusSearchProvider": ...

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]: ...


class FrozenCorpusFetcher:
    provider_id = "frozen-corpus"

    def __init__(self, snapshot: FrozenCorpusSnapshot) -> None: ...

    async def fetch(
        self,
        url: str,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> RawDocument: ...


@dataclass(frozen=True, slots=True)
class FrozenMaterialization:
    source_documents: tuple[SourceDocument, ...]
    evidence_spans: tuple[EvidenceSpan, ...]


class FrozenCorpusMaterializer:
    def __init__(self, snapshot: FrozenCorpusSnapshot) -> None: ...

    def materialize(
        self,
        *,
        selected_evidence_ids: Sequence[str],
        parsed_documents: Mapping[str, ParsedDocument],
        information_need_ids: tuple[str, ...],
    ) -> FrozenMaterialization: ...
~~~

`from_snapshot` delegates to `FrozenCorpusSnapshot.load` and then the public constructor; formal composition loads once and passes that same object identity to SearchProvider, Fetcher and Materializer. Return SearchHit objects using the shared provider type and put only `evidence_id`, `source_id`, `source_family_id`, language and source_type in provider_metadata. Enforce cancellation before tokenization and before returning. Filters may narrow language, source_type and published_at, but unknown filter keys raise INVALID_REQUEST.

`FrozenCorpusFetcher` resolves only canonical URLs present in the loaded task snapshot, constructs the canonical Core `RawDocument` from locked raw bytes/media type/retrieved_at, checks cancellation before decode and return, and raises `REPLAY_MISS` for an unknown URL; it has no delegate or network client. The normal locked Core HTML/PDF Parser parses that RawDocument. `FrozenCorpusMaterializer` then verifies the parser's `parsed_content_hash`, resolves the record's HTML paragraph or PDF page/block container, checks locator/excerpt hashes, creates the canonical `SourceDocument` using the actual parser ID/version, and creates canonical `EvidenceSpan` objects with the **preassigned snapshot evidence IDs** plus the current runtime information-need IDs. This is the only formal StoreEvidence path: it prevents gold/ranking IDs from drifting away from the IDs produced by the agent while revealing no relevance grades, acceptable claims or rubric. Any parser/hash/locator disagreement raises `INVALID_SNAPSHOT` before ranking.

- [ ] **Step 5: 实现离线 snapshot 构建命令**

`benchmarks/scripts/build_snapshot.py` exposes two exact subcommands. `one` accepts `--task-id`, `--documents`, `--output`, `--corpus-version` and `--index-version`. `batch` accepts `--batch`, `--documents-root`, `--output-root`, `--corpus-version` and `--index-version`; it parses the batch as `AnnotatedQuestion`, iterates task IDs in lexical order, resolves exactly `<documents-root>/<task_id>.jsonl`, and invokes the same `one` implementation for each `<output-root>/<task_id>`. Both inputs are private JSONL of `FrozenEvidenceRecord` objects already collected through the normal fetch/parse pipeline. Each child build writes documents.jsonl, index.json, snapshot.json and manifest.sha256 to a sibling staging directory, verifies every locator/content hash, and atomically renames it. Either command must refuse an existing final output; a partial batch reports the exact failed task and leaves completed immutable children intact.

Exact single-task invocation:

    uv run python -m benchmarks.scripts.build_snapshot one --task-id dev-ts-01 --documents benchmarks/private/frozen_ai_cs_60/documents/dev-ts-01.jsonl --output benchmarks/snapshots/frozen_ai_cs_60/dev-ts-01 --corpus-version ai-cs-60-v1 --index-version bm25-mixed-v1

- [ ] **Step 6: 构建 fixture 并验证两次结果逐字节一致**

Run: uv run pytest tests/unit/providers/test_frozen_index.py tests/unit/providers/test_frozen_search.py tests/contracts/test_frozen_search_contract.py -q

Expected: PASS.

Run twice:

    uv run pytest tests/contracts/test_frozen_search_contract.py -q

Expected: identical PASS output, exact snapshot Evidence IDs after materialization, and no network access.

- [ ] **Step 7: 提交**

    git add pyproject.toml uv.lock src/deepresearch/providers/frozen_index.py src/deepresearch/providers/frozen_search.py benchmarks/snapshots benchmarks/scripts/build_snapshot.py tests/fixtures/frozen_corpus tests/unit/providers/test_frozen_index.py tests/unit/providers/test_frozen_search.py tests/contracts/test_frozen_search_contract.py
    git commit -m "feat: add deterministic frozen corpus search"

### Task 4: 实现 DatasetBuilder 和逐批 Validator

**Files:**

- Create: benchmarks/datasets/validator.py
- Create: benchmarks/datasets/builder.py
- Test: tests/unit/benchmarks/test_validator.py
- Test: tests/unit/benchmarks/test_builder.py
- Create: tests/fixtures/benchmark/minimal_private/
- Create: tests/fixtures/benchmark/minimal_runtime/

**Interfaces:** Consumes private batch JSONL and Task 3 snapshot manifests. Produces `BatchValidationReport`, `DatasetValidationReport`, `DatasetFinalizeResult`, `DatasetValidator`, `DatasetBuilder` and the exact validate/export/finalize CLI surface used by all curation tasks.

- [ ] **Step 1: 写完整性和不可覆盖红测**

~~~python
def test_batch_requires_exactly_ten_records(validator, batch_path):
    report = validator.validate_batch(
        batch_path,
        expected_category=TaskCategory.TECHNICAL_SURVEY,
        expected_count=10,
    )
    assert report.valid is False
    assert "expected 10 records" in " ".join(report.errors)


def test_dataset_requires_six_categories_and_thirty_thirty_split(
    validator, incomplete_manifest
):
    report = validator.validate_dataset(incomplete_manifest)
    assert report.valid is False
    assert report.category_counts != {
        category.value: 10 for category in TaskCategory
    }
    assert "expected split counts dev=30 test=30" in " ".join(report.errors)


def test_finalize_refuses_to_replace_frozen_version(builder, frozen_manifest):
    with pytest.raises(DatasetFrozenError, match="new semantic version"):
        builder.finalize(
            dataset_id="frozen_ai_cs_60",
            version=frozen_manifest.version,
            private_root=frozen_manifest.private_root,
            public_root=frozen_manifest.public_root,
            snapshot_root=frozen_manifest.snapshot_root,
            subset_seed=20260829,
        )
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/benchmarks/test_validator.py tests/unit/benchmarks/test_builder.py -q

Expected: FAIL because builder and validator are missing.

- [ ] **Step 3: 实现报告类型和 batch 校验**

~~~python
class BatchValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_id: str
    category: TaskCategory
    record_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sha256: str

    @property
    def valid(self) -> bool:
        return not self.errors


class DatasetValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    version: str
    record_count: int
    split_counts: dict[Literal["dev", "test"], int]
    category_counts: dict[TaskCategory, int]
    batch_reports: tuple[BatchValidationReport, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sha256: str

    @property
    def valid(self) -> bool:
        return not self.errors


class DatasetValidator:
    def validate_batch(
        self,
        path: Path,
        *,
        expected_category: TaskCategory,
        expected_count: int = 10,
    ) -> BatchValidationReport: ...

    def validate_dataset(
        self,
        private_manifest_path: Path,
    ) -> DatasetValidationReport: ...
~~~

The validator must check unique immutable task IDs, category/count, 30/30 split, six categories, information needs, importance range, graded relevance, locator/hash validity, claim links, source families, snapshot/corpus/index existence, exactly 20 test-only cost tasks, batch hashes, gold hash, and duplicate semantic questions.

validator.py also exposes a Typer module entry point with validate-batch and validate-dataset commands. Both print one JSON BatchValidationReport/DatasetValidationReport and exit 0 only when valid.

- [ ] **Step 4: 实现原子 finalize 与 runtime export**

~~~python
class DatasetBuilder:
    def add_batch(
        self,
        batch_id: str,
        records: Sequence[AnnotatedQuestion],
        *,
        expected_category: TaskCategory,
        expected_count: int = 10,
    ) -> BatchValidationReport: ...

    def export_runtime(
        self,
        records: Sequence[AnnotatedQuestion],
        *,
        output_path: Path,
        include_split: Literal["dev", "test"],
    ) -> Path: ...

    def finalize(
        self,
        *,
        dataset_id: str,
        version: str,
        private_root: Path,
        public_root: Path,
        snapshot_root: Path,
        subset_seed: int,
    ) -> "DatasetFinalizeResult": ...


class DatasetFinalizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    public_manifest_path: Path
    private_manifest_path: Path
    public_manifest: DatasetManifest
    private_manifest: PrivateDatasetManifest
~~~

`finalize` discovers exactly the six category batch files under `<private_root>/batches`, validates all 60 referenced child manifests under snapshot_root, freezes the explicit 30/30 IDs, deterministically selects the Task 11 stability/cost/P0/oracle subsets with the documented quotas and subset_seed, exports evaluator-only test runtime files, writes `PrivateDatasetManifest`, hashes it into the public `DatasetManifest`, then validates both. It writes every output to a sibling .staging path, fsyncs, and atomically renames only after all checks pass. If either target manifest/version exists, fail before writing. `export_runtime(..., output_path=...)` maps exactly to CLI `--output FILE`, must call GoldIsolationGuard.runtime_view and serialize one task per line; no directory/file dual interpretation is allowed.

builder.py exposes export-runtime and finalize Typer commands using these same library methods; CLI wrappers contain no separate validation logic.

- [ ] **Step 5: 运行测试**

Run: uv run pytest tests/unit/benchmarks/test_validator.py tests/unit/benchmarks/test_builder.py -q

Expected: PASS.

Run: uv run pyright benchmarks/datasets

Expected: 0 errors.

- [ ] **Step 6: 提交**

    git add benchmarks/datasets/validator.py benchmarks/datasets/builder.py tests/unit/benchmarks/test_validator.py tests/unit/benchmarks/test_builder.py tests/fixtures/benchmark
    git commit -m "feat: validate and freeze benchmark datasets"

### Task 5: 标注 technical_survey 批次

**Files:**

- Create private: benchmarks/private/frozen_ai_cs_60/batches/technical_survey.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/dev-ts-01.jsonl through dev-ts-05.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/test-ts-01.jsonl through test-ts-05.jsonl
- Create: benchmarks/datasets/frozen_ai_cs_60/runtime/dev/technical_survey.jsonl
- Generate: benchmarks/snapshots/frozen_ai_cs_60/dev-ts-01 through dev-ts-05
- Generate private: benchmarks/snapshots/frozen_ai_cs_60/test-ts-01 through test-ts-05

**Interfaces:** Consumes the annotation schema, validator, builder and snapshot batch command. Produces the immutable technical_survey private batch, all ten private snapshots and the five-record public dev runtime view.

- [ ] **Step 1: 先运行空批次验证并确认红灯**

Run:

    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/technical_survey.jsonl --category technical_survey --expected-count 10

Expected: exit 2 with file not found or expected 10 records.

- [ ] **Step 2: 完成十题私有标注**

Use IDs dev-ts-01 through dev-ts-05 and test-ts-01 through test-ts-05. The five public development themes are multimodal-agent technical routes, retrieval-augmented generation design patterns, parameter-efficient fine-tuning families, long-context model techniques, and speculative decoding families. Keep all five sealed-test prompts private.

For every record, annotate at least four atomic information needs, five acceptable claims, eight candidate sources, three independent source families, six gold evidence spans, claim-to-evidence links, and every rubric dimension. At least two questions must require both paper and official-document sources.

- [ ] **Step 3: 构建并校验每题 frozen snapshot**

Run the exact batch command; it builds dev-ts-01…05 and test-ts-01…05 from the IDs in the validated batch:

    uv run python -m benchmarks.scripts.build_snapshot batch --batch benchmarks/private/frozen_ai_cs_60/batches/technical_survey.jsonl --documents-root benchmarks/private/frozen_ai_cs_60/documents --output-root benchmarks/snapshots/frozen_ai_cs_60 --corpus-version ai-cs-60-v1 --index-version bm25-mixed-v1

Expected: ten child snapshot.json/manifest.sha256 pairs are created; rerunning against any existing child exits 4 before modifying it.

- [ ] **Step 4: 绿测批次并只导出 dev**

Run:

    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/technical_survey.jsonl --category technical_survey --expected-count 10
    uv run python -m benchmarks.datasets.builder export-runtime --batch benchmarks/private/frozen_ai_cs_60/batches/technical_survey.jsonl --split dev --output benchmarks/datasets/frozen_ai_cs_60/runtime/dev/technical_survey.jsonl

Expected: valid=true, five dev records exported, zero test records or gold fields in the public file.

- [ ] **Step 5: 提交公开开发视图和可复核 manifest**

    git add benchmarks/datasets/frozen_ai_cs_60/runtime/dev/technical_survey.jsonl benchmarks/snapshots/frozen_ai_cs_60/dev-ts-*/snapshot.json benchmarks/snapshots/frozen_ai_cs_60/dev-ts-*/manifest.sha256
    git commit -m "data: add technical survey development batch"

### Task 6: 标注 method_comparison 批次

**Files:**

- Create private: benchmarks/private/frozen_ai_cs_60/batches/method_comparison.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/dev-mc-01.jsonl through dev-mc-05.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/test-mc-01.jsonl through test-mc-05.jsonl
- Create: benchmarks/datasets/frozen_ai_cs_60/runtime/dev/method_comparison.jsonl
- Generate: benchmarks/snapshots/frozen_ai_cs_60/dev-mc-01 through dev-mc-05
- Generate private: benchmarks/snapshots/frozen_ai_cs_60/test-mc-01 through test-mc-05

**Interfaces:** Consumes the same curation toolchain. Produces the immutable method_comparison private batch, all ten private snapshots and the five-record public dev runtime view.

- [ ] **Step 1: 运行缺失批次验证**

Run:

    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/method_comparison.jsonl --category method_comparison --expected-count 10

Expected: non-zero exit.

- [ ] **Step 2: 完成十题私有标注**

Use IDs dev-mc-01 through dev-mc-05 and test-mc-01 through test-mc-05. Public development themes: LoRA versus adapter tuning, dense versus sparse retrieval, HNSW versus IVF-PQ, DPO versus PPO-style preference optimization, and dense versus mixture-of-experts architectures. Each prompt must name comparison dimensions and forbid a context-free winner.

Each record needs at least three comparable methods, four decision dimensions, two independent primary sources, one limitations claim, and one conflicting or non-comparable result explicitly marked context.

- [ ] **Step 3: 构建十个 snapshot 并运行 hash 校验**

Run:

    uv run python -m benchmarks.scripts.build_snapshot batch --batch benchmarks/private/frozen_ai_cs_60/batches/method_comparison.jsonl --documents-root benchmarks/private/frozen_ai_cs_60/documents --output-root benchmarks/snapshots/frozen_ai_cs_60 --corpus-version ai-cs-60-v1 --index-version bm25-mixed-v1

    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/method_comparison.jsonl --category method_comparison --expected-count 10

Expected: PASS with ten valid manifests; no duplicate source family counted twice because of mirrors or reposts.

- [ ] **Step 4: 导出五题 dev 并证明 test 未泄漏**

Run:

    uv run python -m benchmarks.datasets.builder export-runtime --batch benchmarks/private/frozen_ai_cs_60/batches/method_comparison.jsonl --split dev --output benchmarks/datasets/frozen_ai_cs_60/runtime/dev/method_comparison.jsonl

    rg '"split"\s*:\s*"test"|gold_|rubric|acceptable_claim' benchmarks/datasets/frozen_ai_cs_60/runtime/dev/method_comparison.jsonl

Expected: no matches.

- [ ] **Step 5: 提交**

    git add benchmarks/datasets/frozen_ai_cs_60/runtime/dev/method_comparison.jsonl benchmarks/snapshots/frozen_ai_cs_60/dev-mc-*/snapshot.json benchmarks/snapshots/frozen_ai_cs_60/dev-mc-*/manifest.sha256
    git commit -m "data: add method comparison development batch"

### Task 7: 标注 multi_hop_history 批次

**Files:**

- Create private: benchmarks/private/frozen_ai_cs_60/batches/multi_hop_history.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/dev-mh-01.jsonl through dev-mh-05.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/test-mh-01.jsonl through test-mh-05.jsonl
- Create: benchmarks/datasets/frozen_ai_cs_60/runtime/dev/multi_hop_history.jsonl
- Generate: benchmarks/snapshots/frozen_ai_cs_60/dev-mh-01 through dev-mh-05
- Generate private: benchmarks/snapshots/frozen_ai_cs_60/test-mh-01 through test-mh-05

**Interfaces:** Consumes the same curation toolchain. Produces the immutable multi_hop_history private batch, all ten private snapshots and the five-record public dev runtime view with chronological metadata.

- [ ] **Step 1: 运行红灯验证**

Run validator against the absent multi_hop_history batch.

Expected: non-zero exit.

- [ ] **Step 2: 完成十题私有标注**

Use IDs dev-mh-01 through dev-mh-05 and test-mh-01 through test-mh-05. Public development themes: evolution from sequence-to-sequence attention to Transformers, development of instruction tuning, evolution of RLHF pipelines, progress of retrieval-augmented generation, and the lineage of multimodal foundation models.

Every question must require at least three chronological hops, four distinct publication dates, a causal-link rubric that separates documented influence from temporal succession, and at least one primary source per hop.

- [ ] **Step 3: 构建 snapshot、验证时间字段与 link**

Run:

    uv run python -m benchmarks.scripts.build_snapshot batch --batch benchmarks/private/frozen_ai_cs_60/batches/multi_hop_history.jsonl --documents-root benchmarks/private/frozen_ai_cs_60/documents --output-root benchmarks/snapshots/frozen_ai_cs_60 --corpus-version ai-cs-60-v1 --index-version bm25-mixed-v1
    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/multi_hop_history.jsonl --category multi_hop_history --expected-count 10

Expected: PASS only when `published_at` exists for every source used by a chronological claim and every gold claim link stays inside its task.

- [ ] **Step 4: 导出 dev 并运行隔离绿测**

    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/multi_hop_history.jsonl --category multi_hop_history --expected-count 10
    uv run python -m benchmarks.datasets.builder export-runtime --batch benchmarks/private/frozen_ai_cs_60/batches/multi_hop_history.jsonl --split dev --output benchmarks/datasets/frozen_ai_cs_60/runtime/dev/multi_hop_history.jsonl
    uv run pytest tests/unit/benchmarks/test_isolation.py -q

Expected: PASS with five dev records, valid chronological metadata and no test/gold fields.

- [ ] **Step 5: 提交**

    git add benchmarks/datasets/frozen_ai_cs_60/runtime/dev/multi_hop_history.jsonl benchmarks/snapshots/frozen_ai_cs_60/dev-mh-*/snapshot.json benchmarks/snapshots/frozen_ai_cs_60/dev-mh-*/manifest.sha256
    git commit -m "data: add multi-hop history development batch"

### Task 8: 标注 freshness 批次

**Files:**

- Create private: benchmarks/private/frozen_ai_cs_60/batches/freshness.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/dev-fr-01.jsonl through dev-fr-05.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/test-fr-01.jsonl through test-fr-05.jsonl
- Create: benchmarks/datasets/frozen_ai_cs_60/runtime/dev/freshness.jsonl
- Generate: benchmarks/snapshots/frozen_ai_cs_60/dev-fr-01 through dev-fr-05
- Generate private: benchmarks/snapshots/frozen_ai_cs_60/test-fr-01 through test-fr-05

**Interfaces:** Consumes the same curation toolchain. Produces the immutable freshness private batch, all ten private snapshots and the five-record public dev runtime view with frozen evaluation cutoffs.

- [ ] **Step 1: 运行红灯验证**

Run validator against the absent freshness batch.

Expected: non-zero exit.

- [ ] **Step 2: 完成十题私有标注并冻结 as_of**

Use IDs dev-fr-01 through dev-fr-05 and test-fr-01 through test-fr-05. Every record must set an explicit evaluation_cutoff date, repeat that cutoff in the natural-language ResearchRequest, and reject evidence published after it. The five public development tasks cover model-release comparison, library/API migration, benchmark leaderboard change, open-source license change, and agent-framework feature evolution.

Each source requires canonical `retrieved_at` and either `published_at` or `unknown_published_at_reason`. At least one stale but otherwise relevant evidence span per task must have relevance grade 0 or 1 so freshness is measurable.

- [ ] **Step 3: 构建 snapshot 并验证时间泄漏**

Run:

    uv run python -m benchmarks.scripts.build_snapshot batch --batch benchmarks/private/frozen_ai_cs_60/batches/freshness.jsonl --documents-root benchmarks/private/frozen_ai_cs_60/documents --output-root benchmarks/snapshots/frozen_ai_cs_60 --corpus-version ai-cs-60-v1 --index-version bm25-mixed-v1

Then the Step 4 validator must reject any gold direct-support span whose `published_at` is later than `evaluation_cutoff`.

- [ ] **Step 4: 导出 dev 并运行 freshness 绿测**

    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/freshness.jsonl --category freshness --expected-count 10
    uv run python -m benchmarks.datasets.builder export-runtime --batch benchmarks/private/frozen_ai_cs_60/batches/freshness.jsonl --split dev --output benchmarks/datasets/frozen_ai_cs_60/runtime/dev/freshness.jsonl

Expected: PASS with five dev records and zero direct-support spans newer than evaluation_cutoff.

- [ ] **Step 5: 提交**

    git add benchmarks/datasets/frozen_ai_cs_60/runtime/dev/freshness.jsonl benchmarks/snapshots/frozen_ai_cs_60/dev-fr-*/snapshot.json benchmarks/snapshots/frozen_ai_cs_60/dev-fr-*/manifest.sha256
    git commit -m "data: add freshness development batch"

### Task 9: 标注 bilingual 批次

**Files:**

- Create private: benchmarks/private/frozen_ai_cs_60/batches/bilingual.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/dev-bi-01.jsonl through dev-bi-05.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/test-bi-01.jsonl through test-bi-05.jsonl
- Create: benchmarks/datasets/frozen_ai_cs_60/runtime/dev/bilingual.jsonl
- Generate: benchmarks/snapshots/frozen_ai_cs_60/dev-bi-01 through dev-bi-05
- Generate private: benchmarks/snapshots/frozen_ai_cs_60/test-bi-01 through test-bi-05

**Interfaces:** Consumes the same curation toolchain. Produces the immutable bilingual private batch, all ten private snapshots and the five-record public dev runtime view with language-balanced evidence.

- [ ] **Step 1: 运行红灯验证**

Run validator against the absent bilingual batch.

Expected: non-zero exit.

- [ ] **Step 2: 完成十题中英混合标注**

Use IDs dev-bi-01 through dev-bi-05 and test-bi-01 through test-bi-05. Each task must require at least two Chinese and two English sources, while the requested final answer language alternates across the batch. Public themes cover Chinese model technical reports, bilingual API documentation, Chinese AI policy implementation versus English analysis, multilingual embedding evaluation, and Chinese/English benchmark descriptions.

Gold claims must be language-neutral semantic units. Equivalent translations share claim IDs but retain distinct evidence and source-family IDs.

- [ ] **Step 3: 构建 snapshot 并验证多语 tokenizer**

Run:

    uv run python -m benchmarks.scripts.build_snapshot batch --batch benchmarks/private/frozen_ai_cs_60/batches/bilingual.jsonl --documents-root benchmarks/private/frozen_ai_cs_60/documents --output-root benchmarks/snapshots/frozen_ai_cs_60 --corpus-version ai-cs-60-v1 --index-version bm25-mixed-v1
    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/bilingual.jsonl --category bilingual --expected-count 10
    uv run pytest tests/unit/providers/test_frozen_index.py -q

Expected: PASS, including mixed CJK/Latin tokenization.

- [ ] **Step 4: 导出 dev 并提交**

    uv run python -m benchmarks.datasets.builder export-runtime --batch benchmarks/private/frozen_ai_cs_60/batches/bilingual.jsonl --split dev --output benchmarks/datasets/frozen_ai_cs_60/runtime/dev/bilingual.jsonl
    git add benchmarks/datasets/frozen_ai_cs_60/runtime/dev/bilingual.jsonl benchmarks/snapshots/frozen_ai_cs_60/dev-bi-*/snapshot.json benchmarks/snapshots/frozen_ai_cs_60/dev-bi-*/manifest.sha256
    git commit -m "data: add bilingual development batch"

### Task 10: 标注 source_conflict 批次

**Files:**

- Create private: benchmarks/private/frozen_ai_cs_60/batches/source_conflict.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/dev-sc-01.jsonl through dev-sc-05.jsonl
- Create private: benchmarks/private/frozen_ai_cs_60/documents/test-sc-01.jsonl through test-sc-05.jsonl
- Create: benchmarks/datasets/frozen_ai_cs_60/runtime/dev/source_conflict.jsonl
- Generate: benchmarks/snapshots/frozen_ai_cs_60/dev-sc-01 through dev-sc-05
- Generate private: benchmarks/snapshots/frozen_ai_cs_60/test-sc-01 through test-sc-05

**Interfaces:** Consumes the same curation toolchain and conflict-ranker fixture. Produces the immutable source_conflict private batch, all ten private snapshots and the five-record public dev runtime view preserving both sides of conflicts.

- [ ] **Step 1: 运行红灯验证**

Run validator against the absent source_conflict batch.

Expected: non-zero exit.

- [ ] **Step 2: 完成十题冲突证据标注**

Use IDs dev-sc-01 through dev-sc-05 and test-sc-01 through test-sc-05. Public themes cover conflicting benchmark claims, paper-versus-repository implementation differences, vendor documentation versus measured behavior, differing dataset-license interpretations, and contradictory scaling conclusions.

Every task must include at least one support and one contradict link for the same atomic claim, distinct source families, a resolution information need, and a rubric that rewards preserving unresolved uncertainty.

- [ ] **Step 3: 构建 snapshot 并验证冲突不被去重**

Run:

    uv run python -m benchmarks.scripts.build_snapshot batch --batch benchmarks/private/frozen_ai_cs_60/batches/source_conflict.jsonl --documents-root benchmarks/private/frozen_ai_cs_60/documents --output-root benchmarks/snapshots/frozen_ai_cs_60 --corpus-version ai-cs-60-v1 --index-version bm25-mixed-v1
    uv run pytest tests/integration/replay/test_conflict_research.py -q

Expected: all ten snapshots verify and both independent sides remain candidates in the Planner/Evidence conflict fixture.

- [ ] **Step 4: 导出 dev 并运行冲突绿测**

    uv run python -m benchmarks.datasets.validator validate-batch --batch benchmarks/private/frozen_ai_cs_60/batches/source_conflict.jsonl --category source_conflict --expected-count 10
    uv run python -m benchmarks.datasets.builder export-runtime --batch benchmarks/private/frozen_ai_cs_60/batches/source_conflict.jsonl --split dev --output benchmarks/datasets/frozen_ai_cs_60/runtime/dev/source_conflict.jsonl

Expected: PASS with five dev records and both support/contradict source families preserved.

- [ ] **Step 5: 提交**

    git add benchmarks/datasets/frozen_ai_cs_60/runtime/dev/source_conflict.jsonl benchmarks/snapshots/frozen_ai_cs_60/dev-sc-*/snapshot.json benchmarks/snapshots/frozen_ai_cs_60/dev-sc-*/manifest.sha256
    git commit -m "data: add source conflict development batch"

### Task 11: 冻结 Frozen AI/CS Research 60 v1

**Files:**

- Create: benchmarks/datasets/frozen_ai_cs_60/public_manifest.json
- Create private: benchmarks/private/frozen_ai_cs_60/private_manifest.json
- Generate private: benchmarks/private/frozen_ai_cs_60/runtime/test/*.jsonl
- Test: tests/integration/benchmarks/test_formal_protocols.py

**Interfaces:** Consumes the six validated ten-task batches and sixty verified snapshots. Produces immutable public/private v1 manifests, fixed dev/test splits, preregistered task subsets and evaluator-only test runtime views.

- [ ] **Step 1: 写全数据协议红测**

~~~python
def test_formal_dataset_has_six_balanced_categories(dataset_report):
    assert dataset_report.record_count == 60
    assert dataset_report.split_counts == {"dev": 30, "test": 30}
    assert dataset_report.category_counts == {
        category.value: 10 for category in TaskCategory
    }


def test_locked_subsets_are_test_only_and_stratified(private_manifest):
    assert len(private_manifest.stability_task_ids) == 20
    assert len(private_manifest.cost_subset_task_ids) == 20
    assert len(private_manifest.p0_task_ids) == 10
    assert len(private_manifest.oracle_task_ids) == 10
    assert (
        private_manifest.stability_task_ids
        == private_manifest.cost_subset_task_ids
    )
    assert all(task_id.startswith("test-") for task_id in {
        *private_manifest.stability_task_ids,
        *private_manifest.p0_task_ids,
        *private_manifest.oracle_task_ids,
    })
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/integration/benchmarks/test_formal_protocols.py -q

Expected: FAIL because no finalized manifest exists.

- [ ] **Step 3: 锁定 split 和预注册子集**

Keep the explicit dev/test IDs from Tasks 5–10. Select one shared stability/cost subset with category quotas 4,4,3,3,3,3 using seed 20260829. Select P0 and oracle subsets independently with quotas 2,2,2,2,1,1 using the same seed and persist their exact IDs in the private manifest. Never redraw a subset after seeing results.

- [ ] **Step 4: Finalize**

Run:

    uv run python -m benchmarks.datasets.builder finalize --dataset-id frozen_ai_cs_60 --version 1.0.0 --private-root benchmarks/private/frozen_ai_cs_60 --public-root benchmarks/datasets/frozen_ai_cs_60 --snapshot-root benchmarks/snapshots/frozen_ai_cs_60 --subset-seed 20260829

Expected: public_manifest.json contains counts, versions and cryptographic hashes but no test prompts, rubric, acceptable claims, evidence or private filesystem paths.

- [ ] **Step 5: 全面验证、只导出私有 test runtime**

Run:

    uv run python -m benchmarks.datasets.validator validate-dataset --manifest benchmarks/private/frozen_ai_cs_60/private_manifest.json
    uv run pytest tests/unit/benchmarks tests/integration/benchmarks/test_formal_protocols.py -q

Expected: valid=true and all tests pass.

- [ ] **Step 6: 提交公开 manifest**

    git add benchmarks/datasets/frozen_ai_cs_60/public_manifest.json tests/integration/benchmarks/test_formal_protocols.py
    git commit -m "data: freeze Frozen AI CS Research 60 v1 manifest"

### Task 12: 实现分项 Metrics

**Files:**

- Create: benchmarks/evaluators/__init__.py
- Create: benchmarks/evaluators/metrics.py
- Test: tests/unit/benchmarks/test_metrics.py

**Interfaces:** Consumes public agent artifacts plus evaluator-side gold. Produces `MetricNote`, `MetricValue`, ranking/claim/coverage metrics and `summarize_efficiency(RunManifest, RunEvent)` without exposing private annotations.

- [ ] **Step 1: 写 gold 排名、claim 和效率指标红测**

~~~python
import pytest

from benchmarks.evaluators.metrics import (
    citation_support_precision,
    information_completeness,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    unsupported_claim_rate,
)


def test_rank_metrics_use_grade_two_as_relevant():
    grades = {"e1": 3, "e2": 1, "e3": 2}
    ranked = ["e2", "e1", "e3"]
    assert recall_at_k(ranked, grades, k=2) == pytest.approx(0.5)
    assert mean_reciprocal_rank(ranked, grades) == pytest.approx(0.5)
    assert ndcg_at_k(ranked, grades, k=3) == pytest.approx(
        expected_ndcg([1, 3, 2], ideal=[3, 2, 1])
    )


def test_unsupported_claim_rate_excludes_non_factual_sentences():
    claims = [
        evaluated_claim("c1", factual=True, support=True),
        evaluated_claim("c2", factual=True, support=False),
        evaluated_claim("c3", factual=False, support=False),
    ]
    assert unsupported_claim_rate(claims) == pytest.approx(0.5)


def test_zero_citations_is_zero_precision_not_perfect():
    assert citation_support_precision([]) == 0.0


def test_completeness_uses_information_need_importance():
    needs = [
        evaluated_need("n1", importance=0.8, satisfied=True),
        evaluated_need("n2", importance=0.2, satisfied=False),
    ]
    assert information_completeness(needs) == pytest.approx(0.8)
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/benchmarks/test_metrics.py -q

Expected: FAIL because benchmarks.evaluators.metrics is missing.

- [ ] **Step 3: 实现确定公式**

~~~python
def recall_at_k(
    ranked_ids: Sequence[str],
    graded_relevance: Mapping[str, int],
    k: int,
) -> float:
    relevant = {item for item, grade in graded_relevance.items() if grade >= 2}
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mean_reciprocal_rank(
    ranked_ids: Sequence[str],
    graded_relevance: Mapping[str, int],
) -> float:
    for rank, item in enumerate(ranked_ids, start=1):
        if graded_relevance.get(item, 0) >= 2:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    graded_relevance: Mapping[str, int],
    k: int,
) -> float:
    observed = [graded_relevance.get(item, 0) for item in ranked_ids[:k]]
    ideal = sorted(graded_relevance.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return 0.0 if denominator == 0.0 else dcg(observed) / denominator


class MetricNote(BaseModel):
    code: str
    fields: dict[str, JsonValue] = Field(default_factory=dict)


class MetricValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    value: float
    numerator: float
    denominator: float
    version: str
    notes: tuple[MetricNote, ...] = ()
~~~

Use gain 2**grade - 1 and discount log2(rank + 1). Define:

- claim_coverage_at_k: importance-weighted gold claims having at least one acceptable evidence ID in top k.
- independent_source_coverage: satisfied information needs backed by the required count of distinct source_family_id values.
- citation_support_precision: cited factual claim-evidence links judged support divided by all cited factual claim-evidence links; insufficient, context, contradiction and unknown count as unsupported.
- citation_coverage: factual claims requiring evidence with at least one verified support citation divided by all factual claims requiring evidence.
- unsupported_claim_rate: factual claims with no verified support link divided by all factual claims.
- information_completeness: importance-weighted gold information needs satisfied by an acceptable atomic claim with verified evidence.
- query_redundancy: queries removed by exact/semantic dedupe plus executed queries semantically duplicating earlier queries, divided by all proposed queries.
- execution_adherence: completed planned information needs divided by scheduled non-blocked needs.
- stop_calibration: categorical exact match against evaluator stop label plus separate partial-result flag accuracy.
- marginal_utility_per_search: positive change in information completeness divided by executed searches.
- backtracking_gain: completeness gained after the first directed replan.

Public scalar helpers above remain simple floats for unit-level formula tests; evaluator entrypoints wrap each result in the exact `MetricValue` model. A model validator rejects non-finite fields, negative denominators and a zero denominator without a structured `ZERO_DENOMINATOR` note. `stop_calibration` reads only the evaluator-side AnnotatedQuestion.expected_stop_reason/expected_is_partial fields added in Task 1; those labels never enter RuntimeTask. Never silently return NaN.

- [ ] **Step 4: 加入效率汇总**

~~~python
class EfficiencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    search_calls: int
    fetch_calls: int
    prompt_tokens_by_node: dict[str, int]
    completion_tokens_by_node: dict[str, int]
    total_tokens: int
    p50_tool_latency_ms: float
    p95_tool_latency_ms: float
    wall_time_ms: float
    retries: int
    failures: int
    cost_usd: Decimal | None


def summarize_efficiency(
    manifest: RunManifest,
    events: Sequence[RunEvent],
) -> EfficiencySummary: ...
~~~

Implement the fields from immutable manifest call/node records and the contiguous event stream:

    search_calls
    fetch_calls
    prompt_tokens_by_node
    completion_tokens_by_node
    total_tokens
    p50_tool_latency_ms
    p95_tool_latency_ms
    wall_time_ms
    retries
    failures
    cost_usd

Quality-per-resource ratios must be emitted as separate derived fields and preserve the underlying numerator and denominator.

- [ ] **Step 5: 验证并提交**

Run: uv run pytest tests/unit/benchmarks/test_metrics.py -q

Expected: PASS.

Run: uv run ruff check benchmarks/evaluators/metrics.py tests/unit/benchmarks/test_metrics.py

Expected: PASS.

    git add benchmarks/evaluators/__init__.py benchmarks/evaluators/metrics.py tests/unit/benchmarks/test_metrics.py
    git commit -m "feat: add benchmark quality and efficiency metrics"

### Task 13: 实现 seed 聚合、分层 Bootstrap 与 Pareto 判定

**Files:**

- Create: benchmarks/evaluators/statistics.py
- Create: benchmarks/evaluators/pareto.py
- Test: tests/unit/benchmarks/test_statistics.py
- Test: tests/unit/benchmarks/test_pareto.py

**Interfaces:** Consumes Task 12 `MetricValue` records. Produces `SeedRunRecord`, task-level `TaskVariantAggregate`, paired stratified confidence intervals and two-plane Pareto decisions consumed by summarization.

- [ ] **Step 1: 写统计协议红测**

~~~python
def test_seed_runs_are_averaged_before_bootstrap():
    records = [
        seed_record("t1", "A", 1, 0.4, budget_preset="medium"),
        seed_record("t1", "A", 2, 0.8, budget_preset="medium"),
        seed_record("t1", "D", 1, 0.7, budget_preset="medium"),
        seed_record("t1", "D", 2, 0.9, budget_preset="medium"),
        seed_record("t2", "A", 1, 0.2, budget_preset="medium"),
        seed_record("t2", "D", 1, 0.3, budget_preset="medium"),
    ]
    aggregates = aggregate_seeds(records, metric_name="information_completeness")
    assert len(aggregates) == 4
    assert value_for(aggregates, "t1", "A") == pytest.approx(0.6)


def test_paired_bootstrap_rejects_unpaired_tasks():
    with pytest.raises(ValueError, match="paired task IDs"):
        paired_stratified_bootstrap(
            left=[aggregate("t1", "A", 0.5, budget_preset="medium")],
            right=[aggregate("t2", "D", 0.6, budget_preset="medium")],
            categories={"t1": "technical_survey", "t2": "method_comparison"},
            n_resamples=10_000,
            seed=20260829,
        )


def test_equal_quality_and_cost_is_not_strict_pareto():
    result = pareto_dominance(
        baseline=quality_cost(quality=0.8, cost=1.0),
        candidate=quality_cost(quality=0.8, cost=1.0),
    )
    assert result.dominates is False
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/benchmarks/test_statistics.py tests/unit/benchmarks/test_pareto.py -q

Expected: FAIL because statistics and pareto are missing.

- [ ] **Step 3: 实现 task-level 聚合和 paired bootstrap**

~~~python
class SeedRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    category: TaskCategory
    variant: str
    budget_preset: Literal["low", "medium", "high"]
    seed: int | None = None
    repeat_id: int | None = None
    metrics: dict[str, MetricValue]


class TaskVariantAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    category: TaskCategory
    variant: str
    budget_preset: Literal["low", "medium", "high"]
    metric_name: str
    mean: float
    observed_values: tuple[float, ...]
    run_count: Annotated[int, Field(ge=1)]


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_tasks: int
    n_resamples: int
    seed: int


def aggregate_seeds(
    records: Sequence[SeedRunRecord],
    *,
    metric_name: str,
) -> tuple[TaskVariantAggregate, ...]: ...


def paired_stratified_bootstrap(
    left: Sequence[TaskVariantAggregate],
    right: Sequence[TaskVariantAggregate],
    *,
    categories: Mapping[str, str],
    n_resamples: int = 10_000,
    seed: int,
) -> ConfidenceInterval: ...
~~~

`SeedRunRecord` and `TaskVariantAggregate` live in statistics.py and consume Task 12 MetricValue; they are the only accepted bootstrap inputs. Exactly one of seed/repeat_id must be set. `aggregate_seeds` rejects missing metric names or duplicate `(task, variant, budget_preset, seed/repeat)` keys and averages within `(task, variant,budget_preset)` before any resampling. Primary comparisons filter the sealed primary budget; cost-sensitivity comparisons explicitly select one budget at a time and never pool presets. Validate identical `(task_id, budget_preset)` sets. For each resample, draw task IDs with replacement independently inside each of the six category strata, preserve the original stratum sizes, concatenate paired differences, and take their mean. Compute percentile 2.5% and 97.5% bounds with NumPy's linear quantile method.

- [ ] **Step 4: 实现预注册判断**

Ranker primary result is task-level R2 minus R1 citation_support_precision on the fixed pool. Planner non-inferiority passes only when the lower CI bound for P2 minus P1 information_completeness is greater than -0.03; only then report the secondary reductions in search_calls and query_redundancy as confirmatory.

Implement Pareto on two separate planes:

    citation_support_precision versus cost_usd
    information_completeness versus search_calls

Candidate D dominates A at the sample-mean level only if quality_D >= quality_A, cost_D <= cost_A, and at least one inequality is strict. Separately bootstrap the proportion of task resamples satisfying the same condition.

- [ ] **Step 5: 运行确定性测试**

Run: uv run pytest tests/unit/benchmarks/test_statistics.py tests/unit/benchmarks/test_pareto.py -q

Expected: PASS.

Run the statistics tests twice with PYTHONHASHSEED=0 and confirm identical serialized CI fixtures.

- [ ] **Step 6: 提交**

    git add benchmarks/evaluators/statistics.py benchmarks/evaluators/pareto.py tests/unit/benchmarks/test_statistics.py tests/unit/benchmarks/test_pareto.py
    git commit -m "feat: add paired statistics and pareto analysis"

### Task 14: 锁定 FormalExperimentConfig 和实验变体工厂

**Files:**

- Create: experiments/__init__.py
- Create: experiments/models.py
- Create: experiments/config.py
- Create: experiments/factories.py
- Create: benchmarks/evaluators/oracle.py
- Create: benchmarks/configs/formal.template.yaml
- Create: benchmarks/scripts/lock_model.py
- Create: benchmarks/scripts/capture_inference_environment.py
- Test: tests/unit/experiments/test_config.py
- Test: tests/unit/experiments/test_factories.py
- Test: tests/unit/benchmarks/test_oracle.py

**Interfaces:** Consumes frozen dataset/manifests, Core pricing/usage types and Planner/Ranker factories. Produces `FormalExperimentTemplate`, sealed `FormalExperimentConfig`, immutable experiment records, exact A/B/C/D/P0 component mapping, model/environment lock commands and evaluator-only `OracleReferenceResult` generation.

- [ ] **Step 1: 写配置、锁文件、2×2 映射与 ORACLE 边界红测**

~~~python
def test_formal_config_rejects_missing_result_affecting_field(valid_payload):
    valid_payload.pop("model_snapshot_sha256")
    with pytest.raises(ValidationError):
        FormalExperimentConfig.model_validate(valid_payload)


def test_secret_is_not_part_of_config_or_group_hash(valid_payload):
    with pytest.raises(ValidationError):
        FormalExperimentConfig.model_validate({**valid_payload, "api_key": "secret"})


def test_formal_config_rejects_unmatched_pricing_identity(valid_payload):
    valid_payload["pricing_snapshot"]["model_id"] = "different-model"
    with pytest.raises(ValidationError, match="pricing identity"):
        FormalExperimentConfig.model_validate(valid_payload)


def test_replication_and_budget_sets_are_part_of_group_hash(valid_payload):
    base = FormalExperimentConfig.model_validate(valid_payload)
    other_seed = base.model_copy(
        update={
            "replication": base.replication.model_copy(
                update={"seed_values": (7, 8, 9)}
            )
        }
    )
    other_budgets = base.model_copy(
        update={"budget_sensitivity_presets": ("medium", "high")}
    )
    assert len({
        base.experiment_group_id(),
        other_seed.experiment_group_id(),
        other_budgets.experiment_group_id(),
    }) == 3


def test_abcd_mapping_is_exact(container):
    assert components_for(ExperimentVariant.A, container).ids == ("P1", "R1")
    assert components_for(ExperimentVariant.B, container).ids == ("P1", "R2")
    assert components_for(ExperimentVariant.C, container).ids == ("P2", "R1")
    assert components_for(ExperimentVariant.D, container).ids == ("P2", "R2")
    with pytest.raises(ValueError, match="evaluator-only"):
        components_for(ExperimentVariant.ORACLE, container)


def test_oracle_reference_contains_hashes_not_gold(oracle, frozen_corpus):
    result = oracle.score_reference("t1", frozen_records=frozen_corpus.records_by_id)
    assert result.approved_id_set_sha256 == canonical_sha256(("gold-ev-1", "gold-ev-2"))
    payload = result.model_dump_json()
    assert "gold-ev-1" not in payload
    with pytest.raises(GoldAccessViolation):
        oracle.approved_evidence_ids_for("unknown-task")
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/experiments/test_config.py tests/unit/experiments/test_factories.py tests/unit/benchmarks/test_oracle.py -q

Expected: FAIL because config/factory/lock and evaluator-only ORACLE implementations are missing.

- [ ] **Step 3: 实现唯一实验记录和 sealed config**

~~~python
class ExperimentVariant(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    P0 = "P0"
    ORACLE = "ORACLE"


class RankerComponentVariant(StrEnum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"


class DecodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    temperature: Literal[0.0]
    top_p: Literal[1.0]
    top_k: Literal[-1]
    max_tokens: Annotated[int, Field(gt=0)]
    repetition_penalty: Literal[1.0]
    thinking_mode: Literal["enabled", "disabled"]
    tensor_parallel_size: Annotated[int, Field(gt=0)]
    dtype: Literal["bfloat16", "float16"]
    max_model_len: Annotated[int, Field(gt=0)]


class ReplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed_values: Annotated[tuple[int, ...], Field(min_length=1)]
    candidate_pool_seed: int
    unseeded_repeat_count: Annotated[int, Field(ge=1)]


class ModelFileLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    size: Annotated[int, Field(ge=0)]
    git_blob_or_lfs_oid: str


class ModelSnapshotLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repository_id: str
    requested_revision: str
    resolved_revision: str
    files: tuple[ModelFileLock, ...]
    snapshot_sha256: str


class LockedDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    version: str
    artifact_sha256: str


class InferenceEnvironmentLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    python_version: str
    platform: str
    cuda_version: str
    driver_version: str
    gpu_model: str
    distributions: tuple[LockedDistribution, ...]
    launch_arguments_sha256: str
    model_snapshot_sha256: str
    environment_sha256: str


class ExperimentTaskRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    protocol: Literal["ranker_component", "planner_policy", "end_to_end", "reference"]
    variant: ExperimentVariant | RankerComponentVariant
    planner_id: str
    ranker_id: str
    budget_preset: Literal["low", "medium", "high"]
    seed: int | None = None
    repeat_id: int | None = None
    status: RunStatus
    validity: Literal["valid", "invalid"] = "valid"
    error_code: str | None = None
    candidate_pool_hash: str | None = None
    manifest_path: str
    artifact_ids: tuple[str, ...]
    usage: ResourceUsage
    pricing_snapshot_ids: tuple[str, ...]
    pricing_status: Literal["estimated"]
    cost_label: Literal["estimated_from_normalized_schedule"]


class ExperimentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    group_id: str
    protocol: Literal["ranker_component", "planner_policy", "end_to_end", "reference"]
    variant_components: dict[str, tuple[str, str]]
    runs: tuple[ExperimentTaskRun, ...]


class OracleReferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    dataset_version: str
    private_manifest_sha256: str
    frozen_snapshot_id: str
    approved_id_set_sha256: str
    metric_values: dict[str, MetricValue]
    evaluator_version: str
    created_at: datetime


class EvaluatorReferenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    group_id: str
    private_manifest_sha256: str
    evaluator_version: str
    task_ids_sha256: str
    oracle_results_sha256: str
    created_at: datetime


class FormalExperimentTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    dataset_version: str
    evaluation_timestamp: datetime
    model_id: str
    model_revision: str
    model_lock_path: str
    base_url: AnyHttpUrl
    provider_id: str
    provider_profile_id: str
    endpoint_type: Literal["openai_compatible_chat_completions"]
    decoding: DecodingConfig
    replication: ReplicationConfig
    prompt_version: str
    writer_prompt_version: str
    judge_model_id: str
    judge_model_revision: str
    judge_model_lock_path: str
    judge_prompt_version: str
    r1_model_id: str
    r1_model_revision: str
    r1_model_lock_path: str
    ranker_weights_version: str
    serving_runtime: Literal["vllm"]
    serving_runtime_version: str
    serving_runtime_platform: str
    serving_runtime_artifact_sha256: str
    serving_environment_lock_path: str
    budget_preset: Literal["low", "medium", "high"]
    budget_sensitivity_presets: Annotated[
        tuple[Literal["low", "medium", "high"], ...],
        Field(min_length=1),
    ]
    snapshot_collection_id: str
    corpus_version: str
    index_version: str
    evaluator_version: str
    pricing_status: Literal["estimated"]
    pricing_snapshot: PricingSnapshot


class FormalExperimentConfig(FormalExperimentTemplate):
    private_manifest_sha256: str
    model_snapshot_sha256: str
    judge_model_snapshot_sha256: str
    r1_model_snapshot_sha256: str
    serving_environment_sha256: str
    code_tree_sha256: str
    internal_runtime_task_hashes: dict[str, str]
    main_test_task_ids: tuple[str, ...]
    stability_task_ids: tuple[str, ...]
    cost_subset_task_ids: tuple[str, ...]
    p0_task_ids: tuple[str, ...]
    oracle_task_ids: tuple[str, ...]
    external_config_sha256: str | None = None
    external_lock_sha256: str | None = None
    external_runtime_task_hashes: dict[str, str] = Field(default_factory=dict)

    def experiment_group_id(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))[:16]
~~~

`ExperimentTaskRun` and `ExperimentRunResult` live only in `experiments/models.py`; every agent runner method in Task 15 returns these exact types. Exactly one of seed/repeat_id must be non-null. `RankerComponentVariant` is legal only when `protocol="ranker_component"`; A/B/C/D are legal for planner/end-to-end, P0 only for reference, and ORACLE never creates this record. Every successful agent record copies `pricing_snapshot_ids` and `usage.cost_usd` from its verified Core RunManifest; formal records require exactly the sealed pricing snapshot ID, `pricing_status="estimated"` and the fixed public cost label. Domain RunStatus remains unchanged: strict Replay invalidation is `status="failed", validity="invalid", error_code="REPLAY_MISS"`. Import `PricingSnapshot`/`CostCalculator` from Core and never reproduce their arithmetic.

Every hash field is lowercase SHA-256 and cannot be all zero. The config validator requires pricing identity to equal `(provider_id, endpoint_type, model_id)`, unique `replication.seed_values`, unique `budget_sensitivity_presets` containing the primary `budget_preset`, all internal task lists to be non-empty/unique/test-only and exactly match the private manifest, an exact sorted `internal_runtime_task_hashes` entry for every sealed internal test `RuntimeTask`, judge/model lock identities to match their IDs and revisions, and stability_task_ids to equal cost_subset_task_ids. `config freeze` derives that internal map by loading only the test-runtime files named by the verified private manifest, validating each object with the exact Task 1 schema, requiring its primary budget/profile, and hashing its canonical JSON; the generated map is deliberately absent from the template. The three external-authorization fields are an all-or-none set: primary formal config leaves both hashes null and the map empty; a Portfolio seal requires both non-zero hashes and a non-empty sorted map of canonical external task ID to canonical `RuntimeTask` SHA-256. Freeze/preflight also validates every base `RuntimeTask.request`: `execution_mode="hybrid"`, `access_profile="local"`, `run_purpose="benchmark"`, `provider_profile_id` equal to the sealed field, and `budget_preset` equal to the sealed primary budget. For a cost-sensitivity arm, the evaluator creates a new immutable `RuntimeTask` whose nested request is `base.request.model_copy(update={"budget_preset": selected})` and stages that exact object; the agent never edits it. Agent preflight requires the task ID and base hash to occur in exactly one authorization map (`internal_runtime_task_hashes` or, for namespaced Portfolio tasks, `external_runtime_task_hashes`), verifies the staged hash, requires the staged request budget to equal `AgentRunRequest.budget_preset` and belong to the sealed budget set, then copies only that nested budget back to the primary value and requires the resulting canonical hash to equal the authorized base hash. Thus the staged low/high copy has its own transmitted hash without becoming a second sealed identity. The selected preset is recorded in RunConfig, RunManifest, `ExperimentTaskRun` and its idempotency key. `code_tree_sha256` hashes canonical `(relative_path, file_sha256)` entries under `src/`, `apps/`, `experiments/` and all tracked benchmark Python/config/lock code including `benchmarks/datasets/**/*.py`; it also includes `pyproject.toml`, `uv.lock` and `formal.template.yaml`. It excludes every sealed output matching `benchmarks/configs/formal*.yaml` except the template, private/raw data, snapshots, generated results, Git metadata and caches. The formal runner separately requires a clean result-affecting Git tree (ignored `experiments/<group>/` artifacts do not count) and records the current 40-character commit in group.json and every agent RunManifest.

- [ ] **Step 4: 写不可直接运行的 formal.template.yaml**

Lock these reviewable values; generated hashes and private task IDs are deliberately absent from the template schema:

    dataset_id: frozen_ai_cs_60
    dataset_version: 1.0.0
    evaluation_timestamp: 2026-08-29T00:00:00Z
    model_id: Qwen/Qwen3-8B
    model_revision: cbe31c4effd2c6b8e18d453f0d3230dc6a1d2f18
    model_lock_path: benchmarks/configs/qwen3-8b.lock.json
    base_url: http://127.0.0.1:8001/v1
    provider_id: local-vllm
    provider_profile_id: formal-local-vllm
    endpoint_type: openai_compatible_chat_completions
    decoding:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 8192
      repetition_penalty: 1.0
      thinking_mode: enabled
      tensor_parallel_size: 1
      dtype: bfloat16
      max_model_len: 32768
    replication:
      seed_values: [20260829, 20260830, 20260831]
      candidate_pool_seed: 20260829
      unseeded_repeat_count: 3
    prompt_version: planner-writer-v1
    writer_prompt_version: writer-v1
    judge_model_id: Qwen/Qwen3-8B
    judge_model_revision: cbe31c4effd2c6b8e18d453f0d3230dc6a1d2f18
    judge_model_lock_path: benchmarks/configs/qwen3-8b.lock.json
    judge_prompt_version: evidence-judge-v1
    r1_model_id: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    r1_model_revision: e62509716f15c5fd03a6fd3156a4bc5e43f83f26
    r1_model_lock_path: models/embedding.lock.json
    ranker_weights_version: r2-v1
    serving_runtime: vllm
    serving_runtime_version: 0.28.0
    serving_runtime_platform: manylinux_2_31_x86_64
    serving_runtime_artifact_sha256: addb0ffdaafd8155d75e9b3f5ddb3da28fdee9e8a7097ede91f7db2e9e1a3889
    serving_environment_lock_path: benchmarks/configs/inference-environment.lock.json
    budget_preset: medium
    budget_sensitivity_presets: [low, medium, high]
    snapshot_collection_id: frozen-ai-cs-60-v1
    corpus_version: ai-cs-60-v1
    index_version: bm25-mixed-v1
    evaluator_version: evaluator-v1
    pricing_status: estimated
    pricing_snapshot:
      snapshot_id: formal-local-accounting-2026-08-29
      provider_id: local-vllm
      endpoint_type: openai_compatible_chat_completions
      model_id: Qwen/Qwen3-8B
      effective_at: 2026-08-29T00:00:00Z
      currency: USD
      input_tokens_per_million_usd: "0.10"
      output_tokens_per_million_usd: "0.20"
      cached_tokens_per_million_usd: "0.02"
      reasoning_tokens_per_million_usd: "0.20"

The rate schedule is a normalized local-inference comparison, not hosted retail pricing; every output labels it `estimated_from_normalized_schedule`. The pinned model is the official [Qwen3-8B repository](https://huggingface.co/Qwen/Qwen3-8B), the embedding lock uses the official [multilingual MiniLM repository](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), and the runtime artifact uses the official [vLLM 0.28.0 PyPI release](https://pypi.org/project/vllm/0.28.0/).

- [ ] **Step 5: 实现锁生成、ORACLE scorer 和变体 factory**

`lock_model` resolves the requested immutable repository revision, writes sorted `(path, size, git_blob_or_lfs_oid)` records and their canonical SHA-256, and refuses a moving branch/tag or response revision mismatch. It never downloads weight bodies. `capture_inference_environment`, run inside the actual vLLM server environment, writes sorted package/version/artifact hashes plus Python, OS, CUDA, driver, GPU, launch-arguments hash and model-lock hash; it redacts environment values and credentials. Both write staging-then-rename and are covered offline with response/environment fixtures; real lock capture is deferred to Task 16.

~~~python
class OracleEvidenceProvider:
    """Evaluator-only approved-ID lookup; never injected into an agent graph."""

    def __init__(
        self,
        *,
        approved_ids_by_task: Mapping[str, tuple[str, ...]],
        dataset_version: str,
        private_manifest_sha256: str,
        evaluator_version: str,
        evaluation_timestamp: datetime,
    ) -> None: ...

    def approved_evidence_ids_for(self, task_id: str) -> tuple[str, ...]: ...

    def score_reference(
        self,
        task_id: str,
        *,
        frozen_records: Mapping[str, FrozenEvidenceRecord],
    ) -> OracleReferenceResult: ...
~~~

`OracleEvidenceProvider` is evaluator-only. It accepts private approved-ID mappings, returns IDs only to its own `score_reference`, verifies each ID exists in that task's frozen corpus, computes metrics there, then emits `OracleReferenceResult` and `EvaluatorReferenceManifest`. Both use the sealed config's `evaluation_timestamp`, never wall-clock time, so repeated evaluation is byte-identical. Neither output contains IDs, excerpts, claims, rubrics, prompt text or private paths. Unknown task/ID raises GoldAccessViolation. It never implements SearchProvider and never crosses into an agent process.

`components_for` returns shared Planner/EvidenceRanker protocols for A=(P1,R1), B=(P1,R2), C=(P2,R1), D=(P2,R2); P0 uses P0ReActPlanner/R0SearchOrder. ORACLE raises an evaluator-only error and receives no RunConfig. Every formal A/B/C/D/P0 RunConfig sets workflow_id=research-v1; baseline-v1 remains Core quickstart only. The agent composition root loads one `FrozenCorpusSnapshot` and injects its `FrozenCorpusSearchProvider`, `FrozenCorpusFetcher` and `FrozenCorpusMaterializer` together with the locked Core HTML/PDF parsers. A formal factory must reject a Live fetcher, a materializer from another snapshot ID, or a StoreEvidence handler that regenerates snapshot Evidence IDs.

- [ ] **Step 6: 验证并提交 template 与代码**

Run:

    uv run pytest tests/unit/experiments/test_config.py tests/unit/experiments/test_factories.py tests/unit/benchmarks/test_oracle.py -q
    uv run ruff check experiments benchmarks/evaluators/oracle.py benchmarks/scripts/lock_model.py benchmarks/scripts/capture_inference_environment.py tests/unit/experiments tests/unit/benchmarks/test_oracle.py
    uv run pyright experiments benchmarks/evaluators/oracle.py benchmarks/scripts/lock_model.py benchmarks/scripts/capture_inference_environment.py

Expected: PASS offline. Do not create formal.yaml yet.

    git add experiments benchmarks/configs/formal.template.yaml benchmarks/evaluators/oracle.py benchmarks/scripts/lock_model.py benchmarks/scripts/capture_inference_environment.py tests/unit/experiments tests/unit/benchmarks/test_oracle.py
    git commit -m "feat: define formal experiment configuration"

### Task 15: 实现实验 Runner、三类协议与 CLI

**Files:**

- Create: experiments/runner.py
- Create: experiments/summarize.py
- Create: apps/cli/experiment.py
- Modify: benchmarks/datasets/isolation.py
- Modify: benchmarks/processes/agent.py
- Modify: benchmarks/processes/evaluator.py
- Modify: apps/cli/main.py
- Test: tests/unit/benchmarks/test_isolation.py
- Test: tests/integration/experiments/test_abcd_runner.py
- Test: tests/cli/test_experiment_commands.py

**Interfaces:** Consumes sealed `FormalExperimentConfig`, `ResearchRunner`, component factories and evaluator process launcher. Produces the three isolated experiment protocols, P0 and evaluator-only ORACLE reference artifacts, idempotent raw records and the `deepresearch experiment` CLI.

- [ ] **Step 1: 写 protocol 隔离和 CLI 红测**

~~~python
@pytest.mark.asyncio
async def test_abcd_runner_uses_exact_component_pairs(runner, config):
    result = await runner.run_abcd(config=config)
    assert result.variant_components == {
        "A": ("P1", "R1"),
        "B": ("P1", "R2"),
        "C": ("P2", "R1"),
        "D": ("P2", "R2"),
    }


@pytest.mark.asyncio
async def test_ranker_protocol_reuses_identical_candidate_pool(
    runner, config, spy_launcher
):
    result = await runner.run_ranker_component(config=config, task_ids=["t1"])
    hashes = {run.candidate_pool_hash for run in result.runs}
    assert len(hashes) == 1
    assert {run.ranker_id for run in result.runs} == {"R0", "R1", "R2"}
    assert {run.variant.value for run in result.runs} == {"R0", "R1", "R2"}
    pool_requests = [
        item for item in spy_launcher.requests
        if item.kind == "candidate_pool"
    ]
    assert len(pool_requests) == 1


def test_invalid_formal_config_exits_nonzero(cli_runner, invalid_config):
    result = cli_runner.invoke(
        app,
        ["experiment", "config", "validate", "--config", str(invalid_config)],
    )
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_evaluator_launches_agent_with_public_request_only(
    coordinator, config, spy_launcher
):
    await coordinator.run_one(
        config=config,
        protocol="end_to_end",
        task_id="t1",
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
    )
    request = spy_launcher.only_request
    assert request.kind == "variant_run"
    assert request.protocol == "end_to_end"
    assert "gold" not in request.model_dump_json().casefold()


def test_cost_sweep_cannot_accept_cli_budget_override(cli_runner, config_path):
    result = cli_runner.invoke(
        app,
        [
            "experiment", "run-cost-subset", "--config", str(config_path),
            "--budgets", "low,high",
        ],
    )
    assert result.exit_code != 0


def test_agent_guard_accepts_only_hash_verified_staged_config(
    agent_guard, repo_config_path, staged_config_path, config_sha256
):
    assert agent_guard.resolve_staged_config(
        staged_config_path,
        expected_sha256=config_sha256,
    ) == staged_config_path.resolve()
    with pytest.raises(GoldAccessViolation):
        agent_guard.resolve_staged_config(
            repo_config_path,
            expected_sha256=config_sha256,
        )
    with pytest.raises(GoldAccessViolation):
        agent_guard.resolve_staged_config(
            agent_guard.run_root / "config" / ".." / "requests" / "formal.yaml",
            expected_sha256=config_sha256,
        )


def test_agent_guard_rejects_untrusted_read_inputs_and_hash_mismatch(
    agent_guard,
    repo_root,
    valid_request_path,
    candidate_pool_path,
    candidate_pool_sha256,
    resume_checkpoint_path,
    resume_checkpoint_sha256,
):
    assert agent_guard.resolve_request(valid_request_path) == valid_request_path.resolve()
    with pytest.raises(GoldAccessViolation):
        agent_guard.resolve_request(repo_root / "request.json")
    with pytest.raises(GoldAccessViolation):
        agent_guard.resolve_candidate_pool(
            repo_root / "README.md",
            expected_sha256=candidate_pool_sha256,
        )
    with pytest.raises(GoldAccessViolation, match="hash"):
        agent_guard.resolve_candidate_pool(
            candidate_pool_path,
            expected_sha256="0" * 64,
        )
    with pytest.raises(GoldAccessViolation):
        agent_guard.resolve_resume_checkpoint(
            repo_root / "benchmarks" / "private" / "sentinel.sqlite3",
            expected_sha256=resume_checkpoint_sha256,
        )
    with pytest.raises(GoldAccessViolation, match="hash"):
        agent_guard.resolve_resume_checkpoint(
            resume_checkpoint_path,
            expected_sha256="0" * 64,
        )


@pytest.mark.asyncio
async def test_cost_arm_stages_budget_specific_authorized_runtime_task(
    coordinator, config, spy_launcher
):
    selected = next(
        item
        for item in config.budget_sensitivity_presets
        if item != config.budget_preset
    )
    await coordinator.run_one(
        config=config,
        protocol="end_to_end",
        task_id="t1",
        variant=ExperimentVariant.D,
        budget_preset=selected,
        seed=config.replication.seed_values[0],
    )
    request = spy_launcher.only_request
    staged = RuntimeTask.model_validate_json(
        Path(request.runtime_task_path).read_text(encoding="utf-8")
    )
    rebased = staged.model_copy(
        update={
            "request": staged.request.model_copy(
                update={"budget_preset": config.budget_preset}
            )
        }
    )
    assert staged.request.budget_preset == request.budget_preset == selected
    assert request.runtime_task_sha256 == canonical_sha256(
        staged.model_dump(mode="json")
    )
    assert request.base_runtime_task_sha256 == canonical_sha256(
        rebased.model_dump(mode="json")
    ) == config.internal_runtime_task_hashes["t1"]


def test_agent_rejects_request_runtime_budget_mismatch(
    valid_agent_request, agent_guard
):
    tampered = valid_agent_request.model_copy(update={"budget_preset": "high"})
    with pytest.raises(GoldAccessViolation, match="budget"):
        load_authorized_agent_inputs(tampered, guard=agent_guard)


def test_agent_rejects_checkpoint_identity_not_in_verified_database(
    valid_resume_agent_request, agent_guard
):
    tampered = valid_resume_agent_request.model_copy(
        update={
            "resume_checkpoint_ref": CheckpointRef(
                checkpoint_id="other-checkpoint",
                thread_id="other-thread",
                created_at=valid_resume_agent_request.resume_checkpoint_ref.created_at,
            )
        }
    )
    with pytest.raises(GoldAccessViolation, match="checkpoint identity"):
        load_authorized_agent_inputs(tampered, guard=agent_guard)
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/benchmarks/test_isolation.py tests/integration/experiments/test_abcd_runner.py tests/cli/test_experiment_commands.py -q

Expected: FAIL because runner and CLI command are missing.

- [ ] **Step 3: 实现不可变 run records 和 Runner**

~~~python
class ExperimentRunner:
    async def run_one(
        self,
        *,
        config: FormalExperimentConfig,
        protocol: Literal[
            "ranker_component", "planner_policy", "end_to_end", "reference",
        ],
        task_id: str,
        variant: ExperimentVariant | RankerComponentVariant,
        budget_preset: Literal["low", "medium", "high"],
        seed: int | None = None,
        repeat_id: int | None = None,
        resume: bool = False,
    ) -> ExperimentTaskRun: ...

    async def run_ranker_component(
        self,
        *,
        config: FormalExperimentConfig,
        task_ids: Sequence[str],
    ) -> ExperimentRunResult: ...

    async def run_planner_policy(
        self,
        *,
        config: FormalExperimentConfig,
        task_ids: Sequence[str],
        ranker_id: Literal["R1", "R2"],
    ) -> ExperimentRunResult: ...

    async def run_variant(
        self,
        variant: ExperimentVariant,
        *,
        config: FormalExperimentConfig,
        task_ids: Sequence[str],
    ) -> ExperimentRunResult: ...

    async def run_abcd(
        self,
        *,
        config: FormalExperimentConfig,
    ) -> ExperimentRunResult: ...

    async def run_cost_subset(
        self,
        *,
        config: FormalExperimentConfig,
    ) -> tuple[ExperimentRunResult, ...]: ...
~~~

`ExperimentRunner` is the evaluator-side coordinator; it does not import or instantiate `ResearchRunner`. Its constructor accepts the Task 2 `launch_agent` callable and an evaluator callback. The agent entrypoint alone imports the component factory and `ResearchRunner`, validates this exact request before import, then writes the exact receipt:

~~~python
class AgentRequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime_task_path: str
    runtime_task_sha256: str
    base_runtime_task_sha256: str
    snapshot_dir: str
    run_dir: str
    config_path: str
    config_sha256: str
    seed: int | None = None
    repeat_id: int | None = None
    resume_checkpoint_path: str | None = None
    resume_checkpoint_sha256: str | None = None
    resume_checkpoint_ref: CheckpointRef | None = None


class AgentCandidatePoolRequest(AgentRequestBase):
    kind: Literal["candidate_pool"] = "candidate_pool"
    protocol: Literal["ranker_component"] = "ranker_component"
    planner_id: Literal["P1"] = "P1"
    budget_preset: Literal["low", "medium", "high"]


class AgentVariantRunRequest(AgentRequestBase):
    kind: Literal["variant_run"] = "variant_run"
    protocol: Literal[
        "ranker_component", "planner_policy", "end_to_end", "reference",
    ]
    variant: Literal["A", "B", "C", "D", "P0", "R0", "R1", "R2"]
    budget_preset: Literal["low", "medium", "high"]
    candidate_pool_path: str | None = None
    candidate_pool_sha256: str | None = None


AgentRunRequest: TypeAlias = Annotated[
    AgentCandidatePoolRequest | AgentVariantRunRequest,
    Field(discriminator="kind"),
]


class AgentCandidatePoolReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    candidate_pool_path: str
    candidate_pool_sha256: str
    evidence_ids_sha256: str
    manifest_path: str
    manifest_sha256: str
    usage: ResourceUsage
    pricing_snapshot_ids: tuple[str, ...]


class AgentRunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    status: RunStatus
    error_code: str | None = None
    run_result_path: str
    manifest_path: str
    run_result_sha256: str
    manifest_sha256: str
    artifact_ids: tuple[str, ...]


class StagedRuntimeTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime_task_path: str
    runtime_task_sha256: str
    base_runtime_task_sha256: str


class AgentRuntimeGuard:
    """Task 2 interface with the Task 15 read-only input resolvers added."""

    def resolve_runtime_task(self, path: Path) -> Path: ...
    def resolve_snapshot(self, path: Path) -> Path: ...
    def resolve_request(self, path: Path) -> Path: ...
    def resolve_staged_config(
        self,
        path: Path,
        *,
        expected_sha256: str,
    ) -> Path: ...
    def resolve_candidate_pool(
        self,
        path: Path,
        *,
        expected_sha256: str,
    ) -> Path: ...
    def resolve_resume_checkpoint(
        self,
        path: Path,
        *,
        expected_sha256: str,
    ) -> Path: ...
    def resolve_output(self, path: Path) -> Path: ...
    def validate_payload(self, payload: JsonValue) -> None: ...


def stage_sealed_config(
    source_path: Path,
    *,
    expected_sha256: str,
    group_run_root: Path,
) -> Path: ...


def stage_authorized_runtime_task(
    base_task: RuntimeTask,
    *,
    config: FormalExperimentConfig,
    budget_preset: Literal["low", "medium", "high"],
    agent_input_root: Path,
    request_id: str,
    forbidden_private_root: Path,
) -> StagedRuntimeTask: ...


def load_authorized_agent_inputs(
    request: AgentRunRequest,
    *,
    guard: AgentRuntimeGuard,
) -> tuple[RuntimeTask, FormalExperimentConfig]: ...
~~~

These request/receipt schemas live at the top of `benchmarks/processes/agent.py` and import only stdlib, Pydantic, canonical public models and Core's unchanged `CheckpointRef`. `StagedRuntimeTask`, `stage_sealed_config` and `stage_authorized_runtime_task` live in evaluator.py; `load_authorized_agent_inputs` lives below the startup guard in agent.py and locally imports `FormalExperimentConfig` only after `assert_agent_environment`. `benchmarks/processes/evaluator.py` may import the request schemas, but `ResearchRunner`, component factories and provider clients remain lazy imports inside the agent command **after** `assert_agent_environment`; importing the schema module must not construct an agent or provider. The three `resume_checkpoint_*` fields are an all-null or all-present set; the loader additionally requires `resume_checkpoint_ref.thread_id` and `checkpoint_id` to exist in the verified SQLite checkpoint source. A non-resume request forbids all three.

Group preflight first validates and hashes the repository sealed config, then calls `stage_sealed_config` to copy its exact bytes to `experiments/<group>/config/formal.yaml`. The helper uses a sibling staging file, fsync and atomic rename; an existing identical staged file is verified and reused for sequential commands, while a mismatch is fatal and is never overwritten. It rehashes the destination before returning. `AgentRequestBase.config_path` always names this staged copy and `config_sha256` is the verified file SHA recorded in `group.json`; the repository `benchmarks/configs/formal*.yaml` path never enters a request or child environment.

For every private test task the evaluator loads the evaluator-only base runtime JSONL, proves its canonical hash equals the task's sealed authorization entry, and calls `stage_authorized_runtime_task` into `experiments/<group>/agent-inputs/`. That helper rejects an unsealed budget, creates `base_task.model_copy(update={"request": base_task.request.model_copy(update={"budget_preset": selected})})`, then delegates to Task 2 `materialize_agent_runtime_task` and returns both the selected-copy hash and base hash; neither evaluator nor agent mutates a staged object. Only that staged path and both hashes enter either request. The evaluator writes each discriminated request through fsync plus atomic rename under `experiments/<group>/requests/`, calls `python -m benchmarks.processes.agent --request <path> --receipt <staging-path>`, and reads the atomically renamed receipt only after a zero exit. Before parsing even the request JSON, the agent constructs the guard from its three allow-listed root environment values and requires the CLI path to pass `resolve_request`.

Task 15 extends `AgentRuntimeGuard` with four dedicated read-only resolvers. `resolve_request` accepts only a non-symlink regular file under `<run_root>/requests/`; `resolve_staged_config(path, expected_sha256=...)` accepts only `<run_root>/config/formal.yaml`; `resolve_candidate_pool` accepts only `<run_root>/candidate-pools/*.json`; and `resolve_resume_checkpoint` accepts only `<run_root>/resume-checkpoints/*.sqlite3`. Every resolver rejects traversal, symlinks, repo/private/outside paths and non-regular files; the latter three recompute the requested SHA before parsing. `resolve_output` explicitly rejects `config/`, `agent-inputs/`, `requests/`, `candidate-pools/` and `resume-checkpoints/` so the agent cannot rewrite authorized inputs. The sanitized child environment exposes only agent-input, snapshot and group run roots plus provider endpoint/credential variable names and seed; it removes GOLD_ROOT, the original private runtime/config paths and arbitrary inherited environment values.

After the environment check, `load_authorized_agent_inputs` resolves every path through the appropriate guard method, requires `config_sha256` and `runtime_task_sha256` to match bytes, parses the exact sealed config and canonical `RuntimeTask`, requires the task ID/base hash in exactly one internal-or-external authorization map, checks request/staged/config budget consistency, rebases only the nested budget to the primary value, and verifies that canonical hash against `base_runtime_task_sha256` and the selected authorization map. It also verifies snapshot hashes before any component/provider import. On `--resume`, the evaluator accepts only the prior artifact for the same group and idempotency key, verifies its manifest, atomically stages an immutable copy at `<run_root>/resume-checkpoints/<idempotency-key>.sqlite3`, and sends the staged SHA plus Core `CheckpointRef`. The agent resolver rehashes that source, verifies the requested checkpoint ID/thread exists in it, then copies it into the current attempt's writable run directory before opening Core's checkpointer; the immutable resume source is never updated. The agent then runs the requested operation, writes public artifacts, and fsyncs the receipt. The evaluator validates receipt hashes before opening gold and writing metrics. A child crash or any hash/identity/budget/path failure yields a failed `ExperimentTaskRun`; it never causes in-process fallback or silent budget override.

Ranker component first launches exactly one `AgentCandidatePoolRequest` per `(group, task)` using P1 retrieval, the sealed primary budget and `config.replication.candidate_pool_seed`, with key `SHA-256(group, ranker_component, POOL, task_id, candidate_pool_seed, primary_budget)`. The producer writes a sorted public-only candidate artifact to its writable `<run_dir>/staging/` output, never directly into the protected shared input subtree. After validating the receipt, manifest and candidate SHA, the evaluator atomically promotes the bytes to `<run_root>/candidate-pools/<pool-key>.json`; an existing identical file is reused and an existing mismatch is fatal. The receipt carries and verifies the separate Core RunManifest, usage and pricing snapshot IDs for pool construction; that cost is reported once as protocol setup cost and is never copied into each ranker arm. It then launches R0/R1/R2 `AgentVariantRunRequest` records for the configured replication seeds/repeats with that same promoted `candidate_pool_path` and SHA; those requests perform no search/fetch/model planning and read the artifact only through `resolve_candidate_pool`. `AgentVariantRunRequest` requires both candidate-pool fields for `protocol="ranker_component"` and variants R0/R1/R2, and forbids them for every other protocol/variant. This is how `ExperimentTaskRun.variant` can represent all three ranker-only runs without pretending they are A/B/C/D/P0. Planner policy holds Writer, model, frozen corpus/index, hard budget and ranker constant while allowing different query traces and candidates. End-to-end A/B/C/D shares all frozen fields and records actual differences.

`run_ranker_component.variant_components` is exactly `{"R0": ("P1", "R0"), "R1": ("P1", "R1"), "R2": ("P1", "R2")}`; `run_planner_policy` uses A/C when ranker is R1 and B/D when ranker is R2. These labels, protocol and budget are all present in each raw record, so no downstream summarizer infers a component from a filename.

Before starting, compute group_id from FormalExperimentConfig and atomically create its ignored `group.json` inside the matching child of experiments/. Each task/config/replication/budget uses idempotency key SHA-256(group, protocol, variant, task_id, seed-or-repeat, budget_preset). A completed key is skipped; failed/interrupted keys require explicit --resume. Never overwrite a completed raw record. The runner derives executions only from `config.replication`: use every sealed seed when `seed_supported=true`; otherwise use repeat IDs `1..unseeded_repeat_count` with `seed=null`. The low-level `run_one` validates that its explicit seed/repeat belongs to that sealed policy and that its budget is the primary preset or one of the sealed sensitivity presets; it is not an escape hatch. The runner derives cost runs only from `config.budget_sensitivity_presets`; public formal CLI methods accept no seed/repeat/budget override. Preflight occurs before any model call and verifies the clean result-affecting tree, code/model/R1/environment locks, serving/decoding profile, dataset/snapshot/task-list hashes, pricing identity, provider-reported model identity and current 40-character Git commit. Ignored, hash-verified children under `experiments/<group>/` are allowed so sequential protocol commands and resume remain executable.

- [ ] **Step 4: 实现 CLI**

Expose:

    uv run deepresearch experiment config freeze --source benchmarks/configs/formal.template.yaml --private-manifest benchmarks/private/frozen_ai_cs_60/private_manifest.json --model-lock benchmarks/configs/qwen3-8b.lock.json --r1-model-lock models/embedding.lock.json --serving-environment-lock benchmarks/configs/inference-environment.lock.json --output benchmarks/configs/formal.yaml
    uv run deepresearch experiment config validate --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment run-ranker --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment run-planner --config benchmarks/configs/formal.yaml --ranker R1
    uv run deepresearch experiment run --config benchmarks/configs/formal.yaml --variants A,B,C,D
    uv run deepresearch experiment run-stability --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment run-cost-subset --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment run-reference --config benchmarks/configs/formal.yaml --variants P0,ORACLE
    $groupId = uv run deepresearch experiment config id --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment summarize --experiment-dir "experiments/$groupId" --bootstrap-resamples 10000
    uv run deepresearch experiment summarize --experiment-dir "experiments/$groupId" --verify-only

`config freeze` refuses a dirty worktree, validates every referenced lock, derives exact task lists from the private manifest, calculates the specified code tree hash and writes through a sibling staging file; `--output` must not exist. Unit/CLI tests freeze only a temporary fixture config. The real formal.yaml is intentionally not generated or committed until Task 16.

If the Provider reports seed_supported=false, run exactly `config.replication.unseeded_repeat_count` independent repeats with repeat_id 1…N, save seed=null, and label the report independent_repeats rather than seeds. If it supports seeds, use exactly `config.replication.seed_values`; a CLI-supplied extra seed/repeat/budget is `INVALID_REQUEST` and no group directory is created.

`run-reference` splits its paths deliberately: P0 goes through the agent subprocess and produces ordinary `ExperimentTaskRun`/RunManifest records; ORACLE stays in the evaluator process and writes `oracle-reference.jsonl` plus `evaluator-reference-manifest.json`. ORACLE never creates a RunConfig, ExperimentTaskRun, RunManifest, report or model call. The command verifies that the public outputs contain no approved IDs, excerpts, claims, rubric text or private paths.

- [ ] **Step 5: 实现 summary 输出边界**

summarize writes:

    summary.json
    task_metrics.jsonl
    confidence_intervals.json
    pareto.json
    failures.jsonl
    manifest.sha256

It must keep ranker_component, planner_policy, end_to_end and reference sections distinct. Before writing, it verifies required protocol/budget sections, exact task/replication coverage and the absence of duplicate idempotency keys. It may not include raw model responses, private gold, hidden test prompts or credentials. Reference includes P0 agent metrics and hash-only ORACLE ceiling metadata, clearly labelled as incomparable to an agent execution cost/latency trace. `summarize --verify-only` is the read-only mode over those same result schemas: it verifies every entry in `manifest.sha256` and all completeness invariants without rewriting outputs, recomputing metrics or reading gold.

- [ ] **Step 6: 运行测试并提交**

Run: uv run pytest tests/unit/benchmarks/test_isolation.py tests/integration/experiments/test_abcd_runner.py tests/cli/test_experiment_commands.py -q

Expected: PASS using Replay providers only.

    git add experiments/runner.py experiments/summarize.py apps/cli/experiment.py apps/cli/main.py benchmarks/datasets/isolation.py benchmarks/processes/agent.py benchmarks/processes/evaluator.py tests/unit/benchmarks/test_isolation.py tests/integration/experiments/test_abcd_runner.py tests/cli/test_experiment_commands.py
    git commit -m "feat: run reproducible planner ranker experiments"

### Task 16: 验证 Strict Replay、停止路径和幂等性

**Files:**

- Create: tests/integration/experiments/test_strict_replay.py
- Extend: tests/integration/benchmarks/test_formal_protocols.py
- Create: tests/fixtures/experiments/sufficient/
- Create: tests/fixtures/experiments/conflict/
- Create: tests/fixtures/experiments/plateau/
- Create: tests/fixtures/experiments/budget_exhausted/
- Create: tests/fixtures/experiments/blocked/
- Modify: src/deepresearch/providers/replay.py
- Modify: src/deepresearch/workflow/research_graph.py
- Modify: experiments/runner.py
- Generate: benchmarks/configs/qwen3-8b.lock.json
- Generate: benchmarks/configs/inference-environment.lock.json
- Generate last: benchmarks/configs/formal.yaml

**Interfaces:** Consumes Task 15 runner plus Replay/checkpoint contracts. Produces five deterministic stop-path fixtures, strict no-fallback replay behavior, idempotent resume semantics, the final primary formal configuration seal, and the hash-verified primary ranker/planner/A–D/stability/cost/reference result group consumed by Tasks 18–19.

- [ ] **Step 1: 写端到端 Replay 红测**

~~~python
@pytest.mark.asyncio
async def test_record_then_strict_replay_is_byte_identical(harness):
    recorded = await harness.run("sufficient", mode="record")
    replayed = await harness.run(
        "sufficient",
        mode="strict_replay",
        manifest=recorded.manifest_path,
    )
    assert replayed.report_bytes == recorded.report_bytes
    assert replayed.evaluation_bytes == recorded.evaluation_bytes
    assert replayed.manifest.replay_parent == recorded.manifest.run_id


@pytest.mark.asyncio
async def test_unknown_replay_query_invalidates_run(harness):
    result = await harness.run("unknown-query", mode="strict_replay")
    assert result.status == "failed"
    assert result.validity == "invalid"
    assert result.error_code == "REPLAY_MISS"
    assert result.live_provider_calls == 0


@pytest.mark.asyncio
async def test_checkpoint_resume_does_not_double_charge(harness):
    interrupted = await harness.interrupt_after_node("conflict", "RankEvidence")
    resumed = await harness.resume(interrupted.checkpoint)
    assert resumed.usage == uninterrupted_usage("conflict")
~~~

- [ ] **Step 2: 运行红测并定位缺少的 fixture 或行为**

Run: uv run pytest tests/integration/experiments/test_strict_replay.py tests/integration/benchmarks/test_formal_protocols.py -q

Expected: FAIL on missing fixtures or any replay nondeterminism; no online calls.

- [ ] **Step 3: 完成五类 fixture**

Each fixture contains frozen search records, model records, parsed documents, graph events, expected stop code and expected artifact hashes. Cover SUFFICIENT, one directed conflict search, PLATEAU after two gains below 0.05, BUDGET_EXHAUSTED as partial, and BLOCKED after alternatives fail.

- [ ] **Step 4: 修正列明的 integration glue，不放宽断言**

Only the three source files listed in this task may change: `providers/replay.py` may add missing strict-key/hash validation, `workflow/research_graph.py` may thread the injected deterministic clock/ID factory through existing nodes, and `experiments/runner.py` may finish idempotent resume/result wiring. Do not rename a public contract, alter P1/P2/R1/R2 behavior, weaken stop predicates, or add a live fallback. Ensure replay keys include provider/model/prompt/request/schema versions; cached calls restore recorded usage without charging again. Unknown queries and schema mismatches invalidate the benchmark task.

- [ ] **Step 5: 运行全套离线质量门**

Run:

    uv run pytest tests/unit/benchmarks tests/unit/experiments tests/contracts/test_frozen_search_contract.py tests/integration/experiments tests/integration/benchmarks -q
    uv run ruff check benchmarks experiments apps/cli tests
    uv run pyright benchmarks experiments

Expected: all pass, zero online or paid calls.

- [ ] **Step 6: 捕获真实模型/runtime locks 并提交所有结果相关源码**

Resolve the model metadata from the pinned official revision:

    uv run python -m benchmarks.scripts.lock_model --repo Qwen/Qwen3-8B --revision cbe31c4effd2c6b8e18d453f0d3230dc6a1d2f18 --output benchmarks/configs/qwen3-8b.lock.json

In the exact activated Python environment that launches the formal vLLM server, run:

    python -m benchmarks.scripts.capture_inference_environment --template benchmarks/configs/formal.template.yaml --model-lock benchmarks/configs/qwen3-8b.lock.json --output benchmarks/configs/inference-environment.lock.json

The first command verifies every returned repository file record and resolved commit before writing. The second fails unless installed vLLM is 0.28.0 and its artifact/platform match the template; it captures package/artifact hashes and hardware/runtime metadata but no environment values. Review both canonical JSON files, rerun the full Step 5 gate, then make the pre-seal commit:

    git add src/deepresearch/providers/replay.py src/deepresearch/workflow/research_graph.py experiments/runner.py tests/integration/experiments tests/integration/benchmarks tests/fixtures/experiments benchmarks/configs/qwen3-8b.lock.json benchmarks/configs/inference-environment.lock.json
    git commit -m "test: lock benchmark strict replay protocols"

- [ ] **Step 7: 从干净的 pre-seal commit 生成唯一 formal.yaml**

Run:

    git status --short
    uv run deepresearch experiment config freeze --source benchmarks/configs/formal.template.yaml --private-manifest benchmarks/private/frozen_ai_cs_60/private_manifest.json --model-lock benchmarks/configs/qwen3-8b.lock.json --r1-model-lock models/embedding.lock.json --serving-environment-lock benchmarks/configs/inference-environment.lock.json --output benchmarks/configs/formal.yaml
    uv run deepresearch experiment config validate --config benchmarks/configs/formal.yaml

Expected: status is empty before freeze; formal.yaml has non-zero private/model/R1/environment/code hashes and exact preregistered task IDs. The computed code tree deliberately excludes formal.yaml, so adding the sealed file cannot invalidate it.

    git add benchmarks/configs/formal.yaml
    git commit -m "chore: seal formal experiment configuration"

- [ ] **Step 8: 从 clean seal commit 做最后 preflight**

Run:

    git status --short
    uv run deepresearch experiment config validate --config benchmarks/configs/formal.yaml --require-clean-worktree --verify-current-tree
    uv run pytest tests/integration/benchmarks/test_formal_protocols.py tests/integration/experiments/test_strict_replay.py -q

Expected: empty status, matching code/model/environment hashes, current 40-character seal commit recorded by a dry-run group preflight, all tests pass and no network/provider calls occur. Any later result-affecting source or lock change requires a new sealed config and experiment_group_id.

- [ ] **Step 9: 在 seal commit 上执行完整 primary 协议并冻结 summary**

Start the local vLLM server from the exact environment and launch arguments captured by `inference-environment.lock.json`; the runner preflight must verify its reported model identity before the first task. From the still-clean seal commit run the commands in this exact order:

    uv run deepresearch experiment run-ranker --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment run-planner --config benchmarks/configs/formal.yaml --ranker R1
    uv run deepresearch experiment run --config benchmarks/configs/formal.yaml --variants A,B,C,D
    uv run deepresearch experiment run-stability --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment run-cost-subset --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment run-reference --config benchmarks/configs/formal.yaml --variants P0,ORACLE
    $groupId = uv run deepresearch experiment config id --config benchmarks/configs/formal.yaml
    uv run deepresearch experiment summarize --experiment-dir "experiments/$groupId" --bootstrap-resamples 10000

Every command reuses the same sealed replication and budget sets; no command accepts an unsealed seed/repeat/budget override. `run-reference` launches P0 through the agent boundary and computes ORACLE only in the evaluator. Because `experiments/*/` is ignored, request/result creation does not dirty the result-affecting tree; every file is still immutable and hash-chained inside the group.

Run:

    uv run deepresearch experiment summarize --experiment-dir "experiments/$groupId" --verify-only
    git status --short

Expected: verification exits 0; `summary.json`, `task_metrics.jsonl`, confidence, Pareto, failure and reference artifacts all match `manifest.sha256`; every configured seed or repeat is present exactly once per applicable task/variant/budget; the worktree remains empty. This step, not seal creation alone, is the prerequisite for Task 18 and the primary input to Task 19.

### Task 17: 接入三组外部 Benchmark（Portfolio Full）

**Files:**

- Create: benchmarks/external/__init__.py
- Create: benchmarks/external/base.py
- Create: benchmarks/external/livedrbench.py
- Create: benchmarks/external/frames.py
- Create: benchmarks/external/deepresearchbench.py
- Create: benchmarks/configs/external.yaml
- Create: benchmarks/scripts/fetch_external.py
- Create generated: benchmarks/external/external.lock.json
- Generate private ignored: benchmarks/private/external/raw/**
- Generate private ignored: benchmarks/private/external/staging/**
- Generate ignored: benchmarks/snapshots/external/**
- Create: experiments/external_runner.py
- Modify: experiments/config.py
- Modify: apps/cli/experiment.py
- Modify: .gitignore
- Test: tests/unit/benchmarks/test_external_adapters.py
- Test: tests/integration/experiments/test_external_runner.py
- Generate last: benchmarks/configs/formal-portfolio.yaml

**Interfaces:** Consumes immutable upstream revisions/licenses, canonical Task 1 `FrozenEvidenceRecord`/`RuntimeTask`, Task 2 `materialize_agent_runtime_task`, Task 3 `build_snapshot one` and the Task 15 evaluator→agent process boundary. Produces gold-free external evidence batches, hash-verified external frozen snapshots, evaluator-side `ExternalTaskSelection(runtime_task: RuntimeTask, evaluation_plan: ExternalEvaluationPlan)`, a verified lock, Portfolio task authorization hashes, `ExternalExperimentRunner`, `run-external`, and a separately sealed Portfolio group kept distinct from the internal primary benchmark. The agent loader remains byte-for-byte canonical `RuntimeTask`; no external schema union crosses the process boundary.

- [ ] **Step 1: 写版本锁和统一 adapter 红测**

~~~python
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from benchmarks.datasets.models import RuntimeTask
from deepresearch.providers.errors import ProviderError
from deepresearch.providers.frozen_search import FrozenCorpusSnapshot


@pytest.mark.parametrize(
    "adapter_name,expected_count",
    [
        ("livedrbench", 10),
        ("frames", 20),
        ("deepresearchbench", 10),
    ],
)
def test_external_selection_is_pinned_and_deterministic(
    adapter_factory, lock_file, external_raw_root, external_snapshot_root,
    portfolio_template, adapter_name, expected_count
):
    adapter = adapter_factory(
        adapter_name,
        lock_file=lock_file,
        raw_root=external_raw_root,
        snapshot_root=external_snapshot_root,
    )
    first = adapter.select(
        provider_profile_id=portfolio_template.provider_profile_id,
        budget_preset=portfolio_template.budget_preset,
    )
    second = adapter.select(
        provider_profile_id=portfolio_template.provider_profile_id,
        budget_preset=portfolio_template.budget_preset,
    )
    assert first == second
    assert len(first) == expected_count
    assert len({item.evaluation_plan.external_id for item in first}) == expected_count
    assert all(isinstance(item.runtime_task, RuntimeTask) for item in first)
    assert all(
        set(item.runtime_task.model_dump()) == {
            "task_id", "category", "request", "evaluation_cutoff",
            "snapshot_id", "corpus_version", "index_version",
        }
        for item in first
    )
    for item in first:
        snapshot_lock = adapter.snapshot_lock_for(item.runtime_task.task_id)
        assert (
            item.runtime_task.snapshot_id,
            item.runtime_task.corpus_version,
            item.runtime_task.index_version,
        ) == (
            snapshot_lock.snapshot_id,
            snapshot_lock.corpus_version,
            snapshot_lock.index_version,
        )


def test_lock_requires_commit_hash_dataset_hash_and_license(lock_payload):
    lock_payload["frames"].pop("sha256")
    with pytest.raises(ValidationError):
        ExternalBenchmarkLock.model_validate(lock_payload)


def test_external_raw_root_is_fixed_private_and_ignored(
    repo_root, external_config
):
    assert external_config.raw_root == "benchmarks/private/external/raw"
    check = subprocess.run(
        [
            "git", "check-ignore", "--no-index", "-q",
            "benchmarks/private/external/raw/probe.bin",
        ],
        cwd=repo_root,
        check=False,
    )
    assert check.returncode == 0
    snapshot_check = subprocess.run(
        [
            "git", "check-ignore", "--no-index", "-q",
            "benchmarks/snapshots/external/frames/ext-frames-1/snapshot.json",
        ],
        cwd=repo_root,
        check=False,
    )
    assert snapshot_check.returncode == 0


def test_external_records_build_canonical_gold_free_snapshot(
    external_materializer,
    selection_manifest_path,
    documents_staging_root,
    external_snapshot_root,
):
    built = external_materializer.build(
        selection_manifest_path=selection_manifest_path,
        documents_staging_root=documents_staging_root,
        snapshot_root=external_snapshot_root,
    )
    snapshot = FrozenCorpusSnapshot.load(
        external_snapshot_root / built.snapshots[0].snapshot_relative_path,
        task_id=built.snapshots[0].task_id,
    )
    assert snapshot.manifest.document_count > 0
    assert built.snapshots[0].snapshot_id == snapshot.manifest.snapshot_id
    assert built.snapshots[0].corpus_version == snapshot.manifest.corpus_version
    assert built.snapshots[0].index_version == snapshot.manifest.index_version
    payload = "\n".join(
        record.model_dump_json()
        for record in built.frozen_records_by_task[built.snapshots[0].task_id]
    ).casefold()
    assert all(
        forbidden not in payload
        for forbidden in ("gold", "rubric", "acceptable_claims")
    )


@pytest.mark.parametrize("snapshot_state", ["missing", "hash_mismatch"])
def test_external_adapter_rejects_unverified_snapshot(
    adapter_factory,
    lock_file,
    external_raw_root,
    broken_external_snapshot_root,
    snapshot_state,
):
    adapter = adapter_factory(
        "frames",
        lock_file=lock_file,
        raw_root=external_raw_root,
        snapshot_root=broken_external_snapshot_root(snapshot_state),
    )
    with pytest.raises(ProviderError) as error:
        adapter.select(
            provider_profile_id="formal-local-vllm",
            budget_preset="medium",
        )
    assert error.value.code == "INVALID_SNAPSHOT"


@pytest.mark.asyncio
async def test_external_runner_keeps_gold_plan_out_of_agent_request(
    external_runner, portfolio_config, external_config_path,
    external_lock_path, spy_launcher
):
    result = await external_runner.run(
        config=portfolio_config,
        external_config_path=external_config_path,
        external_lock_path=external_lock_path,
        benchmarks=("frames",),
    )
    assert result.benchmark_counts == {"frames": 20}
    assert "evaluation_plan" not in spy_launcher.only_request.model_dump_json()
    staged = RuntimeTask.model_validate_json(
        Path(spy_launcher.only_request.runtime_task_path).read_text(encoding="utf-8")
    )
    assert isinstance(staged, RuntimeTask)
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/benchmarks/test_external_adapters.py tests/integration/experiments/test_external_runner.py -q

Expected: FAIL because external adapters and runner are missing.

- [ ] **Step 3: 实现锁定下载流程**

`external.yaml` has the exact top-level paths `raw_root: benchmarks/private/external/raw`, `documents_staging_root: benchmarks/private/external/staging` and `snapshot_root: benchmarks/snapshots/external`; it also locks one corpus version per benchmark and the shared index version. Its schema rejects any other roots or blank/moving versions. `fetch_external` resolves those repo-relative paths, proves the first two are under the already ignored `benchmarks/private/` root and the third is under the exact external-snapshot ignore, rejects symlinks/traversal, and writes every upstream payload only below the raw root. `download` resolves each upstream revision exactly once and atomically writes an ignored provisional lock containing the 40-character commit or immutable dataset revision, direct source URL, root-relative raw path, downloaded SHA-256, license identifier, adapter version and fetched_at. The final public lock is created only by `build-snapshots` after materialization succeeds; neither lock contains an absolute/private path or raw bytes. Later calls and every adapter must receive the same resolved roots, join only locked relative paths beneath them, reject symlinks, rehash before reading, and fail on any mismatch; a refresh writes a new final lock version and never edits an existing one.

~~~python
class ExternalSnapshotLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark: Literal["livedrbench", "frames", "deepresearchbench"]
    external_id: str
    task_id: str
    snapshot_id: str
    corpus_version: str
    index_version: str
    snapshot_relative_path: str
    records_sha256: str
    manifest_sha256: str


class ExternalSnapshotBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshots: tuple[ExternalSnapshotLock, ...]
    frozen_records_by_task: dict[str, tuple[FrozenEvidenceRecord, ...]]


class ExternalTaskAdapter(Protocol):
    def frozen_records_for(
        self,
        external_id: str,
    ) -> tuple[FrozenEvidenceRecord, ...]: ...

    def snapshot_lock_for(self, task_id: str) -> ExternalSnapshotLock: ...


class ExternalSnapshotMaterializer:
    def build(
        self,
        *,
        selection_manifest_path: Path,
        documents_staging_root: Path,
        snapshot_root: Path,
    ) -> ExternalSnapshotBuildResult: ...
~~~

Each adapter derives stable source/evidence IDs, raw/parsed/excerpt hashes and canonical locators from its already hash-verified raw item and returns only Task 1 `FrozenEvidenceRecord`; acceptable answers, relevance labels, rubrics and evaluator mappings are forbidden. `fetch_external materialize-records` writes one canonical JSONL per task plus a sorted private selection manifest under `benchmarks/private/external/staging/`. `ExternalSnapshotMaterializer.build` consumes only that manifest/root, invokes Task 3's exact `build_snapshot one` implementation for `benchmarks/snapshots/external/<benchmark>/<task_id>`, then immediately reloads every child with `FrozenCorpusSnapshot.load`. It copies `snapshot_id`, `corpus_version` and `index_version` from the loaded `FrozenCorpusManifest`, verifies the task identity and outer `manifest.sha256`, and returns sorted lock records; that outer hash covers the manifest and therefore its `documents_sha256`/`index_sha256`. Only after all selected 10/20/10 tasks pass does `fetch_external build-snapshots` atomically create `external.lock.json` with the upstream/raw hashes and the complete snapshot-lock tuple.

For each sorted selection row, `build-snapshots` invokes rather than reimplements this exact Task 3 command shape, substituting only lock-derived values:

    uv run python -m benchmarks.scripts.build_snapshot one --task-id <task_id> --documents benchmarks/private/external/staging/<task_id>.jsonl --output benchmarks/snapshots/external/<benchmark>/<task_id> --corpus-version <configured_corpus_version> --index-version <configured_index_version>

Append the exact `.gitignore` rule `benchmarks/snapshots/external/`. The entire external snapshot tree is a reproducible local artifact, while the private staging files are already covered by `benchmarks/private/`; neither may dirty the seal commit. `fetch_external restore --lock ...` is the fresh-checkout path: it downloads only the locked upstream revisions, revalidates every raw hash, regenerates an entirely absent snapshot through Task 3, verifies and reuses an identical complete child, and rejects an incomplete existing child or any hash mismatch without silently selecting a different task. It never rewrites the committed lock.

- [ ] **Step 4: 实现能力匹配和固定抽样**

- LiveDRBench: filter computer-science, prior-art and dataset-discovery tasks; deterministically choose 10 by SHA-256 order of external_id.
- FRAMES: choose 20 tasks that have a locally available multi-document context and evidence mapping; use only Evidence Ranker and multi-document retrieval metrics.
- DeepResearch Bench: choose 10 research-report tasks with usable citation/rubric metadata; use report and citation comparison only.

~~~python
class ExternalEvaluationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark: Literal["livedrbench", "frames", "deepresearchbench"]
    external_id: str
    supported_metric_names: tuple[str, ...]
    private_scoring_reference: str
    upstream_record_sha256: str


class ExternalTaskSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime_task: RuntimeTask
    evaluation_plan: ExternalEvaluationPlan
~~~

Every adapter is constructed with `lock_file`, the exact `raw_root` loaded from `external.yaml`, and `snapshot_root=benchmarks/snapshots/external`, verifies all three before selection, and exposes `select(*, provider_profile_id: str, budget_preset: Literal["low", "medium", "high"]) -> tuple[ExternalTaskSelection, ...]`. Selection is illegal until every chosen task has one matching `ExternalSnapshotLock` and `FrozenCorpusSnapshot.load` reproduces its manifest hash. It deterministically maps upstream items into the **existing exact** `RuntimeTask`: task IDs are namespaced `ext-<benchmark>-<stable-id>`, FRAMES maps to `multi_hop_history`, retrieval/report tasks map to the closest frozen `TaskCategory`, and the canonical `ResearchRequest` is hybrid/local/benchmark with the supplied sealed provider profile and primary budget. `config freeze` passes the fixed roots and values from the loaded formal template; `ExternalExperimentRunner` passes the same fields from the resulting Portfolio config and rejects any regenerated hash mismatch. `snapshot_id`, corpus and index versions come exactly from the verified snapshot lock. The evaluator keeps the wrapper and `ExternalEvaluationPlan`; it calls `materialize_agent_runtime_task(selection.runtime_task, ...)`, so only the seven canonical runtime fields reach the unchanged agent loader. The evaluation plan contains supported metric names and private scoring references and never enters AgentRunRequest, Graph State, prompt, RunManifest or public artifacts. Adapters must not reinterpret a short-answer score as long-report quality.

- [ ] **Step 5: 实现独立 external runner 与 CLI**

~~~python
class ExternalExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    portfolio_group_id: str
    formal_config_sha256: str
    external_lock_sha256: str
    benchmark_counts: dict[str, int]
    runs: tuple[ExperimentTaskRun, ...]
    metrics_artifact_sha256: str


class ExternalExperimentRunner:
    async def run(
        self,
        *,
        config: FormalExperimentConfig,
        external_config_path: Path,
        external_lock_path: Path,
        benchmarks: Sequence[Literal["livedrbench", "frames", "deepresearchbench"]],
    ) -> ExternalExperimentResult: ...
~~~

The runner verifies the external config/lock and current Portfolio code tree, requires the config's resolved raw root to be exactly `<repo>/benchmarks/private/external/raw`, and lets adapters read only hash-matched locked files below that evaluator-only root. Before materializing a task, it loads `<repo>/benchmarks/snapshots/external/<locked-relative-path>` through Task 3, requires its manifest hash and all RuntimeTask snapshot/corpus/index fields to equal `ExternalSnapshotLock`, then recomputes the canonical `RuntimeTask` hash and requires it to equal `config.external_runtime_task_hashes[task_id]`. It materializes exactly one task outside the private/raw external root plus that verified frozen snapshot. It launches the existing sanitized agent subprocess with the canonical loader; neither the raw root nor its path is mounted or passed to the child, and scoring with that task's `ExternalEvaluationPlan` remains evaluator-only. LiveDRBench/DeepResearchBench use end-to-end D; FRAMES uses the Task 15 fixed-pool R2 component path. It writes under `experiments/<portfolio_group_id>/external/<benchmark>/`, includes formal/external lock hashes in every aggregate, and never merges external scores into the internal primary confidence intervals. A strict Replay integration fixture proves the staged RuntimeTask parses with the Task 1 model and the agent request contains no external answer/evidence mapping; missing or mismatched snapshot artifacts fail before agent launch with `INVALID_SNAPSHOT`.

Expose:

    uv run deepresearch experiment run-external --config benchmarks/configs/formal-portfolio.yaml --external-config benchmarks/configs/external.yaml --external-lock benchmarks/external/external.lock.json --benchmarks livedrbench,frames,deepresearchbench

- [ ] **Step 6: 下载、人工核对 license 并跑 contracts**

Run:

    uv run python -m benchmarks.scripts.fetch_external download --config benchmarks/configs/external.yaml --raw-root benchmarks/private/external/raw --write-download-lock benchmarks/private/external/staging/download.lock.json
    uv run python -m benchmarks.scripts.fetch_external materialize-records --config benchmarks/configs/external.yaml --download-lock benchmarks/private/external/staging/download.lock.json --raw-root benchmarks/private/external/raw --documents-staging-root benchmarks/private/external/staging --write-selection-manifest benchmarks/private/external/staging/selection.json
    uv run python -m benchmarks.scripts.fetch_external build-snapshots --config benchmarks/configs/external.yaml --download-lock benchmarks/private/external/staging/download.lock.json --selection-manifest benchmarks/private/external/staging/selection.json --documents-staging-root benchmarks/private/external/staging --snapshot-root benchmarks/snapshots/external --write-lock benchmarks/external/external.lock.json
    uv run python -m benchmarks.scripts.fetch_external verify-snapshots --lock benchmarks/external/external.lock.json --snapshot-root benchmarks/snapshots/external --expected-counts livedrbench=10,frames=20,deepresearchbench=10
    git check-ignore --no-index -q benchmarks/snapshots/external/frames/ext-frames-probe/snapshot.json
    uv run pytest tests/unit/benchmarks/test_external_adapters.py -q
    uv run pytest tests/integration/experiments/test_external_runner.py -q

Expected: 10/20/10 deterministic tasks, exactly 40 loadable snapshot directories, and matching raw/record/manifest hashes in the completed lock. The CLI rejects raw/staging/snapshot roots unequal to the fixed locations; all downloaded/staging payloads are under the existing `benchmarks/private/` ignore, the external snapshot root matches the new exact ignore, and no adapter reads another path. This rule applies regardless of redistribution license; only the lock, adapter and ignore rule are commit candidates.

- [ ] **Step 7: 提交 external 代码与 lock**

    git check-ignore --no-index -q benchmarks/private/external/raw/probe.bin
    git check-ignore --no-index -q benchmarks/snapshots/external/frames/ext-frames-probe/snapshot.json
    git add .gitignore benchmarks/external benchmarks/configs/external.yaml benchmarks/scripts/fetch_external.py experiments/config.py experiments/external_runner.py apps/cli/experiment.py tests/unit/benchmarks/test_external_adapters.py tests/integration/experiments/test_external_runner.py
    git diff --cached --name-only -- benchmarks/private/external benchmarks/snapshots/external
    git commit -m "feat: add pinned external benchmark adapters"

The staged-name check must print nothing; abort the commit if it names any raw file.

- [ ] **Step 8: 在 clean external commit 上生成 Portfolio seal 并运行**

Task 17 changes result-affecting code, so the Task 16 formal.yaml remains immutable evidence for the primary run and must not be overwritten. From the clean Task 17 commit run:

    uv run python -m benchmarks.scripts.fetch_external restore --config benchmarks/configs/external.yaml --raw-root benchmarks/private/external/raw --documents-staging-root benchmarks/private/external/staging --snapshot-root benchmarks/snapshots/external --lock benchmarks/external/external.lock.json
    uv run python -m benchmarks.scripts.fetch_external verify-snapshots --lock benchmarks/external/external.lock.json --snapshot-root benchmarks/snapshots/external --expected-counts livedrbench=10,frames=20,deepresearchbench=10
    uv run deepresearch experiment config freeze --source benchmarks/configs/formal.template.yaml --private-manifest benchmarks/private/frozen_ai_cs_60/private_manifest.json --model-lock benchmarks/configs/qwen3-8b.lock.json --r1-model-lock models/embedding.lock.json --serving-environment-lock benchmarks/configs/inference-environment.lock.json --external-config benchmarks/configs/external.yaml --external-lock benchmarks/external/external.lock.json --output benchmarks/configs/formal-portfolio.yaml
    uv run deepresearch experiment config validate --config benchmarks/configs/formal-portfolio.yaml
    git add benchmarks/configs/formal-portfolio.yaml
    git commit -m "chore: seal portfolio experiment configuration"
    uv run deepresearch experiment config validate --config benchmarks/configs/formal-portfolio.yaml --require-clean-worktree --verify-current-tree
    uv run deepresearch experiment run-external --config benchmarks/configs/formal-portfolio.yaml --external-config benchmarks/configs/external.yaml --external-lock benchmarks/external/external.lock.json --benchmarks livedrbench,frames,deepresearchbench

`restore` is idempotent and reconstructs all ignored runtime artifacts from the committed lock on a fresh checkout; it must report exactly 10/20/10 verified snapshots before freeze and leave `git status --short` empty. `config freeze` with both external arguments runs every adapter read-only, writes no raw or snapshot data, and seals the external config hash, lock hash and sorted canonical RuntimeTask hash map. Supplying only one external argument, an evaluator-plan hash in the runtime map, or a task without the exact locked snapshot manifest is `INVALID_REQUEST`.

Expected: the new config has a distinct group ID/current code hash and non-empty external authorization fields, the original formal.yaml/group is unchanged, and external result counts are exactly 10/20/10. The seal hash excludes both sealed YAML outputs but includes the template, adapter code and external lock.

### Task 18: 实现 A/D 盲化人工评审

**Files:**

- Create: benchmarks/evaluators/human.py
- Test: tests/unit/benchmarks/test_human.py
- Create private: benchmarks/private/human/blinded_packets/
- Create private: benchmarks/private/human/ratings.jsonl

**Interfaces:** Consumes paired A/D reports from the preregistered stability subset. Produces blind packets, validated three-rater `HumanRating` records and aggregate agreement/correlation statistics without publishing identities or blind mappings.

- [ ] **Step 1: 写盲化、评分和一致性红测**

~~~python
def test_blinding_removes_variant_and_run_identifiers(pair):
    packet = blind_pair(pair, seed=20260829)
    payload = packet.model_dump_json()
    assert "variant_a" not in payload.casefold()
    assert "variant_d" not in payload.casefold()
    assert pair.left_run_id not in payload
    assert pair.right_run_id not in payload


def test_every_task_requires_three_distinct_raters(ratings):
    report = validate_human_ratings(ratings, expected_tasks=20, raters_per_task=3)
    assert report.valid
    assert all(count == 3 for count in report.raters_per_task.values())


def test_ordinal_alpha_and_spearman_are_reported(ratings):
    result = summarize_human_ratings(ratings)
    assert set(result.krippendorff_alpha) == set(HUMAN_DIMENSIONS)
    assert set(result.auto_metric_spearman) == set(HUMAN_DIMENSIONS)
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/unit/benchmarks/test_human.py -q

Expected: FAIL because human evaluator is missing.

- [ ] **Step 3: 实现评分 schema**

~~~python
HUMAN_DIMENSIONS = (
    "factual_correctness",
    "evidence_sufficiency",
    "information_coverage",
    "analysis_depth",
    "readability",
    "citation_verifiability",
)


class HumanRating(BaseModel):
    packet_id: str
    task_id: str
    rater_id: str
    report_label: Literal["X", "Y"]
    scores: dict[Literal[
        "factual_correctness",
        "evidence_sufficiency",
        "information_coverage",
        "analysis_depth",
        "readability",
        "citation_verifiability",
    ], Annotated[int, Field(ge=1, le=5)]]
    overall_preference: Literal["X", "Y", "TIE"]
    rationale: str
~~~

blind_pair removes model, variant, cost, token, latency and run identifiers, then seed-randomizes which report is X or Y. Keep the mapping in a separate private file. Rater IDs are pseudonymous and no personal information is committed.

- [ ] **Step 4: 生成 20 题 packet 并收齐三人评分**

Use the pre-registered 20-task stability subset. Pair A and D from the same task and aggregate seeds before choosing the report: select the run whose automatic completeness is the median among its three repeats, breaking ties by run ID. Obtain three distinct AI/CS-background raters per task.

- [ ] **Step 5: 汇总**

Report mean dimension scores, majority A/D/TIE preference, tie rate, ordinal Krippendorff's alpha per dimension, and Spearman correlation between each automatic metric and each human dimension. Preserve missingness rather than imputing a score; formal summary fails if any task has fewer than three valid raters.

- [ ] **Step 6: 验证并提交代码，不提交私有评分**

Run: uv run pytest tests/unit/benchmarks/test_human.py -q

Expected: PASS.

    git add benchmarks/evaluators/human.py tests/unit/benchmarks/test_human.py
    git commit -m "feat: add blinded human evaluation workflow"

### Task 19: 生成结果、失败分析和求职材料

**Files:**

- Create: benchmarks/scripts/render_results.py
- Create: docs/evaluation.md
- Create: docs/results.md
- Generate: docs/assets/results/citation-support-vs-usd.svg
- Generate: docs/assets/results/completeness-vs-search.svg
- Generate: docs/assets/results/abcd-metrics.svg
- Modify: README.md
- Test: tests/cli/test_experiment_commands.py

**Interfaces:** Consumes only hash-verified public summaries and optional aggregate human/external results. Produces deterministic Markdown/SVG portfolio artifacts and README navigation; it never reads raw gold, prompts or provider responses.

- [ ] **Step 1: 写结果渲染红测**

~~~python
def test_renderer_requires_all_three_protocol_sections(summary_path):
    payload = load_json(summary_path)
    payload.pop("planner_policy")
    with pytest.raises(ResultValidationError, match="planner_policy"):
        render_results(payload)


def test_negative_primary_result_is_not_hidden(summary_factory):
    page = render_results(
        summary_factory(
            ranker_primary_ci=(-0.04, -0.01),
            planner_noninferiority=False,
        )
    )
    assert "主假设未成立" in page
    assert "-0.04" in page
    assert "失败分析" in page
~~~

- [ ] **Step 2: 运行红测**

Run: uv run pytest tests/cli/test_experiment_commands.py -q -k render

Expected: FAIL because render_results is missing.

- [ ] **Step 3: 实现只读渲染器**

`render_results` accepts the primary summary/manifest plus the separate Task 17 external directory, verifies both seal/config/lock/artifact hash chains, and writes deterministic Markdown/SVG. It must never copy external metrics into primary confidence intervals. It includes:

- dataset/config/code/model/evaluator versions and evaluation date;
- A/B/C/D table with all quality, evidence, efficiency and failure metrics;
- ranker primary estimate and 95% CI;
- planner non-inferiority result and secondary efficiency results;
- both Pareto planes and bootstrap dominance proportions;
- p50/p95 latency, token, cost and failure breakdown;
- P0 agent reference results and the hash-only evaluator ORACLE ceiling;
- human scores, agreement and correlations when available;
- external 10/20/10 results in a separate section;
- limitations, leaks/re-runs disclosure and negative-result analysis.

Do not expose sealed prompts, private gold, raw provider responses or credentials.

- [ ] **Step 4: 写评测方法文档**

docs/evaluation.md reproduces all commands, process isolation, hardware/model endpoint profile, budget tables, subset policy, metric formulas and statistics. docs/results.md is generated from the frozen summary and starts with a compact factual outcome, not marketing copy.

- [ ] **Step 5: 更新 README 求职入口**

README adds:

    Architecture
    Why Planner and Ranker matter
    Replay quickstart
    Live/local quickstart
    Benchmark protocol
    Results
    Reproduction
    Trade-offs and limitations

Link the design, four implementation plans, evaluation doc, results page and Demo. Embed the two Pareto SVGs and one A/B/C/D chart with accessible alt text.

- [ ] **Step 6: 生成、复核和提交**

Run:

    $groupId = uv run deepresearch experiment config id --config benchmarks/configs/formal.yaml
    $portfolioGroupId = uv run deepresearch experiment config id --config benchmarks/configs/formal-portfolio.yaml
    uv run python -m benchmarks.scripts.render_results --experiment-dir "experiments/$groupId" --external-experiment-dir "experiments/$portfolioGroupId/external" --docs-dir docs
    uv run pytest tests/cli/test_experiment_commands.py -q
    uv run ruff check benchmarks experiments apps tests
    uv run pyright src benchmarks experiments apps
    uv run pytest -q

Expected: all checks pass; renderer output is unchanged on a second run.

Visually inspect all three SVGs and verify labels, axes, legends and color contrast.

    git add benchmarks/scripts/render_results.py docs/evaluation.md docs/results.md docs/assets/results README.md tests/cli/test_experiment_commands.py
    git commit -m "docs: publish reproducible benchmark results"

## Final Acceptance

- [ ] Frozen AI/CS Research 60 contains 30 dev and 30 sealed-test tasks, balanced 10 per category, with verified source/evidence hashes.
- [ ] Agent runtime cannot read gold, rubric, acceptable claims or test annotations; the evaluator runs in a separate path/process boundary.
- [ ] FrozenCorpusSearchProvider is deterministic for arbitrary legal queries and never falls back to Live.
- [ ] Ranker component, Planner policy and 2×2 end-to-end results are separately identified and reproducible.
- [ ] A/B/C/D, P0, three budget presets and repeat/seed agent runs produce immutable Core RunManifests; ORACLE instead produces immutable hash-only `OracleReferenceResult` records plus `EvaluatorReferenceManifest` and never impersonates an agent run.
- [ ] Statistical output aggregates seeds first and uses 10,000 task-level stratified paired bootstrap resamples.
- [ ] Results expose confidence intervals, unsupported claims, failures, cost and both Pareto planes without hiding negative outcomes.
- [ ] Portfolio Full additionally includes pinned external 10/20/10 results and 20-task, three-rater blinded A/D evaluation.

## Execution Handoff

Plan complete. Choose one execution mode before implementation:

1. Subagent-Driven (recommended): use superpowers:subagent-driven-development in this session, one fresh worker and two-stage review per task.
2. Inline Execution: start a separate implementation session with superpowers:executing-plans and run tasks sequentially at the documented checkpoints.

Run Tasks 1–16 for the four-week Core evaluation; Tasks 17–19 are the six-week Portfolio Full extension.
