# Adaptive Planner、Evidence Ranker 与 Research Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在既有 Core Research domain/provider 契约之上，实现可解释、可重放的 P0/P1/P2 Planner、R0/R1/R2 Evidence Ranker、claim-evidence graph、Citation Guard 与受控 LangGraph research graph。

**Architecture:** Planner 通过确定性的 Coverage Ledger、priority、Query Scheduler 和停止策略控制搜索循环；LLM 只负责计划结构化生成、选中信息缺口的查询候选和 claim/evidence 语义判断。Ranker 的 Pass A 负责候选清洗、语义排序和边际效用选择，Pass B 负责 atomic claim、证据关系和引用验证；LangGraph 只保存 ID、预算和决策状态，正文与原始内容留在 Artifact/Evidence Store。

**Tech Stack:** Python 3.12、uv、既有 Core domain/provider Protocol、LangGraph StateGraph/checkpoint、pytest、pytest-asyncio、sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2`、Replay providers、Ruff、Pyright。

**Spec:** [Multi-Agent Deep Research 设计文档](../specs/2026-08-29-multi-agent-deep-research-design.md)

**Predecessor:** [Core Foundation 与 Replay Baseline](./2026-08-29-core-foundation-replay-baseline.md)

## Global Constraints

- 只修改本计划列出的 Planner、Evidence、Workflow 和测试文件；不修改 Core Research plan 负责的共享 domain/provider/runtime 类型。
- 本计划依赖 [Core Foundation 与 Replay Baseline](./2026-08-29-core-foundation-replay-baseline.md) 完成；执行顺序是 Core Task 1–10 及其验收门全部通过 → 本计划 Task 1–12。
- 共享领域类型从 `deepresearch.domain` 导入：`ResearchRequest`、`ResearchPlan`、`SubQuestion`、`InformationNeed`、`FreshnessRequirement`、`CoverageLedgerEntry`、`RunBudget`、`SourceDocument`、`EvidenceSpan`、`HtmlLocator`、`PdfLocator`、`Claim`、`ClaimEvidenceLink`、`RerankScore`、`RunConfig`、`RunResult`、`RunStatus`、`StopReason`。
- Provider request/result 类型从 `deepresearch.providers.types` 导入；`ProviderError` 从 `deepresearch.providers.errors` 导入；async 协议从 `deepresearch.providers.protocols` 导入：`ModelProvider`、`SearchProvider`、`Fetcher`、`Parser`、`Reranker`。任何新模块不得复制这些类或重新声明同名 Protocol。
- `CancellationToken`、`BudgetAccountant`、`ResearchRunner`、`ResourceEstimate`、`BudgetSnapshot` 从 `deepresearch.runtime` 导入并原样消费；`Deadline` 从 `deepresearch.providers.types` 导入，`ResourceUsage` 从 `deepresearch.domain` 导入；本计划不创建替代实现。
- P1FixedPlanner 是 Core FixedPlanner 的协议适配器，R1SimilarityOnly 是 Core SimilarityRanker 的协议适配器；二者必须委托既有实现，不复制 prompt、embedding、cosine 或排序逻辑。
- Core baseline_graph.py 保持 P1+R1 可复现基线不变；本计划创建独立 research_graph.py，并让既有 runner.py 按 RunConfig 选择 baseline 或 optimized graph。
- 对 workflow/state.py 和 workflow/runner.py 的修改只能追加 research-v1 字段/选择分支；必须保留 baseline-v1 行为、ResearchRunner.run 签名和现有 Replay fixture。不得修改 planning/fixed.py、evidence/similarity.py 或 providers/replay.py。
- 所有命令必须通过 `uv run` 执行；质量门使用 `uv run ruff` 和 `uv run pyright`，只支持精确 Python 3.12。
- 所有 Provider 调用均为 async，接收 `deadline` 和 `cancellation_token`，只返回 typed result 或 Core `ProviderError`。
- 所有分数必须是有限浮点数并裁剪到 `[0, 1]`；时间使用带时区 RFC 3339，offset 是规范化 Unicode code point 的左闭右开区间。
- Planner 的公开停止码只允许 `SUFFICIENT`、`PLATEAU`、`BUDGET_EXHAUSTED`、`BLOCKED`；不得保存或展示隐藏思维链。
- Priority 固定为 `importance × evidence_gap × expected_gain / (estimated_cost + 0.05)`；`expected_gain` 和 `estimated_cost` 权重严格采用已批准规格。
- R2 权重严格采用 `0.25/0.20/0.15/0.20/0.10/0.05/-0.15/-0.05`；总分只用于排序，不解释为概率。
- Query Scheduler 的 Unicode NFC、casefold、空白/标点规范化、精确去重和 R1 embedding 语义去重阈值固定为 `0.92`。
- Evidence-first：Writer 只能使用已经存入 Evidence Store 的 Evidence ID；Ranker/Citation Guard 不得现场生成未存档 URL 或引用。
- 缓存 key、idempotency key、artifact/evidence hash 和 checkpoint 行为复用 Core storage contract；LangGraph checkpoint 不是缓存。
- Replay 路径遇到未知 query、未知 model request 或 hash 不匹配必须返回 `REPLAY_MISS`，不得静默 fallback 到 Live provider。
- 除非任务明确写出 `FastAPI` 或 `Streamlit`，本计划不添加服务 API、UI 或部署文件。
- 每个任务先写红测并单独运行，随后实现最小代码、运行绿测、提交独立 commit；实现代码必须遵循本计划列出的签名。

## Core Interface Map（只读依赖）

以下是本计划使用的 Core 契约；具体类定义由 Core Research plan 提交，本计划只导入和调用，不重新定义：

```python
from deepresearch.domain import (
    Claim, ClaimEvidenceLink, CoverageLedgerEntry, EvidenceSpan, HtmlLocator,
    FreshnessRequirement, InformationNeed, PdfLocator, ResearchPlan, ResearchRequest, RerankScore,
    ResourceUsage, RunConfig, RunEvent, RunResult, RunBudget, RunStatus,
    StopReason,
    SourceDocument, SubQuestion,
)
from pydantic import JsonValue
from deepresearch.providers.errors import ProviderError
from deepresearch.providers.protocols import (
    Fetcher, ModelProvider, Parser, Reranker, SearchProvider, TextEmbedder,
)
from deepresearch.providers.types import (
    Deadline, ModelRequest, ModelResult, ParsedDocument, RawDocument,
    SearchHit, StructuredModelResult,
)
from deepresearch.runtime import (
    BudgetAccountant, BudgetSnapshot, CancellationToken, CheckpointRef,
    ResearchRunner, ResourceEstimate,
)
from deepresearch.runtime.checkpoints import checkpoint_serializer
from deepresearch.evidence.similarity import SimilarityRanker
from deepresearch.planning.fixed import FixedPlanner, PlanGenerationError
from deepresearch.workflow.runner import LangGraphResearchRunner
from deepresearch.workflow.state import BaselineBlockedNeed, BaselineState

# Core protocol calls used by this plan; implementations remain in Core.
await model_provider.structured(
    request, output_schema, deadline=deadline,
    cancellation_token=cancellation_token,
)
await search_provider.search(
    query, limit, filters, deadline=deadline,
    cancellation_token=cancellation_token,
)
await fetcher.fetch(
    url, deadline=deadline, cancellation_token=cancellation_token,
)

```

Planner、Ranker、Graph 和 Workflow 只保存 `ResearchPlan`、`CoverageLedgerEntry`、预算快照、状态和 ID；规范化正文、SourceDocument、EvidenceSpan 和完整模型响应由 Core Artifact/Evidence Store 接口持久化。

## Exact File Map

| 文件 | 责任 |
|---|---|
| `src/deepresearch/planning/contracts.py` | Planner 决策、查询候选、Planner state 的本地编排类型；不复制 domain 类型 |
| `src/deepresearch/planning/ledger.py` | Ledger 更新、coverage、独立来源与冲突状态 |
| `src/deepresearch/planning/priority.py` | priority、expected gain、estimated cost 的纯函数 |
| `src/deepresearch/planning/stop.py` | 四个公开停止码及原因 |
| `src/deepresearch/planning/query_scheduler.py` | 查询规范化、精确/语义去重、并发、预算扣账；唯一声明 `QueryBatchResult` |
| `src/deepresearch/planning/planners.py` | P0 ReAct、P1 Fixed Plan、P2 Adaptive Planner |
| `src/deepresearch/evidence/normalize.py` | source family、hash、片段与转载去重 |
| `src/deepresearch/evidence/pass_a.py` | cheap prefilter、语义 rerank、边际效用候选选择；唯一声明 `PassAResult` |
| `src/deepresearch/evidence/features.py` | R2 八项可观察特征、质量/支持 typed observations 与 concrete calculator；唯一声明本任务全部 feature types |
| `src/deepresearch/evidence/rankers.py` | R0/R1/R2、score breakdown 与 feature provenance；唯一声明 `EvidenceRankingResult` |
| `src/deepresearch/evidence/claims.py` | Claim Extractor、Evidence Judge |
| `src/deepresearch/evidence/graph.py` | Claim-Evidence 二部图及一致性验证；唯一声明 `GraphValidationResult` |
| `src/deepresearch/evidence/citation_guard.py` | citation ID、HTML/PDF locator、raw/parsed/excerpt hash 验证；唯一声明 Citation Guard 本地结果与 material resolver protocol |
| `src/deepresearch/workflow/state.py` | 复用 Core checkpoint-safe state 字段；唯一声明 `ResearchState` 与 Planner/`BaselineBlockedNeed` 边界转换函数，不扩 serializer allow-list |
| `src/deepresearch/workflow/research_graph.py` | 新增 Planner/Ranker/Graph 节点、typed claim resolution 和条件路由；唯一声明 `ResearchGraphDependencies`，保留 Core baseline |
| `src/deepresearch/workflow/runner.py` | 通过 Core `ResearchRunner` port 选择并运行 baseline/optimized 图 |
| `tests/unit/planning/test_contracts.py` | Planner 本地契约 |
| `tests/unit/planning/test_ledger.py` | Coverage Ledger |
| `tests/unit/planning/test_priority.py` | priority 公式 |
| `tests/unit/planning/test_budget_integration.py` | Core BudgetAccountant 消费 |
| `tests/unit/planning/test_stop.py` | 四类停止码 |
| `tests/unit/planning/test_query_scheduler.py` | query 规范化/去重/调度 |
| `tests/contracts/test_query_scheduler_contract.py` | Scheduler async contract |
| `tests/unit/planning/test_planners.py` | P0/P1/P2 |
| `tests/unit/evidence/test_normalize.py` | source family 与去重 |
| `tests/unit/evidence/test_pass_a.py` | Pass A |
| `tests/unit/evidence/test_features.py` | R2 八特征公式、typed observations 与 concrete calculator |
| `tests/unit/evidence/test_rankers.py` | R0/R1/R2 |
| `tests/unit/evidence/test_ranker_boundaries.py` | score 边界 |
| `tests/unit/evidence/test_claims.py` | Claim Extractor/Judge |
| `tests/unit/evidence/test_claim_graph.py` | Claim-Evidence graph |
| `tests/unit/evidence/test_citation_guard.py` | 引用/hash |
| `tests/integration/replay/test_planner_paths.py` | Planner Replay 路径 |
| `tests/integration/replay/test_research_graph.py` | research-v1 图 |
| `tests/integration/replay/test_checkpoint_idempotency.py` | 恢复与幂等 |
| `tests/integration/replay/test_replay_provider_paths.py` | Core Replay 消费 |
| `tests/integration/replay/test_stop_paths.py` | 停止分支 |
| `tests/integration/replay/test_conflict_research.py` | 冲突补搜 |
| `tests/integration/replay/test_partial_results.py` | 部分报告 |
| `tests/fixtures/security/invalid_citations.md` | unknown citation、locator/hash 篡改样例 |
| `tests/fixtures/replay/planner_ranker/snapshot.json` | Planner/Ranker Replay bundle 元数据 |
| `tests/fixtures/replay/planner_ranker/search.jsonl` | 确定性搜索记录 |
| `tests/fixtures/replay/planner_ranker/documents.jsonl` | HTML/PDF fetch 与 parse 记录 |
| `tests/fixtures/replay/planner_ranker/model_responses.jsonl` | plan/query/judge/write 结构化模型记录 |
| `tests/fixtures/replay/planner_ranker/embeddings.jsonl` | Query Scheduler、R1、R2 共用的 384 维 embedding 记录 |
| `tests/fixtures/replay/planner_ranker/manifest.sha256` | bundle 文件哈希清单 |
| `tests/fixtures/replay/still_unsupported_after_targeted/` | 一轮定向补搜后仍需 claim resolution 的完整 Replay bundle |
| `tests/fixtures/replay/stop_plateau/` | 两轮低 marginal gain 的完整 Replay bundle |
| `tests/fixtures/replay/stop_budget_exhausted/` | hard budget stop 的完整 Replay bundle |
| `tests/fixtures/replay/stop_blocked/` | 首次失败→替代策略→typed terminal BLOCKED 的完整 Replay bundle |
| `tests/fixtures/replay/conflict_research/` | 一轮冲突定向补搜的完整 Replay bundle |

---

### Task 1: Planner 编排契约与不可变 State

**Files:**
- Create: `src/deepresearch/planning/contracts.py`
- Create: `tests/unit/planning/test_contracts.py`

**Consumes:** Core `ResearchPlan`、`BudgetSnapshot`；Task 2 将提供 `CoverageLedger`，Task 3 将提供 `BlockedNeed`/`StopDecision`。`contracts.py` 使用 `from __future__ import annotations`，并仅在 `TYPE_CHECKING` 分支从 `planning.ledger`/`planning.stop` 前向导入本地类型，避免运行时循环导入。

**Produces:**

```python
@dataclass(frozen=True)
class QueryCandidate:
    subquestion_id: str
    information_need_id: str
    query: str
    priority_hint: float
    estimated_tokens: int
    estimated_search_calls: int
    estimated_seconds: float

@dataclass(frozen=True)
class PlannerState:
    plan: ResearchPlan
    ledger: "CoverageLedger"
    budget_snapshot: BudgetSnapshot
    blocked_needs: tuple["BlockedNeed", ...]
    round_index: int
    recent_marginal_gains: tuple[float, ...]
    query_history: tuple[str, ...]

@dataclass(frozen=True)
class PlannerDecision:
    kind: Literal["SEARCH", "STOP"]
    subquestion_id: str | None
    candidates: tuple[QueryCandidate, ...]
    stop: "StopDecision | None"
    decision_code: str
```

- [ ] **Step 1: Write the failing test**

```python
def test_query_candidate_rejects_negative_costs():
    with pytest.raises(ValueError, match="estimated_tokens"):
        QueryCandidate("sq-1", "need-1", "q", 0.5, -1, 1, 1.0)

def test_planner_decision_stop_cannot_contain_search_candidates():
    with pytest.raises(ValueError, match="STOP"):
        PlannerDecision("STOP", None, (candidate("q"),), stop_decision(), "STOP")

def test_planner_state_rejects_none_ledger():
    with pytest.raises(ValueError, match="ledger"):
        PlannerState(
            plan=make_plan(), ledger=None, budget_snapshot=budget_ok(),
            blocked_needs=(), round_index=0,
            recent_marginal_gains=(), query_history=(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/planning/test_contracts.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'deepresearch.planning.contracts'`.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class QueryCandidate:
    subquestion_id: str
    information_need_id: str
    query: str
    priority_hint: float
    estimated_tokens: int
    estimated_search_calls: int
    estimated_seconds: float

    def __post_init__(self) -> None:
        if not self.subquestion_id or not self.information_need_id:
            raise ValueError("query candidate IDs are required")
        if self.estimated_tokens < 0:
            raise ValueError("estimated_tokens must be non-negative")
        if self.estimated_search_calls < 0:
            raise ValueError("estimated_search_calls must be non-negative")
        if self.estimated_seconds < 0:
            raise ValueError("estimated_seconds must be non-negative")


@dataclass(frozen=True)
class PlannerState:
    plan: ResearchPlan
    ledger: "CoverageLedger"
    budget_snapshot: BudgetSnapshot
    blocked_needs: tuple["BlockedNeed", ...]
    round_index: int
    recent_marginal_gains: tuple[float, ...]
    query_history: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ledger is None:
            raise ValueError("ledger must be a CoverageLedger")
        if self.blocked_needs is None:
            raise ValueError("blocked_needs must be a tuple")


@dataclass(frozen=True)
class PlannerDecision:
    kind: Literal["SEARCH", "STOP"]
    subquestion_id: str | None
    candidates: tuple[QueryCandidate, ...]
    stop: "StopDecision | None"
    decision_code: str

    def __post_init__(self) -> None:
        if self.kind == "STOP" and (
            self.subquestion_id is not None or self.candidates or self.stop is None
        ):
            raise ValueError("STOP decision cannot contain search fields")
        if self.kind == "SEARCH" and (
            self.subquestion_id is None or not self.candidates or self.stop is not None
        ):
            raise ValueError("SEARCH decision requires search fields")
```

Implement the frozen dataclasses around this validation, importing `ResearchPlan` and `BudgetSnapshot` from Core rather than defining replacements. `PlannerState.ledger` is always non-null and `blocked_needs` is always a tuple (empty means no terminally exhausted need). Before Tasks 2–3 exist, Task 1 only validates forward-referenced fields; afterward every state factory must pass `CoverageLedger.empty_for(plan)` plus `blocked_needs=()`, including P0. P0 ignores and preserves those fields instead of using a second state shape.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/planning/test_contracts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/planning/contracts.py tests/unit/planning/test_contracts.py
git commit -m "feat: add planner orchestration contracts"
```

### Task 2: Coverage Ledger、Priority 与预算编排

**Files:**
- Create: `src/deepresearch/planning/ledger.py`
- Create: `src/deepresearch/planning/priority.py`
- Create: `tests/unit/planning/test_ledger.py`
- Create: `tests/unit/planning/test_priority.py`
- Create: `tests/unit/planning/test_budget_integration.py`

**Consumes:** Core domain `ResearchPlan`、`SubQuestion`、`CoverageLedgerEntry`、`EvidenceSpan`、`ClaimEvidenceLink`、`SourceDocument`、`RunBudget`、`ResourceUsage`；`deepresearch.runtime.BudgetAccountant`、`ResourceEstimate`。

**Produces:**

```python
class CoverageLedger:
    def __init__(
        self,
        plan: ResearchPlan,
        entries: Mapping[str, CoverageLedgerEntry] | None = None,
    ) -> None: pass

    @classmethod
    def empty_for(cls, plan: ResearchPlan) -> "CoverageLedger": pass

    def get(self, subquestion_id: str) -> CoverageLedgerEntry: pass
    def entries(self) -> tuple[CoverageLedgerEntry, ...]: pass
    def weighted_coverage(self) -> float: pass
    def replace(self, entry: CoverageLedgerEntry) -> "CoverageLedger": pass

def update_coverage(
    ledger: CoverageLedger,
    subquestion_id: str,
    *,
    selected_evidence: Sequence[EvidenceSpan],
    links: Sequence[ClaimEvidenceLink],
    source_documents: Mapping[str, SourceDocument],
    marginal_gain: float,
    decision_code: str,
) -> CoverageLedger: pass

def compute_priority(
    *, importance: float,
    coverage_score: float,
    new_source_need: float,
    conflict_resolution_need: float,
    token_fraction: float,
    search_fraction: float,
    time_fraction: float,
    recent_gain: float = 0.5,
    historical_success: float = 0.5,
) -> float: pass

```

`update_coverage` 按 `source_family_id` 计独立来源；`contradict` link 进入 unresolved conflict；所有 ledger score 裁剪到 `[0,1]`。预算行为由 Core `deepresearch.runtime.BudgetAccountant` 唯一实现；本任务只增加 Planner/Query Scheduler 对其 exact `reserve(estimate, node, idempotency_key)`、`settle(reservation, actual)`、`release(reservation)` 的消费测试，不创建预算替代类。

- [ ] **Step 1: Write the failing test**

```python
def test_coverage_counts_unique_source_families():
    updated = update_coverage(
        ledger, "sq-1",
        selected_evidence=[span("e-1"), span("e-2")],
        links=[],
        source_documents={"s-1": source("s-1", "f-1"), "s-2": source("s-2", "f-2")},
        marginal_gain=0.3, decision_code="RANKED",
    )
    assert updated.get("sq-1").independent_source_count == 2

def test_empty_ledger_initializes_every_planned_subquestion():
    plan = make_plan(subquestion_ids=("sq-1", "sq-2"))
    ledger = CoverageLedger.empty_for(plan)
    assert tuple(entry.subquestion_id for entry in ledger.entries()) == ("sq-1", "sq-2")
    assert all(entry.coverage_score == 0 for entry in ledger.entries())

def test_priority_uses_spec_weights_and_epsilon():
    assert compute_priority(
        importance=0.8, coverage_score=0.2,
        recent_gain=1.0, new_source_need=0.5,
        conflict_resolution_need=0.5, historical_success=0.5,
        token_fraction=0.5, search_fraction=0.5, time_fraction=0.5,
    ) == pytest.approx(0.8 * 0.8 * 0.575 / 0.55)

def test_priority_uses_neutral_defaults_without_history():
    result = compute_priority(
        importance=0.8, coverage_score=0.2,
        new_source_need=0.5, conflict_resolution_need=0.5,
        token_fraction=0.5, search_fraction=0.5, time_fraction=0.5,
    )
    neutral_gain = 0.40 * 0.5 + 0.25 * 0.5 + 0.20 * 0.5 + 0.15 * 0.5
    assert result == pytest.approx(0.8 * 0.8 * neutral_gain / 0.55)

def test_query_planner_consumes_core_budget_accountant_once():
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    reservation = accountant.reserve(
        ResourceEstimate(search_calls=1, pages=0, tokens=100,
                         wall_seconds=1, cost_usd=Decimal("0")),
        node="Tool", idempotency_key="search:q-1",
    )
    first = accountant.settle(reservation, actual=ResourceUsage.zero())
    second = accountant.settle(reservation, actual=ResourceUsage.zero())
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/planning/test_ledger.py tests/unit/planning/test_priority.py tests/unit/planning/test_budget_integration.py -q`

Expected: FAIL with missing `CoverageLedger`, `compute_priority`, or Planner budget consumption test helpers; Core `BudgetAccountant` remains the only implementation target.

- [ ] **Step 3: Write minimal implementation**

```python
@classmethod
def empty_for(cls, plan: ResearchPlan) -> "CoverageLedger":
    return cls(
        plan,
        {
            subquestion.id: CoverageLedgerEntry(
                subquestion_id=subquestion.id,
                coverage_score=0.0,
                independent_source_count=0,
                unresolved_conflict_ids=(),
                uncertainty_score=1.0,
                last_marginal_gain=0.0,
                evidence_ids=(),
                attempt_count=0,
                last_decision_code="NOT_ATTEMPTED",
            )
            for subquestion in plan.subquestions
        },
    )


PRIORITY_EPSILON = 0.05


expected_gain = (
    0.40 * clip(recent_gain)
    + 0.25 * clip(new_source_need)
    + 0.20 * clip(conflict_resolution_need)
    + 0.15 * clip(historical_success)
)
estimated_cost = clip(
    0.50 * clip(token_fraction)
    + 0.30 * clip(search_fraction)
    + 0.20 * clip(time_fraction)
)
return (
    clip(importance)
    * (1.0 - clip(coverage_score))
    * expected_gain
    / (estimated_cost + PRIORITY_EPSILON)
)
```

Define `PRIORITY_EPSILON = 0.05` in `priority.py`; it is not caller-configurable. Implement immutable ledger replacement and weighted coverage by `SubQuestion.importance`; `CoverageLedger.__init__` validates that entries contain exactly one item per planned subquestion, and `empty_for` is the sole zero-state constructor used by all Planner variants. `compute_priority` owns the no-history defaults `recent_gain=0.5` and `historical_success=0.5`, so callers never invent them. Wire Planner-side estimates to the imported Core `BudgetAccountant` without adding a second budget module or duplicating reservation/settlement logic.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/planning/test_ledger.py tests/unit/planning/test_priority.py tests/unit/planning/test_budget_integration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/planning/ledger.py src/deepresearch/planning/priority.py tests/unit/planning/test_ledger.py tests/unit/planning/test_priority.py tests/unit/planning/test_budget_integration.py
git commit -m "feat: add coverage ledger priority and budget accounting"
```

### Task 3: 四种公开停止码

**Files:**
- Create: `src/deepresearch/planning/stop.py`
- Create: `tests/unit/planning/test_stop.py`

**Consumes:** Task 1 `PlannerState`、Task 2 `CoverageLedger`，Core `BudgetSnapshot` 与 `SubQuestion.evidence_requirements`。

**Produces:**

```python
class StopCode(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PLATEAU = "PLATEAU"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    BLOCKED = "BLOCKED"

@dataclass(frozen=True)
class BlockedNeed:
    need_id: str
    required_source_unavailable: bool
    alternative_strategies_exhausted: bool
    retries_used: int
    max_retries: int

    @property
    def terminal(self) -> bool: pass

@dataclass(frozen=True)
class StopDecision:
    code: StopCode
    reasons: tuple[str, ...]
    uncovered_information_needs: tuple[str, ...]
    is_partial: bool

def evaluate_stop(
    state: PlannerState,
    budget_snapshot: BudgetSnapshot,
    *,
    blocked_needs: Collection[BlockedNeed] = (),
) -> StopDecision | None: pass
```

检查顺序固定为 budget、terminal blocked、sufficient、plateau，避免预算耗尽被误报为充分。`BlockedNeed.terminal` 仅在 required source 已确认不可用、替代检索策略已耗尽且 `retries_used >= max_retries` 时为真；单次 query/provider 失败不能触发 `BLOCKED`。`SUFFICIENT` 需要关键子问题 coverage ≥ `0.85`、独立来源要求满足、整体加权 coverage ≥ `0.80`、无高优先级冲突；默认独立来源数为 2，一手材料自述可单独满足。`PLATEAU` 需要连续两轮 marginal gain `< 0.05`。

- [ ] **Step 1: Write the failing test**

```python
def test_sufficient_requires_coverage_sources_and_no_high_priority_conflict():
    result = evaluate_stop(state_with_coverage(0.82, key=0.90, sources=2), budget_ok())
    assert result.code is StopCode.SUFFICIENT

def test_budget_exhausted_precedes_plateau():
    result = evaluate_stop(state_with_gains(0.04, 0.03), budget_exhausted())
    assert result.code is StopCode.BUDGET_EXHAUSTED

def test_first_failed_query_does_not_block():
    first_failure = BlockedNeed(
        need_id="need-1", required_source_unavailable=True,
        alternative_strategies_exhausted=False, retries_used=1, max_retries=2,
    )
    result = evaluate_stop(
        state_with_uncovered("need-1"), budget_ok(), blocked_needs=[first_failure],
    )
    assert result is None

def test_blocked_requires_unavailable_source_and_exhausted_alternatives_and_retries():
    exhausted = BlockedNeed(
        need_id="need-1", required_source_unavailable=True,
        alternative_strategies_exhausted=True, retries_used=2, max_retries=2,
    )
    result = evaluate_stop(
        state_with_uncovered("need-1"), budget_ok(), blocked_needs=[exhausted],
    )
    assert result.code is StopCode.BLOCKED
    assert result.is_partial is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/planning/test_stop.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'deepresearch.planning.stop'`.

- [ ] **Step 3: Write minimal implementation**

```python
def __post_init__(self) -> None:
    if not self.need_id:
        raise ValueError("need_id is required")
    if self.max_retries < 0 or not 0 <= self.retries_used <= self.max_retries:
        raise ValueError("retry counters are invalid")


@property
def terminal(self) -> bool:
    return (
        self.required_source_unavailable
        and self.alternative_strategies_exhausted
        and self.retries_used >= self.max_retries
    )


if budget_snapshot.exhausted:
    return StopDecision(StopCode.BUDGET_EXHAUSTED, ("hard_budget_limit",), uncovered, True)
terminal_blocked = tuple(sorted(item.need_id for item in blocked_needs if item.terminal))
if terminal_blocked:
    return StopDecision(
        StopCode.BLOCKED,
        ("required_source_and_alternatives_exhausted",),
        terminal_blocked,
        True,
    )
if ledger_meets_sufficient(state.ledger) and not high_priority_conflicts(state.ledger):
    return StopDecision(StopCode.SUFFICIENT, ("coverage_thresholds_met",), (), False)
if len(state.recent_marginal_gains) >= 2 and all(x < 0.05 for x in state.recent_marginal_gains[-2:]):
    return StopDecision(StopCode.PLATEAU, ("two_low_gain_rounds",), uncovered, True)
return None
```

Validate non-empty `need_id`、`0 <= retries_used <= max_retries` and `max_retries >= 0`; implement exact thresholds and structured reasons. The workflow creates/updates `BlockedNeed` only from public provider error codes and retry/strategy counters. Never return hidden rationale or free-form chain-of-thought.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/planning/test_stop.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/planning/stop.py tests/unit/planning/test_stop.py
git commit -m "feat: add public planner stop decisions"
```

### Task 4: Query Scheduler

**Files:**
- Create: `src/deepresearch/planning/query_scheduler.py`
- Create: `tests/unit/planning/test_query_scheduler.py`
- Create: `tests/contracts/test_query_scheduler_contract.py`

**Consumes:** Task 1 `QueryCandidate`，Task 2 priority estimates，Core `SearchProvider`、`SearchHit`、`TextEmbedder`、`ResourceUsage`；`deepresearch.runtime.BudgetAccountant`、`CancellationToken`、`ResourceEstimate`；canonical `pydantic.JsonValue`。

**Produces:**

```python
@dataclass(frozen=True)
class QueryBatchResult:
    results: tuple[tuple[str, tuple[SearchHit, ...]], ...]
    executed_queries: int
    skipped_queries: tuple[str, ...]
    skipped_reason: Literal["BUDGET_EXHAUSTED", "CANCELLED"] | None

class QueryScheduler:
    def __init__(
        self, *, embedder: TextEmbedder, budget: BudgetAccountant,
    ) -> None: pass

    def normalize(self, query: str) -> str: pass

    async def dedupe(
        self, candidates: Sequence[QueryCandidate], *, deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[QueryCandidate]: pass

    async def dispatch(
        self, candidates: Sequence[QueryCandidate], provider: SearchProvider,
        *, limit: int, filters: Mapping[str, JsonValue] | None,
        max_concurrency: int, deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> QueryBatchResult: pass
```

规范化为 Unicode NFC、casefold、空白折叠、标点规范化；先精确去重，再以 R1 注入的同一个 `TextEmbedder` 做 cosine 去重。相似度 ≥ `0.92` 时仅保留 `priority_hint` 最高者；dispatch 使用 semaphore、reservation、取消检查和稳定 query idempotency key。

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_scheduler_semantic_dedupe_keeps_highest_priority():
    scheduler = QueryScheduler(
        embedder=FakeEmbedder([[1, 0], [0.999, 0.02]]),
        budget=budget_with_one_search_call(),
    )
    result = await scheduler.dedupe(
        [candidate("q-low", 0.2), candidate("q-high", 0.9)],
        deadline=100, cancellation_token=token(),
    )
    assert [x.query for x in result] == ["q-high"]

@pytest.mark.asyncio
async def test_dispatch_deducts_budget_before_search():
    result = await scheduler.dispatch(
        [candidate("q-1"), candidate("q-2")], provider,
        limit=5, filters=None, max_concurrency=2,
        deadline=100, cancellation_token=token(),
    )
    assert result.executed_queries == 1
    assert result.skipped_reason == "BUDGET_EXHAUSTED"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/planning/test_query_scheduler.py tests/contracts/test_query_scheduler_contract.py -q`

Expected: FAIL with missing `QueryScheduler` and contract methods.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class QueryBatchResult:
    results: tuple[tuple[str, tuple[SearchHit, ...]], ...]
    executed_queries: int
    skipped_queries: tuple[str, ...]
    skipped_reason: Literal["BUDGET_EXHAUSTED", "CANCELLED"] | None


def normalize(self, query: str) -> str:
    text = unicodedata.normalize("NFC", query).casefold()
    text = collapse_whitespace(text)
    return normalize_punctuation(text)

async def dedupe(self, candidates, *, deadline, cancellation_token):
    exact = keep_highest_priority_by_normalized_query(candidates, self.normalize)
    vectors = await self.embedder.embed([x.query for x in exact], deadline=deadline,
                                        cancellation_token=cancellation_token)
    return keep_non_duplicate_vectors(exact, vectors, threshold=0.92)
```

Declare `QueryBatchResult` only in `planning/query_scheduler.py`; import `SearchHit` from `deepresearch.providers.types` and `Literal`/`dataclass` from the standard library. Implement normalization/dedupe deterministically, inject Core `TextEmbedder` (the same instance used by R1) and `BudgetAccountant`, and call Core `BudgetAccountant.reserve(estimate, node="Tool", idempotency_key=...)`, `settle(reservation, actual)`, and `release(reservation)` methods. Search/Fetch never invent non-canonical budget node labels. `results` is sorted by normalized query, `executed_queries == len(results)`, and every unexecuted normalized query appears once in `skipped_queries`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/planning/test_query_scheduler.py tests/contracts/test_query_scheduler_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/planning/query_scheduler.py tests/unit/planning/test_query_scheduler.py tests/contracts/test_query_scheduler_contract.py
git commit -m "feat: add deterministic query scheduler"
```

### Task 5: P0、P1、P2 Planner

**Files:**
- Create: `src/deepresearch/planning/planners.py`
- Create: `tests/unit/planning/test_planners.py`
- Create: `tests/integration/replay/test_planner_paths.py`

**Consumes:** Task 1 `QueryCandidate`/`PlannerState`/`PlannerDecision`，Tasks 2–4 ledger、priority、stop、scheduler，Core `FixedPlanner`、`ModelProvider`、`ResearchPlan`、`SubQuestion`。

**Produces:**

```python
class PlannerInvariantError(RuntimeError):
    code: Literal["P1_NO_TARGET_WITHOUT_TYPED_STOP_EVIDENCE"] = (
        "P1_NO_TARGET_WITHOUT_TYPED_STOP_EVIDENCE"
    )

class Planner(Protocol):
    variant: Literal["P0", "P1", "P2"]

    async def next_action(
        self, state: PlannerState, *, deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> PlannerDecision: pass

class P0ReActPlanner:
    variant = "P0"

    def __init__(self, *, model_provider: ModelProvider) -> None: pass

class P1FixedPlanner:
    variant = "P1"

    def __init__(self, *, delegate: FixedPlanner) -> None: pass

class P2AdaptivePlanner:
    variant = "P2"

    def __init__(
        self, *, query_scheduler: QueryScheduler,
        model_provider: ModelProvider,
    ) -> None: pass

    async def generate_queries(
        self, target: SubQuestion, state: PlannerState, *,
        deadline: Deadline, cancellation_token: CancellationToken,
    ) -> tuple[QueryCandidate, ...]: pass
```

P0 不读取或更新 Ledger 的 coverage 值，以 `ModelProvider.structured` 产生公开 observation/action code，不保存隐藏思维链。P1 接收已经由 workflow `Plan` 节点生成的 `PlannerState.plan`，从不调用 `FixedPlanner.create_plan`，只调用 Core exact `FixedPlanner.queries_for(plan_id=state.plan.plan_id, subquestion=target, *, deadline, cancellation_token)` 并把字符串结果适配成 `QueryCandidate`；`search_depth` 只属于 Core delegate 的构造参数。P2 每轮先评估停止条件，再按 priority 选择最高价值缺口；仅对关键未覆盖、单一来源、高优先级冲突、时效不符或换策略重试的缺口调用 `ModelProvider.structured` 生成查询候选。

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_p1_does_not_replan_after_initial_plan():
    delegate = FakeFixedPlanner(search_depth=2, queries=("q-1", "q-2"))
    planner = P1FixedPlanner(delegate=delegate)
    plan = make_plan()
    initial = state(plan=plan, ledger=CoverageLedger.empty_for(plan), round_index=0)
    later = replace(initial, round_index=2)
    first = await planner.next_action(initial, deadline=100, cancellation_token=token())
    second = await planner.next_action(later, deadline=100, cancellation_token=token())
    assert second.candidates == first.candidates
    assert delegate.create_plan_calls == 0
    assert delegate.queries_for_calls == 2
    assert delegate.plan_ids == (plan.plan_id, plan.plan_id)

@pytest.mark.asyncio
async def test_p1_fixed_exhaustion_uses_typed_blocked_not_plateau():
    planner = P1FixedPlanner(delegate=FakeFixedPlanner(search_depth=2, queries=()))
    exhausted = BlockedNeed(
        need_id="need-1", required_source_unavailable=True,
        alternative_strategies_exhausted=True, retries_used=2, max_retries=2,
    )
    result = await planner.next_action(
        state_with_all_attempted(blocked_needs=(exhausted,)),
        deadline=100, cancellation_token=token(),
    )
    assert result.stop.code is StopCode.BLOCKED

@pytest.mark.asyncio
async def test_p1_targetless_state_without_stop_evidence_is_invariant_error():
    planner = P1FixedPlanner(delegate=FakeFixedPlanner(search_depth=2, queries=()))
    with pytest.raises(PlannerInvariantError) as error:
        await planner.next_action(
            state_with_all_attempted(blocked_needs=()),
            deadline=100, cancellation_token=token(),
        )
    assert error.value.code == "P1_NO_TARGET_WITHOUT_TYPED_STOP_EVIDENCE"

@pytest.mark.asyncio
async def test_p2_selects_highest_priority_uncovered_subquestion():
    result = await P2AdaptivePlanner(
        query_scheduler=fake_scheduler,
        model_provider=fake_model_provider,
    ).next_action(
        state_with_coverages({"sq-1": 0.2, "sq-2": 0.8}),
        deadline=100, cancellation_token=token(),
    )
    assert result.subquestion_id == "sq-1"

@pytest.mark.asyncio
async def test_p0_never_requires_or_updates_coverage_ledger():
    plan = make_plan()
    initial = state(plan=plan, ledger=CoverageLedger.empty_for(plan))
    before = initial.ledger.entries()
    result = await P0ReActPlanner(model_provider=fake_model_provider).next_action(
        initial, deadline=100, cancellation_token=token(),
    )
    assert result.decision_code == "REACT_REFERENCE"
    assert initial.ledger.entries() == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/planning/test_planners.py tests/integration/replay/test_planner_paths.py -q`

Expected: FAIL with missing planner classes and policy methods.

- [ ] **Step 3: Write minimal implementation**

```python
class P1FixedPlanner:
    variant = "P1"

    def __init__(self, *, delegate: FixedPlanner) -> None:
        self.delegate = delegate

    async def next_action(
        self,
        state: PlannerState,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> PlannerDecision:
        stop = evaluate_stop(
            state, state.budget_snapshot, blocked_needs=state.blocked_needs,
        )
        if stop is not None:
            return PlannerDecision("STOP", None, (), stop, stop.code.value)
        target = next((
            subquestion
            for subquestion in state.plan.subquestions
            if state.ledger.get(subquestion.id).attempt_count == 0
        ), None)
        if target is None:
            raise PlannerInvariantError("fixed targets exhausted without typed stop evidence")
        queries = await self.delegate.queries_for(
            plan_id=state.plan.plan_id, subquestion=target,
            deadline=deadline, cancellation_token=cancellation_token,
        )
        needs = target.information_needs
        candidates = tuple(
            QueryCandidate(
                subquestion_id=target.id,
                information_need_id=needs[index % len(needs)].need_id,
                query=query,
                priority_hint=target.importance,
                estimated_tokens=256,
                estimated_search_calls=1,
                estimated_seconds=5.0,
            )
            for index, query in enumerate(queries)
        )
        return PlannerDecision(
            "SEARCH", target.id, candidates, None, "P1_FIXED_QUERIES",
        )


class P2AdaptivePlanner:
    variant = "P2"

    async def next_action(self, state, *, deadline, cancellation_token):
        stop = evaluate_stop(
            state, state.budget_snapshot, blocked_needs=state.blocked_needs,
        )
        if stop is not None:
            return PlannerDecision("STOP", None, (), stop, stop.code.value)
        target = max_uncovered_by_priority(state.ledger, state.plan)
        candidates = await self.generate_queries(
            target, state, deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return PlannerDecision(
            "SEARCH", target.id, candidates, None, "P2_INCREMENTAL_REPLAN",
        )
```

Implement the shared Planner protocol, deterministic P1 adapter, P2 incremental replan, and P0 structured-action adapter. Import `FixedPlanner` from `deepresearch.planning.fixed`; `P1FixedPlanner` delegates query generation to `queries_for` and adds only target/`PlannerDecision` adaptation. The workflow `Plan` node calls the configured initial generator once before constructing `PlannerState`; `P1FixedPlanner.next_action` therefore never calls `create_plan`. P1 and P2 both route through `evaluate_stop(state, state.budget_snapshot, blocked_needs=state.blocked_needs)`. Fixed depth exhaustion is never relabeled `PLATEAU`: terminal typed exhaustion yields `BLOCKED`, two measured low-gain rounds yield `PLATEAU`, and a targetless state satisfying neither invariant raises the public-safe `P1_NO_TARGET_WITHOUT_TYPED_STOP_EVIDENCE` workflow error for `PersistResults` rather than inventing a stop code. Do not copy the Core prompt or regenerate the entire plan after every round.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/planning/test_planners.py tests/integration/replay/test_planner_paths.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/planning/planners.py tests/unit/planning/test_planners.py tests/integration/replay/test_planner_paths.py
git commit -m "feat: implement P0 P1 and P2 planner policies"
```

### Task 6: Evidence Normalization 与 Pass A

**Files:**
- Create: `src/deepresearch/evidence/normalize.py`
- Create: `src/deepresearch/evidence/pass_a.py`
- Create: `tests/unit/evidence/test_normalize.py`
- Create: `tests/unit/evidence/test_pass_a.py`

**Consumes:** Core `ParsedDocument`、`SourceDocument`、`EvidenceSpan`、`SearchHit`、`SubQuestion`。

**Produces:**

```python
@dataclass(frozen=True)
class EvidenceCandidate:
    evidence: EvidenceSpan
    source: SourceDocument
    search_rank: int
    source_family_id: str

@dataclass(frozen=True)
class PassAResult:
    selected: tuple[EvidenceCandidate, ...]
    rejected_evidence_ids: tuple[str, ...]
    used_context_tokens: int

class EvidenceNormalizer:
    def assign_source_families(
        self, sources: Sequence[SourceDocument],
    ) -> Mapping[str, str]: pass

    def dedupe(
        self, candidates: Sequence[EvidenceCandidate],
    ) -> list[EvidenceCandidate]: pass

class PassASelector:
    async def select(
        self, subquestion: SubQuestion,
        candidates: Sequence[EvidenceCandidate], *, context_budget: int,
        deadline: Deadline, cancellation_token: CancellationToken,
    ) -> PassAResult: pass
```

Source family 规则严格为：canonical URL 或 parsed content hash 相同直接合并；否则 SimHash 汉明距离 ≤ 3 且规范化标题相似度 ≥ 0.90 才合并。Pass A 顺序为 normalize/dedupe、keyword/vector/语言/日期/长度/解析质量 prefilter、semantic rerank、context budget 内的 marginal utility selection；冲突来源不能仅因冲突被删除。

- [ ] **Step 1: Write the failing test**

```python
def test_same_parsed_hash_assigns_same_source_family():
    families = normalizer.assign_source_families([
        source("s-1", parsed_hash="h"), source("s-2", parsed_hash="h")
    ])
    assert families["s-1"] == families["s-2"]

@pytest.mark.asyncio
async def test_pass_a_retains_independent_conflicting_evidence():
    result = await selector.select(subquestion, [supporting_candidate, conflicting_candidate],
                                   context_budget=1000, deadline=100,
                                   cancellation_token=token())
    assert {x.evidence.evidence_id for x in result.selected} == {"e-support", "e-conflict"}

def test_dedupe_keeps_distinct_spans_from_same_source_family():
    first = candidate("e-1", source_id="s-1", parsed_hash="same-doc", excerpt="route A")
    second = candidate("e-2", source_id="s-2", parsed_hash="same-doc", excerpt="route B")
    kept = normalizer.dedupe([first, second])
    assert {item.evidence.evidence_id for item in kept} == {"e-1", "e-2"}
    assert len({item.source_family_id for item in kept}) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/evidence/test_normalize.py tests/unit/evidence/test_pass_a.py -q`

Expected: FAIL with missing normalizer and selector implementations.

- [ ] **Step 3: Write minimal implementation**

```python
def dedupe(self, candidates):
    families = self.assign_source_families([x.source for x in candidates])
    seen_exact_spans = set()
    kept = []
    for item in sorted(candidates, key=lambda x: (x.search_rank, x.evidence.evidence_id)):
        family = families[item.source.source_id]
        exact_key = (family, item.evidence.excerpt_hash)
        if exact_key in seen_exact_spans:
            continue
        kept.append(replace(item, source_family_id=family))
        seen_exact_spans.add(exact_key)
    return kept
```

Declare `EvidenceCandidate` only in `evidence/normalize.py` and `PassAResult` only in `evidence/pass_a.py`; `pass_a.py` imports `EvidenceCandidate` from the former and the Core domain/provider types from their canonical modules. Implement deterministic family assignment and Pass A filtering/selection. Source family is an independence feature, not a deletion key: remove exact repeated spans and proven near-duplicate spans only, retain distinct locators/excerpts (including conflicts) from the same family. Preserve source family and evidence IDs in `PassAResult`, sort `selected` deterministically, and validate `used_context_tokens >= 0`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/evidence/test_normalize.py tests/unit/evidence/test_pass_a.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/evidence/normalize.py src/deepresearch/evidence/pass_a.py tests/unit/evidence
git commit -m "feat: add evidence normalization and pass-a selection"
```

### Task 7: R0、R1、R2 Ranker 与 score breakdown

**Files:**
- Create: `src/deepresearch/evidence/features.py`
- Create: `src/deepresearch/evidence/rankers.py`
- Create: `tests/unit/evidence/test_features.py`
- Create: `tests/unit/evidence/test_rankers.py`
- Create: `tests/unit/evidence/test_ranker_boundaries.py`

**Consumes:** Task 6 `EvidenceCandidate`，Core `InformationNeed`、`SubQuestion`、`RerankScore`、`EvidenceSpan`、`SourceDocument`、`Reranker`、`ModelProvider`，Task 4 注入的同一 `TextEmbedder`。`evaluation_time` 是 run 开始时固定的带时区时间，不得在 Ranker 内调用系统当前时间。

**Produces:**

```python
@dataclass(frozen=True)
class EvidenceFeatures:
    relevance: float
    support_strength: float
    source_quality: float
    coverage_gain: float
    independence: float
    freshness: float
    redundancy: float
    risk: float

class EvidenceQualityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provenance_completeness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    directness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    data_verifiability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    parse_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    is_truncated: bool = False
    is_snippet_only: bool = False
    has_stable_locator: bool = True

class SupportObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    level: Literal["none", "weak", "moderate", "direct"]
    judge_model: str
    prompt_version: str
    decision_code: str

@dataclass(frozen=True)
class FeatureObservation:
    values: EvidenceFeatures
    provenance: Mapping[str, str]

class EvidenceFeatureCalculator(Protocol):
    async def compute(
        self, *, candidate: EvidenceCandidate,
        subquestion: SubQuestion, information_need: InformationNeed,
        subquestion_importance: float, need_importance: float,
        freshness_requirement: FreshnessRequirement | None,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int, evaluation_time: datetime,
        deadline: Deadline, cancellation_token: CancellationToken,
    ) -> FeatureObservation: pass

class EvidenceFeatureMaterialResolver(Protocol):
    def evidence_for_ids(
        self, evidence_ids: Collection[str],
    ) -> tuple[EvidenceSpan, ...]: pass
    def quality_for(self, evidence_id: str) -> EvidenceQualityObservation: pass

class StoreBackedFeatureMaterials(EvidenceFeatureMaterialResolver):
    def __init__(
        self, *, get_evidence: Callable[[str], EvidenceSpan],
        get_quality: Callable[[str], EvidenceQualityObservation],
    ) -> None: pass

class SupportStrengthJudge(Protocol):
    async def observe(
        self, information_need: InformationNeed, evidence: EvidenceSpan,
        *, source: SourceDocument,
        deadline: Deadline, cancellation_token: CancellationToken,
    ) -> SupportObservation: pass

class StructuredSupportStrengthJudge(SupportStrengthJudge):
    def __init__(
        self, *, model_provider: ModelProvider,
        budget: BudgetAccountant, model_id: str, prompt_version: str,
    ) -> None: pass

class DefaultEvidenceFeatureCalculator(EvidenceFeatureCalculator):
    def __init__(
        self, *, embedder: TextEmbedder,
        embedding_model_id: str,
        materials: EvidenceFeatureMaterialResolver,
        support_judge: SupportStrengthJudge,
    ) -> None: pass

def support_score(observation: SupportObservation) -> float: pass
def source_quality(
    source: SourceDocument, quality: EvidenceQualityObservation,
) -> float: pass
def coverage_gain(
    subquestion: SubQuestion, information_need: InformationNeed,
    evidence: EvidenceSpan, coverage_score: float,
) -> float: pass
def freshness_score(
    requirement: FreshnessRequirement | None,
    source: SourceDocument, evaluation_time: datetime,
) -> float: pass
def risk_score(quality: EvidenceQualityObservation) -> float: pass
def independence_score(
    source_family_id: str, selected_source_family_ids: AbstractSet[str],
) -> float: pass
def normalized_cosine_score(left: Sequence[float], right: Sequence[float]) -> float: pass

@dataclass(frozen=True)
class EvidenceRankingResult:
    scores: tuple[RerankScore, ...]
    feature_provenance_by_evidence: Mapping[str, Mapping[str, str]]

class EvidenceRanker(Protocol):
    ranker_id: Literal["R0", "R1", "R2"]

    async def score(
        self, subquestion: SubQuestion, information_need: InformationNeed,
        candidates: Sequence[EvidenceCandidate], *,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int, evaluation_time: datetime,
        deadline: Deadline, cancellation_token: CancellationToken,
    ) -> EvidenceRankingResult: pass

class R0SearchOrder(EvidenceRanker): pass
class R1SimilarityOnly(EvidenceRanker): pass
class R2EvidenceUtility(EvidenceRanker):
    def __init__(
        self, *, feature_calculator: EvidenceFeatureCalculator,
        model_id: str = "evidence-features-v1",
        prompt_version: str = "r2-utility-v1",
    ) -> None: pass
```

R0 使用搜索返回顺序；R1 只使用默认 MiniLM cosine relevance；R2 计算 relevance、support strength、source quality、coverage gain、independence、freshness、redundancy、risk 八个特征，并保存全部特征。`InformationNeed.importance` 与 `SubQuestion.importance` 参与 coverage gain，`coverage_score` 表示该子问题调用前覆盖度，`selected_source_family_ids` 用于 independence/redundancy，`SubQuestion.evidence_requirements.freshness` 与固定 `evaluation_time` 用于 freshness。`source_quality=0.40×source_type+0.20×provenance_completeness+0.20×directness+0.20×data_verifiability`，support strength 使用 `0/⅓/⅔/1`。`FeatureObservation.provenance` 为八个特征逐项记录确定性规则、模型/embedding ID 或输入字段来源；字符串 provenance 单独保存，不塞入只允许浮点数的 `RerankScore.feature_scores`。

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_r0_preserves_search_order_and_deterministic_tie_break():
    ranking = await R0SearchOrder().score(
        subquestion, information_need,
        [candidate("e-2", rank=1), candidate("e-1", rank=1)],
        coverage_score=0.0, selected_evidence_ids=frozenset(),
        selected_source_family_ids=frozenset(), context_budget=100,
        evaluation_time=run_started_at, deadline=100, cancellation_token=token(),
    )
    assert [x.evidence_id for x in ranking.scores] == ["e-1", "e-2"]

@pytest.mark.asyncio
async def test_r1_score_contains_only_relevance():
    ranking = await R1SimilarityOnly(
        delegate=SimilarityRanker(fake_embedder),
    ).score(
        subquestion, information_need, candidates, coverage_score=0.0,
        selected_evidence_ids=frozenset(), selected_source_family_ids=frozenset(),
        context_budget=100, evaluation_time=run_started_at,
        deadline=100, cancellation_token=token(),
    )
    assert set(ranking.scores[0].feature_scores) == {"relevance"}
    assert set(ranking.feature_provenance_by_evidence[ranking.scores[0].evidence_id]) == {"relevance"}

@pytest.mark.asyncio
async def test_r2_clips_total_and_persists_all_features():
    ranking = await R2EvidenceUtility(
        feature_calculator=fake_feature_calculator,
    ).score(
        subquestion_with_freshness, important_need, candidates,
        coverage_score=0.25, selected_evidence_ids=frozenset({"e-existing"}),
        selected_source_family_ids=frozenset({"family-existing"}),
        context_budget=100, evaluation_time=run_started_at,
        deadline=100, cancellation_token=token(),
    )
    assert 0 <= ranking.scores[0].total <= 1
    assert set(ranking.scores[0].feature_scores) == {
        "relevance", "support_strength", "source_quality", "coverage_gain",
        "independence", "freshness", "redundancy", "risk",
    }
    assert set(ranking.feature_provenance_by_evidence[ranking.scores[0].evidence_id]) == set(
        ranking.scores[0].feature_scores
    )
    assert fake_feature_calculator.last_call.coverage_score == 0.25
    assert fake_feature_calculator.last_call.selected_source_family_ids == frozenset({"family-existing"})
    assert fake_feature_calculator.last_call.need_importance == important_need.importance
    assert fake_feature_calculator.last_call.freshness_requirement == subquestion_with_freshness.evidence_requirements.freshness

@pytest.mark.parametrize(
    ("level", "expected"),
    [("none", 0.0), ("weak", 1 / 3), ("moderate", 2 / 3), ("direct", 1.0)],
)
def test_support_strength_uses_exact_four_level_rubric(level, expected):
    assert support_score(support_observation(level)) == pytest.approx(expected)

@pytest.mark.asyncio
async def test_structured_support_judge_records_public_provenance():
    judge = StructuredSupportStrengthJudge(
        model_provider=model_returning_support("moderate"),
        budget=fake_budget, model_id="judge-v1", prompt_version="support-rubric-v1",
    )
    observation = await judge.observe(
        information_need, candidates[0].evidence,
        source=candidates[0].source,
        deadline=100, cancellation_token=token(),
    )
    assert observation == SupportObservation(
        level="moderate", judge_model="judge-v1",
        prompt_version="support-rubric-v1", decision_code="DIRECTNESS_MODERATE",
    )

def test_source_quality_uses_exact_weights_and_source_type_default():
    quality = quality_observation(
        provenance_completeness=0.5, directness=0.75, data_verifiability=0.25,
    )
    assert source_quality(source(source_type="standard"), quality) == pytest.approx(0.70)
    assert source_quality(source(source_type="paper"), quality) == pytest.approx(0.66)
    assert source_quality(source(source_type="unknown"), quality) == pytest.approx(0.30)

def test_missing_quality_metadata_uses_neutral_observable_defaults():
    quality = EvidenceQualityObservation()
    assert (
        quality.provenance_completeness,
        quality.directness,
        quality.data_verifiability,
        quality.parse_confidence,
    ) == (0.5, 0.5, 0.5, 0.5)
    assert risk_score(quality) == 0.5

def test_coverage_gain_uses_need_importance_and_prior_coverage():
    subquestion = subquestion_with_needs((need("n-1", 0.75), need("n-2", 0.25)))
    assert coverage_gain(
        subquestion, subquestion.information_needs[0], span(need_ids=("n-1",)), 0.20,
    ) == pytest.approx(0.60)
    assert coverage_gain(
        subquestion, subquestion.information_needs[0], span(need_ids=("n-2",)), 0.20,
    ) == 0.0

def test_independence_is_binary_by_selected_source_family():
    assert independence_score("family-new", frozenset({"family-old"})) == 1.0
    assert independence_score("family-old", frozenset({"family-old"})) == 0.0

def test_freshness_has_none_unknown_satisfied_and_violated_values():
    assert freshness_score(None, source(published_at=None), run_started_at) == 1.0
    requirement = published_after(date(2025, 1, 1))
    assert freshness_score(requirement, source(published_at=None), run_started_at) == 0.5
    assert freshness_score(requirement, source(published_at=datetime(2025, 2, 1, tzinfo=UTC)), run_started_at) == 1.0
    assert freshness_score(requirement, source(published_at=datetime(2024, 12, 1, tzinfo=UTC)), run_started_at) == 0.0

@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        (quality_observation(parse_confidence=1.0), 0.0),
        (quality_observation(parse_confidence=0.79), 0.5),
        (quality_observation(is_truncated=True), 0.5),
        (quality_observation(is_snippet_only=True), 1.0),
        (quality_observation(has_stable_locator=False), 1.0),
    ],
)
def test_risk_uses_complete_low_confidence_and_snippet_buckets(quality, expected):
    assert risk_score(quality) == expected

def test_relevance_and_redundancy_use_normalized_cosine_and_empty_default():
    assert normalized_cosine_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.5)
    selected_vectors = [[1.0, 0.0], [0.0, 1.0]]
    redundancy = max(
        (normalized_cosine_score([1.0, 0.0], vector) for vector in selected_vectors),
        default=0.0,
    )
    assert redundancy == 1.0
    assert max((), default=0.0) == 0.0

@pytest.mark.asyncio
async def test_default_calculator_consumes_store_quality_and_support_observation():
    calculator = DefaultEvidenceFeatureCalculator(
        embedder=fake_embedder, embedding_model_id="embed-v1",
        materials=fake_materials, support_judge=fake_support_judge("direct"),
    )
    observation = await calculator.compute(
        candidate=candidates[0], subquestion=subquestion, information_need=information_need,
        subquestion_importance=subquestion.importance,
        need_importance=information_need.importance, freshness_requirement=None,
        coverage_score=0.0, selected_evidence_ids=frozenset(),
        selected_source_family_ids=frozenset(), context_budget=100,
        evaluation_time=run_started_at, deadline=100, cancellation_token=token(),
    )
    assert set(asdict(observation.values)) == {
        "relevance", "support_strength", "source_quality", "coverage_gain",
        "independence", "freshness", "redundancy", "risk",
    }
    assert set(observation.provenance) == set(asdict(observation.values))

def test_store_backed_materials_resolve_selected_spans_in_stable_order():
    materials = StoreBackedFeatureMaterials(
        get_evidence=evidence_store.get_evidence,
        get_quality=quality_artifact_store.get_observation,
    )
    assert [item.evidence_id for item in materials.evidence_for_ids({"e-2", "e-1"})] == [
        "e-1", "e-2",
    ]
    assert materials.quality_for("e-1").parse_confidence == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/evidence/test_features.py tests/unit/evidence/test_rankers.py tests/unit/evidence/test_ranker_boundaries.py -q`

Expected: FAIL with missing feature observations/calculator and R0/R1/R2 classes.

- [ ] **Step 3: Write minimal implementation**

```python
SOURCE_TYPE_SCORES = {
    "paper": 0.9,
    "official_documentation": 1.0,
    "standard": 1.0,
    "primary_data": 1.0,
    "first_party_statement": 0.8,
    "secondary_analysis": 0.6,
    "news": 0.4,
    "unknown": 0.0,
}
SUPPORT_SCORES = {"none": 0.0, "weak": 1 / 3, "moderate": 2 / 3, "direct": 1.0}


def support_score(observation):
    return SUPPORT_SCORES[observation.level]


def normalized_cosine_score(left, right):
    return clip((cosine(left, right) + 1.0) / 2.0)


def source_quality(source, quality):
    return clip(
        0.40 * SOURCE_TYPE_SCORES[source.source_type]
        + 0.20 * quality.provenance_completeness
        + 0.20 * quality.directness
        + 0.20 * quality.data_verifiability
    )


def coverage_gain(subquestion, information_need, evidence, coverage_score):
    if information_need.need_id not in evidence.information_need_ids:
        return 0.0
    total_importance = sum(need.importance for need in subquestion.information_needs)
    return clip((1.0 - coverage_score) * information_need.importance / total_importance)


def freshness_score(requirement, source, evaluation_time):
    if requirement is None or requirement.kind == "none":
        return 1.0
    if requirement.kind == "published_after":
        if source.published_at is None:
            return 0.5
        return float(source.published_at.date() >= requirement.published_after)
    age_days = (evaluation_time - source.retrieved_at).total_seconds() / 86_400
    return float(age_days <= requirement.retrieved_within_days)


def risk_score(quality):
    if quality.is_snippet_only or not quality.has_stable_locator:
        return 1.0
    if quality.is_truncated or quality.parse_confidence < 0.80:
        return 0.5
    return 0.0


def independence_score(source_family_id, selected_source_family_ids):
    return float(source_family_id not in selected_source_family_ids)


class StoreBackedFeatureMaterials:
    def __init__(self, *, get_evidence, get_quality) -> None:
        self.get_evidence = get_evidence
        self.get_quality = get_quality

    def evidence_for_ids(self, evidence_ids):
        return tuple(self.get_evidence(item) for item in sorted(evidence_ids))

    def quality_for(self, evidence_id):
        return self.get_quality(evidence_id)


class DefaultEvidenceFeatureCalculator:
    def __init__(
        self, *, embedder, embedding_model_id, materials, support_judge,
    ) -> None:
        self.embedder = embedder
        self.embedding_model_id = embedding_model_id
        self.materials = materials
        self.support_judge = support_judge

    async def compute(
        self, *, candidate, subquestion, information_need,
        subquestion_importance, need_importance, freshness_requirement,
        coverage_score, selected_evidence_ids, selected_source_family_ids,
        context_budget, evaluation_time, deadline, cancellation_token,
    ) -> FeatureObservation:
        if subquestion_importance != subquestion.importance:
            raise ValueError("subquestion importance mismatch")
        if need_importance != information_need.importance:
            raise ValueError("information-need importance mismatch")
        if context_budget <= 0:
            raise ValueError("context_budget must be positive")
        selected = self.materials.evidence_for_ids(sorted(selected_evidence_ids))
        texts = [information_need.text, candidate.evidence.excerpt]
        texts.extend(item.excerpt for item in selected)
        vectors = await self.embedder.embed(
            texts, deadline=deadline, cancellation_token=cancellation_token,
        )
        support = await self.support_judge.observe(
            information_need, candidate.evidence,
            source=candidate.source,
            deadline=deadline, cancellation_token=cancellation_token,
        )
        quality = self.materials.quality_for(candidate.evidence.evidence_id)
        values = EvidenceFeatures(
            relevance=normalized_cosine_score(vectors[0], vectors[1]),
            support_strength=support_score(support),
            source_quality=source_quality(candidate.source, quality),
            coverage_gain=coverage_gain(
                subquestion, information_need, candidate.evidence, coverage_score,
            ),
            independence=independence_score(
                candidate.source_family_id, selected_source_family_ids,
            ),
            freshness=freshness_score(
                freshness_requirement, candidate.source, evaluation_time,
            ),
            redundancy=max(
                (
                    normalized_cosine_score(vectors[1], vector)
                    for vector in vectors[2:]
                ),
                default=0.0,
            ),
            risk=risk_score(quality),
        )
        provenance = {
            "relevance": f"embedding:{self.embedding_model_id}:need_excerpt",
            "support_strength": (
                f"judge:{support.judge_model}:{support.prompt_version}:"
                f"{support.decision_code}"
            ),
            "source_quality": (
                f"source:{candidate.source.source_type}:"
                f"parser:{candidate.source.parser_version}"
            ),
            "coverage_gain": (
                f"need:{information_need.need_id}:importance:{need_importance}:"
                f"coverage:{coverage_score}"
            ),
            "independence": f"source_family:{candidate.source_family_id}",
            "freshness": (
                f"requirement:{freshness_requirement.kind if freshness_requirement else 'none'}:"
                f"published_after:{freshness_requirement.published_after if freshness_requirement else None}:"
                f"retrieved_within_days:{freshness_requirement.retrieved_within_days if freshness_requirement else None}:"
                f"evaluation_time:{evaluation_time.isoformat()}"
            ),
            "redundancy": f"embedding:{self.embedding_model_id}:max_selected",
            "risk": (
                f"parse_confidence:{quality.parse_confidence}:"
                f"truncated:{quality.is_truncated}:snippet:{quality.is_snippet_only}:"
                f"stable_locator:{quality.has_stable_locator}"
            ),
        }
        return FeatureObservation(values=values, provenance=provenance)


class R1SimilarityOnly:
    ranker_id = "R1"

    def __init__(self, *, delegate: SimilarityRanker) -> None:
        self.delegate = delegate

    async def score(
        self,
        subquestion: SubQuestion,
        information_need: InformationNeed,
        candidates: Sequence[EvidenceCandidate],
        *,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int,
        evaluation_time: datetime,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> EvidenceRankingResult:
        del subquestion, coverage_score, selected_evidence_ids
        del selected_source_family_ids, context_budget, evaluation_time
        scores = tuple(await self.delegate.score(
            information_need.text,
            [item.evidence for item in candidates],
            deadline=deadline,
            cancellation_token=cancellation_token,
        ))
        return EvidenceRankingResult(
            scores=scores,
            feature_provenance_by_evidence={
                score.evidence_id: {"relevance": "SimilarityRanker:R1"}
                for score in scores
            },
        )


class R2EvidenceUtility:
    ranker_id = "R2"

    def __init__(
        self, *, feature_calculator: EvidenceFeatureCalculator,
        model_id: str = "evidence-features-v1",
        prompt_version: str = "r2-utility-v1",
    ) -> None:
        self.feature_calculator = feature_calculator
        self.model_id = model_id
        self.prompt_version = prompt_version

    async def score(
        self, subquestion, information_need, candidates, *, coverage_score,
        selected_evidence_ids, selected_source_family_ids, context_budget,
        evaluation_time, deadline, cancellation_token,
    ) -> EvidenceRankingResult:
        scores: list[RerankScore] = []
        provenance: dict[str, Mapping[str, str]] = {}
        for candidate in candidates:
            observation = await self.feature_calculator.compute(
                candidate=candidate, subquestion=subquestion,
                information_need=information_need,
                subquestion_importance=subquestion.importance,
                need_importance=information_need.importance,
                freshness_requirement=subquestion.evidence_requirements.freshness,
                coverage_score=coverage_score,
                selected_evidence_ids=selected_evidence_ids,
                selected_source_family_ids=selected_source_family_ids,
                context_budget=context_budget, evaluation_time=evaluation_time,
                deadline=deadline, cancellation_token=cancellation_token,
            )
            features = observation.values
            score = RerankScore(
                evidence_id=candidate.evidence.evidence_id,
                total=clip(
                    0.25 * features.relevance
                    + 0.20 * features.support_strength
                    + 0.15 * features.source_quality
                    + 0.20 * features.coverage_gain
                    + 0.10 * features.independence
                    + 0.05 * features.freshness
                    - 0.15 * features.redundancy
                    - 0.05 * features.risk
                ),
                feature_scores=asdict(features), model_id=self.model_id,
                prompt_version=self.prompt_version,
            )
            scores.append(score)
            provenance[score.evidence_id] = dict(observation.provenance)
        ordered = tuple(sorted(scores, key=lambda item: (-item.total, item.evidence_id)))
        return EvidenceRankingResult(ordered, provenance)
```

Declare all feature types、`StructuredSupportStrengthJudge` and `DefaultEvidenceFeatureCalculator` only in `evidence/features.py`, and `EvidenceRankingResult` only in `evidence/rankers.py`; import Core types rather than duplicating them. `EvidenceQualityObservation` validates its four numeric fields in `[0,1]`; parse/store handlers create it from parser confidence、truncation、snippet origin、stable locator and source metadata and persist it beside the Evidence ID. `StructuredSupportStrengthJudge.observe` reserves/settles Core budget with `node="Ranker"`, makes one `ModelProvider.structured` call with only the current need/excerpt/source metadata, and accepts exactly the four public levels plus model/prompt/decision provenance. `DefaultEvidenceFeatureCalculator.compute` is the concrete production implementation shown above; tests must use it for formula coverage, while the fake calculator is retained only to isolate R2 weighted summation. Implement R0/R1/R2 behind the same protocol. Import `SimilarityRanker` from `deepresearch.evidence.similarity` and make `R1SimilarityOnly` a thin signature/result adapter; inject its underlying embedder instance into Query Scheduler and the default feature calculator. R0 writes `search_rank:<rank>` provenance, R1 writes the Core delegate ID, and R2 copies all eight strings from `FeatureObservation.provenance`; every result must have exactly the same feature keys in score and provenance. R2 owns only the new weighted formula. Clip all values and write `RerankScore.feature_scores` without treating total as truth probability. The `RankEvidence` node persists `feature_provenance_by_evidence` in its rank artifact before putting only artifact/evidence IDs into State.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/evidence/test_features.py tests/unit/evidence/test_rankers.py tests/unit/evidence/test_ranker_boundaries.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/evidence/features.py src/deepresearch/evidence/rankers.py tests/unit/evidence
git commit -m "feat: implement explainable R0 R1 and R2 rankers"
```

### Task 8: Pass B Claim Extractor、Evidence Judge 与 Claim-Evidence Graph

**Files:**
- Create: `src/deepresearch/evidence/claims.py`
- Create: `src/deepresearch/evidence/graph.py`
- Create: `tests/unit/evidence/test_claims.py`
- Create: `tests/unit/evidence/test_claim_graph.py`

**Consumes:** Core `Claim`、`EvidenceSpan`、`ClaimEvidenceLink`、`ModelProvider`，Task 7 选中的 Evidence ID。

**Produces:**

```python
@dataclass(frozen=True)
class GraphValidationResult:
    valid: bool
    error_codes: tuple[str, ...]

class ClaimExtractor:
    async def extract(
        self, draft_markdown: str, *, evidence_ids: Collection[str],
        deadline: Deadline, cancellation_token: CancellationToken,
    ) -> list[Claim]: pass

class EvidenceJudge:
    async def judge(
        self, claim: Claim, evidence: Sequence[EvidenceSpan], *,
        deadline: Deadline, cancellation_token: CancellationToken,
    ) -> list[ClaimEvidenceLink]: pass

class ClaimEvidenceGraph:
    def add_claim(self, claim: Claim) -> None: pass
    def add_evidence(self, evidence: EvidenceSpan) -> None: pass
    def add_link(self, link: ClaimEvidenceLink) -> None: pass
    def links_for_claim(self, claim_id: str) -> tuple[ClaimEvidenceLink, ...]: pass
    def validate(self) -> GraphValidationResult: pass
    def to_json(self) -> dict[str, object]: pass
```

Pass B 只判断最终 atomic claim 的 `support`、`contradict`、`context`、`insufficient`；judge model、prompt version 和 decision code 必须写入 link。Evidence 不充分时先查已有未链接证据，再由 Workflow 最多触发一轮定向补搜，仍不充分则改写/删除/放入限制章节。

- [ ] **Step 1: Write the failing test**

```python
def test_graph_rejects_link_to_unknown_evidence():
    graph = ClaimEvidenceGraph()
    graph.add_claim(claim("c-1"))
    with pytest.raises(ValueError, match="unknown evidence"):
        graph.add_link(link("c-1", "e-missing", relation="support"))

def test_graph_keeps_support_and_contradiction_edges():
    graph = graph_with_claim_and_evidence()
    graph.add_link(link("c-1", "e-1", relation="support"))
    graph.add_link(link("c-1", "e-2", relation="contradict"))
    assert {x.relation for x in graph.links_for_claim("c-1")} == {"support", "contradict"}

def test_valid_graph_returns_typed_validation_result():
    result = graph_with_claim_and_evidence().validate()
    assert result == GraphValidationResult(valid=True, error_codes=())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/evidence/test_claims.py tests/unit/evidence/test_claim_graph.py -q`

Expected: FAIL with missing extractor, judge, and graph implementations.

- [ ] **Step 3: Write minimal implementation**

```python
def add_link(self, link: ClaimEvidenceLink) -> None:
    if link.claim_id not in self._claims:
        raise ValueError("unknown claim")
    if link.evidence_id not in self._evidence:
        raise ValueError("unknown evidence")
    self._links.append(link)
```

Declare `GraphValidationResult` only in `evidence/graph.py`, beside `ClaimEvidenceGraph`; import `Claim`、`EvidenceSpan` and `ClaimEvidenceLink` from `deepresearch.domain`. Use Core `ModelProvider.structured` for extraction/judging; `validate()` returns sorted public error codes such as `UNKNOWN_CLAIM`、`UNKNOWN_EVIDENCE` and `DUPLICATE_LINK`, validates referential integrity, and preserves all relation types.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/evidence/test_claims.py tests/unit/evidence/test_claim_graph.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/evidence/claims.py src/deepresearch/evidence/graph.py tests/unit/evidence
git commit -m "feat: add claim evidence graph and pass-b judging"
```

### Task 9: Citation Guard

**Files:**
- Create: `src/deepresearch/evidence/citation_guard.py`
- Create: `tests/unit/evidence/test_citation_guard.py`
- Create: `tests/fixtures/security/invalid_citations.md`

**Consumes:** Task 8 `ClaimEvidenceGraph`，Core `EvidenceSpan`、`SourceDocument`、`HtmlLocator`、`PdfLocator`、Artifact/Evidence Store。Workflow 通过本任务的只读 resolver adapter 读取材料；原始 bytes 或正文不得进入 Graph State。

**Produces:**

```python
@dataclass(frozen=True)
class CitationGuardResult:
    valid: bool
    errors: tuple[str, ...]
    checked_citation_ids: tuple[str, ...]

class CitationMaterialResolver(Protocol):
    def raw_bytes_for_source(self, source_id: str) -> bytes: pass
    def normalized_document_text(self, source_id: str) -> str: pass
    def html_paragraph_text(self, source_id: str, paragraph_id: str) -> str: pass
    def pdf_block_text(
        self, source_id: str, page_index: int, block_index: int,
    ) -> str: pass

class CitationGuard:
    def verify(
        self, report_markdown: str, graph: ClaimEvidenceGraph,
        evidence: Mapping[str, EvidenceSpan],
        sources: Mapping[str, SourceDocument], *,
        materials: CitationMaterialResolver,
    ) -> CitationGuardResult: pass
```

验证每个行内 citation ID、claim/evidence/source 存在性、Html/Pdf locator、Unicode code point offset、raw `content_hash`、`parsed_content_hash` 和 `excerpt_hash`。resolver 对 HTML 按 `paragraph_id` 返回容器文本，对 PDF 按零基 `page_index + block_index` 返回块文本；Guard 再在该容器内应用左闭右开 offset，绝不把 PDF offset 当成整个文档的平面下标。Guard 不能创建新证据、URL 或 citation；未知、缺失或任一 hash 不匹配均失败。

- [ ] **Step 1: Write the failing test**

```python
def test_guard_rejects_unknown_citation_id():
    result = guard.verify("[E-missing] claim", valid_graph(), valid_evidence(), valid_sources(),
                          materials=valid_materials())
    assert result.valid is False
    assert "unknown citation" in " ".join(result.errors)

def test_guard_rejects_excerpt_hash_mismatch():
    result = guard.verify("[E-1] claim", valid_graph(), tampered_evidence(), valid_sources(),
                          materials=valid_materials())
    assert result.valid is False
    assert "excerpt hash" in " ".join(result.errors)

def test_guard_resolves_pdf_page_and_block_before_offsets():
    result = guard.verify(
        "claim [E-pdf]", pdf_graph(), pdf_evidence(page_index=1, block_index=3),
        pdf_sources(), materials=pdf_materials(block_text="prefix evidence suffix"),
    )
    assert result.valid is True

def test_guard_rejects_raw_content_hash_mismatch():
    result = guard.verify(
        "claim [E-1]", valid_graph(), valid_evidence(), valid_sources(),
        materials=materials_with_tampered_raw_bytes(),
    )
    assert result.valid is False
    assert "content hash" in " ".join(result.errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/evidence/test_citation_guard.py -q`

Expected: FAIL with missing `CitationGuard`.

- [ ] **Step 3: Write minimal implementation**

```python
for citation_id in extract_citation_ids(report_markdown):
    span = evidence.get(citation_id)
    if span is None:
        errors.append(f"unknown citation: {citation_id}")
        continue
    source = sources[span.source_id]
    if sha256(materials.raw_bytes_for_source(span.source_id)).hexdigest() != source.content_hash:
        errors.append(f"content hash mismatch: {span.source_id}")
        continue
    normalized = materials.normalized_document_text(span.source_id)
    if sha256_text(normalized) != source.parsed_content_hash:
        errors.append(f"parsed content hash mismatch: {span.source_id}")
        continue
    if isinstance(span.locator, HtmlLocator):
        container = materials.html_paragraph_text(
            span.source_id, span.locator.paragraph_id,
        )
    elif isinstance(span.locator, PdfLocator):
        container = materials.pdf_block_text(
            span.source_id, span.locator.page_index, span.locator.block_index,
        )
    else:
        errors.append(f"unsupported locator: {citation_id}")
        continue
    if not (0 <= span.locator.start_char < span.locator.end_char <= len(container)):
        errors.append(f"locator out of bounds: {citation_id}")
        continue
    excerpt = container[span.locator.start_char:span.locator.end_char]
    if excerpt != span.excerpt or sha256_text(excerpt) != span.excerpt_hash:
        errors.append(f"excerpt hash mismatch: {citation_id}")
```

Declare `CitationGuardResult` and `CitationMaterialResolver` only in `evidence/citation_guard.py`; the resolver is a read-only adapter over Core stores, not a second storage implementation. Parse stable citation IDs, resolve graph edges, branch explicitly on `HtmlLocator`/`PdfLocator`, verify range bounds before slicing, recompute all three SHA-256 layers, and return all public validation errors without exposing raw material, provider secrets or hidden reasoning.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/evidence/test_citation_guard.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/evidence/citation_guard.py tests/unit/evidence/test_citation_guard.py tests/fixtures/security/invalid_citations.md
git commit -m "feat: verify claim citations against evidence hashes"
```

### Task 10: LangGraph Research Graph 集成

**Files:**
- Modify: `src/deepresearch/workflow/state.py`
- Create: `src/deepresearch/workflow/research_graph.py`
- Modify: `src/deepresearch/workflow/runner.py`
- Create: `tests/integration/replay/test_research_graph.py`
- Create: `tests/integration/replay/test_checkpoint_idempotency.py`
- Create: `tests/fixtures/replay/still_unsupported_after_targeted/`

**Consumes:** Tasks 1–9 的 Planner、Scheduler、Ranker、Graph、Citation Guard；Core `BaselineState`/`BaselineBlockedNeed`、`FixedPlanner`/`PlanGenerationError`、既有 baseline node handlers、既有 `LangGraphResearchRunner` 与 `ResearchRunner.run` port、`CheckpointRef`、`checkpoint_serializer()`、`ResearchRequest`、`RunBudget`、`SearchProvider`、`Fetcher`、`Parser`、`ModelProvider`、Artifact/Evidence Store、checkpoint contract。`ResearchGraphDependencies` 只接收 composition root 注入的 handlers，不复制 Core handler 实现。

**Produces:**

```python
@dataclass(frozen=True)
class ClaimResolutionRecord:
    claim_id: str
    action: Literal["DELETE", "REWRITE", "MOVE_TO_LIMITATIONS"]
    reason_code: Literal[
        "UNSUPPORTED_FACT", "OVERSTATED_SUPPORT", "CONTRADICTED", "UNCERTAIN",
    ]
    replacement_text: str | None

class ResearchState(BaselineState, total=False):
    planner_round_index: int
    decision_route: Literal["SEARCH", "STOP"]
    verification_route: Literal[
        "TARGETED_RESEARCH", "RESOLVE_UNSUPPORTED", "FINALIZE",
    ]
    rank_artifact_id: str | None
    claim_ids: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]
    claim_resolution_artifact_id: str | None
    citation_guard_artifact_id: str | None
    directional_research_rounds: int

def blocked_need_from_checkpoint(record: BaselineBlockedNeed) -> BlockedNeed: pass
def blocked_need_to_checkpoint(item: BlockedNeed) -> BaselineBlockedNeed: pass

NodeHandler: TypeAlias = Callable[
    [ResearchState], Awaitable[Mapping[str, object]],
]

class InitialPlanNode(Protocol):
    initial_plan_generator: FixedPlanner

    async def __call__(
        self, state: ResearchState,
    ) -> Mapping[str, object]: pass

@dataclass(frozen=True)
class ResearchGraphDependencies:
    validate_request: NodeHandler
    initial_plan_generator: FixedPlanner
    plan: InitialPlanNode
    decide_next: NodeHandler
    search: NodeHandler
    fetch: NodeHandler
    parse_and_normalize: NodeHandler
    store_evidence: NodeHandler
    rank_evidence: NodeHandler
    draft_report: NodeHandler
    extract_claims: NodeHandler
    verify_claims: NodeHandler
    targeted_research: NodeHandler
    resolve_unsupported_claims: NodeHandler
    finalize_citations: NodeHandler
    persist_results: NodeHandler
    checkpointer: BaseCheckpointSaver

def build_research_graph(
    dependencies: ResearchGraphDependencies,
) -> CompiledStateGraph: pass

# Add this private dispatch method to Core's existing LangGraphResearchRunner.
# Its existing public run(...) signature remains byte-for-byte unchanged.
def _graph_for_config(
    self: LangGraphResearchRunner, config: RunConfig,
) -> CompiledStateGraph: pass
```

`state.py` 从 Core `BaselineState` 派生唯一的 `ResearchState`，直接继承其 `coverage_ledger`、`blocked_needs: tuple[BaselineBlockedNeed, ...]`、`recent_marginal_gains` 和 `budget_snapshot`，绝不以不同类型重声明这些键。Core `BaselineBlockedNeed` 是仅含 str/bool/int 的 TypedDict；`BlockedNeed` dataclass 只在 Planner 节点内存中使用。节点入口以 `blocked_need_from_checkpoint` 将 inherited primitive records 转为 Planner dataclass，节点出口以 `blocked_need_to_checkpoint` 转回 `BaselineBlockedNeed`，因此 checkpoint 永不保存 `BlockedNeed`，也不修改 Core serializer allow-list。`research_graph.py` 唯一声明 `ClaimResolutionRecord`、`NodeHandler` 与 `ResearchGraphDependencies`，从 `langgraph.checkpoint.base` 导入 `BaseCheckpointSaver`、从 `langgraph.graph` 导入 `END`/`StateGraph`、从 `langgraph.graph.state` 导入 `CompiledStateGraph`。`ClaimResolutionRecord` 要求 DELETE 的 replacement 为 `None`，REWRITE/MOVE_TO_LIMITATIONS 的 replacement 为非空公开文本。`initial_plan_generator` 是 Core `FixedPlanner` 的同一实例，composition root 将其绑定进 `plan` handler；因此 research-v1 继承 Core 的 validator 与“最多一次 repair，二次失败抛 `PLAN_INVALID`”语义。research_graph.py 复用 Core handlers 但创建独立的 `ValidateRequest → Plan → DecideNext → Search → Fetch → ParseAndNormalize → StoreEvidence → RankEvidence → DraftReport → ExtractClaims → VerifyClaims → TargetedResearch/ResolveUnsupportedClaims → FinalizeCitations → PersistResults` 图，加入 CoverageLedger、typed exhaustion、Pass B claim/judge 和 Citation Guard；不得修改 baseline_graph.py。`DecideNext` 先统一调用 `evaluate_stop`，未停止才路由 P0/P1/P2 的 SEARCH；四种停止码都走 STOP。首次证据不足可走一次 `TargetedResearch`；再次验证仍不足必须进入 `ResolveUnsupportedClaims`，按 typed resolution action 删除 unsupported factual/numeric claim、改写过强 claim 或把 contradicted/uncertain claim 移入限制章节并保留证据引用，然后才可 Finalize。State 只保存计划/账本/预算/primitive typed exhaustion、状态和 artifact/evidence/claim IDs；完整正文与 feature provenance 在 Artifact Store。每个有副作用节点先检查 Core cache/artifact 和稳定 idempotency key，再调用 provider；checkpoint 恢复不得重复扣账。

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_graph_routes_sufficient_to_terminal_without_replan():
    result = await harness.run_graph_fixture("sufficient")
    assert result.status == "completed"
    assert result.stop_reason == "SUFFICIENT"
    decisions = [
        event.public_payload for event in harness.events
        if event.node == "DecideNext"
    ]
    assert [item["decision_code"] for item in decisions] == ["SUFFICIENT"]

@pytest.mark.asyncio
async def test_graph_resume_reuses_artifacts_and_does_not_repeat_usage():
    checkpoint = await harness.run_graph_fixture_until_checkpoint("after_search")
    provider_calls_before = tuple(harness.provider_call_ids)
    usage_before = harness.budget.snapshot()
    result = await harness.resume_graph(checkpoint)
    assert isinstance(result, RunResult)
    assert tuple(harness.provider_call_ids) == provider_calls_before
    assert harness.budget.snapshot() == usage_before

def test_research_state_blocked_need_strictly_roundtrips_with_core_serializer():
    record: BaselineBlockedNeed = {
        "need_id": "need-1",
        "required_source_unavailable": True,
        "alternative_strategies_exhausted": True,
        "retry_count": 2,
        "max_retries": 2,
    }
    state = research_state(
        blocked_needs=(record,), recent_marginal_gains=(0.04, 0.03),
    )
    serializer = checkpoint_serializer()
    restored = serializer.loads_typed(serializer.dumps_typed(state))
    assert restored["blocked_needs"] == (record,)
    assert restored["recent_marginal_gains"] == (0.04, 0.03)
    assert type(restored["blocked_needs"][0]) is dict
    internal = blocked_need_from_checkpoint(restored["blocked_needs"][0])
    assert internal.terminal is True
    assert blocked_need_to_checkpoint(internal) == record

def test_existing_runner_dispatches_both_workflows_without_a_second_runner_class():
    assert type(harness.runner) is LangGraphResearchRunner
    assert harness.runner._graph_for_config(baseline_config()) is harness.baseline_graph
    assert harness.runner._graph_for_config(research_config()) is harness.research_graph

@pytest.mark.asyncio
async def test_research_graph_uses_core_one_repair_then_plan_invalid():
    generator = FixedPlanner(
        model=always_invalid_plan_model,
        artifact_store=harness.artifact_store,
        budget=harness.run_budget,
        search_depth=2,
    )
    result = await harness.run_with_initial_generator(generator)
    assert harness.dependencies.initial_plan_generator is generator
    assert result.status == "failed"
    assert result.error_code == "PLAN_INVALID"
    assert always_invalid_plan_model.complete_calls == 2
    assert not [event for event in harness.events if event.node == "Search"]

@pytest.mark.asyncio
async def test_still_unsupported_claims_are_resolved_before_citations():
    result = await harness.run_graph_fixture("still_unsupported_after_targeted")
    resolution_event = next(
        event for event in harness.events if event.node == "ResolveUnsupportedClaims"
    )
    resolution = harness.read_json_artifact(
        resolution_event.public_payload["claim_resolution_artifact_id"]
    )
    assert resolution["actions"] == {
        "c-unsupported": "DELETE", "c-uncertain": "MOVE_TO_LIMITATIONS",
    }
    assert result.evidence_graph_artifact_id is not None
    graph_payload = harness.read_json_artifact(result.evidence_graph_artifact_id)
    assert graph_payload["citation_guard"]["valid"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/replay/test_research_graph.py tests/integration/replay/test_checkpoint_idempotency.py -q`

Expected: FAIL with missing `build_research_graph` and graph nodes.

- [ ] **Step 3: Write minimal implementation**

```python
def blocked_need_from_checkpoint(record):
    return BlockedNeed(
        need_id=record["need_id"],
        required_source_unavailable=record["required_source_unavailable"],
        alternative_strategies_exhausted=record["alternative_strategies_exhausted"],
        retries_used=record["retry_count"],
        max_retries=record["max_retries"],
    )


def blocked_need_to_checkpoint(item):
    return {
        "need_id": item.need_id,
        "required_source_unavailable": item.required_source_unavailable,
        "alternative_strategies_exhausted": item.alternative_strategies_exhausted,
        "retry_count": item.retries_used,
        "max_retries": item.max_retries,
    }


if dependencies.plan.initial_plan_generator is not dependencies.initial_plan_generator:
    raise ValueError("Plan node must own the configured initial_plan_generator")
graph = StateGraph(ResearchState)
graph.add_node("ValidateRequest", dependencies.validate_request)
graph.add_node("Plan", dependencies.plan)
graph.add_node("DecideNext", dependencies.decide_next)
graph.add_node("Search", dependencies.search)
graph.add_node("Fetch", dependencies.fetch)
graph.add_node("ParseAndNormalize", dependencies.parse_and_normalize)
graph.add_node("StoreEvidence", dependencies.store_evidence)
graph.add_node("RankEvidence", dependencies.rank_evidence)
graph.add_node("DraftReport", dependencies.draft_report)
graph.add_node("ExtractClaims", dependencies.extract_claims)
graph.add_node("VerifyClaims", dependencies.verify_claims)
graph.add_node("TargetedResearch", dependencies.targeted_research)
graph.add_node("ResolveUnsupportedClaims", dependencies.resolve_unsupported_claims)
graph.add_node("FinalizeCitations", dependencies.finalize_citations)
graph.add_node("PersistResults", dependencies.persist_results)

graph.set_entry_point("ValidateRequest")
graph.add_edge("ValidateRequest", "Plan")
graph.add_edge("Plan", "DecideNext")
graph.add_conditional_edges(
    "DecideNext", route_after_decide,
    {"SEARCH": "Search", "STOP": "DraftReport"},
)
graph.add_edge("Search", "Fetch")
graph.add_edge("Fetch", "ParseAndNormalize")
graph.add_edge("ParseAndNormalize", "StoreEvidence")
graph.add_edge("StoreEvidence", "RankEvidence")
graph.add_edge("RankEvidence", "DecideNext")
graph.add_edge("DraftReport", "ExtractClaims")
graph.add_edge("ExtractClaims", "VerifyClaims")
graph.add_conditional_edges(
    "VerifyClaims", route_after_verify,
    {
        "TARGETED_RESEARCH": "TargetedResearch",
        "RESOLVE_UNSUPPORTED": "ResolveUnsupportedClaims",
        "FINALIZE": "FinalizeCitations",
    },
)
graph.add_edge("TargetedResearch", "Search")
graph.add_edge("ResolveUnsupportedClaims", "FinalizeCitations")
graph.add_edge("FinalizeCitations", "PersistResults")
graph.add_edge("PersistResults", END)
return graph.compile(checkpointer=dependencies.checkpointer)


# Methods added inside Core's existing LangGraphResearchRunner.
def _graph_for_config(self, config):
    if config.workflow_id == "baseline-v1":
        return self._baseline_graph
    if config.workflow_id == "research-v1" and self._research_graph is not None:
        return self._research_graph
    raise ValueError("research-v1 graph is not configured")
```

Implement `route_after_decide(state) -> Literal["SEARCH", "STOP"]` by returning the validated `decision_route`, and `route_after_verify(state) -> Literal["TARGETED_RESEARCH", "RESOLVE_UNSUPPORTED", "FINALIZE"]` by returning the validated `verification_route`; reject any other value before graph routing. `VerifyClaims` chooses targeted research only when `directional_research_rounds == 0`; afterward any remaining unsupported IDs choose resolution, and an empty unsupported set chooses finalization. `ResolveUnsupportedClaims` persists a typed action map plus rewritten report artifact before returning only IDs. The builder above registers and connects every declared research-v1 node, including baseline request validation and `PersistResults`. Persist public structured decisions before returning node results; handle `SUFFICIENT`, `PLATEAU`, `BUDGET_EXHAUSTED`, and partial `BLOCKED` as specified. `tests` use the canonical `RunResult` fields only; provider call IDs、checkpoint IDs、events and intermediate budget snapshots belong to the Replay harness/spy, never to an extended `RunResult`.

Modify Core's existing `LangGraphResearchRunner` in runner.py; do not define a second runner class or change its public `run(...)` signature. Append one keyword-only constructor dependency `research_graph: CompiledStateGraph | None = None` after all existing constructor parameters, retain the existing baseline graph field, store the new graph as `_research_graph`, and have the existing `run` call `_graph_for_config(config)`. `workflow_id=baseline-v1` continues to use the exact Core baseline graph (and its P1/R1 validation); `workflow_id=research-v1` requires and uses the injected research graph. All A/B/C/D experiments set research-v1; the Core Replay quickstart and pre-existing constructor calls remain baseline-v1-compatible because the new argument defaults to `None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/replay/test_research_graph.py tests/integration/replay/test_checkpoint_idempotency.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/workflow/state.py src/deepresearch/workflow/research_graph.py src/deepresearch/workflow/runner.py tests/integration/replay/test_research_graph.py tests/integration/replay/test_checkpoint_idempotency.py tests/fixtures/replay/still_unsupported_after_targeted
git commit -m "feat: integrate planner ranker and evidence graph"
```

### Task 11: 消费 Core Replay Providers 与研究轨迹集成

**Files:**
- Create: `tests/integration/replay/test_replay_provider_paths.py`
- Create: `tests/fixtures/replay/planner_ranker/snapshot.json`
- Create: `tests/fixtures/replay/planner_ranker/search.jsonl`
- Create: `tests/fixtures/replay/planner_ranker/documents.jsonl`
- Create: `tests/fixtures/replay/planner_ranker/model_responses.jsonl`
- Create: `tests/fixtures/replay/planner_ranker/embeddings.jsonl`
- Create: `tests/fixtures/replay/planner_ranker/manifest.sha256`

**Consumes:** Core `deepresearch.providers.replay.ReplayBundle`、`ReplayModelProvider`、`ReplaySearchProvider`、`ReplayFetcher`、`ReplayTextEmbedder`、`ProviderError`；Task 10 graph。

**Produces:**

严格 Replay 的 cache key/request hash 必须匹配 `snapshot_id + canonical request`；未知 query、model request、hash mismatch 由 Core Replay provider 抛出 `ProviderError(code="REPLAY_MISS")`。本任务不添加 provider 类、不添加 Live delegate，也不改变 Core Replay 实现。

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_replay_unknown_query_returns_replay_miss():
    provider = ReplaySearchProvider(bundle)
    with pytest.raises(ProviderError) as exc:
        await provider.search("unknown query", 5, None,
                                   deadline=100, cancellation_token=token())
    assert exc.value.code == "REPLAY_MISS"

@pytest.mark.asyncio
async def test_replay_embedder_is_shared_by_every_semantic_consumer():
    await harness.run_replay_fixture("planner_ranker")
    embedder = harness.components.text_embedder
    assert isinstance(embedder, ReplayTextEmbedder)
    assert harness.components.query_scheduler.embedder is embedder
    assert harness.components.r1.delegate.embedder is embedder
    assert harness.components.r2.feature_calculator.embedder is embedder
    vectors = await embedder.embed(
        ["known replay query", "known evidence excerpt"],
        deadline=100, cancellation_token=token(),
    )
    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)

@pytest.mark.asyncio
async def test_replay_graph_produces_claim_graph_and_citation_guard_result():
    result = await harness.run_replay_fixture("planner_ranker")
    assert isinstance(result, RunResult)
    assert result.evidence_graph_artifact_id is not None
    graph_payload = harness.read_json_artifact(result.evidence_graph_artifact_id)
    assert graph_payload["claims"]
    assert graph_payload["validation"]["valid"] is True
    assert graph_payload["citation_guard"]["valid"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/replay/test_replay_provider_paths.py -q`

Expected: FAIL with missing replay fixture records or graph wiring; Core Replay provider classes are already supplied by the predecessor plan.

- [ ] **Step 3: Write minimal implementation**

```python
bundle = ReplayBundle.load(bundle_root)
verification = bundle.verify()
assert verification.valid is True
search_provider = ReplaySearchProvider(bundle)
text_embedder = ReplayTextEmbedder(bundle)
hits = await search_provider.search(
    "known replay query", 5, None,
    deadline=100.0, cancellation_token=CancellationToken(),
)
assert hits[0].provider_metadata["source_id"] == "src-1"
vectors = await text_embedder.embed(
    ["known replay query"], deadline=100.0,
    cancellation_token=CancellationToken(),
)
assert len(vectors[0]) == 384
```

Create the six listed replay artifacts, including sorted `embeddings.jsonl`, and validate them through Core `ReplayBundle.load(bundle_root).verify()`. Instantiate Core Replay providers plus one shared `ReplayTextEmbedder`; inject that same instance into `SimilarityRanker`、`QueryScheduler` and `DefaultEvidenceFeatureCalculator`, so R1、semantic query dedupe and R2 are all strict Replay. `harness.read_json_artifact` reads through the Core Artifact Store and the test never adds fields to canonical `RunResult`. Do not add another Replay provider/embedder module or Live fallback.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/replay/test_replay_provider_paths.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/replay/test_replay_provider_paths.py tests/fixtures/replay/planner_ranker
git commit -m "test: exercise core replay providers in research graph"
```

### Task 12: 停止、冲突、预算与部分结果集成覆盖

**Files:**
- Modify: `src/deepresearch/workflow/research_graph.py`
- Create: `tests/integration/replay/test_stop_paths.py`
- Create: `tests/integration/replay/test_conflict_research.py`
- Create: `tests/integration/replay/test_partial_results.py`
- Create: `tests/fixtures/replay/stop_plateau/`
- Create: `tests/fixtures/replay/stop_budget_exhausted/`
- Create: `tests/fixtures/replay/stop_blocked/`
- Create: `tests/fixtures/replay/conflict_research/`

**Consumes:** Tasks 1–11 全部接口。

**Produces:** 可独立审阅的 canonical `RunResult` + `RunEvent` + artifacts 观察契约，以及 `research_graph.py` 中唯一的 `result_status_for(stop_code: StopCode, report_artifact_id: str | None) -> tuple[RunStatus, bool]`。停止码来自 `RunResult.stop_reason`；未覆盖 needs、selected evidence IDs、claim graph 和 Citation Guard 结果来自 public event/artifact；补搜次数来自 `TargetedResearch` 事件数；预算来自 `RunResult.final_usage` 或 harness 的 Core `BudgetSnapshot`。不得向 `RunResult` 添加这些中间字段。

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_conflict_triggers_at_most_one_directional_research_round():
    result = await harness.run_fixture("conflict_research")
    targeted = [event for event in harness.events if event.node == "TargetedResearch"]
    assert len(targeted) == 1
    assert result.stop_reason in {"SUFFICIENT", "PLATEAU"}

@pytest.mark.asyncio
async def test_budget_exhausted_is_partial_not_sufficient():
    result = await harness.run_fixture("stop_budget_exhausted")
    assert result.stop_reason == "BUDGET_EXHAUSTED"
    assert result.is_partial is True

@pytest.mark.asyncio
async def test_blocked_without_minimum_report_is_failed():
    result = await harness.run_fixture("stop_blocked")
    assert result.status == "failed"
    assert result.stop_reason == "BLOCKED"
    assert result.report_artifact_id is None

@pytest.mark.asyncio
async def test_first_provider_failure_tries_alternative_before_blocked():
    await harness.run_fixture("stop_blocked")
    public_steps = [
        (event.node, event.kind, event.public_payload.get("decision_code"))
        for event in harness.events
    ]
    first_failure = next(i for i, step in enumerate(public_steps) if step[1] == "PROVIDER_FAILURE")
    alternative = next(
        i for i, step in enumerate(public_steps)
        if step[2] == "ALTERNATIVE_STRATEGY"
    )
    blocked = next(i for i, step in enumerate(public_steps) if step[2] == "BLOCKED")
    assert first_failure < alternative < blocked
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/replay/test_stop_paths.py tests/integration/replay/test_conflict_research.py tests/integration/replay/test_partial_results.py -q`

Expected: FAIL because stop fixtures and graph routes do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def result_status_for(
    stop_code: StopCode,
    report_artifact_id: str | None,
) -> tuple[RunStatus, bool]:
    if report_artifact_id is None:
        return "failed", stop_code is not StopCode.SUFFICIENT
    if stop_code is StopCode.SUFFICIENT:
        return "completed", False
    return "completed", True


assert len([event for event in harness.events if event.node == "TargetedResearch"]) <= 1
assert len([event for event in harness.events if event.kind == "PROVIDER_RETRY"]) <= 2
persist_payload = next(
    event.public_payload for event in harness.events if event.node == "PersistResults"
)
assert result.stop_reason != "SUFFICIENT" or (
    persist_payload["uncovered_information_needs"] == []
)
```

Add deterministic fixtures and graph assertions for `PLATEAU`, `BUDGET_EXHAUSTED`, `BLOCKED`, conflict-directed re-search, partial report restrictions, cancellation-safe boundaries, and retry exhaustion. Modify `research_graph.py` only to add/use the exact `result_status_for` terminal mapping above and to enforce the already-declared one-round `TargetedResearch` route; `BLOCKED` without a report artifact is unconditionally `FAILED`, while a non-sufficient run with a valid minimum report is `COMPLETED` and partial.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/replay/test_stop_paths.py tests/integration/replay/test_conflict_research.py tests/integration/replay/test_partial_results.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepresearch/workflow/research_graph.py tests/integration/replay tests/fixtures/replay/stop_plateau tests/fixtures/replay/stop_budget_exhausted tests/fixtures/replay/stop_blocked tests/fixtures/replay/conflict_research
git commit -m "test: cover planner stop and conflict replay paths"
```

## Verification Checklist

- [ ] `uv run pytest tests/unit/planning tests/unit/evidence -q`
- [ ] `uv run pytest tests/contracts tests/integration/replay -q`
- [ ] `uv run ruff check src/deepresearch tests`
- [ ] `uv run pyright src/deepresearch`
- [ ] 固定 fixture 上可分别运行 P0/P1/P2 与 R0/R1/R2。
- [ ] R2 八项公式、neutral quality defaults、feature provenance、source family、coverage ledger、priority 和停止原因均有 concrete calculator 测试且可序列化。
- [ ] 首次 provider/query 失败继续替代策略；只有 checkpoint-safe `BlockedNeed.terminal` 才产生 BLOCKED，固定深度不得伪装成 PLATEAU。
- [ ] research-v1 从 ValidateRequest 开始，使用 Core FixedPlanner 的一次 repair，第二次无效以 PLAN_INVALID 失败且不搜索。
- [ ] 一轮定向补搜后仍不充分的 claim 必须经过 typed ResolveUnsupportedClaims，之后 Citation Guard 才可通过。
- [ ] 任意 report citation 都能回到现有 EvidenceSpan、SourceDocument、HTML paragraph/PDF page+block locator、raw/parsed/excerpt hash。
- [ ] Replay miss、ReplayTextEmbedder、预算耗尽、冲突补搜、checkpoint 恢复和幂等调用均有测试证据，strict Replay 不构造 Live embedder。
- [ ] LangGraph state 不包含完整正文、原始 Provider 响应、密钥或隐藏思维链。

## Execution Handoff

Plan complete. Choose one execution mode before implementation:

1. Subagent-Driven (recommended): use superpowers:subagent-driven-development in this session, one fresh worker and two-stage review per task.
2. Inline Execution: start a separate implementation session with superpowers:executing-plans and run tasks sequentially at the documented checkpoints.

Do not start formal benchmark execution until every verification item above is satisfied.
