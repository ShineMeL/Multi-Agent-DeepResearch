# Research-v1 production demo implementation plan

> Execute task-by-task with TDD and a review checkpoint after each task.

**Goal:** Compose a deterministic, auditable `research-v1` Replay Showcase
using the existing P1/R1 Core pipeline and make the demo/UI documentation
truthful.

**Spec:** `docs/superpowers/specs/2026-09-11-research-v1-production-demo-design.md`

## Constraints

- Do not change baseline-v1 semantics, replay fixture bytes, pricing, provider
  identity, service authorization, or checkpoint continuation behavior.
- Do not use the previously exposed Kimi key or any live/paid provider.
- Research-v1 accepts only the explicitly implemented P1/R1 deterministic
  composition; P2/R2 remains a stable fail-closed capability error.
- Every new artifact is canonical and content-addressed; unsupported claims are
  never included as verified findings.

### Task 1 — research graph composition seam and deterministic handlers

Files:

- Modify `src/deepresearch/workflow/research_graph.py`
- Create `src/deepresearch/workflow/research_handlers.py`
- Create `tests/unit/workflow/test_research_handlers.py`
- Create/extend `tests/integration/replay/test_research_v1_production.py`

1. Add a typed optional node-wrapper/audit-composition seam to
   `ResearchGraphDependencies`; keep the existing test dependency behavior
   unchanged when the seam is absent.
2. Implement a production facade around `BaselineNodeHandlers` that delegates
   shared nodes, runs deterministic `ClaimExtractor`/`EvidenceJudge`, persists
   a canonical claim graph, and resolves unsupported claims conservatively.
3. Add failing tests first for canonical claim artifacts, supported/unsupported
   routing, and research-state preservation; then implement the smallest code
   that passes.

### Task 2 — wire the builder and service path

Files:

- Modify `src/deepresearch/runtime/runner_factory.py`
- Extend `tests/unit/runtime/test_runner_factory.py`
- Extend `tests/unit/runtime/test_runner_factory_execution.py`

1. Add a failing composition test proving that a valid replay P1/R1 config
   builds a runner with a research graph and that P2/R2 remains fail-closed.
2. Build the research graph from the same frozen routes, pricing, checkpointer,
   content boundary, and audit composition as baseline.
3. Execute one end-to-end replay through the runner and assert completion,
   manifest validity, and no provider calls beyond the replay bundle.

### Task 3 — public demo/UI/docs and release verification

Files:

- Modify `apps/ui/replay.py`, `apps/ui/app.py`, `docs/deployment.md`, `README.md`
- Create/extend `tests/integration/replay/test_showcase_ui.py`
- Update this plan and the ignored SDD ledger with evidence.

1. Add a research-v1 replay payload that explicitly selects P1/R1; keep the
   existing baseline payload as a compatibility fixture.
2. Update UI copy and deployment instructions to distinguish deterministic
   research demo from unavailable P2/R2/live paths.
3. Run focused tests, Ruff, Pyright, readiness/preflight commands, and the
   offline HTTP/SSE demo. Record Docker/Postgres, online Provider, and formal
   benchmark gates as environment-blocked rather than passing.

### Exit criteria

- Research-v1 P1/R1 replay completes through the public API with durable events
  and artifacts.
- Baseline replay tests remain green for all applicable tests.
- New code passes focused pytest, Ruff, Pyright, and `git diff --check`.
- No secret values or fixture rewrites appear in the worktree or commit.
- Release preflight output remains honest: external gates are blocked for their
  missing prerequisites, and `docs/results.md` remains unsealed.

## Execution evidence (2026-09-11)

- Research/UI/runtime focused suite: **69 passed**.
- Cross-module offline regression suite (`tests/unit`, `tests/contracts`,
  replay/API/deployment integration, excluding `online`): **1,752 passed, 5
  skipped, 1 deselected**. The only output warning is the existing Starlette/
  HTTPX deprecation notice.
- `pyright src apps`: **0 errors, 0 warnings, 0 informations**; `ruff check .`
  and `git diff --check` pass.
- The public showcase now selects the deterministic P1/R1 `research-v1` graph.
  Strict replay executes `ExtractClaims` and `VerifyClaims`, preserves the
  research state through durable-event recovery, and emits a Core-valid
  `research-graph-v1` manifest. Baseline-v1 remains available unchanged.
- P2/R2, live/paid provider smoke, Docker/PostgreSQL startup, formal benchmark
  inputs, and publication sealing remain explicitly blocked by missing external
  prerequisites; no result or credential is fabricated.
