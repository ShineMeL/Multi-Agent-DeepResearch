# Multi-Agent DeepResearch

This repository is the reproducible Core foundation for a multi-agent research
system. The delivered baseline compiles a LangGraph workflow with a fixed P1
planner and R1 evidence ranker, runs it against a strict offline replay bundle,
and exports a citation-backed Markdown report.

The current milestone is intentionally narrow: Core foundation / P1+R1 strict
Replay. It does not claim trained planner or ranker variants, A/B/C/D experiments,
empirical quality or cost metrics, confidence intervals, manual review, or a
public service/UI deployment. Those are follow-on milestones.

## Quickstart

Use Python 3.12 and the locked dependencies. With a standalone `uv` executable:

```powershell
uv sync --extra dev
```

In environments where only the pinned module is available, the equivalent form
is `python -m uv sync --extra dev`. The project requires `==3.12.*`; verify the
lock before running a clean checkout:

```powershell
uv lock --check
```

Live runs read credentials from `.env`, never from a request or command-line
argument. Start from the empty template and fill it locally (do not commit the
result):

```powershell
Copy-Item .env.example .env
```

The strict Replay path is fully offline and needs no credentials, network, model
download, or Settings construction:

```powershell
uv run deepresearch research `
  --question "Compare planner strategies" `
  --mode replay `
  --replay-root tests/fixtures/replay/baseline `
  --budget medium `
  --checkpoint-db artifacts/replay-check.sqlite3 `
  --output artifacts/replay-check
```

The output directory must not already exist. Choose a fresh checkpoint/output
pair for each manual run. A successful run prints `status=completed` and
`stop_reason=SUFFICIENT`, then writes exactly:

```text
report.md
evidence.json
run-manifest.json
```

`report.md` cites evidence with IDs of the form `[E-<64 lowercase hex>]` and
contains a canonical `## References` section. Each cited ID resolves to an
`EvidenceSpan` in `evidence.json`; raw page bodies are not exported.

A Live invocation uses the same command surface, but it is not an offline demo
and can incur provider charges. It requires valid local `.env` settings plus the
verified embedding lock/model files:

```powershell
uv run deepresearch research `
  --question "Compare planner strategies" `
  --mode live `
  --budget low `
  --checkpoint-db artifacts/live-check.sqlite3 `
  --output artifacts/live-check
```

Keep the budget explicit for Live work. The local-unpriced policy records live
cost as unknown when no approved pricing catalog is configured; it does not
pretend that external calls are free.

## Architecture boundaries

- `deepresearch.domain` owns the canonical request, plan, evidence, usage, event,
  configuration, and result models.
- `deepresearch.providers.protocols` defines the async provider contracts:
  model, search, fetch, parser, embedder, and reranker calls receive an absolute
  deadline and cancellation token. SDK-specific code stays below
  `deepresearch.providers`.
- `deepresearch.runtime.ports` owns runner/checkpoint ports. Runtime budget,
  cache, checkpoints, manifests, and content-addressed stores keep large bodies
  outside graph state.
- `deepresearch.workflow` composes the compiled LangGraph baseline. State stores
  IDs, typed summaries, counters, and decisions—not raw provider responses,
  fetched bodies, credentials, or SDK objects.
- `apps/cli/main.py` is the composition boundary. It validates options, binds a
  content-addressed provider profile, selects Replay/Live providers, holds the
  derived runtime lock, validates public artifacts, and publishes the three
  output files without overwriting an existing target.

Strict Replay rejects unknown request keys and never falls back to Live. The
recording/resume surface and richer service composition are documented follow-on
work; this baseline quickstart only claims the tested offline path above.

## Roadmap and design documents

- [System design](docs/superpowers/specs/2026-08-29-multi-agent-deep-research-design.md)
- [Core foundation and replay baseline](docs/superpowers/plans/2026-08-29-core-foundation-replay-baseline.md)
- [Planner and evidence optimization](docs/superpowers/plans/2026-08-29-planner-evidence-optimization.md)
- [Benchmark and evaluation](docs/superpowers/plans/2026-08-29-benchmark-evaluation.md)
- [Service demo and deployment](docs/superpowers/plans/2026-08-29-service-demo-deployment.md)

The planner/evidence plan owns P2/R2 and the optimization experiments. The
benchmark plan owns baselines, metrics, confidence intervals, failure analysis,
and cost reporting. The service plan owns the user-facing UI/API and deployment.

## Offline quality gates

The repository is intended to pass these gates without external calls:

```powershell
uv run ruff check .
uv run pyright src apps
uv run pytest tests/unit tests/contracts tests/integration/replay tests/cli -q
uv run python -m compileall -q src apps benchmarks experiments
```

Use the `python -m uv ...` spelling when the standalone executable is not on
`PATH`.
