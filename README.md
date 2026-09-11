# Multi-Agent DeepResearch

This repository is the reproducible Core foundation for a multi-agent research
system. The delivered baseline compiles a LangGraph workflow with a fixed P1
planner and R1 evidence ranker, runs it against a strict offline replay bundle,
and exports a citation-backed Markdown report.

The current checkout keeps the Core foundation / P1+R1 strict Replay path
reproducible. Benchmark publication wiring is present, but formal A/B/C/D
quality, cost, confidence-interval, external, and human results remain
unsealed unless a hash-verified public summary is supplied. This checkout does
not claim formal A/B/C/D results or a hosted Live endpoint.

## Navigation

- [Architecture](#architecture)
- [Why Planner and Ranker matter](#why-planner-and-ranker-matter)
- [Replay quickstart](#replay-quickstart)
- [Live/local quickstart](#livelocal-quickstart)
- [Benchmark protocol](#benchmark-protocol)
- [Results](#results)
- [Reproduction](#reproduction)
- [Trade-offs and limitations](#trade-offs-and-limitations)
- [Demo and deployment plan](#demo-and-deployment-plan)

## Replay quickstart

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

Hash-addressed Replay fixtures require their checked-in LF bytes. Use a clean LF
checkout (`core.autocrlf=false` before checkout) and do not update recorded
hashes to accommodate CRLF conversion. Before starting either local Replay
path, verify the packaged catalog, pricing, and public baseline bundle:

```powershell
uv run python scripts/release_readiness.py
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

## Live/local quickstart

The Live command above is the local quickstart. It requires credentials and can
incur provider charges; use Replay when validating the repository or CI path.

## Compose replay showcase

The Compose stack runs the FastAPI service, API-only Streamlit UI, and Postgres
with named `artifact-data` and `postgres-data` volumes. Set
`POSTGRES_PASSWORD`, an independently URI-encoded `DATABASE_URL`, and a
nonblank, 32-byte-or-longer `SESSION_SIGNING_KEY` in your shell or an
uncommitted environment file before starting it. Construct the URL by encoding
the password as a URI component; this matters for passwords containing `@`,
`:`, `/`, `#`, or `%`:

```powershell
$env:POSTGRES_PASSWORD = "replace-with-a-local-password"
$encodedPassword = [uri]::EscapeDataString($env:POSTGRES_PASSWORD)
$env:DATABASE_URL = "postgresql+asyncpg://deepresearch:${encodedPassword}@postgres:5432/deepresearch"
$env:SESSION_SIGNING_KEY = "replace-with-a-local-signing-key-at-least-32-bytes"
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

The Compose service uses the official image's named `postgres` account rather
than a brittle numeric UID, so its entrypoint can initialize the named data
volume while the database remains non-root.

This is a local replay showcase with `DEPLOYMENT_ACCESS_PROFILE=local`. It
forces only the `replay` execution mode and the packaged `replay-default`
provider profile, so it needs no model or search credential. The image contains
`deploy/replay/profiles.json`, `deploy/replay/pricing.json`, and the public
baseline bundle at `tests/fixtures/replay/baseline`. Submit the recorded input
`Compare planner strategies`; arbitrary questions produce a Replay miss, never
a Live fallback. The UI cannot supply or override server policy or provider
routes. The showcase uses the supported deterministic `research-v1` P1/R1
composition, which adds claim extraction and evidence verification to the
audited retrieval/report pipeline; the `baseline-v1` payload remains available
for compatibility. Research-v1 P2/R2 (the API defaults) is still rejected with
`RESEARCH_GRAPH_UNAVAILABLE` rather than silently falling back, and interrupted
runs whose real Core checkpoint is not resumable are rejected with
`CHECKPOINT_RESUME_UNAVAILABLE`.

Kimi Live is a separate, model-only opt-in. Configure its key server-side as
`MODEL_API_KEY`, provide a separate Tavily search route and a complete pricing
catalog, and use a Live-specific provider profile. Native Kimi
`$web_search` is not supported by this service.

For public deployment, inject database and session secrets from the platform's
secret store; do not add them to an image or Compose file. Set the server
deployment profile to `public_live`, enable `COOKIE_SECURE=true`, and explicitly
set the provider/mode/purpose/budget allowlists and trusted-proxy CIDRs for that
deployment.

## Architecture

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
shipped service composition supports the packaged baseline Replay path.
Creating new recordings and resumable Core checkpoints remain follow-on work;
interrupted service runs can therefore return `CHECKPOINT_RESUME_UNAVAILABLE`,
as documented above.

## Why Planner and Ranker matter

The planner controls search breadth, redundancy, and stopping decisions. The
evidence ranker controls which retrieved spans support a claim. They are
reported as separate protocols so a quality change is not incorrectly
attributed to a resource reduction or to the other component.

## Benchmark protocol

The benchmark uses fixed task/snapshot hashes, sealed model and environment
locks, explicit budgets, and deterministic replication. Seeds are aggregated at
the task level before paired confidence intervals. Optional human and external
results remain separate from the primary agent intervals. See the [evaluation
protocol](docs/evaluation.md) for commands, isolation boundaries, formulas, and
missingness rules.

## Results

The [results page](docs/results.md) is generated only from a verified public
summary. Until formal aggregates are sealed, it says so explicitly and retains
negative-result analysis. The checked-in figures are deterministic placeholders
with accessible descriptions; they do not represent formal CI, human, or
external data.

![Citation support versus estimated USD (deterministic placeholder; no formal result is sealed)](docs/assets/results/citation-support-vs-usd.svg)

![Information completeness versus search calls (deterministic placeholder; no formal result is sealed)](docs/assets/results/completeness-vs-search.svg)

![A/B/C/D metric overview (deterministic placeholder; no formal result is sealed)](docs/assets/results/abcd-metrics.svg)

## Reproduction

Use the [benchmark plan](docs/superpowers/plans/2026-08-29-benchmark-evaluation.md)
and the commands in [evaluation.md](docs/evaluation.md). A formal publication
requires the same sealed config, manifest hashes, model/environment locks,
replication policy, and evaluator version; a second renderer run must be byte
stable.

## Trade-offs and limitations

Hash verification and process isolation make publication auditable, but they
also mean missing human/external artifacts cannot be filled in by the renderer.
USD values are labelled estimated when derived from the approved pricing
schedule. The renderer never reads private gold, sealed prompts, raw provider
responses, or credentials.

## Demo and deployment plan

The user-facing Streamlit/Gradio service is specified separately in the
[service demo and deployment plan](docs/superpowers/plans/2026-08-29-service-demo-deployment.md).
No hosted service or live provider endpoint is claimed by this checkout.

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
