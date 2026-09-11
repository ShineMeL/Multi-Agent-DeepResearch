# Release B/C Execution and Publication Design

## Goal

Finish the deployable service validation and the reproducible benchmark
publication path without manufacturing benchmark values, weakening replay
integrity, or exposing provider credentials.

This design treats **Release B** as live/deployed service verification and
**Release C** as formal benchmark execution plus publication. Release A remains
the credential-free replay showcase already merged into `main`.

## Current baseline and blockers

The implementation starts from `main` merge commit `2731b6d`, which contains
Release A commit `74a2f21`. The service-scope offline gate is already present;
this work must not change its public replay policy.

The current machine has no `docker`, `docker compose`, `postgres`, `psql`,
`pg_isready`, or `vllm` executable. The repository also intentionally lacks the
formal model/environment locks, the private Frozen AI/CS Research 60 manifest,
the external benchmark lock, and `formal.yaml`/`formal-portfolio.yaml`.
`benchmarks/configs/external.yaml` contains review-only `example.invalid`
sources and pending licenses. These are prerequisites, not values that can be
inferred or replaced with placeholders.

The previously exposed Kimi key is considered compromised and is not an input
to this release. A rotated key must be supplied through a secret manager or
process environment only. No command in this design prints secret values.

## Scope

### Release B — deployed and live service verification

Release B has two independently reportable gates:

1. **B1 deployment gate.** Validate the Docker/Compose topology, build the API
   and UI images, start the Postgres-backed stack with an independently
   URI-encoded `DATABASE_URL`, and verify live/readiness health, durable run
   completion, artifact downloads, owner isolation, idempotency, SSE replay,
   and graceful shutdown. The default packaged profile remains the zero-cost
   `replay-default` profile.
2. **B2 online smoke gate.** On an explicitly authorized manual or scheduled
   run, load a complete `online-smoke` provider/pricing catalog and require
   `MODEL_API_KEY`, `SEARCH_API_KEY`, and `SESSION_SIGNING_KEY` from the secret
   store. Execute only the bounded smoke scenario from
   `tests/integration/deployment/test_smoke.py`, record estimated usage/cost,
   and publish only redacted public artifacts. Missing credentials or an
   incomplete pricing set must skip before app construction/provider calls;
   neither condition is a pass.

Release B must never silently fall back from live to replay, accept client
route overrides, or treat a skipped online smoke as a completed live result.

### Release C — formal benchmark and publication

Release C is gated by immutable inputs and is split into four stages:

1. **C1 primary seal.** Obtain and verify the pinned Qwen/Qwen3-8B model lock,
   the exact vLLM 0.28.0 serving-environment lock, the private dataset manifest,
   and the clean result-affecting source tree. Generate and validate exactly
   one `benchmarks/configs/formal.yaml` from the template. The seal commit and
   all hashes become immutable; changing result-affecting code requires a new
   group ID and seal.
2. **C2 primary execution.** From the clean seal commit, run the strict replay
   and stop-path gates, then run the ranker, planner, A/B/C/D, stability,
   cost-sensitivity, P0, and evaluator-only ORACLE protocols with the exact
   sealed seed/repeat/budget policy. Verify every result manifest and aggregate
   before summarizing with 10,000 deterministic stratified paired bootstrap
   resamples. No raw prompts, private gold, provider responses, or credentials
   cross the publication boundary.
3. **C3 Portfolio and human evaluation.** Only after C2 is independently
   verified, acquire licensed immutable inputs for the three external adapters,
   restore and hash exactly 10/20/10 external snapshots, and create a separate
   `formal-portfolio.yaml` and experiment group. Separately collect three
   pseudonymous ratings for each of the 20 pre-registered A/D pairs. External
   metrics and human ratings remain separate aggregates; missing inputs are
   reported as missing and never imputed.
4. **C4 publication.** Run the deterministic renderer against hash-verified
   primary, optional external, and optional human aggregates. Update
   `docs/results.md`, the SVGs, and README only when all referenced manifests,
   configuration hashes, and sidecar hashes verify. Preserve negative results,
   confidence intervals, failure breakdowns, limitations, and provenance.

## Architecture and interfaces

The existing service and benchmark modules remain the owners of runtime
behavior. Release work adds only orchestration and verification around them:

- `scripts/release_preflight.py` (new) reports a machine-readable capability
  matrix for Docker/Compose, Postgres, GPU/vLLM, required lock files, private
  manifests, external licenses, and secret *presence* (never values). It exits
  non-zero for a requested gate whose prerequisites are absent and explains
  whether the gate is `blocked`, `skipped`, or `ready`.
- `scripts/release_b_gate.py` (new) wraps the documented B1 commands and
  records health, Compose, and artifact assertions without changing the API
  composition root. It accepts an explicit profile (`replay` or `online-smoke`)
  and refuses an unconfigured profile before any provider call.
- `scripts/release_c_gate.py` (new) validates C1–C4 inputs and delegates actual
  benchmark execution to the existing `deepresearch experiment` CLI and
  renderer. It must fail closed when locks, private data, licenses, human
  aggregates, or manifest sidecars are absent.
- The existing `.gitattributes` remains authoritative for hash-addressed files.
  A clean-LF checkout is a prerequisite for replay verification on Windows;
  no script may rewrite a tracked fixture or regenerate a manifest hash to
  bless CRLF bytes.
- Existing CI keeps normal verification secret-free and keeps B2 in a manual or
  scheduled job. Release C credentials and private data never enter CI logs,
  images, browser code, or committed files.

The new scripts are diagnostics/orchestration only. They do not introduce a
second provider registry, rate limiter, lifecycle manager, result schema, or
statistical implementation.

## State and failure rules

Each gate emits a stable status and a reason code:

| Gate | Ready condition | Missing-input behavior |
| --- | --- | --- |
| B1 | Docker/Compose, Postgres stack, URI-safe DSN, and health/artifact checks pass | `DEPLOYMENT_PREREQUISITE_MISSING`; do not claim deployment |
| B2 | Complete live catalog/pricing plus all rotated secrets; bounded smoke completes | `ONLINE_SMOKE_NOT_AUTHORIZED` or `ONLINE_SMOKE_INCOMPLETE`; skip before provider calls |
| C1 | Model, environment, dataset, and clean-tree hashes produce validated formal seal | `FORMAL_INPUT_MISSING` or `FORMAL_TREE_DIRTY`; do not create a partial seal |
| C2 | Every configured protocol record and manifest verifies | `FORMAL_RESULT_INVALID`; retain failed run artifacts for diagnosis |
| C3 | Licensed 10/20/10 snapshots and 20×3 valid ratings verify | `PORTFOLIO_INPUT_MISSING` or `HUMAN_AGGREGATE_INCOMPLETE`; keep primary separate |
| C4 | Renderer verifies every input and is byte-stable on a second run | `PUBLICATION_UNSEALED`; leave `docs/results.md` explicitly unsealed |

Interrupted or failed gates retain their evidence and never overwrite a
completed group. A result-affecting source or lock change invalidates the
current seal and requires a new group. A skipped gate is never converted to a
zero, pass, or completed result.

## Verification requirements

Before any Release B/C claim, run and record:

1. `uv lock --check --offline` and the relevant locked dependency sync.
2. Release A readiness and the service-scope offline suite.
3. The new preflight command with a missing-input fixture and a complete
   synthetic replay profile; confirm no secret values appear in output.
4. B1 only on a host with Docker/Compose/Postgres; B2 only through the
   authorized online job and rotated secrets.
5. C1–C4 only from clean LF checkouts and immutable, independently supplied
   model/data/license/rating inputs; run the existing unit/contracts suites plus
   the exact gate-specific integration tests.
6. `ruff`, configured Pyright, `compileall`, `git diff --check`, and a second
   renderer run whose output hashes are identical.

The current repository remains honest until these prerequisites exist:
`docs/results.md` continues to state that the primary result is not sealed,
and deterministic SVG placeholders remain labeled as placeholders.

## Non-goals

- Do not install system software, enable Docker, provision Postgres, or incur
  provider charges without an available authorized environment and explicit
  credentials.
- Do not use or repeat the previously exposed Kimi key.
- Do not synthesize benchmark scores, human ratings, external snapshots,
  licenses, model locks, or confidence intervals.
- Do not change Core replay semantics, implement unsupported checkpoint
  continuation, or relax SSRF, owner, redaction, pricing, or admission rules.
- Do not publish `docs/results.md` as sealed merely because the orchestration
  scripts or tests pass.

