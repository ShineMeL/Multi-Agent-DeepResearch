# Research-v1 production demo design

## Goal

Make the already-shipped `research-v1` workflow runnable in the credential-free
Replay Showcase without weakening the audited baseline workflow. The first
production composition is intentionally deterministic: it reuses the existing
P1 fixed planner, R1 similarity retrieval, and report writer, then adds the
research-only claim extraction, evidence verification, and unsupported-claim
resolution stages using the deterministic Core implementations. P2/R2 model
optimization remains a separate follow-up and is not advertised by this
release.

## Boundaries

- `baseline-v1` remains byte- and behavior-compatible.
- `research-v1` is accepted only for the supported P1/R1 replay composition;
  unsupported planner/ranker combinations fail closed with the existing stable
  capability error.
- Replay routes, pricing snapshots, provider identities, and `replay_parent`
  remain server-owned and unchanged.
- Research-only state is preserved in checkpoints and validated by
  `validate_research_state`; baseline audit receipts continue to cover the
  shared retrieval/report nodes through a narrow state-shadow adapter.
- Claims and links are stored as canonical, content-addressed artifacts. An
  unsupported claim is deleted or moved to a limitations note; it is never
  silently presented as verified.
- No live/paid provider, external search, credential, or fabricated benchmark
  result is introduced.

## Runtime shape

```text
Validate → Plan → Search/Fetch/Parse/Store/Rank loop → Draft
    → ExtractClaims → VerifyClaims
       ├─ supported → FinalizeCitations → PersistResults
       └─ unsupported → ResolveUnsupportedClaims → FinalizeCitations → PersistResults
```

The production builder creates the normal baseline handlers once, derives a
research handler facade from them, and compiles both graphs with the same
checkpointer and audit composition. A wrapper presents a validated baseline
state to the baseline audit envelope and merges research fields back into the
LangGraph state. This avoids a second event stream or a second persistence
protocol.

## Observable contract

- A research replay returns HTTP 202, durable node events, downloadable report,
  evidence graph, and manifest artifacts.
- The public manifest remains Core-valid and contains no research-only secret or
  prompt material.
- The run result uses the existing `RunResult` fields; research metadata remains
  checkpoint/audit state and does not alter the public API shape.
- UI copy names the current implementation accurately (`research-v1` demo,
  P1/R1 deterministic replay) and states that P2/R2/live execution are not
  enabled in this release.

## Failure policy

Missing artifacts, malformed claims, invalid evidence links, or unsupported
configuration produce the existing typed failure boundary. The builder never
falls back from `research-v1` to baseline or from replay to a provider.
