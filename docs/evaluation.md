# Evaluation protocol

This page records the reproducible benchmark procedure.  It is intentionally
usable before a formal result is sealed: absent artifacts remain marked as
missing, and this repository does not manufacture human, external, or CI data.

## Commands

The sealed `formal.yaml` and `formal-portfolio.yaml` paths below are generated
artifacts, not templates.  They are intentionally absent from this checkout
until the pinned model/environment locks and licensed external sources are
available; running the commands without those seals must fail closed.

```powershell
uv run deepresearch experiment config id --config benchmarks/configs/formal.yaml
uv run deepresearch experiment run --config benchmarks/configs/formal.yaml --variants A,B,C,D
uv run deepresearch experiment summarize --experiment-dir experiments/<group-id>
uv run python -m benchmarks.scripts.render_results --experiment-dir experiments/<group-id> --docs-dir docs
```

For the optional Portfolio extension, pass a separately sealed external
directory with `--external-experiment-dir`.  The directory must contain a
hash-verified `manifest.sha256` for `metrics.json`; primary confidence
intervals are never pooled with external metrics.

## Isolation and provenance

Agents receive RuntimeTask projections and hash-addressed snapshots.  The
evaluator owns private gold and rubric data in a separate process boundary.
Publication consumes only `summary.json`, the public `manifest.sha256`, and
other manifest-listed aggregate artifacts.  Raw run records, prompts, provider
responses, credentials, and private gold are outside this boundary.

## Metrics and statistics

Quality, evidence, efficiency, latency, token, cost, and failure metrics are
reported with their metric versions and missingness.  Seeds are aggregated at
the task level before paired comparisons.  Formal paired intervals use the
sealed budget and replication policy with 10,000 stratified bootstrap draws;
the planner non-inferiority margin is the pre-registered completeness margin.

The primary paired effect is computed after within-task seed aggregation:
`effect(task) = candidate(task) - baseline(task)`.  The reported estimate is
the mean task effect; each bootstrap replicate resamples tasks within category
and recomputes that mean.  A two-sided 95% interval is the 2.5th and 97.5th
percentiles of 10,000 deterministic replicates.  Planner non-inferiority is
accepted only when the lower bound is at least the sealed completeness margin.
Pareto dominance is the fraction of bootstrap replicates in which one variant
is no worse on quality while costing no more and using no more search calls.

The six human dimensions use 1–5 ordinal ratings.  Missing ratings remain
missing; no score is imputed.  Agreement is ordinal Krippendorff's alpha on
pairable task/rater units, and automatic-vs-human association is Spearman's
rank correlation on complete pairs only.

## Execution profile and budgets

The formal template records the intended endpoint and lock identities:
Qwen/Qwen3-8B at the pinned revision, local vLLM 0.28.0, bfloat16, one tensor
parallel worker, and a 32,768-token context.  The actual model, hardware,
CUDA/runtime, and environment hashes are reported only after the corresponding
lock files are captured; this checkout does not claim that profile is sealed.

| Preset | Max search calls | Max pages | Max tokens | Wall time | Estimated USD cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| low | 4 | 8 | 20,000 | 180 s | 0.25 |
| medium | 8 | 12 | 40,000 | 300 s | 0.50 |
| high | 12 | 20 | 70,000 | 480 s | 1.00 |

Costs are labelled estimated when derived from the approved pricing snapshot;
they are not treated as observed provider billing.

## Budget and subset policy

The formal configuration fixes the primary budget, sensitivity presets, task
subsets, seeds/repeats, model/environment locks, pricing snapshot, and
evaluation timestamp.  A rerun must use the same sealed values.  Cost is
labelled estimated when it comes from the normalized pricing schedule.

## Reproducibility limitations

The checked-in SVGs are deterministic, accessible placeholders until a public
formal summary is available.  Human ratings and external 10/20/10 results are
optional aggregate inputs.  File-based human/external aggregates require a
sidecar hash manifest; no score is imputed when those inputs are absent.
