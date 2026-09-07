# Evaluation protocol

This page records the reproducible benchmark procedure.  It is intentionally
usable before a formal result is sealed: absent artifacts remain marked as
missing, and this repository does not manufacture human, external, or CI data.

## Commands

```powershell
uv run deepresearch experiment config id --config benchmarks/configs/formal.yaml
uv run deepresearch experiment run --config benchmarks/configs/formal.yaml --variants A,B,C,D
uv run deepresearch experiment summarize --experiment-dir experiments/<group-id>
uv run python -m benchmarks.scripts.render_results --experiment-dir experiments/<group-id> --docs-dir docs
```

For the optional Portfolio extension, pass a separately sealed external
directory with `--external-experiment-dir`.  Primary confidence intervals are
never pooled with external metrics.

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

## Budget and subset policy

The formal configuration fixes the primary budget, sensitivity presets, task
subsets, seeds/repeats, model/environment locks, pricing snapshot, and
evaluation timestamp.  A rerun must use the same sealed values.  Cost is
labelled estimated when it comes from the normalized pricing schedule.

## Reproducibility limitations

The checked-in SVGs are deterministic, accessible placeholders until a public
formal summary is available.  Human ratings and external 10/20/10 results are
optional aggregate inputs; no score is imputed when those inputs are absent.
