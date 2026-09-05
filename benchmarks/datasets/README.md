# Benchmark data access boundary

The evaluator is the only process that reads sealed gold records. Agent
processes receive one validated `RuntimeTask` and a snapshot directory; they
never receive the private benchmark root or a path that can resolve into it.

Public development runs use:

```text
DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT=benchmarks/datasets/frozen_ai_cs_60/runtime/dev
DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT=benchmarks/snapshots/frozen_ai_cs_60
```

Sealed formal runs use an isolated child directory:

```text
DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT=experiments/<group>/agent-inputs
DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT=benchmarks/snapshots/frozen_ai_cs_60
DEEPRESEARCH_BENCHMARK_GOLD_ROOT=benchmarks/private/frozen_ai_cs_60
```

The evaluator reads `benchmarks/private/frozen_ai_cs_60/runtime/test`, stages
one redacted task outside the private root for each child launch, and removes
`DEEPRESEARCH_BENCHMARK_GOLD_ROOT` from the child environment. The API, UI and
agent container never mount the private root. A sealed test question becomes
publishable only after its first formal result is frozen and hash-verified.

The probe entrypoints provide a permanent startup check:

```text
python -m benchmarks.processes.agent --probe-runtime-task TASK.json \
  --runtime-root RUNTIME --snapshot-dir SNAPSHOT --run-root RUN --output RUN/probe.json
python -m benchmarks.processes.evaluator probe-agent --private-task PRIVATE_TASK.json \
  --private-root benchmarks/private/frozen_ai_cs_60 \
  --snapshot-dir SNAPSHOT --agent-input-root AGENT_INPUT --run-root RUN
```
