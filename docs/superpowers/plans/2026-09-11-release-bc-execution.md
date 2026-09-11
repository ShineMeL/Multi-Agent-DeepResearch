# Release B/C Execution and Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fail-closed, secret-safe orchestration for Release B deployment/live checks and Release C formal benchmark publication, then execute every gate that the supplied environment and authorized inputs can support.

**Architecture:** Three small standard-library orchestration modules expose deterministic reports and delegate runtime behavior to the existing Compose, service, benchmark CLI, and renderer interfaces. A capability preflight runs before any external command; missing tools, credentials, model/data locks, licenses, or aggregates produce stable blocked/skipped reasons and never create synthetic results. Existing service, benchmark, replay, and statistical modules remain the only owners of domain behavior.

**Tech Stack:** Python 3.12, `argparse`, `dataclasses`, `subprocess`, `shutil`, `hashlib`, existing `uv`/pytest/Ruff/Pyright tooling, Docker Compose, FastAPI service, and the existing `deepresearch experiment` and `render_results` CLIs.

**Spec:** `docs/superpowers/specs/2026-09-11-release-bc-execution-design.md`

## Global Constraints

- Release A replay defaults remain credential-free and zero-cost: `ALLOWED_EXECUTION_MODES=["replay"]`, `ALLOWED_PROVIDER_PROFILE_IDS=["replay-default"]`, purposes `demo`/`test`, and low/medium budgets.
- A skipped or blocked gate is never reported as a pass, zero score, completed run, or sealed result.
- Provider secrets are checked by presence only; values never enter reports, subprocess output, fixtures, logs, images, or Git.
- The previously exposed Kimi key is compromised and must not be read or used; live smoke accepts only a newly rotated secret supplied by the operator/secret store.
- Hash-addressed replay fixtures remain byte-for-byte immutable. Windows verification uses a clean LF checkout; no command rewrites fixtures or regenerates hashes to bless CRLF.
- Do not change Core replay semantics, unsupported checkpoint continuation, service lifecycle/authorization, pricing/admission, SSRF, owner isolation, redaction, or benchmark statistical formulas.
- `docs/results.md` remains explicitly unsealed until C1–C4 inputs and manifest/sidecar hashes verify.
- New code follows TDD: each behavior gets a failing test, the expected failure is observed, then the minimal implementation is added and re-run.

---

### Task 1: Implement capability preflight and stable gate reports

**Files:**
- Create: `scripts/release_preflight.py`
- Test: `tests/unit/test_release_preflight.py`

**Interfaces:**
- Produces `GateName = Literal["b1", "b2", "c1", "c2", "c3", "c4"]` and `GateStatus = Literal["ready", "blocked", "skipped"]`.
- Produces frozen `CheckResult(name: str, present: bool, detail: str)` and `GateReport(gate: GateName, status: GateStatus, reason: str | None, checks: tuple[CheckResult, ...])` dataclasses.
- Produces `assess_gate(repository: Path, gate: GateName, *, experiment_dir: Path | None = None, external_experiment_dir: Path | None = None, human_summary: Path | None = None, environ: Mapping[str, str] | None = None, command_exists: Callable[[str], bool] = shutil.which) -> GateReport`.
- CLI contract: `python scripts/release_preflight.py --gate {b1,b2,c1,c2,c3,c4,all} [--repository PATH] [--experiment-dir PATH] [--external-experiment-dir PATH] [--human-summary PATH] [--format text|json]`; exit `0` only when every requested report is `ready`, otherwise exit `1` with a stable reason and no secret values.

- [x] **Step 1: Write the failing tests for statuses, secret-safe output, and all gates**

```python
def test_b1_is_blocked_without_docker(tmp_path, monkeypatch):
    report = assess_gate(
        tmp_path,
        "b1",
        environ={"DATABASE_URL": "postgresql+asyncpg://user:p%40ss@db/research"},
        command_exists=lambda name: False,
    )
    assert report.status == "blocked"
    assert report.reason == "DEPLOYMENT_PREREQUISITE_MISSING"
    assert all("p%40ss" not in check.detail for check in report.checks)


def test_b2_reports_presence_without_secret_values(tmp_path):
    env = {
        "MODEL_API_KEY": "new-secret-value",
        "SEARCH_API_KEY": "another-secret-value",
        "SESSION_SIGNING_KEY": "a" * 32,
        "PROVIDER_PROFILE_CATALOG_PATH": str(tmp_path / "providers.json"),
        "PRICING_CATALOG_PATH": str(tmp_path / "pricing.json"),
    }
    (tmp_path / "providers.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pricing.json").write_text("{}", encoding="utf-8")
    report = assess_gate(tmp_path, "b2", environ=env, command_exists=lambda _: True)
    encoded = json.dumps(report, default=lambda value: value.__dict__)
    assert "new-secret-value" not in encoded
    assert "another-secret-value" not in encoded
    assert "present" in encoded


def test_c1_requires_formal_inputs_and_lf_fixture_bytes(tmp_path):
    report = assess_gate(tmp_path, "c1", environ={}, command_exists=lambda _: True)
    assert report.status == "blocked"
    assert report.reason == "FORMAL_INPUT_MISSING"


def test_c4_is_not_ready_when_results_are_unsealed(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "results.md").write_text(
        "Factual outcome: primary result is not yet sealed.\n", encoding="utf-8"
    )
    report = assess_gate(tmp_path, "c4", environ={}, command_exists=lambda _: True)
    assert report.status == "blocked"
    assert report.reason == "PUBLICATION_UNSEALED"
```

- [x] **Step 2: Run the focused tests and verify they fail for the missing module/contract**

Run: `python -m uv run --no-sync pytest -q tests/unit/test_release_preflight.py`

Expected: collection fails because `scripts.release_preflight` and `assess_gate` do not exist yet.

- [x] **Step 3: Implement the minimal preflight module**

Use exact reason codes from the spec. B1 checks Docker/Compose commands and
`docker-compose.yml`; B2 checks the three required secret names plus catalog
files; C1 checks model/environment/R1 locks, private manifest, `formal.template`,
clean Git status, and CRLF-free hash-addressed fixtures; C2 checks a validated
`formal.yaml` and a supplied experiment directory; C3 checks external config,
external lock, portfolio config, external `metrics.json` plus its sidecar, and a
human aggregate plus its sidecar; C4 checks the primary `summary.json` and
manifest, optional separate external/human inputs, and returns
`PUBLICATION_UNSEALED` before other publication checks when `docs/results.md`
still contains its explicit unsealed marker. Details contain only booleans, relative labels, and
stable missing-file names. JSON output serializes the same fields in sorted
order; text output is one line per check and a final `release_gate ...` line.

- [x] **Step 4: Re-run the focused tests and the script contract**

Run: `python -m uv run --no-sync pytest -q tests/unit/test_release_preflight.py`

Expected: all focused tests pass, and `python scripts/release_preflight.py --gate all --format json` exits `1` with no credential values on this machine.

- [x] **Step 5: Run lint/type checks for the new module**

Run: `python -m uv run --no-sync ruff check scripts/release_preflight.py tests/unit/test_release_preflight.py --no-cache`

Expected: `All checks passed!`. Run `python -m compileall -q scripts` and expect exit `0`.

- [x] **Step 6: Commit the preflight unit**

```bash
git add scripts/release_preflight.py tests/unit/test_release_preflight.py
git commit -m "feat: add Release B and C capability preflight"
```

### Task 2: Add Release B deployment and online-smoke orchestration

**Files:**
- Create: `scripts/release_b_gate.py`
- Test: `tests/integration/deployment/test_release_b_gate.py`
- Modify: `docs/deployment.md`

**Interfaces:**
- Produces frozen `BStep(name: str, status: Literal["ready", "blocked", "failed"], detail: str)` and `BRunResult(status: Literal["ready", "blocked", "failed"], reason: str | None, steps: tuple[BStep, ...])` dataclasses, plus `run_b_gate(repository: Path, *, profile: Literal["replay", "online-smoke"], keep_up: bool = False, runner: Callable[..., CompletedProcess[str]] = subprocess.run) -> BRunResult`.
- The runner never receives raw secret values in command arguments. It accepts only server-side catalog paths and environment names.
- `--profile replay` performs B1; `--profile online-smoke` performs B2 after preflight. `--dry-run` prints the exact redacted command plan without running Docker/provider commands. Exit `0` only for `ready`.

- [x] **Step 1: Write failing tests for fail-closed ordering and command redaction**

```python
def test_b1_stops_before_docker_when_preflight_is_blocked(tmp_path, monkeypatch):
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "scripts.release_b_gate.assess_gate",
        lambda *args, **kwargs: GateReport("b1", "blocked", "DEPLOYMENT_PREREQUISITE_MISSING", ()),
    )

    result = run_b_gate(
        tmp_path,
        profile="replay",
        runner=lambda *args, **kwargs: calls.append(args),
    )
    assert result.status == "blocked"
    assert result.reason == "DEPLOYMENT_PREREQUISITE_MISSING"
    assert calls == []


def test_b1_runs_config_build_up_health_and_cleanup_in_order(tmp_path, monkeypatch):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.release_b_gate.assess_gate",
        lambda *args, **kwargs: GateReport("b1", "ready", None, ()),
    )
    calls: list[tuple[object, ...]] = []

    def fake_runner(*args, **kwargs):
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    result = run_b_gate(tmp_path, profile="replay", runner=fake_runner)
    assert result.status == "ready"
    assert [step.name for step in result.steps] == [
        "compose_config", "image_build", "stack_up", "health_live", "health_ready", "stack_down"
    ]


def test_b2_requires_complete_catalog_and_does_not_echo_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "secret-that-must-not-print")
    monkeypatch.setenv("SEARCH_API_KEY", "another-secret-that-must-not-print")
    monkeypatch.setenv("SESSION_SIGNING_KEY", "a" * 32)
    (tmp_path / "providers.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pricing.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROVIDER_PROFILE_CATALOG_PATH", str(tmp_path / "providers.json"))
    monkeypatch.setenv("PRICING_CATALOG_PATH", str(tmp_path / "pricing.json"))
    result = run_b_gate(tmp_path, profile="online-smoke", runner=lambda *args, **kwargs: None)
    payload = json.dumps(result, default=lambda value: value.__dict__)
    assert "secret-that-must-not-print" not in payload
    assert "another-secret-that-must-not-print" not in payload
    assert result.status == "blocked"
    assert result.reason == "ONLINE_SMOKE_INCOMPLETE"
```

- [x] **Step 2: Run the tests to observe the missing module failure**

Run: `python -m uv run --no-sync pytest -q tests/integration/deployment/test_release_b_gate.py`

Expected: collection fails because `scripts.release_b_gate` is absent.

- [x] **Step 3: Implement B1/B2 with explicit subprocess boundaries**

Use `assess_gate` before any external command. B1 executes the documented
sequence: `docker compose config --quiet`, `docker build --build-arg
DEEPRESEARCH_CODE_COMMIT=<validated HEAD> -t multi-agent-deep-research .`,
`docker compose up -d`, HTTP GET `/health/live`, HTTP GET `/health/ready`, and
`docker compose down` unless `keep_up=True`. A failed step returns
`DEPLOYMENT_COMMAND_FAILED` with the step name only; cleanup runs in `finally`.
B2 writes catalog JSON supplied by the operator to a mode-0600 temporary file,
sets only the path variables for the subprocess, runs
`uv run pytest -q tests/integration/deployment/test_smoke.py -m online`, and
deletes temporary files. It refuses missing credentials, missing catalog files,
or incomplete pricing before starting the service. Never include environment
values in `BStep.detail`.

- [x] **Step 4: Re-run focused B tests and a dry-run on this host**

Run: `python -m uv run --no-sync pytest -q tests/integration/deployment/test_release_b_gate.py`

Expected: focused tests pass. Then run:
`python scripts/release_b_gate.py --profile replay --dry-run`.

Expected: exit `1`, reason `DEPLOYMENT_PREREQUISITE_MISSING`, and no attempt to invoke Docker.

- [x] **Step 5: Document operator commands and limits**

Add a Release B section to `docs/deployment.md` that links the preflight and
gate commands, states that the default profile is replay, gives the manual
online-smoke invocation, requires a rotated secret set, and names the exact
`DEPLOYMENT_PREREQUISITE_MISSING`, `ONLINE_SMOKE_NOT_AUTHORIZED`, and
`ONLINE_SMOKE_INCOMPLETE` outcomes. State that `docker compose down` preserves
named volumes unless the operator explicitly chooses a destructive cleanup.

- [x] **Step 6: Run lint and commit B orchestration**

Run: `python -m uv run --no-sync ruff check scripts/release_b_gate.py tests/integration/deployment/test_release_b_gate.py --no-cache`

Expected: `All checks passed!`.

```bash
git add scripts/release_b_gate.py tests/integration/deployment/test_release_b_gate.py docs/deployment.md
git commit -m "feat: add Release B deployment gate"
```

### Task 3: Add Release C formal/portfolio/publication orchestration

**Files:**
- Create: `scripts/release_c_gate.py`
- Test: `tests/integration/benchmarks/test_release_c_gate.py`
- Modify: `docs/evaluation.md`

**Interfaces:**
- Produces `CStage = Literal["c1", "c2", "c3", "c4"]`, frozen `CStep(name: str, status: Literal["ready", "blocked", "failed"], detail: str)`, frozen `CRunResult(status: Literal["ready", "blocked", "failed"], reason: str | None, steps: tuple[CStep, ...])`, and `run_c_gate(repository: Path, *, stage: CStage, experiment_dir: Path | None = None, external_experiment_dir: Path | None = None, human_summary: Path | None = None, runner: Callable[..., CompletedProcess[str]] = subprocess.run) -> CRunResult`.
- C1 never writes `formal.yaml`; C2 delegates existing experiment commands only after C1 validation; C3 keeps external/human artifacts separate; C4 invokes the existing renderer only after every input and hash sidecar verifies.
- `--stage c1|c2|c3|c4`, `--experiment-dir`, `--external-experiment-dir`, and `--human-summary` are the complete CLI surface. Exit `0` only when the selected stage is ready.

- [x] **Step 1: Write failing tests for missing formal inputs, dirty trees, and publication stop conditions**

```python
def test_c1_does_not_create_partial_formal_config(tmp_path):
    output = tmp_path / "benchmarks/configs/formal.yaml"
    result = run_c_gate(tmp_path, stage="c1")
    assert result.status == "blocked"
    assert result.reason == "FORMAL_INPUT_MISSING"
    assert not output.exists()


def test_c2_never_launches_provider_when_formal_config_is_absent(tmp_path):
    calls: list[tuple[object, ...]] = []

    def recording_runner(*args, **kwargs):
        calls.append(tuple(args))
        raise AssertionError("formal gate must fail before command delegation")

    result = run_c_gate(tmp_path, stage="c2", experiment_dir=tmp_path / "group", runner=recording_runner)
    assert result.status == "blocked"
    assert result.reason == "FORMAL_INPUT_MISSING"
    assert calls == []


def test_c3_requires_separate_external_and_human_sidecars(tmp_path):
    result = run_c_gate(tmp_path, stage="c3")
    assert result.status == "blocked"
    assert result.reason in {"PORTFOLIO_INPUT_MISSING", "HUMAN_AGGREGATE_INCOMPLETE"}


def test_c4_keeps_unsealed_results_unchanged(tmp_path):
    results = tmp_path / "docs" / "results.md"
    results.parent.mkdir(parents=True)
    results.write_text("Factual outcome: primary result is not yet sealed.\n", encoding="utf-8")
    before = results.read_bytes()
    result = run_c_gate(tmp_path, stage="c4")
    assert result.status == "blocked"
    assert result.reason == "PUBLICATION_UNSEALED"
    assert results.read_bytes() == before
```

- [x] **Step 2: Run focused tests and verify the expected missing-module failure**

Run: `python -m uv run --no-sync pytest -q tests/integration/benchmarks/test_release_c_gate.py`

Expected: collection fails because `scripts.release_c_gate` is absent.

- [x] **Step 3: Implement C1–C4 validation and delegation**

C1 checks `qwen3-8b.lock.json`, `inference-environment.lock.json`,
`models/embedding.lock.json`, `benchmarks/private/frozen_ai_cs_60/private_manifest.json`,
the public manifest/snapshots, and a clean result-affecting Git tree. It also
rejects CRLF in every hash-addressed fixture and returns `FORMAL_TREE_DIRTY`
separately from `FORMAL_INPUT_MISSING`. C2 requires an existing validated
`formal.yaml` and group directory, then delegates the exact strict-replay and
formal protocol commands from `docs/evaluation.md`; it never accepts ad-hoc
seeds, budgets, or variants. C3 requires `external.yaml`, a verified external
lock and 10/20/10 snapshot output, plus a human aggregate validated by
`validate_human_ratings` with 20 tasks and 3 distinct raters; it never merges
external or human metrics into primary confidence intervals. C4 runs
`render_results` in a temporary output location twice, compares all output
hashes, and only then promotes deterministic public Markdown/SVG files. Any
missing or invalid manifest/sidecar returns `PUBLICATION_UNSEALED` without
touching `docs/results.md`.

- [x] **Step 4: Re-run focused C tests and verify the current machine is blocked honestly**

Run: `python -m uv run --no-sync pytest -q tests/integration/benchmarks/test_release_c_gate.py`

Expected: all focused tests pass. Then run:
`python scripts/release_c_gate.py --stage c1`.

Expected: exit `1` with `FORMAL_INPUT_MISSING` because the pinned model lock and private manifest are absent; no `formal.yaml` is created.

- [x] **Step 5: Document C execution order and publication stop conditions**

Update `docs/evaluation.md` with the C1→C4 gate commands, the clean-LF
requirement, the 10,000-resample command, the separate external/human inputs,
and the exact behavior when an input is missing. Keep all current statements
that formal YAML, model locks, external licenses, and ratings are intentionally
absent in this checkout.

- [x] **Step 6: Run lint/type checks and commit C orchestration**

Run: `python -m uv run --no-sync ruff check scripts/release_c_gate.py tests/integration/benchmarks/test_release_c_gate.py --no-cache`

Expected: `All checks passed!`. Run configured Pyright for `src apps benchmarks experiments`; expect zero diagnostics.

```bash
git add scripts/release_c_gate.py tests/integration/benchmarks/test_release_c_gate.py docs/evaluation.md
git commit -m "feat: add Release C publication gate"
```

### Task 4: Make Windows LF verification reproducible and wire the gates into CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/deployment.md`
- Modify: `docs/evaluation.md`
- Test: `tests/contracts/test_release_runbook.py`

**Interfaces:**
- CI invokes `python scripts/release_preflight.py --gate c1 --format json` in a clean Linux checkout and retains B2 as a manual/scheduled job.
- Runbooks use `git -c core.autocrlf=false clone --branch main https://github.com/ShineMeL/Multi-Agent-DeepResearch.git "$env:TEMP\\deepresearch-lf"` or an equivalent fresh LF checkout; they never call a hash-rewriting repair command.

- [x] **Step 1: Write failing runbook/CI contract tests**

```python
def test_runbooks_instruct_clean_lf_checkout_and_never_regenerate_hashes():
    deployment = Path("docs/deployment.md").read_text(encoding="utf-8")
    evaluation = Path("docs/evaluation.md").read_text(encoding="utf-8")
    combined = deployment + evaluation
    assert "core.autocrlf=false" in combined
    assert "Do not regenerate expected hashes" in combined
    assert "release_preflight.py" in combined


def test_ci_keeps_online_smoke_manual_and_secret_free_verify_job():
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert "secrets" not in json.dumps(workflow["jobs"]["verify"])
    triggers = json.dumps(workflow.get(True, workflow.get("on", {})))
    assert "workflow_dispatch" in triggers
    assert "schedule" in triggers
    assert "online-smoke" in json.dumps(workflow["jobs"])
```

- [x] **Step 2: Run the tests and observe the missing runbook/CI references**

Run: `python -m uv run --no-sync pytest -q tests/contracts/test_release_runbook.py`

Expected: the assertions for `release_preflight.py` fail before documentation/CI changes.

- [x] **Step 3: Add exact LF and gate instructions**

Document a PowerShell-safe fresh checkout, the preflight command, and the
canonical replay readiness invocation. Add the C preflight to the secret-free
CI verification job without trying to make absent private inputs pass. Keep the
online job conditional on all five configured secret/catalog inputs and retain
its manual/scheduled trigger. Do not add secrets to the normal verification
job.

- [x] **Step 4: Re-run contract tests, lint, and diff checks**

Run: `python -m uv run --no-sync pytest -q tests/contracts/test_release_runbook.py`

Expected: all tests pass. Run `python -m uv run --no-sync ruff check . --no-cache`, `python -m uv run --no-sync pyright src apps benchmarks experiments`, `python -m compileall -q scripts`, and `git diff --check`; all must exit `0`.

- [x] **Step 5: Commit the reproducibility wiring**

```bash
git add .github/workflows/ci.yml docs/deployment.md docs/evaluation.md tests/contracts/test_release_runbook.py
git commit -m "docs: wire Release B and C reproducibility gates"
```

### Task 5: Execute environment-dependent gates and publish only verified results

**Files:**
- No source edits are authorized until the corresponding external prerequisite is verified.
- Runtime-only outputs belong under ignored `experiments/<group>/`, temporary secret stores, or operator-managed Docker volumes.

**Interfaces:** Consumes the Task 1–4 gate commands and existing service/benchmark contracts. Produces only verified run manifests, summaries, and (for a successful C4) public docs assets.

- [x] **Step 1: Re-run the capability matrix from a clean LF checkout**

Run:

```powershell
python scripts/release_preflight.py --gate all --format json
```

Record each gate as `ready`, `blocked`, or `skipped`; do not edit `docs/results.md` based on this command.

- [x] **Step 2: Execute B1 only when Docker/Compose is available** — recorded `DEPLOYMENT_PREREQUISITE_MISSING`; this host has no Docker/Compose.

Run:

```powershell
python scripts/release_b_gate.py --profile replay
```

On a host without Docker, retain `DEPLOYMENT_PREREQUISITE_MISSING` as the
result and report the host limitation. On a capable host, preserve the Compose
health/artifact/SSE evidence and use a separate URI-encoded Postgres password.

- [x] **Step 3: Execute B2 only with rotated authorized credentials** — recorded `ONLINE_SMOKE_NOT_AUTHORIZED`; no provider call was made.

Supply a complete `online-smoke` catalog and prices through the secret store,
then run:

```powershell
python scripts/release_b_gate.py --profile online-smoke
```

If any required input is absent, retain the stable skip/block reason and make no
provider call. Never use the compromised Kimi key.

- [x] **Step 4: Execute C1/C2 only after model/data locks exist** — recorded `FORMAL_INPUT_MISSING`; no formal config was created.

From a clean LF seal commit, run the existing lock, freeze, validate, strict
replay, formal protocol, and summarize commands in `docs/evaluation.md`. Verify
all manifest hashes and 10,000 bootstrap outputs before proceeding.

- [x] **Step 5: Execute C3 only with licensed external data and human ratings** — recorded `PORTFOLIO_INPUT_MISSING`; no external or human result was fabricated.

Restore and verify exactly 10/20/10 external snapshots and a separate
20-task/3-rater human aggregate. Reject pending/example licenses, incomplete
sidecars, or duplicated raters; keep primary/external/human groups separate.

- [x] **Step 6: Run C4 and seal results only after all checks pass** — correctly stopped at `PUBLICATION_UNSEALED`; `docs/results.md` was not edited.

Run the renderer twice, compare output hashes, inspect all three SVGs, then
update `docs/results.md`, README links, and public figures from verified
aggregates. If any gate is blocked or skipped, leave the page's explicit
“primary result is not yet sealed” wording unchanged.

- [x] **Step 7: Final verification and handoff**

Run the service-scope offline suite, gate-specific tests, Ruff, configured
Pyright, `compileall`, `git diff --check`, and a clean-LF readiness check. Record
Docker/Postgres/provider/model/data limitations explicitly. Do not claim Release
B or C completion unless the corresponding gate exits `0` with verified output.

## Completion checklist

- [x] Preflight reports stable, secret-free statuses for B1/B2/C1/C2/C3/C4.
- [x] B1 Docker/Postgres evidence is captured, or its prerequisite block is recorded.
- [x] B2 online smoke is authorized and completed, or skipped before provider calls.
- [x] C1 formal inputs and seal are verified, or no partial seal is created.
- [ ] C2 primary results are manifest-verified with 10,000 deterministic bootstrap resamples.
- [ ] C3 external 10/20/10 and 20×3 human inputs are independently verified, or remain explicitly missing.
- [ ] C4 output is byte-stable and only then can `docs/results.md` become sealed.
- [x] Release A replay behavior, security boundaries, and cost/route policies remain unchanged.

### Execution record — 2026-09-11

- Release A readiness: passed with bundle SHA-256 `d3132abffb9e62c0c4dd5ae6996edd93872fbeab5483553fb1c53235281eaed6`.
- Offline service regression: `1742 passed, 5 skipped, 1 deselected, 1 warning`.
- New gate/preflight tests: `21 passed`; Ruff, configured Pyright, compileall, and diff checks passed.
- Clean LF clone: `crlf_files=0`; readiness passed after locked dependency sync.
- Published commits: `167654b` on `main`; review branch `feature/release-bc-readiness` was pushed and merged.
- Remaining environment blockers are intentionally unchanged: Docker/PostgreSQL/vLLM binaries, authorized online credentials/catalogs, pinned formal model/data locks, licensed external snapshots, and 20×3 human ratings. `docs/results.md` remains unsealed.
