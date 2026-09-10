# Release A Launch Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package and verify a reproducible, offline Replay Showcase that can be started from the repository with a complete server-owned catalog, durable HTTP/SSE behavior, and an auditable readiness gate.

**Architecture:** Keep the existing FastAPI lifespan, `SqlAlchemyRunStore`, durable SSE cursor, `LimitManager`, `RunManager`, and `DefaultCoreRunnerBuilder` as the only service composition. Add a checked-in replay catalog and pricing snapshot whose routes point to the public baseline bundle, package those inputs in the image, and expose a read-only readiness command that validates the bundle and prints only non-secret release metadata. Live Kimi access and the `research-v1` graph remain opt-in follow-on releases and are not enabled by this plan.

**Tech Stack:** Python 3.12, `uv` locked dependencies, FastAPI/Uvicorn, Streamlit, LangGraph Core replay adapters, SQLite/Postgres, Docker Compose, pytest/respx, Ruff, Pyright, Git attributes.

**Spec:** `docs/superpowers/specs/2026-09-09-launch-readiness-design.md`

## Global Constraints

- Use Python `==3.12.*`; run `uv lock --check` and install with `uv sync --all-extras --locked` (or the equivalent frozen `--extra dev` local command); do not re-resolve dependencies in this release.
- Release A defaults remain `ALLOWED_EXECUTION_MODES=["replay"]`, `ALLOWED_PROVIDER_PROFILE_IDS=["replay-default"]`, purposes `demo`/`test`, and low/medium budgets; a client cannot select Live or replace a route.
- Strict Replay must validate the checked-in bundle, derive `replay_parent` from its snapshot, reject an unknown question as a typed Replay miss, and never fall back to a provider or network call.
- Keep credentials out of route catalogs, request payloads, checkpoints, manifests, events, SSE, logs, browser code, Docker layers, and Git. Never place the user-supplied Kimi key in a file or command; rotate it before any future Live test.
- Do not rewrite or re-hash tracked frozen fixtures to compensate for Windows CRLF. The Git blob is authoritative; canonical Replay verification uses a clean LF checkout or a temporary LF copy in tests.
- Preserve the existing stable failures: `RESEARCH_GRAPH_UNAVAILABLE` for production `research-v1` and `CHECKPOINT_RESUME_UNAVAILABLE` for unsupported Core continuation. Do not claim either capability in Release A.
- Keep `docs/results.md` explicitly unsealed. Do not invent quality, cost, confidence interval, external, or human-evaluation values.
- Use the existing security boundary, owner isolation, accounting, policy, SSE, and UI contracts; do not add a second manager, store, rate limiter, event stream, or provider protocol.
- Compose remains a local demonstration topology. Public hosting still requires TLS, a secret manager, trusted proxy CIDRs, secure cookies, backups, and an independently reviewed Postgres deployment.

## File Map

| File | Responsibility |
| --- | --- |
| `.gitattributes` | Force LF only for hash-addressed dataset, snapshot, replay, manifest, and result-sidecar files. |
| `deploy/replay/profiles.json` | Server-owned `replay-default` route catalog for the shipped baseline bundle. |
| `deploy/replay/pricing.json` | Zero-cost `PricingSnapshot` entries for every metered baseline operation. |
| `Dockerfile` | Package the deployment catalog, readiness command, and public baseline bundle without including secrets. |
| `docker-compose.yml` | Point the API at packaged replay inputs and keep the local profile replay-only. |
| `scripts/release_readiness.py` | Validate the Release A package and emit a stable version/status/count/hash line. |
| `tests/contracts/test_fixture_eol.py` | Cross-platform Git-attribute contract. |
| `tests/integration/deployment/test_release_package.py` | Catalog, pricing, bundle, Docker, and readiness-package contract. |
| `tests/unit/providers/test_openai_compatible.py` | Mock Kimi-compatible regional endpoint/error/redaction contract; no network. |
| `.github/workflows/ci.yml` | Run the Release A readiness command and strict replay gate in Linux CI. |
| `README.md`, `docs/deployment.md`, `docs/results.md` | Document the runnable Demo boundary, optional Live boundary, and unsealed Benchmark status. |

---

### Task 1: Lock hash-addressed fixtures to LF

**Files:**
- Create: `.gitattributes`
- Create: `tests/contracts/test_fixture_eol.py`

**Interfaces:**
- Consumes: Git's `check-attr` output for tracked paths.
- Produces: `eol=lf` attributes for every hash-addressed class used by Replay and formal snapshots; no change to existing source/Markdown line-ending policy.

- [ ] **Step 1: Write the failing Git-attribute contract test**

```python
from __future__ import annotations

import subprocess


def test_hash_addressed_assets_are_checked_out_as_lf() -> None:
    paths = [
        "benchmarks/datasets/frozen_ai_cs_60/public_manifest.json",
        "benchmarks/datasets/frozen_ai_cs_60/runtime/dev/technical_survey.jsonl",
        "benchmarks/snapshots/frozen_ai_cs_60/dev-ts-01/snapshot.json",
        "benchmarks/snapshots/frozen_ai_cs_60/dev-ts-01/manifest.sha256",
        "tests/fixtures/frozen_corpus/task-fixture/documents.jsonl",
        "tests/fixtures/frozen_corpus/task-fixture/index.json",
        "tests/fixtures/frozen_corpus/task-fixture/snapshot.json",
        "tests/fixtures/frozen_corpus/task-fixture/manifest.sha256",
        "tests/fixtures/replay/baseline/documents.jsonl",
        "tests/fixtures/replay/baseline/embeddings.jsonl",
        "tests/fixtures/replay/baseline/expected-evidence.json",
        "tests/fixtures/replay/baseline/expected-report.md",
        "tests/fixtures/replay/baseline/model_responses.jsonl",
        "tests/fixtures/replay/baseline/search.jsonl",
        "tests/fixtures/replay/baseline/snapshot.json",
        "tests/fixtures/replay/baseline/manifest.sha256",
    ]
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    attributes = {}
    for line in result.stdout.splitlines():
        path, attribute, value = line.rsplit(": ", 2)
        attributes[path.replace("\\", "/")] = (attribute, value)
    assert set(attributes) == set(paths)
    assert all(attribute == "eol" and value == "lf" for attribute, value in attributes.values())
```

- [ ] **Step 2: Run the contract before adding attributes**

Run: `python -m uv run pytest -q tests/contracts/test_fixture_eol.py`

Expected: FAIL because each path reports `unspecified` or does not yet have an `eol=lf` rule.

- [ ] **Step 3: Add narrowly scoped attributes**

Create `.gitattributes` with these exact rules; do not add a repository-wide `*.json text eol=lf` rule and do not normalize the existing worktree:

```gitattributes
benchmarks/datasets/frozen_ai_cs_60/public_manifest.json text eol=lf
benchmarks/datasets/frozen_ai_cs_60/runtime/**/*.jsonl text eol=lf
benchmarks/snapshots/frozen_ai_cs_60/**/snapshot.json text eol=lf
benchmarks/snapshots/frozen_ai_cs_60/**/manifest.sha256 text eol=lf
tests/fixtures/frozen_corpus/**/documents.jsonl text eol=lf
tests/fixtures/frozen_corpus/**/index.json text eol=lf
tests/fixtures/frozen_corpus/**/snapshot.json text eol=lf
tests/fixtures/frozen_corpus/**/manifest.sha256 text eol=lf
tests/fixtures/replay/**/documents.jsonl text eol=lf
tests/fixtures/replay/**/embeddings.jsonl text eol=lf
tests/fixtures/replay/**/expected-evidence.json text eol=lf
tests/fixtures/replay/**/expected-report.md text eol=lf
tests/fixtures/replay/**/model_responses.jsonl text eol=lf
tests/fixtures/replay/**/search.jsonl text eol=lf
tests/fixtures/replay/**/snapshot.json text eol=lf
tests/fixtures/replay/**/manifest.sha256 text eol=lf
```

- [ ] **Step 4: Run the contract after adding attributes**

Run: `python -m uv run pytest -q tests/contracts/test_fixture_eol.py`

Expected: PASS. `git ls-files --eol tests/fixtures/replay/baseline` may still show `w/crlf` until a clean LF checkout; that is expected and is not fixed by rewriting the fixture.

- [ ] **Step 5: Commit the line-ending contract**

```powershell
git add .gitattributes tests/contracts/test_fixture_eol.py
git commit -m "build: pin hash-addressed assets to LF"
```

### Task 2: Ship a complete Replay catalog and package inputs

**Files:**
- Create: `deploy/replay/profiles.json`
- Create: `deploy/replay/pricing.json`
- Create: `tests/integration/deployment/test_release_package.py`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `tests/integration/deployment/test_compose_config.py`

**Interfaces:**
- Consumes: `FileProviderRouteCatalog`, `FilePricingCatalog`, `ReplayBundle`, `ReplayProviderSnapshot`, and the existing `create_app` composition.
- Produces: one server-selected profile named `replay-default`, five replay routes, six pricing snapshots, and container paths `/app/deploy/replay/{profiles,pricing}.json` plus `/app/tests/fixtures/replay/baseline`.

- [ ] **Step 1: Write the failing package contract**

Create `tests/integration/deployment/test_release_package.py` with a test that copies `tests/fixtures/replay/baseline` to a temporary directory while replacing only `CRLF` bytes with `LF`, copies the two deployment JSON files into that temporary repository, rewrites only the temporary catalog's `bundle_path` to the copied bundle, and asserts:

```python
from deepresearch.providers.replay import ReplayBundle
from deepresearch.runtime.runner_factory import FilePricingCatalog, FileProviderRouteCatalog


def test_packaged_replay_catalog_is_complete(tmp_path: Path) -> None:
    bundle = _lf_copy(Path("tests/fixtures/replay/baseline"), tmp_path / "bundle")
    profiles = _catalog_with_bundle(Path("deploy/replay/profiles.json"), bundle, tmp_path)
    pricing = FilePricingCatalog.load(Path("deploy/replay/pricing.json"))
    routes = FileProviderRouteCatalog.load(profiles).resolve("replay-default")
    snapshots = pricing.resolve("replay-default")

    assert routes.execution_mode == "replay"
    assert {route.operation for route in routes.routes} == {
        "model", "search", "fetch", "parse", "embed"
    }
    assert ReplayBundle.load(bundle).verify().valid
    required = {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured") if route.operation == "model" else (route.operation,)
        )
    }
    available = {(item.provider_id, item.endpoint_type, item.model_id) for item in snapshots}
    assert required <= available
assert len(snapshots) == 6
```

Use these helpers so the test never changes tracked fixture bytes or relies on the current checkout's line endings:

```python
def _lf_copy(source: Path, destination: Path) -> Path:
    destination.mkdir()
    for item in source.iterdir():
        if item.is_file():
            destination.joinpath(item.name).write_bytes(
                item.read_bytes().replace(b"\\r\\n", b"\\n")
            )
    return destination


def _catalog_with_bundle(source: Path, bundle: Path, tmp_path: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for route in payload["profiles"]["replay-default"]["routes"]:
        if route["operation"] != "parse":
            route["parameters"]["bundle_path"] = str(bundle)
    destination = tmp_path / "profiles.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination
```

Add a second assertion that every non-parse route in the checked-in catalog has the exact relative bundle path `tests/fixtures/replay/baseline`.

- [ ] **Step 2: Run the package test before adding deployment inputs**

Run: `python -m uv run pytest -q tests/integration/deployment/test_release_package.py`

Expected: FAIL because `deploy/replay/profiles.json` and `deploy/replay/pricing.json` do not exist.

- [ ] **Step 3: Add the server-owned route catalog**

Create `deploy/replay/profiles.json` with one profile and these five routes. Keep all nullable fields explicit so the frozen Pydantic model receives the same shape in every environment:

```json
{
  "profiles": {
    "replay-default": {
      "execution_mode": "replay",
      "routes": [
        {
          "operation": "model",
          "provider_id": "baseline-model",
          "endpoint_type": "chat.completions",
          "model_id": "baseline-model-v1",
          "model_revision": "cccccccccccccccccccccccccccccccccccccccc",
          "base_url": null,
          "credential_ref": null,
          "fallback_rank": 0,
          "parameters": {"bundle_path": "tests/fixtures/replay/baseline"}
        },
        {
          "operation": "search",
          "provider_id": "baseline-search",
          "endpoint_type": "search",
          "model_id": null,
          "model_revision": null,
          "base_url": null,
          "credential_ref": null,
          "fallback_rank": 0,
          "parameters": {"bundle_path": "tests/fixtures/replay/baseline"}
        },
        {
          "operation": "fetch",
          "provider_id": "baseline-fetch",
          "endpoint_type": "fetch",
          "model_id": null,
          "model_revision": null,
          "base_url": null,
          "credential_ref": null,
          "fallback_rank": 0,
          "parameters": {"bundle_path": "tests/fixtures/replay/baseline"}
        },
        {
          "operation": "parse",
          "provider_id": "baseline-parser-router",
          "endpoint_type": "parse",
          "model_id": null,
          "model_revision": null,
          "base_url": null,
          "credential_ref": null,
          "fallback_rank": 0,
          "parameters": {}
        },
        {
          "operation": "embed",
          "provider_id": "baseline-embed",
          "endpoint_type": "embed",
          "model_id": "baseline-embed-v1",
          "model_revision": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
          "base_url": null,
          "credential_ref": null,
          "fallback_rank": 0,
          "parameters": {"bundle_path": "tests/fixtures/replay/baseline"}
        }
      ]
    }
  }
}
```

- [ ] **Step 4: Add the matching zero-cost pricing snapshot**

Create `deploy/replay/pricing.json` with the six rows below. `complete` and `structured` are both required for the model; parser pricing uses provider `baseline-parser-router` and model ID `parse`.

```json
{
  "profiles": {
    "replay-default": [
      {"snapshot_id":"baseline-model-complete","provider_id":"baseline-model","endpoint_type":"complete","model_id":"baseline-model-v1","effective_at":"2026-08-29T00:00:00Z","currency":"USD","input_tokens_per_million_usd":"0","output_tokens_per_million_usd":"0","cached_tokens_per_million_usd":"0","reasoning_tokens_per_million_usd":"0"},
      {"snapshot_id":"baseline-model-structured","provider_id":"baseline-model","endpoint_type":"structured","model_id":"baseline-model-v1","effective_at":"2026-08-29T00:00:00Z","currency":"USD","input_tokens_per_million_usd":"0","output_tokens_per_million_usd":"0","cached_tokens_per_million_usd":"0","reasoning_tokens_per_million_usd":"0"},
      {"snapshot_id":"baseline-search-search","provider_id":"baseline-search","endpoint_type":"search","model_id":"search","effective_at":"2026-08-29T00:00:00Z","currency":"USD","input_tokens_per_million_usd":"0","output_tokens_per_million_usd":"0","cached_tokens_per_million_usd":"0","reasoning_tokens_per_million_usd":"0"},
      {"snapshot_id":"baseline-fetch-fetch","provider_id":"baseline-fetch","endpoint_type":"fetch","model_id":"fetch","effective_at":"2026-08-29T00:00:00Z","currency":"USD","input_tokens_per_million_usd":"0","output_tokens_per_million_usd":"0","cached_tokens_per_million_usd":"0","reasoning_tokens_per_million_usd":"0"},
      {"snapshot_id":"baseline-parser-router-parse","provider_id":"baseline-parser-router","endpoint_type":"parse","model_id":"parse","effective_at":"2026-08-29T00:00:00Z","currency":"USD","input_tokens_per_million_usd":"0","output_tokens_per_million_usd":"0","cached_tokens_per_million_usd":"0","reasoning_tokens_per_million_usd":"0"},
      {"snapshot_id":"baseline-embed-embed","provider_id":"baseline-embed","endpoint_type":"embed","model_id":"baseline-embed-v1","effective_at":"2026-08-29T00:00:00Z","currency":"USD","input_tokens_per_million_usd":"0","output_tokens_per_million_usd":"0","cached_tokens_per_million_usd":"0","reasoning_tokens_per_million_usd":"0"}
    ]
  }
}
```

- [ ] **Step 5: Package the catalog and fixture in the image**

After the existing source/model copies in `Dockerfile`, add:

```dockerfile
COPY deploy ./deploy
COPY scripts ./scripts
COPY tests/fixtures/replay/baseline ./tests/fixtures/replay/baseline
```

Leave `.dockerignore`'s `artifacts/` rule in place; the image must contain only the tracked public baseline fixture, not local run output. Keep the existing non-root `deepresearch` user and locked `uv sync` command.

- [ ] **Step 6: Point Compose at the packaged inputs and preserve exact Replay identity**

In `docker-compose.yml`, set the API environment to:

```yaml
      DEPLOYMENT_ACCESS_PROFILE: local
      PROVIDER_PROFILE_CATALOG_PATH: /app/deploy/replay/profiles.json
      PRICING_CATALOG_PATH: /app/deploy/replay/pricing.json
      ALLOWED_EXECUTION_MODES: '["replay"]'
      ALLOWED_PROVIDER_PROFILE_IDS: '["replay-default"]'
```

The `local` profile is intentional: the public baseline was recorded before the prompt-boundary marker and the existing lifespan preserves its exact identity only for local replay. The client still sends `access_profile: showcase`; server policy normalization remains authoritative. Do not expose `local` as a public-live deployment profile.

- [ ] **Step 7: Extend static Compose/package assertions**

Update `test_compose_config.py` to assert the two catalog environment paths, `DEPLOYMENT_ACCESS_PROFILE == "local"`, and the two Dockerfile `COPY` lines. Keep assertions for URI-independent `DATABASE_URL`, non-root API/UI, Postgres `postgres` user, `cap_drop: ALL`, `no-new-privileges`, and secret-free build files.

- [ ] **Step 8: Run package and Compose tests**

Run: `python -m uv run pytest -q tests/integration/deployment/test_release_package.py tests/integration/deployment/test_compose_config.py`

Expected: PASS. If the working tree is CRLF-contaminated, the package test must pass because its helper creates an LF temporary copy; it must not mutate `tests/fixtures/replay/baseline`.

- [ ] **Step 9: Commit the packaged Replay release**

```powershell
git add deploy/replay Dockerfile docker-compose.yml tests/integration/deployment/test_release_package.py tests/integration/deployment/test_compose_config.py
git commit -m "build: package the replay showcase"
```

### Task 3: Add a secret-free readiness command and CI gate

**Files:**
- Create: `scripts/release_readiness.py`
- Create: `tests/integration/deployment/test_release_readiness.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: a repository root, the packaged catalog/pricing files, and `ReplayBundle.verify()`.
- Produces: `assess(repository: Path) -> ReadinessSummary` and a CLI whose successful stdout has exactly `release_a version=... status=pass profile_count=... route_count=... pricing_count=... bundle_sha256=...`; no path, credential, traceback, or provider response is printed.

- [ ] **Step 1: Write the failing readiness test**

Create a test that copies the deployment JSON and an LF-normalized baseline bundle into a temporary repository, invokes `python scripts/release_readiness.py --repository repository` using the test's `repository` variable, and asserts:

```python
assert result.returncode == 0, result.stderr
assert re.fullmatch(
    r"release_a version=0\.1\.0 status=pass profile_count=1 route_count=5 pricing_count=6 bundle_sha256=[0-9a-f]{64}\n",
    result.stdout,
)
assert "MODEL-SECRET" not in result.stdout
assert "MODEL-SECRET" not in result.stderr
```

Add a second test with the bundle's `manifest.sha256` changed in the temporary copy; assert a nonzero exit, the same field-only `status=fail` stdout with a 64-zero hash, and a static stderr token such as `readiness_error=bundle_verification` without the temporary path.

- [ ] **Step 2: Run the readiness tests before adding the command**

Run: `python -m uv run pytest -q tests/integration/deployment/test_release_readiness.py`

Expected: FAIL because `scripts/release_readiness.py` is missing.

- [ ] **Step 3: Implement the deterministic package assessment**

Implement `scripts/release_readiness.py` with these rules:

```python
class ReadinessFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ReadinessSummary:
    profile_count: int
    route_count: int
    pricing_count: int
    bundle_sha256: str


def assess(repository: Path) -> ReadinessSummary:
    profile_path = repository / "deploy" / "replay" / "profiles.json"
    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    bundle_root = repository / "tests" / "fixtures" / "replay" / "baseline"
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    pricing_payload = json.loads(pricing_path.read_text(encoding="utf-8"))
    if set(profile_payload.get("profiles", {})) != {"replay-default"}:
        raise ReadinessFailure("profile_shape")
    if set(pricing_payload.get("profiles", {})) != {"replay-default"}:
        raise ReadinessFailure("pricing_profile_shape")
    routes = FileProviderRouteCatalog.load(profile_path).resolve("replay-default")
    prices = FilePricingCatalog.load(pricing_path).resolve("replay-default")
    verification = ReplayBundle.load(bundle_root).verify()
    if not verification.valid:
        raise ReadinessFailure("bundle_verification")
    if routes.execution_mode != "replay" or len(routes.routes) != 5:
        raise ReadinessFailure("route_shape")
    if len(prices) != 6:
        raise ReadinessFailure("pricing_shape")
    for route in routes.routes:
        if route.operation == "parse":
            continue
        if route.parameters.get("bundle_path") != "tests/fixtures/replay/baseline":
            raise ReadinessFailure("bundle_path")
    required = {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured") if route.operation == "model" else (route.operation,)
        )
    }
    available = {(item.provider_id, item.endpoint_type, item.model_id) for item in prices}
    if not required <= available:
        raise ReadinessFailure("pricing_coverage")
    digest_input = json.dumps(
        {name: verification.file_sha256[name] for name in REPLAY_FILES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ReadinessSummary(
        profile_count=1,
        route_count=len(routes.routes),
        pricing_count=len(prices),
        bundle_sha256=hashlib.sha256(digest_input).hexdigest(),
    )
```

The CLI must accept `--repository` for the isolated test, default it to `Path(__file__).resolve().parents[1]`, catch only the custom readiness failure and file/validation errors, emit the field-only failure line with a 64-zero digest, and return `1`. Never include exception text, absolute paths, catalog contents, environment values, or secrets in stdout/stderr.

- [ ] **Step 4: Run the command tests after implementation**

Run: `python -m uv run pytest -q tests/integration/deployment/test_release_readiness.py`

Expected: PASS.

- [ ] **Step 5: Add the Linux CI Release A gate**

In `.github/workflows/ci.yml`, after the locked sync and before the broad offline suite, add:

```yaml
      - name: Release A readiness
        run: |
          uv run python scripts/release_readiness.py
          uv run pytest -q tests/integration/deployment/test_release_package.py tests/integration/replay/test_full_service.py::test_full_service_strict_replay_completes_against_verified_baseline_bundle
```

Keep the existing secret-free verification job and its `-m "not online"` selection. The online job remains manual/scheduled and must not be used as Release A evidence.

- [ ] **Step 6: Commit the readiness gate**

```powershell
git add scripts/release_readiness.py tests/integration/deployment/test_release_readiness.py .github/workflows/ci.yml
git commit -m "ci: gate release a replay readiness"
```

### Task 4: Pin the Kimi-compatible model contract without enabling Live

**Files:**
- Modify: `tests/unit/providers/test_openai_compatible.py`

**Interfaces:**
- Consumes: `OpenAICompatibleModelProvider` and existing `ProviderError`/`respx` contracts.
- Produces: offline proof that regional Kimi OpenAI-compatible URLs use `/chat/completions`, credentials are header-only, upstream errors/timeouts are typed, and no key enters public text. It does not add a Kimi provider or enable a Live profile.

- [ ] **Step 1: Add failing regional success tests**

Add a parameterized test for `https://api.moonshot.cn/v1` and `https://api.moonshot.ai/v1` that registers only `POST {base_url}/chat/completions`, invokes `complete`, asserts `result.output == "answer"`, asserts the request's authorization is `Bearer KIMI-TEST-SECRET`, and asserts `repr(provider)` does not contain `KIMI-TEST-SECRET`. The test must fail if the provider appends a different path or sends the key in JSON.

- [ ] **Step 2: Add failing error and timeout tests**

Use a mocked `401` response body containing `KIMI-TEST-SECRET` and assert `ProviderError.code == "AUTHENTICATION"`, no key in `str(error)`, and no cause/context text. Use `httpx.MockTransport` whose handler raises `httpx.ReadTimeout("fixture timeout")` and assert `ProviderError.code == "TIMEOUT"` with no secret in the public error. Close the injected client in `finally`.

- [ ] **Step 3: Run the provider tests**

Run: `python -m uv run pytest -q tests/unit/providers/test_openai_compatible.py`

Expected: PASS with the existing provider implementation. If a test exposes a regression, change only the provider's existing HTTP/error boundary and add the smallest matching regression; do not add a native `$web_search` adapter in this release.

- [ ] **Step 4: Commit the Kimi contract**

```powershell
git add tests/unit/providers/test_openai_compatible.py
git commit -m "test: pin kimi compatible model contract"
```

### Task 5: Publish accurate Demo, Live, and Benchmark boundaries

**Files:**
- Modify: `README.md`
- Modify: `docs/deployment.md`
- Modify: `docs/results.md`

**Interfaces:**
- Consumes: the packaged Compose paths and `release_readiness.py` command from Tasks 2–3.
- Produces: operator documentation that can start the offline Demo without credentials while clearly separating optional Kimi Live and unsealed formal Benchmark work.

- [ ] **Step 1: Write documentation assertions**

Add or extend a documentation contract test in `tests/integration/deployment/test_release_package.py` that asserts README/deployment text contains:

```python
assert "uv run python scripts/release_readiness.py" in readme
assert "/app/deploy/replay/profiles.json" in deployment
assert "research-v1" in readme and "RESEARCH_GRAPH_UNAVAILABLE" in deployment
assert "not sealed" in results
assert "MODEL_API_KEY" in deployment and "KIMI" in deployment
```

The test must also construct a synthetic secret sentinel as `"synthetic-" + "secret-sentinel"` and assert that its resulting value is absent from every checked-in text file; never copy a real credential into the test or documentation.

- [ ] **Step 2: Update the README Replay/Compose quickstart**

Document `uv lock --check`, the clean-LF checkout requirement for hash-addressed fixtures, `uv run python scripts/release_readiness.py`, and the exact Compose commands with URI-encoded `DATABASE_URL`, `POSTGRES_PASSWORD`, and a 32-byte session key. State that the image now contains `deploy/replay/profiles.json`, `deploy/replay/pricing.json`, and the public baseline bundle; remove the statement that the default catalog is deliberately empty. Keep the known-input example `Compare planner strategies` and state that arbitrary questions produce a Replay miss.

State explicitly that Compose is a local replay showcase (`DEPLOYMENT_ACCESS_PROFILE=local`), that Kimi is a separate model-only opt-in using server-side `MODEL_API_KEY` plus separately configured Tavily/Serper search and pricing, and that native Kimi `$web_search` is not supported. Retain the statement that formal A/B/C/D results and hosted Live endpoints are not claimed.

- [ ] **Step 3: Update deployment operations**

Change the opening deployment text from “empty routes/no bundled recording” to the packaged replay paths and the readiness command. Keep read-only catalog override instructions for operators with a different verified bundle. Preserve the public deployment table, proxy/CIDR rule, health/shutdown behavior, accounting recovery, and Docker/Postgres verification caveats. Add the exact stop condition: any secret leak, owner isolation failure, hash/manifest mismatch, Replay-to-Live fallback, policy bypass, unknown-cost reservation loss, missing research node, or unsealed Benchmark input stops publication.

- [ ] **Step 4: Correct only the results-page encoding while preserving its status**

Rewrite the malformed failure-analysis sentence in `docs/results.md` as a UTF-8 sentence such as `Failure analysis: no public aggregate is sealed, so no failure-rate attribution is reported.` Do not change `not sealed`, `not reported`, deterministic figure captions, or provenance fields.

- [ ] **Step 5: Run documentation and secret scans**

Run:

```powershell
python -m uv run pytest -q tests/integration/deployment/test_release_package.py
$sentinel = "synthetic-" + "secret-sentinel"
rg -n --hidden --glob '!\.git/**' --glob '!\.venv/**' --fixed-strings $sentinel .
```

Expected: the documentation test passes and `rg` prints no matching line. A user-supplied credential is never added to the repository.

- [ ] **Step 6: Commit the operator documentation**

```powershell
git add README.md docs/deployment.md docs/results.md tests/integration/deployment/test_release_package.py
git commit -m "docs: document replay release boundaries"
```

### Task 6: Run the Release A acceptance gate and archive the candidate

**Files:**
- Modify only if a previous task's test or documentation assertion requires a scoped correction.
- Do not commit `.superpowers/sdd/2026-08-29-service-demo-deployment/progress.md`; it is an ignored execution ledger.

**Interfaces:**
- Consumes: all package, API/SSE, security, limits, UI, health, Compose, and readiness contracts.
- Produces: a clean branch with reproducible Replay evidence and an explicit record of any environment-limited checks.

- [ ] **Step 1: Verify dependency, lint, type, and diff gates**

Run from a clean LF checkout:

```powershell
python -m uv lock --check
python -m uv sync --extra dev --frozen
python -m uv run ruff check .
python -m uv run pyright src apps benchmarks experiments
python -m compileall -q scripts
git diff --check
```

Expected: all commands exit zero. Do not substitute whole-repository test-only Pyright for the production/application gate; legacy fixture diagnostics remain separate.

- [ ] **Step 2: Run the service regression and readiness suites**

```powershell
python -m uv run python scripts/release_readiness.py
python -m uv run pytest -q tests/unit tests/contracts tests/integration/replay tests/integration/api tests/integration/deployment -m "not online"
```

Expected readiness output has only version/status/count/hash fields and `status=pass`. The offline service suite must retain the known Windows CRLF/Core-clock exclusions documented in `docs/deployment.md`; do not re-hash fixtures or turn those environment failures into release numbers.

- [ ] **Step 3: Validate Compose and build the image**

```powershell
$env:DATABASE_URL = "postgresql+asyncpg://deepresearch:ci-only@postgres:5432/deepresearch"
$env:POSTGRES_PASSWORD = "ci-only"
$env:SESSION_SIGNING_KEY = "ci-only-signing-key-at-least-32-bytes"
docker compose config --quiet
docker build --build-arg DEEPRESEARCH_CODE_COMMIT=$(git rev-parse HEAD) -t multi-agent-deep-research:release-a .
```

Expected: Compose config and Docker build pass on a host with Docker Engine. If Docker/Postgres is unavailable on the current Windows host, record the exact commands as environment-limited and require Linux CI/target-environment acceptance before claiming a deployed service.

- [ ] **Step 4: Perform the HTTP/SSE evidence check**

Start the packaged stack with `docker compose up --build`, open `http://127.0.0.1:8501`, submit `Compare planner strategies`, and verify through the API that the run reaches `completed`, emits `run_completed`, replays events after `Last-Event-ID`, serves report/evidence/manifest artifacts, and records `manifest.replay_parent == snapshot.run_id`. Submit an unknown question and verify a stable Replay miss with no Live/provider request. Use two isolated browser sessions and verify foreign runs/artifacts/events all return the same 404.

- [ ] **Step 5: Record stop conditions and push only a clean candidate**

Stop and retain the previous commit if any release gate observes secret text, owner leakage, route-policy bypass, hash mismatch, Replay Live fallback, lost unknown-cost reservation, missing graph node, or unsealed Benchmark data presented as a result. Otherwise run:

```powershell
git status --short
git log -1 --oneline
git push origin feature/service-demo-deployment
```

The handoff must report the readiness line, focused test counts, production Pyright/Ruff results, Docker/Postgres/online limitations, and the unchanged Release B/C blockers. Release B requires a separately reviewed Kimi model-only profile with complete search/embedding/pricing and authorized smoke; Release C requires real `ResearchGraphDependencies`, formal model/environment/data locks, private Gold isolation, external artifacts, and human ratings before any Benchmark result can be sealed.

## Release A completion checklist

- [ ] A clean LF checkout loads `replay-default` from the packaged catalog and verifies the public baseline bundle without credentials.
- [ ] The Compose API uses `/app/deploy/replay/profiles.json` and `/app/deploy/replay/pricing.json`; no catalog or key is supplied by the browser.
- [ ] The readiness command prints only version, status, counts, and a bundle hash; CI runs it without secrets.
- [ ] Strict HTTP/SSE replay, artifact downloads, owner isolation, reconnect cursors, redaction, quotas, health, and shutdown regressions remain green.
- [ ] README/deployment docs distinguish Replay Demo, optional Kimi Live, and unsealed Benchmark results.
- [ ] The candidate is pushed only after the branch is clean and all environment-limited checks are explicitly labeled.

## Explicitly deferred gates

Release B is not enabled by this plan because it needs an independently supplied `MODEL_API_KEY`, search credentials, complete provider/pricing/embedding locks, and authorized paid smoke. Release C is not enabled because production `research-v1` handlers, formal 60-task inputs, Qwen/vLLM/GPU locks, private Gold, external raw/lock/results, and three-person human ratings are absent. These are separate plans and must remain fail-closed rather than being represented by substituted values or fabricated measurements.
