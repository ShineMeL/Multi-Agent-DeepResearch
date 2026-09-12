# Service deployment and operator runbook

The API and Streamlit client can start with SQLite or the supplied Compose
Postgres deployment. The image packages the Replay catalog at
`/app/deploy/replay/profiles.json`, pricing at
`/app/deploy/replay/pricing.json`, and the verified public baseline bundle at
`/app/tests/fixtures/replay/baseline`. From a clean LF checkout, verify those
release inputs before startup:

```powershell
uv lock --check
uv run python scripts/release_readiness.py
```

Strict replay derives Core's `replay_parent` from the verified bundle snapshot
and uses the built-in parser router for mixed HTML/PDF recordings. The
local-only replay composition preserves the shipped baseline request identity;
live/public compositions retain the untrusted-content boundary. The packaged
input is `Compare planner strategies`. Arbitrary questions cannot replay
recordings for different inputs, and there is no automatic live fallback on
`REPLAY_MISS`.

The implemented production composition supports `baseline-v1` with `P1` / `R1`
and the deterministic `research-v1` P1/R1 research showcase. The latter adds
claim extraction, evidence verification, and conservative unsupported-claim
resolution while reusing the audited retrieval/report pipeline. Explicitly
choose P1/R1 for this composition. The API's default research settings
(`research-v1` with P2/R2) remain unavailable and fail closed with
`RESEARCH_GRAPH_UNAVAILABLE`; there is no silent baseline fallback.

## Build provenance

The service accepts `DEEPRESEARCH_CODE_COMMIT` (a lowercase 40- or 64-character
Git object hash) and `DEEPRESEARCH_DEPENDENCY_LOCK_SHA256` (64 lowercase hex
characters). Invalid values fail before provider composition. If packaged
`uv.lock` exists, its bytes determine the lock digest and a conflicting supplied
digest is rejected. Source checkouts discover Git only when metadata is present;
Git is optional at runtime. Wheels without checkout metadata use explicit all-zero
hashes for unavailable commit/lock identities. These are unknown sentinels, never
claims of reproducible build provenance. Supply both values when distributing a
wheel without its lock file.

Docker includes the actual lock file. Build release images with
`docker build --build-arg DEEPRESEARCH_CODE_COMMIT=<full-source-commit> -t multi-agent-deep-research .`.
CI supplies its checked-out SHA. The default zero commit allows local image runs
without `.git` or a Git executable and remains visible as unknown in the manifest;
it must not be presented as a verified release revision. Provider/configuration
hashes and canonical manifest validation retain their existing semantics.

## PowerShell preparation

The commands below support Windows PowerShell 5.1 and PowerShell 7. Define this
helper once in the terminal before either startup sequence. It uses the instance
RNG API available in .NET Framework; each call returns 32 random bytes encoded
as Base64. Assign its result as shown below to keep it out of terminal output.

```powershell
function New-DeploymentSecret {
    $secretBytes = New-Object byte[] 32
    $secretRng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $secretRng.GetBytes($secretBytes)
        return [Convert]::ToBase64String($secretBytes)
    }
    finally {
        $secretRng.Dispose()
    }
}
```

Assign the function result to an environment variable as shown below; invoking
the function without assignment would print its result.

## Local SQLite

Run from the repository root with Python 3.12 and uv 0.11.28. These PowerShell
commands create local state under `artifacts` and `deepresearch.db`:

```powershell
uv sync --all-extras --locked
$env:SESSION_SIGNING_KEY = New-DeploymentSecret
$env:LANGGRAPH_STRICT_MSGPACK = 'true'
$env:DATABASE_URL = 'sqlite+aiosqlite:///./deepresearch.db'
$env:ARTIFACT_ROOT = './artifacts'
$env:CHECKPOINT_SQLITE_PATH = './artifacts/checkpoints.sqlite'
$env:PROVIDER_PROFILE_CATALOG_PATH = './deploy/replay/profiles.json'
$env:PRICING_CATALOG_PATH = './deploy/replay/pricing.json'
$env:DEPLOYMENT_ACCESS_PROFILE = 'local'
$env:ALLOWED_EXECUTION_MODES = '["replay"]'
$env:ALLOWED_PROVIDER_PROFILE_IDS = '["replay-default"]'
$env:ALLOWED_RUN_PURPOSES = '["demo","test"]'
$env:ALLOWED_BUDGET_PRESETS = '["low","medium"]'
uv run uvicorn apps.api.main:create_app --factory --host 127.0.0.1 --port 8000 --no-proxy-headers
```

In another terminal:

```powershell
$env:DEEPRESEARCH_API_URL = 'http://127.0.0.1:8000'
uv run streamlit run apps/ui/app.py --server.address 127.0.0.1 --server.port 8501
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Persist the generated signing key through your secret manager if sessions must
survive restarts. Regenerating it invalidates existing owner cookies. The API's
environment names are unprefixed; the CLI's `.env.example` uses separate
`DEEPRESEARCH_*` variables and does not configure this service. The server does
not automatically load `.env`; inject environment variables explicitly.

The SQLite service store and Core checkpointer own separate connections/files.
The lifespan opens Core's SQLite saver through `open_service_checkpointer`,
initializes its schema, upgrades the service schema, reconciles startup state,
then opens admission. Keep `CHECKPOINT_SQLITE_PATH` inside `ARTIFACT_ROOT`;
symlink checkpoint files are rejected. Use one API process: run tasks and
rate-limit buckets are process-local. Do not add Uvicorn workers or replicas
without an external coordination design.

## Compose Postgres

Docker Engine and the Compose plugin must be installed and running. Supply the
raw database password separately from the URI-encoded connection URL:

```powershell
$env:POSTGRES_PASSWORD = New-DeploymentSecret
$env:SESSION_SIGNING_KEY = New-DeploymentSecret
$encodedPassword = [uri]::EscapeDataString($env:POSTGRES_PASSWORD)
$env:DATABASE_URL = "postgresql+asyncpg://deepresearch:${encodedPassword}@postgres:5432/deepresearch"
docker compose config --quiet
docker build -t multi-agent-deep-research .
docker compose up --build --detach --wait --wait-timeout 180
docker compose ps
```

These generated values are for a fresh local installation. On an existing
volume, reuse its database credential: changing `POSTGRES_PASSWORD` does not
rotate the password in an initialized Postgres database. Do not print expanded
Compose configuration to shared logs: it contains injected credentials.

The API is available at `http://127.0.0.1:8000` and the UI at
`http://127.0.0.1:8501`. Compose keeps the container listeners on `0.0.0.0` so
the services remain reachable over the private Compose network, but publishes
both host ports explicitly on IPv4 loopback. To avoid a local port collision,
set `API_HOST_PORT` or `UI_HOST_PORT` before `docker compose config` and use the
selected port in host-side health or smoke commands; for example:

```powershell
$env:API_HOST_PORT = '18000'
$env:UI_HOST_PORT = '18501'
```

Compose uses local HTTP with `COOKIE_SECURE=false`
and applies the `local` replay policy. It allows only Replay execution.
Loopback binding limits ordinary remote access; it does not make this
configuration suitable for public hosting, and Docker Engine releases before
28.0.0 have a documented local-network reachability caveat for published
localhost ports. Postgres has no published host port and is reachable only
inside the Compose network by default.
See Docker's [port-publishing guidance](https://docs.docker.com/engine/network/port-publishing/)
for the host-binding and pre-28.0.0 caveats.

API and UI startup invokes the Python modules already installed in the locked
image environment, so container startup performs no dependency resolution. The
UI is explicitly headless with usage telemetry disabled. Both run as the image's
unprivileged UID/GID 10001 account; Postgres uses the official image's
`postgres` account. All three retain `init: true`, dropped capabilities and
`no-new-privileges`. Uvicorn keeps proxy-header processing disabled; configure
trusted proxy CIDRs only through the application's policy when building a
separate TLS-terminating public deployment.

The API service store uses SQLAlchemy/asyncpg. The checkpoint adapter converts
that URL to a psycopg DSN and opens `AsyncPostgresSaver`, calls `setup()` before
yielding it, then closes it on shutdown. Both savers use the strict msgpack
serializer with its explicit Core type allowlist. Set
`LANGGRAPH_STRICT_MSGPACK=true`; startup refuses missing/false values. First
startup requires permissions for both service migrations and checkpointer
schema setup. Backup the database and artifact volume together during a quiet
period with the API stopped.

Named `postgres-data` persists service rows, event logs, usage reservations and
Postgres checkpoints. Named `artifact-data` mounts at
`/var/lib/deepresearch/artifacts` and persists content-addressed reports,
evidence and manifests. The configured SQLite checkpoint path remains under
that root but is unused when Postgres is selected. `docker compose stop` and
`docker compose down` retain named volumes; **`down -v` destroys them**.

Postgres tests in the ordinary local suite use controlled adapters unless an
external database is explicitly configured. Static Compose validation is not a
Postgres restart/recovery test. On the current Windows implementation host,
Docker was unavailable, so no Docker image build, container startup, or real
Postgres recovery was verified locally. CI includes image build/config checks;
it is also configured to start the full Compose stack, exercise the replay HTTP
acceptance below, and clean up only its uniquely named disposable stack and
volumes. A separate disposable PostgreSQL service activates the store contract
tests. These are verification gates, not a claim that a particular CI run has
passed; operators still need to validate their actual database and volumes.

After the services are ready, validate the complete offline demonstration:

```powershell
uv run python -m scripts.smoke_replay --api-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501 --timeout 90
```

The command submits only the advertised fixed Replay example, retains its
owner cookie, checks durable completion and zero cost, and verifies the report,
evidence and manifest downloads against their SHA-256 identifiers. It does not
fall back to Live or call paid providers. Failure returns a nonzero exit status
and a static error code without raw response bodies. A timed-out accepted run
receives a bounded cancellation attempt; run records and artifacts are retained.
Omit `--ui-url` for an API-only check and use your chosen host ports if overridden.

## Provider catalogs and requests

Set `PROVIDER_PROFILE_CATALOG_PATH` and `PRICING_CATALOG_PATH` to UTF-8 JSON
files readable by the API process. Both use a top-level `profiles` mapping.
Provider entries map profile IDs to `execution_mode` and a `routes` array.
Pricing entries map profile IDs to arrays of `PricingSnapshot` objects.
The exact validated schemas are `FrozenProviderRoute`, `FrozenProviderRoutes`
and `PricingSnapshot` in `src/deepresearch/runtime/runner_factory.py` and
`src/deepresearch/runtime/manifest.py`; the offline composition example is
`tests/unit/runtime/test_runner_factory_execution.py`.

Each supported baseline profile needs exactly one route for `model`, `search`,
`fetch`, `parse` and `embed`; unsupported fallback routes are rejected. Route
identity includes provider/model/revision, endpoint, base URL, credential
reference, parameters and fallback rank. Credentials are environment variable
**names**, never values, in `credential_ref`. Only names in
`PROVIDER_CREDENTIAL_ENV_NAMES` can be resolved; include the same names in
`REDACTION_SECRET_ENV_NAMES`. Defaults cover `MODEL_API_KEY`, `SEARCH_API_KEY`
and, for redaction only, `SESSION_SIGNING_KEY`.

Replay requires recorded provider identities and `parameters.bundle_path`
pointing at the matching verified bundle. The catalog's provider identity must
match its adapter; simply renaming every provider to `replay` is not a valid
recording. Operators substituting a different verified bundle must mount its
catalogs and replay data read-only through a Compose override, then set
`PROVIDER_PROFILE_CATALOG_PATH` and `PRICING_CATALOG_PATH` to those container
paths. Use container paths in route parameters, never host paths. The default
Compose configuration instead uses the release inputs already packaged in the
image.

### Optional KIMI model route

Kimi exposes an OpenAI-compatible chat-completions API. For a live baseline
profile, use the `openai-compatible` model route with a server-side
`credential_ref` of `MODEL_API_KEY`; select the endpoint for the account's
region (for example, `https://api.moonshot.cn/v1` for a CN account) and set the
model ID to one enabled for that account, such as `kimi-k3`. Keep the key out of
the catalog and inject it only as `MODEL_API_KEY`. The profile still needs a
separately configured Tavily search route and all required pricing snapshots
before a public/live run is admitted. Kimi is therefore a model-only Live
opt-in, not part of the credential-free Replay showcase.

Kimi's official built-in `$web_search` tool is a different tool-call protocol;
this service does not treat it as a `SearchProvider` or silently convert its
encrypted tool output into evidence. Wire it through a dedicated provider
adapter and tests before enabling it in a live catalog. See the [Kimi API
overview](https://platform.kimi.ai/docs/api/overview) and [official web-search
guide](https://platform.kimi.ai/docs/guide/use-web-search).

Admission freezes non-secret routes and their canonical configuration SHA-256,
plus pricing snapshots, into durable run state and audit data. No client can
override these server-selected routes. Public live and formal benchmark runs
need every `(provider_id, endpoint_type, model_id)` price before invocation;
cost-limited runs also require pricing. Include both `complete` and `structured`
model endpoints and all tool operations. Current Core requires the two model
endpoint rates to agree and tool rates to be zero; unsupported tariffs fail
preflight. Complete supported catalogs yield estimated, not billed, costs.
Resumes reuse the persisted pricing tuple and frozen routes or refuse before a
provider call if those inputs cannot be reconstructed safely.

## Public access policy and safety

For public hosting, terminate TLS at a controlled proxy, set
`DEPLOYMENT_ACCESS_PROFILE=public_live`, `COOKIE_SECURE=true`, a stable
32-byte-or-longer `SESSION_SIGNING_KEY`, and explicit server allowlists:

| Variable | Format / purpose |
| --- | --- |
| `ALLOWED_EXECUTION_MODES` | JSON array, e.g. `["live"]` |
| `ALLOWED_PROVIDER_PROFILE_IDS` | JSON array of configured catalog IDs |
| `ALLOWED_RUN_PURPOSES` | JSON array, e.g. `["test"]` |
| `ALLOWED_BUDGET_PRESETS` | JSON array; public accepts low/medium, never high |
| `DAILY_COST_LIMIT_USD` | Finite nonnegative decimal; default `5.00` |
| `TRUSTED_PROXY_CIDRS` | JSON array of actual proxy networks; default `[]` |
| `PROVIDER_CREDENTIAL_ENV_NAMES` | JSON array of provider credential names |
| `REDACTION_SECRET_ENV_NAMES` | JSON array of all secrets to scrub |

The server normalizes client access profile and enforces its mode, provider,
purpose and budget choices. Client `local`/`showcase` values cannot lower public
restrictions. Forwarded IP headers count only when the direct peer is trusted;
do not trust all networks. Always launch Uvicorn with `--no-proxy-headers`, as in
the Docker default, Compose command and local command here: otherwise Uvicorn
can overwrite the peer before the application's CIDR checks run. Configure
`TRUSTED_PROXY_CIDRS` in the application, not Uvicorn's forwarded allowlist.
Preserve owner cookies and streaming responses at the
proxy. Owner identity binds the signed session and resolved client IP: an IP
change can make prior runs inaccessible. Different owners receive the same 404
for run, events, all artifacts, resume and cancel, including unknown IDs.

Keep secrets out of catalogs, images, source control, browser code and requests.
The backend wraps untrusted fetched content and applies SSRF checks, pinned-peer
fetching and redaction. These controls do not replace TLS, proxy access controls,
egress policy, backups, and careful secret distribution. Secret rotation can
invalidate sessions; coordinate it with operators.

## Health, shutdown, and recovery

`GET /health/live` returns 200 for a responding process.
`GET /health/ready` returns 200 only when database, artifact access, schema,
checkpointer initialization and admission are available; failure returns 503
with static statuses, without DSNs, exception text or secret values. Readiness
does not validate provider credentials, bundle coverage or graph availability.
If readiness fails, inspect API logs privately, database availability/migrations,
artifact permissions, and strict-msgpack configuration.

Use Ctrl+C locally or `docker compose stop -t 30` to leave time for the API's
20-second shutdown grace period. Shutdown closes create/resume admission,
drains work within that grace period, then interrupts remaining tasks and closes
savers and engine connections. Startup reconstructs usage lower bounds from
durable event deltas and reconciles persisted reservations. Known final costs
are settled, including zero for work proven never to have started. If a final
bill is unknown, its reservation and run link remain intact across failure,
restart and repeated cancellation; the process execution slot is freed. Such
reservations continue counting against that UTC day's limit until an operator
reconciles the actual bill. Missing runs are provable admission orphans and
their reservations can be released. Do not manually delete ledger rows to clear
capacity or assume an unknown bill means no spend.

Durable events support `Last-Event-ID` with an integer sequence: reconnect
returns events strictly after it. Keep the same owner cookie and client address.
This event replay is separate from graph resume. The current real Core
cooperative shutdown records a terminal cancellation checkpoint that service
resume rejects with `CHECKPOINT_RESUME_UNAVAILABLE`; do not promise that every
interrupted run can resume. Existing contracts cover resumable checkpoint
shapes, but real Postgres interruption/restart needs environment verification.

## Verification and known platform limits

The deployment type gate covers `src apps benchmarks experiments` in strict
mode, plus the replay acceptance script and critical regression modules listed in CI. This
follows the plan's production/application completion checklist and retains
strict research-tooling coverage. Whole-repository test typing is still an
opt-in diagnostic: older unannotated fixtures generate thousands of errors;
no blanket ignores or production diagnostic exclusions were added.

```powershell
uv run pytest -q tests/unit tests/contracts tests/integration/replay tests/integration/api tests/integration/deployment -m "not online"
uv run ruff check .
uv run pyright src apps benchmarks experiments
uv run pyright scripts/smoke_replay.py tests/integration/api/test_accounting_recovery.py tests/unit/runtime/test_build_provenance.py tests/integration/deployment/test_proxy_identity.py tests/integration/replay/test_full_service.py tests/integration/deployment/test_smoke.py tests/integration/deployment/test_replay_smoke.py
docker compose config --quiet
docker build -t multi-agent-deep-research .
```

Compose validation requires the three disposable local environment values shown
above; CI supplies clearly labeled CI-only values and requires no provider
secrets. The passing service e2e records deterministic offline model, search,
fetch and embedding providers through the real HTTP/SSE, SQLite and Core
pipeline with the production HTML parser. It verifies artifact downloads, owner
isolation, idempotency, event reconnection and reopening persisted rows.
A separate strict-replay case starts the production `main.create_app` lifespan
with on-disk route/pricing catalogs and a verified LF copy of the shipped baseline
bundle, submits `execution_mode="replay"`, constructs the replay adapters and
parser router through the shipped registry, and asserts HTTP 202, durable
completion, the recorded parent identity, artifacts, and event replay. Neither
case claims that arbitrary questions can use that bundle. Existing contract tests
cover health failure, policy, SSRF, redaction and unavailable graph/resume paths.
The ASGI e2e uses Core's deterministic clock fixture. The production service
factory now pairs wall timestamps with a monotonic source to avoid the former
independent-clock envelope mismatch. The HTTP acceptance command exercises
those production clocks and does not inject test time.

Windows checkouts can rewrite hash-addressed fixture bytes to CRLF. A
`REPLAY_CORRUPT`/hash mismatch in a frozen fixture must be investigated against
its checked-in Git bytes. Use a clean LF checkout (`core.autocrlf=false` before
checkout) for canonical replay verification. Do not regenerate expected hashes
to bless newline conversion. Linux CI uses the recorded LF fixture bytes.

The separate `online-smoke` job runs only on manual dispatch or its weekly
schedule and only when all five secrets/catalog inputs are supplied. Missing
credentials skip the step; incomplete pricing snapshots skip the test before
provider invocation. Invalid configured routes or unsupported pricing fail
validation rather than being silently treated as success. This job can incur
provider charges and exercises only `baseline-v1`/P1/R1 under the public medium
budget. To run it locally, explicitly supply `MODEL_API_KEY`, `SEARCH_API_KEY`,
`SESSION_SIGNING_KEY`, `PRICING_CATALOG_PATH` and
`PROVIDER_PROFILE_CATALOG_PATH` with the live `online-smoke` profile, then run:

```powershell
uv run pytest -q tests/integration/deployment/test_smoke.py -m online
```

The online smoke uses temporary SQLite state and checks health, real artifact
completion and response redaction. It is not a live Postgres or deployed-proxy
test. No live smoke was run during local implementation without configuration.

## Release B deployment gates

Release B is a capability- and evidence-gated deployment check. Run the
secret-free preflight first; it never starts Docker or a provider:

```powershell
uv run python scripts/release_preflight.py --gate b1 --format text
uv run python scripts/release_b_gate.py --profile replay
```

The replay profile validates Compose and builds the actual Compose service
images with the validated `DEEPRESEARCH_CODE_COMMIT`. It starts the
API/Postgres/UI stack with `--no-build --wait --wait-timeout 180`, then checks
`/health/live` and `/health/ready` with bounded retries (30 seconds per probe).
See the official [Compose startup options](https://docs.docker.com/reference/cli/docker/compose/up/).
It finally runs the replay-only HTTP acceptance script: a completed,
non-partial, zero-provider-cost report, durable SSE replay, all artifact SHA-256
checks, and UI health must pass before the gate reports success. The script
uses a 90-second deadline; each external command also has a bounded timeout.

Set `API_HOST_PORT` / `UI_HOST_PORT` in the gate process environment to override
8000 / 8501. The same explicit ports and validated environment are forwarded to
Compose and the acceptance client, including cleanup; invalid ports fail
before stack mutation. A source checkout or validated `GITHUB_SHA` is required
for image provenance. This orchestration is regression-tested with controlled
process/HTTP boundaries; that is not evidence of a local Docker engine run.

The gate then runs `docker compose down`. Named
volumes are retained by that cleanup. Only an explicitly chosen `down -v`
operation removes persisted volumes. Use `--keep-up` when an operator needs
to inspect a running stack, and `--dry-run` to print the redacted command plan
without invoking Docker. A missing Docker/Compose binary, Compose file, or
Postgres configuration stops before any external command with
`DEPLOYMENT_PREREQUISITE_MISSING`.

The online smoke profile is a separate, authorized operation and can incur
provider charges:

```powershell
uv run python scripts/release_preflight.py --gate b2 --format text
uv run python scripts/release_b_gate.py --profile online-smoke
```

Supply a newly rotated `MODEL_API_KEY`, `SEARCH_API_KEY`, and a
32-byte-or-longer `SESSION_SIGNING_KEY` through the secret store, together
with complete `online-smoke` provider/pricing catalogs. The gate accepts
catalog paths (`PROVIDER_PROFILE_CATALOG_PATH` and
`PRICING_CATALOG_PATH`) or CI-provided JSON values; secret values are passed
only to the child test process and never appear in command arguments, reports,
or logs. Missing credentials return `ONLINE_SMOKE_NOT_AUTHORIZED`; an
incomplete or invalid catalog returns `ONLINE_SMOKE_INCOMPLETE` before the
service or provider is started. The previously exposed Kimi key is
compromised and must not be used.

## Publication stop conditions

Any secret leak, owner isolation failure, hash/manifest mismatch,
Replay-to-Live fallback, policy bypass, unknown-cost reservation loss, missing
research node, or unsealed Benchmark input stops publication.
