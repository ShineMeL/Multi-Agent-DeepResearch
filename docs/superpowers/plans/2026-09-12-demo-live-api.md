# Demo stop status and live API integration

User-authorized scope: fix confusing offline completion status and enable API-backed demos.

## Findings

- Persisted packaged replay runs finish with `SUFFICIENT`; modified questions fail
  with `REPLAY_MISS`. The UI displays English codes, allows incompatible recording
  parameters, and displays `Not reported` both while working and after failures.
- API and UI are running from the previous research-v1 worktree. No model/search
  credentials are configured in the process or local environment files.
- The service selects legacy replay prompt compatibility for the entire deployment.
  Enabling live alongside replay will incorrectly change the bundled replay request.
- Provider adapters exist, but the UI has no live mode and no turnkey local API
  configuration. Kimi has model-specific sampling/thinking parameters to support.

## Implementation

1. Fix UI completion/error messages and terminal refresh behavior; pin the built-in
   offline example to its recorded request. Prove `SUFFICIENT`, actual partial stops,
   pending, and replay mismatch remain distinct in Streamlit behavior tests.
2. Add server capability discovery and an explicit local demo launcher/config file.
   Replay stays available; live requires model and Tavily keys. Credentials are
   server-only, configuration status is public and contains no keys or paths.
3. Configure real model/search/fetch/parser/ranking composition. Resolve content
   boundaries per run when live and replay coexist. Use a supported Kimi adapter
   with explicit frozen wire behavior; retain generic chat-completions support.
   Local unpriced live runs must report unknown cost and keep token/search/time
   limits; public or formal cost requirements are unchanged.
4. Test real service composition with mocked external HTTP responses, real replay
   and Streamlit acceptance, source lint/type checks. Restart local demo, verify
   actual endpoints and update the usage documentation. Run paid smoke only with
   locally configured credentials; report missing credentials as a concrete blocker.

## Verification

- Real loopback HTTP replay completed with `SUFFICIENT`, `is_partial=false`,
  report/evidence/manifest HTTP 200, terminal SSE and zero cost (run
  `aa2f7b6f-dcdc-4c48-b0cc-ce45a7af6f2f`). Both service health checks returned 200.
- UI/API/provider/launcher focus: 70 passed before the final gated-poll regression;
  revised UI suite: 56 passed. Server/provider/launcher additions: 17 passed.
- Core/provider/API/replay/architecture regressions: 425 passed, 2 skipped.
  Updated deployment plus full-service replay suite: 52 passed.
- First full run: 1887 passed, 10 skipped, 2 failed. Both failures asserted retired
  wiring (global replay boundary, UI-owned question literal); the tests now cover
  per-run boundary selection and API-owned example documentation. Full rerun is pending.
- Production Pyright and Ruff lint pass. Whole-repository formatting has existing
  drift (127 files on the first check); unrelated code was not mechanically rewritten.
- Actual paid Kimi/Tavily smoke remains unverified: no fresh local credentials.
  External HTTP is mocked in online integration tests; replay never substitutes
  for live verification. Browser automation is unavailable, so UI behavior uses
  Streamlit AppTest and real HTTP service acceptance, not screenshot evidence.

## Additional findings resolved during implementation

- Real structured provider results lost their concrete output type during usage
  settlement; preserve the typed result through Pydantic copy/validation.
- Kimi normalization must happen before cache and audit identity, not only at
  HTTP encoding. Replay request bytes are preserved unchanged.
- A finite P1 plan can exhaust its query strategies before sufficient coverage.
  Preserve partial evidence/report with `BLOCKED`, rather than fail with
  `NO_LEGAL_CONTINUATION`. Sufficiency thresholds were not reduced.
- Capability discovery uses the manager's startup catalog and only modes/budgets
  supported by this thin client. Configuration changes require restart.
