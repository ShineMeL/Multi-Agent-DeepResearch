# Task 1 report: UI repair and HTTP-only live client

## Outcome

- Added strict typed `/capabilities` models and a session-owned `GET` using the existing
  cookie-preserving `httpx.Client`. `ShowcaseSession` caches one result per browser session and
  exposes an explicit refresh path; a failed configuration fetch remains a visible failure.
- Replaced editable replay inputs with the server-advertised fixed example. The compatibility
  helpers `replay_payload` and `research_replay_payload` remain intact.
- Added the `离线示例 / 在线 API` mode switch. Online submission uses only the advertised profile
  and creates a live `baseline-v1`/P1/R1, seed-null, freshness-none Markdown request. No key input
  or secret value is rendered; unavailable live mode explains API-host `.env.demo` setup and the
  restart command. Unpriced live mode warns that cost is unknown, not free.
- Added truthful bilingual terminal/partial status and actionable `REPLAY_MISS` and
  `AUTHENTICATION` messages. Unknown run error codes remain visible in the details expander;
  exception text is never rendered.
- Removed the eager reader-completion `st.rerun()` race. The event reader now finishes, the
  bounded final status GET drains into the view, and only the terminal automatic-refresh
  transition requests one outer rerun. Cursor, cookie, idempotency, downloads, explicit recovery,
  bounded polling, and terminal polling stop behavior remain covered.

## TDD evidence

RED (before production changes):

```text
.venv/Scripts/python.exe -m pytest \
  tests/contracts/ui/test_api_client.py::test_capabilities_are_typed_and_fetched_with_the_session_owned_http_client \
  tests/contracts/ui/test_api_client.py::test_live_payload_posts_server_selected_baseline_profile_without_seed \
  tests/integration/replay/test_showcase_ui.py::test_run_status_message_never_labels_partial_or_failed_as_success -q

ImportError: cannot import name 'live_payload' from 'apps.ui.replay'
ERROR: found no collectors ...
exit 4
```

GREEN (fresh final focused run):

```text
python -m uv run --no-sync pytest tests/contracts/ui \
  tests/integration/replay/test_showcase_ui.py -q
52 passed, 1 warning in 4.64s
exit 0
```

The warning is the repository's existing Starlette `httpx` TestClient deprecation warning.

Former intermittent timeout regression, five consecutive fresh runs:

```text
1..5 | % { python -m uv run --no-sync pytest \
  tests/integration/replay/test_showcase_ui.py::test_app_submits_replay_shows_downloads_metrics_and_preserves_session \
  -q --disable-warnings }
1 passed in 2.43s / 2.50s / 2.43s / 2.43s / 2.19s
all exit 0
```

Broader replay regression run:

```text
python -m uv run --no-sync pytest tests/integration/replay -q
298 passed, 1 warning in 200.05s
exit 0
```

Static verification:

```text
python -m uv run --no-sync ruff check apps/ui tests/contracts/ui \
  tests/integration/replay/test_showcase_ui.py --no-cache
All checks passed!

python -m uv run --no-sync pyright src apps benchmarks experiments
0 errors, 0 warnings, 0 informations

git diff --check
exit 0
```

No online/paid provider call was made.

## Changed files

- `apps/ui/api_client.py`
- `apps/ui/app.py`
- `apps/ui/replay.py`
- `tests/contracts/ui/test_api_client.py`
- `tests/integration/replay/test_showcase_ui.py`
- `.superpowers/sdd/2026-09-12-demo-live-api/task-1-report.md`

## Self-review

- Scope: only the assigned UI files/tests and this report are included in the commit; concurrent
  controller server/provider changes are not staged.
- Contract: the strict UI models match the exact brief and the controller's current endpoint.
- Truthfulness: only completed, non-partial SUFFICIENT (or legacy null reason) uses success;
  partial, failed, cancelled, interrupted, blocked, and in-progress states remain explicit.
- Security: production UI imports only public domain contracts and HTTP; no server/provider/storage
  imports, credential collection, exception strings, or secret-bearing capability fields exist.
- Known gap: no paid live-provider smoke call was authorized or attempted. The live request is
  verified at the HTTP boundary with a mock transport; controller owns real adapter validation.

## Review fix round

Review found two uncovered interleavings: a contradictory partial `SUFFICIENT` view inherited
success wording, and an SSE EOF terminal GET was intentionally consumed inside the HTTP event
iterator while an earlier running poll kept the next UI status read five seconds away.

RED:

```text
python -m uv run --no-sync pytest \
  tests/integration/replay/test_showcase_ui.py::test_run_status_message_never_labels_partial_or_failed_as_success \
  tests/integration/replay/test_showcase_ui.py::test_partial_completed_sufficient_is_never_rendered_as_success \
  tests/integration/replay/test_showcase_ui.py::test_reader_completion_forces_one_immediate_final_status_after_running_poll -q
3 failed, 9 passed, 1 warning in 2.83s
exit 1
```

The failures showed the exact defects: partial `SUFFICIENT` returned the success sentence, the
AppTest had no incomplete warning, and the session view remained `running` after SSE completion.

GREEN:

```text
python -m uv run --no-sync pytest <the three review regression selections above> -q
12 passed, 1 warning in 2.08s
exit 0

python -m uv run --no-sync pytest tests/contracts/ui \
  tests/integration/replay/test_showcase_ui.py -q
55 passed, 1 warning in 5.16s
exit 0

python -m uv run --no-sync ruff check apps/ui tests/contracts/ui \
  tests/integration/replay/test_showcase_ui.py --no-cache
All checks passed!

python -m uv run --no-sync pyright apps/ui
0 errors, 0 warnings, 0 informations

git diff --check
exit 0
```

The Starlette `httpx` TestClient deprecation remains visible as the recorded dependency warning;
dependencies and warning handling were not changed. `ShowcaseSession` now observes each event
reader completion once and forces exactly one immediate background status read when its current
view is absent/running. A terminal response then stops polling. `_run_panel` no longer carries the
unused `watching_at_render` argument. Partial state is checked before any stop-reason success copy.
