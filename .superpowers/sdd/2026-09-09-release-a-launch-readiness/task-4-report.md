# Task 4 Report: Pin the Kimi-Compatible Model Contract without Enabling Live

## Status

DONE

## Changes

- Added a parameterized offline `respx` regression covering the CN and global Kimi-compatible base URLs.
- Verified both URLs call exactly `/chat/completions`, return the expected completion, send the synthetic credential only as `Authorization: Bearer ...`, and keep it out of the JSON payload and provider representation.
- Added a public-safe 401 regression with a secret-bearing upstream body; it asserts `ProviderError.code == "AUTHENTICATION"`, no key in public text, and no cause/context leakage.
- Added an `httpx.MockTransport` timeout regression; it asserts `ProviderError.code == "TIMEOUT"`, no key in public text, and closes the injected client in `finally`.
- No provider implementation, Kimi provider, `$web_search` adapter, or Live profile was added or changed.

## TDD and Verification Evidence

1. Added the requested tests first against the existing provider contract. The implementation already satisfies the brief, so no production change was needed.

2. Provider test suite:

   ```text
   python -m uv run pytest -q tests/unit/providers/test_openai_compatible.py
   38 passed in 6.78s
   ```

3. Self-review:

   ```text
   git diff --check
   exit 0
   ```

4. Commit scope is limited to `tests/unit/providers/test_openai_compatible.py` and this report.

## Concerns

- None. The existing provider implementation passed all new regional URL, header-only credential, typed error, and redaction checks.
