import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepresearch.domain import ResourceUsage
from deepresearch.storage import (
    CacheConflictError,
    CacheEntry,
    CacheIntegrityError,
    EmbedCacheKey,
    FetchCacheKey,
    FileCache,
    ModelCacheKey,
    ParseCacheKey,
    SearchCacheKey,
    cache_key_json,
    cache_key_sha256,
)


def _search_key() -> SearchCacheKey:
    return SearchCacheKey(
        snapshot_id="snapshot-1",
        normalized_query="deterministic research",
        provider_id="search-provider",
        endpoint_type="web",
        locale="en-US",
        complete_parameters={"limit": 10, "filters": {"safe": True}},
        time_policy="as-of-snapshot",
    )


def _entry(key_sha256: str, *, producer_version: str = "v1") -> CacheEntry:
    return CacheEntry(
        key_sha256=key_sha256,
        value_artifact_id=f"sha256:{'b' * 64}",
        producer_version=producer_version,
        usage=ResourceUsage.zero(cost_known=True),
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
        metadata={"nested": {"stable": [True, 2, "value"]}},
    )


def _symlink_directory(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("snapshot_id", "snapshot-2"),
        ("locale", "zh-CN"),
        ("time_policy", "published-after-2025"),
    ],
)
def test_search_cache_key_hash_includes_every_policy_field(
    field: str, replacement: str
) -> None:
    key = _search_key()
    changed = key.model_copy(update={field: replacement})

    assert cache_key_sha256(changed) != cache_key_sha256(key)


def test_cache_key_json_is_sorted_deterministic_and_secret_free() -> None:
    first = _search_key()
    second = SearchCacheKey(
        snapshot_id="snapshot-1",
        normalized_query="deterministic research",
        provider_id="search-provider",
        endpoint_type="web",
        locale="en-US",
        complete_parameters={"filters": {"safe": True}, "limit": 10},
        time_policy="as-of-snapshot",
    )

    assert cache_key_json(first) == cache_key_json(second)
    assert cache_key_json(first) == json.dumps(
        json.loads(cache_key_json(first)),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(ValidationError, match="secret"):
        _search_key().model_copy(update={"complete_parameters": {"api_key": "hidden"}})


@pytest.mark.parametrize(
    "secret_key",
    [
        "client_secret",
        "clientSecret",
        "private_key",
        "x-api-key",
        "session_token",
        "accessToken",
        "refresh-token",
        "api_token",
    ],
)
def test_cache_rejects_compound_secret_names_recursively(secret_key: str) -> None:
    with pytest.raises(ValidationError, match="secret"):
        _search_key().model_copy(
            update={"complete_parameters": {"nested": [{secret_key: "hidden"}]}}
        )


def test_cache_allows_benign_token_accounting_names() -> None:
    parameters = {
        name: 1
        for name in (
            "max_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "prompt_tokens",
            "completionTokens",
            "token-count",
            "max_completion_tokens",
            "tokenLimit",
            "token-budget",
            "tokenUsage",
        )
    }

    key = _search_key().model_copy(update={"complete_parameters": parameters})

    assert key.complete_parameters == parameters


@pytest.mark.parametrize(
    "secret_key",
    [
        "token",
        "access_token",
        "refreshToken",
        "session-token",
        "security_token",
        "apiToken",
        "auth_token",
        "bearerToken",
        "csrf-token",
        "id_token",
        "signature",
        "privateKey",
        "authorization",
        "cookie",
        "credential",
    ],
)
def test_cache_rejects_exact_credential_token_families(secret_key: str) -> None:
    with pytest.raises(ValidationError, match="secret"):
        _search_key().model_copy(
            update={"complete_parameters": {"nested": [{secret_key: "hidden"}]}}
        )


def test_fetch_cache_key_rejects_url_userinfo() -> None:
    with pytest.raises(ValidationError, match="credentials|userinfo|secret"):
        FetchCacheKey(
            snapshot_id="s1",
            canonical_url="https://user:password@example.com/path",
            fetch_policy="fresh",
            accepted_content_types=("text/html",),
        )


@pytest.mark.parametrize(
    "parameter_name",
    [
        "api_key",
        "access_token",
        "token",
        "X-Amz-Credential",
        "X-Amz-Signature",
        "X-Amz-Security-Token",
        "X-Goog-Credential",
        "X-Goog-Signature",
        "X-Goog-Security-Token",
        "GoogleAccessId",
    ],
)
def test_fetch_cache_key_rejects_secret_query_parameters(parameter_name: str) -> None:
    with pytest.raises(ValidationError, match="secret|credential|query"):
        FetchCacheKey(
            snapshot_id="s1",
            canonical_url=f"https://example.com/path?{parameter_name}=hidden",
            fetch_policy="fresh",
            accepted_content_types=("text/html",),
        )


def test_all_operation_cache_key_fields_validate_and_affect_hash() -> None:
    keys = (
        FetchCacheKey(
            snapshot_id="s1",
            canonical_url="https://example.com/path",
            fetch_policy="fresh",
            accepted_content_types=("text/html",),
        ),
        ParseCacheKey(
            snapshot_id="s1",
            raw_content_hash="1" * 64,
            parser_id="html",
            parser_version="1",
            normalization_version="1",
        ),
        ModelCacheKey(
            provider_id="model-provider",
            endpoint_type="responses",
            model_id="m1",
            prompt_version="p1",
            system_prompt_hash="2" * 64,
            tool_schema_hash="",
            output_schema_hash="",
            temperature=Decimal("0.0"),
            seed=None,
            canonical_request_hash="3" * 64,
        ),
        EmbedCacheKey(
            model_id="e1",
            model_revision="r1",
            snapshot_sha256="4" * 64,
            normalize_embeddings=True,
            canonical_texts_hash="5" * 64,
        ),
    )

    assert len({cache_key_sha256(key) for key in keys}) == len(keys)


def _model_key_with_temperature(temperature: Decimal) -> ModelCacheKey:
    return ModelCacheKey(
        provider_id="model-provider",
        endpoint_type="responses",
        model_id="m1",
        prompt_version="p1",
        system_prompt_hash="2" * 64,
        tool_schema_hash="",
        output_schema_hash="",
        temperature=temperature,
        seed=None,
        canonical_request_hash="3" * 64,
    )


@pytest.mark.parametrize(
    "temperatures",
    [
        (Decimal("0.0"), Decimal("-0.000"), Decimal("0E+10")),
        (Decimal("0.10"), Decimal("0.1000"), Decimal("1E-1")),
        (Decimal("10.0"), Decimal("10.000"), Decimal("1E+1")),
    ],
)
def test_model_temperature_has_one_canonical_decimal_representation(
    temperatures: tuple[Decimal, Decimal, Decimal],
) -> None:
    keys = tuple(_model_key_with_temperature(value) for value in temperatures)

    assert len({cache_key_json(key) for key in keys}) == 1
    assert len({cache_key_sha256(key) for key in keys}) == 1
    assert all(isinstance(key.temperature, Decimal) for key in keys)


def test_cache_keys_reject_extra_nonfinite_and_unvalidated_copy_updates() -> None:
    with pytest.raises(ValidationError):
        SearchCacheKey(**(_search_key().model_dump() | {"unknown": "x"}))
    with pytest.raises(ValidationError):
        ModelCacheKey(
            provider_id="p",
            endpoint_type="e",
            model_id="m",
            prompt_version="v",
            system_prompt_hash="1" * 64,
            tool_schema_hash="",
            output_schema_hash="",
            temperature=Decimal("NaN"),
            canonical_request_hash="2" * 64,
        )
    with pytest.raises(ValidationError):
        _search_key().model_copy(update={"operation": "fetch"})
    with pytest.raises(ValidationError):
        _search_key().model_copy(update={"complete_parameters": {"score": float("nan")}})


def test_cache_entry_requires_aware_time_matching_ids_and_immutable_secret_free_metadata() -> None:
    key = _search_key()
    digest = cache_key_sha256(key)

    with pytest.raises(ValidationError, match="timezone"):
        _entry(digest).model_copy(
            update={"created_at": datetime(2026, 8, 29)}  # noqa: DTZ001 - invalid fixture
        )
    with pytest.raises(ValidationError, match="secret"):
        _entry(digest).model_copy(update={"metadata": {"nested": [{"password": "x"}]}})

    entry = _entry(digest)
    with pytest.raises(TypeError):
        entry.metadata["new"] = "value"
    nested = entry.metadata["nested"]
    assert isinstance(nested, dict)
    with pytest.raises(TypeError):
        nested["stable"] = False


def test_cache_entry_revalidates_nonfinite_and_constructed_invalid_usage() -> None:
    digest = cache_key_sha256(_search_key())
    nonfinite = ResourceUsage(
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=0,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=float("inf"),
        cost_usd=None,
    )
    invalid_constructed = ResourceUsage.model_construct(
        input_tokens=-1,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=-1,
        search_calls=-1,
        pages=0,
        retries=0,
        wall_seconds=-1.0,
        cost_usd=Decimal(-1),
    )
    nonfinite_cost = ResourceUsage.model_construct(
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=0,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=0.0,
        cost_usd=Decimal("Infinity"),
    )

    with pytest.raises(ValidationError, match="usage|finite|wall|cost"):
        _entry(digest).model_copy(update={"usage": nonfinite})
    with pytest.raises(ValidationError, match="usage|greater|equal"):
        _entry(digest).model_copy(update={"usage": invalid_constructed})
    with pytest.raises(ValidationError, match="usage|finite|cost"):
        _entry(digest).model_copy(update={"usage": nonfinite_cost})


def test_file_cache_put_if_absent_is_idempotent_and_never_overwrites(
    tmp_path: Path,
) -> None:
    cache = FileCache(tmp_path)
    key = _search_key()
    first = _entry(cache_key_sha256(key))
    different = _entry(cache_key_sha256(key), producer_version="v2")

    assert cache.put_if_absent(key, first) == first
    assert cache.put_if_absent(key, first) == first
    assert cache.get(key) == first
    with pytest.raises(CacheConflictError):
        cache.put_if_absent(key, different)
    assert cache.get(key) == first


def test_file_cache_rejects_symlinked_store_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _symlink_directory(tmp_path / "root" / "cache", outside)
    key = _search_key()

    with pytest.raises(CacheIntegrityError, match="symlink|reparse|contain"):
        FileCache(tmp_path / "root").put_if_absent(key, _entry(cache_key_sha256(key)))
    assert not list(outside.rglob("*.json"))


def test_file_cache_rejects_symlinked_shard_directory(tmp_path: Path) -> None:
    key = _search_key()
    digest = cache_key_sha256(key)
    cache = FileCache(tmp_path / "root")
    outside = tmp_path / "outside"
    _symlink_directory(tmp_path / "root" / "cache" / digest[:2], outside)

    with pytest.raises(CacheIntegrityError, match="symlink|reparse|contain"):
        cache.put_if_absent(key, _entry(digest))
    assert not list(outside.rglob("*.json"))


def test_file_cache_concurrent_writers_are_atomic(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    key = _search_key()
    entry = _entry(cache_key_sha256(key))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: cache.put_if_absent(key, entry), range(32)))

    assert results == (entry,) * 32
    assert cache.get(key) == entry
    assert not list(tmp_path.rglob("*.tmp"))


def test_file_cache_ignores_preexisting_unlocked_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FileCache(tmp_path)
    key = _search_key()
    digest = cache_key_sha256(key)
    lock_path = tmp_path / "cache" / digest[:2] / f"{digest}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"stale owner")
    ticks = iter((0.0, 11.0))
    monkeypatch.setattr("deepresearch.storage.cache.time.monotonic", lambda: next(ticks))

    entry = _entry(digest)

    assert cache.put_if_absent(key, entry) == entry


def test_file_cache_detects_corruption_and_key_value_mismatch(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    key = _search_key()
    digest = cache_key_sha256(key)

    with pytest.raises(CacheIntegrityError, match="key_sha256"):
        cache.put_if_absent(key, _entry("0" * 64))

    cache.put_if_absent(key, _entry(digest))
    record_path = next(tmp_path.rglob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    payload = record.get("payload", record)
    payload["producer_version"] = "tampered-but-valid"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(CacheIntegrityError):
        cache.get(key)
