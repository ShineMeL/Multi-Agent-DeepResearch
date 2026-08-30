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


def test_file_cache_concurrent_writers_are_atomic(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    key = _search_key()
    entry = _entry(cache_key_sha256(key))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: cache.put_if_absent(key, entry), range(32)))

    assert results == (entry,) * 32
    assert cache.get(key) == entry
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


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
