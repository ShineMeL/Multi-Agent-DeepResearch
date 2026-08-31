from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from deepresearch import runtime as runtime_package
from deepresearch.domain import CoverageLedgerEntry, FreshnessRequirement, ResearchRequest
from deepresearch.runtime import CheckpointRef
from deepresearch.runtime.checkpoints import (
    CheckpointIdentityError,
    CheckpointSerializationError,
    checkpoint_config,
    checkpoint_ref_from_tuple,
    checkpoint_serializer,
    open_sqlite_checkpointer,
)


def request() -> ResearchRequest:
    return ResearchRequest(
        question="Which methods are documented?",
        output_requirements={"answer_shape": "brief"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="test",
        budget_preset="low",
    )


def test_runtime_publicly_exports_checkpoint_factories() -> None:
    assert runtime_package.checkpoint_serializer is checkpoint_serializer
    assert runtime_package.open_sqlite_checkpointer is open_sqlite_checkpointer
    assert "checkpoint_serializer" in runtime_package.__all__
    assert "open_sqlite_checkpointer" in runtime_package.__all__


def ledger() -> CoverageLedgerEntry:
    return CoverageLedgerEntry(
        subquestion_id="sq-1",
        coverage_score=0.9,
        independent_source_count=2,
        unresolved_conflict_ids=(),
        uncertainty_score=0.1,
        last_marginal_gain=0.2,
        evidence_ids=("E-1", "E-2"),
        attempt_count=1,
        last_decision_code="RANKED",
    )


def test_checkpoint_serializer_preserves_models_bytes_tuples_and_mapping_collisions() -> None:
    serde = checkpoint_serializer()
    collision = {
        "lc": 2,
        "type": "constructor",
        "id": ["deepresearch", "runtime", "checkpoints", "_TupleEnvelope"],
        "kwargs": {"items": ["ordinary", "mapping"]},
    }
    value = {
        "request": request(),
        "coverage": (ledger(),),
        "nested": ("outer", ("inner", 3)),
        "legacy_bytes": b"checkpoint-bytes",
        "collision": collision,
    }

    restored = serde.loads_typed(serde.dumps_typed(value))

    assert restored == value
    assert type(restored["coverage"]) is tuple
    assert type(restored["nested"]) is tuple
    assert type(restored["nested"][1]) is tuple
    assert restored["collision"] == collision


@pytest.mark.parametrize(
    "value",
    (object(), Path("blocked"), lambda: None, RuntimeError("secret detail")),
    ids=("object", "path", "callable", "exception"),
)
def test_checkpoint_serializer_rejects_unapproved_values_without_repr(value: object) -> None:
    with pytest.raises(CheckpointSerializationError) as error:
        checkpoint_serializer().dumps_typed({"value": value})

    assert str(error.value) == "checkpoint value is not serializable"
    assert "secret detail" not in str(error.value)


def test_checkpoint_serializer_rejects_tuple_subclasses_without_iterating() -> None:
    iterated = False

    class TupleSubclass(tuple[object, ...]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            nonlocal iterated
            iterated = True
            return super().__iter__()

    with pytest.raises(CheckpointSerializationError):
        checkpoint_serializer().dumps_typed(TupleSubclass(("a", "b")))

    assert iterated is False


def test_checkpoint_serializer_content_revalidates_approved_models() -> None:
    invalid = ResearchRequest.model_construct(question="only field")
    permissive = JsonPlusSerializer(
        allowed_json_modules=((ResearchRequest.__module__, ResearchRequest.__name__),),
        allowed_msgpack_modules=(ResearchRequest,),
        pickle_fallback=False,
    )
    invalid_payload = permissive.dumps_typed(invalid)

    with pytest.raises(CheckpointSerializationError):
        checkpoint_serializer().dumps_typed(invalid)
    with pytest.raises(CheckpointSerializationError):
        checkpoint_serializer().loads_typed(invalid_payload)


@pytest.mark.parametrize(
    "data",
    (("unknown", b"payload"), ("msgpack", b"\xc7\x01\x7f\x00")),
)
def test_checkpoint_serializer_rejects_unknown_or_corrupt_payloads(
    data: tuple[str, bytes],
) -> None:
    with pytest.raises(CheckpointSerializationError):
        checkpoint_serializer().loads_typed(data)


def test_checkpoint_serializer_propagates_memory_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dumps(serializer: object, value: object) -> tuple[str, bytes]:
        raise MemoryError

    def fail_loads(serializer: object, data: tuple[str, bytes]) -> object:
        raise MemoryError

    monkeypatch.setattr(JsonPlusSerializer, "dumps_typed", fail_dumps)
    with pytest.raises(MemoryError):
        checkpoint_serializer().dumps_typed({"safe": "value"})

    monkeypatch.setattr(JsonPlusSerializer, "loads_typed", fail_loads)
    with pytest.raises(MemoryError):
        checkpoint_serializer().loads_typed(("null", b""))


@pytest.mark.parametrize(
    "data",
    (("null", b"garbage"), ("json", b'{ "safe": true }')),
)
def test_checkpoint_serializer_rejects_noncanonical_label_payloads(
    data: tuple[str, bytes],
) -> None:
    with pytest.raises(CheckpointSerializationError):
        checkpoint_serializer().loads_typed(data)


@pytest.mark.parametrize("path", (Path("relative.sqlite3"), Path(Path.cwd().anchor)))
async def test_sqlite_checkpointer_rejects_relative_or_anchor_paths(path: Path) -> None:
    with pytest.raises(ValueError, match="absolute.*file"):
        async with open_sqlite_checkpointer(path):
            pytest.fail("unsafe path was opened")


async def test_sqlite_checkpointer_rejects_directory_and_final_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="file"):
        async with open_sqlite_checkpointer(tmp_path):
            pytest.fail("directory was opened")

    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    link = tmp_path / "link.sqlite3"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    with pytest.raises(ValueError, match="symlink|reparse"):
        async with open_sqlite_checkpointer(link):
            pytest.fail("symlink was opened")


class CounterState(TypedDict):
    value: int


async def test_sqlite_checkpointer_survives_reopen_and_exact_resume(
    tmp_path: Path,
) -> None:
    calls = 0

    async def increment(state: CounterState) -> CounterState:
        nonlocal calls
        calls += 1
        return {"value": state["value"] + 1}

    path = (tmp_path / "checkpoints.sqlite3").resolve()
    async with open_sqlite_checkpointer(path) as saver:
        builder = StateGraph(CounterState)
        builder.add_node("increment", increment)
        builder.set_entry_point("increment")
        builder.add_edge("increment", END)
        graph = builder.compile(checkpointer=saver)
        output = await graph.ainvoke(
            {"value": 0},
            {"configurable": {"thread_id": "thread-1"}},
        )
        saved = await saver.aget_tuple(
            {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
        )
        assert saved is not None
        ref = checkpoint_ref_from_tuple(saved)

    async with open_sqlite_checkpointer(path) as saver:
        resumed_builder = StateGraph(CounterState)
        resumed_builder.add_node("increment", increment)
        resumed_builder.set_entry_point("increment")
        resumed_builder.add_edge("increment", END)
        resumed = resumed_builder.compile(checkpointer=saver)
        resumed_output = await resumed.ainvoke(None, checkpoint_config(ref))

    assert output == resumed_output == {"value": 1}
    assert calls == 1


def test_checkpoint_identity_is_strict_and_timezone_aware() -> None:
    created = datetime(2026, 9, 1, 3, 4, 5, tzinfo=UTC)
    valid = SimpleNamespace(
        config={
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        },
        checkpoint={"id": "cp-1", "ts": created.isoformat()},
    )
    ref = checkpoint_ref_from_tuple(cast("object", valid))

    assert ref == CheckpointRef(
        checkpoint_id="cp-1",
        thread_id="thread-1",
        created_at=created,
    )
    assert checkpoint_config(ref) == {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-1",
        }
    }

    for configurable, checkpoint in (
        ({"thread_id": "thread-1", "checkpoint_id": "cp-1"}, {"id": "cp-2", "ts": created.isoformat()}),
        ({"thread_id": "thread-1", "checkpoint_ns": "nested", "checkpoint_id": "cp-1"}, {"id": "cp-1", "ts": created.isoformat()}),
        ({"thread_id": 1, "checkpoint_ns": "", "checkpoint_id": "cp-1"}, {"id": "cp-1", "ts": created.isoformat()}),
        ({"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "cp-1"}, {"id": "cp-1", "ts": "2026-09-01T03:04:05"}),
    ):
        malformed = SimpleNamespace(
            config={"configurable": configurable},
            checkpoint=checkpoint,
        )
        with pytest.raises(CheckpointIdentityError) as error:
            checkpoint_ref_from_tuple(cast("object", malformed))
        assert error.value.code == "CHECKPOINT_MISMATCH"


@pytest.mark.parametrize(
    ("checkpoint_id", "thread_id", "created_at"),
    (
        (1, "thread-1", datetime(2026, 9, 1, tzinfo=UTC)),
        ("cp-1", 2, datetime(2026, 9, 1, tzinfo=UTC)),
        ("cp-1", "thread-1", "2026-09-01T00:00:00+00:00"),
        ("", "thread-1", datetime(2026, 9, 1, tzinfo=UTC)),
        ("cp-1", "", datetime(2026, 9, 1, tzinfo=UTC)),
    ),
)
def test_checkpoint_ref_strictly_rejects_invalid_field_types(
    checkpoint_id: object,
    thread_id: object,
    created_at: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CheckpointRef(
            checkpoint_id=checkpoint_id,  # type: ignore[arg-type]
            thread_id=thread_id,  # type: ignore[arg-type]
            created_at=created_at,  # type: ignore[arg-type]
        )


def test_checkpoint_ref_rejects_datetime_subclass_without_calling_it() -> None:
    called = False

    class DatetimeSubclass(datetime):
        def utcoffset(self):  # type: ignore[no-untyped-def]
            nonlocal called
            called = True
            return super().utcoffset()

    value = DatetimeSubclass(2026, 9, 1, tzinfo=UTC)

    with pytest.raises(TypeError, match="created_at is invalid"):
        CheckpointRef(checkpoint_id="cp-1", thread_id="thread-1", created_at=value)

    assert called is False
