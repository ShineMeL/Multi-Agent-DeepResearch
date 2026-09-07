from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.datasets.isolation import GoldAccessViolation
from benchmarks.evaluators.oracle import OracleEvidenceProvider
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from experiments.config import canonical_sha256


def test_oracle_hash_only_outputs_and_determinism():
    snapshot = FrozenCorpusSnapshot.load(
        Path("tests/fixtures/frozen_corpus/task-fixture"), task_id="task-fixture"
    )
    records = {record.evidence_id: record for record in snapshot.records}
    ids = tuple(sorted(records))
    oracle = OracleEvidenceProvider(
        approved_ids_by_task={"task-fixture": ids},
        dataset_version="1.0.0",
        private_manifest_sha256="a" * 64,
        evaluator_version="evaluator-v1",
        evaluation_timestamp=datetime(2026, 8, 29, tzinfo=UTC),
    )
    result = oracle.score_reference("task-fixture", frozen_records=records)
    assert result.approved_id_set_sha256 == canonical_sha256(ids)
    assert result == oracle.score_reference("task-fixture", frozen_records=records)
    for evidence_id in ids:
        assert evidence_id not in result.model_dump_json()
    assert result.metric_values["approved_evidence_recall"].value == 1.0
    manifest = oracle.reference_manifest(group_id="group", results=(result,))
    assert manifest.created_at == result.created_at
    assert "manifest_path" not in result.model_dump()
    with pytest.raises(GoldAccessViolation):
        oracle.approved_evidence_ids_for("unknown-task")
    with pytest.raises(GoldAccessViolation):
        oracle.score_reference("task-fixture", frozen_records={})
    foreign = {
        key: record.model_copy(update={"task_id": "other"}) for key, record in records.items()
    }
    with pytest.raises(GoldAccessViolation):
        oracle.score_reference("task-fixture", frozen_records=foreign)


def test_formal_oracle_requires_and_uses_verified_snapshot_binding():
    snapshot = FrozenCorpusSnapshot.load(
        Path("tests/fixtures/frozen_corpus/task-fixture"), task_id="task-fixture"
    )
    records = {record.evidence_id: record for record in snapshot.records}
    arguments = {
        "approved_ids_by_task": {"task-fixture": tuple(records)},
        "dataset_version": "1.0.0",
        "private_manifest_sha256": "a" * 64,
        "evaluator_version": "evaluator-v1",
        "evaluation_timestamp": datetime(2026, 8, 29, tzinfo=UTC),
    }
    with pytest.raises(GoldAccessViolation, match="snapshot"):
        OracleEvidenceProvider(**arguments, formal=True)
    oracle = OracleEvidenceProvider(
        **arguments, formal=True, snapshots_by_task={"task-fixture": snapshot}
    )
    result = oracle.score_reference("task-fixture", frozen_records=records)
    assert result.frozen_snapshot_id == "snapshot-task-fixture-v1"
    with pytest.raises(GoldAccessViolation, match="snapshot"):
        oracle.score_reference(
            "task-fixture",
            frozen_records={
                key: value.model_copy(update={"title": "Unsealed title"})
                for key, value in records.items()
            },
        )
