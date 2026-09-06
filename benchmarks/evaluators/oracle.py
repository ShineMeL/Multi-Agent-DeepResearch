"""Evaluator-only approved evidence reference. Never a SearchProvider or agent record."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType

from benchmarks.datasets.isolation import GoldAccessViolation
from benchmarks.datasets.models import FrozenEvidenceRecord
from benchmarks.evaluators.metrics import ratio_metric
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from experiments.models import (
    EvaluatorReferenceManifest,
    OracleReferenceResult,
    canonical_sha256,
    require_hash,
)


class OracleEvidenceProvider:
    def __init__(
        self,
        *,
        approved_ids_by_task: Mapping[str, tuple[str, ...]],
        dataset_version: str,
        private_manifest_sha256: str,
        evaluator_version: str,
        evaluation_timestamp: datetime,
        formal: bool = False,
        snapshots_by_task: Mapping[str, FrozenCorpusSnapshot] | None = None,
    ) -> None:
        require_hash(private_manifest_sha256)
        if evaluation_timestamp.tzinfo is None:
            raise ValueError("sealed evaluation_timestamp must be timezone aware")
        approved = {task: tuple(sorted(ids)) for task, ids in approved_ids_by_task.items()}
        if not approved or any(
            not task or not ids or len(set(ids)) != len(ids) or any(not item for item in ids)
            for task, ids in approved.items()
        ):
            raise GoldAccessViolation("approved mapping must contain unique non-empty IDs")
        self._approved = MappingProxyType(approved)
        self._dataset_version = dataset_version
        self._private_hash = private_manifest_sha256
        self._evaluator_version = evaluator_version
        self._timestamp = evaluation_timestamp
        self._snapshots: dict[str, FrozenCorpusSnapshot] = {}
        if formal and snapshots_by_task is None:
            raise GoldAccessViolation("formal ORACLE requires verified snapshot bindings")
        if snapshots_by_task is not None:
            if set(snapshots_by_task) != set(approved):
                raise GoldAccessViolation("snapshot bindings must exactly match oracle tasks")
            for task_id, snapshot in snapshots_by_task.items():
                verified = FrozenCorpusSnapshot.load(snapshot.root, task_id=task_id)
                if verified.manifest != snapshot.manifest:
                    raise GoldAccessViolation("snapshot binding disagrees with verified manifest")
                self._snapshots[task_id] = verified

    def approved_evidence_ids_for(self, task_id: str) -> tuple[str, ...]:
        try:
            return self._approved[task_id]
        except KeyError:
            raise GoldAccessViolation("unknown oracle task") from None

    def score_reference(
        self, task_id: str, *, frozen_records: Mapping[str, FrozenEvidenceRecord]
    ) -> OracleReferenceResult:
        ids = self.approved_evidence_ids_for(task_id)
        snapshot = self._snapshots.get(task_id)
        if snapshot is not None and dict(frozen_records) != {
            record.evidence_id: record for record in snapshot.records
        }:
            raise GoldAccessViolation("records disagree with verified snapshot binding")
        for identity in ids:
            record = frozen_records.get(identity)
            if record is None or record.task_id != task_id or record.evidence_id != identity:
                raise GoldAccessViolation("approved evidence is absent from the task frozen corpus")
            FrozenEvidenceRecord.model_validate_json(record.model_dump_json())
        # IDs/excerpts remain here. This reference measures approved retrieval availability;
        # answer/claim quality requires a generated report and is not fabricated for ORACLE.
        metric = ratio_metric("approved_evidence_recall", len(ids), len(ids))
        return OracleReferenceResult(
            task_id=task_id,
            dataset_version=self._dataset_version,
            private_manifest_sha256=self._private_hash,
            frozen_snapshot_id=snapshot.manifest.snapshot_id
            if snapshot
            else "offline-corpus-"
            + canonical_sha256(
                [record.model_dump(mode="json") for _, record in sorted(frozen_records.items())]
            ),
            approved_id_set_sha256=canonical_sha256(ids),
            metric_values={metric.name: metric},
            evaluator_version=self._evaluator_version,
            created_at=self._timestamp,
        )

    def reference_manifest(
        self, *, group_id: str, results: Sequence[OracleReferenceResult]
    ) -> EvaluatorReferenceManifest:
        ordered = tuple(sorted(results, key=lambda result: result.task_id))
        if not ordered or len({result.task_id for result in ordered}) != len(ordered):
            raise GoldAccessViolation("reference results must be non-empty and task-unique")
        for result in ordered:
            snapshot = self._snapshots.get(result.task_id)
            if snapshot is not None and result.frozen_snapshot_id != snapshot.manifest.snapshot_id:
                raise GoldAccessViolation("reference result snapshot identity mismatch")
            if (
                result.task_id not in self._approved
                or result.approved_id_set_sha256 != canonical_sha256(self._approved[result.task_id])
                or (
                    result.private_manifest_sha256,
                    result.evaluator_version,
                    result.created_at,
                    result.dataset_version,
                )
                != (
                    self._private_hash,
                    self._evaluator_version,
                    self._timestamp,
                    self._dataset_version,
                )
            ):
                raise GoldAccessViolation(
                    "reference result does not match sealed evaluator identity"
                )
        return EvaluatorReferenceManifest(
            group_id=group_id,
            private_manifest_sha256=self._private_hash,
            evaluator_version=self._evaluator_version,
            task_ids_sha256=canonical_sha256(tuple(result.task_id for result in ordered)),
            oracle_results_sha256=canonical_sha256([r.model_dump(mode="json") for r in ordered]),
            created_at=self._timestamp,
        )
