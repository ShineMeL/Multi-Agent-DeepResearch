from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from deepresearch.domain import EvidenceSpan, SourceDocument
from deepresearch.retrieval import near_duplicate_signals


@dataclass(frozen=True)
class EvidenceCandidate:
    evidence: EvidenceSpan
    source: SourceDocument
    search_rank: int
    source_family_id: str

    def __post_init__(self) -> None:
        if type(self.search_rank) is not int or self.search_rank < 1:
            raise ValueError("search_rank must be a positive integer")
        if self.evidence.source_id != self.source.source_id:
            raise ValueError("evidence source_id must match source source_id")
        if type(self.source_family_id) is not str or not self.source_family_id.strip():
            raise ValueError("source_family_id is required")


class EvidenceNormalizer:
    def assign_source_families(
        self,
        sources: Sequence[SourceDocument],
    ) -> Mapping[str, str]:
        ordered = sorted(sources, key=lambda source: source.source_id)
        if len({source.source_id for source in ordered}) != len(ordered):
            raise ValueError("source IDs must be unique")
        parent = list(range(len(ordered)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left_index, left in enumerate(ordered):
            for right_index in range(left_index + 1, len(ordered)):
                right = ordered[right_index]
                same_url = str(left.canonical_url) == str(right.canonical_url)
                same_hash = left.parsed_content_hash == right.parsed_content_hash
                near_duplicate = near_duplicate_signals(
                    left_text=left.title,
                    right_text=right.title,
                    left_title=left.title,
                    right_title=right.title,
                ).is_near_duplicate
                if same_url or same_hash or near_duplicate:
                    union(left_index, right_index)

        components: dict[int, list[SourceDocument]] = {}
        for index, source in enumerate(ordered):
            components.setdefault(find(index), []).append(source)

        families: dict[str, str] = {}
        for members in components.values():
            representative = members[0]
            family_id = representative.source_family_id.strip()
            if not family_id:
                seed = f"{representative.canonical_url}|{representative.parsed_content_hash}"
                family_id = "family-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
            for source in members:
                families[source.source_id] = family_id
        return families

    def dedupe(self, candidates: Sequence[EvidenceCandidate]) -> list[EvidenceCandidate]:
        if not candidates:
            return []
        families = self.assign_source_families([item.source for item in candidates])
        seen_exact_spans: set[tuple[str, str]] = set()
        kept: list[EvidenceCandidate] = []
        for item in sorted(candidates, key=lambda candidate: (candidate.search_rank, candidate.evidence.evidence_id)):
            family = families[item.source.source_id]
            exact_key = (family, item.evidence.excerpt_hash)
            if exact_key in seen_exact_spans:
                continue
            duplicate = False
            for previous in kept:
                if previous.source_family_id != family:
                    continue
                signals = near_duplicate_signals(
                    left_text=previous.evidence.excerpt,
                    right_text=item.evidence.excerpt,
                    left_title=previous.source.title,
                    right_title=item.source.title,
                )
                if signals.is_near_duplicate:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(replace(item, source_family_id=family))
            seen_exact_spans.add(exact_key)
        return kept


__all__ = ["EvidenceCandidate", "EvidenceNormalizer"]
