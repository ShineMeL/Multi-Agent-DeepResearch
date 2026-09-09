from __future__ import annotations

import subprocess


def test_hash_addressed_assets_are_checked_out_as_lf() -> None:
    paths = [
        "benchmarks/datasets/frozen_ai_cs_60/public_manifest.json",
        "benchmarks/datasets/frozen_ai_cs_60/runtime/dev/technical_survey.jsonl",
        "benchmarks/snapshots/frozen_ai_cs_60/dev-ts-01/snapshot.json",
        "benchmarks/snapshots/frozen_ai_cs_60/dev-ts-01/manifest.sha256",
        "tests/fixtures/frozen_corpus/task-fixture/documents.jsonl",
        "tests/fixtures/frozen_corpus/task-fixture/index.json",
        "tests/fixtures/frozen_corpus/task-fixture/snapshot.json",
        "tests/fixtures/frozen_corpus/task-fixture/manifest.sha256",
        "tests/fixtures/replay/baseline/documents.jsonl",
        "tests/fixtures/replay/baseline/embeddings.jsonl",
        "tests/fixtures/replay/baseline/expected-evidence.json",
        "tests/fixtures/replay/baseline/expected-report.md",
        "tests/fixtures/replay/baseline/model_responses.jsonl",
        "tests/fixtures/replay/baseline/search.jsonl",
        "tests/fixtures/replay/baseline/snapshot.json",
        "tests/fixtures/replay/baseline/manifest.sha256",
    ]
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    )
    attributes = {}
    for line in result.stdout.splitlines():
        path, attribute, value = line.rsplit(": ", 2)
        attributes[path.replace("\\", "/")] = (attribute, value)
    assert set(attributes) == set(paths)
    assert all(attribute == "eol" and value == "lf" for attribute, value in attributes.values())
