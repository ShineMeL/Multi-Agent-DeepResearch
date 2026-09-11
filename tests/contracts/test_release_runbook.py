from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def test_runbooks_instruct_clean_lf_checkout_and_never_regenerate_hashes() -> None:
    deployment = Path("docs/deployment.md").read_text(encoding="utf-8")
    evaluation = Path("docs/evaluation.md").read_text(encoding="utf-8")
    combined = deployment + evaluation

    assert "core.autocrlf=false" in combined
    assert "Do not regenerate expected hashes" in combined
    assert "release_preflight.py" in combined


def test_ci_keeps_online_smoke_manual_and_secret_free_verify_job() -> None:
    workflow: dict[str, Any] = yaml.safe_load(
        Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    )

    assert "secrets" not in json.dumps(workflow["jobs"]["verify"])
    triggers = json.dumps(workflow.get(True, workflow.get("on", {})))
    assert "workflow_dispatch" in triggers
    assert "schedule" in triggers
    assert "online-smoke" in json.dumps(workflow["jobs"])


def test_ci_runs_preflight_before_optional_formal_work() -> None:
    workflow: dict[str, Any] = yaml.safe_load(
        Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    commands = "\n".join(
        step.get("run", "") for step in workflow["jobs"]["verify"]["steps"]
    )

    assert "scripts/release_preflight.py --gate c1 --format json" in commands
    assert "scripts/release_b_gate.py --profile online-smoke" not in commands
