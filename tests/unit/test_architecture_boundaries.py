from __future__ import annotations

import ast
import inspect
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, cast, get_args, get_origin, get_type_hints

import pytest
from pydantic import SecretStr

from deepresearch.domain import (
    CoverageLedgerEntry,
    FreshnessRequirement,
    ResearchRequest,
    RunBudget,
)
from deepresearch.providers import ModelProvider
from deepresearch.runtime import (
    BudgetAccountant,
    BudgetSnapshot,
    CheckpointRef,
    checkpoint_serializer,
)
from deepresearch.runtime.manifest import RunManifest
from deepresearch.runtime.ports import CheckpointRef as PortsCheckpointRef
from deepresearch.workflow import (
    BaselineBlockedNeed,
    BaselineState,
    StateValidationError,
    validate_baseline_state,
)

ROOT = Path(__file__).parents[2]
SOURCE_ROOTS = (ROOT / "src" / "deepresearch", ROOT / "apps")
PROVIDER_ROOT = ROOT / "src" / "deepresearch" / "providers"
SDK_PREFIXES = (
    "fitz",
    "huggingface_hub",
    "httpx",
    "openai",
    "pymupdf",
    "sentence_transformers",
    "tavily",
    "trafilatura",
)


def _python_files() -> list[Path]:
    return sorted(
        path
        for root in SOURCE_ROOTS
        for path in root.rglob("*.py")
        if path.is_file()
    )


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
    return found


def _diagnostic(path: Path, line: int, module: str) -> str:
    return f"{path.relative_to(ROOT).as_posix()}:{line}:{module}"


def _starts_with_module(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def test_external_sdks_are_imported_only_in_provider_modules() -> None:
    violations: list[str] = []
    for path in _python_files():
        if path.is_relative_to(PROVIDER_ROOT):
            continue
        for line, module in _imports(path):
            if any(_starts_with_module(module, prefix) for prefix in SDK_PREFIXES):
                violations.append(_diagnostic(path, line, module))
    assert violations == []


def test_langgraph_imports_stay_in_checkpoint_or_workflow_orchestration() -> None:
    allowed = (
        ROOT / "src" / "deepresearch" / "runtime" / "checkpoints.py",
        ROOT / "src" / "deepresearch" / "workflow",
    )
    violations: list[str] = []
    for path in _python_files():
        if any(path == root or path.is_relative_to(root) for root in allowed):
            continue
        for line, module in _imports(path):
            if _starts_with_module(module, "langgraph"):
                violations.append(_diagnostic(path, line, module))
    assert violations == []


def test_core_does_not_depend_on_later_service_packages() -> None:
    violations: list[str] = []
    core_root = ROOT / "src" / "deepresearch"
    for path in sorted(core_root.rglob("*.py")):
        for line, module in _imports(path):
            if module.startswith(("apps.api", "apps.ui")):
                violations.append(_diagnostic(path, line, module))
    assert violations == []


def test_cross_layer_imports_do_not_use_private_deepresearch_modules() -> None:
    violations: list[str] = []
    for path in _python_files():
        for line, module in _imports(path):
            components = module.split(".")
            if components[0] != "deepresearch":
                continue
            if any(component.startswith("_") for component in components[1:]):
                violations.append(_diagnostic(path, line, module))
    assert violations == []


def test_public_contract_definitions_have_one_owner() -> None:
    owned = (
        (ResearchRequest, "deepresearch.domain"),
        (CoverageLedgerEntry, "deepresearch.domain"),
        (ModelProvider, "deepresearch.providers.protocols"),
        (CheckpointRef, "deepresearch.runtime.ports"),
        (PortsCheckpointRef, "deepresearch.runtime.ports"),
        (RunManifest, "deepresearch.runtime.manifest"),
    )
    for symbol, module_prefix in owned:
        module = inspect.getmodule(symbol)
        assert module is not None
        assert module.__name__ == module_prefix or module.__name__.startswith(
            f"{module_prefix}."
        )


def _leaf_types(annotation: object) -> set[object]:
    origin = get_origin(annotation)
    if origin is Annotated:
        return _leaf_types(get_args(annotation)[0])
    if origin is Literal:
        return set()
    if origin in (types.UnionType, getattr(types, "UnionType", object)):
        leaves: set[object] = set()
        for item in get_args(annotation):
            leaves.update(_leaf_types(item))
        return leaves
    if origin is not None:
        leaves: set[object] = set()
        for item in get_args(annotation):
            if item is not Ellipsis:
                leaves.update(_leaf_types(item))
        return leaves
    if annotation is BaselineBlockedNeed:
        leaves: set[object] = set()
        for item in get_type_hints(BaselineBlockedNeed, include_extras=True).values():
            leaves.update(_leaf_types(item))
        return leaves
    return {annotation}


def test_graph_state_leaf_types_exclude_raw_payloads_and_secrets() -> None:
    annotations = get_type_hints(BaselineState, include_extras=True)
    leaves: set[object] = set()
    for annotation in annotations.values():
        leaves.update(_leaf_types(annotation))

    forbidden = {
        object,
        bytes,
        bytearray,
        SecretStr,
    }
    assert leaves.isdisjoint(forbidden)
    allowed = {
        type(None),
        bool,
        int,
        float,
        str,
        ResearchRequest,
        CoverageLedgerEntry,
        BudgetSnapshot,
    }
    assert leaves <= allowed, sorted(repr(item) for item in leaves - allowed)


def _minimal_state() -> dict[str, object]:
    budget = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    snapshot = BudgetAccountant(budget, run_scope="architecture-boundary").snapshot()
    request = ResearchRequest(
        question="architecture boundary",
        output_requirements={"answer_shape": "markdown"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="local",
        provider_profile_id="baseline-replay-v1:" + "a" * 64,
        run_purpose="test",
        budget_preset="low",
    )
    return {
        "run_id": "run-architecture",
        "thread_id": "run-architecture",
        "request": request,
        "config_sha256": "b" * 64,
        "plan_id": None,
        "plan_artifact_id": None,
        "pending_subquestion_ids": (),
        "active_subquestion_id": None,
        "query_ids": (),
        "source_ids": (),
        "evidence_ids": (),
        "selected_evidence_ids": (),
        "coverage_ledger": (),
        "high_priority_unresolved_conflict_ids": (),
        "blocked_needs": (),
        "recent_marginal_gains": (),
        "baseline_work_artifact_ids": (),
        "budget_snapshot": snapshot,
        "stop_reason": None,
        "is_partial": False,
        "draft_artifact_id": None,
        "report_artifact_id": None,
        "evidence_graph_artifact_id": None,
        "manifest_artifact_id": None,
        "next_event_seq": 1,
        "failed_node": None,
        "elapsed_wall_seconds": 0.0,
        "error_code": None,
    }


def test_state_validator_and_checkpoint_serializer_keep_payloads_compact() -> None:
    state = _minimal_state()
    validated = validate_baseline_state(state)
    label, payload = checkpoint_serializer().dumps_typed(validated)
    restored = checkpoint_serializer().loads_typed((label, payload))
    assert isinstance(restored, Mapping)
    validate_baseline_state(cast("Mapping[str, object]", restored))

    state_with_raw_payload = dict(state)
    state_with_raw_payload["raw_document"] = {"body": b"not checkpoint state"}
    with pytest.raises(StateValidationError):
        validate_baseline_state(state_with_raw_payload)
