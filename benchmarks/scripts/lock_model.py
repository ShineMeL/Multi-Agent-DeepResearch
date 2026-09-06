"""Lock immutable Hugging Face metadata without downloading model weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

import httpx
import typer
from pydantic import BaseModel, ConfigDict, Field

from benchmarks.datasets.validator import canonical_json_bytes
from experiments.config import write_immutable
from experiments.models import ModelFileLock, ModelSnapshotLock, canonical_sha256, require_revision


class _Lfs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sha256: str
    size: int


class _Sibling(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rfilename: str
    size: int
    blobId: str
    lfs: _Lfs | None = None


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    sha: str
    siblings: Annotated[tuple[_Sibling, ...], Field(min_length=1)]


def lock_model(
    *,
    repository_id: str,
    revision: str,
    output_path: Path,
    metadata: Mapping[str, object] | None = None,
) -> ModelSnapshotLock:
    require_revision(revision)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    if metadata is None:
        url = f"https://huggingface.co/api/models/{quote(repository_id, safe='/')}/revision/{revision}"
        response = httpx.get(url, params={"blobs": "true"}, timeout=60, follow_redirects=False)
        response.raise_for_status()
        parsed = _Metadata.model_validate(response.json())
    else:
        parsed = _Metadata.model_validate(metadata)
    if (parsed.id, parsed.sha) != (repository_id, revision):
        raise ValueError("repository identity or response revision mismatch")
    files: list[ModelFileLock] = []
    for item in sorted(parsed.siblings, key=lambda item: item.rfilename):
        if item.lfs is not None and item.size != item.lfs.size:
            raise ValueError("LFS metadata size mismatch")
        files.append(
            ModelFileLock(
                path=item.rfilename,
                size=item.size,
                git_blob_or_lfs_oid=item.lfs.sha256 if item.lfs else item.blobId,
            )
        )
    lock = ModelSnapshotLock(
        repository_id=repository_id,
        requested_revision=revision,
        resolved_revision=parsed.sha,
        files=tuple(files),
        snapshot_sha256=canonical_sha256([item.model_dump(mode="json") for item in files]),
    )
    write_immutable(output_path, canonical_json_bytes(lock.model_dump(mode="json")))
    return lock


def main(
    repo: Annotated[str, typer.Option()],
    revision: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    lock_model(repository_id=repo, revision=revision, output_path=output)


if __name__ == "__main__":
    typer.run(main)
