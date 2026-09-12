"""Native Windows file operations fail before touching handles on other hosts."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import apps.cli.main as cli
from deepresearch.providers import recording


class ForbiddenNativeHandle:
    def __getattr__(self, name: str) -> object:
        pytest.fail(f"unsupported platform attempted native operation: {name}")


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda path: recording._open_windows_marker(path, create_new=True), id="marker"
        ),
        pytest.param(recording._open_windows_directory, id="directory"),
        pytest.param(
            lambda _: recording._windows_handle_identity(ForbiddenNativeHandle(), 1), id="identity"
        ),
        pytest.param(
            lambda _: recording._close_windows_handle(ForbiddenNativeHandle(), 1), id="close"
        ),
        pytest.param(
            lambda _: recording._write_windows_marker(ForbiddenNativeHandle(), 1, b"owned"),
            id="write",
        ),
        pytest.param(
            lambda _: recording._read_windows_marker(ForbiddenNativeHandle(), 1, 5), id="read"
        ),
        pytest.param(
            lambda _: recording._set_windows_delete_disposition(ForbiddenNativeHandle(), 1),
            id="delete-disposition",
        ),
        pytest.param(recording._flush_windows_directory, id="recording-flush"),
        pytest.param(cli._flush_windows_directory, id="cli-flush"),
    ],
)
def test_native_windows_operations_reject_unsupported_platform_before_io(
    operation: Callable[[Path], object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes

    def forbidden_dll(*_: object, **__: object) -> object:
        pytest.fail("unsupported platform attempted to load a Windows DLL")

    monkeypatch.setattr(recording, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(ctypes, "WinDLL", forbidden_dll, raising=False)

    with pytest.raises(NotImplementedError, match="Windows"):
        operation(tmp_path / "untouched")

    assert not tuple(tmp_path.iterdir())
