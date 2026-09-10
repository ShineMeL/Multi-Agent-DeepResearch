"""The shipped server command must preserve the transport peer for app policy."""

import shlex
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from uvicorn.main import main


def test_compose_server_disables_uvicorn_proxy_rewriting(monkeypatch: pytest.MonkeyPatch) -> None:
    config: Any = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    arguments = shlex.split(config["services"]["api"]["command"])
    captured: dict[str, Any] = {}

    def capture_server(app: str, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(import_module("uvicorn.main"), "run", capture_server)
    result = CliRunner().invoke(main, arguments[arguments.index("uvicorn") + 1 :])
    assert result.exit_code == 0, result.output
    assert captured["proxy_headers"] is False
