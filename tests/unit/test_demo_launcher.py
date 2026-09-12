import json
from pathlib import Path

from apps.api.demo import demo_environment, prepare_demo


def test_launcher_readiness_probes_use_the_actual_public_health_route(tmp_path):
    from fastapi.testclient import TestClient

    from apps.api.main import create_app
    from scripts.run_demo import readiness_urls

    prepared = prepare_demo(Path.cwd(), tmp_path, environ={})
    api_url, ui_url = readiness_urls(api_port=8000, ui_port=8501)
    with TestClient(create_app(prepared.settings), client=("127.0.0.1", 1234)) as client:
        assert client.get(api_url).status_code == 200
    assert ui_url == "http://127.0.0.1:8501/_stcore/health"


def test_explicit_demo_env_file_aliases_and_process_precedence(tmp_path):
    (tmp_path / ".env.demo").write_text(
        "MOONSHOT_API_KEY=fixture-file-model\nTAVILY_API_KEY=fixture-file-search\nMODEL_ID=kimi-k2.6\n",
        encoding="utf-8",
    )
    values = demo_environment(tmp_path, {"MODEL_API_KEY": "fixture-process-model"})
    assert values["MODEL_API_KEY"] == "fixture-process-model"
    assert values["SEARCH_API_KEY"] == "fixture-file-search"


def test_demo_generated_state_contains_no_provider_secret_and_retains_owner_key(tmp_path):
    env = {
        "MODEL_API_KEY": "fixture-test-secret-model",
        "SEARCH_API_KEY": "fixture-test-secret-search",
    }
    first = prepare_demo(Path.cwd(), tmp_path, environ=env)
    again = prepare_demo(Path.cwd(), tmp_path, environ=env)
    assert first.settings.session_signing_key == again.settings.session_signing_key
    payload = (tmp_path / "profiles.json").read_text(encoding="utf-8")
    assert "fixture-test-secret" not in payload
    assert "fixture-test-secret" not in repr(first)
    profile = json.loads(payload)["profiles"]["live-default"]
    assert profile["execution_mode"] == "live"
    assert {r["provider_id"] for r in profile["routes"]} == {
        "kimi-instant",
        "tavily",
        "httpx-fetcher",
        "baseline-parser-router",
        "lexical-hash",
    }


def test_launcher_keeps_both_servers_on_loopback_and_credentials_out_of_ui():
    from scripts.run_demo import demo_processes

    env = {
        "MODEL_API_KEY": "fixture-model",
        "MOONSHOT_API_KEY": "fixture-alias",
        "SEARCH_API_KEY": "fixture-search",
        "SESSION_SIGNING_KEY": "fixture-signing",
        "PATH": "unchanged",
    }
    api, ui = demo_processes(Path.cwd(), python="python", api_port=8000, ui_port=8501, environ=env)
    assert "--no-proxy-headers" in api.command
    assert "127.0.0.1" in api.command
    assert "--server.address=127.0.0.1" in ui.command
    assert api.environment["MODEL_API_KEY"] == "fixture-model"
    assert ui.environment["DEEPRESEARCH_API_URL"] == "http://127.0.0.1:8000"
    assert ui.environment["PATH"] == "unchanged"
    assert not any("fixture" in value for value in ui.environment.values())
    assert env["MODEL_API_KEY"] == "fixture-model"
