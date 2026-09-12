"""Start the local demo: python -m scripts.run_demo [--check]."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from apps.api.demo import prepare_demo

_SECRET_NAMES = {
    "MODEL_API_KEY",
    "SEARCH_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "TAVILY_API_KEY",
    "SESSION_SIGNING_KEY",
    "DATABASE_URL",
}


@dataclass(frozen=True)
class DemoProcess:
    command: tuple[str, ...]
    environment: dict[str, str] = field(repr=False)


def demo_processes(
    repository: Path,
    *,
    python: str,
    api_port: int,
    ui_port: int,
    environ: Mapping[str, str],
) -> tuple[DemoProcess, DemoProcess]:
    api_environment = dict(environ)
    api_environment["PYTHONUTF8"] = "1"
    ui_environment = {name: value for name, value in environ.items() if name not in _SECRET_NAMES}
    ui_environment["PYTHONUTF8"] = "1"
    ui_environment["DEEPRESEARCH_API_URL"] = f"http://127.0.0.1:{api_port}"
    return (
        DemoProcess(
            (
                python,
                "-m",
                "uvicorn",
                "apps.api.demo:create_demo_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--no-proxy-headers",
            ),
            api_environment,
        ),
        DemoProcess(
            (
                python,
                "-m",
                "streamlit",
                "run",
                str(repository / "apps/ui/app.py"),
                "--server.headless=true",
                "--server.address=127.0.0.1",
                f"--server.port={ui_port}",
                "--browser.gatherUsageStats=false",
            ),
            ui_environment,
        ),
    )


def _port(value: str) -> int:
    parsed = int(value)
    if not 1024 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("use a port from 1024 through 65535")
    return parsed


def readiness_urls(*, api_port: int, ui_port: int) -> tuple[str, str]:
    return (
        f"http://127.0.0.1:{api_port}/health/ready",
        f"http://127.0.0.1:{ui_port}/_stcore/health",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check configuration without paid calls"
    )
    parser.add_argument("--api-port", type=_port, default=8000)
    parser.add_argument("--ui-port", type=_port, default=8501)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    state = Path(os.environ.get("DEEPRESEARCH_DEMO_STATE", str(repository / "artifacts/demo")))
    try:
        prepared = prepare_demo(repository, state, environ=os.environ)
    except (OSError, ValueError):
        print(
            "Demo configuration invalid. Check .env.demo and the deployment paths; no secrets printed."
        )
        return 2
    configured = all(
        prepared.environment.get(name, "").strip() for name in ("MODEL_API_KEY", "SEARCH_API_KEY")
    )
    if args.check:
        print(
            json.dumps(
                {
                    "offline": "configured",
                    "live": "configured" if configured else "missing_credentials",
                    "model_key_configured": bool(
                        prepared.environment.get("MODEL_API_KEY", "").strip()
                    ),
                    "search_key_configured": bool(
                        prepared.environment.get("SEARCH_API_KEY", "").strip()
                    ),
                    "paid_api_verified": False,
                }
            )
        )
        return 0
    if args.api_port == args.ui_port:
        print("API and UI need different ports.")
        return 2
    for port in (args.api_port, args.ui_port):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                print(f"Port {port} is in use. Stop your previous demo or choose another port.")
                return 2
    specs = demo_processes(
        repository,
        python=sys.executable,
        api_port=args.api_port,
        ui_port=args.ui_port,
        environ=os.environ,
    )
    children: list[subprocess.Popen[bytes]] = []
    try:
        for spec in specs:
            children.append(
                subprocess.Popen(
                    spec.command,
                    cwd=repository,
                    env=spec.environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            )
        opener = build_opener(ProxyHandler({}))
        for url in readiness_urls(api_port=args.api_port, ui_port=args.ui_port):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if any(child.poll() is not None for child in children):
                    print("A demo process exited. Inspect the startup message above.")
                    return 1
                try:
                    with opener.open(url, timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, URLError):
                    time.sleep(0.25)
            else:
                print("Demo readiness timed out; child processes will be stopped.")
                return 1
        print(
            f"Demo: http://127.0.0.1:{args.ui_port} | API docs: http://127.0.0.1:{args.api_port}/docs",
            flush=True,
        )
        print(
            "Live API configured (may incur charges)."
            if configured
            else "Offline ready. For live API set MODEL_API_KEY and SEARCH_API_KEY in .env.demo, then restart.",
            flush=True,
        )
        print("Keep this terminal running. Ctrl+C stops only these demo processes.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
