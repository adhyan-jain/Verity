"""
Shared process-orchestration helpers for scripts/dev_up.py and scripts/smoke_test.py.
Not a Verity component — internal tooling only.
"""

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, uvicorn import target, port, health path)
SERVICES = [
    ("agent", "agent.api:app", 8000, "/health"),
    ("fraud", "engines.fraud.api:app", 8001, "/health"),
    ("ledger", "engines.ledger.api:app", 8002, "/api/v1/ledger/health"),
    ("typology", "engines.typology.api:app", 8003, "/api/v1/typology/health"),
]


def load_env_file(path: str = os.path.join(ROOT, ".env")) -> None:
    """Loads KEY=VALUE pairs from `path` into os.environ, without overriding
    variables the caller's shell already set (shell env always wins)."""
    if not os.path.exists(path):
        print(f"[env] {path} not found; copy .env.example to .env first.", file=sys.stderr)
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def ensure_fraud_model_trained() -> None:
    """Fraud engine's /health and /explain routes hard-fail without a trained
    artifact. model.pkl is intentionally gitignored (binary), so a fresh
    checkout must train it once before the fraud service can start."""
    artifact_path = os.path.join(ROOT, "engines", "fraud", "model.pkl")
    if os.path.exists(artifact_path):
        return
    print("[setup] engines/fraud/model.pkl missing — training fraud model (~2 min)...")
    subprocess.run([sys.executable, "-m", "engines.fraud.train"], cwd=ROOT, check=True)


def start_service(target: str, port: int, extra_env: dict | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", target, "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_healthy(port: int, health_path: str, timeout: float = 60.0, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}{health_path}"
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def start_all_backend_services() -> dict[str, subprocess.Popen]:
    """Starts agent+fraud+ledger+typology, waits for all four to report
    healthy, and returns the running processes keyed by service name. Raises
    RuntimeError (after killing whatever did start) if any fails to come up."""
    procs: dict[str, subprocess.Popen] = {}
    for name, target, port, _ in SERVICES:
        print(f"[start] {name} -> http://127.0.0.1:{port}", flush=True)
        procs[name] = start_service(target, port)

    for name, _, port, health_path in SERVICES:
        proc = procs[name]
        if not wait_healthy(port, health_path, proc=proc):
            output = ""
            try:
                if proc.stdout:
                    output = proc.stdout.read()
            except Exception:
                pass
            stop_all(procs)
            err_msg = f"{name} service did not become healthy on port {port} within timeout."
            if output:
                err_msg += f"\n--- {name} output ---\n{output.strip()}\n--- end output ---"
            raise RuntimeError(err_msg)
        print(f"[ready] {name} healthy", flush=True)

    return procs


def stop_all(procs: dict[str, subprocess.Popen]) -> None:
    for name, proc in procs.items():
        if proc.poll() is None:
            if os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
    for name, proc in procs.items():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
