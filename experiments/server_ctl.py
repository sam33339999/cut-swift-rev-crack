"""Stop the local vLLM on port 8000 and start the experiment server."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = ROOT / "experiments" / "001-baseline-and-mine" / "start_server.sh"
MODELS_URL = "http://127.0.0.1:8000/v1/models"
EXPECTED_MODEL = "Ornith-1.5-9B"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def listener_pids(port: int = 8000) -> set[int]:
    try:
        out = subprocess.check_output(["ss", "-lntp"], text=True)
    except subprocess.CalledProcessError:
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        if f":{port}" not in line:
            continue
        for match in re.findall(r"pid=(\d+)", line):
            pids.add(int(match))
    return pids


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    line = out.strip().splitlines()[0]
    return int(line)


def stop_listeners(port: int = 8000, wait_s: float = 40) -> list[int]:
    """SIGTERM the process groups listening on port, then SIGKILL if needed."""
    pids = listener_pids(port)
    if not pids:
        return []
    my_pgid = os.getpgrp()
    pgids = set()
    for pid in pids:
        try:
            pgids.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    for pgid in pgids:
        if pgid in (0, 1, my_pgid):
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        else:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + wait_s
    while time.time() < deadline and listener_pids(port):
        time.sleep(1)
    if listener_pids(port):
        for pgid in pgids:
            if pgid in (0, 1, my_pgid):
                continue
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(2)
    still = listener_pids(port)
    if still:
        raise RuntimeError(f"port {port} still held by {sorted(still)}")
    # The engine can release the port before the GPU allocation is gone.
    gpu_deadline = time.time() + 30
    while time.time() < gpu_deadline:
        used = gpu_used_mib()
        if used is None or used < 2000:
            break
        time.sleep(1)
    return sorted(pids)


def fetch_models(timeout: float = 5) -> tuple[int, str]:
    request = urllib.request.Request(MODELS_URL)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return response.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def assert_model_body(body: str) -> dict:
    payload = json.loads(body)
    names = [item.get("id") for item in payload.get("data", [])]
    if EXPECTED_MODEL not in names:
        raise RuntimeError(f"models endpoint did not list {EXPECTED_MODEL}: {names}")
    return payload


def start_server(log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    log_handle.write(f"\n--- start {now()} ---\n".encode())
    log_handle.flush()
    return subprocess.Popen(
        ["bash", str(START_SCRIPT)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd="/content",
    )


def reload_server(log_path: Path, boot_timeout_s: float = 360) -> tuple[str, str, int]:
    """Stop whatever is on port 8000, start the experiment server, return the models body.

    The returned body is the raw HTTP response. This function does not sample.
    """
    stopped = stop_listeners()
    proc = start_server(log_path)
    deadline = time.time() + boot_timeout_s
    last_error = "no response yet"
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"vLLM exited with code {proc.returncode} before it was ready.\n{tail}"
            )
        try:
            status, body = fetch_models()
        except Exception as exc:  # noqa: BLE001 — boot is still in progress
            last_error = str(exc)
            time.sleep(3)
            continue
        if status == 200:
            assert_model_body(body)
            return body, now(), proc.pid
        last_error = f"HTTP {status}: {body[:400]}"
        time.sleep(3)
    tail = ""
    if log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    raise RuntimeError(
        f"vLLM did not become ready in {boot_timeout_s}s after stopping {stopped}. "
        f"Last error: {last_error}\n{tail}"
    )
