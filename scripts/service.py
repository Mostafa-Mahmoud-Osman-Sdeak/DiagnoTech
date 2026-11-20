#!/usr/bin/env python3
"""Local service control: start / stop / restart / status.

The server runs as a detached background process with a PID file and a
log file, survives terminal closure, refuses duplicate instances, and
can be supervised by the OS scheduler/systemd using the same commands.
"""
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PID_FILE = PROJECT_ROOT / "logs" / "diagnotech.pid"
LOG_FILE = PROJECT_ROOT / "logs" / "server.log"


def _pid_alive(pid: int) -> bool:
    """Cross-platform process-liveness check.

    ``os.kill(pid, 0)`` is the POSIX idiom, but on Windows it raises
    ``OSError [WinError 11]`` instead of ProcessLookupError, so Windows
    uses OpenProcess instead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
        except ValueError:
            PID_FILE.unlink(missing_ok=True)
            return None
        if _pid_alive(pid):
            return pid
        PID_FILE.unlink(missing_ok=True)
    return None


def start() -> None:
    if _read_pid():
        print("already running")
        return
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    candidates = [PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
                  PROJECT_ROOT / ".venv" / "bin" / "python"]
    python = next((p for p in candidates if p.exists()), None)
    executable = str(python) if python is not None else sys.executable
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"))
    with LOG_FILE.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [executable, "-m", "diagnotech.main"],
            cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=log,
            start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    print(f"started (pid {proc.pid}); log: {LOG_FILE}")


def stop() -> None:
    pid = _read_pid()
    if not pid:
        print("not running")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    PID_FILE.unlink(missing_ok=True)
    print("stopped")


def status() -> None:
    pid = _read_pid()
    print(f"process: {'running (pid ' + str(pid) + ')' if pid else 'not running'}")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=3) as resp:
            print("health:", resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        print("health: unreachable", exc)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"start": start, "stop": stop, "status": status,
     "restart": lambda: (stop(), start())}[command]()
