"""
Starts Telltale: the dashboard (which is also the control panel) and Nemotron.
Everything else -- running live or on a recording, recording data, training --
is done from the page at http://localhost:8080.

    python telltale.py                    # then pick a source on the page
    python telltale.py --live             # also start the detector on the ESP32 right away
    python telltale.py --replay data/normal_205808.csv

Ctrl+C (or closing the window) stops everything, including the motor.
Run it from src/, with the venv active.
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
procs = []


def find_dashboard():
    for p in (HERE / "dashboard.py", HERE.parent / "frontend" / "dashboard.py"):
        if p.exists():
            return p
    sys.exit("Can't find dashboard.py (looked in src/ and ../frontend/).")


def find_port():
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        sys.exit("No ESP32 found (no /dev/ttyUSB* or /dev/ttyACM*). Check the USB cable, "
                 "or use --replay to run on a recording.")
    return ports[0]


def ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def start(name, args, ready_text=None, quiet_repeats=False):
    """Launch a program, prefix its output with [name], and optionally wait until it prints ready_text."""
    # own session: Ctrl+C reaches only this launcher, which then stops each part cleanly
    p = subprocess.Popen([PY, "-u", *args], cwd=HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    procs.append((name, p))
    ready = threading.Event()

    def pump():
        last = None
        for line in p.stdout:
            line = line.rstrip()
            if ready_text and ready_text in line:
                ready.set()
            if quiet_repeats:                       # detector prints 4x/sec: show changes only
                key = " ".join(line.split()[:2])
                if key == last and not line.startswith(">>>"):
                    continue
                last = key
            print(f"[{name}] {line}", flush=True)
        ready.set()                                 # process ended: don't wait forever
    threading.Thread(target=pump, daemon=True).start()
    if ready_text:
        if not ready.wait(timeout=180) or p.poll() is not None:
            raise SystemExit(f"{name} didn't start. Its output is above.")
    return p


def stop_all(*_):
    print("\nstopping everything...", flush=True)
    for name, p in reversed(procs):                 # detector first, so it stops the motor
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    for f in ("probe.json", "motor_cmd.txt"):
        (HERE / f).unlink(missing_ok=True)
    print("all stopped.", flush=True)
    os._exit(0)


def post(path, body):
    req = urllib.request.Request("http://localhost:8080" + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="start straight away on this recording")
    ap.add_argument("--live", action="store_true", help="start straight away on the ESP32")
    ap.add_argument("--motor", type=int, default=70, help="normal motor speed %% (match your recordings)")
    ap.add_argument("--probe-speed", type=int, default=45, help="active-test speed %%")
    ap.add_argument("--no-agent", action="store_true", help="skip Nemotron")
    args = ap.parse_args()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_all)

    # The dashboard is the control panel: it starts and stops the detector, records and trains.
    start("dashboard", [str(find_dashboard()), "--motor", str(args.motor),
                        "--probe-speed", str(args.probe_speed)], ready_text="dashboard on")
    print("control panel ready: http://localhost:8080", flush=True)

    if not args.no_agent:
        if not ollama_up():
            print("warning: Ollama isn't answering on port 11434. Try: ollama serve", flush=True)
        print("loading Nemotron (first time can take a minute)...", flush=True)
        start("agent", ["agent.py", "--base-speed", str(args.motor),
                        "--probe-speed", str(args.probe_speed)], ready_text="agent watching")

    if args.replay or args.live:
        r = post("/run", {"mode": "live" if args.live else "replay", "file": args.replay})
        print(r.get("error") or "detector started.", flush=True)
    print("ready. Use the page to run, record and train. Ctrl+C here stops everything.", flush=True)

    while True:                                     # if a core part crashes, shut the rest down
        time.sleep(1)
        for name, p in list(procs):
            if p.poll() is not None:
                print(f"{name} exited (code {p.returncode}).", flush=True)
                stop_all()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:                       # a startup step failed: don't leave parts running
        if procs:
            print(e, flush=True)
            stop_all()
        raise
