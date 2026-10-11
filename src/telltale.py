"""
Starts the whole Telltale system with one command, and stops it all with Ctrl+C.

    python telltale.py                    # live: finds the ESP32, starts everything
    python telltale.py --replay data/loose_fastener_m70_214501.csv   # no hardware
    python telltale.py --retrain          # retrain on data/ first, then start

Order: dashboard -> agent (waits until Nemotron is loaded) -> detector.
Ctrl+C (or closing the window) stops all three and the motor.
Run it from src/, with the venv active. Dashboard: http://localhost:8080
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import threading
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="run on a recorded CSV instead of the ESP32")
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--motor", type=int, default=70, help="normal motor speed %% (match your recordings)")
    ap.add_argument("--retrain", action="store_true", help="retrain on data/ before starting")
    ap.add_argument("--no-agent", action="store_true", help="skip Nemotron")
    args = ap.parse_args()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_all)

    if args.retrain or not (HERE / "model.pt").exists():
        print("training the detector...", flush=True)
        if subprocess.call([PY, "train.py"], cwd=HERE) != 0:
            sys.exit("training failed; see the output above.")

    if args.replay and not Path(args.replay).exists() and not (HERE / args.replay).exists():
        sys.exit(f"No such recording: {args.replay}")
    port = None if args.replay else (args.port or find_port())

    start("dashboard", [str(find_dashboard())], ready_text="dashboard on")
    print("dashboard ready: http://localhost:8080", flush=True)

    if not args.no_agent:
        if not ollama_up():
            print("warning: Ollama isn't answering on port 11434. Try: ollama serve", flush=True)
        print("loading Nemotron (first time can take a minute)...", flush=True)
        start("agent", ["agent.py", "--base-speed", str(args.motor),
                        *(["--probe", "off"] if args.replay else [])],
              ready_text="agent watching")

    detect = ["detect_live.py", "--motor", str(args.motor)]
    detect += ["--replay", args.replay] if args.replay else ["--port", port]
    start("detect", detect, quiet_repeats=True)
    print(f"running ({'replay: ' + args.replay if args.replay else 'live on ' + port}). "
          "Ctrl+C stops everything, including the motor.", flush=True)

    while True:                                     # if any part crashes, shut the rest down
        time.sleep(1)
        for name, p in list(procs):
            if p.poll() is None:
                continue
            if name == "detect" and args.replay and p.returncode == 0:
                print("replay finished. The dashboard stays up; Ctrl+C to stop.", flush=True)
                procs.remove((name, p))
                continue
            print(f"{name} exited (code {p.returncode}).", flush=True)
            stop_all()


if __name__ == "__main__":
    main()
