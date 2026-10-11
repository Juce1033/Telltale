"""
Telltale dashboard + control panel. Run it from src/ (telltale.py does this for you).
No internet, no extra packages -- just Python's standard library.

    python ../frontend/dashboard.py     # then open http://localhost:8080

From the page you can:
  Run    -- start the detector live on the ESP32 or on any recording, and stop it
  Record -- record a labelled recording with logger.py (healthy, a fault, knocks, baseline)
  Train  -- retrain the detector on everything in data/
Reads:  state.json (detector, 4x/sec), agent.json (Nemotron's diagnosis)
Writes: motor_cmd.txt (motor buttons; detect_live.py forwards it to the ESP32)

Fault drills: press "Mark fault created" (or Space) the moment you create a fault;
the dashboard times how long the detector takes to flag it. Press "Mark fault cleared"
(or C) once the fault is undone -- a drill the detector never caught is recorded as
missed. Pressing the button again while timing (or Esc) cancels a drill started by
mistake. A FAULT with no drill running counts as a false alarm.
"""
import argparse
import collections
import glob
import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE, AGENT, CMD = Path("state.json"), Path("agent.json"), Path("motor_cmd.txt")
PAGE = Path(__file__).with_name("dashboard.html")

PY = sys.executable
MOTOR = 70                                       # normal running speed, set by --motor
PROBE_SPEED = 45                                 # speed the agent's active test uses

lock = threading.Lock()
history = collections.deque(maxlen=240)          # 60 s at 4 Hz
board = {"trials": [], "false_alarms": 0}
pending = None                                   # the injection currently being timed
last_status = "OK"


_last_good = {}


def read_json(path):
    """Read a JSON file; if it's mid-update, fall back to the last good copy."""
    try:
        data = json.loads(path.read_text())
        _last_good[path] = data
        return data
    except FileNotFoundError:
        _last_good.pop(path, None)
        return None
    except json.JSONDecodeError:
        return _last_good.get(path)


def poller():
    """Sample state.json 4x/sec into history and update the scoreboard."""
    global last_status, pending
    last_t = None
    while True:
        s = read_json(STATE)
        if s and s.get("time") != last_t:
            last_t = s["time"]
            with lock:
                history.append({"t": s["time"], "cur": s["current_mean_a"],
                                "vib": s["vibration_rms"], "score": s["anomaly_score"],
                                "status": s["status"]})
                # a new fault = healthy (or a knock) turning into a fault; returning from an
                # active test with the fault still present is the same fault, not a new one
                if s["status"] == "FAULT" and last_status in ("OK", "NOTICE"):
                    if pending and pending["detect_s"] is None:
                        pending["detect_s"] = round(time.time() - pending["t"], 2)
                        pending["fault"] = s["fault"]
                        pending = None                         # caught: this drill is done
                    elif not pending:
                        board["false_alarms"] += 1
                last_status = s["status"]
        time.sleep(0.1)


# ---------------- control panel: the detector and one-off tasks ----------------
class Proc:
    """A child program whose last lines of output the page can show."""
    def __init__(self, name, args, desc, seconds=None):
        self.name, self.desc, self.seconds = name, desc, seconds
        self.log = collections.deque(maxlen=40)
        self.started = time.time()
        self.stopped = False                     # True when stopped on purpose (Stop / Cancel)
        self.p = subprocess.Popen([PY, "-u", *args], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.log.append(line.rstrip())

    def running(self):
        return self.p.poll() is None

    def stop(self):
        if self.running():
            self.stopped = True
            self.p.terminate()                   # detect_live / logger stop the motor on exit
            try:
                self.p.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.p.kill()

    def info(self):
        return {"name": self.name, "desc": self.desc, "running": self.running(),
                "exit": self.p.poll(), "stopped": self.stopped, "elapsed": round(time.time() - self.started, 1),
                "seconds": self.seconds, "log": list(self.log)[-25:]}


ctl = {"detect": None, "task": None}
ctl_lock = threading.Lock()


def esp32_ports():
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def recordings():
    files = sorted(glob.glob("data/*.csv")) + sorted(glob.glob("tests/*.csv"))
    return files


def busy_task():
    t = ctl["task"]
    return t if t and t.running() else None


def start_detect(mode, file=None):
    if busy_task():
        return "Wait for the current recording or training to finish."
    if ctl["detect"]:
        ctl["detect"].stop()
    if mode == "live":
        ports = esp32_ports()
        if not ports:
            return "No ESP32 found. Plug it into the ASUS and try again."
        args, desc = ["--port", ports[0]], f"Live on {ports[0]}"
    else:
        if file not in recordings():
            return "That recording no longer exists."
        args, desc = ["--replay", file], f"Replaying {Path(file).name}"
    if not Path("model.pt").exists():
        return "No trained model yet. Record some data and press Train first."
    AGENT.unlink(missing_ok=True)                # a new run starts with a clean agent panel
    ctl["detect"] = Proc("detect", ["detect_live.py", "--motor", str(MOTOR), *args], desc)
    return None


def start_task(kind, body):
    if busy_task():
        return "Something is already recording or training."
    if ctl["detect"]:
        ctl["detect"].stop()                     # the serial port / model file must be free
    if kind == "train":
        ctl["task"] = Proc("train", ["train.py"], "Training the detector")
        return None
    ports = esp32_ports()
    if not ports:
        return "No ESP32 found. Plug it into the ASUS and try again."
    label = body.get("label", "normal")
    secs = max(10, min(900, int(body.get("seconds", 120))))
    args = ["logger.py", "--port", ports[0], "--seconds", str(secs)]
    if label == "baseline":
        args += ["--label", "normal", "--motor", str(PROBE_SPEED), "--outdir", "data/baselines"]
        desc = f"Recording healthy at {PROBE_SPEED}% (active-test baseline)"
    else:
        args += ["--label", label, "--motor", str(MOTOR)]
        desc = f"Recording {label.replace('_', ' ')}"
    ctl["task"] = Proc("record", args, desc, seconds=secs)
    return None


def control_info():
    d, t = ctl["detect"], ctl["task"]
    return {"detect": d.info() if d else None, "task": t.info() if t else None,
            "ports": esp32_ports(), "recordings": recordings(),
            "motor": MOTOR, "probe_speed": PROBE_SPEED}


def summary():
    trials = board["trials"]
    detected = [t for t in trials if t["detect_s"] is not None]
    missed = [t for t in trials if t.get("missed")]
    times = [t["detect_s"] for t in detected]
    return {
        "injected": len(trials),
        "detected": len(detected),
        "missed": len(missed),
        "rate": round(len(detected) / len(trials), 3) if trials else None,
        "avg_detect_s": round(sum(times) / len(times), 2) if times else None,
        "last_detect_s": times[-1] if times else None,
        "false_alarms": board["false_alarms"],
        "timing": bool(pending),
        "trials": trials[-8:],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                   # keep the terminal quiet
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api":
            s = read_json(STATE)
            with lock:
                return self.send(200, {
                    "now": time.time(),
                    "state": s,
                    "stale": (s is None) or (time.time() - s["time"] > 2),
                    "agent": read_json(AGENT),
                    "history": list(history),
                    "score": summary(),
                    "control": control_info(),
                })
        self.send(404, {"error": "not found"})

    def do_POST(self):
        global pending
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path in ("/run", "/stop", "/record", "/train", "/cancel"):
            with ctl_lock:                       # separate lock: stopping a program can take seconds
                err = None
                if self.path == "/run":
                    err = start_detect(body.get("mode"), body.get("file"))
                elif self.path == "/stop" and ctl["detect"]:
                    ctl["detect"].stop()
                elif self.path == "/record":
                    err = start_task("record", body)
                elif self.path == "/train":
                    err = start_task("train", body)
                elif self.path == "/cancel" and busy_task():
                    ctl["task"].stop()
                return self.send(409 if err else 200, {"error": err, "control": control_info()})
        with lock:
            if self.path == "/mark":
                kind = body.get("type")
                if kind == "inject" and not pending:
                    pending = {"t": time.time(), "detect_s": None, "fault": None, "missed": False}
                    board["trials"].append(pending)
                    s = read_json(STATE)
                    if s and s.get("status") == "FAULT":      # already flagged before the mark
                        pending["detect_s"], pending["fault"] = 0.0, s["fault"]
                        pending = None
                elif kind == "cancel" and pending:
                    if pending in board["trials"]:
                        board["trials"].remove(pending)       # started by mistake: drop it
                    pending = None
                elif kind == "clear" and pending:
                    if pending["detect_s"] is None:
                        pending["missed"] = True              # fault was undone, never caught
                    pending = None
            elif self.path == "/reset":
                board["trials"].clear()
                board["false_alarms"] = 0
                pending = None
            elif self.path == "/motor":
                pct = max(0, min(100, int(body.get("percent", 0))))
                CMD.write_text(f"S{pct}")
            else:
                return self.send(404, {"error": "not found"})
            return self.send(200, summary())


def shutdown(*_):
    for k in ("task", "detect"):
        if ctl[k]:
            ctl[k].stop()
    sys.exit(0)


def main():
    global MOTOR, PROBE_SPEED
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--motor", type=int, default=70, help="normal motor speed %%")
    p.add_argument("--probe-speed", type=int, default=45, help="active-test speed %%")
    args = p.parse_args()
    MOTOR, PROBE_SPEED = args.motor, args.probe_speed
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, shutdown)
    threading.Thread(target=poller, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)   # bind first, then announce
    print(f"dashboard on http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
