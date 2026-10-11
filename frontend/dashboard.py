"""
Live dashboard server. Runs on the GX10 alongside detect_live.py and agent.py.
No internet, no extra packages -- just Python's standard library.

    python dashboard.py                 # then open http://localhost:8080
                                        # or http://<GX10-IP>:8080 from a laptop

Reads:  state.json (detector, 4x/sec), agent.json (Nemotron's diagnosis)
Writes: motor_cmd.txt (Start/Stop buttons; detect_live.py forwards it to the ESP32)

Fault drills: press "Mark fault created" (or Space) the moment you create a fault;
the dashboard times how long the detector takes to flag it. Press "Mark fault cleared"
(or C) once the fault is undone -- a drill the detector never caught is recorded as
missed. Pressing the button again while timing (or Esc) cancels a drill started by
mistake. A FAULT with no drill running counts as a false alarm.
"""
import argparse
import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE, AGENT, CMD = Path("state.json"), Path("agent.json"), Path("motor_cmd.txt")
PAGE = Path(__file__).with_name("dashboard.html")

lock = threading.Lock()
history = collections.deque(maxlen=240)          # 60 s at 4 Hz
board = {"trials": [], "false_alarms": 0}
pending = None                                   # the injection currently being timed
last_status = "OK"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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
                })
        self.send(404, {"error": "not found"})

    def do_POST(self):
        global pending
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    threading.Thread(target=poller, daemon=True).start()
    print(f"dashboard on http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
