"""
The Nemotron agent. Runs alongside detect_live.py.

It watches state.json. When the detector declares a FAULT, the agent:
  1. Safety: if current is dangerously high (a stall), stops the motor at once.
  2. Active test (live mode): drops the motor to a probe speed, measures, returns to
     normal speed and measures again. A fault that vanishes on retest was transient
     (a knock); one that persists is real, and its response to speed is evidence.
  3. Asks Nemotron to weigh the detector's guess against the test results and the
     sensor trends, and writes a plain-English diagnosis to agent.json.
Motor commands go through motor_cmd.txt, which detect_live.py forwards to the ESP32.

    python agent.py                       # uses nemotron3:33b via Ollama
    python agent.py --probe off           # skip the active test
    python agent.py --model nemotron-3-super

For the active test to compare against healthy running at the probe speed, record a
short baseline once:  python logger.py --label normal --motor 45 --seconds 120 --outdir data/baselines
"""
import argparse
import collections
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OLLAMA = "http://localhost:11434/api/chat"
STATE, OUT, CMD = Path("state.json"), Path("agent.json"), Path("motor_cmd.txt")
PROBE = Path("probe.json")
SAFETY_STOP_A = 0.6        # stop immediately above this current, no questions asked

SYSTEM = """You are the maintenance agent for a small test machine: a 12V gearmotor drives a
GT2 belt to a pulley on a shaft held by two pillow-block bearings. You only observe it from
outside: motor current (amps) and vibration (accelerometer + gyro, bolted near the output bearing).
A trained detector model flags anomalies and guesses a fault class.

Known faults and their physical signatures:
- loose_fastener: a bearing-block screw backed off. Vibration RMS rises (rattling, impacts once
  per revolution); current roughly unchanged. Risk: bearing misalignment, wear. Fix: retighten.
- belt_slip: belt tension lost. Current DROPS (the motor is less loaded); vibration irregular.
  Risk: output stops being driven, belt wear/heat. Fix: re-tension belt via slotted motor mount.
- stall: something is braking the output. Current RISES sharply, motor slows. Risk: motor
  overheating, driver stress. Normal running is 0.15-0.6 A; the sensor saturates near 0.8 A.

Before you are called, the system may have run an ACTIVE TEST on the machine: it dropped the
motor to a lower probe speed, measured, returned to normal speed, and measured again.
How to read it:
- persisted_after_retest = false: the anomaly vanished on retest. It was transient (most likely
  an external knock or bump to the bench), NOT a machine fault. Say so, with low concern.
- persisted_after_retest = true: the fault is real.
- At the probe speed, compare to the healthy probe-speed baseline when one is given:
  vibration well above healthy at both speeds supports loose_fastener (a rattle is present at
  any speed); current below healthy supports belt_slip; current above healthy supports stall.
- safety_stop = true: current was dangerous, so the motor was stopped before any test.

Your job when a fault is flagged:
1. Read the active test results if present. Call tools if you need more evidence.
2. Decide whether the evidence actually supports the detector's guess. Say so if it doesn't.
3. If it is a stall with current above 0.6 A and the motor is still running, stop it. For other
   faults, do not stop the motor unless the evidence shows something dangerous.
4. Reply in exactly this format, short, for an operator glancing at a screen:
FAULT: <name> (<your confidence: high/medium/low>)
EVIDENCE: <one sentence citing the actual numbers>
ACTION: <what the operator should physically do, and anything you did>"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_detector_output",
        "description": "Latest detector verdict: status, fault guess, confidence, anomaly score "
                       "(1.0 = threshold), class probabilities, current and vibration now.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_sensor_trend",
        "description": "Compare current and vibration over the last N seconds to the healthy "
                       "baseline recorded before the fault started.",
        "parameters": {"type": "object", "properties": {
            "seconds": {"type": "number", "description": "how far back to look, e.g. 5"}},
            "required": ["seconds"]}}},
    {"type": "function", "function": {
        "name": "stop_motor",
        "description": "Immediately stop the motor. Use for stalls or anything dangerous.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "set_motor_speed",
        "description": "Set motor speed in percent (0-100) to reduce stress without stopping.",
        "parameters": {"type": "object", "properties": {
            "percent": {"type": "number"}, "reason": {"type": "string"}},
            "required": ["percent", "reason"]}}},
]

history = collections.deque(maxlen=480)   # last ~2 minutes of detector states
actions = []


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def tool_get_detector_output():
    return history[-1] if history else {"error": "no detector data yet"}


def tool_get_sensor_trend(seconds=5):
    now = time.time()
    recent = [s for s in history if now - s["time"] <= float(seconds)]
    healthy = [s for s in history if s["status"] == "OK"]
    return {
        "window_s": seconds,
        "current_now_a": mean([s["current_mean_a"] for s in recent]),
        "current_baseline_a": mean([s["current_mean_a"] for s in healthy]),
        "vibration_rms_now": mean([s["vibration_rms"] for s in recent]),
        "vibration_rms_baseline": mean([s["vibration_rms"] for s in healthy]),
        "max_anomaly_score": max((s["anomaly_score"] for s in recent), default=None),
        "seconds_in_fault": recent[-1]["fault_active_s"] if recent else 0,
    }


def send_motor(cmd, reason, log=True):
    CMD.write_text(cmd)
    if log:
        actions.append(f"{cmd} ({reason})")
    print(f"  [agent action] {cmd}: {reason}")
    return {"ok": True, "sent": cmd}


# ---------- active test ----------
def window_stats(states):
    if not states:
        return None
    return {"current_a": mean([s["current_mean_a"] for s in states]),
            "vibration_rms": mean([s["vibration_rms"] for s in states]),
            "anomaly_score": mean([s["anomaly_score"] for s in states])}


def collect(seconds):
    """Wait, then return the detector states that arrived during those seconds."""
    t0 = time.time()
    time.sleep(seconds)
    return [s for s in list(history) if s["time"] >= t0]


def load_baseline(speed, folder="data/baselines"):
    """Healthy current and vibration at a given motor speed, from a recording made with
    logger.py --motor <speed> --outdir data/baselines. Same math as detect_live.py."""
    files = sorted(Path(folder).glob(f"*_m{speed}_*.csv"))
    for f in reversed(files):                    # newest usable recording wins
        try:
            df = pd.read_csv(f)
            acc = df[["accel_x", "accel_y", "accel_z"]].to_numpy(float)
            cur = df["current_a"].to_numpy(float)
            if len(df) >= 500:
                break
        except Exception:
            pass
        print(f"skipping unusable baseline recording {f.name}")
    else:
        return None
    n = len(df) // 500
    vib = [float(np.sqrt(((acc[i*500:(i+1)*500] - acc[i*500:(i+1)*500].mean(0)) ** 2).mean()))
           for i in range(n)]
    return {"current_a": round(float(cur.mean()), 4), "vibration_rms": round(float(np.mean(vib)), 4),
            "file": f.name}


def ratio(a, b):
    return round(a / b, 2) if a is not None and b else None


def run_active_test(trigger, base_speed, probe_speed, baseline_low):
    """Probe the machine at a lower speed, then retest at normal speed."""
    before = window_stats([s for s in list(history)[-8:]])          # last ~2 s before the test
    if before and before["current_a"] and before["current_a"] > SAFETY_STOP_A:
        send_motor("S0", f"current {before['current_a']:.2f} A above {SAFETY_STOP_A} A")
        return {"safety_stop": True, "before": before}

    PROBE.write_text(json.dumps({"speed": probe_speed, "suspect": trigger["fault"], "started": time.time()}))
    try:
        send_motor(f"S{probe_speed}", "active test", log=False)
        time.sleep(2.5)                                              # settle + refill the window
        low = window_stats(collect(2.0))
        send_motor(f"S{base_speed}", "active test done", log=False)
        time.sleep(2.5)
        back = window_stats(collect(2.0))
    finally:
        PROBE.unlink(missing_ok=True)

    result = {"base_speed_pct": base_speed, "probe_speed_pct": probe_speed,
              "before": before, "at_probe_speed": low, "after_retest": back,
              "persisted_after_retest": bool(back and back["anomaly_score"] and back["anomaly_score"] > 1)}
    if baseline_low and low:
        result["healthy_at_probe_speed"] = baseline_low
        result["vibration_vs_healthy_at_probe_speed"] = ratio(low["vibration_rms"], baseline_low["vibration_rms"])
        result["current_vs_healthy_at_probe_speed"] = ratio(low["current_a"], baseline_low["current_a"])
    return result


def summarize_test(t):
    """One line for the dashboard."""
    if not t:
        return None
    if t.get("safety_stop"):
        return f"Current hit {t['before']['current_a']:.2f} A, so the motor was stopped before testing."
    parts = [f"Slowed to {t['probe_speed_pct']}% and back to {t['base_speed_pct']}%."]
    if t.get("vibration_vs_healthy_at_probe_speed") is not None:
        parts.append(f"At {t['probe_speed_pct']}%: vibration {t['vibration_vs_healthy_at_probe_speed']}x healthy, "
                     f"current {t['current_vs_healthy_at_probe_speed']}x healthy.")
    parts.append("The fault persisted on retest." if t["persisted_after_retest"]
                 else "The anomaly did not return on retest.")
    return " ".join(parts)


def run_tool(name, args):
    if name == "get_detector_output":
        return tool_get_detector_output()
    if name == "get_sensor_trend":
        return tool_get_sensor_trend(args.get("seconds", 5))
    if name == "stop_motor":
        return send_motor("S0", args.get("reason", ""))
    if name == "set_motor_speed":
        pct = max(0, min(100, int(args.get("percent", 0))))
        return send_motor(f"S{pct}", args.get("reason", ""))
    return {"error": f"unknown tool {name}"}


THINK = {"value": False}   # reasoning mode off = much faster replies; --think turns it on


def chat(model, messages, tools=TOOLS):
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": "2h",                      # keep the model loaded in memory
               "options": {"temperature": 0.2}}
    if tools:
        payload["tools"] = tools
    if THINK["value"] is not None:
        payload["think"] = THINK["value"]
    req = urllib.request.Request(OLLAMA, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["message"]
    except urllib.error.HTTPError as e:
        if "think" in payload and e.code == 400:         # model doesn't accept the think flag
            THINK["value"] = None
            return chat(model, messages, tools)
        raise


def warm_up(model):
    """Load the model into memory now, so the first real diagnosis isn't slow."""
    t0 = time.time()
    chat(model, [{"role": "user", "content": "Reply with: ready"}], tools=None)
    print(f"model loaded and warm ({time.time() - t0:.1f}s)")


def investigate(model, trigger, cfg):
    t0 = time.time()
    actions.clear()
    test = None
    if cfg["probe"] == "auto" and trigger.get("source") == "live":
        print(f"  [active test] {cfg['base_speed']}% -> {cfg['probe_speed']}% -> {cfg['base_speed']}%")
        test = run_active_test(trigger, cfg["base_speed"], cfg["probe_speed"], cfg["baseline_low"])
        print(f"  [active test] {summarize_test(test)}")
    note = (f"Active test results: {json.dumps(test)}" if test else
            "No active test was run (replay data or testing disabled); rely on the tools.")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"The detector just flagged: {trigger['fault']} "
                                            f"(confidence {trigger['confidence']}). {note} Investigate."}]
    reply = ""
    for _ in range(6):                       # at most 6 rounds of tool use
        msg = chat(model, messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            reply = msg.get("content", "")
            break
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            result = run_tool(name, args)
            print(f"  [tool] {name}({args})")
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})

    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    out = {"time": time.time(), "detector_fault": trigger["fault"], "diagnosis": reply,
           "actions": list(actions), "latency_s": round(time.time() - t0, 1),
           "test": test, "test_summary": summarize_test(test)}
    tmp = OUT.with_suffix(".json.tmp")           # write then swap, so the dashboard never reads half a file
    tmp.write_text(json.dumps(out))
    tmp.replace(OUT)
    print(f"\n=== AGENT ({out['latency_s']}s) ===\n{reply}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nemotron3:33b")
    p.add_argument("--think", action="store_true", help="enable reasoning mode (slower, smarter)")
    p.add_argument("--cooldown", type=float, default=20, help="seconds before re-diagnosing same fault")
    p.add_argument("--probe", choices=["auto", "off"], default="auto", help="run the active test")
    p.add_argument("--probe-speed", type=int, default=45)
    p.add_argument("--base-speed", type=int, default=70, help="must match detect_live.py --motor")
    args = p.parse_args()

    THINK["value"] = args.think
    cfg = {"probe": args.probe, "probe_speed": args.probe_speed, "base_speed": args.base_speed,
           "baseline_low": load_baseline(args.probe_speed)}
    if args.probe == "auto":
        b = cfg["baseline_low"]
        print(f"active test on: healthy baseline at {args.probe_speed}% = "
              + (f"{b['current_a']} A, {b['vibration_rms']} g ({b['file']})" if b else
                 "none recorded (test still checks whether faults persist)"))
    warm_up(args.model)
    busy, last_fault = threading.Event(), None
    timing = {"last": 0.0, "ok_since": None}
    print(f"agent watching state.json with {args.model}")
    while True:
        try:
            s = json.loads(STATE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.25)
            continue
        if not history or s["time"] != history[-1]["time"]:
            history.append(s)

        new_fault = s["status"] == "FAULT" and (
            s["fault"] != last_fault or time.time() - timing["last"] > args.cooldown)
        if new_fault and not busy.is_set():
            last_fault, timing["last"] = s["fault"], time.time()
            busy.set()
            print(f"detector flagged {s['fault']} -> investigating")

            def job(trigger=s):
                try:
                    investigate(args.model, trigger, cfg)
                except Exception as e:
                    print(f"agent error: {e}")
                finally:
                    timing["last"] = time.time()     # cooldown counts from the end of the work
                    busy.clear()
            threading.Thread(target=job, daemon=True).start()
        # forget the last fault only after the machine has been healthy for a while,
        # so the brief gap after an active test doesn't trigger a second investigation
        if s["status"] == "OK":
            timing["ok_since"] = timing["ok_since"] or time.time()
            if time.time() - timing["ok_since"] > 5:
                last_fault = None
        else:
            timing["ok_since"] = None
        time.sleep(0.25)


if __name__ == "__main__":
    main()
