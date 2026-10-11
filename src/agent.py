"""
The Nemotron agent. Runs alongside detect_live.py.

It watches state.json. When the detector declares a FAULT, the agent
investigates with tools (detector output, sensor trends vs. baseline),
writes a plain-English diagnosis to agent.json, and can stop or slow the
motor (via motor_cmd.txt, which detect_live.py forwards to the ESP32).

    python agent.py                       # uses nemotron3:33b via Ollama
    python agent.py --model nemotron-3-super
"""
import argparse
import collections
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434/api/chat"
STATE, OUT, CMD = Path("state.json"), Path("agent.json"), Path("motor_cmd.txt")

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

Your job when a fault is flagged:
1. Call tools to check the detector output and the sensor trends against the healthy baseline.
2. Decide whether the evidence actually supports the detector's guess. Say so if it doesn't.
3. If it is a stall with current above 0.6 A, stop the motor to protect it. For other faults,
   do not stop the motor unless the evidence shows something dangerous.
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


def send_motor(cmd, reason):
    CMD.write_text(cmd)
    actions.append(f"{cmd} ({reason})")
    print(f"  [agent action] {cmd}: {reason}")
    return {"ok": True, "sent": cmd}


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


def chat(model, messages):
    body = json.dumps({"model": model, "messages": messages, "tools": TOOLS,
                       "stream": False, "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]


def investigate(model, trigger):
    t0 = time.time()
    actions.clear()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"The detector just flagged: {trigger['fault']} "
                                            f"(confidence {trigger['confidence']}). Investigate."}]
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
           "actions": list(actions), "latency_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out))
    print(f"\n=== AGENT ({out['latency_s']}s) ===\n{reply}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nemotron3:33b")
    p.add_argument("--cooldown", type=float, default=20, help="seconds before re-diagnosing same fault")
    args = p.parse_args()

    busy, last_fault, last_time = threading.Event(), None, 0.0
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
            s["fault"] != last_fault or time.time() - last_time > args.cooldown)
        if new_fault and not busy.is_set():
            last_fault, last_time = s["fault"], time.time()
            busy.set()
            print(f"detector flagged {s['fault']} -> investigating")

            def job(trigger=s):
                try:
                    investigate(args.model, trigger)
                except Exception as e:
                    print(f"agent error: {e}")
                finally:
                    busy.clear()
            threading.Thread(target=job, daemon=True).start()
        if s["status"] == "OK":
            last_fault = None
        time.sleep(0.25)


if __name__ == "__main__":
    main()
