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
- loose_fastener: a bearing-block screw backed off. The vibration PATTERN changes (rattling, impacts
  once per revolution); overall vibration RMS may rise a lot or barely at all, depending on where
  the sensor sits. Current roughly unchanged. Risk: bearing misalignment, wear. Fix: retighten.
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
   The detector is a trained model and its alarm is reliable: if it flagged a fault (anomaly
   score above 1), never conclude "no action needed" unless the active test showed the anomaly
   did NOT persist. If the evidence doesn't fit the named fault, say the cause is uncertain and
   still recommend an inspection. A ratio near 1.0 (0.9-1.1) means essentially unchanged.
   The detector judges the full frequency pattern, not just overall RMS: a confident detector guess
   (confidence above 0.7) with nothing contradicting it (e.g. current unchanged for a loose
   fastener) should be reported as that fault, not as uncertain.
3. Do NOT stop the motor yourself for a stall: once a second reviewer confirms your stall
   diagnosis, the system stops the motor automatically. In ACTION for a stall, say the motor is
   being stopped and what to check (what is braking the output). For other faults, do not stop
   the motor unless the evidence shows something dangerous.
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


_healthy_cache = {}


def healthy_from_data():
    """For a model.pt trained before it stored the healthy reference: measure it from data/."""
    if "ref" not in _healthy_cache:
        try:
            from features import healthy_reference
            _healthy_cache["ref"] = healthy_reference("data")
        except Exception as e:
            print(f"couldn't read healthy recordings: {e}")
            _healthy_cache["ref"] = None
    return _healthy_cache["ref"]


def tool_get_sensor_trend(seconds=5):
    # last N seconds of DATA (not of the clock): still works after a replay ends or is stopped
    now = history[-1]["time"] if history else time.time()
    recent = [s for s in history if now - s["time"] <= float(seconds) and s["status"] != "PROBE"]
    faulty = [s for s in recent if s["status"] == "FAULT"]
    recent = faulty or recent                   # describe the fault itself, not the moments around it
    ref = (history[-1].get("healthy_ref") if history else None) or healthy_from_data()
    if ref:      # the healthy machine as measured in the training recordings
        base_cur, base_vib = ref["current_a"], ref["vibration_rms"]
        source = "training recordings of the healthy machine"
    else:        # older model without a reference: fall back to recent healthy readings
        healthy = [s for s in history if s["status"] == "OK"]
        base_cur = mean([s["current_mean_a"] for s in healthy])
        base_vib = mean([s["vibration_rms"] for s in healthy])
        source = "recent readings marked healthy (less reliable)"
    vib_now = mean([s["vibration_rms"] for s in recent])
    cur_now = mean([s["current_mean_a"] for s in recent])
    return {
        "window_s": seconds,
        "units": "current in amps (A); vibration is acceleration RMS in g",
        "readings_used": len(recent),
        "baseline_source": source,
        "current_now_a": cur_now,
        "current_baseline_a": base_cur,
        "current_ratio": round(cur_now / base_cur, 2) if cur_now and base_cur else None,
        "vibration_rms_now": vib_now,
        "vibration_rms_baseline": base_vib,
        "vibration_ratio": round(vib_now / base_vib, 2) if vib_now and base_vib else None,
        "healthy_vibration_upper_limit": ref.get("vibration_rms_p99") if ref else None,
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


def chat(model, messages, tools=TOOLS, temperature=0.2):
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": "2h",                      # keep the model loaded in memory
               "options": {"temperature": temperature}}
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
            return chat(model, messages, tools, temperature)
        raise


def warm_up(model):
    """Load the model into memory now, so the first real diagnosis isn't slow."""
    t0 = time.time()
    chat(model, [{"role": "user", "content": "Reply with: ready"}], tools=None)
    print(f"model loaded and warm ({time.time() - t0:.1f}s)")


REVIEWER = """You are the second-opinion reviewer for a maintenance agent watching a small test
machine (12V gearmotor, belt drive, two bearing blocks; sensed by motor current and vibration).
Another agent has written a diagnosis. You get the raw measurements and its draft. Check it.
You cannot touch the machine; you only approve or send it back.

Fault signatures (ratios compare to the healthy machine; 0.9-1.1 means unchanged):
- loose_fastener: vibration pattern changes (overall size may or may not rise), current about unchanged.
- belt_slip: current DOWN (ratio below about 0.9).
- stall: current UP sharply; dangerous above 0.6 A.
- If the active test says persisted_after_retest = false, it was a transient knock, not a fault.

The measurements include "signature_check", computed in code from the ratios. It is
arithmetic, not opinion: trust it. If it says CONSISTENT, check 2 passes. Overall vibration
size is NOT a reason to fail a draft: the detector judges the vibration pattern, which the
ratios don't capture.

Checklist:
1. Every number in EVIDENCE must match the measurements (small rounding is fine).
2. The named fault must not contradict its signature: use signature_check.
3. ACTION must match the severity: a confirmed fault needs a physical repair step; "no action"
   is only acceptable when the active test showed the anomaly did not persist.
4. Format must be the three lines FAULT / EVIDENCE / ACTION.
A value that is null or missing means no data, NOT evidence against the draft: never fail a draft
because of it; judge only the values that are present. Do not fail a draft over wording or style. Approve unless something above is actually wrong.

Reply with exactly one of:
VERDICT: APPROVE
or
VERDICT: REVISE
CHECK: <the number of the check that failed: 1, 2, 3 or 4>
REASON: <one sentence: what is wrong and what the measurements actually show>"""

FAULT_NAMES = ("loose_fastener", "belt_slip", "stall", "disturbance", "unknown_anomaly", "normal")


def named_fault(draft):
    line = parse_lines(draft).get("FAULT", "").lower().replace("-", "_").replace(" ", "_")
    return next((f for f in FAULT_NAMES if f in line), None)


def signature_check(fault, trend, test):
    """Does the named fault match its physical signature? Plain arithmetic on the ratios."""
    v, c = trend.get("vibration_ratio"), trend.get("current_ratio")
    if test and test.get("persisted_after_retest") is False:
        return {"fault": fault, "fits": fault in ("disturbance", "normal"),
                "why": "active test: the anomaly vanished on retest, so it was transient"}
    if fault is None or v is None or c is None:
        return {"fault": fault, "fits": None, "why": "not enough data to check"}
    # Only veto on a CONTRADICTION. Current is the reliable signal; vibration size alone is not
    # (where the sensor sits changes it, and the detector judges the whole frequency pattern).
    rules = {
        "loose_fastener": (0.8 <= c <= 1.25, "current should stay roughly unchanged (0.8-1.25x)"),
        "belt_slip":      (c <= 1.05, "current should not rise (at most 1.05x)"),
        "stall":          (c >= 0.95, "current should not drop (at least 0.95x)"),
    }
    if fault not in rules:          # unknown_anomaly etc.: nothing specific to contradict
        return {"fault": fault, "fits": None, "why": "no fixed signature for this name"}
    ok, need = rules[fault]
    return {"fault": fault, "fits": ok,
            "why": f"{fault}: {need}; measured current {c}x, vibration {v}x -> "
                   f"{'CONSISTENT' if ok else 'CONTRADICTS'}"}

NO_ACTION = re.compile(r"\b(no (further )?(action|repair|maintenance|intervention)s?( is)?( needed| required| necessary)?"
                       r"|not (needed|required|necessary)|nothing to (do|fix)|all (good|fine)"
                       r"|continue (normal )?operation)\b", re.I)


def parse_lines(text):
    out = {}
    for l in (text or "").splitlines():
        m = re.match(r"\s*(FAULT|EVIDENCE|ACTION)\s*:\s*(.*)$", l, re.I)
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
    return out


def rule_checks(draft, facts):
    """Hard checks in plain code. The reviewer can't overrule these."""
    problems, lines = [], parse_lines(draft)
    missing = [k for k in ("FAULT", "EVIDENCE", "ACTION") if k not in lines]
    if missing:
        problems.append(f"missing the {', '.join(missing)} line(s) of the required format")
    test = facts.get("active_test") or {}
    transient = test.get("persisted_after_retest") is False
    if lines.get("ACTION") and NO_ACTION.search(lines["ACTION"]) and not transient:
        problems.append("it said no repair was needed, but the fault is confirmed and the active test didn't clear it")
    return problems


def review(model, draft, facts):
    """Second Nemotron call: approve the draft or say what's wrong with it.
    Code has the last word on anything code can calculate."""
    problems = rule_checks(draft, facts)
    sig = signature_check(named_fault(draft), facts["sensor_trend_last_5s"], facts.get("active_test"))
    if sig["fits"] is False:
        problems.append(f"the numbers contradict {sig['fault']}: {sig['why']}")
    facts = {**facts, "signature_check": sig}
    msg = chat(model, [{"role": "system", "content": REVIEWER},
                       {"role": "user", "content": f"MEASUREMENTS:\n{json.dumps(facts, indent=1)}\n\n"
                                                   f"DRAFT DIAGNOSIS:\n{draft}"}], tools=None, temperature=0)
    text = re.sub(r"<think>.*?</think>", "", msg.get("content", ""), flags=re.S).strip()
    reason = re.search(r"REASON\s*:\s*(.+)", text, re.I)
    check = re.search(r"CHECK\s*:\s*(\d)", text, re.I)
    check = check.group(1) if check else None
    if re.search(r"VERDICT\s*:\s*REVISE", text, re.I):
        why = reason.group(1).strip() if reason else "the reviewer found a problem"
        # overrule the reviewer where code already knows the answer
        if check is None and sig["fits"] is True and not problems:
            print(f"  [review] overruled (no check named; code checks all pass): {why}")
        elif check == "2" and sig["fits"] is True:
            print(f"  [review] overruled (signature consistent by calculation): {why}")
        elif check == "4" and not any("format" in p for p in problems):
            print(f"  [review] overruled (format is fine): {why}")
        elif check == "3" and not any("no repair" in p for p in problems) and not NO_ACTION.search(
                parse_lines(draft).get("ACTION", "")):
            print(f"  [review] overruled (action gives a repair step): {why}")
        else:
            problems.append(why)
    elif not re.search(r"VERDICT\s*:\s*APPROVE", text, re.I) and not problems:
        print(f"  [review] unclear reply, treating as approved: {text[:80]!r}")
    return problems


def set_stage(stage, trigger):
    """Tell the dashboard what the agents are doing right now (no diagnosis yet)."""
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"time": time.time(), "stage": stage, "detector_fault": trigger["fault"]}))
    tmp.replace(OUT)


def investigate(model, trigger, cfg):
    t0 = time.time()
    actions.clear()
    test = None
    stall = trigger["fault"] == "stall"
    if stall and (trigger.get("current_mean_a") or 0) > SAFETY_STOP_A:
        # past the danger line: protect the motor and driver now, don't wait for two AI calls
        send_motor("S0", f"safety stop: current {trigger['current_mean_a']} A is above {SAFETY_STOP_A} A")
    if cfg["probe"] == "auto" and trigger.get("source") == "live" and not stall:
        # (no active test for a stall: changing speed while the output is braked tells us nothing)
        print(f"  [active test] {cfg['base_speed']}% -> {cfg['probe_speed']}% -> {cfg['base_speed']}%")
        baseline = load_baseline(cfg["probe_speed"])     # re-read: it may have been recorded since startup
        test = run_active_test(trigger, cfg["base_speed"], cfg["probe_speed"], baseline)
        print(f"  [active test] {summarize_test(test)}")
    note = (f"Active test results: {json.dumps(test)}" if test else
            "No active test was run (replay data or testing disabled); rely on the tools.")
    set_stage("diagnosing", trigger)
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

    # ---- second opinion: a fresh Nemotron call checks the draft against the raw numbers
    rev = None
    if cfg.get("review", True):
        t1 = time.time()
        set_stage("reviewing", trigger)
        facts = {"detector": {k: trigger.get(k) for k in ("fault", "confidence", "anomaly_score")},
                 "sensor_trend_last_5s": tool_get_sensor_trend(5),
                 "active_test": test, "motor_commands_sent": list(actions)}
        if not facts["sensor_trend_last_5s"]["readings_used"]:
            problems, rev = [], {"status": "skipped", "reason": "no sensor readings left to check against"}
        else:
            problems = review(model, reply, facts)
            rev = {"status": "approved", "reason": None, "first_draft": None}
        if problems:
            reason = "; ".join(problems)
            print(f"  [review] sent back: {reason}")
            set_stage("revising", trigger)
            messages.append({"role": "user", "content":
                f"A second reviewer rejected your diagnosis: {reason}. Measurements: {json.dumps(facts)}. "
                "Write a corrected diagnosis in exactly the same three-line format."})
            fixed = re.sub(r"<think>.*?</think>", "", chat(model, messages, tools=None).get("content", ""),
                           flags=re.S).strip()
            still = review(model, fixed, facts) if fixed else ["empty revision"]
            rev = {"status": "revised" if not still else "disputed", "reason": reason,
                   "still": "; ".join(still) or None, "first_draft": reply}
            reply = fixed or reply
        else:
            print(f"  [review] {rev['status']}")
        rev["latency_s"] = round(time.time() - t1, 1)

    # ---- stall confirmed by both agents -> stop the motor
    confirmed = rev is None or rev["status"] in ("approved", "revised")
    if named_fault(reply) == "stall" and confirmed and not any(a.startswith("S0") for a in actions):
        send_motor("S0", "stall confirmed by the reviewer" if rev else "stall diagnosed")
    elif named_fault(reply) == "stall" and not confirmed:
        print("  [stall] reviewer disputes the diagnosis: motor left running for the operator to decide")

    out = {"time": time.time(), "detector_fault": trigger["fault"], "diagnosis": reply,
           "actions": list(actions), "latency_s": round(time.time() - t0, 1),
           "test": test, "test_summary": summarize_test(test), "review": rev}
    tmp = OUT.with_suffix(".json.tmp")           # write then swap, so the dashboard never reads half a file
    tmp.write_text(json.dumps(out))
    tmp.replace(OUT)
    print(f"\n=== AGENT ({out['latency_s']}s) ===\n{reply}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nemotron3:33b")
    p.add_argument("--no-review", action="store_true", help="skip the second-opinion reviewer")
    p.add_argument("--think", action="store_true", help="enable reasoning mode (slower, smarter)")
    p.add_argument("--probe", choices=["auto", "off"], default="auto", help="run the active test")
    p.add_argument("--probe-speed", type=int, default=45)
    p.add_argument("--base-speed", type=int, default=70, help="must match detect_live.py --motor")
    args = p.parse_args()

    THINK["value"] = args.think
    cfg = {"review": not args.no_review, "probe": args.probe, "probe_speed": args.probe_speed, "base_speed": args.base_speed,
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
        if history and s["time"] - history[-1]["time"] > 3:
            # a gap in the readings means a new run started: forget the previous one
            history.clear()
            last_fault, timing["ok_since"] = None, None
        if not history or s["time"] != history[-1]["time"]:
            history.append(s)

        # one diagnosis per fault: re-investigate only for a different fault, or the same one
        # coming back after the machine has been healthy for a while (last_fault reset below)
        new_fault = s["status"] == "FAULT" and s["fault"] != last_fault
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
