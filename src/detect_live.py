"""
Runs the trained models on a live stream (or replays a CSV for testing).
Every 0.25 s it prints a verdict and writes state.json, which the
Nemotron agent and the dashboard read.

    python detect_live.py --port /dev/ttyUSB0          # live from ESP32
    python detect_live.py --replay data/fake_stall.csv # test without hardware
"""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import COLS, FS, HOP, WIN, featurize
from train import AutoEncoder, Classifier

CONFIRM = 3     # consecutive anomalous windows before declaring a fault (~0.75 s)


def load_models(path="model.pt"):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    ae = AutoEncoder(ck["d"]); ae.load_state_dict(ck["ae"]); ae.eval()
    clf = Classifier(ck["d"], len(ck["classes"])); clf.load_state_dict(ck["clf"]); clf.eval()
    return ae, clf, ck


SER = None      # open serial port, shared so agent motor commands can be sent
CMD_FILE = Path("motor_cmd.txt")


def forward_motor_cmd():
    """The agent writes e.g. 'S0' to motor_cmd.txt; pass it to the ESP32."""
    if not CMD_FILE.exists():
        return
    cmd = CMD_FILE.read_text().strip()
    CMD_FILE.unlink()
    if SER:
        SER.write((cmd + "\n").encode())
    print(f">>> motor command {cmd}" + ("" if SER else " (replay mode: not sent)"))


def serial_rows(port, baud, motor):
    global SER
    import serial
    ser = SER = serial.Serial(port, baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(f"S{motor}\n".encode())     # ESP32 boots with motor stopped; start it
    while True:
        parts = ser.readline().decode("ascii", errors="ignore").strip().split(",")
        if len(parts) != 8:
            continue
        try:
            yield [float(p) for p in parts]
        except ValueError:
            continue


def replay_rows(path, speed=1.0):
    for row in pd.read_csv(path)[COLS].to_numpy():
        yield list(row)
        time.sleep(1 / (FS * speed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--replay")
    p.add_argument("--motor", type=int, default=70, help="motor speed %% (live mode)")
    p.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    args = p.parse_args()

    ae, clf, ck = load_models()
    mu, sd, thr, classes = ck["mu"], ck["sd"], ck["threshold"], ck["classes"]
    rows = replay_rows(args.replay, args.speed) if args.replay else serial_rows(args.port, args.baud, args.motor)

    buf = collections.deque(maxlen=WIN)
    streak, since_hop, fault_start = 0, 0, None
    for row in rows:
        buf.append(row)
        since_hop += 1
        if len(buf) < WIN or since_hop < HOP:
            continue
        since_hop = 0
        forward_motor_cmd()

        x = torch.tensor((featurize(np.array(buf)) - mu) / sd, dtype=torch.float32)[None]
        with torch.no_grad():
            score = float(((ae(x) - x) ** 2).mean()) / thr          # >1 means anomalous
            probs = torch.softmax(clf(x), 1)[0].numpy()

        streak = streak + 1 if score > 1 else 0
        arr = np.array(buf)
        if streak >= CONFIRM:
            ranked = [(classes[i], float(probs[i])) for i in np.argsort(-probs) if classes[i] != "normal"]
            fault, conf = ranked[0]
            fault_start = fault_start or time.time()
        else:
            fault, conf, fault_start = "normal", float(probs[classes.index("normal")]), None

        state = {
            "time": time.time(),
            "status": "FAULT" if fault != "normal" else "OK",
            "fault": fault,
            "confidence": round(conf, 3),
            "anomaly_score": round(score, 3),          # 1.0 = threshold
            "fault_active_s": round(time.time() - fault_start, 2) if fault_start else 0,
            "class_probs": {c: round(float(pr), 3) for c, pr in zip(classes, probs)},
            "current_mean_a": round(float(arr[:, 1].mean()), 4),
            "vibration_rms": round(float(np.sqrt(((arr[:, 2:5] - arr[:, 2:5].mean(0)) ** 2).mean())), 4),
        }
        with open("state.json", "w") as f:
            json.dump(state, f)
        print(f"{state['status']:<6}{fault:<16}score={score:6.2f}  conf={conf:.2f}  I={state['current_mean_a']:.3f}A")


if __name__ == "__main__":
    main()
