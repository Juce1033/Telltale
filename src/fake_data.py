"""
Generates FAKE sensor CSVs in the locked stream format so the detector
pipeline can be built and tested before the rig produces real data.
Delete data/*.csv from this script once real recordings exist.

    python fake_data.py            # writes data/fake_<label>.csv
"""
import csv
from pathlib import Path

import numpy as np

FS = 500            # samples per second; match the ESP32
SECONDS = 120       # per condition
rng = np.random.default_rng(0)


def simulate(label, seconds=SECONDS):
    n = seconds * FS
    t = np.arange(n) / FS
    out_hz, motor_hz = 2.5, 75.0       # output shaft (150 RPM) and motor shaft before gearbox
    load_i = 0.25                      # amps
    speed = 1.0
    rattle = 0.0
    belt_noise = 0.0

    if label == "loose_fastener":
        rattle = 1.0
    elif label == "belt_slip":
        load_i = 0.18                  # output decoupled -> less load
        belt_noise = 1.0
    elif label == "stall":
        load_i = 0.55                  # brake adds load
        speed = 0.7                    # motor slows

    jitter = 1 + 0.02 * np.sin(2 * np.pi * 0.05 * t)   # slow natural speed drift
    phase_out = 2 * np.pi * out_hz * speed * np.cumsum(jitter) / FS
    phase_mot = 2 * np.pi * motor_hz * speed * np.cumsum(jitter) / FS

    vib = (0.02 * np.sin(phase_out) + 0.03 * np.sin(phase_mot)
           + 0.01 * np.sin(2 * phase_mot))
    if rattle:
        # impacts once per output revolution, ringing at ~120 Hz
        impacts = (np.sin(phase_out) > 0.98).astype(float)
        ring = np.convolve(impacts, np.exp(-np.arange(40) / 6) * np.sin(2 * np.pi * 120 * np.arange(40) / FS))[:n]
        vib += 0.15 * ring + 0.05 * np.sin(2 * phase_out)
    if belt_noise:
        vib += 0.03 * rng.standard_normal(n) * (1 + np.sin(phase_out))   # irregular slip bursts

    ax = vib + 0.01 * rng.standard_normal(n)
    ay = 0.7 * vib + 0.01 * rng.standard_normal(n)
    az = 1.0 + 0.5 * vib + 0.01 * rng.standard_normal(n)       # gravity on z
    gx, gy, gz = (k * vib * 50 + 0.3 * rng.standard_normal(n) for k in (1.0, 0.6, 0.3))
    cur = load_i + 0.02 * np.sin(phase_mot / 6) + 0.005 * rng.standard_normal(n)
    if label == "stall":
        cur += 0.05 * np.sin(2 * np.pi * 0.3 * t)               # brake pressure wobble
    t_ms = (t * 1000).astype(int)
    return np.column_stack([t_ms, cur, ax, ay, az, gx, gy, gz])


def main():
    Path("data").mkdir(exist_ok=True)
    for label in ["normal", "loose_fastener", "belt_slip", "stall"]:
        rows = simulate(label, SECONDS * (3 if label == "normal" else 1))
        path = Path("data") / f"fake_{label}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_ms", "current_a", "accel_x", "accel_y", "accel_z",
                        "gyro_x", "gyro_y", "gyro_z", "label"])
            for r in rows:
                w.writerow([int(r[0])] + [f"{v:.5f}" for v in r[1:]] + [label])
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
