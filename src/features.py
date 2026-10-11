"""
Turns raw sensor rows into feature vectors the models learn from.

One "window" = 1 second of samples. For each window we compute:
  - current: mean, std, peak-to-peak
  - each of the 6 vibration channels: RMS + FFT energy in 25 frequency bands
That's the machine's "fingerprint" for that second.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FS = 500                    # samples/sec -- MUST match the ESP32 stream
WIN = FS                    # 1-second windows
HOP = FS // 4               # new window every 0.25 s (windows overlap)
N_BANDS = 25
COLS = ["timestamp_ms", "current_a", "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z"]

_hann = np.hanning(WIN)
_band_edges = np.linspace(0, WIN // 2 + 1, N_BANDS + 1).astype(int)


def featurize(win):
    """win: array [WIN, 8] in COLS order -> 1-D feature vector."""
    cur = win[:, 1]
    feats = [cur.mean(), cur.std(), cur.max() - cur.min()]
    for ch in range(2, 8):
        x = win[:, ch] - win[:, ch].mean()           # drop gravity / DC offset
        feats.append(np.sqrt(np.mean(x ** 2)))      # RMS
        spec = np.abs(np.fft.rfft(x * _hann))
        bands = [spec[a:b].sum() for a, b in zip(_band_edges[:-1], _band_edges[1:])]
        feats.extend(np.log1p(bands))
    return np.asarray(feats, dtype=np.float32)


def windows(arr):
    """Slide over a [N, 8] array, yielding featurized windows."""
    for start in range(0, len(arr) - WIN + 1, HOP):
        yield featurize(arr[start:start + WIN])


def check_rate(df, path):
    dt = np.diff(df["timestamp_ms"].to_numpy())
    if len(dt) == 0:
        return
    rate = 1000 / np.median(dt) if np.median(dt) > 0 else float("nan")
    gaps = int((dt > 3 * np.median(dt)).sum())
    if abs(rate - FS) > 0.1 * FS or gaps:
        print(f"  WARNING {path.name}: ~{rate:.0f} Hz (expected {FS}), {gaps} gaps")


# Labels that are not machine faults: things the model should recognise and report,
# but never raise an alarm for. Healthy-type labels can be augmented.
NOT_FAULTS = {"normal", "disturbance"}
AUGMENT = {"normal": 3}          # extra synthetic copies per real training window


def augment_window(win, rng):
    """One realistic variation of a healthy 1-second window:
    vibration level +-15%, motor speed +-3%, a little extra sensor noise, current +-5%."""
    w = win.copy()
    f = rng.uniform(0.97, 1.03)                       # speed change = time stretch
    n = len(w)
    src = np.linspace(0, n - 1, n) * f
    src = np.clip(src, 0, n - 1)
    for ch in range(1, 8):
        w[:, ch] = np.interp(src, np.arange(n), win[:, ch])
    w[:, 1] *= rng.uniform(0.95, 1.05)                # current level
    for ch in range(2, 8):                            # vibration level + noise, gravity kept
        mean = w[:, ch].mean()
        dev = (w[:, ch] - mean) * rng.uniform(0.85, 1.15)
        dev += rng.normal(0, 0.10 * (dev.std() + 1e-9), n)
        w[:, ch] = mean + dev
    return w


def load_dataset(folder="data", augment=True, train_frac=0.8, seed=0):
    """Returns X [n, feats], labels [n], file_ids [n], positions 0..1 within file [n],
    and is_aug [n] (True for synthetic copies; those only ever land in the training split)."""
    rng = np.random.default_rng(seed)
    X, y, fid, pos, aug = [], [], [], [], []
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {folder}/")
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        check_rate(df, path)
        label = str(df["label"].iloc[0]) if "label" in df else path.stem.split("_")[0]
        label = label.replace("-", "_")
        arr = df[COLS].to_numpy(dtype=np.float64)
        starts = list(range(0, len(arr) - WIN + 1, HOP))
        p = np.linspace(0, 1, len(starts))
        copies = AUGMENT.get(label, 0) if augment else 0
        n_aug = 0
        for st, pp in zip(starts, p):
            win = arr[st:st + WIN]
            X.append(featurize(win)); y.append(label); fid.append(i); pos.append(pp); aug.append(False)
            if pp <= train_frac:                      # augment training windows only
                for _ in range(copies):
                    X.append(featurize(augment_window(win, rng)))
                    y.append(label); fid.append(i); pos.append(pp); aug.append(True)
                    n_aug += 1
        extra = f" (+{n_aug} augmented)" if n_aug else ""
        print(f"  {path.name}: {len(starts)} windows{extra}, label={label}")
    return np.stack(X), np.array(y), np.array(fid), np.array(pos), np.array(aug)


def healthy_reference(folder="data"):
    """Typical healthy current and vibration (same math as detect_live.py), measured over
    every 1-second window of the 'normal' recordings. The agent compares against this."""
    cur, vib = [], []
    for path in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(path)
        if str(df["label"].iloc[0]).replace("-", "_") != "normal":
            continue
        arr = df[COLS].to_numpy(dtype=np.float64)
        for st in range(0, len(arr) - WIN + 1, WIN):
            w = arr[st:st + WIN]
            cur.append(w[:, 1].mean())
            acc = w[:, 2:5] - w[:, 2:5].mean(0)
            vib.append(float(np.sqrt((acc ** 2).mean())))
    if not cur:
        return None
    return {"current_a": round(float(np.mean(cur)), 4),
            "current_a_range": [round(float(np.percentile(cur, 1)), 4), round(float(np.percentile(cur, 99)), 4)],
            "vibration_rms": round(float(np.mean(vib)), 4),
            "vibration_rms_p99": round(float(np.percentile(vib, 99)), 4)}
