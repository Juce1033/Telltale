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


def load_dataset(folder="data"):
    """Returns X [n_windows, n_feats], labels [n_windows], file_ids, positions (0..1 within file)."""
    X, y, fid, pos = [], [], [], []
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {folder}/")
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        check_rate(df, path)
        label = df["label"].iloc[0] if "label" in df else path.stem.split("_")[0]
        feats = list(windows(df[COLS].to_numpy(dtype=np.float64)))
        X += feats
        y += [label] * len(feats)
        fid += [i] * len(feats)
        pos += list(np.linspace(0, 1, len(feats)))
        print(f"  {path.name}: {len(feats)} windows, label={label}")
    return np.stack(X), np.array(y), np.array(fid), np.array(pos)
