"""
Serial logger: reads sensor lines from the ESP32 and saves them to a CSV.

Expects the ESP32 to print one line per reading, 8 comma-separated values:
    timestamp_ms,current_a,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z

Usage (on the GX10):
    pip install pyserial
    python logger.py --port /dev/ttyUSB0 --label normal --seconds 600

Find the port with:  ls /dev/ttyACM* /dev/ttyUSB*
"""
import argparse
import csv
import time
from pathlib import Path

import serial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--label", default="normal", help="normal, loose_fastener, belt_slip, stall")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--outdir", default="data")
    p.add_argument("--motor", type=int, default=70, help="motor speed %% to run while recording (0 = off)")
    args = p.parse_args()

    Path(args.outdir).mkdir(exist_ok=True)
    out_path = Path(args.outdir) / f"{args.label}_{time.strftime('%H%M%S')}.csv"

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)              # ESP32 resets when the port opens
    ser.reset_input_buffer()   # throw away anything half-sent
    ser.write(f"S{args.motor}\n".encode())   # ESP32 boots with motor stopped; start it
    time.sleep(1.5)            # let the motor get up to speed before recording
    ser.reset_input_buffer()

    n_rows, n_bad = 0, 0
    start = time.time()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "current_a", "accel_x", "accel_y", "accel_z",
                         "gyro_x", "gyro_y", "gyro_z", "label"])
        while time.time() - start < args.seconds:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            parts = line.split(",")
            if len(parts) != 8:
                n_bad += 1     # partial or garbled line; skip it
                continue
            try:
                row = [int(parts[0])] + [float(x) for x in parts[1:]]
            except ValueError:
                n_bad += 1
                continue
            writer.writerow(row + [args.label])
            n_rows += 1
            if n_rows % 1000 == 0:
                rate = n_rows / (time.time() - start)
                print(f"{n_rows} rows, ~{rate:.0f} samples/s, {n_bad} bad lines")

    ser.write(b"S0\n")        # stop the motor when done
    ser.close()
    print(f"Saved {n_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
