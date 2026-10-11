# Telltale

Telltale is a hardware-monitoring and AI-fault-detection project built for a UCSC Silicon Valley hackathon. It combines a tiny sensor rig, live inference code, and a lightweight frontend/dashboard to detect machine faults such as loose fasteners, belt slip, and stalls from current and vibration data.

## What this project does

The system records current and vibration data from a small motor-driven test bench, extracts signal features, trains a fault detector model, and then runs live inference to classify faults and help diagnose what is wrong.

It includes:

- ESP32 firmware for sampling current and vibration
- Python feature extraction and model training
- Live detection pipeline for streaming sensor data
- Optional agent-based diagnosis using Ollama / Nemotron
- Local dashboard UI for monitoring faults and timing detection latency

## Repository layout

- `failure_bench/` — ESP32 firmware that reads the INA226 and MPU6050 sensors and streams CSV-style samples over serial
- `frontend/` — browser dashboard and local server for monitoring detector output and fault timing
- `src/` — Python code for feature engineering, training, live detection, and agent-based diagnosis
- `requirements.txt` — Python dependencies for the ML/data pipeline
- `LICENSE` — MIT license

## Hardware and sensor pipeline

The bench firmware in `failure_bench/failure_bench.ino` samples:

- motor current via INA226
- vibration and motion via MPU6050
- serial output at 921600 baud

Each row looks like:

```text
timestamp_ms,current_a,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
```

This makes it easy to pipe the data into feature extraction and detection logic in Python.

## Python pipeline

The main code lives under `src/`:

- `features.py` — converts raw sensor samples into time-window features used by the models
- `train.py` — trains an autoencoder and classifier and saves `model.pt`
- `detect_live.py` — runs the trained models on live serial input or a CSV replay
- `agent.py` — watches `state.json`, investigates anomalies, and may send motor commands
- `logger.py` — helper for logging/tracing detector output
- `fake_data.py` — utilities for generating synthetic data for testing

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the detector model:

```bash
cd src
python train.py
```

3. Run the live detector against a serial port or a replay CSV:

```bash
cd src
python detect_live.py --port /dev/ttyUSB0
```

Or replay recorded data:

```bash
cd src
python detect_live.py --replay data/fake_stall.csv
```

4. Launch the dashboard:

```bash
cd frontend
python dashboard.py
```

Then open http://localhost:8080 in a browser.

## Agent mode

The project also includes an agent loop for diagnosis. It watches detector state and uses Ollama-hosted models to reason about faults and suggest actions.

Example:

```bash
cd src
python agent.py --model nemotron3:33b
```

This expects an Ollama server running locally on `http://localhost:11434`.

## Typical fault classes

The detector is trained to recognize faults such as:

- `normal`
- `loose_fastener`
- `belt_slip`
- `stall`

The system combines anomaly detection with classification so it can flag unusual behavior and identify likely faults.

## Notes

This project is designed for rapid experimentation and local monitoring, not production deployment. It is intended to run on a small, self-contained hardware bench with no internet dependency for the core live loop.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
