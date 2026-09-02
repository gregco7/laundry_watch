# LaundryWatch

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![MicroPython](https://img.shields.io/badge/MicroPython-ESP32-2B2728?logo=micropython&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)

![A held-out wash cycle: predicted phase against the phases marked by hand](docs/held-out-cycle.png)

The top bar is what a person marked on the washer's own panel; the bottom is what
the model predicted from vibration alone, on a cycle it had never seen. It all runs
on one laptop — an ESP32 taped to the machine POSTs accelerometer windows over the
LAN, and a FastAPI service turns each one into a phase and serves it to a React
dashboard.

## Layout

```
firmware/   MicroPython sampler for the ESP32
pipeline/   features, calibration, classifier, HMM, rule baseline (pure numpy)
server/     FastAPI + SQLite: ingest, cycles, marks, live inference
analysis/   labeling, training, evaluation — the things you run by hand
web/        Vite + React + Tailwind dashboard
tests/      24 tests, mostly against a tone whose answer is known in advance
```
