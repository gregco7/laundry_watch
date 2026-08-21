# LaundryWatch — Build Plan & Repo Guide

A local-first system that learns what a wash or dry cycle looks like in the
frequency domain and reports which phase a machine is in. Everything runs on
one laptop: the database, the ML pipeline, and the web app. Sensor nodes POST
readings over the home WiFi. Anyone can clone this and train on their own
machines.

**How this repo is being built:** scaffold first, then code **function by
function** with AI assistance, so every piece is understood before moving on.
Each stub file has a module docstring and a commented list of the functions it
will contain. Implement them one at a time, in the order below.

---

## Architecture in one breath

```
machine vibrates -> sensor -> ESP32 reads it -> POST /readings (LAN, plain HTTP)
   -> normalize (z-score vs machine calibration)
   -> classifier predicts phase
   -> HMM smooths over recent windows
   -> machine state + cycle record updated in SQLite
   -> GET /machines -> React dashboard
```

Local-first on purpose: no cloud, no HTTPS, no reverse proxy. Trade-off — the
laptop must be running to log data. Fine for a home project; the same code runs
unchanged on a Raspberry Pi if always-on is ever wanted.

---

## Repo map

```
laundry_watch/
├── analysis/                 # things you run by hand
│   ├── label.py              # spectrogram labeling tool
│   ├── train.py              # fit the classifier -> models/clf-v1.joblib
│   ├── evaluate.py           # held-out-by-cycle metrics, model vs baseline
│   ├── recordings/           # raw JSONL training data — COMMITTED
│   └── labels/               # phase-boundary JSON — COMMITTED
├── pipeline/                 # the reusable core (pure logic, no web)
│   ├── features.py           # windowing + FFT -> feature vector
│   ├── calibrate.py          # per-machine baseline + z-scoring
│   ├── rules.py              # rule-based baseline state machine
│   ├── model.py              # sklearn train / predict / save / load
│   └── hmm.py                # transition matrix + Viterbi smoothing
├── server/                   # the local web service
│   ├── main.py               # FastAPI routes (POST /readings, GET /machines)
│   ├── db.py                 # SQLite engine + session
│   ├── models.py             # SQLAlchemy tables
│   └── schemas.py            # Pydantic request/response shapes
├── web/                      # Vite + React + Tailwind dashboard (init later)
├── models/                   # clf-v1.joblib (generated, committed — it's small)
├── requirements.txt
├── docker-compose.yml        # optional; SQLite means you don't need it
└── README.md
```

`pipeline/` knows nothing about the web — it's plain functions over numpy
arrays. `server/` imports `pipeline/` and adds HTTP + storage. `analysis/`
imports `pipeline/` and adds plots + scripts. Keep that dependency direction.

---

## Build order (function by function)

Work bottom-up: the pipeline core first (testable in isolation), then the
scripts that use it, then the server, then the web.

### 1. `pipeline/features.py` — raw window -> fingerprint
The foundation everything else rests on. Per 256-sample window (~2.56 s, 100 Hz,
50% overlap): subtract mean (remove gravity) -> Hann window -> FFT -> reduce to
~15 numbers (8 log-spaced band energies 0.5–50 Hz, spectral centroid, RMS,
top-3 band deltas, mid-band rolling mean). Write a unit test against a synthetic
sine wave before trusting it.

### 2. `pipeline/calibrate.py` — normalization
Learn a machine's quiet baseline (mean/std per feature), z-score every later
feature against it. **This is the idea that lets one model cover both machines.**

### 3. `pipeline/rules.py` — the baseline
Thresholds on normalized band energy with hysteresis. Build it *before* the ML.
Measure its cycle-end timing error and record the number — it's the yardstick.

### 4. `analysis/label.py` — the labeling tool
Render a recording's spectrogram, click phase boundaries, write label JSON.
Build it before you have a pile of cycles to label.

### 5. `pipeline/model.py` — the classifier
`HistGradientBoostingClassifier`. train / predict / predict_proba / save / load.
Kept small on purpose; a neural net overfits this data and adds nothing.

### 6. `pipeline/hmm.py` — temporal smoothing
Transition matrix from laundry domain rules, Viterbi over the classifier's
per-window probabilities. Replaces hand-tuned hysteresis with something
defensible.

### 7. `analysis/train.py` + `analysis/evaluate.py`
Wire it together offline: features -> z-score -> label -> fit -> save; then
evaluate held-out-by-cycle and compare against the baseline.

### 8. `server/` — expose it
`db.py` (SQLite) -> `models.py` (tables) -> `schemas.py` (Pydantic) ->
`main.py` (routes). The `POST /readings` handler runs the pipeline inline.

### 9. `web/` — one page
Vite + React + Tailwind. A card per machine reading from `GET /machines`.

---

## The three things you can't get wrong

1. **Normalize per machine.** Z-score every feature against that machine's own
   quiet baseline. This is what makes one model general.
2. **Split test data by whole cycle, never random windows.** Adjacent windows
   are correlated; a random split leaks the answer and inflates accuracy into
   fiction. `analysis/evaluate.py` must split by cycle.
3. **Iterate offline.** Raw recordings are saved as files, so retrain against
   them in seconds. Never tune against a live load.

---

## Data shapes

```json
// analysis/recordings/*.jsonl  — one object per window (raw / dev mode)
{"t": 1723500123.44, "node": "washer-01", "mode": "raw",
 "hz": 100, "axis": "z", "samples": [ /* 256 ints */ ]}

// production reading — small feature summary POSTed live
{"t": 1723500123.44, "node": "washer-01",
 "bands": [ /* 8 floats */ ], "centroid": 12.4, "rms": 0.44}

// analysis/labels/*.json — human-drawn phase spans
{"file": "wash-a.jsonl", "machine": "washer-01",
 "phases": [{"start": ..., "end": ..., "label": "agitate"},
            {"start": ..., "end": ..., "label": "spin"}]}
```

Phase labels: `idle` · `fill` · `agitate` · `spin` · `tumble` · `done`.
Collapse whatever the data won't separate (e.g. `fill` may be indistinguishable
from `idle`) and say so — that's a finding, not a failure.

---

## Conventions

- Python 3.11+. `pip install -r requirements.txt` in a venv.
- `pipeline/` is pure and import-light; no FastAPI or SQLAlchemy in there.
- Sampling is nominally 100 Hz but **measure the real rate** — loops drift.
- Run the server: `uvicorn server.main:app --host 0.0.0.0 --reload`
- Commit `recordings/` and `labels/` — the dataset is part of the deliverable.
