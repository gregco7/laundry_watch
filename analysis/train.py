"""Fit the classifier from every labelled recording on this machine.

    recordings + labels -> features -> calibration -> z-score -> fit -> joblib

Run:  venv/bin/python analysis/train.py

Everything here is offline and re-runnable in seconds, which is the point.
Raw windows are kept as files precisely so that a change to features.py means
retraining against the same six washes rather than washing six more loads.
Nothing in this file should ever be tuned against a live machine.

WHAT THIS FILE OWNS, AND WHY IT IS ONE FILE:

The z-scoring happens exactly once, here. pipeline/model.py cannot tell whether
it was handed raw or normalized features -- both are (n, 15) arrays of finite
floats -- so a second code path that forgets to normalize would train a model
that works today and breaks the day the sensor is re-taped, with no error to
follow. One place does it, and the server reads the same stored calibration
back out rather than recomputing anything.

The calibration is written to machines.calibration, not into the model file.
That split is the whole "one model, any washer" idea: the MODEL is shared and
machine-independent, the BASELINE is per machine. Bundling them would make a
second washer need its own model.

WHAT IT DELIBERATELY DOES NOT DO:

No holdout, no accuracy claim. Every window goes into the fit, and the number
this prints at the end is training accuracy, which is a measure of how much
the model memorized -- adjacent windows are near-copies, so it will look
excellent and mean nothing. analysis/evaluate.py holds out whole cycles, and
that is the only number worth quoting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.label import LABELS, RECORDINGS, label_windows, load_labels, load_recording  # noqa: E402
from pipeline import calibrate, model  # noqa: E402
from pipeline.features import FEATURE_NAMES, extract_sequence  # noqa: E402
from server.db import SessionLocal  # noqa: E402
from server.models import Machine  # noqa: E402

# Phases the classifier is asked to distinguish. `done` is absent by design:
# it is a still washer and sounds exactly like idle, so it is detected as a
# transition in pipeline/hmm.py rather than heard. See analysis/label.py.
TRAIN_PHASES = ("idle", "fill", "wash", "rinse", "spin")


class CycleData(NamedTuple):
    """One labelled recording, ready to train on."""

    cycle_id: int
    machine_id: str
    name: str
    X: np.ndarray          # (n, 15) raw features
    y: np.ndarray          # (n,) phase strings, "" where no span covers the window
    times: np.ndarray      # (n,) window centres, epoch seconds

    @property
    def labelled(self) -> np.ndarray:
        """Mask of windows that have a usable label. An open cycle's tail is
        deliberately unlabelled and must not be trained on."""
        return np.isin(self.y, TRAIN_PHASES)


def load_dataset(machine_id: Optional[str] = None) -> list[CycleData]:
    """Every label file in analysis/labels/, joined to its recording.

    A missing recording is a warning, not an error: analysis/labels/ is tracked
    in git and analysis/recordings/ is not, so anyone who clones this repo has
    the labels and none of the data. Failing hard there would make the repo
    look broken rather than empty.
    """
    out: list[CycleData] = []
    for label_path in sorted(LABELS.glob("*.json")):
        label = load_labels(label_path)
        if machine_id and label["machine"] != machine_id:
            continue
        recording = RECORDINGS / label["file"]
        if not recording.exists():
            print(f"  ! {label['file']} is missing from analysis/recordings/ -- skipped")
            continue

        windows = load_recording(recording)
        X, times = extract_sequence(windows)
        y = np.array([lab or "" for lab in label_windows(windows, label["phases"])])
        out.append(CycleData(label["cycle_id"], label["machine"], label["file"], X, y, times))
    return out


def build_calibration(dataset: list[CycleData]) -> calibrate.Baseline:
    """Learn one baseline from the idle windows of every cycle.

    Pooled across cycles rather than fitted per cycle: the baseline is supposed
    to describe the MACHINE, and a per-cycle baseline would quietly absorb the
    thing it is meant to expose -- a mount that shifted between washes would be
    normalized away instead of showing up as a drift.
    """
    idle = np.vstack([c.X[c.y == "idle"] for c in dataset if np.any(c.y == "idle")])
    return calibrate.learn_baseline(idle)


def store_calibration(machine_id: str, baseline: calibrate.Baseline) -> bool:
    """Write the baseline to machines.calibration so the server can z-score
    live windows with exactly the numbers the model was trained against."""
    with SessionLocal() as session:
        machine = session.get(Machine, machine_id)
        if machine is None:
            return False
        machine.calibration = calibrate.serialize(baseline)
        session.commit()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit models/clf-v1.joblib from labelled recordings.")
    ap.add_argument("--machine", default="washer-01")
    ap.add_argument("--out", type=Path, default=model.DEFAULT_PATH)
    ap.add_argument("--no-db", action="store_true",
                    help="do not write the calibration to machines.calibration")
    ap.add_argument("--unbalanced", action="store_true",
                    help="drop class weighting (the set is ~a third idle; you probably don't want this)")
    args = ap.parse_args()

    print(f"reading labels from {LABELS.relative_to(REPO_ROOT)}/")
    dataset = load_dataset(args.machine)
    if not dataset:
        print("no labelled recordings found -- run analysis/label.py first")
        return

    print(f"\n{'cycle':>6} {'windows':>8} {'labelled':>9}  distribution")
    for c in dataset:
        counts = {p: int(np.sum(c.y[c.labelled] == p)) for p in TRAIN_PHASES}
        print(f"{c.cycle_id:>6} {len(c.y):>8} {int(c.labelled.sum()):>9}  "
              + " ".join(f"{p}={n}" for p, n in counts.items() if n))

    baseline = build_calibration(dataset)
    X_all = np.vstack([c.X for c in dataset])
    warnings = calibrate.check_baseline(baseline, X_all)
    print(f"\ncalibration: {baseline}")
    for w in warnings:
        print(f"  ! {w}")
    if not warnings:
        print("  no warnings")

    X = np.vstack([calibrate.zscore(c.X[c.labelled], baseline) for c in dataset])
    y = np.concatenate([c.y[c.labelled] for c in dataset])

    counts = {p: int(np.sum(y == p)) for p in TRAIN_PHASES}
    rarest, n_rare = min(counts.items(), key=lambda kv: kv[1])
    print(f"\ntraining on {len(y)} windows from {len(dataset)} cycles")
    print("  " + "  ".join(f"{p}={n}" for p, n in counts.items()))
    if n_rare < 100:
        # Not fatal, but it decides how much to believe the per-class numbers
        # evaluate.py prints later.
        print(f"  ! {rarest} has only {n_rare} examples; treat its recall as provisional")

    t0 = time.time()
    clf = model.train(X, y, balanced=not args.unbalanced)
    elapsed = time.time() - t0

    stored = (not args.no_db) and store_calibration(args.machine, baseline)
    path = model.save(clf, args.out, meta={
        "machine": args.machine,
        "cycles": [c.cycle_id for c in dataset],
        "recordings": [c.name for c in dataset],
        "n_windows": int(len(y)),
        "class_counts": counts,
        "balanced": not args.unbalanced,
        "calibration_n": baseline.n,
        "calibration_stored": stored,
        "trained_at": time.time(),
    })

    train_acc = float(np.mean(model.predict(clf, X) == y))
    print(f"\nfitted in {elapsed:.2f}s -> {path.relative_to(REPO_ROOT)}")
    print(f"calibration {'written to machines.calibration' if stored else 'NOT stored'}"
          + ("" if stored else f" (no machine row for {args.machine!r})"))
    print(f"training accuracy {train_acc:.3f} -- memorization, not skill. "
          "Run analysis/evaluate.py for a number worth quoting.")


if __name__ == "__main__":
    main()
