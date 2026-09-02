"""Leave-one-cycle-out evaluation. The only numbers in this repo worth quoting.

    for each recorded cycle:
        train on the OTHERS, predict this one, score it

NON-NEGOTIABLE, AND THE REASON THIS FILE EXISTS: whole cycles are held out,
never random windows. Windows overlap in content and adjacent ones are close
to copies of each other, so a random 80/20 split puts near-duplicates of every
test window in the training set. The accuracy that comes back is ~99% and
describes nothing except the model's ability to recognize a window it has
already seen. With three cycles there is no room for a fixed holdout either,
so every cycle takes a turn as the test set.

THE CALIBRATION IS ALSO HELD OUT. Fitting the baseline on all cycles and then
testing on one of them leaks the test cycle's own idle windows into the
normalization -- a small leak, and exactly the kind nobody notices. Each fold
learns its baseline from its training cycles only.

FOUR THINGS ARE SCORED, because "the model gets 84%" is meaningless alone:

    prior     a control, not a method: the HMM run on UNIFORM emissions, so
              it hears nothing at all and simply plays back the average
              schedule. Its score is the floor that any real result has to
              clear. On cycles this repetitive it is embarrassingly high, and
              knowing that is the difference between a measurement and a
              flattering number.
    rules     the hand-written baseline. If the model cannot beat this, the
              model is not worth its joblib file.
    raw       the classifier, per window, no smoothing. Expect this to look
              bad on the loud segments and do not treat that as failure --
              a drain spin IS a spin to anything that sees 2.56 seconds.
    viterbi   the classifier plus the sequence prior, offline, whole recording.
    online    the same prior run causally, one window at a time -- what the
              server can actually do live. The gap between it and viterbi is
              the price of not being able to see the future.

TWO METRICS, and the second one is the point:

    per-window accuracy   how often the phase label is right.
    cycle-end error       how many seconds off the finish time is. This is the
                          number the project exists to shrink: nobody wants to
                          know it is rinsing, they want to know it has stopped.

Run:  venv/bin/python analysis/evaluate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.train import TRAIN_PHASES, CycleData, build_calibration, load_dataset  # noqa: E402
from pipeline import calibrate, hmm, model, rules  # noqa: E402

# Rules speak a slightly larger vocabulary than the labels do: they emit `done`
# where a label file says `idle`, because analysis/label.py deliberately keeps
# `done` out of the label set (it is a silence, not a sound). Folding it back
# here compares like with like instead of scoring a vocabulary difference as an
# error -- which cost the baseline 20 points on cycle 3 before it was noticed.
FOLD = {"done": "idle"}


def fold(labels: np.ndarray) -> np.ndarray:
    return np.array([FOLD.get(p, p) for p in labels])


def truth_end(cycle: CycleData, label_meta: dict) -> Optional[float]:
    """The moment the wash really finished: the `done` mark if there is one,
    otherwise ended_at. Cycle 1 has no done mark, cycle 2 has both."""
    return label_meta.get("done_at") or label_meta.get("ended_at")


def confusion(y_true: np.ndarray, y_pred: np.ndarray, phases=TRAIN_PHASES) -> np.ndarray:
    idx = {p: i for i, p in enumerate(phases)}
    M = np.zeros((len(phases), len(phases)), dtype=int)
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    return M


def print_confusion(M: np.ndarray, phases=TRAIN_PHASES) -> None:
    print(f"\n{'true \\ pred':>12}" + "".join(f"{p:>8}" for p in phases) + f"{'recall':>9}")
    for i, p in enumerate(phases):
        row = M[i]
        rec = row[i] / row.sum() if row.sum() else float("nan")
        print(f"{p:>12}" + "".join(f"{v:>8}" for v in row) + f"{rec:>9.2f}")
    prec = [M[:, j][j] / M[:, j].sum() if M[:, j].sum() else float("nan") for j in range(len(phases))]
    print(f"{'precision':>12}" + "".join(f"{v:>8.2f}" for v in prec))


def macro_f1(M: np.ndarray) -> float:
    f1s = []
    for i in range(M.shape[0]):
        tp, fp, fn = M[i, i], M[:, i].sum() - M[i, i], M[i].sum() - M[i, i]
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s)) if f1s else float("nan")


def evaluate(dataset: list[CycleData], label_meta: dict[int, dict], power: float) -> None:
    results: dict[str, list[float]] = {k: [] for k in ("prior", "rules", "raw", "viterbi", "online")}
    errors: dict[str, list[Optional[float]]] = {k: [] for k in results}
    matrices = {k: np.zeros((len(TRAIN_PHASES), len(TRAIN_PHASES)), dtype=int) for k in results}

    for held in dataset:
        train_set = [c for c in dataset if c.cycle_id != held.cycle_id]

        # Baseline from the TRAINING cycles only -- see the header.
        baseline = build_calibration(train_set)
        X_tr = np.vstack([calibrate.zscore(c.X[c.labelled], baseline) for c in train_set])
        y_tr = np.concatenate([c.y[c.labelled] for c in train_set])
        clf = model.train(X_tr, y_tr)

        m = held.labelled
        Z = calibrate.zscore(held.X, baseline)
        y_true = held.y[m]
        times = held.times
        end_truth = truth_end(held, label_meta[held.cycle_id])

        # --- the four contenders, all on the same z-scored features
        rule_pred, rule_done = rules.run(Z, times)

        proba_all = model.predict_proba(clf, Z)
        cols = model.class_columns(clf, hmm.STATES)          # by NAME, never by position
        proba_states = proba_all[:, cols]
        raw_pred = np.array([hmm.STATES[i] for i in np.argmax(proba_states, axis=1)])
        vit_pred = hmm.smooth(proba_states, power=power)

        # filt.phase, NOT argmax of the posterior: the reported phase is the
        # belief plus output hysteresis, and the hysteresis is the whole reason
        # the live system does not flip phase on one ambiguous window. Scoring
        # the raw argmax measures a system nobody runs.
        filt = hmm.Filter(power=power)
        online_pred, online_done = [], None
        for row, tw in zip(proba_states, times):
            filt.update(row)
            online_pred.append(filt.phase)
            # The filter's own verdict, at the moment it reached it -- not
            # re-derived afterwards from the smoothed label sequence, which
            # would measure a different (slower) mechanism than the one the
            # server actually runs.
            if filt.finished and online_done is None:
                online_done = float(tw)
        online_pred = np.array(online_pred)

        # The control: same transition matrix, no evidence whatsoever.
        uniform = np.full_like(proba_states, 1.0 / len(hmm.STATES))
        prior_pred = hmm.smooth(uniform, power=power)

        preds = {"prior": prior_pred, "rules": fold(rule_pred), "raw": raw_pred,
                 "viterbi": vit_pred, "online": online_pred}
        dones = {
            "prior": hmm.done_time(prior_pred, times),
            "rules": rule_done,
            "raw": hmm.done_time(raw_pred, times),
            "viterbi": hmm.done_time(vit_pred, times),
            "online": online_done,
        }

        print(f"\n=== cycle {held.cycle_id}  ({int(m.sum())} labelled windows, "
              f"trained on {[c.cycle_id for c in train_set]})")
        print(f"{'method':>9}{'window acc':>12}{'macro F1':>10}{'cycle-end error':>18}")
        for name, pred in preds.items():
            acc = float(np.mean(pred[m] == y_true))
            M = confusion(y_true, pred[m])
            matrices[name] += M
            results[name].append(acc)
            err = (dones[name] - end_truth) if (dones[name] and end_truth) else None
            errors[name].append(err)
            shown = f"{err:+.0f} s" if err is not None else "not detected"
            print(f"{name:>9}{acc:>12.3f}{macro_f1(M):>10.3f}{shown:>18}")

    print("\n\n" + "=" * 64)
    print("POOLED OVER ALL HELD-OUT CYCLES")
    print("=" * 64)
    print(f"{'method':>9}{'mean acc':>10}{'macro F1':>10}{'mean |end error|':>19}{'detected':>10}")
    for name in results:
        errs = [e for e in errors[name] if e is not None]
        mean_err = f"{np.mean(np.abs(errs)):.0f} s" if errs else "-"
        print(f"{name:>9}{np.mean(results[name]):>10.3f}{macro_f1(matrices[name]):>10.3f}"
              f"{mean_err:>19}{f'{len(errs)}/{len(dataset)}':>10}")

    for name in ("raw", "viterbi", "online"):
        print(f"\n--- {name}")
        print_confusion(matrices[name])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--machine", default="washer-01")
    ap.add_argument("--power", type=float, default=hmm.EMISSION_POWER,
                    help="emission tempering; 1.0 trusts the classifier fully")
    args = ap.parse_args()

    dataset = load_dataset(args.machine)
    if len(dataset) < 2:
        print(f"need at least 2 labelled cycles to hold one out; found {len(dataset)}")
        return

    from analysis.label import LABELS, load_labels
    meta = {}
    for p in sorted(LABELS.glob("*.json")):
        d = load_labels(p)
        meta[d["cycle_id"]] = d

    print(f"leave-one-cycle-out over {len(dataset)} cycles, emission power {args.power}")
    evaluate(dataset, meta, args.power)


if __name__ == "__main__":
    main()
