"""Score what the server actually said, live, against what you marked afterwards.

    server predicts, window by window  ->  you mark the panel  ->  this compares

This is the strongest test in the repo, and the only one that is blind by
construction. analysis/evaluate.py holds cycles out of training, which is
honest but retrospective: the code doing the scoring already knows how the wash
ended. Here the predictions were written to disk BEFORE anyone marked anything,
by the online filter, with no lookahead and no second chance. Nothing about the
cycle can leak backwards into the model, because the model was already trained
and the file was already written.

What it needs:
  1. a cycle recorded with the model loaded (server/main.py logs every
     prediction to analysis/predictions/ automatically), and
  2. the phase marks, tapped as usual.

DO NOT RETRAIN BEFORE SCORING. The moment analysis/train.py runs with this
cycle in the labels directory, the model has seen it and the result stops
meaning anything. Score first, then retrain -- and the retrained model gets
scored on the NEXT wash.

Run:  venv/bin/python analysis/blind.py 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402

from analysis.label import label_windows, spans_from_marks  # noqa: E402
from server.db import SessionLocal  # noqa: E402
from server.models import Cycle, Mark  # noqa: E402

PREDICTIONS = Path(os.environ.get("LAUNDRY_PREDICTIONS") or REPO_ROOT / "analysis" / "predictions")
PHASES = ("idle", "fill", "wash", "rinse", "spin")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Read a prediction log. The last line may be half-written -- the server
    is appending to this file while you read it."""
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    return sorted(rows, key=lambda r: r["t"])


def first_done(rows: list[dict[str, Any]]) -> Optional[float]:
    """When the running system decided the wash was over: the first return to
    idle after any spin. Exactly the rule pipeline/hmm.py uses, applied to what
    was actually said at the time rather than to a re-run."""
    seen_spin = False
    for r in rows:
        if r["phase"] == "spin":
            seen_spin = True
        elif seen_spin and r["phase"] == "idle":
            return float(r["tc"])
    return None


def clock(t: Optional[float]) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) if t else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cycle_id", type=int)
    args = ap.parse_args()

    with SessionLocal() as session:
        cycle = session.get(Cycle, args.cycle_id)
        if cycle is None:
            print(f"no cycle {args.cycle_id}")
            return
        marks = [(m.t, m.phase) for m in
                 session.query(Mark).filter(Mark.cycle_id == args.cycle_id).order_by(Mark.t)]
        recording, ended_at = cycle.recording, cycle.ended_at

    if not recording:
        print(f"cycle {args.cycle_id} has no recording name")
        return
    path = PREDICTIONS / (Path(recording).stem + ".pred.jsonl")
    if not path.exists():
        print(f"no prediction log at {path}\n"
              "The server writes one only when a model is loaded -- was it running?")
        return

    rows = load_predictions(path)
    if not rows:
        print("prediction log is empty")
        return
    if not marks:
        print("no marks on this cycle; nothing to score against")
        return

    # The same span construction analysis/label.py uses, so "what you marked"
    # means exactly the same thing here as it does in training.
    spans, done_at, unlabelled_from = spans_from_marks(
        marks, first_t=rows[0]["tc"], last_t=rows[-1]["tc"], ended_at=ended_at,
    )
    fake_windows = [{"t": r["tc"], "n": 0, "hz": 1.0} for r in rows]  # centres already
    truth = label_windows(fake_windows, spans)

    pred = np.array([r["phase"] for r in rows])
    true = np.array([t or "" for t in truth])
    conf = np.array([r["conf"] for r in rows])
    m = np.isin(true, PHASES)

    correct = pred[m] == true[m]
    print(f"\n=== cycle {args.cycle_id}  {recording}")
    print(f"    {len(rows)} live predictions, {int(m.sum())} of them inside a marked span")
    print(f"    predictions written {clock(rows[0]['t'])} -> {clock(rows[-1]['t'])}")

    print(f"\n    PER-WINDOW ACCURACY  {correct.mean():.3f}")
    print(f"\n    {'phase':>7}{'windows':>9}{'recall':>8}{'mean conf':>11}")
    for ph in PHASES:
        k = (true == ph) & m
        if not k.sum():
            continue
        print(f"    {ph:>7}{int(k.sum()):>9}{np.mean(pred[k] == ph):>8.2f}{np.mean(conf[k]):>11.2f}")

    # Confidence when right against confidence when wrong. A model that is
    # unsure exactly where it is wrong is usable; one that is certain and wrong
    # is the dangerous kind, and the two look identical in an accuracy number.
    if correct.any() and (~correct).any():
        print(f"\n    mean confidence when right {conf[m][correct].mean():.2f}, "
              f"when wrong {conf[m][~correct].mean():.2f}")

    truth_end = done_at or ended_at
    said = first_done(rows)
    print("\n    CYCLE END")
    print(f"    {'you marked done':<22}{clock(truth_end)}")
    print(f"    {'the server said done':<22}{clock(said)}")
    if said and truth_end:
        err = said - truth_end
        print(f"    {'error':<22}{err:+.0f} s"
              + ("  (early -- as every method here is; the mark is a human walking over)"
                 if err < 0 else ""))
    elif truth_end and not said:
        print("    the server never called it finished -- that is the failure that matters")

    print("\n    WHAT THE SERVER SAID, AS IT SAID IT")
    last = None
    for r in rows:
        if r["phase"] != last:
            print(f"      {clock(r['tc'])}  {r['phase']:<6} (confidence {r['conf']:.2f})")
            last = r["phase"]
    print("\n    WHAT YOU MARKED")
    for t, ph in marks:
        print(f"      {clock(t)}  {ph}")
    if unlabelled_from:
        print(f"\n    ! cycle still open; {clock(unlabelled_from)} onwards is unscored")


if __name__ == "__main__":
    main()
