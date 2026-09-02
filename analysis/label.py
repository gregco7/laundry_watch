"""Turn a cycle's phase marks into a label file -- and let a human check it.

The marks are already in SQLite: you tapped Fill / Wash / Rinse / Spin / Done on
the phone and the SERVER stamped each one, on the same clock it stamps windows
with. So there is nothing to click here. The job is a join by time, plus enough
of a picture that the join can be eyeballed before ten cycles are trained on it.

    marks (points in time)  x  windows (a stream)  ->  labels/<recording>.json

What the join has to get right, and what each rule is protecting against:

  Pre-roll is idle.  A cycle's file starts up to RING_SECONDS BEFORE the row's
  started_at, because tapping Start flushes the ring buffer. Those windows are
  real and they are almost always a still washer. Calling them idle is an
  assumption, not a fact, so the summary prints their count and median RMS --
  idle reads ~34 counts here, fill ~46. If that first row comes back at 150 you
  tapped Record mid-wash and should pass --no-preroll.

  The end boundary is the `done` mark if there is one, else ended_at.  Cycle 1
  has no done mark; cycle 2 has one AND an ended_at three minutes later (known
  bug #5 stranded the End button). Trusting ended_at blindly would label three
  minutes of an empty room as `spin` in one cycle and not the other -- an
  inconsistency the classifier would happily learn.

  Everything after that boundary is `idle`, not `done`.  `done` is acoustically
  a still washer; there is nothing in the vibration to separate it from idle and
  a class that cannot be heard poisons the ones that can. It survives as
  `done_at` at the top of the label file, which is the ground truth
  analysis/evaluate.py measures cycle-end timing error against -- the number
  this whole project exists to produce.

  An OPEN cycle's tail is unlabelled, not idle and not the last mark.  While
  ended_at IS NULL the machine is still running; the windows after the final
  mark could be more spin or could be a finished wash nobody has tapped Done on.
  Guessing either way manufactures training data. They are left out of every
  span and train.py skips windows no span covers. Re-run this after tapping End.

Deliberately NOT done here: nudging boundaries to match the audio. The `spin`
mark lands ~3 minutes before the drum actually ramps, in both recorded cycles,
so a spin span opens with a few minutes of rinse agitation. That bias is
consistent, which is worth more than being right -- the model sees the same
offset every time and the duration prior absorbs it. "Correcting" it by hand
would be inventing data, and inventing it differently in each cycle.

Run:
    python analysis/label.py                 # list cycles and what they'd yield
    python analysis/label.py 1 2 --plot      # write labels, render a PNG to check
    python analysis/label.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# analysis/ is run as a script, so sys.path[0] is analysis/ and `import server`
# would fail. Prepending the repo root is what makes this runnable as
# `python analysis/label.py` from anywhere, which is how it will be run.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server.db import SessionLocal  # noqa: E402
from server.models import Cycle, Mark  # noqa: E402
from server.schemas import PHASES  # noqa: E402

# Same two directories server/main.py uses, spelled out again rather than
# imported from it: importing server.main constructs the FastAPI app and mounts
# the static dashboard, which is a lot of side effect for one Path.
import os  # noqa: E402

RECORDINGS = Path(os.environ.get("LAUNDRY_RECORDINGS") or REPO_ROOT / "analysis" / "recordings")
LABELS = REPO_ROOT / "analysis" / "labels"
PLOTS = REPO_ROOT / "analysis" / "plots"

# Phases that describe a sound. `done` is excluded on purpose (see module
# docstring): it is a moment, stored as done_at, not a span to be classified.
ACOUSTIC_PHASES = tuple(p for p in PHASES if p != "done")


# --------------------------------------------------------------------------
# Reading a recording
# --------------------------------------------------------------------------


def load_recording(path: Path) -> list[dict[str, Any]]:
    """Parse a .jsonl recording into windows, oldest first.

    Two defences, both for things that have actually happened or are known to be
    possible in this repo:

    A truncated final line is normal, not corruption -- this file is being
    appended to by a live server while you read it, and the last write may be
    half a line long. Bailing on it would make the tool unusable during a wash,
    which is exactly when you want it. Any OTHER unparseable line is a real
    problem and raises.

    Windows are sorted by t and de-duplicated by seq because the pre-roll flush
    is not atomic (known bug #3): server/main.py snapshots the ring buffer under
    the lock and writes it outside, so a window arriving in that gap can land
    twice or out of order. Sorting here means downstream code never has to care,
    and the counts are reported so a silent duplicate does not stay silent.
    """
    raw: list[dict[str, Any]] = []
    with path.open() as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # live append caught mid-write
            raise

    seen: set[int] = set()
    windows = []
    for w in sorted(raw, key=lambda w: w["t"]):
        if w["seq"] in seen:
            continue
        seen.add(w["seq"])
        windows.append(w)
    return windows


def window_center(w: dict[str, Any]) -> float:
    """The time a window is ABOUT, not the time it arrived.

    `t` is stamped by the server when the POST lands, i.e. after all n samples
    were taken, so a window nominally covers [t - n/hz, t]. Labelling by t alone
    puts every window ~1.3 s later than the sound it contains. Small next to a
    3-minute mark bias, but it costs one line and it is the kind of constant
    offset that is impossible to find later.
    """
    return w["t"] - w["n"] / (2.0 * w["hz"])


def window_rms(w: dict[str, Any], axis: str = "z") -> float:
    """Mean-removed RMS in raw counts -- the same number tools/strip.py plots and
    the same one the phase table in the notes is written in, so they compare."""
    s = np.asarray(w[axis], dtype=float)
    return float(np.sqrt(np.mean((s - s.mean()) ** 2)))


def recording_health(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Gaps and drops, so a recording that lost its fill phase says so.

    Cycle 1 lost 5.2 minutes -- its entire fill -- to a blocking POST on a dead
    WiFi link, and that was only discovered by looking. A gap is invisible in a
    label file: the spans still tile the timeline, they just have no data under
    part of one.
    """
    if not windows:
        return {"windows": 0}
    ts = np.array([w["t"] for w in windows])
    seqs = np.array([w["seq"] for w in windows])
    dt = np.diff(ts)
    gaps = [(float(ts[i]), float(dt[i])) for i in np.where(dt > 10.0)[0]]
    return {
        "windows": len(windows),
        "duration_s": float(ts[-1] - ts[0]),
        # seq is the node's own counter: the difference between how many it took
        # and how many arrived is windows lost in flight, which arrival times
        # alone cannot distinguish from a slow sample loop.
        "missing": int(seqs[-1] - seqs[0] + 1 - len(seqs)),
        "gaps_over_10s": gaps,
        "hz_median": float(np.median([w["hz"] for w in windows])),
        "late_total": int(sum(w["late"] for w in windows)),
    }


# --------------------------------------------------------------------------
# Marks -> spans
# --------------------------------------------------------------------------


def spans_from_marks(
    marks: list[tuple[float, str]],
    *,
    first_t: float,
    last_t: float,
    ended_at: Optional[float],
    preroll_idle: bool = True,
) -> tuple[list[dict[str, Any]], Optional[float], Optional[float]]:
    """Build phase spans for one cycle. Returns (spans, done_at, unlabelled_from).

    `marks` is (t, phase) in time order. All the boundary reasoning from the
    module docstring lives here and nowhere else.
    """
    done_at = next((t for t, p in reversed(marks) if p == "done"), None)
    phase_marks = [(t, p) for t, p in marks if p != "done"]

    # done wins over ended_at; open cycles have neither.
    end_boundary = done_at if done_at is not None else ended_at

    spans: list[dict[str, Any]] = []
    unlabelled_from: Optional[float] = None

    if not phase_marks:
        # Nothing marked. Either a cycle opened by accident or one where the
        # phone never reached the server -- emit nothing rather than one giant
        # idle span over a real wash.
        return spans, done_at, first_t

    if preroll_idle and first_t < phase_marks[0][0]:
        spans.append({"start": first_t, "end": phase_marks[0][0], "label": "idle"})

    for i, (t, phase) in enumerate(phase_marks):
        if i + 1 < len(phase_marks):
            spans.append({"start": t, "end": phase_marks[i + 1][0], "label": phase})
        elif end_boundary is not None:
            spans.append({"start": t, "end": end_boundary, "label": phase})
        else:
            # Open cycle: the final mark has no end. Everything from it onward is
            # unknown -- see the docstring. Left out of the spans entirely.
            unlabelled_from = t

    if end_boundary is not None and end_boundary < last_t:
        spans.append({"start": end_boundary, "end": last_t, "label": "idle"})

    return spans, done_at, unlabelled_from


def label_windows(
    windows: list[dict[str, Any]], spans: list[dict[str, Any]]
) -> list[Optional[str]]:
    """One label per window, or None where no span covers it.

    Half-open [start, end) so a mark's own instant belongs to the phase it
    starts, and no window can be claimed by two adjacent spans.
    """
    if not spans:
        return [None] * len(windows)
    starts = np.array([s["start"] for s in spans])
    ends = np.array([s["end"] for s in spans])
    labels = [s["label"] for s in spans]

    out: list[Optional[str]] = []
    for w in windows:
        c = window_center(w)
        hit = np.where((starts <= c) & (c < ends))[0]
        out.append(labels[hit[0]] if len(hit) else None)
    return out


# --------------------------------------------------------------------------
# Writing + reading label files
# --------------------------------------------------------------------------


def write_labels(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_labels(path: Path) -> dict[str, Any]:
    """Used by analysis/train.py. Here so the file format has exactly one
    reader and one writer."""
    return json.loads(Path(path).read_text())


def build_payload(cycle: Cycle, marks: list[tuple[float, str]], windows: list[dict[str, Any]],
                  *, preroll_idle: bool = True) -> dict[str, Any]:
    # The outer edges of the data, not the centres of the edge windows: spans
    # are half-open [start, end), so ending the last one at the last window's
    # centre leaves that window uncovered -- one silently dropped training row
    # per cycle, at the exact moment the wash finishes.
    first, last = windows[0], windows[-1]
    first_t = first["t"] - first["n"] / first["hz"]
    last_t = last["t"]
    spans, done_at, unlabelled_from = spans_from_marks(
        marks, first_t=first_t, last_t=last_t,
        ended_at=cycle.ended_at, preroll_idle=preroll_idle,
    )
    return {
        # The documented shape (CLAUDE.md "Data shapes") is file/machine/phases.
        # Everything else is additive: evaluate.py needs done_at, and train.py
        # needs to know an open cycle's tail is missing rather than empty.
        "file": cycle.recording,
        "machine": cycle.machine_id,
        "cycle_id": cycle.id,
        "started_at": cycle.started_at,
        "ended_at": cycle.ended_at,
        "done_at": done_at,
        "open": cycle.ended_at is None,
        "unlabelled_from": unlabelled_from,
        "generated_at": time.time(),
        "phases": spans,
    }


# --------------------------------------------------------------------------
# The eyeball check
# --------------------------------------------------------------------------


def plot_cycle(windows: list[dict[str, Any]], payload: dict[str, Any], out_path: Path,
               axis: str = "z") -> Path:
    """Spectrogram + amplitude with the spans drawn on top.

    This is the checkpoint: if the marks and the windows are joined correctly,
    the shaded boxes line up with the places the picture visibly changes. If
    they are offset by minutes, every label made today is wrong the same way and
    it is worth knowing at 9am.

    Agg backend, PNG out -- an interactive window is one more thing to close on
    a laptop that is also running the server that is recording the wash.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = windows[0]["t"]
    mins = np.array([(w["t"] - t0) / 60.0 for w in windows])
    hz = float(np.median([w["hz"] for w in windows]))

    # Spectrogram straight from the raw windows: mean-removed (gravity is a
    # constant ~17000 counts and its DC bin would be the only thing visible),
    # Hann-tapered (a hard window edge smears one tone across every bin).
    n = windows[0]["n"]
    taper = np.hanning(n)
    mags = []
    for w in windows:
        s = np.asarray(w[axis], dtype=float)
        s = (s - s.mean()) * taper
        mags.append(np.abs(np.fft.rfft(s)))
    spec = 20 * np.log10(np.array(mags) + 1.0)
    freqs = np.fft.rfftfreq(n, 1.0 / hz)

    rms = np.array([window_rms(w, axis) for w in windows])
    labels = label_windows(windows, payload["phases"])

    colors = {"idle": "#94a3b8", "fill": "#38bdf8", "wash": "#22c55e",
              "rinse": "#f59e0b", "spin": "#ef4444"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.pcolormesh(mins, freqs, spec.T, shading="nearest", cmap="magma")
    ax1.set_ylabel(f"Hz  ({axis}-axis)")
    ax1.set_title(f"{payload['file']}  --  cycle {payload['cycle_id']}"
                  + ("  [OPEN, still recording]" if payload["open"] else ""))

    # Log amplitude: idle 34 and spin 572 are a factor of 17 apart, and on a
    # linear axis everything below spin is a flat line at the bottom.
    ax2.semilogy(mins, np.maximum(rms, 1.0), lw=0.8, color="#111827")
    ax2.set_ylabel(f"RMS counts ({axis})")
    ax2.set_xlabel(f"minutes since {time.strftime('%H:%M:%S', time.localtime(t0))}")

    for span in payload["phases"]:
        a, b = (span["start"] - t0) / 60.0, (span["end"] - t0) / 60.0
        for ax in (ax1, ax2):
            ax.axvspan(a, b, color=colors.get(span["label"], "#000000"), alpha=0.18)
            ax.axvline(a, color=colors.get(span["label"], "#000000"), lw=1.2)
        ax2.text(a, ax2.get_ylim()[1], f" {span['label']}", va="top", fontsize=9,
                 color=colors.get(span["label"], "#000000"))

    if payload["done_at"]:
        for ax in (ax1, ax2):
            ax.axvline((payload["done_at"] - t0) / 60.0, color="k", ls="--", lw=1.2)
    if payload["unlabelled_from"]:
        for ax in (ax1, ax2):
            ax.axvspan((payload["unlabelled_from"] - t0) / 60.0, mins[-1],
                       facecolor="none", hatch="//", edgecolor="#6b7280", lw=0)

    # Sanity check on the join itself, printed where you cannot miss it.
    covered = sum(1 for x in labels if x is not None)
    fig.text(0.01, 0.01, f"{covered}/{len(labels)} windows labelled", fontsize=8, color="#6b7280")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _clock(t: Optional[float]) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) if t else "-"


def process(cycle_id: int, *, plot: bool, preroll_idle: bool, write: bool) -> None:
    with SessionLocal() as session:
        cycle = session.get(Cycle, cycle_id)
        if cycle is None:
            print(f"no cycle {cycle_id}")
            return
        marks = [(m.t, m.phase) for m in
                 session.query(Mark).filter(Mark.cycle_id == cycle_id).order_by(Mark.t)]
        recording = cycle.recording
        machine_id, started, ended = cycle.machine_id, cycle.started_at, cycle.ended_at

    if not recording:
        print(f"cycle {cycle_id}: no recording file on the row")
        return
    path = RECORDINGS / recording
    if not path.exists():
        print(f"cycle {cycle_id}: {path} is missing")
        return

    windows = load_recording(path)
    if not windows:
        print(f"cycle {cycle_id}: {recording} has no windows")
        return

    health = recording_health(windows)
    with SessionLocal() as session:
        payload = build_payload(session.get(Cycle, cycle_id), marks, windows,
                                preroll_idle=preroll_idle)
    labels = label_windows(windows, payload["phases"])

    print(f"\n=== cycle {cycle_id}  {machine_id}  {recording}")
    print(f"    {_clock(started)} -> {_clock(ended) if ended else 'OPEN'}   "
          f"{health['windows']} windows, {health['duration_s'] / 60:.1f} min, "
          f"{health['hz_median']:.1f} Hz, {health['missing']} missing, "
          f"{health['late_total']} late")
    for at, dur in health["gaps_over_10s"]:
        print(f"    !! gap of {dur:.0f}s at {_clock(at)}"
              + ("  <-- longer than a phase" if dur > 120 else ""))

    print(f"    {'phase':6} {'from':>9} {'min':>6} {'windows':>8} {'medRMS':>7}")
    for span in payload["phases"]:
        idx = [i for i, w in enumerate(windows)
               if span["start"] <= window_center(w) < span["end"]]
        med = np.median([window_rms(windows[i]) for i in idx]) if idx else float("nan")
        print(f"    {span['label']:6} {_clock(span['start']):>9} "
              f"{(span['end'] - span['start']) / 60:>6.1f} {len(idx):>8} {med:>7.0f}")
    if payload["done_at"]:
        print(f"    done_at {_clock(payload['done_at'])}"
              + ("  (ended_at is later; the mark wins)" if ended and ended > payload["done_at"] + 1 else ""))
    if payload["unlabelled_from"]:
        n = sum(1 for x in labels if x is None)
        print(f"    {n} windows from {_clock(payload['unlabelled_from'])} left UNLABELLED "
              f"(cycle still open -- re-run after End)")

    if write:
        out = LABELS / (path.stem + ".json")
        write_labels(payload, out)
        print(f"    -> {out.relative_to(REPO_ROOT)}")
    if plot:
        png = plot_cycle(windows, payload, PLOTS / (path.stem + ".png"))
        print(f"    -> {png.relative_to(REPO_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cycles", nargs="*", type=int, help="cycle ids; omit to list")
    ap.add_argument("--all", action="store_true", help="every cycle in the database")
    ap.add_argument("--plot", action="store_true", help="also render analysis/plots/<name>.png")
    ap.add_argument("--no-preroll", action="store_true",
                    help="do not label the pre-Start ring-buffer windows as idle")
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    args = ap.parse_args()

    ids = args.cycles
    if args.all or not ids:
        with SessionLocal() as session:
            rows = [(c.id, c.machine_id, c.started_at, c.ended_at, c.recording)
                    for c in session.query(Cycle).order_by(Cycle.id)]
        if not args.all:
            print(f"{'id':>3} {'machine':10} {'started':>9} {'ended':>9}  recording")
            for cid, mid, st, en, rec in rows:
                print(f"{cid:>3} {mid:10} {_clock(st):>9} {_clock(en) if en else '     OPEN':>9}  {rec}")
            print("\npass cycle ids to write labels, e.g.  python analysis/label.py 1 2 --plot")
            return
        ids = [r[0] for r in rows]

    for cid in ids:
        process(cid, plot=args.plot, preroll_idle=not args.no_preroll, write=not args.dry_run)


if __name__ == "__main__":
    main()
