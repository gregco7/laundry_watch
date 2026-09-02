"""FastAPI app -- the local server the node POSTs to and the phone drives.

Two clients, two very different rates:

  ESP32  -> POST /readings          ~1400 windows/hour, forever, never stops
  phone  -> POST /cycles, /mark     a handful of taps per wash

The node is deliberately dumb. It has no idea whether anything is being
recorded; it just ships every window it takes. THIS file decides what to keep.
That split is the whole reliability story: a node holding no state cannot
desync, and a "start recording" command lost to a WiFi blip cannot cost you a
50-minute wash.

How keeping works:

  Every arriving window goes into an in-memory ring buffer holding the last
  RING_SECONDS. Windows older than that are dropped and forgotten. When you
  open a cycle, the buffer is flushed to a JSONL file first -- so tapping Start
  two minutes late still captures the fill you already missed -- and every
  window after that is appended as it arrives.

Cycle boundaries and phase marks go to SQLite. Eight hours of 256-int arrays
do not.

Run:  uvicorn server.main:app --host 0.0.0.0 --reload
      0.0.0.0, not the default -- the default binds to loopback and the ESP32
      cannot reach it.
"""

import json
import os
import threading
import time
import numpy as np

from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from pipeline import calibrate, hmm
from pipeline import model as classifier
from pipeline.features import MAX_CONTEXT_GAP, extract
from server.db import REPO_ROOT, get_session, init_db
from server.models import Cycle, Machine, Mark
from server.schemas import (
    CycleOut,
    CycleStart,
    EmptyIn,
    MachineOut,
    MarkIn,
    MarkOut,
    PhaseEvent,
    ReadingAck,
    ReadingIn,
    StatusOut,
)

# Five minutes of pre-roll. Long enough that a late Start still catches the
# fill, short enough that the buffer is a few megabytes rather than a leak:
# ~117 windows x 768 ints.
RING_SECONDS = 300.0

# What GET /live hands the dashboard. ~60 windows is about 2.5 minutes at
# 100 Hz -- enough of a strip chart to see agitate turn into spin.
LIVE_WINDOWS = 60

# How long the node can go quiet before the screen stops claiming to know what
# the washer is doing. Windows arrive every ~2.6 s, so 20 s is roughly seven
# missed in a row -- past the point where a WiFi retry explains it.
SILENT_AFTER = 20.0

RECORDINGS = Path(os.environ.get("LAUNDRY_RECORDINGS") or REPO_ROOT / "analysis" / "recordings")


# --------------------------------------------------------------------------
# In-memory state
#
# FastAPI runs every non-async endpoint in a threadpool, so two requests really
# can touch these at the same time -- a window arriving while you tap Start is
# exactly the interleaving that matters. One lock guards both the buffer and
# the file append; they are never modified independently.
# --------------------------------------------------------------------------

_lock = threading.Lock()
_buffer: deque[dict[str, Any]] = deque()

# Per-node liveness, so GET /live can answer "is the node actually talking?"
# without a database round trip. seq is the node's own counter: a gap means
# windows were dropped in flight, which is invisible from arrival times alone.
_nodes: dict[str, dict[str, Any]] = {}

# --------------------------------------------------------------------------
# Inference
#
# Loaded once at startup, not per request: unpickling the model takes longer
# than the 2.56 s between windows, so a per-request load would fall behind the
# node and never catch up.
#
# None is a normal state, not an error. Until analysis/train.py has run there
# is no models/clf-v1.joblib, and the status screen already knows how to say
# "this is the last phase a human marked" -- a server that refused to start
# without a model would make the recording tools unusable on a fresh clone,
# which is exactly when you need them.
# --------------------------------------------------------------------------

_MODEL: Optional[tuple[Any, dict[str, Any]]] = None

# Per machine: the online filter and the temporal context the delta features
# need. Kept here rather than in the database because it is a belief about
# right now, worth nothing after a restart -- and rebuilding it from the ring
# buffer on startup would be a second code path that could disagree with this
# one about what the machine is doing.
_infer: dict[str, dict[str, Any]] = {}

# How long a cached calibration is trusted before it is re-read from the
# machines row. Long enough to cost nothing, short enough that retraining
# takes effect without a restart -- analysis/train.py writes a new baseline,
# and a minute later the running server is using it.
CALIBRATION_TTL = 60.0

# How long a model-detected finish keeps the screen on "done" when there is no
# cycle row to acknowledge. Without a cycle there is no "I've taken it out"
# button that does anything, so this is what eventually returns the screen to
# idle instead of leaving a finished wash on it forever.
MODEL_DONE_TTL = 30 * 60.0

# Every prediction the server makes, appended as it is made. Cheap (~150 KB a
# wash) and the only way to score the model HONESTLY against a cycle it has
# never seen: re-running it offline afterwards proves the model works, while
# this proves the running system worked, live, with no lookahead and no second
# chance. Blind by construction -- it is written before anyone marks anything.
PREDICTIONS = Path(os.environ.get("LAUNDRY_PREDICTIONS") or REPO_ROOT / "analysis" / "predictions")


def _trim(now: float) -> None:
    """Drop windows that have aged out. Called under _lock."""
    cutoff = now - RING_SECONDS
    while _buffer and _buffer[0]["t"] < cutoff:
        _buffer.popleft()


def _append_jsonl(path: Path, windows: list[dict[str, Any]]) -> None:
    """Append windows to a cycle's recording. Called under _lock.

    Opened per call rather than held open for the length of a wash: an open
    handle is state that survives a crash badly, and one open() per 2.56 s is
    free next to the 768 ints it writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for w in windows:
            fh.write(json.dumps(w, separators=(",", ":")) + "\n")


def _open_cycle(session: Session, machine_id: str) -> Optional[Cycle]:
    """The one query the ingest path runs per window. Covered by
    ix_cycles_machine_open."""
    return session.scalar(
        select(Cycle).where(Cycle.machine_id == machine_id, Cycle.ended_at.is_(None))
    )


def _reset_inference(machine_id: str) -> None:
    """Start believing nothing in particular again.

    Called when a cycle opens and when the node has been silent long enough
    that the belief is stale. A filter carried across a gap is worse than no
    filter: it reports the phase the machine was in before the outage with
    full confidence, and nothing on the screen distinguishes that from a
    reading taken a second ago.
    """
    _infer[machine_id] = {
        "filter": hmm.Filter(),
        "prev": None,
        "history": [],
        "last_t": None,
        "phase": None,
        "phase_since": None,
        "calibration": None,
        "calibration_at": 0.0,
        "finished_at": None,
    }


def _correct_inference(machine_id: str, phase: str, t: float) -> bool:
    """Push a human's mark into the running filter.

    This is the whole point of the Correct button. Without it a mark is a row
    in a table that the live system never reads -- which is exactly what
    happened on 2026-09-01, when the screen sat on the wrong phase through
    sixteen taps saying otherwise.
    """
    state = _infer.get(machine_id)
    if state is None or state.get("filter") is None:
        return False

    filt = state["filter"]
    if phase == "done":
        # Not a state the filter carries; it is the end of the cycle.
        state["finished_at"] = t
        return True

    filt.correct(phase)
    state["phase"], state["phase_since"] = filt.phase, t
    # A correction to an earlier phase un-finishes the wash: the human is
    # saying it is still going.
    state["finished_at"] = None
    return True


def _calibration(session: Session, machine_id: str, state: dict[str, Any], now: float):
    """The machine's stored baseline, cached for CALIBRATION_TTL seconds."""
    if state["calibration"] is None or now - state["calibration_at"] > CALIBRATION_TTL:
        machine = session.get(Machine, machine_id)
        state["calibration"] = calibrate.deserialize(machine.calibration if machine else None)
        state["calibration_at"] = now
    return state["calibration"]


def _log_prediction(node: str, cycle: Optional[Cycle], row: dict[str, Any]) -> None:
    """Append one prediction. Grouped by recording when a cycle is open so the
    log and the windows it describes stay together, and by day otherwise."""
    if cycle is not None and cycle.recording:
        name = Path(cycle.recording).stem + ".pred.jsonl"
    else:
        name = f"{node}-{time.strftime('%Y%m%d', time.localtime(row['t']))}.pred.jsonl"
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    with (PREDICTIONS / name).open("a") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def _infer_window(session: Session, reading: ReadingIn, t: float,
                  cycle: Optional[Cycle] = None) -> None:
    """Run one window through features -> z-score -> classifier -> filter.

    Everything here is best-effort: an exception on this path must not cost a
    window of the recording, which is unrepeatable, while a prediction is
    recomputed 23 times a minute. So the caller wraps it and the failure mode
    is a stale phase on the screen rather than a 500 at the node.
    """
    if _MODEL is None:
        return
    state = _infer.get(reading.node) or _infer.setdefault(reading.node, {})
    if not state:
        _reset_inference(reading.node)
        state = _infer[reading.node]

    baseline = _calibration(session, reading.node, state, t)
    if baseline is None:
        return  # uncalibrated machine: run analysis/train.py

    last_t = state["last_t"]
    if last_t is not None and t - last_t > SILENT_AFTER:
        # A hole big enough that the belief is about a different situation.
        _reset_inference(reading.node)
        state = _infer[reading.node]
    elif last_t is not None and t - last_t > MAX_CONTEXT_GAP:
        # Smaller hole: the deltas would describe a change that never happened,
        # but what the machine is doing has probably not changed. Same rule as
        # pipeline/features.extract_sequence, for the same reason.
        state["prev"], state["history"] = None, []

    frame = np.array([reading.x, reading.y, reading.z], dtype=float)
    vec, context = extract(frame, reading.hz, prev=state["prev"], history=state["history"])

    clf, _bundle = _MODEL
    proba = classifier.predict_proba(clf, calibrate.zscore(vec, baseline))[0]
    # By NAME. sklearn orders classes alphabetically and hmm.STATES is in cycle
    # order; lining them up by position gives a working system that is wrong
    # about which phase it is in.
    filt = state["filter"]
    filt.update(proba[classifier.class_columns(clf, hmm.STATES)])

    if filt.phase != state["phase"]:
        state["phase"], state["phase_since"] = filt.phase, t

    # Latched, not recomputed: the wash finishes once. Without the latch a
    # door slam an hour later would set the machine back to "running" and the
    # screen would un-finish a load somebody already took out.
    if filt.finished and state.get("finished_at") is None:
        state["finished_at"] = t

    state["prev"], state["last_t"] = context, t
    state["history"].append(context[1])
    if len(state["history"]) > 5:
        state["history"].pop(0)

    _log_prediction(reading.node, cycle, {
        "t": t,
        # The window's centre as well as its arrival, because analysis/label.py
        # and pipeline/features.py both work in centres. Storing one and
        # deriving the other later is how a 1.3-second offset gets into a
        # scoring script and never comes out.
        "tc": t - reading.n / (2.0 * reading.hz),
        "seq": reading.seq,
        "phase": filt.phase,
        "conf": round(filt.confidence, 4),
        # The whole posterior, not just the winner: a wrong call at 0.35 and a
        # wrong call at 0.99 are different failures, and only one of them is
        # worth changing the model over.
        "p": {name: round(float(v), 4) for name, v in zip(filt.states, filt.alpha)},
        "cycle_id": cycle.id if cycle is not None else None,
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MODEL
    init_db()
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    _MODEL = classifier.try_load()
    if _MODEL is None:
        print("no model at models/clf-v1.joblib -- /status will report marks, not predictions")
    else:
        meta = _MODEL[1].get("meta", {})
        print(f"model loaded: trained on cycles {meta.get('cycles')} "
              f"({meta.get('n_windows')} windows)")
    yield


app = FastAPI(title="LaundryWatch", lifespan=lifespan)


# --------------------------------------------------------------------------
# Ingest -- the node's only endpoint
# --------------------------------------------------------------------------


@app.post("/readings", response_model=ReadingAck)
def post_reading(reading: ReadingIn, session: Session = Depends(get_session)) -> ReadingAck:
    """One window from the node.

    The server stamps `t`, not the node: the ESP32 has no RTC, so its idea of
    the time is whatever it booted to. Marks and windows must share one clock
    or every label you draw on day two is offset by an unknown amount.
    """
    t = time.time()
    window = {"t": t, **reading.model_dump()}

    cycle = _open_cycle(session, reading.node)

    with _lock:
        _buffer.append(window)
        _trim(t)

        node = _nodes.setdefault(reading.node, {"dropped": 0, "last_seq": None})
        last = node["last_seq"]
        if last is not None and reading.seq > last + 1:
            # The node numbers every window it takes. A gap means windows were
            # lost between here and there -- a WiFi retry, the laptop asleep.
            # Arrival times alone cannot tell that from a slow sample loop.
            node["dropped"] += reading.seq - last - 1
        node["last_seq"] = reading.seq
        node["last_seen"] = t
        node["hz"] = reading.hz
        node["late"] = reading.late

        if cycle is not None and cycle.recording:
            _append_jsonl(RECORDINGS / cycle.recording, [window])

    # After the lock, not inside it: an FFT and a tree ensemble take a few
    # milliseconds, and holding the ingest lock across them would make every
    # other window and every dashboard poll wait behind a prediction.
    try:
        _infer_window(session, reading, t, cycle)
    except Exception as exc:  # noqa: BLE001 -- see _infer_window's docstring
        # A window is unrepeatable and a prediction is not. Never let inference
        # cost the recording.
        print(f"inference failed for {reading.node}: {exc!r}")

    return ReadingAck(
        ok=True,
        recorded=cycle is not None,
        cycle_id=cycle.id if cycle is not None else None,
    )


# --------------------------------------------------------------------------
# Cycles -- what the phone drives
# --------------------------------------------------------------------------


@app.post("/cycles", response_model=CycleOut, status_code=201)
def start_cycle(body: CycleStart, session: Session = Depends(get_session)) -> Cycle:
    """Open a cycle and flush the pre-roll into its file."""
    if _open_cycle(session, body.machine_id) is not None:
        # Refusing is the safe half of the trade: two open cycles would both
        # match the ingest query, and which one won would depend on row order.
        raise HTTPException(409, f"{body.machine_id} already has an open cycle")

    machine = session.get(Machine, body.machine_id)
    if machine is None:
        # Created on demand. A home project should not need a setup step
        # before it can record, and the node already told us the name.
        machine = Machine(id=body.machine_id, label=body.machine_id)
        session.add(machine)

    started = time.time()
    name = f"{body.machine_id}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}.jsonl"
    cycle = Cycle(
        machine_id=body.machine_id,
        started_at=started,
        recording=name,
        setting=body.setting,
        load_size=body.load_size,
        notes=body.notes,
    )
    session.add(cycle)
    session.commit()

    # A new wash starts from the prior, not from whatever the filter believed
    # about the last one -- otherwise a load started right after a spin begins
    # life convinced the machine is already finishing.
    _reset_inference(body.machine_id)

    # The whole point of the ring buffer: whatever is still in memory is
    # already part of this wash, so it belongs in this file. Press Start two
    # minutes late and you still get the fill.
    with _lock:
        _trim(time.time())
        preroll = list(_buffer)
    if preroll:
        _append_jsonl(RECORDINGS / name, preroll)

    return cycle


@app.post("/cycles/{cycle_id}/end", response_model=CycleOut)
def end_cycle(cycle_id: int, session: Session = Depends(get_session)) -> Cycle:
    cycle = session.get(Cycle, cycle_id)
    if cycle is None:
        raise HTTPException(404, f"no cycle {cycle_id}")
    if cycle.ended_at is not None:
        raise HTTPException(409, f"cycle {cycle_id} already ended")
    cycle.ended_at = time.time()
    session.commit()
    return cycle


@app.post("/cycles/{cycle_id}/mark", response_model=MarkOut, status_code=201)
def add_mark(cycle_id: int, body: MarkIn, session: Session = Depends(get_session)) -> Mark:
    """Record a phase boundary. Accepted on a closed cycle too -- you will
    notice a missed mark after tapping End, and the alternative is losing it."""
    cycle = session.get(Cycle, cycle_id)
    if cycle is None:
        raise HTTPException(404, f"no cycle {cycle_id}")

    # Default to now rather than trusting the phone's clock: the windows this
    # mark will be joined against are stamped by this same clock.
    mark = Mark(cycle_id=cycle_id, t=body.t if body.t is not None else time.time(), phase=body.phase)
    session.add(mark)
    session.commit()

    # The screen must change the instant you tap, or you cannot tell a mark
    # that landed from one that failed -- and with a model loaded the phase on
    # screen comes from the filter, so recording the mark alone changes nothing
    # visible. Steering the filter is both the correction and the receipt.
    with _lock:
        _correct_inference(cycle.machine_id, mark.phase, mark.t)

    return mark


@app.get("/cycles/{cycle_id}", response_model=CycleOut)
def get_cycle(cycle_id: int, session: Session = Depends(get_session)) -> Cycle:
    cycle = session.get(Cycle, cycle_id)
    if cycle is None:
        raise HTTPException(404, f"no cycle {cycle_id}")
    return cycle


# --------------------------------------------------------------------------
# Read-only views
# --------------------------------------------------------------------------


@app.get("/machines", response_model=list[MachineOut])
def list_machines(session: Session = Depends(get_session)) -> list[MachineOut]:
    out = []
    for machine in session.scalars(select(Machine).order_by(Machine.id)):
        cycle = _open_cycle(session, machine.id)
        out.append(MachineOut.from_machine(machine, open_cycle_id=cycle.id if cycle else None))
    return out


@app.get("/live")
def live() -> dict[str, Any]:
    """The mounting check and the marking instrument, in one payload.

    Returns a per-window amplitude rather than the raw samples -- 60 windows of
    768 ints each is 3 MB a phone would have to re-download every refresh.

    The amplitude here is a mean-removed RMS, which is a placeholder: it is
    band-blind, so it cannot tell a loud low rumble from a loud high one. That
    is what pipeline/features.py is for. It is still enough to answer the
    question day one actually asks -- is spin obviously louder than agitate,
    or is the sensor mounted somewhere that resonates?
    """
    with _lock:
        recent = list(_buffer)[-LIVE_WINDOWS:]
        nodes = {k: dict(v) for k, v in _nodes.items()}

    series = []
    for w in recent:
        point = {"t": w["t"], "seq": w["seq"], "hz": w["hz"], "late": w["late"]}
        for axis in ("x", "y", "z"):
            samples = w[axis]
            mean = sum(samples) / len(samples)
            # Mean removal is not optional: gravity is a constant ~16384 on the
            # vertical axis, and its square would swamp the vibration entirely.
            point[axis] = (sum((s - mean) ** 2 for s in samples) / len(samples)) ** 0.5
        series.append(point)

    now = time.time()
    for name, node in nodes.items():
        node["silent_for"] = round(now - node.get("last_seen", now), 1)

    return {"now": now, "buffered": len(_buffer), "nodes": nodes, "series": series}


@app.post("/cycles/{cycle_id}/empty", response_model=CycleOut)
def empty_cycle(cycle_id: int, body: EmptyIn, session: Session = Depends(get_session)) -> Cycle:
    """Someone took the laundry out."""
    cycle = session.get(Cycle, cycle_id)
    if cycle is None:
        raise HTTPException(404, f"no cycle {cycle_id}")
    cycle.emptied_at = time.time()
    cycle.emptied_by = body.by
    session.commit()
    return cycle


@app.get("/status", response_model=StatusOut)
def status(machine_id: str = "washer-01", session: Session = Depends(get_session)) -> StatusOut:
    """One request, one of four states. Everything the status screen renders.

    `phase` comes from the model when one is loaded and has seen a window, and
    from the last human mark otherwise. `predicted` says which, so the screen
    is honest about whether the system worked it out or is reading back what
    you told it.
    """
    now = time.time()

    with _lock:
        node = dict(_nodes.get(machine_id, {}))
    last_seen = node.get("last_seen")
    silent_for = None if last_seen is None else round(now - last_seen, 1)
    sensor_ok = silent_for is not None and silent_for <= SILENT_AFTER

    def marks_of(cycle: Cycle) -> list[Mark]:
        return list(session.scalars(select(Mark).where(Mark.cycle_id == cycle.id).order_by(Mark.t)))

    open_cycle = _open_cycle(session, machine_id)
    latest = session.scalar(
        select(Cycle).where(Cycle.machine_id == machine_id).order_by(Cycle.started_at.desc())
    )

    # Offline outranks everything. A screen that keeps showing the last known
    # phase while the sensor is gone looks exactly like a screen that knows.
    if not sensor_ok:
        last_phase = last_at = None
        if latest is not None:
            ms = marks_of(latest)
            if ms:
                last_phase, last_at = ms[-1].phase, ms[-1].t
        return StatusOut(
            mode="offline", now=now, sensor_ok=False, silent_for=silent_for,
            last_known_phase=last_phase, last_known_at=last_at,
        )

    inferred = _infer.get(machine_id) or {}
    predicted = _MODEL is not None and inferred.get("phase") is not None
    model_phase = inferred.get("phase") if predicted else None
    model_done = inferred.get("finished_at") if predicted else None

    # With a model, a cycle row is a RECORDING SESSION, not the washer's state.
    # Nobody has to tap Record for the machine to be running, so when there is
    # no open cycle the screen follows the sensor instead of the database.
    # cycle_id stays None, which the dashboard already handles -- the phase card
    # reads nothing but phase, phase_since and history.
    if open_cycle is None and predicted:
        if model_done is not None and now - model_done < MODEL_DONE_TTL:
            return StatusOut(
                mode="done", now=now, sensor_ok=True, silent_for=silent_for,
                finished_at=model_done, predicted=True,
            )
        if model_phase != "idle" and model_done is None:
            return StatusOut(
                mode="running", now=now, sensor_ok=True, silent_for=silent_for,
                phase=model_phase, phase_since=inferred.get("phase_since"),
                predicted=True,
            )

    # An open cycle is more interesting than a finished one -- a new wash
    # started means the last one stopped mattering, emptied or not.
    cycle = open_cycle or latest

    if cycle is not None:
        ms = marks_of(cycle)
        history = [PhaseEvent(at=m.t, phase=m.phase) for m in ms]

        # Finished means either a `done` mark or a closed cycle. The mark comes
        # first because it is the real event: in the finished product the model
        # spots that transition and nobody taps End at all. Without this, a
        # cycle marked done but left open reads as still running forever.
        done_mark = next((m for m in reversed(ms) if m.phase == "done"), None)
        finished_at = done_mark.t if done_mark is not None else cycle.ended_at

        # The point of the whole project: with a model, nobody has to tap Done.
        # A human mark still wins where one exists -- it is a direct observation
        # and this is an inference -- but its absence is no longer the same as
        # the wash not being over.
        if finished_at is None and predicted and inferred.get("finished_at"):
            finished_at = inferred["finished_at"]

        if finished_at is not None and cycle.emptied_at is None:
            return StatusOut(
                mode="done", now=now, sensor_ok=True, silent_for=silent_for,
                cycle_id=cycle.id, finished_at=finished_at, history=history,
                predicted=predicted and done_mark is None and cycle.ended_at is None,
            )

        if finished_at is None and open_cycle is not None:
            current = ms[-1] if ms else None
            return StatusOut(
                mode="running", now=now, sensor_ok=True, silent_for=silent_for,
                cycle_id=cycle.id,
                phase=inferred["phase"] if predicted else (current.phase if current else None),
                phase_since=(inferred["phase_since"] if predicted
                             else (current.t if current else cycle.started_at)),
                history=history,
                predicted=predicted,
            )

        # Finished and emptied: idle, but the last load is still worth showing.
        return StatusOut(
            mode="idle", now=now, sensor_ok=True, silent_for=silent_for,
            last_finished_at=finished_at,
            last_emptied_at=cycle.emptied_at,
            last_emptied_by=cycle.emptied_by,
        )

    return StatusOut(mode="idle", now=now, sensor_ok=True, silent_for=silent_for)


@app.get("/health")
def health() -> dict[str, Any]:
    with _lock:
        buffered = len(_buffer)
    return {"ok": True, "buffered": buffered, "recordings": str(RECORDINGS)}


# --------------------------------------------------------------------------
# The dashboard
#
# Mounted LAST so it cannot shadow an API route, and only if a build exists.
# Serving the built bundle from here rather than from `npm run dev` means the
# marking instrument does not die with a dev server -- and there is one thing
# to start before a wash instead of two.
#
#   cd web && npm run build      then open http://<laptop-ip>:8000/
# --------------------------------------------------------------------------

WEB_DIST = REPO_ROOT / "web" / "dist"
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
