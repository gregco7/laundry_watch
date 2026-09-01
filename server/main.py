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
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.db import REPO_ROOT, get_session, init_db
from server.models import Cycle, Machine, Mark
from server.schemas import (
    CycleOut,
    CycleStart,
    MachineOut,
    MarkIn,
    MarkOut,
    ReadingAck,
    ReadingIn,
)

# Five minutes of pre-roll. Long enough that a late Start still catches the
# fill, short enough that the buffer is a few megabytes rather than a leak:
# ~117 windows x 768 ints.
RING_SECONDS = 300.0

# What GET /live hands the dashboard. ~60 windows is about 2.5 minutes at
# 100 Hz -- enough of a strip chart to see agitate turn into spin.
LIVE_WINDOWS = 60

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    RECORDINGS.mkdir(parents=True, exist_ok=True)
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


@app.get("/health")
def health() -> dict[str, Any]:
    with _lock:
        buffered = len(_buffer)
    return {"ok": True, "buffered": buffered, "recordings": str(RECORDINGS)}
