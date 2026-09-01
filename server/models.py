"""SQLAlchemy tables -- three of them, and they stay three.

`machines` is the thing that vibrates, `cycles` is one wash from start to
finish, `marks` is a human saying "spin starts now". That is the entire
schema, because the bulk data never comes here: 256 samples x 3 axes every
2.56 seconds goes to append-only JSONL in analysis/recordings/, and the
training pipeline reads those as files.

Two conventions run through all three tables:

Times are float epoch seconds, not DateTime. The JSONL windows are stamped by
the server as floats, and analysis/label.py exists to join marks against those
windows by time. Storing one side as DateTime means a conversion at every join,
written on day two, while tired.

An open cycle is `ended_at IS NULL`. There is no is_recording flag and no
status column -- a second source of truth is a second thing that can desync,
which is the same reason cycle state doesn't live on the ESP32.
"""

import time
from typing import Any, Optional

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.db import Base


class Machine(Base):
    """One washer. There is exactly one today; the table exists so a second
    one is an INSERT rather than a migration."""

    __tablename__ = "machines"

    # Natural string key, not an autoincrement int. The node already puts
    # "washer-01" in every POST body, so an integer key would mean a lookup on
    # the hot path just to translate a name the client already sent.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(128))
    installed_at: Mapped[float] = mapped_column(Float, default=time.time)

    # Per-machine baseline: mean/std per feature, written by pipeline/calibrate.py.
    # A JSON blob rather than columns because the feature vector's shape is still
    # moving -- features.py isn't written yet, and pinning ~15 float columns now
    # would mean a schema change the first time a band is added or dropped.
    # NULL until the machine has been calibrated.
    calibration: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    cycles: Mapped[list["Cycle"]] = relationship(
        back_populates="machine", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Machine {self.id!r}>"


class Cycle(Base):
    """One wash. Open while ended_at is NULL; that is what gates whether an
    arriving window gets written to disk or dropped from the ring buffer."""

    __tablename__ = "cycles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # No index=True here: the composite index at the bottom of the class starts
    # with machine_id, so a standalone one would be a second copy of the same
    # prefix -- more to write on every insert, nothing more to read.
    machine_id: Mapped[str] = mapped_column(
        # ondelete only does anything because db.py sets PRAGMA foreign_keys=ON
        # per connection. SQLite parses this clause and ignores it by default.
        ForeignKey("machines.id", ondelete="CASCADE")
    )

    started_at: Mapped[float] = mapped_column(Float, default=time.time)
    ended_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Filename of this cycle's JSONL, e.g. "wash-2026-08-31-1403.jsonl". The
    # link between a row here and ~1400 windows on disk. Nullable because the
    # row is created the instant you tap Start, before the file has a name.
    recording: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Recorded so you can answer "did the model only ever see Normal?" months
    # from now. You will not remember, and there is no way to recover it after
    # the fact. Two nullable columns is a cheap insurance policy.
    setting: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    load_size: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Someone took the laundry out. Separate from ended_at because they are
    # genuinely different events -- the machine finishing and a human noticing
    # can be an hour apart, and the gap between them is the only thing this
    # whole project is trying to shrink.
    emptied_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    emptied_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    machine: Mapped["Machine"] = relationship(back_populates="cycles")

    # order_by is not cosmetic: label.py derives a phase span from one mark to
    # the next, so it needs them in time order. Doing it here means every read
    # is sorted and no caller can forget.
    marks: Mapped[list["Mark"]] = relationship(
        back_populates="cycle", cascade="all, delete-orphan", order_by="Mark.t"
    )

    # The one query the ingest path runs on every window: "is there an open
    # cycle for this machine?" -- machine_id = ? AND ended_at IS NULL.
    __table_args__ = (Index("ix_cycles_machine_open", "machine_id", "ended_at"),)

    def __repr__(self) -> str:
        state = "open" if self.ended_at is None else "closed"
        return f"<Cycle {self.id} {self.machine_id!r} {state}>"


class Mark(Base):
    """A phase boundary, as a point in time -- not a span.

    You tap "spin" when spin starts, and that is all that is stored. label.py
    sorts a cycle's marks and reads span i as mark[i] -> mark[i+1]. Storing
    spans instead would let two rows overlap or contradict each other, with
    nothing in the data to say which one is right.

    `phase` is a plain string on purpose. The label set is a finding, not a
    given -- CLAUDE.md already expects `fill` to be indistinguishable from
    `idle` -- and an enum column would turn that discovery into a migration.
    The vocabulary is enforced one layer up, in schemas.PHASES.
    """

    __tablename__ = "marks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(
        ForeignKey("cycles.id", ondelete="CASCADE"), index=True
    )
    t: Mapped[float] = mapped_column(Float, default=time.time)
    phase: Mapped[str] = mapped_column(String(32))

    cycle: Mapped["Cycle"] = relationship(back_populates="marks")

    def __repr__(self) -> str:
        return f"<Mark {self.phase!r} @ {self.t}>"
