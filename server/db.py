"""SQLite engine + session (SQLAlchemy 2.0).

Local single-file database -- no server process, no Docker. Keeps
"clone and run" to a single step.

What lives here is small on purpose: cycle boundaries and phase marks, the
handful of rows a human creates by pressing a button. The bulk data -- 256
samples per axis, every 2.56 seconds, all day -- goes to append-only JSONL in
analysis/recordings/. Eight hours of windows is ~11k rows of 768 ints each;
SQLite would swallow it, but nothing downstream wants it back in row form.
The training pipeline reads files.

Everything below is a decision about SQLite's defaults, all of which are
wrong for a server that is written to and inspected at the same time.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Anchored to the repo root, not the working directory. A bare "laundry.db"
# means `uvicorn` launched from ~/ and `uvicorn` launched from the repo open
# two different files, and the failure looks like "my marks vanished" rather
# than anything to do with paths. The env override exists so tests can point
# at a tmp file instead of stomping the real recording.
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("LAUNDRY_DB") or REPO_ROOT / "laundry.db")

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    # sqlite3 refuses by default to use a connection from any thread other
    # than the one that opened it. FastAPI hands every non-async endpoint to a
    # threadpool worker, so that check fires more or less at random under load.
    # Safe to disable here because SessionLocal hands each request its own
    # session and never shares one across threads.
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record) -> None:
    """Applied per connection -- PRAGMAs are connection state, not file state."""
    cur = dbapi_conn.cursor()

    # Default journaling takes an exclusive lock for the duration of a write,
    # so the moment you open `sqlite3 laundry.db` in a second terminal to check
    # whether a mark actually landed, the next write comes back "database is
    # locked". WAL lets one writer and any number of readers coexist. You will
    # be poking at this file by hand constantly while a wash is running.
    cur.execute("PRAGMA journal_mode=WAL")

    # SQLite parses FOREIGN KEY, stores it, and then ignores it unless this is
    # on -- per connection, every time. Off, a mark can reference a cycle id
    # that never existed and nothing complains until label.py tries to join and
    # silently drops the row.
    cur.execute("PRAGMA foreign_keys=ON")

    cur.close()


class Base(DeclarativeBase):
    """Declarative base. models.py imports this; this module must not import
    models.py at import time, or the two form a cycle. See init_db()."""


SessionLocal = sessionmaker(
    bind=engine,
    # Off: an autoflush fires a partially-built object at the database the
    # moment an unrelated query runs, and the resulting NOT NULL error points
    # at the query rather than at the half-filled object that caused it.
    autoflush=False,
    # Off: by default every attribute is expired on commit and re-read on next
    # access. In FastAPI the session closes before the response is serialized,
    # so that re-read has no connection and raises DetachedInstanceError on
    # what looks like a plain attribute access.
    expire_on_commit=False,
)


def get_session() -> Iterator[Session]:
    """FastAPI dependency -- one session per request, always closed.

    Deliberately does not commit: the endpoint decides what constitutes a unit
    of work. Teardown runs after the response is sent.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create any missing tables. Idempotent; call it on server startup.

    The import is function-scoped rather than top-of-file to break the cycle
    (db -> models -> db), but it is also load-bearing: create_all only knows
    about tables that have been registered on Base.metadata by an import, so
    importing models is what makes the tables exist at all.

    Caveat: create_all creates, it never alters. Add a column to an existing
    table and this call will not notice -- during the build, delete the file
    and start over.
    """
    from server import models  # noqa: F401  -- registers tables on Base.metadata

    Base.metadata.create_all(engine)
