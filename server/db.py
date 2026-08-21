"""SQLite engine + session (SQLAlchemy).

Local single-file database -- no server process, no Docker. Keeps
"clone and run" to a single step.

To implement (fill in one at a time):
- engine            -> SQLAlchemy engine over a local .db file
- SessionLocal      -> session factory
- get_session()     -> FastAPI dependency
- init_db()         -> create tables
"""
