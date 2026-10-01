"""SQLite engine + session factory.

One SQLite file backs both the FastAPI request handlers (which run in a threadpool) and the
APScheduler fire callback (which runs on the event loop), so the connection is opened with
``check_same_thread=False``. Volume is low (a home's worth of tasks), so SQLite's own file
locking is plenty.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from models import Base


def make_engine(db_path: str) -> Engine:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine: Engine) -> None:
    """Add columns introduced after a database was created; create_all skips them."""
    added = {"tasks": {"user": "VARCHAR(64)"}, "settings": {"users": "JSON"}}
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, columns in added.items():
            have = {c["name"] for c in insp.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in have:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN "{name}" {sql_type}'))


def make_session_factory(engine: Engine) -> sessionmaker:
    # expire_on_commit=False so a Task read inside a `with Session()` block stays usable for
    # serialization after commit (we build response models before the session closes).
    return sessionmaker(bind=engine, expire_on_commit=False)
