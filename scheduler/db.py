"""SQLite engine + session factory.

One SQLite file backs both the FastAPI request handlers (which run in a threadpool) and the
APScheduler fire callback (which runs on the event loop), so the connection is opened with
``check_same_thread=False``. Volume is low (a home's worth of tasks), so SQLite's own file
locking is plenty.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine
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
    return engine


def make_session_factory(engine: Engine) -> sessionmaker:
    # expire_on_commit=False so a Task read inside a `with Session()` block stays usable for
    # serialization after commit (we build response models before the session closes).
    return sessionmaker(bind=engine, expire_on_commit=False)
