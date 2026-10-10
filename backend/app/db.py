"""SQLAlchemy engine and session plumbing."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

_connect_args = {"check_same_thread": False} if settings.db_url.startswith("sqlite") else {}

engine = create_engine(settings.db_url, connect_args=_connect_args, future=True)


if settings.db_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def configure_sqlite(conn, _):
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """For background workers, which have no request lifecycle."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  (registers mappers)

    models.Base.metadata.create_all(engine)
    # Additive migration for existing local databases; never recreate user data.
    columns = {c["name"] for c in inspect(engine).get_columns("lectures")}
    with engine.begin() as conn:
        for name, kind in (("source_path", "VARCHAR(1024)"), ("asr_config", "JSON"), ("summary", "TEXT")):
            if name not in columns:
                conn.execute(text(f"ALTER TABLE lectures ADD COLUMN {name} {kind}"))
        chat_columns = {c["name"] for c in inspect(conn).get_columns("chat_messages")}
        if "studio_item_id" not in chat_columns:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN studio_item_id VARCHAR(32)"))
