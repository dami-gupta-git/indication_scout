from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from indication_scout.config import get_settings


def _make_engine():
    return create_engine(get_settings().database_url)


def _make_session_factory():
    return sessionmaker(autocommit=False, autoflush=False, bind=_make_engine())


def get_db() -> Generator[Session, None, None]:
    db = _make_session_factory()()
    try:
        yield db
    finally:
        db.close()
