"""Create the pgvector extension and all ORM tables in the configured database.

Used by CI to stand up an empty Postgres before a regression run. Idempotent:
existing tables are left alone.
"""

import logging

from sqlalchemy import text

from indication_scout.db.base import Base
from indication_scout.db.session import make_session_factory

# Imported for their side effect: registering the mapped classes on Base.metadata.
from indication_scout.sqlalchemy import analysis_runs, pubmed_abstracts  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    engine = make_session_factory().kw["bind"]
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    logger.info("Created %d tables", len(Base.metadata.tables))


if __name__ == "__main__":
    main()
