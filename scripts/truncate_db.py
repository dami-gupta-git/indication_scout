"""Truncate all user tables in the main database.

Queries information_schema to find all tables in the public schema
(excluding alembic_version), then truncates them with CASCADE.

Usage:
    python scripts/truncate_db.py
"""

import logging
import os
from pathlib import Path

from sqlalchemy import text

# Ensure .env is found when the script is run from any working directory.
os.chdir(Path(__file__).resolve().parent.parent)

from indication_scout.db.session import get_db

logger = logging.getLogger(__name__)

_EXCLUDED_TABLES = {"alembic_version"}


def main() -> None:
    db = next(get_db())

    rows = db.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
    ).fetchall()

    tables = [row[0] for row in rows if row[0] not in _EXCLUDED_TABLES]

    if not tables:
        logger.info("No user tables found — nothing to truncate.")
        return

    table_list = ", ".join(tables)
    logger.info("Truncating: %s", table_list)
    db.execute(text(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE"))
    db.commit()
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
