"""Prune snapshot reports: keep only the most recent report per drug, move older ones to bak/.

Report files are named `{drug}_{YYYY-MM-DD_HH-MM-SS}.md`. The timestamp format sorts
lexically, so the most recent report per drug is the lexically-largest filename.

Usage (run from the project root):
    python scripts/prune_snapshots.py            # move older reports to snapshots/bak/
    python scripts/prune_snapshots.py --dry-run  # show what would move, change nothing
"""

import logging
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("indication_scout.prune_snapshots")

SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "snapshots"
BAK_DIR = SNAPSHOTS_DIR / "bak"


def drug_from_filename(name: str) -> str | None:
    """Return the drug name from `{drug}_{date}_{time}.md`, or None if it doesn't match."""
    stem = name[:-3] if name.endswith(".md") else name
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    drug, date, time = parts
    if not drug:
        return None
    return drug


def main(dry_run: bool = False) -> None:
    if not SNAPSHOTS_DIR.is_dir():
        logger.error("snapshots dir not found: %s", SNAPSHOTS_DIR)
        return

    by_drug: dict[str, list[Path]] = defaultdict(list)
    for path in SNAPSHOTS_DIR.iterdir():
        if not path.is_file() or path.suffix != ".md":
            continue
        drug = drug_from_filename(path.name)
        if drug is None:
            logger.info("skipping unrecognized filename: %s", path.name)
            continue
        by_drug[drug].append(path)

    if not dry_run:
        BAK_DIR.mkdir(exist_ok=True)

    moved = 0
    for drug, paths in sorted(by_drug.items()):
        paths.sort(key=lambda p: p.name)
        keep = paths[-1]
        old = paths[:-1]
        logger.info("%s: keeping %s, moving %d older", drug, keep.name, len(old))
        for path in old:
            dest = BAK_DIR / path.name
            if dry_run:
                logger.info("  [dry-run] would move %s -> bak/", path.name)
                continue
            if dest.exists():
                logger.warning("  destination exists, skipping: bak/%s", path.name)
                continue
            shutil.move(str(path), str(dest))
            moved += 1

    logger.info("done: moved %d file(s) to bak/", moved)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show what would move without moving")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
