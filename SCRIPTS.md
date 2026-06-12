# Scripts

Developer/operational scripts under `scripts/`. Run with the project venv:
`.venv/bin/python scripts/<name>.py`.

## seed_examples_from_reports.py

Populates `seed_examples/` from saved `test_reports/` payloads so the committed
seed reports (served on the landing page and as the analyse shortcut) reflect the
latest local runs.

**What it does** — for each requested drug:

1. Finds the latest `test_reports/{drug}_{timestamp}.json` payload (timestamp
   parsed from the filename).
2. Validates it as a `SupervisorOutput`; skips with a warning if it won't load.
3. Copies it to `seed_examples/{drug}.json`.
4. Records the capture time (epoch, from the filename timestamp) in
   `seed_examples/captured_at.json`.

Drug names are normalized (`normalize_drug_name`) so synonyms resolve to the same
seed file, matching the consumer in `services/seed_reports.py`. The manifest is
merged — entries for drugs not processed are left untouched. Drugs with no
`test_reports/` payload warn and keep any existing seed.

**Usage**

```bash
# Default: all of EXAMPLE_DRUGS (metformin, semaglutide, bupropion, sildenafil)
.venv/bin/python scripts/seed_examples_from_reports.py

# Specific drugs
.venv/bin/python scripts/seed_examples_from_reports.py metformin bupropion
```

**Source of payloads** — `scout find -d <drug>` writes the `SupervisorOutput`
payload to `test_reports/{drug}_{timestamp}.json`. This script consumes those.
