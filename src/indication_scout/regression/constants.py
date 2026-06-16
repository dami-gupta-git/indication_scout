"""Tunable thresholds for the regression harness.

Kept in one place so calibration changes are reviewed atomically rather than
scattered across test bodies.
"""

# Minimum Jaccard overlap between the golden and current candidate-disease
# sets. Below this, the test fails. Calibrate after 2-3 back-to-back recorded
# runs of the same input show how stable the candidate set is at temperature=0.
CANDIDATE_SET_JACCARD_MIN = 0.7

# Same threshold for top_diseases (the supervisor's ranked top-3).
TOP_DISEASES_JACCARD_MIN = 0.6

# For each shared candidate, tolerated absolute drift on numeric fields.
EVIDENCE_COUNT_TOLERANCE = 5
SCORE_TOLERANCE = 0.05

# Free-text fields are checked for presence + length bounds only, never exact
# match. These bounds catch silent regressions (e.g. empty summary, runaway
# generation) without flagging legitimate prose drift.
SUMMARY_MIN_LEN = 50
SUMMARY_MAX_LEN = 20_000
BLURB_PROSE_MIN_LEN = 20
BLURB_PROSE_MAX_LEN = 2_000
