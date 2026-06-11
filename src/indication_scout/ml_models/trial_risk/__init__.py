"""Trial-termination risk classifier.

Standalone subpackage. Trains a calibrated logistic-regression model on cached
ClinicalTrials.gov data (`_scout_cache/ct_completed/`, `_scout_cache/ct_terminated/`) plus
date-bounded PubMed literature signals via the existing retrieval pipeline.

Run via:
    python -m indication_scout.ml_models.trial_risk.train
    python -m indication_scout.ml_models.trial_risk.score
"""
