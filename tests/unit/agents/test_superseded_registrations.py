"""Unit tests for sibling-registration collapsing.

A protocol run under the EU Clinical Trials Regulation gets one ClinicalTrials.gov registration
per site or per indication, and the registry models no relation between them. The INTENSIFY
protocol is the worked case: NCT05603104 is the terminated umbrella whose stop reason names the EU
CT numbers still recruiting, and NCT05973786 (bipolar) carries one of those numbers as its
org_study_id. Counting both is a double-count, and describing the terminated one as a separate
programme that ended is wrong.

The controls matter as much as the positive case: "terminated with an enrollment of 0 or 1" looks
like the same pattern and is not, so those records must survive untouched.
"""

import pytest

from indication_scout.agents._trial_signals import (
    format_derived_signals,
    superseded_registrations,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import TrialSignals
from indication_scout.models.model_clinical_trials import Trial

_INTENSIFY_STOP = (
    "The site in Israel stopped recruitment due to logistic reasons. Other sites are "
    "recruiting for 2023-506602-39-00, 2023-506605-19-00, 2023-506617-21-00."
)

# The terminated umbrella registration: 1 participant enrolled, stop reason names its successors.
_UMBRELLA = Trial(
    nct_id="NCT05603104",
    org_study_id="2022-502185-24-00 (EU CT #)",
    title="Intensified Pharmacological Treatment for Schizophrenia, MDD and Bipolar Depression",
    phase="Phase 3",
    overall_status="TERMINATED",
    why_stopped=_INTENSIFY_STOP,
    enrollment=1,
)

# The continuing bipolar registration: its org_study_id is one of the ids named above.
_SUCCESSOR = Trial(
    nct_id="NCT05973786",
    org_study_id="2023-506605-19-00 (EU CT#)",
    title="Six Week Intensified Pharmacological Treatment for Bipolar Depression",
    phase="Phase 3",
    overall_status="RECRUITING",
)

# Terminated with an enrollment of 1 and no reference to anything — the proxy that must NOT fire.
_UNRELATED_STOP = Trial(
    nct_id="NCT01219673",
    org_study_id="2010-0334",
    title="Symptom Burden in Head and Neck Cancer",
    phase="Phase 1",
    overall_status="TERMINATED",
    why_stopped="slow accrual",
    enrollment=1,
)


def test_sibling_registration_is_collapsed_into_its_successor():
    assert superseded_registrations([_UMBRELLA, _SUCCESSOR]) == {
        "NCT05603104": "NCT05973786"
    }


@pytest.mark.parametrize(
    "trials",
    [
        pytest.param([_UMBRELLA], id="successor_absent_from_set"),
        pytest.param([_UNRELATED_STOP], id="terminated_tiny_enrollment_no_reference"),
        pytest.param([_SUCCESSOR], id="active_trial_alone"),
    ],
)
def test_nothing_is_collapsed_without_a_resolving_reference(trials):
    """Only an explicit pointer that RESOLVES to another trial in the set collapses a record.

    A stop reason naming ids absent from the set is not enough: the sibling may be out of scope
    for this pair, and dropping the record would lose a trial the report should describe.
    """
    assert superseded_registrations(trials) == {}


def test_unrelated_terminations_survive_alongside_a_real_sibling_pair():
    """The tiny-enrollment proxy must not be swept up when a genuine pair is also present."""
    assert superseded_registrations([_UMBRELLA, _SUCCESSOR, _UNRELATED_STOP]) == {
        "NCT05603104": "NCT05973786"
    }


def test_an_active_trial_citing_another_protocol_is_never_collapsed():
    """Supersession requires a stopped status; an active trial that cites a protocol stays."""
    citing_active = _UMBRELLA.model_copy(
        update={"overall_status": "RECRUITING", "nct_id": "NCT09999999"}
    )
    assert superseded_registrations([citing_active, _SUCCESSOR]) == {}


def test_legacy_eudract_number_resolves_via_secondary_ids():
    """Older records carry the EudraCT number as a secondary id rather than the org study id.

    The EU CT form carries two extra digits (2023-506605-19-00) over the legacy EudraCT form
    (2012-004321-16), so the legacy pattern also matches a prefix of any EU CT number. Extraction
    must not turn one EU CT id into both itself and a spurious truncated twin.
    """
    legacy_stop = Trial(
        nct_id="NCT00000001",
        overall_status="WITHDRAWN",
        why_stopped="Superseded; recruitment continues under 2012-004321-16.",
    )
    legacy_successor = Trial(
        nct_id="NCT00000002",
        secondary_ids=["2012-004321-16"],
        overall_status="RECRUITING",
    )
    assert superseded_registrations([legacy_stop, legacy_successor]) == {
        "NCT00000001": "NCT00000002"
    }


def test_direct_nct_reference_in_a_stop_reason_resolves():
    """Some sponsors name the successor's NCT id outright rather than a protocol number."""
    stopped = Trial(
        nct_id="NCT00000003",
        overall_status="TERMINATED",
        why_stopped="Study was re-registered as NCT00000004.",
    )
    successor = Trial(nct_id="NCT00000004", overall_status="RECRUITING")
    assert superseded_registrations([stopped, successor]) == {
        "NCT00000003": "NCT00000004"
    }


def test_collapsed_registration_leaves_the_relevant_id_list():
    """The collapse must remove the id from relevant_nct_ids, not just from a local list.

    Coverage counts, the report's trial listings and the prose all key off relevant_nct_ids. The
    2026-11-17 bupropion run collapsed the record only in the list handed to the prose writer, so
    the section read "Three relevant trials" above "Trial activity: 4 relevant trial(s)" and still
    listed the collapsed NCT05603104 under terminated trials.
    """
    relevant = ["NCT05603104", "NCT05973786", "NCT00012558"]
    superseded = superseded_registrations([_UMBRELLA, _SUCCESSOR])
    remaining = [n for n in relevant if n not in superseded]

    assert remaining == ["NCT05973786", "NCT00012558"]


def test_fact_block_tells_the_prose_writer_not_to_describe_the_collapsed_record():
    """The count silently dropping by one would read as a retrieval gap without this line."""
    block = format_derived_signals(
        TrialSignals(superseded_registrations={"NCT05603104": "NCT05973786"})
    )
    assert "NCT05603104 -> NCT05973786" in block
    assert "do not describe it as a separate trial" in block
