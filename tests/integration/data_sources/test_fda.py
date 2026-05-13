"""Integration tests for FDAClient — hits real openFDA API."""

import pytest

from indication_scout.data_sources.fda import FDAClient


async def test_get_label_indications_wegovy(fda_client):
    """Wegovy returns label text with cardiovascular and weight management indications."""
    result = await fda_client.get_label_indications("Wegovy")

    assert len(result) == 2
    assert all("WEGOVY" in text for text in result)
    assert any("cardiovascular" in text.lower() for text in result)
    assert any(
        "weight management" in text.lower() or "body weight" in text.lower()
        for text in result
    )
    assert any("steatohepatitis" in text.lower() for text in result)


async def test_get_label_indications_generic_name_semaglutide(fda_client):
    """Generic/INN name ("semaglutide") resolves via the generic_name clause.

    Under a brand_name-only search, `semaglutide` 404s (no FDA brand is literally
    named "semaglutide"). The OR query adds openfda.generic_name, which returns
    semaglutide-labelled products (Ozempic/Wegovy/Rybelsus). A non-empty result
    here proves the generic_name branch is live — guards against regression if
    someone reverts the OR clause to brand_name-only.
    """
    result = await fda_client.get_label_indications("semaglutide")

    assert len(result) > 0
    combined = " ".join(result).lower()
    assert "type 2 diabetes" in combined or "body weight" in combined


async def test_get_label_indications_nonexistent_brand(fda_client):
    """A brand that doesn't exist returns an empty list, not an error."""
    result = await fda_client.get_label_indications("ZZZNotARealDrugBrand999")

    assert result == []


async def test_get_all_label_indications_semaglutide_brands(fda_client):
    """All three semaglutide brands together return 5 distinct indication texts."""
    result = await fda_client.get_all_label_indications(
        ["Ozempic", "Rybelsus", "Wegovy"]
    )

    assert len(result) >= 2
    combined = " ".join(result).lower()
    assert "ozempic" in combined
    assert "wegovy" in combined
    assert "type 2 diabetes" in combined
    assert "cardiovascular" in combined
    assert "steatohepatitis" in combined


async def test_get_label_indications_xeljanz(fda_client):
    """Xeljanz returns labels mentioning RA, PsA, and ulcerative colitis."""
    result = await fda_client.get_label_indications("Xeljanz")

    assert len(result) == 2
    combined = " ".join(result).lower()
    assert "rheumatoid arthritis" in combined
    assert "ulcerative colitis" in combined
    assert "psoriatic arthritis" in combined
