import pandas as pd
import pytest

from src.data.loader import (
    BACUDA_FILE, LEAK_COLUMNS, RAW_DIR, TARGET, add_hs_levels, check_schema,
    load_bacuda, temporal_split,
)


def _toy():
    return pd.DataFrame({
        "year": [2013, 2013, 2014, 2014, 2014],
        "month": [1, 12, 1, 6, 7],
        "TARIFF.CODE": ["870323", "0101210000", "1234567890", "2", "8703230000"],
    })


def test_zfill_et_regroupements_sh():
    d = add_hs_levels(_toy())
    assert (d["TARIFF.CODE"].str.len() == 10).all()
    assert d.loc[0, "TARIFF.CODE"] == "0000870323"
    assert d.loc[1, "HS6"] == "010121" and d.loc[1, "HS4"] == "0101" and d.loc[1, "HS2"] == "01"


def test_temporal_split_ordre_et_bornes():
    s = temporal_split(_toy())
    assert s.tolist() == ["train", "train", "valid", "valid", "test"]


def test_check_schema_detecte_colonne_manquante():
    with pytest.raises(ValueError):
        check_schema(pd.DataFrame({"year": [2013]}))


def test_la_cible_et_la_fuite_sont_declarees():
    assert TARGET == "illicit"
    assert "RAISED_TAX_AMOUNT_USD" in LEAK_COLUMNS


@pytest.mark.skipif(not (RAW_DIR / BACUDA_FILE).exists(), reason="Donnees BACUDA absentes de data/raw/")
def test_valeurs_de_reference_sur_les_vraies_donnees():
    df = load_bacuda()
    assert len(df) == 200_000
    assert int(df[TARGET].sum()) == 14_370
    assert (df["TARIFF.CODE"].str.len() == 10).all()
    # la colonne de fuite est positive exactement pour les fraudes
    assert ((df["RAISED_TAX_AMOUNT_USD"] > 0) == (df[TARGET] == 1)).all()
    tailles = temporal_split(df).value_counts().to_dict()
    assert tailles == {"train": 113_223, "valid": 46_348, "test": 40_429}
