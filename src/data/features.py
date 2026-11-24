"""Construction des variables.

Règle anti-fuite : toutes les statistiques (fréquences des catégories, médianes de prix
unitaire) sont apprises sur le TRAIN uniquement par `ajuster_stats`, puis appliquées
à n'importe quelle partition par `construire_features`. Ni la cible ni les colonnes
de LEAK_COLUMNS ne sont utilisées.
"""
import numpy as np
import pandas as pd

FREQ_COLUMNS = {
    "OFFICE": "office",
    "ORIGIN.CODE": "origin",
    "IMPORTER.TIN": "importer",
    "HS6": "hs6",
    "TARIFF.CODE": "tariff",
}


def ajuster_stats(d: pd.DataFrame) -> dict:
    """Apprend les statistiques sur d (le train). N'utilise aucune étiquette."""
    s = {}
    for col in FREQ_COLUMNS:
        s[col] = d[col].value_counts()
    lup = np.log10(d["CIF_USD_EQUIVALENT"] / d["QUANTITY"])
    s["med_hs6"] = lup.groupby(d["HS6"]).agg(["median", "count"])
    s["med_hs4"] = lup.groupby(d["HS4"]).median()
    s["med_global"] = float(lup.median())
    return s


def construire_features(d: pd.DataFrame, s: dict) -> pd.DataFrame:
    """Transforme d en variables numériques, avec les statistiques s apprises sur le train."""
    out = pd.DataFrame(index=d.index)
    cif, qty = d["CIF_USD_EQUIVALENT"], d["QUANTITY"]
    wgt, tax = d["GROSS.WEIGHT"], d["TOTAL.TAXES.USD"].clip(lower=0)

    out["log_cif"] = np.log10(cif + 1)
    out["log_qty"] = np.log10(qty + 1)
    out["log_weight"] = np.log10(wgt + 1)
    out["log_taxes"] = np.log10(tax + 1)
    out["log_unit_price"] = np.log10(cif / qty)
    out["log_cif_per_kg"] = np.log10(cif / wgt)
    out["log_weight_per_unit"] = np.log10(wgt / qty)
    out["tax_rate_eff"] = (tax / cif).clip(upper=1.5)

    out["taxes_is_5"] = (tax == 5).astype(int)
    out["taxes_is_0"] = (tax == 0).astype(int)
    out["weight_is_1"] = (wgt == 1).astype(int)
    out["qty_is_1"] = (qty == 1).astype(int)
    out["month"] = d["month"]

    for col, nom in FREQ_COLUMNS.items():
        out["logfreq_" + nom] = np.log10(1 + d[col].map(s[col]).fillna(0))

    ref = d["HS6"].map(s["med_hs6"]["median"].where(s["med_hs6"]["count"] >= 5))
    ref = ref.fillna(d["HS4"].map(s["med_hs4"])).fillna(s["med_global"])
    out["ecart_norme"] = out["log_unit_price"] - ref
    return out