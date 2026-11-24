"""Chargement du jeu BACUDA (synthetic_data2.csv) et découpage temporel."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
BACUDA_FILE = "bacuda_synthetic_data2.csv"

TARGET = "illicit"
# Supplément de taxe récupéré après inspection : positif uniquement pour les fraudes.
# Cette colonne contient la réponse, elle ne doit JAMAIS servir de variable explicative.
LEAK_COLUMNS = ["RAISED_TAX_AMOUNT_USD"]

EXPECTED_COLUMNS = [
    "year", "month", "day", "OFFICE", "IMPORTER.TIN", "TARIFF.CODE", "ORIGIN.CODE",
    "CIF_USD_EQUIVALENT", "QUANTITY", "GROSS.WEIGHT", "TOTAL.TAXES.USD",
    "RAISED_TAX_AMOUNT_USD", TARGET,
]


def check_schema(df: pd.DataFrame) -> None:
    """Lève une erreur si une colonne attendue est absente."""
    manquantes = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")


def add_hs_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les regroupements du code tarifaire (6, 4 et 2 premiers chiffres)."""
    df = df.copy()
    df["TARIFF.CODE"] = df["TARIFF.CODE"].astype(str).str.zfill(10)
    df["HS6"] = df["TARIFF.CODE"].str[:6]
    df["HS4"] = df["TARIFF.CODE"].str[:4]
    df["HS2"] = df["TARIFF.CODE"].str[:2]
    return df


def load_bacuda(path=None) -> pd.DataFrame:
    """Charge le fichier BACUDA avec les bons types et les regroupements SH."""
    path = Path(path) if path else RAW_DIR / BACUDA_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Télécharger le fichier dans data/raw/.")
    df = pd.read_csv(path, dtype={"TARIFF.CODE": str})
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    check_schema(df)
    return add_hs_levels(df)


def temporal_split(df: pd.DataFrame) -> pd.Series:
    """Découpage temporel : train = 2013, valid = janv.-juin 2014, test = juil.-déc. 2014.

    On s'appuie sur l'année et le mois, car la colonne `day` contient des jours impossibles.
    """
    return pd.Series(
        np.where(df["year"] == 2013, "train", np.where(df["month"] <= 6, "valid", "test")),
        index=df.index,
        name="split",
    )