import numpy as np
import pandas as pd

from src.data.loader import LEAK_COLUMNS, TARGET, add_hs_levels
from src.data.features import ajuster_stats, construire_features


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "year": 2013,
        "month": rng.integers(1, 13, n),
        "OFFICE": rng.choice([f"O{i}" for i in range(6)], n),
        "IMPORTER.TIN": rng.choice([f"I{i}" for i in range(40)], n),
        "TARIFF.CODE": rng.choice(["8703230000", "0101210000", "3926909000"], n),
        "ORIGIN.CODE": rng.choice(["C1", "C2", "C3"], n),
        "CIF_USD_EQUIVALENT": rng.random(n) * 5000 + 10,
        "QUANTITY": rng.integers(1, 50, n).astype(float),
        "GROSS.WEIGHT": rng.integers(1, 5000, n).astype(float),
        "TOTAL.TAXES.USD": rng.choice([0.0, 5.0, 120.0, 800.0], n),
        "RAISED_TAX_AMOUNT_USD": 0.0,
        TARGET: rng.integers(0, 2, n),
    })
    return add_hs_levels(df)


def test_pas_de_cible_ni_de_fuite_dans_les_features():
    d = _toy()
    F = construire_features(d, ajuster_stats(d))
    interdits = set(LEAK_COLUMNS) | {TARGET}
    assert not (interdits & set(F.columns))


def test_aucune_valeur_manquante_ou_infinie():
    d = _toy()
    F = construire_features(d, ajuster_stats(d))
    assert F.shape == (len(d), 19)
    assert not F.isna().any().any()
    assert not np.isinf(F.to_numpy(dtype=float)).any()


def test_ajuster_stats_n_utilise_pas_la_cible():
    d = _toy()
    s_avec = ajuster_stats(d)
    s_sans = ajuster_stats(d.drop(columns=[TARGET, *LEAK_COLUMNS]))
    assert s_avec["IMPORTER.TIN"].equals(s_sans["IMPORTER.TIN"])
    assert s_avec["med_global"] == s_sans["med_global"]


def test_entite_inconnue_du_train_a_une_frequence_nulle():
    train, autre = _toy(400, 0), _toy(100, 1)
    autre["IMPORTER.TIN"] = "JAMAIS_VU"
    F = construire_features(autre, ajuster_stats(train))
    assert (F["logfreq_importer"] == 0).all()


def test_les_features_d_une_ligne_ne_dependent_pas_des_autres_partitions():
    train = _toy(400, 0)
    s = ajuster_stats(train)
    a = _toy(50, 1)
    b = pd.concat([a, _toy(200, 2)], ignore_index=True)
    Fa = construire_features(a, s)
    Fb = construire_features(b, s).iloc[: len(a)]
    pd.testing.assert_frame_equal(Fa.reset_index(drop=True), Fb.reset_index(drop=True))
