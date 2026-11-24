import numpy as np

from src.models.evaluation import bootstrap, evaluer, ic, prec_at_k


def test_classement_parfait():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.05, 0.0, 0.3, 0.15, 0.12, 0.01])
    assert prec_at_k(y, p, 0.2) == 1.0
    r = evaluer(y, p)
    assert r["PR-AUC"] == 1.0 and r["ROC-AUC"] == 1.0 and r["rappel@20%"] == 1.0


def test_modele_aleatoire_proche_du_taux_de_base():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.07).astype(int)
    p = rng.random(20000)
    r = evaluer(y, p)
    assert abs(r["ROC-AUC"] - 0.5) < 0.03
    assert abs(r["pr\u00e9cision@10%"] - y.mean()) < 0.02


def test_bootstrap_contient_la_valeur_observee():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.1).astype(int)
    p = rng.random(3000) + 0.3 * y
    B = bootstrap(y, [p], lambda a, b: prec_at_k(a, b, 0.1), n_boot=200)
    lo, hi = ic(B[:, 0])
    assert lo <= prec_at_k(y, p, 0.1) <= hi
