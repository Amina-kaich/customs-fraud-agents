"""Modeles de reference (BACUDA) : regression logistique et LightGBM.

Usage depuis la racine du projet :
    python -m src.models.baseline            # evalue sur la VALIDATION seulement
    python -m src.models.baseline --test     # evalue AUSSI sur le TEST (une seule fois, a la fin)
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import ajuster_stats, construire_features
from src.data.loader import TARGET, load_bacuda, temporal_split
from src.models.evaluation import bootstrap, evaluer, ic

REPORTS = Path(__file__).resolve().parents[2] / "reports"

# cles produites par evaluer() (les accents sont ecrits en \u pour garder un fichier ASCII)
K_P80 = "pr\u00e9cision \u00e0 rappel 80%"
K_P10 = "pr\u00e9cision@10%"
K_R10 = "rappel@10%"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


def preparer():
    """Charge les donnees, decoupe dans le temps et construit les variables (stats du train)."""
    df = load_bacuda()
    split = temporal_split(df)
    tr, va, te = (split == s for s in ("train", "valid", "test"))
    F = construire_features(df, ajuster_stats(df[tr]))
    y = df[TARGET].to_numpy()
    return F, y, tr.to_numpy(), va.to_numpy(), te.to_numpy()


def entrainer_logistique(X, y):
    modele = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return modele.fit(X, y)


def entrainer_lightgbm(Xtr, ytr, Xva, yva, seed=42):
    """Entraine LightGBM ; l'arret anticipe utilise la validation (jamais le test)."""
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(
        {**LGB_PARAMS, "seed": seed}, dtr, 3000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def _resume(yy, p, n_boot):
    res = {k: float(v) for k, v in evaluer(yy, p).items()}
    B = bootstrap(yy, [p], average_precision_score, n_boot=n_boot)
    lo, hi = ic(B[:, 0])
    res["PR-AUC IC95"] = [float(lo), float(hi)]
    return res


def executer(avec_test=False, seed=42, n_boot=300):
    F, y, tr, va, te = preparer()
    lr = entrainer_logistique(F[tr], y[tr])
    gbm = entrainer_lightgbm(F[tr], y[tr], F[va], y[va], seed=seed)
    preds = {
        "regression_logistique": lambda X: lr.predict_proba(X)[:, 1],
        "lightgbm": lambda X: gbm.predict(X, num_iteration=gbm.best_iteration),
    }
    resultats = {"seed": seed, "iterations_lightgbm": int(gbm.best_iteration)}
    parties = {"valid": va}
    if avec_test:
        parties["test"] = te
    for nom_part, masque in parties.items():
        resultats[nom_part] = {
            nom: _resume(y[masque], f(F[masque]), n_boot) for nom, f in preds.items()
        }
        resultats[nom_part]["taux_de_base"] = float(np.mean(y[masque]))
    return resultats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="evaluer aussi sur le test (une seule fois)")
    args = parser.parse_args()
    res = executer(avec_test=args.test)
    REPORTS.mkdir(exist_ok=True)
    nom = "baseline_metrics_test.json" if args.test else "baseline_metrics_valid.json"
    (REPORTS / nom).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for part in ("valid", "test"):
        if part in res:
            print(f"== {part} (taux de base {res[part]['taux_de_base']:.4f})")
            for modele in ("regression_logistique", "lightgbm"):
                r = res[part][modele]
                print(f"  {modele:22s} PR-AUC {r['PR-AUC']:.3f} IC95 [{r['PR-AUC IC95'][0]:.3f} ; {r['PR-AUC IC95'][1]:.3f}]"
                      f"  ROC-AUC {r['ROC-AUC']:.3f}  prec@rappel80 {r[K_P80]:.3f}"
                      f"  prec@10% {r[K_P10]:.3f}  rappel@10% {r[K_R10]:.3f}")
    print("Fichier ecrit :", REPORTS / nom)


if __name__ == "__main__":
    main()
