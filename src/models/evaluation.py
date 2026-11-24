"""Métriques d'évaluation orientées budget d'inspection, et intervalles de confiance."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def prec_at_k(yy, p, k: float = 0.10) -> float:
    """Part de fraudes parmi les k % de déclarations les mieux notées."""
    yy, p = np.asarray(yy), np.asarray(p)
    n = max(1, int(len(yy) * k))
    top = np.argsort(-p, kind="stable")[:n]
    return float(yy[top].mean())


def evaluer(yy, p) -> dict:
    """PR-AUC, ROC-AUC, précision à rappel 80 %, précision et rappel @5, 10 et 20 %."""
    yy, p = np.asarray(yy), np.asarray(p)
    pr, rc, _ = precision_recall_curve(yy, p)
    i = np.where(rc >= 0.80)[0][-1]
    out = {
        "PR-AUC": average_precision_score(yy, p),
        "ROC-AUC": roc_auc_score(yy, p),
        "précision à rappel 80%": pr[i],
    }
    for k in (0.05, 0.10, 0.20):
        n = max(1, int(len(yy) * k))
        top = np.argsort(-p, kind="stable")[:n]
        out[f"précision@{int(k * 100)}%"] = yy[top].mean()
        out[f"rappel@{int(k * 100)}%"] = yy[top].sum() / yy.sum()
    return out


def bootstrap(yy, ps, metric, n_boot: int = 300, seed: int = 0) -> np.ndarray:
    """Valeurs bootstrap de metric(y, p) pour chaque modèle de ps, sur les mêmes tirages."""
    yy = np.asarray(yy)
    ps = [np.asarray(p) for p in ps]
    rng = np.random.default_rng(seed)
    n = len(yy)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append([metric(yy[idx], p[idx]) for p in ps])
    return np.array(vals)


def ic(v, alpha: float = 0.05):
    """Intervalle de confiance à (1 - alpha) à partir de valeurs bootstrap."""
    return np.quantile(v, [alpha / 2, 1 - alpha / 2])