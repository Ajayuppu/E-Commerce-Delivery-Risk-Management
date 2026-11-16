"""Predictive metrics suited to a rare positive class (late ~4-10%)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss


def predictive_metrics(y, p, probabilistic=True):
    """Rules produce rankings, not probabilities: Brier/log-loss only when probabilistic=True."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    n = len(y)
    out = {"n": n, "positives": int(y.sum()), "base_rate": float(y.mean()),
           "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "brier": float(brier_score_loss(y, p)) if probabilistic else np.nan,
           "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))) if probabilistic else np.nan}
    for frac in (0.01, 0.05):
        top = order[: max(1, int(n * frac))]
        out[f"precision_top{int(frac*100)}pct"] = float(y[top].mean())
        out[f"recall_top{int(frac*100)}pct"] = float(y[top].sum() / max(y.sum(), 1))
    return out
