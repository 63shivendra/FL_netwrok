"""
Fixed federated server (v2). Run:  python federated_v2/server.py path/to/finaldataset.csv

Differences from federated/server.py:
- Same 30/30/20/20 stratified split (random_state=42), so results are directly comparable.
- Federated StandardScaler: hospitals share only (count, sum, sum of squares).
- Agreement is measured AFTER fine-tuning (and before, for reference).
- Reports macro-F1, balanced accuracy, ROC-AUC and per-class recall, not just accuracy.
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

try:
    from .client import MLPClient, SVMClient, local_stats
except ImportError:
    from client import MLPClient, SVMClient, local_stats


def load_and_split(path):
    df = pd.read_csv(path)
    X = df.drop(["id", "prognosis"], axis=1).values
    y = df["prognosis"].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_ref, X_test, y_ref, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=42)
    return (X1, y1), (X2, y2), X_ref, (X_test, y_test)


def federated_scaler(*parts):
    """Combine each hospital's (n, sum, sumsq) into a global mean/std without pooling rows."""
    stats = [local_stats(p) for p in parts]
    n = sum(s[0] for s in stats)
    mean = sum(s[1] for s in stats) / n
    var = sum(s[2] for s in stats) / n - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    std[std == 0] = 1.0
    return lambda X: (X - mean) / std


def metrics(y, prob):
    pred = prob.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"acc": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "bal_acc": float(balanced_accuracy_score(y, pred)), "auc_ovr": float(roc_auc_score(y, prob, multi_class="ovr")),
            "per_class": {str(k): {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k])} for k in range(3)},
            "cm": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def run(path, rounds=5, local_epochs=10, ft_epochs=1, log=print, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (X1, y1), (X2, y2), Xr, (Xt, yt) = load_and_split(path)
    scale = federated_scaler(X1, X2)
    X1, X2, Xr, Xt = scale(X1), scale(X2), scale(Xr), scale(Xt)
    h1, h2 = MLPClient(X1, y1), SVMClient(X2, y2)
    out = {"rounds": [], "train_losses": [], "ft_losses": []}
    log("Starting Federated Learning (v2)...\n")
    t0 = time.time()
    for rnd in range(1, rounds + 1):
        log(f"Round {rnd}/{rounds}")
        out["train_losses"] += h1.train_local(local_epochs, log=log)
        h2.train_local(log=log)
        pre = {"mlp": metrics(yt, h1.predict_proba(Xt)), "svm": metrics(yt, h2.predict_proba(Xt))}
        p1, p2 = h1.predict_proba(Xr), h2.predict_proba(Xr)
        agree_before = float((p1.argmax(1) == p2.argmax(1)).mean())
        p_bar = (p1 + p2) / 2
        out["ft_losses"] += h1.fine_tune(Xr, p_bar, epochs=ft_epochs, log=log)
        h2.fine_tune(Xr, p_bar, log=log)
        q1, q2 = h1.predict_proba(Xr), h2.predict_proba(Xr)
        agree_after = float((q1.argmax(1) == q2.argmax(1)).mean())
        m1, m2 = metrics(yt, h1.predict_proba(Xt)), metrics(yt, h2.predict_proba(Xt))
        out["rounds"].append({"round": rnd, "mlp": m1, "svm": m2, "pre_distill": pre,
                              "agreement_before": agree_before, "agreement_after": agree_after,
                              "consensus_top_class": np.bincount(p_bar.argmax(1), minlength=3).tolist(),
                              "seconds": time.time() - t0})
        log(f"  Hospital 1 (MLP): acc {m1['acc']:.4f}  macro-F1 {m1['macro_f1']:.4f}  bal-acc {m1['bal_acc']:.4f}  AUC {m1['auc_ovr']:.4f}")
        log(f"  Hospital 2 (SVM): acc {m2['acc']:.4f}  macro-F1 {m2['macro_f1']:.4f}  bal-acc {m2['bal_acc']:.4f}  AUC {m2['auc_ovr']:.4f}")
        log(f"  Agreement on reference set: before {agree_before:.4f} -> after {agree_after:.4f}\n")
    out["total_seconds"] = time.time() - t0
    out["split"] = {"hospital1": len(y1), "hospital2": len(y2), "reference": len(Xr), "test": len(yt)}
    return out, (X1, y1, X2, y2, Xt, yt)


if __name__ == "__main__":
    res, _ = run(sys.argv[1] if len(sys.argv) > 1 else "finaldataset.csv")
    f = res["rounds"][-1]
    print("Final Metrics (v2):")
    for name in ("mlp", "svm"):
        m = f[name]
        print(f"  {name.upper()}: acc {m['acc']:.4f}  macro-F1 {m['macro_f1']:.4f}  balanced acc {m['bal_acc']:.4f}  ROC-AUC {m['auc_ovr']:.4f}")
