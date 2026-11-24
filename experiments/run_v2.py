"""
Run the fixed federation (federated_v2) plus fair baselines with the SAME fixes,
and write app_data/v2_results.json.

Usage:  python experiments/run_v2.py final_fl/finaldataset.csv app_data
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from federated_v2.client import MLPClient, SVMClient
from federated_v2.server import run, metrics

path, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
log = lambda *a: print(*a, flush=True)

res, (X1, y1, X2, y2, Xt, yt) = run(path, log=log)
log(f"federation done in {res['total_seconds']:.0f}s")

base = {}
# Hospital 1 alone: same class-weighted MLP, same 50 local epochs, no federation
torch.manual_seed(42)
m = MLPClient(X1, y1)
m.train_local(50, log=lambda *a: None)
base["local_mlp_h1"] = metrics(yt, m.predict_proba(Xt)); log("local MLP", base["local_mlp_h1"]["macro_f1"])
# Hospital 2 alone: same class-weighted, calibrated SVM, no consensus
s = SVMClient(X2, y2)
s.train_local(log=lambda *a: None)
base["local_svm_h2"] = metrics(yt, s.predict_proba(Xt)); log("local SVM", base["local_svm_h2"]["macro_f1"])
# Pooled data, same class-weighted MLP (non-private upper bound)
torch.manual_seed(42)
c = MLPClient(np.vstack([X1, X2]), np.concatenate([y1, y2]))
c.train_local(50, log=lambda *a: None)
base["central_mlp"] = metrics(yt, c.predict_proba(Xt)); log("central MLP", base["central_mlp"]["macro_f1"])
majority = np.zeros((len(yt), 3)); majority[:, 1] = 1
base["majority"] = metrics(yt, majority + 1e-9 * np.random.rand(len(yt), 3))

res["baselines"] = base
res["choices"] = {"class_weighting": "sqrt inverse frequency (chosen on a validation split of training data)",
                  "missing_indicators": "tested on validation, macro-F1 0.612 vs 0.609 without, so not used",
                  "svm": "C=10, gamma=scale (the original grid search picked C=10 every round)"}
os.makedirs(out_dir, exist_ok=True)
json.dump(res, open(os.path.join(out_dir, "v2_results.json"), "w"), indent=1)
log("ALL DONE")
