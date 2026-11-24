"""
Faithful re-run of FL_netwrok/final_fl (client.py + server.py) with extra metric capture,
plus baselines (majority, local-only, centralized) and a data export for the in-browser simulator.

Usage:
  unzip final_fl.zip            (gives final_fl/ with finaldataset.csv, client.py, server.py)
  pip install torch scikit-learn pandas numpy
  python experiments/run_experiment.py final_fl app_data
"""
import sys, os, json, time, functools
import numpy as np
import pandas as pd

FL_DIR, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, FL_DIR)
os.chdir(FL_DIR)  # server.py reads 'finaldataset.csv' relatively

import torch, torch.nn as nn, torch.nn.functional as F
import sklearn.model_selection as skms
from sklearn.metrics import (f1_score, precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.svm import SVC

import client as client_mod
# Parallelise grid search only (results identical, just faster).
client_mod.GridSearchCV = functools.partial(skms.GridSearchCV, n_jobs=-1)
import server as server_mod
from client import Client, MLP

torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(max(1, os.cpu_count() // 2))

def log(*a):
    print(*a, flush=True)

# ---------------- data profile ----------------
df = pd.read_csv('finaldataset.csv')
feat_cols = [c for c in df.columns if c not in ('id', 'prognosis')]
profile = {
    'rows': int(len(df)), 'cols': int(df.shape[1]), 'n_features': len(feat_cols),
    'class_counts': {str(k): int(v) for k, v in df['prognosis'].value_counts().sort_index().items()},
    'gender_counts': {str(k): int(v) for k, v in df['gender'].value_counts().sort_index().items()},
    'missing_total': int(df.isna().sum().sum()),
}
ages = df['age'].values
hist, edges = np.histogram(ages, bins=np.arange(0, 120, 10))
profile['age_hist'] = {'edges': edges.tolist(), 'counts': hist.tolist()}
profile['age_by_class'] = {str(k): float(v) for k, v in df.groupby('prognosis')['age'].mean().items()}
# Feature importance proxy: one-way ANOVA F between classes on standardized features
from sklearn.feature_selection import f_classif
Fv, pv = f_classif(df[feat_cols].values, df['prognosis'].values)
order = np.argsort(-np.nan_to_num(Fv))
z = (df[feat_cols] - df[feat_cols].mean()) / (df[feat_cols].std() + 1e-9)
cls_means = z.groupby(df['prognosis']).mean()
profile['top_features'] = [{
    'name': feat_cols[i].replace('obs_', ''), 'F': float(np.nan_to_num(Fv[i])),
    'class_means_z': {str(k): float(cls_means.loc[k, feat_cols[i]]) for k in cls_means.index}
} for i in order[:15]]
profile['feature_names'] = [c.replace('obs_', '') for c in feat_cols]
json.dump(profile, open(os.path.join(OUT, 'profile.json'), 'w'), indent=1)
log('profile done', profile['class_counts'])

# ---------------- splits (identical to server.py) ----------------
(X1, y1), (X2, y2), Xc, (Xt, yt) = server_mod.load_and_split_data()
splits = {'hospital1': len(y1), 'hospital2': len(y2), 'comm': len(Xc), 'test': len(yt)}
Xt_t = torch.tensor(Xt, dtype=torch.float32); yt_t = torch.tensor(yt, dtype=torch.long)
Xc_t = torch.tensor(Xc, dtype=torch.float32)

def preds_of(c, X_t):
    if isinstance(c.model, nn.Module):
        c.model.eval()
        with torch.no_grad():
            out = c.model(X_t)
        return out.argmax(1).numpy(), F.softmax(out, 1).numpy()
    Xn = X_t.numpy()
    dv = c.model.decision_function(Xn)
    e = np.exp(dv - dv.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
    return c.model.predict(Xn), p

def metrics(y, pred, prob=None):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    m = {'acc': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro')),
         'weighted_f1': float(f1_score(y, pred, average='weighted')),
         'per_class': {str(k): {'precision': float(p[k]), 'recall': float(r[k]), 'f1': float(f[k]), 'support': int(s[k])} for k in range(3)},
         'cm': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}
    if prob is not None:
        try:
            m['auc_ovr'] = float(roc_auc_score(y, prob, multi_class='ovr'))
        except Exception:
            pass
    return m

# ---------------- faithful FL run ----------------
num_rounds, local_epochs = 5, 10
c1, c2 = server_mod.create_clients()
rounds = []; train_losses = []; ft_losses = []
t_all = time.time()
for rnd in range(num_rounds):
    log(f'Round {rnd+1}')
    t0 = time.time()
    l1 = c1.train_local(epochs=local_epochs, hospital_name='Hospital 1'); train_losses.extend(l1)
    t_mlp_train = time.time() - t0
    t0 = time.time(); c2.train_local(epochs=local_epochs, hospital_name='Hospital 2'); t_svm_train = time.time() - t0
    # accuracy right after local training (before distillation) - extra insight
    pre1, pp1 = preds_of(c1, Xt_t); pre2, pp2 = preds_of(c2, Xt_t)
    pre_local = {'mlp': metrics(yt, pre1, pp1), 'svm': metrics(yt, pre2, pp2)}
    probs_1 = c1.compute_logits(Xc_t); probs_2 = c2.compute_logits(Xc_t)
    avg = (probs_1 + probs_2) / 2
    t0 = time.time(); lf = c1.fine_tune(avg, Xc_t, epochs=local_epochs, hospital_name='Hospital 1'); ft_losses.extend(lf); t_mlp_ft = time.time() - t0
    t0 = time.time(); c2.fine_tune(avg, Xc_t, epochs=local_epochs, hospital_name='Hospital 2'); t_svm_ft = time.time() - t0
    a1 = c1.evaluate(Xt_t, yt_t); a2 = c2.evaluate(Xt_t, yt_t)
    agree = server_mod.compute_agreement(probs_1, probs_2)
    p1, pr1 = preds_of(c1, Xt_t); p2, pr2 = preds_of(c2, Xt_t)
    rounds.append({'round': rnd + 1, 'acc_mlp': float(a1), 'acc_svm': float(a2), 'agreement': float(agree),
                   'mlp': metrics(yt, p1, pr1), 'svm': metrics(yt, p2, pr2), 'pre_distill': pre_local,
                   'svm_params': {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in c2.model_params.items()},
                   'avg_prob_class_share': avg.argmax(1).bincount(minlength=3).tolist(),
                   'times': {'mlp_train': t_mlp_train, 'svm_train': t_svm_train, 'mlp_ft': t_mlp_ft, 'svm_ft': t_svm_ft}})
    log(f'  acc mlp {a1:.4f} svm {a2:.4f} agree {agree:.4f} macroF1 mlp {rounds[-1]["mlp"]["macro_f1"]:.3f} svm {rounds[-1]["svm"]["macro_f1"]:.3f}  ({time.time()-t_all:.0f}s)')

fl = {'rounds': rounds, 'train_losses': train_losses, 'ft_losses': ft_losses, 'splits': splits,
      'total_seconds': time.time() - t_all,
      'final': {'acc_mlp': rounds[-1]['acc_mlp'], 'acc_svm': rounds[-1]['acc_svm'],
                'avg_train_loss': float(np.mean(train_losses)), 'avg_ft_loss': float(np.mean(ft_losses)),
                'agreement': rounds[-1]['agreement'],
                'improve_mlp': rounds[-1]['acc_mlp'] - rounds[0]['acc_mlp'],
                'improve_svm': rounds[-1]['acc_svm'] - rounds[0]['acc_svm']}}
json.dump(fl, open(os.path.join(OUT, 'fl_results.json'), 'w'), indent=1)
log('FL done', fl['final'])

# ---------------- baselines ----------------
base = {}
maj = np.bincount(y1).argmax()
base['majority'] = metrics(yt, np.full_like(yt, maj))

def train_mlp(X, y, epochs=50, seed=42):
    torch.manual_seed(seed)
    m = MLP(); opt = torch.optim.Adam(m.parameters(), lr=0.0005, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    for _ in range(epochs):
        m.train()
        for bx, by in dl:
            opt.zero_grad(); F.cross_entropy(m(bx), by).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        out = m(Xt_t)
    return out.argmax(1).numpy(), F.softmax(out, 1).numpy()

p, pr = train_mlp(X1, y1); base['local_mlp_h1'] = metrics(yt, p, pr); log('local mlp', base['local_mlp_h1']['acc'])
Xall = np.vstack([X1, X2]); yall = np.concatenate([y1, y2])
p, pr = train_mlp(Xall, yall); base['central_mlp'] = metrics(yt, p, pr); log('central mlp', base['central_mlp']['acc'])
# class-weighted MLP to show imbalance handling
torch.manual_seed(42)
w = torch.tensor(len(yall) / (3 * np.bincount(yall)), dtype=torch.float32)
m = MLP(); opt = torch.optim.Adam(m.parameters(), lr=0.0005, weight_decay=1e-4)
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(Xall, dtype=torch.float32), torch.tensor(yall)), batch_size=64, shuffle=True)
for _ in range(50):
    m.train()
    for bx, by in dl:
        opt.zero_grad(); F.cross_entropy(m(bx), by, weight=w).backward(); opt.step()
m.eval()
with torch.no_grad():
    out = m(Xt_t)
base['central_mlp_weighted'] = metrics(yt, out.argmax(1).numpy(), F.softmax(out, 1).numpy()); log('central weighted', base['central_mlp_weighted']['macro_f1'])

t0 = time.time()
s = SVC(kernel='rbf', decision_function_shape='ovr'); s.fit(X2, y2)
dv = s.decision_function(Xt); e = np.exp(dv - dv.max(1, keepdims=True))
base['local_svm_h2'] = metrics(yt, s.predict(Xt), e / e.sum(1, keepdims=True)); log('local svm', base['local_svm_h2']['acc'], time.time() - t0)
json.dump(base, open(os.path.join(OUT, 'baselines.json'), 'w'), indent=1)

# ---------------- sample for in-browser simulator ----------------
# Standardized features, no ids. Stratified sample of 3000 train-pool rows + 1000 test rows.
rng = np.random.RandomState(7)
def strat(X, y, n):
    idx = []
    for k in range(3):
        ks = np.where(y == k)[0]
        take = max(1, int(round(n * len(ks) / len(y))))
        idx.extend(rng.choice(ks, take, replace=False))
    idx = np.array(idx); rng.shuffle(idx); return X[idx], y[idx]
Xs, ys = strat(Xall, yall, 3000); Xts, yts = strat(Xt, yt, 1000)
clip = lambda a: np.clip(a, -6, 6).round(2).tolist()
json.dump({'X': clip(Xs), 'y': ys.tolist(), 'Xt': clip(Xts), 'yt': yts.tolist(),
           'features': profile['feature_names']}, open(os.path.join(OUT, 'sim_sample.json'), 'w'), separators=(',', ':'))
log('ALL DONE')
