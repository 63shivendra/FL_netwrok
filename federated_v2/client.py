"""
Fixed federated clients (v2). Differences from federated/client.py:

1. Class imbalance: both hospitals weight their loss by sqrt(inverse class frequency),
   computed from their OWN labels (nothing shared).
2. SVM keeps its private knowledge: fine-tuning retrains on local labels PLUS the
   consensus pseudo-labels, instead of replacing the model with one trained only on
   pseudo-labels.
3. Real probabilities from the SVM: Platt-calibrated on a held-out 10% of the hospital's
   own data (instead of a softmax over raw decision values).
4. MLP fine-tune mixes the soft consensus loss with its local loss (no forgetting), in
   mini-batches over the whole reference set.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

NUM_CLASSES = 3


def class_weights(y, power=0.5):
    """sqrt-inverse-frequency weights from a hospital's own labels, normalised to mean 1."""
    cnt = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    w = (len(y) / (NUM_CLASSES * np.maximum(cnt, 1))) ** power
    return w / w.mean()


def local_stats(X):
    """What a hospital shares to build a federated StandardScaler: count, sum, sum of squares."""
    return len(X), X.sum(axis=0), (X ** 2).sum(axis=0)


class MLP(nn.Module):
    def __init__(self, input_dim=44, hidden1=128, hidden2=64, output_dim=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden2, output_dim))

    def forward(self, x):
        return self.net(x)


class MLPClient:
    def __init__(self, X, y, lr=5e-4):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.w = torch.tensor(class_weights(y), dtype=torch.float32)
        self.model = MLP(X.shape[1])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loader = DataLoader(TensorDataset(self.X, self.y), batch_size=64, shuffle=True)

    def train_local(self, epochs=10, log=print):
        self.model.train()
        losses = []
        for ep in range(epochs):
            tot = 0.0
            for bx, by in self.loader:
                self.opt.zero_grad()
                loss = F.cross_entropy(self.model(bx), by, weight=self.w)
                loss.backward()
                self.opt.step()
                tot += loss.item()
            losses.append(tot / len(self.loader))
            log(f"  Hospital 1 (MLP) - Epoch {ep + 1}/{epochs}, Loss: {losses[-1]:.4f}")
        return losses

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        return F.softmax(self.model(torch.as_tensor(X, dtype=torch.float32)), dim=1).numpy()

    def fine_tune(self, X_ref, p_bar, epochs=1, alpha=0.5, log=print):
        """alpha * soft-label CE on the reference set + (1 - alpha) * weighted CE on local data."""
        self.model.train()
        Xr = torch.tensor(X_ref, dtype=torch.float32)
        P = torch.tensor(p_bar, dtype=torch.float32)
        sw = P @ self.w  # class-balance the consensus too
        ref_loader = DataLoader(TensorDataset(Xr, P, sw), batch_size=64, shuffle=True)
        losses = []
        for ep in range(epochs):
            local_iter = iter(self.loader)
            tot = 0.0
            for bx, bp, bw in ref_loader:
                try:
                    lx, ly = next(local_iter)
                except StopIteration:
                    local_iter = iter(self.loader)
                    lx, ly = next(local_iter)
                self.opt.zero_grad()
                soft = (-(bp * F.log_softmax(self.model(bx), dim=1)).sum(1) * bw).mean()
                hard = F.cross_entropy(self.model(lx), ly, weight=self.w)
                loss = alpha * soft + (1 - alpha) * hard
                loss.backward()
                self.opt.step()
                tot += loss.item()
            losses.append(tot / len(ref_loader))
            log(f"  Hospital 1 (MLP) - Fine-tune Epoch {ep + 1}/{epochs}, Loss: {losses[-1]:.4f}")
        return losses


class SVMClient:
    def __init__(self, X, y, C=10.0, gamma="scale", calib_frac=0.1, seed=42):
        self.Xf, self.Xcal, self.yf, self.ycal = train_test_split(X, y, test_size=calib_frac, stratify=y, random_state=seed)
        self.cw = class_weights(y)
        self.C, self.gamma = C, gamma
        self.model = None

    def _fit(self, X, y, sw):
        svc = SVC(kernel="rbf", C=self.C, gamma=self.gamma, decision_function_shape="ovr").fit(X, y, sample_weight=sw)
        # Platt scaling on the hospital's own held-out rows, in the same class-balanced space
        self.model = CalibratedClassifierCV(FrozenEstimator(svc), method="sigmoid")
        self.model.fit(self.Xcal, self.ycal, sample_weight=self.cw[self.ycal])

    def train_local(self, log=print):
        if self.model is None:  # later rounds already contain the local data
            self._fit(self.Xf, self.yf, self.cw[self.yf])
            log("  Hospital 2 (SVM) - Trained locally (class-weighted, calibrated)")
        else:
            log("  Hospital 2 (SVM) - Local model carried over from last round")

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fine_tune(self, X_ref, p_bar, lam=0.5, log=print):
        """Retrain on local labels + confidence-weighted consensus pseudo-labels."""
        pseudo = p_bar.argmax(1)
        conf = p_bar.max(1)
        X = np.vstack([self.Xf, X_ref])
        y = np.concatenate([self.yf, pseudo])
        sw = np.concatenate([self.cw[self.yf], lam * conf * self.cw[pseudo]])
        self._fit(X, y, sw)
        log(f"  Hospital 2 (SVM) - Fine-tuned on {len(self.yf):,} local + {len(pseudo):,} consensus rows")
