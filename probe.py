"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (fit,
fit_hyperparameters, predict, predict_proba) keep their signatures.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """
    Binary MLP probe for hallucination detection.

    Architecture:
      - StandardScaler preprocessing
      - 2-layer MLP: input → 512 → 128 → 1
      - BatchNorm + ReLU + Dropout(0.3) after each hidden layer
      - BCEWithLogitsLoss with pos_weight for class imbalance
      - AdamW optimizer, mini-batch SGD, early stopping on train loss
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Module | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Call fit() before forward().")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)

        if self._net is None:
            self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=5e-4, weight_decay=1e-3
        )

        self.train()
        max_epochs  = 100
        batch_size  = 64
        best_loss   = float("inf")
        patience    = 12
        no_improve  = 0
        n_samples   = X_t.size(0)

        for _ in range(max_epochs):
            perm       = torch.randperm(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                loss = criterion(self(X_t[idx]), y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)

            epoch_loss /= n_samples
            if epoch_loss + 1e-4 < best_loss:
                best_loss  = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 201)])
        )
        best_thr, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_thr = score, float(t)
        self._threshold = best_thr
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(self._scaler.transform(X)).float()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self(X_t)).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)