"""Reconstruction-error autoencoder for unsupervised fraud detection.

We train a denoising autoencoder on (mostly) clean applications.
At inference time, applications with high reconstruction error are
flagged — including novel fraud patterns the supervised models
weren't trained on.

Loss surface is intentionally simple: MSE on normalised features.
For an MVP this beats fancier VAEs and is much easier to debug.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger
from .base import FraudModel

log = get_logger(__name__)


class _AENet(nn.Module):
    def __init__(self, input_dim: int, encoder_dims: list[int], dropout: float):
        super().__init__()
        # Build encoder
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in encoder_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Symmetric decoder back to input_dim
        dec_layers: list[nn.Module] = []
        for h in reversed(encoder_dims[:-1]):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderFraudModel(FraudModel):
    name = "autoencoder"

    DEFAULT_PARAMS = {
        "encoder_dims": [128, 64, 32],
        "dropout": 0.2,
        "batch_size": 512,
        "epochs": 30,
        "learning_rate": 1e-3,
        "early_stopping_patience": 5,
        "noise_std": 0.05,
    }

    def __init__(self, params: dict | None = None, device: str | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scaler = StandardScaler()
        self.net: _AENet | None = None
        self.score_min_ = 0.0
        self.score_max_ = 1.0
        self._trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None, **kwargs) -> "AutoencoderFraudModel":
        # If labels available, train only on negatives ("clean" data) — semi-supervised.
        if y is not None:
            mask = y.values == 0
            X_train = X.loc[mask]
            log.info(
                f"[AE] training on {mask.sum():,} negative samples "
                f"(skipping {(~mask).sum():,} positives)"
            )
        else:
            X_train = X

        X_arr = self.scaler.fit_transform(X_train.values).astype(np.float32)
        input_dim = X_arr.shape[1]

        self.net = _AENet(
            input_dim=input_dim,
            encoder_dims=self.params["encoder_dims"],
            dropout=self.params["dropout"],
        ).to(self.device)

        optim = torch.optim.Adam(self.net.parameters(), lr=self.params["learning_rate"])
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True)

        best_loss = float("inf")
        patience = 0
        for epoch in range(self.params["epochs"]):
            self.net.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                noisy = batch + torch.randn_like(batch) * self.params["noise_std"]
                recon = self.net(noisy)
                loss = loss_fn(recon, batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(ds)

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.params["early_stopping_patience"]:
                    log.info(f"[AE] early stopping at epoch {epoch}, best_loss={best_loss:.5f}")
                    break

        # Compute reconstruction-error range for normalisation
        self.net.eval()
        with torch.no_grad():
            recon = self.net(torch.from_numpy(X_arr).to(self.device))
            err = ((recon - torch.from_numpy(X_arr).to(self.device)) ** 2).mean(dim=1).cpu().numpy()
        self.score_min_ = float(np.percentile(err, 1))
        self.score_max_ = float(np.percentile(err, 99))
        self._trained = True
        log.info(
            f"[AE] trained. recon_err range [{self.score_min_:.4f}, {self.score_max_:.4f}]"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained or self.net is None:
            raise RuntimeError("Model not trained")
        self.net.eval()
        X_arr = self.scaler.transform(X.values).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(X_arr).to(self.device)
            recon = self.net(t)
            err = ((recon - t) ** 2).mean(dim=1).cpu().numpy()
        span = max(self.score_max_ - self.score_min_, 1e-9)
        prob = (err - self.score_min_) / span
        return np.clip(prob, 0.0, 1.0)

    def get_params(self) -> dict[str, Any]:
        return {"params": self.params}
